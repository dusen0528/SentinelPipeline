"""
SentinelPipeline 진입점

애플리케이션 초기화, 컴포넌트 배선, FastAPI 서버 시작을 담당합니다.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable

from sentinel_pipeline.application.config.manager import ConfigManager
from sentinel_pipeline.application.event.emitter import EventEmitter
from sentinel_pipeline.application.pipeline.pipeline import PipelineEngine
from sentinel_pipeline.application.stream.manager import StreamManager
from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.domain.models.event import Event
from sentinel_pipeline.infrastructure.transport.http_client import create_http_transport
from sentinel_pipeline.infrastructure.transport.ws_client import create_ws_transport
from sentinel_pipeline.infrastructure.video.ffmpeg_publisher import FFmpegPublisher
from sentinel_pipeline.infrastructure.video.rtsp_decoder import RTSPDecoder
from sentinel_pipeline.interface.api.app import create_app
from sentinel_pipeline.interface.api.dependencies import set_app_context
from sentinel_pipeline.interface.config.loader import ConfigLoader
from sentinel_pipeline.plugins import load_plugins_from_config

logger = get_logger(__name__)

# 전역 컴포넌트 (종료 시 정리용)
_stream_manager: StreamManager | None = None
_pipeline_engine: PipelineEngine | None = None
_event_emitter: EventEmitter | None = None
_transport_close_funcs: list[Callable[[], None]] = []


def setup_transports(event_config: dict) -> tuple[Callable[[list[Event]], bool], list[Callable[[], None]]]:
    """
    이벤트 전송 클라이언트를 설정합니다.
    
    Args:
        event_config: 이벤트 설정 딕셔너리
        
    Returns:
        (transport_function, close_functions) 튜플
    """
    transports = event_config.get("transports", [])
    if not transports:
        logger.warning("이벤트 전송 설정이 없습니다")
        return lambda events: True, []
    
    enabled_transports = [t for t in transports if t.get("enabled", True)]
    if not enabled_transports:
        logger.warning("활성화된 전송 설정이 없습니다")
        return lambda events: True, []
    
    transport_functions: list[Callable[[list[Event]], bool]] = []
    close_functions: list[Callable[[], None]] = []
    
    for transport_cfg in enabled_transports:
        transport_type = transport_cfg.get("type")
        url = transport_cfg.get("url")
        
        if not url:
            logger.warning(f"전송 URL이 없습니다: {transport_cfg}")
            continue
        
        try:
            if transport_type == "http":
                # HTTP 전송 설정
                headers = transport_cfg.get("headers", {})
                transport_func, close_func = create_http_transport(
                    base_url=url,
                    endpoint="/api/events",
                    headers=headers,
                )
                transport_functions.append(transport_func)
                if close_func:
                    close_functions.append(close_func)
                logger.info(f"HTTP 전송 설정 완료: {url}")
                
            elif transport_type == "ws":
                # WebSocket 전송 설정
                transport_func, close_func = create_ws_transport(url)
                transport_functions.append(transport_func)
                if close_func:
                    close_functions.append(close_func)
                logger.info(f"WebSocket 전송 설정 완료: {url}")
                
            else:
                logger.warning(f"지원하지 않는 전송 타입: {transport_type}")
                
        except Exception as e:
            logger.error(f"전송 설정 실패 ({transport_type}): {e}", error=str(e))
    
    if not transport_functions:
        logger.warning("설정된 전송이 없습니다")
        return lambda events: True, []
    
    # 여러 전송을 순차적으로 호출하는 래퍼
    def combined_transport(events: list[Event]) -> bool:
        """여러 전송을 순차적으로 호출합니다."""
        success = True
        for transport_func in transport_functions:
            try:
                result = transport_func(events)
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"전송 오류: {e}", error=str(e))
                success = False
        return success
    
    return combined_transport, close_functions


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def initialize_components(config_path: str | Path | None = None) -> tuple:
    """
    모든 컴포넌트를 초기화하고 배선합니다.
    
    Args:
        config_path: 설정 파일 경로 (None이면 config.json)
        
    Returns:
        (stream_manager, pipeline_engine, config_manager, config_loader, event_emitter) 튜플
    """
    # 1. 설정 로드
    loader = ConfigLoader()
    if config_path:
        app_config = loader.load_from_file(config_path)
    else:
        app_config = loader.load_from_file("config.json")
    
    # 교차 검증
    is_valid, errors = loader.validate(app_config)
    if not is_valid:
        logger.error("설정 검증 실패", errors=errors)
        raise ValueError(f"설정 검증 실패: {errors}")
    
    # 런타임 번들 생성
    bundle = loader.to_runtime_bundle(app_config)
    runtime_app = bundle["app"]
    pipeline_cfg = bundle["pipeline"]
    event_cfg = bundle["event"]
    observability_cfg = bundle["observability"]
    streams_cfg = bundle["streams"]
    global_cfg = runtime_app.global_config
    
    logger.info("설정 로드 완료", module_count=len(runtime_app.modules))
    
    # 2. ConfigManager 초기화 (bundle 보관)
    config_manager = ConfigManager()
    config_manager.load_config(runtime_app, bundle=bundle)
    
    # 3. PipelineEngine 초기화 (pipeline 설정 적용)
    pipeline_engine = PipelineEngine(
        max_consecutive_errors=pipeline_cfg.get("max_consecutive_errors", 5),
        max_consecutive_timeouts=pipeline_cfg.get(
            "max_consecutive_timeouts",
            pipeline_cfg.get("auto_disable_threshold", 10),
        ),
        max_workers=pipeline_cfg.get("max_workers", 4),
        error_window_seconds=pipeline_cfg.get("error_window_seconds", 60),
        cooldown_seconds=pipeline_cfg.get("cooldown_seconds", 300),
    )
    
    # 4. EventEmitter 초기화 (event 설정 적용)
    event_emitter = EventEmitter(
        max_queue_size=event_cfg.get("max_queue_size", 1000),
        batch_size=event_cfg.get("batch_size", 10),
        flush_interval_ms=event_cfg.get("flush_interval_ms", 100),
        drop_strategy=event_cfg.get("drop_strategy", "drop_oldest"),
    )
    
    # 전송 클라이언트 설정
    transport_func, close_funcs = setup_transports(event_cfg)
    event_emitter.set_transport(transport_func)
    # close_funcs는 전역 변수에 저장하여 종료 시 호출
    global _transport_close_funcs
    _transport_close_funcs.extend(close_funcs)
    
    # PipelineEngine의 이벤트를 EventEmitter로 전달
    pipeline_engine.set_on_events(event_emitter.emit_batch)
    
    # EventEmitter 시작
    event_emitter.start()
    
    # 5. StreamManager 초기화
    stream_manager = StreamManager()
    
    # 종속성 주입 (이벤트 핸들링용)
    stream_manager.set_dependencies(
        pipeline_engine=pipeline_engine, event_emitter=event_emitter
    )
    
    # 전역 설정 적용
    stream_manager.apply_global_config(
        max_fps=global_cfg.max_fps,
        downscale=global_cfg.downscale,
        target_width=global_cfg.target_width,
        target_height=global_cfg.target_height,
    )
    
    # 디코더/퍼블리셔 팩토리 설정
    def create_decoder(stream_id: str | None = None):
        return RTSPDecoder(
            stream_id=stream_id or "unknown", event_emitter=event_emitter
        )
    
    def create_publisher(
        stream_id: str | None = None,
        output_url: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ):
        # 필수값이 없으면 기본값을 채우되, 호출 시점에 전달된 값을 우선 사용
        return FFmpegPublisher(
            stream_id=stream_id or "unknown",
            output_url=output_url or "rtsp://localhost:8554/unknown",
            width=width or 640,
            height=height or 480,
            fps=fps or 15,
        )
    
    stream_manager.set_decoder_factory(create_decoder)
    stream_manager.set_publisher_factory(create_publisher)
    
    # 프레임 처리 콜백 설정
    def frame_callback(frame, metadata):
        return pipeline_engine.process_frame(frame, metadata)
    
    stream_manager.set_frame_callback(frame_callback)
    
    # 6. 플러그인 로드 및 등록
    strict_missing = _str_to_bool(os.getenv("REQUIRE_REGISTERED_MODULES"), default=False)
    if strict_missing:
        logger.info("플러그인 로딩: 미등록 모듈 발견 시 실패 모드")
    else:
        logger.info("플러그인 로딩: 미등록 모듈 발견 시 경고 후 스킵")
    modules_cfg = app_config.model_dump().get("modules", [])
    modules = load_plugins_from_config(modules_cfg, strict_missing=strict_missing)
    for module in modules:
        pipeline_engine.register_module(module)

    # 모듈 통계 주기 전송 (WS)
    stats_interval = float(os.getenv("MODULE_STATS_INTERVAL", "5.0"))
    pipeline_engine.start_stats_publisher(interval_sec=stats_interval)
    
    # 7. 초기 스트림 시작 (enabled=True인 것만)
    for stream_config in streams_cfg:
        if stream_config.enabled:
            try:
                stream_manager.start_stream(
                    stream_id=stream_config.stream_id,
                    rtsp_url=stream_config.rtsp_url,
                    max_fps=stream_config.max_fps,
                    downscale=stream_config.downscale,
                    output_url=stream_config.output_url,
                    target_width=stream_config.target_width,
                    target_height=stream_config.target_height,
                )
                logger.info(f"초기 스트림 시작: {stream_config.stream_id}")
            except Exception as e:
                logger.error(
                    f"스트림 시작 실패: {stream_config.stream_id} - {e}",
                    stream_id=stream_config.stream_id,
                    error=str(e),
                )
    
    logger.info("컴포넌트 초기화 완료")
    
    return stream_manager, pipeline_engine, config_manager, loader, event_emitter


def apply_bundle_to_components(
    bundle: dict[str, Any],
    pipeline_engine: PipelineEngine,
    event_emitter: EventEmitter,
    stream_manager: StreamManager,
) -> None:
    """
    설정 bundle을 런타임 컴포넌트에 재적용합니다.

    PUT /api/config로 설정이 변경되었을 때 호출됩니다.
    """
    pipeline_cfg = bundle.get("pipeline", {}) or {}
    event_cfg = bundle.get("event", {}) or {}
    streams_cfg = bundle.get("streams", []) or []
    app_cfg = bundle.get("app", {}) or {}
    global_cfg = app_cfg.get("global_config", {}) if app_cfg else {}
    observability_cfg = bundle.get("observability", {}) or {}

    # 1) PipelineEngine 설정 재적용
    if pipeline_cfg:
        scheduler = pipeline_engine.scheduler
        scheduler.max_consecutive_errors = pipeline_cfg.get("max_consecutive_errors", 5)
        scheduler.max_consecutive_timeouts = pipeline_cfg.get(
            "max_consecutive_timeouts",
            pipeline_cfg.get("auto_disable_threshold", 10),
        )
        scheduler.error_window_seconds = pipeline_cfg.get("error_window_seconds", 60)
        scheduler.cooldown_seconds = pipeline_cfg.get("cooldown_seconds", 300)
        logger.info("PipelineEngine 설정 재적용 완료")

    # 2) EventEmitter 설정 재적용
    if event_cfg:
        # 기존 transport 종료 및 재시작
        event_emitter.shutdown()
        global _transport_close_funcs
        _transport_close_funcs.clear()

        event_emitter.max_queue_size = event_cfg.get("max_queue_size", 1000)
        event_emitter.batch_size = event_cfg.get("batch_size", 10)
        event_emitter.flush_interval_ms = event_cfg.get("flush_interval_ms", 100)
        drop_strategy = event_cfg.get("drop_strategy", "drop_oldest")
        from sentinel_pipeline.application.event.emitter import DropStrategy
        event_emitter.drop_strategy = DropStrategy(drop_strategy)

        transport_func, close_funcs = setup_transports(event_cfg)
        event_emitter.set_transport(transport_func)
        _transport_close_funcs.extend(close_funcs)

        event_emitter.start()
        logger.info("EventEmitter 설정 재적용 완료")

    # 3) StreamManager 전역 설정 재적용
    if global_cfg:
        stream_manager.apply_global_config(
            max_fps=global_cfg.get("max_fps", 15),
            downscale=global_cfg.get("downscale", 0.5),
            target_width=global_cfg.get("target_width"),
            target_height=global_cfg.get("target_height"),
        )
        logger.info("StreamManager 전역 설정 재적용 완료")

    # 4) Streams 재적용 (변경 감지 시 재시작)
    restart_streams_on_change = _str_to_bool(
        os.getenv("STREAM_RESTART_ON_CHANGE"), default=True
    )

    if streams_cfg:
        active_states = {s.stream_id: s for s in stream_manager.get_active_streams()}
        new_enabled_streams = {s.stream_id for s in streams_cfg if getattr(s, "enabled", False)}

        # 비활성화된 스트림 중지
        for stream_id in active_states.keys() - new_enabled_streams:
            try:
                stream_manager.stop_stream(stream_id)
                logger.info(f"스트림 중지: {stream_id}")
            except Exception as e:
                logger.error(f"스트림 중지 실패: {stream_id} - {e}", stream_id=stream_id, error=str(e))

        # 활성 스트림 재시작 필요 여부 판단
        for stream_config in streams_cfg:
            if not getattr(stream_config, "enabled", False):
                continue
            current_state = active_states.get(stream_config.stream_id)
            needs_restart = False
            if current_state:
                cfg = current_state.config
                if (
                    cfg.max_fps != stream_config.max_fps
                    or cfg.downscale != stream_config.downscale
                    or cfg.rtsp_url != stream_config.rtsp_url
                    or cfg.output_url != stream_config.output_url
                ):
                    needs_restart = True

            if current_state is None or (needs_restart and restart_streams_on_change):
                if needs_restart and restart_streams_on_change:
                    try:
                        stream_manager.stop_stream(stream_config.stream_id, force=True)
                        logger.info(f"스트림 재시작(설정 변경): {stream_config.stream_id}")
                    except Exception as e:
                        logger.error(
                            f"스트림 중지 실패(재시작): {stream_config.stream_id} - {e}",
                            stream_id=stream_config.stream_id,
                            error=str(e),
                        )
                try:
                    stream_manager.start_stream(
                        stream_id=stream_config.stream_id,
                        rtsp_url=stream_config.rtsp_url,
                        max_fps=stream_config.max_fps,
                        downscale=stream_config.downscale,
                        output_url=stream_config.output_url,
                    )
                except Exception as e:
                    logger.error(
                        f"스트림 시작 실패: {stream_config.stream_id} - {e}",
                        stream_id=stream_config.stream_id,
                        error=str(e),
                    )

    # 5) Observability 재적용 (로그 레벨만 즉시 반영)
    if observability_cfg:
        try:
            from sentinel_pipeline.common.logging import configure_logging

            log_level = observability_cfg.get("log_level", "INFO")
            configure_logging(level=str(log_level).upper())
            logger.info("Observability 설정 재적용 완료", log_level=log_level)
        except Exception as e:
            logger.warning("Observability 설정 재적용 실패", error=str(e))

    logger.info("설정 bundle 재적용 완료")


def shutdown_components() -> None:
    """모든 컴포넌트를 종료합니다."""
    global _stream_manager, _pipeline_engine, _event_emitter, _transport_close_funcs
    
    logger.info("컴포넌트 종료 시작")
    
    if _stream_manager:
        try:
            _stream_manager.stop_all_streams()
        except Exception as e:
            logger.error(f"스트림 종료 오류: {e}", error=str(e))
    
    if _pipeline_engine:
        try:
            _pipeline_engine.shutdown()
        except Exception as e:
            logger.error(f"파이프라인 종료 오류: {e}", error=str(e))
    
    if _event_emitter:
        try:
            _event_emitter.shutdown()
        except Exception as e:
            logger.error(f"이벤트 발행자 종료 오류: {e}", error=str(e))
    
    # 전송 클라이언트 종료
    for close_func in _transport_close_funcs:
        try:
            close_func()
        except Exception as e:
            logger.error(f"전송 클라이언트 종료 오류: {e}", error=str(e))
    
    _transport_close_funcs.clear()
    
    logger.info("컴포넌트 종료 완료")


def setup_signal_handlers() -> None:
    """시그널 핸들러를 설정합니다."""
    def signal_handler(signum, frame):
        logger.info(f"시그널 수신: {signum}")
        shutdown_components()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main() -> None:
    """메인 진입점."""
    import uvicorn
    
    # 시그널 핸들러 설정
    setup_signal_handlers()
    
    # 설정 파일 경로 (환경변수 또는 기본값)
    config_path = os.getenv("CONFIG_PATH", "config.json")
    
    try:
        # 컴포넌트 초기화
        global _stream_manager, _pipeline_engine, _event_emitter
        (
            _stream_manager,
            _pipeline_engine,
            config_manager,
            config_loader,
            _event_emitter,
        ) = initialize_components(config_path)
        
        # FastAPI 앱 생성
        allowed_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else None
        app = create_app(allowed_origins=allowed_origins)
        
        # DI 컨텍스트 설정
        from sentinel_pipeline.interface.api.dependencies import AppContext
        
        app_context = AppContext(
            stream_manager=_stream_manager,
            pipeline_engine=_pipeline_engine,
            config_manager=config_manager,
            config_loader=config_loader,
            event_emitter=_event_emitter,
            transport_close_funcs=_transport_close_funcs,
        )
        set_app_context(app, app_context)
        
        # 서버 시작
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        
        logger.info(f"서버 시작: http://{host}:{port}")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )
        
    except KeyboardInterrupt:
        logger.info("사용자 중단")
    except Exception as e:
        logger.exception("초기화 오류", error=str(e))
        sys.exit(1)
    finally:
        shutdown_components()


if __name__ == "__main__":
    main()
