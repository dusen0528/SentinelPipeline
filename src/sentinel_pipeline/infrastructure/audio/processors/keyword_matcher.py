"""
Fast Path: 확장 키워드 사전 기반 매칭

역인덱싱을 통한 O(1) 조회를 제공하는 키워드 매처입니다.
"""

import json
import logging
from pathlib import Path
from typing import Any

from sentinel_pipeline.domain.models.event import EventType

logger = logging.getLogger(__name__)


class ExpandedKeywordDict:
    """
    확장 키워드 사전 - O(1) 조회를 위한 Fast Path
    
    JSON 키워드 사전을 로드하여 {변형단어: 대표키워드} 형태의
    Flat Lookup Table을 생성합니다.
    """

    def __init__(self, dict_path: str | Path | None = None):
        """
        Args:
            dict_path: 키워드 사전 JSON 파일 경로
                      None이면 기본 경로(models/audio/keywords.json) 사용
        """
        if dict_path is None:
            # 기본 경로: 프로젝트 루트 기준
            dict_path = Path("models/audio/keywords.json")
        
        self.dict_path = Path(dict_path)
        self.lookup_table: dict[str, str] = {}  # {변형단어: 대표키워드}
        self.event_type_map: dict[str, EventType] = {}  # {대표키워드: EventType}
        self.base_keywords: list[str] = []  # 대표키워드 목록
        
        self._load_and_flatten()

    def _load_and_flatten(self):
        """JSON 키워드 사전을 로드하여 Flat Lookup Table 생성"""
        if not self.dict_path.exists():
            logger.warning(f"키워드 사전 파일을 찾을 수 없습니다: {self.dict_path}")
            return

        try:
            with open(self.dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for base_keyword, config in data.items():
                if not isinstance(config, dict):
                    continue
                
                variations = config.get("variations", [])
                event_type_str = config.get("event_type", "CUSTOM")
                
                try:
                    event_type = EventType(event_type_str)
                except ValueError:
                    logger.warning(
                        f"알 수 없는 EventType: {event_type_str}, "
                        f"키워드: {base_keyword}, CUSTOM으로 설정합니다."
                    )
                    event_type = EventType.CUSTOM

                # 대표키워드 자체도 lookup_table에 추가
                self.lookup_table[base_keyword] = base_keyword
                self.event_type_map[base_keyword] = event_type
                self.base_keywords.append(base_keyword)

                # 변형 키워드들 추가
                for variation in variations:
                    if variation and isinstance(variation, str):
                        # 공백 제거된 버전도 추가 (STT 결과 처리)
                        clean_variation = variation.replace(" ", "")
                        self.lookup_table[variation] = base_keyword
                        if clean_variation != variation:
                            self.lookup_table[clean_variation] = base_keyword

            logger.info(
                f"키워드 사전 로드 완료: {len(self.base_keywords)}개 대표키워드, "
                f"{len(self.lookup_table)}개 변형"
            )

        except json.JSONDecodeError as e:
            logger.error(f"키워드 사전 JSON 파싱 오류: {e}")
        except Exception as e:
            logger.error(f"키워드 사전 로드 오류: {e}")

    def contains_keyword(self, text: str) -> tuple[bool, str | None, EventType | None]:
        """
        텍스트에서 키워드 포함 여부 확인 (완전 일치)
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            (매칭 여부, 대표키워드, EventType) 튜플
        """
        if not text:
            return False, None, None

        # 공백 및 구두점 제거
        clean_text = text.replace(" ", "").replace(".", "").replace(",", "").replace("!", "")

        # 완전 일치 조회 (O(1))
        if clean_text in self.lookup_table:
            base_keyword = self.lookup_table[clean_text]
            event_type = self.event_type_map.get(base_keyword)
            return True, base_keyword, event_type

        # 부분 문자열 매칭 (더 느리지만 더 포괄적)
        for variation, base_keyword in self.lookup_table.items():
            if variation in clean_text:
                event_type = self.event_type_map.get(base_keyword)
                return True, base_keyword, event_type

        return False, None, None

    def get_all_keywords(self) -> list[str]:
        """모든 대표키워드 목록 반환"""
        return self.base_keywords.copy()

    def get_variations(self, base_keyword: str) -> list[str]:
        """특정 대표키워드의 모든 변형 반환"""
        variations = [
            variation
            for variation, base in self.lookup_table.items()
            if base == base_keyword
        ]
        return list(set(variations))  # 중복 제거

    def get_all_variations_dict(self) -> dict[str, list[str]]:
        """모든 대표키워드와 그 변형들의 딕셔너리 반환
        
        Returns:
            {대표키워드: [변형1, 변형2, ...], ...}
        """
        result = {}
        for base_keyword in self.base_keywords:
            result[base_keyword] = self.get_variations(base_keyword)
        return result

