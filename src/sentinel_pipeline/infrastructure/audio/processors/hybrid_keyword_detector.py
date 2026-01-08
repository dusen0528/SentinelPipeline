"""
하이브리드 키워드 감지기

3단계 계층형 아키텍처를 통합하여 키워드를 감지합니다.
- Fast Path: 해시맵 조회 (O(1))
- Medium Path: 형태소 분석 + 편집 거리
- Heavy Path: 의미 유사도 (비동기)
"""

import logging
from pathlib import Path
from typing import Any

from sentinel_pipeline.domain.models.event import EventType
from sentinel_pipeline.infrastructure.audio.processors.keyword_matcher import (
    ExpandedKeywordDict,
)
from sentinel_pipeline.infrastructure.audio.processors.morphological_matcher import (
    MorphologicalMatcher,
)
from sentinel_pipeline.infrastructure.audio.processors.semantic_matcher import (
    SemanticSimilarityMatcher,
)

logger = logging.getLogger(__name__)


class HybridKeywordDetector:
    """
    3단계 하이브리드 키워드 감지기
    
    깔때기(Funnel) 구조로 가장 빠른 방법부터 순차적으로 시도합니다.
    """

    def __init__(
        self,
        keyword_dict_path: str | Path | None = None,
        enable_medium_path: bool = True,
        enable_heavy_path: bool = True,
        heavy_path_async: bool = True,
        semantic_threshold: float = 0.7,
        use_korean_model: bool = False,
    ):
        """
        Args:
            keyword_dict_path: 키워드 사전 JSON 파일 경로
            enable_medium_path: Medium Path 활성화 여부
            enable_heavy_path: Heavy Path 활성화 여부
            heavy_path_async: Heavy Path 비동기 처리 여부
            semantic_threshold: 의미 유사도 임계값
            use_korean_model: 한국어 특화 모델 사용 여부
        """
        # Fast Path 초기화
        self.fast_path = ExpandedKeywordDict(dict_path=keyword_dict_path)

        self.enable_medium_path = enable_medium_path
        self.enable_heavy_path = enable_heavy_path
        self.heavy_path_async = heavy_path_async

        # Medium Path 초기화 (모든 variations와 비교하도록 개선)
        self.medium_path: MorphologicalMatcher | None = None
        if enable_medium_path:
            try:
                keywords = self.fast_path.get_all_keywords()
                event_type_map = {
                    kw: self.fast_path.event_type_map.get(kw, EventType.CUSTOM)
                    for kw in keywords
                }
                # 모든 variations 딕셔너리 가져오기
                all_variations = self.fast_path.get_all_variations_dict()
                
                self.medium_path = MorphologicalMatcher(
                    keywords=keywords, 
                    event_type_map=event_type_map,
                    all_variations=all_variations,
                    similarity_threshold=0.65,  # STT 오인식 고려하여 임계값 낮춤
                )
                logger.info(f"Medium Path 초기화 완료 (variations: {sum(len(v) for v in all_variations.values())}개)")
            except Exception as e:
                logger.warning(f"Medium Path 초기화 실패: {e}, 비활성화됩니다.")
                self.enable_medium_path = False

        # Heavy Path 초기화 (의미 유사도 - 모든 variations 포함)
        self.heavy_path: SemanticSimilarityMatcher | None = None
        if enable_heavy_path:
            try:
                # 대표 키워드뿐만 아니라 주요 variations도 포함
                keywords = self.fast_path.get_all_keywords()
                all_variations = self.fast_path.get_all_variations_dict()
                
                # 의미 유사도 비교용 키워드 목록 (대표 + 주요 variations)
                semantic_keywords = list(keywords)
                for base_kw, variations in all_variations.items():
                    # 각 카테고리에서 의미가 명확한 variations 추가
                    for var in variations[:10]:  # 상위 10개만
                        if var not in semantic_keywords:
                            semantic_keywords.append(var)
                
                event_type_map = {}
                for kw in semantic_keywords:
                    base = self.fast_path.lookup_table.get(kw, kw)
                    event_type_map[kw] = self.fast_path.event_type_map.get(base, EventType.CUSTOM)
                
                self.heavy_path = SemanticSimilarityMatcher(
                    keywords=semantic_keywords,
                    event_type_map=event_type_map,
                    similarity_threshold=max(0.5, semantic_threshold - 0.1),  # 임계값 낮춤
                    use_korean_model=True,  # 한국어 모델 기본 사용
                )
                logger.info(f"Heavy Path 초기화 완료 (keywords: {len(semantic_keywords)}개, 비동기 로딩 중)")
            except Exception as e:
                logger.warning(f"Heavy Path 초기화 실패: {e}, 비활성화됩니다.")
                self.enable_heavy_path = False

    def detect(
        self, text: str
    ) -> dict[str, Any]:
        """
        텍스트에서 위험 키워드 감지
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            {
                'is_dangerous': bool,
                'event_type': EventType | None,
                'keyword': str | None,
                'confidence': float,
                'path': str,  # 'fast', 'medium', 'heavy'
                'original_text': str
            }
        """
        if not text:
            return {
                "is_dangerous": False,
                "event_type": None,
                "keyword": None,
                "confidence": 0.0,
                "path": "none",
                "original_text": text,
            }

        # 1단계: Fast Path (O(1) 조회)
        matched, base_keyword, event_type = self.fast_path.contains_keyword(text)
        if matched and base_keyword:
            return {
                "is_dangerous": True,
                "event_type": event_type,
                "keyword": base_keyword,
                "confidence": 1.0,
                "path": "fast",
                "original_text": text,
            }

        # 2단계: Medium Path (형태소 분석 + 편집 거리)
        if self.enable_medium_path and self.medium_path:
            try:
                matched, keyword, event_type, confidence = self.medium_path.match(text)
                if matched and keyword:
                    return {
                        "is_dangerous": True,
                        "event_type": event_type,
                        "keyword": keyword,
                        "confidence": confidence,
                        "path": "medium",
                        "original_text": text,
                    }
            except Exception as e:
                logger.warning(f"Medium Path 처리 중 오류: {e}")

        # 3단계: Heavy Path (의미 유사도)
        # 비동기 모드에서는 실시간 블로킹하지 않음
        if self.enable_heavy_path and self.heavy_path:
            if not self.heavy_path_async:
                # 동기 모드: 즉시 처리
                try:
                    matched, keyword, event_type, confidence = self.heavy_path.match(text)
                    if matched and keyword:
                        return {
                            "is_dangerous": True,
                            "event_type": event_type,
                            "keyword": keyword,
                            "confidence": confidence,
                            "path": "heavy",
                            "original_text": text,
                        }
                except Exception as e:
                    logger.warning(f"Heavy Path 처리 중 오류: {e}")
            # 비동기 모드: 후처리 분석용 (실시간 알림은 1-2단계에서 처리)

        # 매칭 실패
        return {
            "is_dangerous": False,
            "event_type": None,
            "keyword": None,
            "confidence": 0.0,
            "path": "none",
            "original_text": text,
        }

    def analyze(self, text: str) -> dict[str, Any]:
        """
        analyze 메서드 (기존 RiskKeywordMatcher와 호환성 유지)
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            {
                'is_dangerous': bool,
                'event_type': EventType | None,
                'keyword': str | None,
                'confidence': float,
                'original_text': str
            }
        """
        result = self.detect(text)
        return {
            "is_dangerous": result["is_dangerous"],
            "event_type": result["event_type"],
            "keyword": result["keyword"],
            "confidence": result["confidence"],
            "path": result["path"],
            "original_text": result["original_text"],
        }

