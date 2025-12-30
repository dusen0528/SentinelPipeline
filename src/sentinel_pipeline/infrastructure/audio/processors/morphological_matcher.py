"""
Medium Path: 형태소 분석 + 편집 거리 기반 매칭

Kiwi 형태소 분석기와 Levenshtein 거리를 사용하여
STT 오인식 및 한국어 교착어 특성을 처리합니다.
"""

import logging
from functools import lru_cache
from typing import Any

from sentinel_pipeline.domain.models.event import EventType

logger = logging.getLogger(__name__)

try:
    from kiwipiepy import Kiwi
    from Levenshtein import distance as levenshtein_distance
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    logger.warning(
        "kiwipiepy 또는 python-Levenshtein이 설치되지 않았습니다. "
        "Medium Path가 비활성화됩니다."
    )


class MorphologicalMatcher:
    """
    형태소 분석 + 편집 거리 기반 키워드 매처
    
    Kiwi를 사용하여 형태소 분석 및 정규화를 수행하고,
    Levenshtein 거리로 발음 유사성을 처리합니다.
    """

    def __init__(
        self,
        keywords: list[str] | None = None,
        event_type_map: dict[str, EventType] | None = None,
        max_edit_distance: int | None = None,
        normalize_coda: bool = True,
    ):
        """
        Args:
            keywords: 검사할 키워드 목록 (대표키워드)
            event_type_map: 키워드별 EventType 매핑
            max_edit_distance: 최대 편집 거리 (None이면 자동 계산)
            normalize_coda: Kiwi의 받침 정규화 옵션
        """
        if not KIWI_AVAILABLE:
            logger.error("Kiwi가 사용 불가능합니다. MorphologicalMatcher를 초기화할 수 없습니다.")
            raise ImportError("kiwipiepy 또는 python-Levenshtein이 필요합니다.")

        self.keywords = keywords or []
        self.event_type_map = event_type_map or {}
        self.normalize_coda = normalize_coda

        # 동적 임계값 계산을 위한 최대 편집 거리
        if max_edit_distance is None:
            # 키워드 길이의 30% 또는 최소 1자
            self.max_edit_distance = max(
                1, int(min(len(kw) for kw in self.keywords) * 0.3) if self.keywords else 1
            )
        else:
            self.max_edit_distance = max_edit_distance

        # Kiwi 초기화 (기본 모델 사용, sbg는 별도 설치 필요)
        try:
            self.kiwi = Kiwi()
            logger.info("Kiwi 형태소 분석기 초기화 완료")
        except Exception as e:
            logger.error(f"Kiwi 초기화 실패: {e}")
            raise

    def _calculate_threshold(self, keyword: str) -> int:
        """
        키워드 길이에 따른 동적 임계값 계산
        
        Args:
            keyword: 키워드
            
        Returns:
            최대 허용 편집 거리
        """
        # 키워드 길이의 30% 또는 최소 1자
        return max(1, int(len(keyword) * 0.3))

    def _extract_core_morphemes(self, text: str) -> str:
        """
        핵심 형태소(명사/동사/형용사)만 추출하여 노이즈 제거
        
        Args:
            text: 원본 텍스트
            
        Returns:
            핵심 형태소만 포함된 텍스트
        """
        try:
            tokens = self.kiwi.tokenize(text, normalize_coda=self.normalize_coda)
            core_tokens = []

            for token in tokens:
                tag = token.tag
                # 명사(N), 동사(V), 형용사(VA)만 추출
                if tag.startswith(("N", "V")):
                    core_tokens.append(token.form)

            return "".join(core_tokens)

        except Exception as e:
            logger.warning(f"형태소 분석 실패: {e}, 원본 텍스트 사용")
            return text

    @lru_cache(maxsize=1000)
    def _cached_tokenize(self, text: str) -> str:
        """형태소 분석 결과 캐싱"""
        return self._extract_core_morphemes(text)

    def match(self, text: str) -> tuple[bool, str | None, EventType | None, float]:
        """
        텍스트에서 키워드 매칭 시도
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            (매칭 여부, 대표키워드, EventType, 신뢰도) 튜플
        """
        if not text or not self.keywords:
            return False, None, None, 0.0

        # 1. Kiwi 정규화 및 핵심 형태소 추출
        normalized_text = self._cached_tokenize(text)

        if not normalized_text:
            return False, None, None, 0.0

        # 2. 각 키워드와 편집 거리 계산
        best_match = None
        best_distance = float("inf")
        best_keyword = None

        for keyword in self.keywords:
            # 키워드도 형태소 분석하여 비교
            normalized_keyword = self._cached_tokenize(keyword)

            # 편집 거리 계산
            dist = levenshtein_distance(normalized_text, normalized_keyword)

            # 동적 임계값 적용
            threshold = self._calculate_threshold(normalized_keyword)

            if dist <= threshold and dist < best_distance:
                best_distance = dist
                best_keyword = keyword
                best_match = normalized_keyword

        if best_match is not None:
            # 신뢰도 계산: 거리가 가까울수록 높은 신뢰도
            max_threshold = self._calculate_threshold(best_match)
            confidence = max(0.0, 1.0 - (best_distance / max_threshold))
            event_type = self.event_type_map.get(best_keyword)

            return True, best_keyword, event_type, confidence

        return False, None, None, 0.0

    def update_keywords(
        self, keywords: list[str], event_type_map: dict[str, EventType]
    ):
        """키워드 목록 및 EventType 매핑 업데이트"""
        self.keywords = keywords
        self.event_type_map = event_type_map
        # 캐시 초기화
        self._cached_tokenize.cache_clear()

