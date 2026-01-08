"""
Medium Path: 형태소 분석 + 편집 거리 + 한국어 발음 유사도 기반 매칭

STT 오인식을 처리하기 위해 다층 매칭 전략을 사용합니다:
1. 형태소 분석으로 핵심 단어 추출
2. 한국어 자모 분리 후 편집 거리 계산
3. 발음 유사 자모 치환 (ㅅ/ㅆ, ㅈ/ㅊ 등)
"""

import logging
import re
from functools import lru_cache
from typing import Any

from sentinel_pipeline.domain.models.event import EventType

logger = logging.getLogger(__name__)

try:
    from kiwipiepy import Kiwi
    from Levenshtein import distance as levenshtein_distance, ratio as levenshtein_ratio
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    logger.warning(
        "kiwipiepy 또는 python-Levenshtein이 설치되지 않았습니다. "
        "Medium Path가 비활성화됩니다."
    )


# 한국어 자모 분리/조합 유틸리티
CHOSUNG = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]
JUNGSUNG = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]
JONGSUNG = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# 발음 유사 자모 그룹 (STT 오인식 패턴)
SIMILAR_CHOSUNG = {
    'ㄱ': ['ㄱ', 'ㄲ', 'ㅋ'],
    'ㄲ': ['ㄱ', 'ㄲ', 'ㅋ'],
    'ㅋ': ['ㄱ', 'ㄲ', 'ㅋ'],
    'ㄷ': ['ㄷ', 'ㄸ', 'ㅌ'],
    'ㄸ': ['ㄷ', 'ㄸ', 'ㅌ'],
    'ㅌ': ['ㄷ', 'ㄸ', 'ㅌ'],
    'ㅂ': ['ㅂ', 'ㅃ', 'ㅍ'],
    'ㅃ': ['ㅂ', 'ㅃ', 'ㅍ'],
    'ㅍ': ['ㅂ', 'ㅃ', 'ㅍ'],
    'ㅅ': ['ㅅ', 'ㅆ'],
    'ㅆ': ['ㅅ', 'ㅆ'],
    'ㅈ': ['ㅈ', 'ㅉ', 'ㅊ'],
    'ㅉ': ['ㅈ', 'ㅉ', 'ㅊ'],
    'ㅊ': ['ㅈ', 'ㅉ', 'ㅊ'],
}

SIMILAR_JUNGSUNG = {
    'ㅐ': ['ㅐ', 'ㅔ', 'ㅒ', 'ㅖ'],
    'ㅔ': ['ㅐ', 'ㅔ', 'ㅒ', 'ㅖ'],
    'ㅒ': ['ㅐ', 'ㅔ', 'ㅒ', 'ㅖ'],
    'ㅖ': ['ㅐ', 'ㅔ', 'ㅒ', 'ㅖ'],
    'ㅗ': ['ㅗ', 'ㅛ', 'ㅜ'],
    'ㅛ': ['ㅗ', 'ㅛ'],
    'ㅜ': ['ㅜ', 'ㅠ', 'ㅗ'],
    'ㅠ': ['ㅜ', 'ㅠ'],
    'ㅓ': ['ㅓ', 'ㅕ', 'ㅏ'],
    'ㅕ': ['ㅓ', 'ㅕ'],
    'ㅏ': ['ㅏ', 'ㅑ', 'ㅓ'],
    'ㅑ': ['ㅏ', 'ㅑ'],
    'ㅡ': ['ㅡ', 'ㅣ', 'ㅜ'],
    'ㅣ': ['ㅣ', 'ㅡ', 'ㅢ'],
    'ㅢ': ['ㅢ', 'ㅣ', 'ㅡ'],
}


def decompose_korean(text: str) -> str:
    """한글을 자모로 분리"""
    result = []
    for char in text:
        if '가' <= char <= '힣':
            code = ord(char) - 0xAC00
            cho = code // (21 * 28)
            jung = (code % (21 * 28)) // 28
            jong = code % 28
            result.append(CHOSUNG[cho])
            result.append(JUNGSUNG[jung])
            if jong > 0:
                result.append(JONGSUNG[jong])
        else:
            result.append(char)
    return ''.join(result)


def normalize_pronunciation(text: str) -> str:
    """발음 유사 자모를 대표 자모로 정규화"""
    result = []
    for char in text:
        # 초성 정규화
        if char in SIMILAR_CHOSUNG:
            result.append(SIMILAR_CHOSUNG[char][0])
        # 중성 정규화
        elif char in SIMILAR_JUNGSUNG:
            result.append(SIMILAR_JUNGSUNG[char][0])
        else:
            result.append(char)
    return ''.join(result)


def korean_phonetic_distance(text1: str, text2: str) -> float:
    """
    한국어 발음 유사도 기반 거리 계산
    
    Returns:
        0.0 ~ 1.0 (1.0이 완전 일치)
    """
    # 자모 분리
    jamo1 = decompose_korean(text1)
    jamo2 = decompose_korean(text2)
    
    # 발음 정규화 후 비교
    norm1 = normalize_pronunciation(jamo1)
    norm2 = normalize_pronunciation(jamo2)
    
    # Levenshtein ratio (유사도)
    return levenshtein_ratio(norm1, norm2)


class MorphologicalMatcher:
    """
    형태소 분석 + 편집 거리 + 한국어 발음 유사도 기반 키워드 매처
    
    STT 오인식을 처리하기 위해 다층 매칭 전략을 사용:
    1. 원본 텍스트 → 모든 variations와 직접 비교
    2. 형태소 분석 후 핵심 단어 추출 → 비교
    3. 자모 분리 + 발음 정규화 → 비교
    """

    def __init__(
        self,
        keywords: list[str] | None = None,
        event_type_map: dict[str, EventType] | None = None,
        all_variations: dict[str, list[str]] | None = None,
        similarity_threshold: float = 0.7,
        normalize_coda: bool = True,
    ):
        """
        Args:
            keywords: 대표키워드 목록
            event_type_map: 키워드별 EventType 매핑
            all_variations: {대표키워드: [변형1, 변형2, ...]} 매핑
            similarity_threshold: 유사도 임계값 (0.0 ~ 1.0)
            normalize_coda: Kiwi의 받침 정규화 옵션
        """
        if not KIWI_AVAILABLE:
            logger.error("Kiwi가 사용 불가능합니다.")
            raise ImportError("kiwipiepy 또는 python-Levenshtein이 필요합니다.")

        self.keywords = keywords or []
        self.event_type_map = event_type_map or {}
        self.similarity_threshold = similarity_threshold
        self.normalize_coda = normalize_coda

        # 모든 variations를 flat하게 저장: {variation: base_keyword}
        self.variation_to_base: dict[str, str] = {}
        self.all_variations_list: list[str] = []
        
        if all_variations:
            for base_keyword, variations in all_variations.items():
                for var in variations:
                    self.variation_to_base[var] = base_keyword
                    self.all_variations_list.append(var)
                # 대표 키워드도 추가
                self.variation_to_base[base_keyword] = base_keyword
                self.all_variations_list.append(base_keyword)
        else:
            # all_variations가 없으면 키워드 자체만 사용
            for kw in self.keywords:
                self.variation_to_base[kw] = kw
                self.all_variations_list.append(kw)

        # Kiwi 초기화
        try:
            self.kiwi = Kiwi()
            logger.info(f"Kiwi 형태소 분석기 초기화 완료 (variations: {len(self.all_variations_list)}개)")
        except Exception as e:
            logger.error(f"Kiwi 초기화 실패: {e}")
            raise

    def _extract_core_text(self, text: str) -> str:
        """핵심 형태소(명사/동사/형용사 어간)만 추출"""
        try:
            tokens = self.kiwi.tokenize(text, normalize_coda=self.normalize_coda)
            core_tokens = []
            for token in tokens:
                tag = token.tag
                # 명사(N), 동사(V), 형용사(VA), 감탄사(IC) 추출
                if tag.startswith(("N", "V", "IC")):
                    core_tokens.append(token.form)
            return "".join(core_tokens) if core_tokens else text
        except Exception as e:
            logger.debug(f"형태소 분석 실패: {e}")
            return text

    def _clean_text(self, text: str) -> str:
        """텍스트 정제 (공백, 구두점 제거)"""
        return re.sub(r'[^\w가-힣]', '', text)

    def match(self, text: str) -> tuple[bool, str | None, EventType | None, float]:
        """
        텍스트에서 키워드 매칭 시도 (다층 전략)
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            (매칭 여부, 대표키워드, EventType, 신뢰도) 튜플
        """
        if not text or not self.all_variations_list:
            return False, None, None, 0.0

        clean_input = self._clean_text(text)
        if not clean_input:
            return False, None, None, 0.0

        best_match = None
        best_similarity = 0.0
        best_keyword = None
        
        # === 1단계: 원본 텍스트 → 모든 variations와 직접 발음 유사도 비교 ===
        for variation in self.all_variations_list:
            clean_var = self._clean_text(variation)
            if not clean_var:
                continue
                
            # 발음 유사도 계산
            similarity = korean_phonetic_distance(clean_input, clean_var)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = variation
                best_keyword = self.variation_to_base.get(variation)

        # === 2단계: 부분 문자열 포함 확인 ===
        if best_similarity < self.similarity_threshold:
            for variation in self.all_variations_list:
                clean_var = self._clean_text(variation)
                if not clean_var:
                    continue
                    
                # 입력에 variation이 포함되어 있는지 확인
                if clean_var in clean_input or clean_input in clean_var:
                    # 포함된 경우 길이 비율로 유사도 계산
                    len_ratio = min(len(clean_var), len(clean_input)) / max(len(clean_var), len(clean_input))
                    if len_ratio > 0.5:  # 50% 이상 겹쳐야 함
                        similarity = 0.85  # 부분 일치 점수
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = variation
                            best_keyword = self.variation_to_base.get(variation)

        # === 3단계: 형태소 분석 후 비교 ===
        if best_similarity < self.similarity_threshold:
            core_input = self._extract_core_text(text)
            if core_input and core_input != clean_input:
                for variation in self.all_variations_list:
                    core_var = self._extract_core_text(variation)
                    if not core_var:
                        continue
                        
                    similarity = korean_phonetic_distance(core_input, core_var)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = variation
                        best_keyword = self.variation_to_base.get(variation)

        # 임계값 이상이면 매칭 성공
        if best_similarity >= self.similarity_threshold and best_keyword:
            event_type = self.event_type_map.get(best_keyword)
            logger.debug(
                f"Medium Path 매칭: '{text}' → '{best_match}' "
                f"(base={best_keyword}, similarity={best_similarity:.2f})"
            )
            return True, best_keyword, event_type, best_similarity

        return False, None, None, 0.0

    def update_variations(
        self, 
        all_variations: dict[str, list[str]], 
        event_type_map: dict[str, EventType]
    ):
        """variations 및 EventType 매핑 업데이트"""
        self.event_type_map = event_type_map
        self.variation_to_base.clear()
        self.all_variations_list.clear()
        
        for base_keyword, variations in all_variations.items():
            for var in variations:
                self.variation_to_base[var] = base_keyword
                self.all_variations_list.append(var)
            self.variation_to_base[base_keyword] = base_keyword
            self.all_variations_list.append(base_keyword)
        
        logger.info(f"variations 업데이트 완료: {len(self.all_variations_list)}개")
