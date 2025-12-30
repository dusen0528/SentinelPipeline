"""
Heavy Path: 의미 기반 유사도 매칭

Sentence Transformers를 사용하여 텍스트를 임베딩 벡터로 변환하고
코사인 유사도를 계산하여 의미적으로 유사한 키워드를 감지합니다.
"""

import logging
import threading
from functools import lru_cache
from typing import Any

import numpy as np

from sentinel_pipeline.domain.models.event import EventType

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers가 설치되지 않았습니다. "
        "Heavy Path가 비활성화됩니다."
    )


class SemanticSimilarityMatcher:
    """
    의미 기반 유사도 매처
    
    Sentence Transformers를 사용하여 텍스트를 임베딩 벡터로 변환하고
    코사인 유사도를 계산합니다.
    """

    def __init__(
        self,
        keywords: list[str] | None = None,
        event_type_map: dict[str, EventType] | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        use_korean_model: bool = False,
    ):
        """
        Args:
            keywords: 검사할 키워드 목록 (대표키워드)
            event_type_map: 키워드별 EventType 매핑
            model_name: Sentence Transformer 모델 이름
            similarity_threshold: 코사인 유사도 임계값
            use_korean_model: 한국어 특화 모델 사용 여부
                              True면 jhgan/ko-sroberta-multitask 사용
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error(
                "Sentence Transformers가 사용 불가능합니다. "
                "SemanticSimilarityMatcher를 초기화할 수 없습니다."
            )
            raise ImportError("sentence-transformers가 필요합니다.")

        self.keywords = keywords or []
        self.event_type_map = event_type_map or {}
        self.similarity_threshold = similarity_threshold

        # 한국어 모델 선택
        if use_korean_model:
            model_name = "jhgan/ko-sroberta-multitask"
            logger.info("한국어 특화 모델 사용: jhgan/ko-sroberta-multitask")

        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self._model_ready = False
        self._init_lock = threading.Lock()

        # 키워드 임베딩 사전 계산
        self.keyword_embeddings: dict[str, np.ndarray] = {}

        # 비동기 초기화
        self._init_thread = threading.Thread(target=self._load_model, daemon=True)
        self._init_thread.start()

    def _load_model(self):
        """모델 비동기 로드"""
        try:
            with self._init_lock:
                if self._model_ready:
                    return

                logger.info(f"Sentence Transformer 모델 로딩 중: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self._model_ready = True
                logger.info("Sentence Transformer 모델 로드 완료")

                # 키워드 임베딩 사전 계산
                if self.keywords:
                    self._precompute_embeddings()

        except Exception as e:
            logger.error(f"Sentence Transformer 모델 로드 실패: {e}")

    def _precompute_embeddings(self):
        """키워드 임베딩 사전 계산"""
        if not self._model_ready or self.model is None:
            return

        try:
            logger.info(f"{len(self.keywords)}개 키워드 임베딩 사전 계산 중...")
            embeddings = self.model.encode(
                self.keywords, show_progress_bar=False, convert_to_numpy=True
            )

            for keyword, embedding in zip(self.keywords, embeddings):
                self.keyword_embeddings[keyword] = embedding

            logger.info("키워드 임베딩 사전 계산 완료")

        except Exception as e:
            logger.error(f"키워드 임베딩 사전 계산 실패: {e}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @lru_cache(maxsize=500)
    def _cached_encode(self, text: str) -> tuple[float, ...]:
        """임베딩 결과 캐싱 (튜플로 변환하여 hashable하게 만듦)"""
        if not self._model_ready or self.model is None:
            return tuple()

        try:
            embedding = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
            return tuple(embedding.tolist())
        except Exception as e:
            logger.warning(f"임베딩 실패: {e}")
            return tuple()

    def match(self, text: str) -> tuple[bool, str | None, EventType | None, float]:
        """
        텍스트에서 의미적으로 유사한 키워드 매칭
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            (매칭 여부, 대표키워드, EventType, 신뢰도) 튜플
        """
        if not text or not self._model_ready or self.model is None:
            return False, None, None, 0.0

        if not self.keyword_embeddings:
            # 아직 사전 계산이 안 된 경우
            return False, None, None, 0.0

        try:
            # 텍스트 임베딩 (캐시 사용)
            cached_result = self._cached_encode(text)
            if not cached_result:
                return False, None, None, 0.0

            text_embedding = np.array(cached_result)

            # 각 키워드와 코사인 유사도 계산
            best_similarity = 0.0
            best_keyword = None

            for keyword, keyword_embedding in self.keyword_embeddings.items():
                similarity = self._cosine_similarity(text_embedding, keyword_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_keyword = keyword

            # 임계값 확인
            if best_similarity >= self.similarity_threshold and best_keyword:
                event_type = self.event_type_map.get(best_keyword)
                return True, best_keyword, event_type, best_similarity

            return False, None, None, 0.0

        except Exception as e:
            logger.warning(f"의미 유사도 매칭 실패: {e}")
            return False, None, None, 0.0

    def update_keywords(
        self, keywords: list[str], event_type_map: dict[str, EventType]
    ):
        """키워드 목록 및 EventType 매핑 업데이트"""
        self.keywords = keywords
        self.event_type_map = event_type_map
        # 캐시 초기화
        self._cached_encode.cache_clear()
        # 임베딩 재계산
        if self._model_ready:
            self._precompute_embeddings()

