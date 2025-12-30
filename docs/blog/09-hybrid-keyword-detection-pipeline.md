# [SentinelPipeline] 하이브리드 키워드 감지 파이프라인

현재 진행 중인 SentinelPipeline 프로젝트의 핵심 기능 중 하나는 오디오 기반 위험 감지입니다. 기본적으로 ResNet을 이용해 비명(Scream) 소리를 감지하지만, 비명이 아닌 "살려줘", "도와주세요" 같은 긴급 키워드(Spoken Keywords)도 놓치지 않아야 합니다.

처음에는 단순하게 Whisper(STT) -> String Matching으로 구현했으나, 실시간성을 유지하려다 보니 정확도가 떨어지고, 정확도를 높이려다 보니 속도가 느려지는 딜레마에 빠졌습니다. 오늘은 이 문제를 해결하기 위해 도입한 **'3단계 하이브리드 감지 아키텍처'**와 'Java 없는 형태소 분석기 Kiwi' 도입기를 정리해봅니다.

## 1. 문제 상황 (Problem)

초기 파이프라인은 다음과 같았습니다.

```
[음성] -> [ResNet 비명감지 모델] -> [Whisper Tiny 모델] -> [단순 텍스트 포함 여부 확인]
```

하지만 현장에서 테스트해보니 세 가지 치명적인 문제가 발생했습니다.

- **Whisper 경량 모델의 한계**: "불이야"를 "뿌리야"로, "도와줘"를 "도와조"로 인식하는 등 발음이 조금만 부정확해도 오탐(False Negative)이 발생했습니다.
- **한국어의 교착어 특성**: 컴퓨터 입장에서 "살려"와 "살려주세요", "살려주시라요"는 완전히 다른 문자열입니다. 단순 매칭(`if keyword in text`)으로는 수많은 어미 변형을 커버할 수 없었습니다.
- **무거운 의존성 (Java)**: 한국어 처리를 위해 KoNLPy(Okt)를 쓰려니, 파이썬 프로젝트에 JDK(Java)를 설치해야 했습니다. 이는 Docker 이미지 크기를 비대하게 만들고 배포 복잡도를 높였습니다.

## 2. 해결 전략 (Solution)

### 2.1 형태소 분석기 교체: Why Kiwi?

기존의 KoNLPy 대신 **Kiwi(kiwipiepy)**를 도입했습니다.

- **No Java**: C++로 작성되어 빠르고, JVM 설치가 필요 없습니다. (Docker 이미지가 훨씬 가벼워집니다.)
- **Typo Correction**: `normalize_coda` 옵션을 켜면 "했대" -> "했데" 같은 받침/맞춤법 오류를 기가 막히게 잡아줍니다. STT 후처리 용도로 제격입니다.

### 2.2 3단계 계층형 아키텍처 (Tiered Approach)

모든 텍스트에 대해 무거운 연산을 돌릴 순 없습니다. 가장 빠르고 싼 연산부터 순차적으로 실행하는 깔때기(Funnel) 구조를 설계했습니다.

![3단계 하이브리드 아키텍처](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FbWjpJk%2FdJMcahXnsqi%2FAAAAAAAAAAAAAAAAAAAAACbjKu4dLK4O04sbfBV4v80fyvlED1PG0jbmSB9lFBCp%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3D4%252BhWLAJnYylzMil5ZBDcx4u7Xds%253D)

- **1단계 (Fast Path)**: 해시맵(Dict) 조회. O(1) 속도. 90%의 명확한 케이스 처리.
- **2단계 (Medium Path)**: Kiwi 형태소 분석 + 레벤슈타인 거리. 오타 및 어미 변형 처리.
- **3단계 (Heavy Path)**: 임베딩 벡터 유사도. 문맥적 의미 파악. (리소스 문제로 비동기 처리)

## 3. 구현 상세 (Implementation)

### 3.1 Fast Path: 역인덱싱(Inverted Index) 최적화

단순히 for 루프를 돌면 키워드가 늘어날수록 느려집니다. 이를 방지하기 위해 `{ "변형된단어": "대표키워드" }` 형태로 Flat한 Lookup Table을 미리 만들어둡니다.

```python
# src/infrastructure/audio/processors/keyword_matcher.py

class ExpandedKeywordDict:
    """확장 키워드 사전 - O(1) 조회를 위한 Fast Path"""

    def __init__(self, dict_path: str):
        self.lookup_table = {}
        self._load_and_flatten(dict_path)

    def _load_and_flatten(self, dict_path):
        # JSON 예: {"구조요청": ["살려줘", "도와주세요", "사람살려"]}
        # 이를 {"살려줘": "구조요청", "도와주세요": "구조요청"...} 으로 변환
        with open(dict_path, 'r') as f:
            data = json.load(f)
            for base, variations in data.items():
                self.lookup_table[base] = base
                for v in variations:
                    self.lookup_table[v] = base

    def contains_keyword(self, text: str):
        # 가장 빠른 완전 일치 조회
        if text in self.lookup_table:
            return True, self.lookup_table[text]
        # 필요시 부분 문자열 로직 추가
        return False, ""
```

### 3.2 Medium Path: Kiwi + 편집 거리

1단계에서 걸러지지 않은 "살려조"(오타) 같은 케이스를 잡습니다. 여기서 Kiwi의 정규화 능력이 빛을 발합니다.

```python
# src/infrastructure/audio/processors/morphological_matcher.py

from kiwipiepy import Kiwi
from Levenshtein import distance

class MorphologicalMatcher:
    def __init__(self):
        # sbg: 속도 최적화 모델 사용
        self.kiwi = Kiwi(model_type='sbg')
        # ... 키워드 전처리 로직 ...

    def match(self, text: str):
        # 1. Kiwi 정규화: "살려조" -> "살려줘" 등으로 자동 보정 시도
        tokens = self.kiwi.tokenize(text, normalize_coda=True)

        # 2. 핵심 형태소(명사/동사/형용사)만 추출하여 노이즈 제거
        normalized_text = "".join([t.form for t in tokens if t.tag.startswith(('N', 'V', 'V'))])

        # 3. 편집 거리(Levenshtein) 계산
        # 텍스트 길이에 따라 유동적인 임계값(Threshold) 적용
        for keyword in self.keywords:
            dist = distance(normalized_text, keyword)
            if dist <= self._calculate_threshold(keyword):
                return True, keyword
        return False, ""
```

### 3.3 Heavy Path: 임베딩 벡터 유사도

문맥적 의미를 파악하기 위해 Sentence Transformers를 사용합니다. 경량 모델(`all-MiniLM-L6-v2`)을 사용하여 메모리 효율성을 확보했습니다.

```python
# src/infrastructure/audio/processors/semantic_matcher.py

from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSimilarityMatcher:
    def __init__(self):
        # 경량 모델 사용 (약 80MB)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # 키워드 임베딩 사전 계산
        self.keyword_embeddings = self.model.encode(self.keywords)

    def match(self, text: str):
        # 텍스트 임베딩
        text_embedding = self.model.encode(text)
        
        # 코사인 유사도 계산
        similarities = np.dot(self.keyword_embeddings, text_embedding) / (
            np.linalg.norm(self.keyword_embeddings, axis=1) * np.linalg.norm(text_embedding)
        )
        
        # 임계값 확인
        if similarities.max() >= self.similarity_threshold:
            return True, self.keywords[similarities.argmax()]
        return False, ""
```

## 4. 통합 하이브리드 감지기

3단계를 순차적으로 호출하는 깔때기 구조입니다.

```python
# src/infrastructure/audio/processors/hybrid_keyword_detector.py

class HybridKeywordDetector:
    def detect(self, text: str):
        # 1단계: Fast Path (O(1) 조회)
        matched, keyword = self.fast_path.contains_keyword(text)
        if matched:
            return {'is_dangerous': True, 'keyword': keyword, 'path': 'fast'}
        
        # 2단계: Medium Path (형태소 분석 + 편집 거리)
        matched, keyword, confidence = self.medium_path.match(text)
        if matched:
            return {'is_dangerous': True, 'keyword': keyword, 'path': 'medium', 'confidence': confidence}
        
        # 3단계: Heavy Path (의미 유사도) - 비동기 처리
        # 실시간 알림은 1-2단계에서 처리
        # 3단계는 후처리 분석용
        
        return {'is_dangerous': False}
```

## 5. 데이터 흐름

```
[Whisper STT] → 텍스트
    ↓
[HybridKeywordDetector]
    ├─ [1단계: ExpandedKeywordDict] → O(1) 조회
    │   └─ 매칭 성공 → 즉시 반환 (90% 케이스)
    │
    ├─ [2단계: MorphologicalMatcher] → Kiwi + Levenshtein
    │   └─ 매칭 성공 → 반환 (오타/어미 변형)
    │
    └─ [3단계: SemanticSimilarityMatcher] → 임베딩 + 코사인
        └─ 비동기 처리 (후처리 분석)
```

## 6. 핵심 기술 요약

### 6.1 한국어 처리 파이프라인

```
STT 텍스트 입력
    ↓
[Kiwi 형태소 분석]
    ├─ normalize_coda: 받침/맞춤법 오류 자동 보정
    ├─ 조사/어미 제거
    ├─ 핵심 형태소 추출 (명사/동사/형용사)
    └─ 기본형 정규화
    ↓
[벡터화 (임베딩) - 3단계에서만]
    ├─ 텍스트 → 숫자 벡터 변환
    └─ 의미 공간에 배치
    ↓
[유사도 계산]
    ├─ 레벤슈타인 거리 (발음 유사성, 2단계)
    ├─ 코사인 유사도 (의미적 유사성, 3단계)
    └─ 자카드 지수 (단어 집합 유사성, 선택적)
    ↓
[위험 키워드 매칭]
```

### 6.2 성능 최적화 전략

1. **오프라인 사전 구축**: 키워드 확장은 미리 수행하여 JSON 파일로 저장
2. **역인덱싱 최적화**: Flat Lookup Table로 O(1) 조회 속도 보장
3. **임베딩 사전 계산**: 키워드 임베딩은 초기화 시 한 번만 계산
4. **배치 처리**: 의미 기반 매칭은 배치로 처리하여 GPU 효율성 향상
5. **캐싱**: 형태소 분석 결과 캐싱 (LRU Cache)
6. **하이브리드 접근**: 빠른 경로 우선, 필요 시에만 무거운 처리
7. **비동기 처리**: 3단계는 실시간 블로킹 없이 백그라운드 처리

## 7. 결론

이번 리팩토링을 통해 얻은 성과는 다음과 같습니다.

- **Latency 개선**: 90% 이상의 케이스가 1단계(Fast Path)에서 처리되어 1ms 이내에 판별됩니다.
- **정확도 향상**: Kiwi의 정규화 덕분에 STT 오인식 상황에서도 강건하게 키워드를 잡아냅니다.
- **배포 간소화**: Java 의존성 제거로 Docker 이미지 크기 감소 및 배포 프로세스 단순화.

## 8. 참고 자료

- **Kiwi (kiwipiepy)**: https://github.com/bab2min/kiwipiepy
  - C++ 기반 고성능 한국어 형태소 분석기
  - Java 의존성 없음, normalize_coda 옵션으로 오타 보정
- **Levenshtein 거리**: https://en.wikipedia.org/wiki/Levenshtein_distance
  - 문자열 편집 거리 계산 알고리즘
- **Sentence Transformers**: https://www.sbert.net/
  - 문장 임베딩 및 유사도 계산
- **한국어 임베딩 모델**: https://github.com/jhgan00/ko-sroberta-multitask
  - 한국어 특화 Sentence Transformer 모델
- **코사인 유사도**: https://en.wikipedia.org/wiki/Cosine_similarity
  - 벡터 공간에서의 의미적 유사성 측정

