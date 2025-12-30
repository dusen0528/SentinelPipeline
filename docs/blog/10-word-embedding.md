# Word Embedding

## 목차

1. [One-Hot Encoding](#one-hot-encoding)
2. [희소 표현 vs 밀집 표현](#희소-표현-vs-밀집-표현)
3. [Word2Vec](#word2vec)
4. [GloVe](#glove)
5. [FastText](#fasttext)
6. [정적 임베딩의 한계](#정적-임베딩의-한계)
7. [유사도 측정 방법 (코사인 유사도)](#유사도-측정-방법-코사인-유사도)

---

## One-Hot Encoding

컴퓨터는 문자열을 직접 이해할 수 없기 때문에, 범주형 변수를 이진 형식으로 변환하는 방법이 필요합니다. **One-Hot Encoding**은 각 범주에 대해 새로운 열을 생성하여 1은 해당 범주가 존재함, 0은 해당 범주가 존재하지 않음을 나타냅니다.

| **단어** | **원-핫 인코딩 표현** | **설명** |
|---------|-------------------|---------|
| **사과** | \[1, 0, 0\] | 첫 번째 자리만 1 (Hot) |
| **배** | \[0, 1, 0\] | 두 번째 자리만 1 (Hot) |
| **포도** | \[0, 0, 1\] | 세 번째 자리만 1 (Hot) |

원-핫 인코딩의 가장 큰 문제는 단어 사이의 **관계**를 알 수 없다는 특징이 있습니다.

- 예: 사과 `[1, 0, 0]`, 배 `[0, 1, 0]`
- 컴퓨터 입장에서 이 두 벡터는 그냥 완전히 다른 숫자일 뿐입니다. (직교하므로 내적값 0)

그래서 **워드 임베딩**은 단어를 **벡터로 표현**하는 방법으로 단어를 밀집 표현으로 변환합니다.

---

## 희소 표현 vs 밀집 표현

### 희소 표현 (Sparse Representation)

행렬이나 벡터의 값 대부분이 0으로 채워진 표현 방식을 말합니다.

이러한 희소 벡터의 경우 다음과 같은 문제점이 있습니다.

**문제점 1: 극심한 공간 낭비 (차원의 저주)**

- **차원 폭발:** 표현하고 싶은 단어가 늘어날수록 벡터의 크기(차원)도 정비례해서 무한정 커집니다.
  - *예) 단어 10,000개 → 벡터 길이 10,000 (9,999개의 0이 낭비됨)*
- **비효율성:** 유의미한 값은 딱 하나인데 불필요한 `0`을 저장하기 위해 너무 많은 메모리 공간을 차지합니다. (DTM 또한 같은 이유로 비효율적인 희소 행렬입니다.)

**문제점 2: 단어의 '의미'를 담지 못함**

- 단순히 순서(인덱스)에 따라 좌표만 찍어줄 뿐, 단어 간의 **유사성이나 관계, 속뜻을 전혀 표현하지 못합니다.**
  - *예) '강아지'와 '개'는 비슷한 단어지만, 원-핫 벡터 상에서는 완전히 다른(직교하는) 남남으로 취급됩니다.*

### 밀집 표현 (Dense Representation)

희소 표현과는 반대로 밀집 표현이 있는데, 이는 벡터의 차원을 단어 집합의 크기로 정하는 것이 아닌 사용자가 설정한 값으로 모든 단어의 벡터 표현 차원을 맞춥니다.

이 과정에서 0과 1이 아닌 실수 값을 가지게 됩니다.

만약 밀집 표현을 사용해 차원을 128로 지정한다면:

> **희소 표현**  
> Ex) 강아지 = \[ 0 0 0 0 1 0 0 0 0 0 0 0 ... 중략 ... 0\] # 이때 1 뒤의 0의 수는 9995개. 차원은 10,000

> **밀집 표현**  
> Ex) 강아지 = \[0.2 1.8 1.1 -2.1 1.1 2.8 ... 중략 ...\] # 이 벡터의 차원은 128

이 때 벡터 차원이 밀집하게 된다 하여 밀집 벡터라고 합니다.


---

## Word2Vec

단어를 밀집 벡터로 가장 대표적인 방법은 구글의 **Word2Vec**입니다.

- 비슷한 위치에 등장하는 단어들은 비슷한 의미를 가진다

이러한 Word2Vec의 학습 방식이 2가지가 존재합니다.

### 1) CBOW (Continuous Bag of Words)

- **방식:** 주변 단어들을 보고 **중간에 있는 단어**를 맞추는 문제입니다.
- 예시: "The fat **cat** sat on the mat"
  - 입력: `The`, `fat`, `sat`, `on`
  - 맞춰야 할 정답: `cat`
- 특징: 학습이 빠르지만, 드물게 나오는 단어에는 약할 수 있습니다.
![1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2Fllinc%2FdJMcaf6jddY%2FAAAAAAAAAAAAAAAAAAAAAPBvEXu-UuvE3D2E8C6H6oolm6TjgD-BrEOVl7EpAfUp%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DPlNXAbVgI8PE%252BwskbJHBB1iPdqU%253D)

![2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FlKTGU%2FdJMcaf6jdd8%2FAAAAAAAAAAAAAAAAAAAAAKLgJ4bhEFAVq4-JdFARefAoTnfCbFWvE75AhVxSF_u6%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DtSSIbjWntxbp4yUM35oSt9wNQaE%253D)
### 2) Skip-gram

- **방식:** 중간에 있는 단어를 보고 **주변 단어들**을 맞추는 문제입니다. (CBOW의 반대)
- 예시:
  - 입력: `cat`
  - 맞춰야 할 정답: `The`, `fat`, `sat`, `on`
- 특징: 전반적으로 성능이 더 좋아서 **가장 많이 사용**됩니다.

> 윈도우 크기가 정해지면 윈도우를 옆으로 움직여서 주변 단어와 중심 단어의 선택을 변경해가며 학습을 위한 데이터 셋을 만드는데 이 방법을 **슬라이딩 윈도우(sliding window)**라고 합니다.

![3](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FbIgQ7e%2FdJMcaf6jdei%2FAAAAAAAAAAAAAAAAAAAAAPqgezRbsduVv42bfk2XbyzXhgEJJhur6aPzZ9AHA2XK%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3D3gmxVOP4bWr%252FsQB7bY1kfIFzJ14%253D)

![4](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2Fcol0X7%2FdJMcaf6jdet%2FAAAAAAAAAAAAAAAAAAAAAElgftNYNASb5LbyKtonvEB9m0pAgME2SnsOdV3rqfr3%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3D9QVE2LT%252BRFXw0koraA%252FjU9J1Lpg%253D)
---

## GloVe

Word2Vec은 주변 단어(Window)만 봅니다.

즉, 문장 하나하나를 훑으면서 학습하느라 전체적인 통계 정보(이 단어가 전체 문서에서 몇 번 나왔는지 등)는 좀 소홀히 하는 경향이 있습니다.

- 기존의 **카운트 기반**(전체 통계, LSA) 방법과 **예측 기반**(Word2Vec) 방법의 장점을 합쳤습니다.
- "단어의 동시 등장 확률"을 계산해서 학습합니다.

GloVe가 똑똑한 이유는 단순히 같이 나왔다가 아니라 **얼마나 뚜렷하게 관계가 있는가?**를 따지기 때문입니다.

예를 들어 **얼음(Ice)**과 **증기(Steam)**는 둘 다 물과 관련이 있지만, 얼음은 고체, 증기는 기체로 서로 다른 상태입니다. GloVe는 이런 미묘한 차이를 학습할 수 있습니다.

![GloVe 설명](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2Fcol0X7%2FdJMcaf6jdet%2FAAAAAAAAAAAAAAAAAAAAAElgftNYNASb5LbyKtonvEB9m0pAgME2SnsOdV3rqfr3%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3D9QVE2LT%252BRFXw0koraA%252FjU9J1Lpg%253D)

---

## FastText

Facebook에서 개발한 FastText는 Word2Vec의 확장 버전으로, 단어를 **글자 단위(subword)**로 쪼개서 학습합니다.

### FastText의 핵심 아이디어

- 단어를 글자 단위로 분해하여 학습
- 예: "eating" → `e`, `ea`, `eat`, `eati`, `eating`, `ing`, `ng`, `g`
- 이렇게 하면 학습 데이터에 없는 단어(Out-of-Vocabulary, OOV)도 처리할 수 있습니다.

### FastText의 장점

**1. OOV 문제 해결**

- Word2Vec: "먹고"라는 단어를 학습하지 않았으면 처리 불가
- FastText: "먹"과 "고"라는 글자 조합을 학습했기 때문에, "먹고"가 나와도 먹이라는 공통 분모를 찾아내 이 단어는 현재 먹는 것과 관련된 거구나 라는 방식으로 유추합니다.

**한국어에서 FastText를 사용한다면**

- 한국어 동사의 활용은 끝이 없는데, 어간(뿌리)만 살아있으면 뜻을 파악합니다. (예: `가(go)`만 알면 `가니`, `가고`, `가서` 다 이해)
- 오타 교정
  - Word2Vec: "안녕ㅎ세요? 누구세요?"
  - FastText: "`안녕`이랑 `세요`는 살아있네? `ㅎ`는 오타인가 보다. 인사로 처리하자."
- 모든 변형을 다 수집하지 않아도, 기본 블록(글자, 자소)만 잘 학습되면 파생어들을 다 처리할 수 있습니다.

---

## 정적 임베딩의 한계

Word2Vec, GloVe, FastText는 모두 **단어 하나당 하나의 벡터값만** 가집니다.  
이를 정적 임베딩이라고 하는데, 치명적인 단점이 하나 있습니다.

**문제점: 문맥(Context)을 파악하지 못함 (동음이의어)**

- 같은 단어라도 문장에 따라 뜻이 달라지는데, 이를 구분하지 못합니다.
- **예:**
  - "맛있는 **배**를 먹었다" (과일) → 벡터 \[0.2, 0.5\]
  - "항구에서 **배**를 탔다" (탈것) → 벡터 \[0.2, 0.5\]
- 컴퓨터는 문맥을 보지 않고 글자 모양만 보기 때문에, 이 두 '배'를 완전히 똑같은 벡터(의미)로 저장해버립니다.
- (참고: 이 문제를 해결하기 위해 나중에 문맥을 보는 BERT, GPT 같은 모델이 등장합니다.)

---

## 유사도 측정 방법 (코사인 유사도)

단어를 밀집 벡터로 바꾼 후, **그래서 단어끼리 얼마나 비슷한데?**를 계산하는 방법입니다.  
가장 많이 쓰는 방식은 코사인 유사도(Cosine Similarity)입니다.

**원리: 거리(Distance)가 아닌 각도(Angle)를 봅니다.**

- 두 벡터가 가리키는 **방향이 얼마나 일치하는지**를 봅니다.
  - **1에 가까움:** 방향이 같음 (아주 유사함)
  - **0:** 90도 직교 (전혀 관계없음, 원-핫 인코딩 상태)
  - **-1:** 정반대 방향
- 단순히 거리를 재면, 단어의 빈도수(벡터의 크기)에 따라 오차가 생길 수 있기 때문에 **각도**를 보는 것이 더 정확합니다.

### 코사인 유사도 공식

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

여기서:
- `A · B`: 두 벡터의 내적
- `||A||`, `||B||`: 각 벡터의 크기(노름)

---
