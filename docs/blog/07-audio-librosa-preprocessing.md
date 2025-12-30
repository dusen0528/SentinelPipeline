# [Python] 음성 데이터와 Librosa를 활용한 데이터 전처리

## 우리가 듣는 소리는 공기의 진동인 아날로그 신호

컴퓨터는 0과 1밖에 모르는 디지털 기계입니다.

이러한 아날로그 파동을 컴퓨터가 이해할 수 있는 숫자로 바꾸는 과정이 있는데 이를 **ADC(Analog-to-Digital Conversion)** 라고 합니다.

이 과정에서 가장 중요한 개념이 바로 **샘플링**인데:

- 매끄러운 곡선(아날로그 소리)을 일정한 간격으로 점을 찍어(샘플링) 계단 모양의 데이터로 만드는 과정입니다.

영상의 경우 1초에 24장, 1초의 60장 등.. 여러장의 사진을 연속적으로 보여주면 움직이는 것처럼 보이는데, 소리도 이게 존재합니다.

소리를 자연스럽게 저장하려면 1초에 몇번정도의 샘플을 찍어야 할까요?

→ 과학자들의 연구결과 어떤 소리를 "정확하게" 기록하기 위해선, 그 소리의 주파수보다 최소 2배는 더 자주 점을 찍어야 한다는 법칙을 발견했습니다. (**나이퀴스트 이론**)

인간의 귀는 1초에 2만번 진동하는 고음을 들을 수 있고 이론에 따르면 최소 2배 더 자주 찍어야합니다.

그래서 오디오 CD의 표준은 **44,100Hz**로 정해져있고 덕분에 우리는 소리를 컴퓨터에 숫자로 완벽하게 저장할 수 있게 됐습니다.

## 주파수

### 시간 vs 주파수

- **시간 영역 (햇빛)**: 그냥 눈으로 보면 하얀 빛 한줄기입니다. 그안에 무슨색이 있는지 알기 힘듭니다.
- **주파수 영역(무지개)**: 프리즘을 통과시키면 빛이 빨/주/노/초 .. 로 분리되며, 이 빛에는 빨간색이 많이 들어있구나!를 분석할 수 있게 됩니다.

소리도 마찬가지로 복잡한 소리 파형을 수학적인 프리즘(푸리에 변환, FFT 등..)에 통과 시키면, 그 소리가 저음(낮은 주파수)인지 고음(높은 주파수)로 이루어져있는지 분해할 수 있습니다.

### 그렇다면 비명은?

비명은 보통 찢어지는 듯한 소리인데, 이를 주파수 영역으로 분해해서 보면 어떤 주파수 성분이 강하게 나타날까요?

→ 당연히 **고주파 영역에서 에너지가 폭발하는 특징**이 있습니다.

AI로 이를 탐지 하기 위해서는 소리를 그림으로 바꿔서 AI에게 보여주는 방법이 있습니다.

이 그림을 **스펙트로그램**이라고 합니다.

### 스펙트로그램

![스펙트로그램 설명](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FkCy5J%2FdJMcagYoKCX%2FAAAAAAAAAAAAAAAAAAAAAI-s8_MlIGMSnwQ1D3ecxewY6azTCDNe19mzZDFBrysh%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3D4CVC7efWyo68G3dKKbEnJAsnUYU%253D)

- **가로축 (X축)**: 시간의 흐름 (왼쪽에서 오른쪽으로)
- **세로축 (Y축)**: 소리의 높낮이 (주파수) 위로 갈수록 고음 ⬆️ 아래로 갈수록 저음 ⬇️
  - 위로 갈수록 고음 ⬆️
  - 아래로 갈수록 저음 ⬇️
- **색깔의 진하기**: 소리의 크기 (진할수록 시끄러움)

그럼 일상생활에서 들리는 소리 중에, 사람 비명은 아닌데 '고음'이고 '시끄러운' 소리가 무엇이 있을까요?

| 소리 종류 | 특징 | 스펙트로그램 모양 (상상해보기) |
|---------|------|---------------------------|
| 비명 😱 | "끼아악!" 하고 불규칙하게 떨림 | 윗부분에 진한 선이 있는데, 선이 지저분하고 떨리는 모양 |
| 사이렌 🚨 | "위잉위~잉" 하고 규칙적임 | 윗부분에 진한 선이 아주 매끄럽게 오르락내리락하는 모양 |

AI는 단순히 "위에 색깔이 진하네?"만 보는 게 아니라, "그 진한 선이 지저분한가(비명)? 매끄러운가(사이렌)?" 라는 패턴(무늬)을 보고 판단해야 합니다.

## 데이터셋을 활용해 학습 하기

[Kaggle Dataset: Human Screaming Detection Dataset](http://www.kaggle.com/datasets/what2000/human-screaming-detection-dataset/data)

나이퀴스트 이론에 따라 **44100 RATE**로 샘플링 진행하기

### 만약 낮다면?

![낮은 샘플링 레이트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FbpH54q%2FdJMcagjNgMy%2FAAAAAAAAAAAAAAAAAAAAAJaErFk4e__wTZ1eFbX9mu9hF60QxpIpDSlQvsHP8jxM%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DDdezS9wC9fAnuRW9vdD%252BGIRbTgY%253D)

- 이와 같이 파란선(디지털)이 빨간선(원본)을 잘 따라가지 못함

### 만약 보통이라면?

![보통 샘플링 레이트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FOPeCl%2FdJMcagjNgNk%2FAAAAAAAAAAAAAAAAAAAAAFet1e4oO3m_CJLTWcRPsi6RojVnQ5M7lbqr_nkXAlce%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3D23Z5cvQW2XLAQZ3oDqVtEzRYVHw%253D)

- 어느정도 비슷하지만 아직 디테일이 부족

### 만약 높다면?

![높은 샘플링 레이트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2F8871N%2FdJMcagYoKDt%2FAAAAAAAAAAAAAAAAAAAAADksmslmG_0JkkhU_YORsa4MM8uGPohSqPdTtnTAgxcI%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DeiZb1ueI6PUBEMWrd2bWHPnDdaU%253D)

- 샘플이 많아 디지털소리가 원본 소리와 거의 완벽하게 일치

## 오디오 특징 추출하는 함수

### 1. MFCC : 목소리의 지문

우리는 보통 친한 친구의 목소리를 들으면 바로 알 수 있다. 이게 가능한 이유는 우리의 귀와 뇌는 친한 친구의 고유한 톤과 질감을 기억하기 때문입니다.

**MFCC**는 바로 이 고유한 톤과 질감을 컴퓨터가 알아볼 수 있는 숫자로 바꿔주는 기술입니다.

"Mel Scale" 이라는 인간 청각 구조를 반영한 주파수 단계를 사용합니다. 단순히 높낮이가 아니라, "누가" 내는 소리인지, "어떤 발음"인지(아 vs 으)를 파악하는 데 쓰입니다.

**비명 탐지시에는:**

- 비명은 일반 대화와 달리 성대가 극도로 긴장된 상태의 독특한 스펙트럼 형태(지문)를 가집니다.

### 2. Spectral Contrast (스펙트럼 대비)

소리의 선명도를 의미합니다.

우리가 소리를 주파수별 (가로축)으로 쪼개서 에너지를 보면 에너지가 가장 강한 곳(산봉우리, Peak)이 있고 약한 곳(골짜기, Valley)가 있습니다.

스펙트럼 대비는 가장 높은 산봉우리와 그 옆의 깊은 골짜기 사이의 격차를 계산한 값입니다.

- 그래프에서 파란선은 소리의 에너지가 가장 강한 부분(산꼭대기)
- 빨간색 점선은 소리의 에너지가 가장 약한 부분 (골짜기 바닥)
- 여기서 스펙트럼 대비는 파란선과 빨간선 사이의 간격입니다.

![Spectral Contrast 그래프](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FUlwJr%2FdJMcaiu6LWx%2FAAAAAAAAAAAAAAAAAAAAADD229jYacjqvHSvVG8BumQNzIWVb1Z4bOpLcCwv_oWi%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DmAgwDVYm1cp48MFVwhUC0hytOEo%253D)

**대비가 큼 (간격이 넓음) = 맑고 또렷한 소리 🎵**
- 예: 피아노 소리, 맑은 노랫소리, 휘파람.
- 이런 소리는 특정 음(주파수)만 확실히 강하고, 나머지는 조용합니다. 산은 높고 골짜기는 깊어서 대비가 큽니다.

**대비가 작음 (간격이 좁음) = 탁하고 거친 소리 📢**
- 예: 비명 소리, 치지직거리는 노이즈, 폭포 소리.
- 비명은 목을 긁으면서 내는 소리라, 특정 음만 나는 게 아니라 주변의 잡음(골짜기)까지 시끄럽게 채워버립니다. 그래서 산과 골짜기의 차이가 줄어듭니다.

```python
librosa.feature.spectral_contrast(y=y, sr=RATE)
```

### 3. Chroma Feature

![Chroma Feature 그래프](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FchNBWI%2FdJMcaaYbQH6%2FAAAAAAAAAAAAAAAAAAAAAAQmjlNuaIJDaLvMrCebod2vQwUGh0ypdyCGwYtmzzzV%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DdhU%252F9P%252B5tZHwbUyy%252FmZG9xFAomw%253D)

크로마 특징은 소리의 음계 성분을 분석하는 것입니다.

쉽게 말하면 이 소리가 피아노 건반중 어떤 음에 가까운가를 알려주는 지표입니다.

- **비명 (Scream)**: 비명은 보통 특정한 고음을 길게 유지하죠? ("끼아아악!" 할 때 특정 음정이 유지됨) 그래서 Chroma 그래프를 보면 특정 음계 부분이 진하게 나옵니다.
- **말소리 (Speech)**: 말할 때는 음의 높낮이가 계속 변합니다. Chroma가 정신없이 바뀝니다.
- **소음 (Noise)**: 바람 소리나 자동차 소음은 특정 음정이 없습니다. (피아노 건반을 쾅! 하고 다 누른 것과 같음) 그래서 Chroma가 전체적으로 흐릿하게 퍼져 나옵니다.

> "MFCC가 '목소리 지문', Spectral Contrast가 '목소리 맑기'라면, Chroma는 '소리의 음정(도레미...)'을 봅니다."

비명은 음악적인 조성이 거의 없습니다.

반면 배경음악이나 노래는 특정 키(Key)가 뚜렷합니다. 이를 통해 오경보(음악을 비명으로 착각)를 줄이는 데 도움을 줍니다.

### 4. Zero Crossing Rate (영교차율)

![Zero Crossing Rate 파형](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2F5fHMe%2FdJMcacVZ12w%2FAAAAAAAAAAAAAAAAAAAAAMm4RZKbwfbUCemTnGzrqSL6anAftOoy3Mdp6bSLefSB%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3Dq1sFpq87EOTIUWJN3eaZFRwXKik%253D)

파형이 "0"을 지나는 횟수 (거칠기/노이즈)를 의미합니다.

파형 그래프가 +에서 -로, -에서 +로 얼마나 자주 바뀌는지 셉니다.

부드러운 소리(저음)는 천천히 바뀌고, 치찰음(쉿, 칙)이나 타악기 소리는 엄청 자주 바뀝니다.

**비명 탐지시에는:**

- **비명 (High ZCR)**: "끼아아악!" 하는 비명은 단순히 높은 음일뿐만 아니라, 목을 긁으면서 내는 아주 거칠고 지저분한 소리입니다. 파형이 자글자글하게 떨리기 때문에 0을 엄청 자주 건너갑니다.
- **노래/말소리 (Low ZCR)**: 반면, "아~" 하고 맑게 노래를 부르거나 부드럽게 말할 때는 파형이 매끄럽습니다. 0을 건너가는 횟수가 상대적으로 적습니다.

> Zero Crossing Rate는 '소리의 거칠기'를 봅니다.

AI는 이 값이 높게 나오면 "어? 소리가 되게 거칠고 자글자글하네? 비명일 확률이 높아!" 라고 판단합니다.

### 5. Spectral Centroid (스펙트럼 중심)

![Spectral Centroid 그래프](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FkTrtV%2FdJMcajneUqH%2FAAAAAAAAAAAAAAAAAAAAADBwdueHJAWioZzntX5vd7syXsd69LOXGpqxvqxDZjpO%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DiFwdYIwEH95aCEVMAbdxSv8c5BU%253D)

주파수 성분의 평균 위치입니다.

- 값이 높다 → 고음이 많다라는 의미로 소리가 밝고 날카롭게 들리고
- 값이 낮다 → 저음이 많다라는 의미로 소리가 둔하고 무겁게 들립니다.

**비명 탐지시에는:**

비명은 고주파 성분이 많기 때문에 Spectral Centroid 값이 높게 나타납니다.

### 6. Spectral Bandwidth (스펙트럼 대역폭)

![Spectral Bandwidth 그래프](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FFmfoL%2FdJMcafkTgXr%2FAAAAAAAAAAAAAAAAAAAAANp9jf0x92HJ2y7HKaD9hztXF6a9vi9rVyVT-KqzY_NO%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DIpug8fdJGdz5LvwK3A9hlJ878KQ%253D)

주파수 성분이 얼마나 넓게 퍼져있는지를 나타냅니다.

- 값이 높다 → 다양한 주파수가 섞여있음 (잡음이 많음)
- 값이 낮다 → 특정 주파수에 집중되어 있음 (맑고 깨끗한 소리)

**비명 탐지시에는:**

비명은 다양한 주파수 성분이 섞여있어서 Spectral Bandwidth 값이 높게 나타납니다.

### 7. RMSE (Root Mean Square Energy)

![RMSE 그래프](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FJ2945%2FdJMcaaYbQJX%2FAAAAAAAAAAAAAAAAAAAAAKwWom0q61B9KNEhvDxSlSi8UY4-aC2ECOsI6TA2DYJp%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DpLU9WENnBj6udrwjrPg2RqwRPWc%253D)

소리의 전체적인 에너지(크기)를 나타냅니다.

- 값이 높다 → 소리가 크다
- 값이 낮다 → 소리가 작다

**비명 탐지시에는:**

비명은 일반적으로 큰 소리이기 때문에 RMSE 값이 높게 나타납니다.

### 8. Spectral Roll-off

주파수 스펙트럼에서 에너지의 85%가 집중되어 있는 주파수 지점입니다.

- 값이 높다 → 고주파 성분이 많다
- 값이 낮다 → 저주파 성분이 많다

**비명 탐지시에는:**

비명은 고주파 성분이 많기 때문에 Spectral Roll-off 값이 높게 나타납니다.

### 9. Tempo (BPM)

소리의 리듬과 템포를 나타냅니다.

- 값이 높다 → 빠른 리듬
- 값이 낮다 → 느린 리듬

**비명 탐지시에는:**

비명은 일반적으로 일정한 리듬이 없고 불규칙한 패턴을 가지므로, Tempo 값이 일정하지 않거나 높게 나타날 수 있습니다.

## 왜 이렇게 많은 특징을 쓰나요?

하나의 특징만으로는 비명을 정확하게 구분하기 어렵습니다. 예를 들어:

- Spectral Centroid만 보면: 고음이 많은 사이렌도 비명으로 오인식될 수 있습니다.
- RMSE만 보면: 큰 소리만 비명으로 오인식될 수 있습니다.
- ZCR만 보면: 거친 소리만 비명으로 오인식될 수 있습니다.

하지만 여러 특징을 **조합**하면:

- 고주파(Spectral Centroid 높음) + 거친 소리(ZCR 높음) + 탁한 소리(Spectral Contrast 낮음) + 불규칙한 패턴(Tempo 불규칙) = **비명일 확률 높음**

이렇게 여러 특징을 종합적으로 분석하여 오경보를 줄이고 정확도를 높일 수 있습니다.

---

**참고 자료:**
- [Kaggle: Human Screaming Detection Dataset](http://www.kaggle.com/datasets/what2000/human-screaming-detection-dataset/data)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

