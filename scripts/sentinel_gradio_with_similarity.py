#!/usr/bin/env python
"""
Sentinel AI 정밀 분석기 (Medium Path 유사도 검색 포함)

Fast Path (정확 일치) + Medium Path (발음 유사도) 적용

필요 패키지:
    pip install gradio faster-whisper kiwipiepy python-Levenshtein
"""

import os
import ssl
import time
import signal
import subprocess
import csv
import re
from datetime import datetime
from difflib import SequenceMatcher

import gradio as gr
from faster_whisper import WhisperModel

# ======================================================
# [설정] 모델 및 파일 경로
# ======================================================
MODEL_NAME = "large-v3-turbo" 
LOG_FILE = "stt_keyword_test_log.csv"

# ======================================================
# [키워드 사전] - variations 포함
# ======================================================
KEYWORDS_DATA = {
    "구조요청": {
        "variations": [
            "살려주세요", "살려줘", "사람살려", "구해줘", 
            "도와줘", "도와주세요", "에스오에스", "헬프", "세이브미"
        ],
        "event_type": "CUSTOM"
    },
    "화재": {
        "variations": [
            "불이야", "불났다", "화재", "연기발생", "타는냄새"
        ],
        "event_type": "FIRE"
    },
    "응급의료": {
        "variations": [
            "119", "일일구", "긴급", "구급차", "앰뷸런스", "응급실"
        ],
        "event_type": "SYSTEM_ALERT"
    },
    "신고": {
        "variations": [
            "신고", "일일이", "경찰", "112"
        ],
        "event_type": "SYSTEM_ALERT"
    }
}

# Flat lookup table 생성 (variation -> category)
KEYWORDS_MAP = {}
ALL_VARIATIONS = []
for category, info in KEYWORDS_DATA.items():
    for var in info["variations"]:
        KEYWORDS_MAP[var] = category
        ALL_VARIATIONS.append(var)

# ======================================================
# [한국어 발음 유사도] Medium Path
# ======================================================
try:
    from Levenshtein import ratio as levenshtein_ratio
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    print("[경고] python-Levenshtein이 설치되지 않았습니다. pip install python-Levenshtein")

# 한국어 자모
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
           'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
            'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
            'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
            'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 발음 유사 자모 그룹 (STT 오인식 패턴)
SIMILAR_CHOSUNG = {
    'ㄱ': ['ㄱ', 'ㄲ', 'ㅋ'], 'ㄲ': ['ㄱ', 'ㄲ', 'ㅋ'], 'ㅋ': ['ㄱ', 'ㄲ', 'ㅋ'],
    'ㄷ': ['ㄷ', 'ㄸ', 'ㅌ'], 'ㄸ': ['ㄷ', 'ㄸ', 'ㅌ'], 'ㅌ': ['ㄷ', 'ㄸ', 'ㅌ'],
    'ㅂ': ['ㅂ', 'ㅃ', 'ㅍ'], 'ㅃ': ['ㅂ', 'ㅃ', 'ㅍ'], 'ㅍ': ['ㅂ', 'ㅃ', 'ㅍ'],
    'ㅅ': ['ㅅ', 'ㅆ'], 'ㅆ': ['ㅅ', 'ㅆ'],
    'ㅈ': ['ㅈ', 'ㅉ', 'ㅊ'], 'ㅉ': ['ㅈ', 'ㅉ', 'ㅊ'], 'ㅊ': ['ㅈ', 'ㅉ', 'ㅊ'],
}

SIMILAR_JUNGSUNG = {
    'ㅐ': ['ㅐ', 'ㅔ', 'ㅒ', 'ㅖ'], 'ㅔ': ['ㅐ', 'ㅔ', 'ㅒ', 'ㅖ'],
    'ㅗ': ['ㅗ', 'ㅛ', 'ㅜ'], 'ㅛ': ['ㅗ', 'ㅛ'],
    'ㅜ': ['ㅜ', 'ㅠ', 'ㅗ'], 'ㅠ': ['ㅜ', 'ㅠ'],
    'ㅓ': ['ㅓ', 'ㅕ', 'ㅏ'], 'ㅕ': ['ㅓ', 'ㅕ'],
    'ㅏ': ['ㅏ', 'ㅑ', 'ㅓ'], 'ㅑ': ['ㅏ', 'ㅑ'],
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
        if char in SIMILAR_CHOSUNG:
            result.append(SIMILAR_CHOSUNG[char][0])
        elif char in SIMILAR_JUNGSUNG:
            result.append(SIMILAR_JUNGSUNG[char][0])
        else:
            result.append(char)
    return ''.join(result)


def korean_phonetic_similarity(text1: str, text2: str) -> float:
    """한국어 발음 유사도 계산 (0.0 ~ 1.0)"""
    if not LEVENSHTEIN_AVAILABLE:
        # Fallback: 단순 문자열 유사도
        return SequenceMatcher(None, text1, text2).ratio()
    
    # 공백/특수문자 제거
    clean1 = re.sub(r'[^\w가-힣]', '', text1)
    clean2 = re.sub(r'[^\w가-힣]', '', text2)
    
    # 자모 분리
    jamo1 = decompose_korean(clean1)
    jamo2 = decompose_korean(clean2)
    
    # 발음 정규화
    norm1 = normalize_pronunciation(jamo1)
    norm2 = normalize_pronunciation(jamo2)
    
    return levenshtein_ratio(norm1, norm2)


def detect_keyword_hybrid(stt_text: str, threshold: float = 0.65):
    """
    하이브리드 키워드 감지 (Fast Path + Medium Path)
    
    Returns:
        (keyword, category, path, confidence)
    """
    if not stt_text:
        return None, None, "none", 0.0
    
    # 전처리
    clean_text = re.sub(r'[^\w가-힣]', '', stt_text)
    
    # ==========================================
    # 1단계: Fast Path (정확 일치 / 부분 문자열 포함)
    # ==========================================
    for variation in ALL_VARIATIONS:
        clean_var = variation.replace(" ", "")
        if clean_var in clean_text or clean_text in clean_var:
            category = KEYWORDS_MAP.get(variation)
            return variation, category, "fast", 1.0
    
    # ==========================================
    # 2단계: Medium Path (발음 유사도)
    # ==========================================
    best_variation = None
    best_category = None
    best_score = 0.0
    
    for variation in ALL_VARIATIONS:
        score = korean_phonetic_similarity(clean_text, variation)
        if score > best_score:
            best_score = score
            best_variation = variation
            best_category = KEYWORDS_MAP.get(variation)
    
    if best_score >= threshold:
        return best_variation, best_category, "medium", best_score
    
    return None, None, "none", 0.0


# ======================================================
# [1단계] 포트 청소
# ======================================================
print("포트 7860 청소 중...")
try:
    cmd = "netstat -nlp | grep :7860"
    result = subprocess.check_output(cmd, shell=True).decode()
    if result:
        lines = result.strip().split('\n')
        for line in lines:
            parts = line.split()
            for part in parts:
                if '/' in part and part.split('/')[0].isdigit():
                    pid = int(part.split('/')[0])
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        time.sleep(2)
except Exception:
    pass

# ======================================================
# [2단계] SSL 보안 무시
# ======================================================
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['WGETRC'] = '' 
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ======================================================
# [3단계] 분석 로직
# ======================================================
def calculate_accuracy(intended, recognized):
    """정확도 분석"""
    if not intended:
        return "", 0.0

    def normalize(text):
        return re.sub(r'[\s\.,!?\-\'\"]+', '', text.lower())

    norm_intended = normalize(intended)
    norm_recognized = normalize(recognized)
    
    similarity = SequenceMatcher(None, norm_intended, norm_recognized).ratio()

    if intended == recognized:
        level = "exact"
    elif norm_intended == norm_recognized:
        level = "normalized"
    elif similarity >= 0.8:
        level = "similar"
    elif similarity >= 0.5:
        level = "different"
    else:
        level = "wrong"
        
    return level, similarity


def save_exact_log(timestamp, model, intended, stt_result, stt_time, stt_level, stt_score, 
                   is_dangerous, detect_path, keyword, confidence, is_emergency, correct):
    """CSV 로그 저장"""
    file_exists = os.path.exists(LOG_FILE)
    
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "timestamp", "model", "intended", "stt_result", "stt_time", 
                "stt_level", "stt_score", "is_dangerous", "detect_path", 
                "keyword", "confidence", "is_emergency", "correct"
            ])
            
        writer.writerow([
            timestamp,
            model,
            intended,
            stt_result,
            f"{stt_time:.2f}",
            stt_level,
            f"{stt_score:.2f}",
            is_dangerous,
            detect_path,
            keyword if keyword else "",
            f"{confidence:.2f}",
            is_emergency,
            correct
        ])


# ======================================================
# [4단계] 모델 로드
# ======================================================
print(f"{MODEL_NAME} 모델 로딩 중...")
model = WhisperModel(MODEL_NAME, device="cuda", compute_type="int8")
print("모델 로드 완료!")


def analyze_audio(audio_path, intended_text):
    if audio_path is None:
        return "오디오 없음", "N/A", "저장 안됨"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    
    try:
        # 1. STT 변환
        segments, _ = model.transcribe(
            audio_path, beam_size=1, language="ko", 
            condition_on_previous_text=False, vad_filter=True
        )
        stt_text = " ".join([s.text for s in segments]).strip()
        stt_time = time.time() - start_time
        
        # 2. 하이브리드 키워드 감지 (Fast + Medium Path)
        detected_kw, detected_category, detect_path, confidence = detect_keyword_hybrid(stt_text)
        is_dangerous = detected_kw is not None
        
        # 3. 정확도 분석
        stt_level, stt_score = calculate_accuracy(intended_text, stt_text)
        
        # 4. 정답 여부 판단
        is_emergency_situation = False
        if intended_text:
            for kw in ALL_VARIATIONS:
                if kw in intended_text:
                    is_emergency_situation = True
                    break
        
        is_correct = (is_dangerous == is_emergency_situation)
        
        # 5. 로그 저장
        save_exact_log(
            timestamp=timestamp,
            model=MODEL_NAME,
            intended=intended_text,
            stt_result=stt_text,
            stt_time=stt_time,
            stt_level=stt_level,
            stt_score=stt_score,
            is_dangerous=is_dangerous,
            detect_path=detect_path,
            keyword=detected_category,
            confidence=confidence,
            is_emergency=is_emergency_situation,
            correct=is_correct
        )
        
        # UI 출력
        status_icon = "위험" if is_dangerous else "안전"
        path_info = f"[{detect_path}]" if detect_path != "none" else ""
        accuracy_msg = f"({stt_level}, {stt_score:.0%})" if intended_text else ""
        
        return (
            stt_text, 
            f"{status_icon} {path_info} (키워드: {detected_category or '없음'}, 신뢰도: {confidence:.2f})", 
            f"저장됨 {accuracy_msg}"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"에러: {str(e)}", "ERROR", "저장 실패"


# ======================================================
# [5단계] 웹 UI
# ======================================================
with gr.Blocks(title="Sentinel Data Logger", analytics_enabled=False) as interface:
    gr.Markdown(f"## Sentinel AI 정밀 분석기 ({MODEL_NAME})")
    gr.Markdown("**Fast Path** (정확 일치) + **Medium Path** (발음 유사도) 적용")
    
    with gr.Row():
        with gr.Column():
            txt_intended = gr.Textbox(label="말할 내용 (정답지)", placeholder="예: 살려줘", lines=1)
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="마이크 입력")
            btn_submit = gr.Button("분석 및 로그 저장", variant="primary")
            
        with gr.Column():
            txt_stt = gr.Textbox(label="STT 인식 결과")
            txt_status = gr.Textbox(label="판정 결과")
            txt_log = gr.Textbox(label="로그 저장 상태")

    btn_submit.click(
        fn=analyze_audio,
        inputs=[mic_input, txt_intended],
        outputs=[txt_stt, txt_status, txt_log]
    )

print(f"서버 시작... https://192.168.2.104:7860")
interface.launch(
    server_name="0.0.0.0", 
    server_port=7860,
    ssl_certfile="cert.pem", 
    ssl_keyfile="key.pem",
    ssl_verify=False
)

