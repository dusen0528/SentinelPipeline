# 스트림 제어 API 명세서

**Base URL**: `http://13.209.141.252:8000/api/streams`

---

## 1. 스트림 목록 조회

**GET** `/api/streams`

모든 스트림의 상태 목록을 조회합니다.

### 응답 예시
```json
{
  "success": true,
  "data": [
    {
      "stream_id": "d4633bc5",
      "status": "running",
      "fps": 15.0,
      "last_frame_ts": 1234567890.123,
      "error_count": 0
    }
  ]
}
```

---

## 2. 스트림 상세 조회

**GET** `/api/streams/{stream_id}`

특정 스트림의 상세 정보를 조회합니다.

### 경로 파라미터
- `stream_id` (string): 스트림 ID

### 응답 예시
```json
{
  "success": true,
  "data": {
    "stream_id": "d4633bc5",
    "status": "running",
    "config": { ... },
    "stats": { ... }
  }
}
```

---

## 3. 스트림 시작

**POST** `/api/streams/{stream_id}/start`

스트림을 시작합니다. RUNNING 상태가 될 때까지 대기합니다.

### 경로 파라미터
- `stream_id` (string): 스트림 ID

### 요청 Body
```json
{
  "rtsp_url": "rtsp://15.165.156.169:5554/01057432025-01057432025",
  "max_fps": 15,
  "downscale": 1.0,
  "output_url": "rtsp://15.165.156.169:5554/01057432025-blur"
}
```

### 응답 예시
```json
{
  "success": true,
  "data": {
    "stream_id": "d4633bc5",
    "status": "running",
    "config": { ... }
  }
}
```

---

## 4. 스트림 중지

**POST** `/api/streams/{stream_id}/stop`

스트림을 중지합니다.

### 경로 파라미터
- `stream_id` (string): 스트림 ID

### 응답 예시
```json
{
  "success": true,
  "data": {
    "stream_id": "d4633bc5",
    "status": "STOPPED"
  }
}
```

---

## 5. 스트림 재시작

**POST** `/api/streams/{stream_id}/restart`

스트림을 재시작합니다. RUNNING 상태가 될 때까지 대기합니다.

### 경로 파라미터
- `stream_id` (string): 스트림 ID

### 응답 예시
```json
{
  "success": true,
  "data": {
    "stream_id": "d4633bc5",
    "status": "running"
  }
}
```

---

## 6. 스트림 연결 끊기 (완전 정리)

**POST** `/api/streams/{stream_id}/disconnect`

스트림을 중지하고 모든 리소스(FFmpeg, 디코더 등)를 정리한 후 삭제합니다.

### 경로 파라미터
- `stream_id` (string): 스트림 ID

### 동작 방식
1. 스트림 중지 요청 (`_stop_event.set()`)
2. 스레드 종료 대기 (최대 5초)
3. 리소스 정리 (디코더, 퍼블리셔 해제)
4. 스트림 삭제 (메모리에서 제거)

### 주의사항
- **정상 종료**: 스트림 루프가 `should_stop()` 체크를 통해 정상 종료
- **블로킹 상황**: 네트워크 I/O나 외부 프로세스가 응답하지 않으면 5초 이상 걸릴 수 있음
- **예외 처리**: 이미 삭제된 스트림이어도 성공 응답 반환

### 응답 예시
```json
{
  "success": true,
  "data": {
    "stream_id": "d4633bc5",
    "status": "DISCONNECTED",
    "message": "연결이 완전히 종료되었습니다."
  }
}
```

---

## 7. 입력 URL로 출력 URL 조회

**GET** `/api/streams/by-input?input_url={rtsp_url}`

입력 RTSP URL로 스트림을 찾거나 생성하고, 출력 URL을 반환합니다.

### 쿼리 파라미터
- `input_url` (string, required): 입력 RTSP URL

### 예시
```
GET http://13.209.141.252:8000/api/streams/by-input?input_url=rtsp://15.165.156.169:5554/01057432025-01057432025
```

### 응답 예시 (기존 스트림이 있는 경우)
```json
{
  "success": true,
  "data": {
    "stream_id": "d4633bc5",
    "input_url": "rtsp://15.165.156.169:5554/01057432025-01057432025",
    "output_url": "rtsp://15.165.156.169:5554/01057432025-blur",
    "name": "d4633bc5",
    "status": "running",
    "message": "기존 스트림을 찾았습니다."
  }
}
```

### 응답 예시 (새로 생성된 경우)
```json
{
  "success": true,
  "data": {
    "stream_id": "auto-a1b2c3d4",
    "input_url": "rtsp://15.165.156.169:5554/01057432025-01057432025",
    "output_url": "rtsp://15.165.156.169:5554/01057432025-blur",
    "name": "auto-a1b2c3d4",
    "status": "running",
    "message": "새로운 스트림이 생성되었습니다."
  }
}
```

### 동작 방식
- **기존 스트림이 있으면**: 출력 URL만 반환 (상태와 무관)
- **스트림이 없으면**: 새로 생성하고 출력 URL 반환
- **새로 생성하는 경우**: RUNNING 상태가 될 때까지 최대 10초 대기

---

## 8. 스트림 재등록 (강제 재시작)

**POST** `/api/streams/register`

입력 RTSP URL로 스트림을 강제로 재등록합니다. 기존 스트림이 있으면 먼저 연결을 끊고 새로 등록합니다.


### 요청 Body
```json
{
  "rtsp_url": "rtsp://15.165.156.169:5554/01057432025-01057432025"
}
```
### 동작 방식
1. 입력 RTSP URL로 기존 스트림 검색
2. **기존 스트림이 있으면**: 
   - 기존 스트림 연결 끊기 (`disconnect`)
   - 새로 등록 → RUNNING 상태 대기 (최대 10초) → 출력 RTSP URL 반환
3. **스트림이 없으면**: 
   - 새로 등록 → RUNNING 상태 대기 (최대 10초) → 출력 RTSP URL 반환

### 응답 예시
```json
{
  "success": true,
  "data": {
    "output_url": "rtsp://15.165.156.169:5554/01057432025-blur"
  }
}
```

### 주의사항
- **기존 스트림 강제 종료**: 기존 스트림이 있으면 무조건 연결을 끊고 새로 시작
- **RUNNING 상태 확인**: 최대 10초 동안 RUNNING 상태가 될 때까지 대기
- **타임아웃**: 10초 내에 RUNNING 상태가 되지 않으면 503 에러 반환

### 사용 예시
```bash
# 첫 호출: 스트림 생성
POST http://13.209.141.252:8000/api/streams/register
Body: {"rtsp_url": "rtsp://15.165.156.169:5554/01057432025-01057432025"}
→ {"success": true, "data": {"output_url": "rtsp://15.165.156.169:5554/01057432025-blur"}}

# 두 번째 호출: 기존 스트림 끊고 재등록
POST http://13.209.141.252:8000/api/streams/register
Body: {"rtsp_url": "rtsp://15.165.156.169:5554/01057432025-01057432025"}
→ {"success": true, "data": {"output_url": "rtsp://15.165.156.169:5554/01057432025-blur"}}
```

### 특징
- ✅ 강제 재시작 (기존 스트림 무조건 끊기)
- ✅ RUNNING 상태 확인 후 반환
- ✅ 항상 새로운 스트림 인스턴스 생성

---

## 공통 사항

### 에러 응답 형식
```json
{
  "code": "STREAM_NOT_FOUND",
  "message": "스트림을 찾을 수 없습니다: {stream_id}",
  "details": {
    "stream_id": "d4633bc5"
  }
}
```

### 상태 코드
- `200 OK`: 성공
- `400 Bad Request`: 잘못된 요청 (RTSP URL 형식 오류 등)
- `404 Not Found`: 스트림을 찾을 수 없음
- `500 Internal Server Error`: 서버 오류
- `502 Bad Gateway`: 스트림 연결 실패
- `503 Service Unavailable`: 타임아웃 (10초 내 RUNNING 상태 미달성)

### RTSP URL 형식
- `rtsp://[hostname]:[port]/[path]`
- 예: `rtsp://15.165.156.169:5554/01057432025-01057432025`

### 출력 URL 생성 규칙
입력 URL의 마지막 경로 세그먼트에 `-blur`를 추가합니다.
- 입력: `rtsp://15.165.156.169:5554/01057432025-01057432025`
- 출력: `rtsp://15.165.156.169:5554/01057432025-blur`

---

## 주요 사용 시나리오

### 시나리오 1: 출력 RTSP URL 조회 (기존 유지, 권장)

**목적**: 입력 URL로 출력 URL을 얻고 싶을 때. 기존 스트림이 있으면 그대로 유지.

**API**: `GET /api/streams/by-input?input_url={rtsp_url}`

**동작**:
1. 입력 RTSP URL로 기존 스트림 검색
2. **기존 스트림이 있으면**: 출력 URL만 즉시 반환 (상태와 무관, 삭제/재시작 안 함)
3. **스트림이 없으면**: 새로 생성 → RUNNING 상태 대기 (최대 10초) → 출력 URL 반환

**사용 예시**:
```bash
# 첫 호출: 스트림 생성 후 URL 반환
GET /api/streams/by-input?input_url=rtsp://15.165.156.169:5554/01057432025-01057432025
→ {"output_url": "rtsp://15.165.156.169:5554/01057432025-blur", "status": "running"}

# 두 번째 호출: 기존 스트림 유지, URL만 반환 (빠름)
GET /api/streams/by-input?input_url=rtsp://15.165.156.169:5554/01057432025-01057432025
→ {"output_url": "rtsp://15.165.156.169:5554/01057432025-blur", "status": "running"}
```

**특징**:
- ✅ Idempotent (여러 번 호출해도 안전)
- ✅ 기존 스트림 보존 (불필요한 재시작 방지)
- ✅ 빠른 응답 (기존 스트림이 있으면 즉시 반환)

---

### 시나리오 2: 스트림 연결 끊기 (완전 정리)

**목적**: 스트림을 완전히 종료하고 모든 리소스를 정리.

**API**: `POST /api/streams/{stream_id}/disconnect`

**동작**:
1. 스트림 중지 요청 (정상 종료 시도)
2. 스레드 종료 대기 (최대 5초)
3. 리소스 정리 (디코더, 퍼블리셔 해제)
4. 스트림 삭제 (메모리에서 제거)

**사용 예시**:
```bash
# 스트림 완전 종료
POST /api/streams/d4633bc5/disconnect
→ {"status": "DISCONNECTED", "message": "연결이 완전히 종료되었습니다."}
```

**특징**:
- ✅ 모든 리소스 정리
- ✅ 예외 처리 포함 (이미 삭제된 경우에도 성공)

---

### 시나리오 3: 스트림 시작 (명시적 제어)

**목적**: stream_id를 지정하고 설정을 커스터마이징하여 스트림 시작.

**API**: `POST /api/streams/{stream_id}/start`

**동작**:
1. Body에 rtsp_url, max_fps, downscale, output_url 설정
2. 스트림 시작 → RUNNING 상태 대기 (최대 10초) → 상태 반환

**사용 예시**:
```bash
POST /api/streams/my-stream-1/start
Body: {
  "rtsp_url": "rtsp://15.165.156.169:5554/01057432025-01057432025",
  "max_fps": 15,
  "downscale": 1.0,
  "output_url": "rtsp://15.165.156.169:5554/01057432025-blur"
}
→ {"status": "running", "config": {...}}
```

**특징**:
- ✅ stream_id 직접 지정
- ✅ 상세 설정 가능 (FPS, downscale, output_url)
- ✅ RUNNING 상태 확인 후 반환

---

### 시나리오 4: 스트림 재시작

**목적**: 멈춰있거나 에러인 스트림을 재시작.

**API**: `POST /api/streams/{stream_id}/restart`

**동작**:
1. 기존 스트림 중지
2. 같은 설정으로 재시작
3. RUNNING 상태 대기 (최대 10초) → 상태 반환

**사용 예시**:
```bash
POST /api/streams/d4633bc5/restart
→ {"status": "running"}
```

**특징**:
- ✅ 편의성 API (stop + start 조합)
- ✅ RUNNING 상태 확인 후 반환

---

### 시나리오 5: 스트림 중지 (일시 정지)

**목적**: 스트림을 일시 중지 (삭제하지 않음).

**API**: `POST /api/streams/{stream_id}/stop`

**동작**:
1. 스트림 중지 요청
2. 스레드 종료 대기 (최대 5초)
3. 상태를 STOPPED로 변경 (메모리에는 유지)

**사용 예시**:
```bash
POST /api/streams/d4633bc5/stop
→ {"status": "STOPPED"}
```

**특징**:
- ✅ 일시 중지 (나중에 restart 가능)
- ✅ 삭제하지 않음 (메모리에 유지)

---

## 시나리오별 API 선택 가이드

| 상황 | 권장 API | 이유 |
|------|---------|------|
| **출력 URL만 필요** | `GET /by-input` | 기존 유지, 빠른 응답 |
| **스트림 완전 종료** | `POST /{stream_id}/disconnect` | 리소스 정리 |
| **커스텀 설정으로 시작** | `POST /{stream_id}/start` | 상세 설정 가능 |
| **스트림 재시작** | `POST /{stream_id}/restart` | 편의성 |
| **일시 중지** | `POST /{stream_id}/stop` | 나중에 재시작 가능 |

---

## 중요 사항

### ✅ 모든 새로 생성하는 API는 RUNNING 상태 확인 후 반환
- `GET /by-input` (스트림 없을 때)
- `POST /{stream_id}/start`
- `POST /{stream_id}/restart`

**동작**: 최대 10초 동안 RUNNING 상태가 될 때까지 대기 후 응답

### ✅ 기존 스트림 유지 정책
- `GET /by-input`: 기존 스트림이 있으면 **삭제/재시작 없이** URL만 반환
- 불필요한 재시작 방지로 안정성 향상

### ✅ 출력 URL 생성 규칙
- 입력 URL의 마지막 경로에 `-blur` 추가
- 예: `rtsp://.../01057432025-01057432025` → `rtsp://.../01057432025-blur`

