#!/usr/bin/env bash
BASE_URL="http://localhost:8000"
API_KEY=""  # ADMIN_WS_API_KEY 사용 시 값 입력, 없으면 빈 값
AUTH_HEADER=()
if [ -n "$API_KEY" ]; then
  AUTH_HEADER=(-H "X-API-Key: $API_KEY")
fi

stream_id="dummy_cam"
module_name="DummyModule"
ts=$(date +%s)

INTERVAL="${1:-3}"  # seconds; override with first arg, e.g., ./curl_test.sh 1

echo "Streaming test data to $BASE_URL/admin/debug/broadcast every ${INTERVAL}s (Ctrl+C to stop)"

seq=0
while true; do
  seq=$((seq + 1))
  fps=$(python - <<'PY'
import random; import sys
print(round(random.uniform(10,20),1))
PY
)
  err=$((seq % 5 == 0 ? 1 : 0))
  timeouts=$((seq % 7 == 0 ? 1 : 0))
  ts=$(date +%s)

  # stream_update
  curl -s -X POST "$BASE_URL/admin/debug/broadcast" \
  -H "Content-Type: application/json" "${AUTH_HEADER[@]}" \
    -d "{\"kind\":\"stream_update\",\"payload\":{\"stream_id\":\"$stream_id\",\"status\":\"RUNNING\",\"fps\":$fps,\"error_count\":$err,\"last_frame_ts\":$ts}}" >/dev/null

  # module_stats
  curl -s -X POST "$BASE_URL/admin/debug/broadcast" \
  -H "Content-Type: application/json" "${AUTH_HEADER[@]}" \
    -d "{\"kind\":\"module_stats\",\"payload\":{\"modules\":{\"$module_name\":{\"success_count\":$((10+seq)) ,\"error_count\":$err,\"timeout_count\":$timeouts}},\"ts\":$ts}}" >/dev/null

  # event
  curl -s -X POST "$BASE_URL/admin/debug/broadcast" \
  -H "Content-Type: application/json" "${AUTH_HEADER[@]}" \
    -d "{\"kind\":\"event\",\"payload\":{\"type\":\"event\",\"stream_id\":\"$stream_id\",\"module\":\"$module_name\",\"ts\":$ts,\"count\":1,\"types\":[\"CUSTOM\"]}}" >/dev/null

  echo "sent seq=$seq fps=$fps err=$err timeout=$timeouts ts=$ts"
  sleep "$INTERVAL"
done