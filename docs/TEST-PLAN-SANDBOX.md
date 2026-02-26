# Open Speech — Sandbox Automated Test Plan

**Environment:** build-server (203.0.113.20) — CPU only, no GPU  
**Purpose:** Autonomous testing Will can run without Jeremy's Windows PC  
**Image:** `jwindsor1/open-speech:cpu` or local build  
**Port:** 8200 (avoids conflict with jw-pc on 8100)

---

## Setup

```bash
# Pull CPU image
docker pull jwindsor1/open-speech:cpu

# Run on port 8200, CPU mode
docker run -d \
  --name open-speech-test \
  -p 8200:8100 \
  -e STT_DEVICE=cpu \
  -e STT_COMPUTE_TYPE=int8 \
  -e STT_MODEL=Systran/faster-whisper-base \
  -e TTS_DEVICE=cpu \
  -e TTS_MODEL=pocket-tts \
  -e OS_HTTPS_ENABLED=false \
  jwindsor1/open-speech:cpu

# Or build from local repo
cd /home/claude/repos/open-speech
docker build -f Dockerfile.cpu -t open-speech:test-cpu .
docker run -d --name open-speech-test -p 8200:8100 \
  -e STT_DEVICE=cpu -e STT_COMPUTE_TYPE=int8 \
  -e STT_MODEL=Systran/faster-whisper-base \
  -e TTS_DEVICE=cpu -e OS_HTTPS_ENABLED=false \
  open-speech:test-cpu
```

---

## Phase 1: Container Health

```bash
BASE=http://localhost:8200

# S1: Health check
curl -s $BASE/health | python3 -m json.tool

# S2: Version correct
curl -s $BASE/health | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['version']=='0.5.1', d['version']; print('✅ version ok')"

# S3: Models list loads
curl -s $BASE/api/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"✅ {len(d['models'])} models listed\")"

# S4: Docker logs clean (no permission errors)
docker logs open-speech-test --tail 30 | grep -i "error\|permission denied" && echo "❌ errors found" || echo "✅ clean startup"
```

---

## Phase 2: STT — CPU Models

```bash
# Generate test audio
python3 -c "
import wave, struct, math
with wave.open('/tmp/test.wav', 'w') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
    # Silence placeholder — real test uses actual speech file
    f.writeframes(b'\x00' * 32000)
"

# T1: Install provider
curl -s -X POST $BASE/api/providers/install \
  -H 'Content-Type: application/json' \
  -d '{"provider":"faster-whisper"}' | python3 -m json.tool

# T2: Load base model (CPU-friendly, ~150MB)
curl -s -X POST $BASE/v1/audio/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"Systran/faster-whisper-base"}' | python3 -m json.tool

# T3: Transcribe test file
curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/test.wav" \
  -F "model=Systran/faster-whisper-base"

# T4: Transcribe from voice skill
~/.openclaw/skills/voice/scripts/stt-whisper /tmp/test.wav \
  --server http://localhost:8200
```

---

## Phase 3: TTS — CPU Models

### 3.1 Pocket TTS (primary CPU target)

```bash
# Install provider
curl -s -X POST $BASE/api/providers/install \
  -H 'Content-Type: application/json' \
  -d '{"provider":"pocket-tts"}' | python3 -m json.tool

# Poll install status
JOB_ID=$(curl -s -X POST $BASE/api/providers/install \
  -H 'Content-Type: application/json' \
  -d '{"provider":"pocket-tts"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# Check status
curl -s $BASE/api/providers/install/$JOB_ID | python3 -m json.tool

# Load model
curl -s -X POST $BASE/v1/audio/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"pocket-tts"}' | python3 -m json.tool

# Synthesize — time it
time curl -s -X POST $BASE/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"pocket-tts","input":"Open Speech running on CPU. Pocket TTS test.","voice":"alba"}' \
  -o /tmp/pocket-tts-test.wav

echo "File size: $(wc -c < /tmp/pocket-tts-test.wav) bytes"

# Get audio duration
python3 -c "
import wave
with wave.open('/tmp/pocket-tts-test.wav') as f:
    dur = f.getnframes() / f.getframerate()
    print(f'Audio duration: {dur:.1f}s')
"
```

### 3.2 Piper (lightweight, CPU)

```bash
# Install + load
curl -s -X POST $BASE/api/providers/install \
  -H 'Content-Type: application/json' \
  -d '{"provider":"piper"}' | python3 -m json.tool

curl -s -X POST $BASE/v1/audio/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"piper/en_US-lessac-medium"}' | python3 -m json.tool

# Synthesize
time curl -s -X POST $BASE/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"piper/en_US-lessac-medium","input":"Piper TTS test on CPU.","voice":"default"}' \
  -o /tmp/piper-test.wav

echo "Piper file: $(wc -c < /tmp/piper-test.wav) bytes"
```

---

## Phase 4: Latency Benchmarks

```bash
# Benchmark script — compare backends on same text
TEXT="The quick brown fox jumps over the lazy dog. This sentence is used for testing text to speech latency."

for BACKEND in pocket-tts piper; do
  echo "=== $BACKEND ==="
  START=$(date +%s%N)
  curl -s -X POST $BASE/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$BACKEND\",\"input\":\"$TEXT\"}" \
    -o /tmp/bench-$BACKEND.wav
  END=$(date +%s%N)
  MS=$(( (END - START) / 1000000 ))
  DUR=$(python3 -c "import wave; f=wave.open('/tmp/bench-$BACKEND.wav'); print(round(f.getnframes()/f.getframerate(),1))" 2>/dev/null || echo "?")
  echo "  Generation: ${MS}ms | Audio: ${DUR}s"
done
```

---

## Phase 5: API Compatibility

```bash
# OpenAI-compatible TTS
curl -s -X POST $BASE/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"OpenAI compat test","voice":"alloy"}' \
  -o /tmp/compat-test.wav && echo "✅ TTS compat ok" || echo "❌ TTS compat failed"

# OpenAI-compatible STT
curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/test.wav" \
  -F "model=whisper-1" | python3 -m json.tool

# Models list
curl -s $BASE/v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'✅ {len(d[\"data\"])} models in /v1/models')
"
```

---

## Phase 6: Cleanup

```bash
docker stop open-speech-test
docker rm open-speech-test
rm -f /tmp/pocket-tts-test.wav /tmp/piper-test.wav /tmp/bench-*.wav /tmp/test.wav
```

---

## Automated Run Script

```bash
#!/bin/bash
# run-sandbox-tests.sh — Will runs this autonomously
set -e
BASE=http://localhost:8200

echo "🧪 Starting Open Speech sandbox tests..."

# Start container
docker run -d --name open-speech-test -p 8200:8100 \
  -e STT_DEVICE=cpu -e STT_COMPUTE_TYPE=int8 \
  -e STT_MODEL=Systran/faster-whisper-base \
  -e TTS_DEVICE=cpu -e OS_HTTPS_ENABLED=false \
  jwindsor1/open-speech:cpu

# Wait for startup
sleep 15

# Health
curl -sf $BASE/health > /dev/null && echo "✅ Health ok" || { echo "❌ Health failed"; exit 1; }

# Version
VER=$(curl -s $BASE/health | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
echo "✅ Version: $VER"

# Models
COUNT=$(curl -s $BASE/api/models | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models']))")
echo "✅ $COUNT models listed"

echo "🎉 Smoke tests passed. Container running on port 8200."
echo "Run Phase 2-5 tests manually or extend this script."
```

---

## Results Log

| Date | Image | Phase | Result | Notes |
|------|-------|-------|--------|-------|
| — | — | — | — | First run TBD |

---

## CPU Model Candidates

| Model | Type | Size | Speed (CPU) | Quality |
|-------|------|------|-------------|---------|
| Systran/faster-whisper-tiny | STT | 75MB | Very fast | Acceptable |
| Systran/faster-whisper-base | STT | 150MB | Fast | Good |
| Systran/faster-whisper-small | STT | 500MB | Moderate | Better |
| pocket-tts | TTS | 220MB | 3.3x RT | Good |
| piper/en_US-lessac-medium | TTS | 35MB | Very fast | Decent |
