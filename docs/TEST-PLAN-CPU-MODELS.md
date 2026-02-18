# Open Speech — CPU Model Test Plan

**Author:** Will 🗿  
**Reviewed by:** Forge (peer review — F → fixes applied)  
**Environment:** debian-sandbox (192.0.2.26) — 8 cores, 32GB RAM, no GPU  
**Purpose:** Validate every CPU-capable model end-to-end on sandbox  
**Validation method:** Generated OGG audio sent to Jeremy via Discord voice messages

---

## CPU Model Matrix

| Model | Type | Size | CPU? | Streaming | Priority |
|-------|------|------|------|-----------|----------|
| Systran/faster-whisper-tiny | STT | 75MB | ✅ | — | High |
| Systran/faster-whisper-tiny.en | STT | 75MB | ✅ | — | High |
| Systran/faster-whisper-base | STT | 150MB | ✅ | — | High |
| Systran/faster-whisper-base.en | STT | 150MB | ✅ | — | High |
| Systran/faster-whisper-small | STT | 500MB | ✅ | — | Medium |
| Systran/faster-whisper-small.en | STT | 500MB | ✅ | — | Medium |
| moonshine/tiny | STT | 35MB | ✅ | — | Medium |
| moonshine/base | STT | 70MB | ✅ | — | Medium |
| pocket-tts | TTS | 220MB | ✅ CPU-only | ✅ | **Critical** |
| piper/en_US-lessac-medium | TTS | 35MB | ✅ CPU-only | ❌ | High |
| piper/en_US-lessac-high | TTS | 75MB | ✅ CPU-only | ❌ | High |
| piper/en_US-amy-medium | TTS | 35MB | ✅ CPU-only | ❌ | Medium |
| piper/en_US-ryan-medium | TTS | 35MB | ✅ CPU-only | ❌ | Medium |
| piper/en_GB-alan-medium | TTS | 35MB | ✅ CPU-only | ❌ | Medium |
| piper/en_GB-cori-medium | TTS | 35MB | ✅ CPU-only | ❌ | Medium |
| kokoro | TTS | 330MB | ✅ (slow) | ❌ | Low |

**Not CPU-viable on sandbox:** qwen3, fish-speech, f5-tts (GPU strongly required)

---

## Container Setup

```bash
# Build from local source (latest main)
cd /home/claude/repos/open-speech
docker build -f Dockerfile.cpu -t open-speech:cpu-test .

# Run CPU-mode on port 8200
# NOTE: OS_SSL_ENABLED=false (NOT OS_HTTPS_ENABLED — that var doesn't exist)
docker run -d \
  --name open-speech-test \
  -p 8200:8100 \
  -p 10401:10400 \
  -e STT_DEVICE=cpu \
  -e STT_COMPUTE_TYPE=int8 \
  -e OS_SSL_ENABLED=false \
  open-speech:cpu-test

# Wait for startup
sleep 20
curl -s http://localhost:8200/health | python3 -m json.tool
```

---

## Helper Functions

```bash
BASE=http://localhost:8200

# Install provider and wait for completion (fixes async race condition)
install_provider() {
  local PROVIDER=$1
  local MAX_WAIT=${2:-120}  # seconds
  
  JOB_ID=$(curl -s -X POST $BASE/api/providers/install \
    -H 'Content-Type: application/json' \
    -d "{\"provider\":\"$PROVIDER\"}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))")
  
  if [ -z "$JOB_ID" ]; then
    echo "❌ No job_id returned for $PROVIDER"
    return 1
  fi
  
  echo "Installing $PROVIDER (job: $JOB_ID)..."
  for i in $(seq 1 $MAX_WAIT); do
    STATUS=$(curl -s $BASE/api/providers/install/$JOB_ID | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))")
    if [ "$STATUS" = "done" ]; then
      echo "✅ $PROVIDER installed"
      return 0
    elif [ "$STATUS" = "failed" ]; then
      curl -s $BASE/api/providers/install/$JOB_ID | python3 -c "import sys,json; d=json.load(sys.stdin); print('❌ Install failed:', d.get('error',''))"
      return 1
    fi
    sleep 1
  done
  echo "❌ Timeout waiting for $PROVIDER install"
  return 1
}

# Load model via correct endpoint (STT = /api/models/{id}/load, TTS = /v1/audio/models/load)
load_stt_model() {
  local MODEL_ID=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$1', safe=''))")
  curl -s -X POST "$BASE/api/models/$MODEL_ID/load" \
    -H 'Content-Type: application/json' | python3 -m json.tool
}

load_tts_model() {
  curl -s -X POST $BASE/v1/audio/models/load \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$1\"}" | python3 -m json.tool
}

# Synthesize and save as OGG (correct format for Discord + voice skill parity)
synth() {
  local MODEL=$1 INPUT=$2 VOICE=$3 OUTFILE=$4
  curl -s -X POST $BASE/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"input\":\"$INPUT\",\"voice\":\"$VOICE\",\"response_format\":\"wav\"}" \
    -o "$OUTFILE"
  local SIZE=$(wc -c < "$OUTFILE")
  if [ "$SIZE" -lt 1000 ]; then
    echo "❌ $OUTFILE too small ($SIZE bytes) — likely error"
    cat "$OUTFILE"
    return 1
  fi
  echo "✅ $OUTFILE ($SIZE bytes)"
}

# Get WAV duration in seconds
wav_dur() {
  python3 -c "import wave; f=wave.open('$1'); print(round(f.getnframes()/f.getframerate(),1))"
}

# Test sentences
SHORT="Hello. Open Speech is running."
MEDIUM="The quick brown fox jumps over the lazy dog. Open Speech is running on CPU with no GPU required."
LONG="Open Speech is a unified self-hosted speech platform supporting multiple text-to-speech and speech-to-text backends. It runs locally on your hardware with no cloud dependencies, no per-character billing, and full OpenAI API compatibility."
```

---

## Test Inputs (real speech for STT)

```bash
# Download a real speech sample for STT testing (LibriVox public domain)
curl -sL "https://www.librivox.org/rss/6" | grep -o 'http[^"]*mp3' | head -1 | xargs curl -sL -o /tmp/test-speech.mp3 || true

# Fallback: use a pre-generated voice clip from TTS-1 (once pocket-tts is working)
# Will generate /tmp/test-speech.wav from pocket-tts during test run
```

---

## Phase 1: Smoke

```bash
# S1: Health
curl -sf $BASE/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'✅ v{d[\"version\"]} models_loaded={d[\"models_loaded\"]}')"

# S2: Models list
curl -s $BASE/api/models | python3 -c "
import sys,json
d=json.load(sys.stdin)
tts=[m for m in d['models'] if m.get('type')=='tts']
stt=[m for m in d['models'] if m.get('type')=='stt']
print(f'✅ {len(stt)} STT + {len(tts)} TTS models listed')
"

# S3: Clean logs
docker logs open-speech-test --tail 20 | grep -iE "permission denied|error" && echo "❌ errors in logs" || echo "✅ clean logs"
```

---

## Phase 2: STT Tests

```bash
# Install faster-whisper provider once
install_provider "faster-whisper"

# STT-1: tiny — speed baseline
load_stt_model "Systran/faster-whisper-tiny"
time curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/test-speech.wav;type=audio/wav" \
  -F "model=Systran/faster-whisper-tiny" | python3 -m json.tool
# PASS: Returns {"text":"..."} within 5s

# STT-2: tiny.en — English-only variant
load_stt_model "Systran/faster-whisper-tiny.en"
time curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/test-speech.wav;type=audio/wav" \
  -F "model=Systran/faster-whisper-tiny.en" | python3 -m json.tool
# PASS: Same or faster than STT-1

# STT-3: base — quality step up
load_stt_model "Systran/faster-whisper-base"
time curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/test-speech.wav;type=audio/wav" \
  -F "model=Systran/faster-whisper-base" | python3 -m json.tool
# PASS: Returns within 8s

# STT-4: base.en
load_stt_model "Systran/faster-whisper-base.en"
time curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/test-speech.wav;type=audio/wav" \
  -F "model=Systran/faster-whisper-base.en" | python3 -m json.tool

# STT-5: small (larger, more accurate)
load_stt_model "Systran/faster-whisper-small"
time curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/test-speech.wav;type=audio/wav" \
  -F "model=Systran/faster-whisper-small" | python3 -m json.tool
# PASS: Returns within 15s

# STT-6: moonshine/tiny (alternative provider)
install_provider "moonshine" 180
load_stt_model "moonshine/tiny"
time curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/test-speech.wav;type=audio/wav" \
  -F "model=moonshine/tiny" | python3 -m json.tool
# PASS: Loads and returns transcript

# STT-7: error path — invalid model ID
curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/test-speech.wav;type=audio/wav" \
  -F "model=fake/model-that-doesnt-exist" | python3 -m json.tool
# PASS: Returns {"error":"..."} not 500

# STT-8: error path — empty file
echo "" > /tmp/empty.wav
curl -s -X POST $BASE/v1/audio/transcriptions \
  -F "file=@/tmp/empty.wav;type=audio/wav" \
  -F "model=Systran/faster-whisper-tiny" | python3 -m json.tool
# PASS: Returns error, not crash
```

---

## Phase 3: TTS Tests

### Pocket TTS (CPU-native, streaming)

```bash
install_provider "pocket-tts" 180
load_tts_model "pocket-tts"

# TTS-1: All 8 voices — one each, timed
for VOICE in alba marius javert jean fantine cosette eponine azelma; do
  START=$(date +%s%3N)
  synth "pocket-tts" "$SHORT" "$VOICE" "/tmp/pocket-$VOICE.wav"
  END=$(date +%s%3N)
  DUR=$(wav_dur /tmp/pocket-$VOICE.wav 2>/dev/null || echo "?")
  echo "$VOICE: $((END-START))ms gen | ${DUR}s audio"
done

# TTS-2: Latency benchmark (3 lengths, 3 runs each for averaging)
for LEN in short medium long; do
  TEXT=$([ "$LEN" = "short" ] && echo "$SHORT" || [ "$LEN" = "medium" ] && echo "$MEDIUM" || echo "$LONG")
  TOTAL=0
  for RUN in 1 2 3; do
    START=$(date +%s%3N)
    synth "pocket-tts" "$TEXT" "alba" "/tmp/pocket-bench-$LEN-$RUN.wav" > /dev/null
    END=$(date +%s%3N)
    TOTAL=$((TOTAL + END - START))
  done
  AVG=$((TOTAL / 3))
  DUR=$(wav_dur /tmp/pocket-bench-$LEN-1.wav 2>/dev/null || echo "?")
  FACTOR=$(python3 -c "print(round(${DUR:-0} / (${AVG}/1000+0.001), 2))" 2>/dev/null || echo "?")
  echo "$LEN: avg=${AVG}ms | audio=${DUR}s | RT_factor=${FACTOR}x"
done
# PASS: RT factor >= 2.0x for medium text

# TTS-3: Streaming endpoint
curl -s -X POST $BASE/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"pocket-tts","input":"Streaming test for pocket TTS.","voice":"alba","response_format":"wav","stream":true}' \
  -o /tmp/pocket-stream.wav
echo "Stream output: $(wc -c < /tmp/pocket-stream.wav) bytes"
# PASS: Returns audio (streamed or buffered)

# TTS-4: Error paths
# Empty input
curl -s -X POST $BASE/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"pocket-tts","input":"","voice":"alba"}' | python3 -m json.tool
# PASS: Returns {"error":"..."} not 500

# Invalid voice — should fallback to default (alba)
synth "pocket-tts" "$SHORT" "nonexistent_voice" "/tmp/pocket-fallback.wav"
# PASS: Returns audio (fallback voice)
```

### Piper (CPU-native, multiple voices)

```bash
install_provider "piper" 180

# Load and test each model separately (unload between to avoid piper first-loaded bias)
for MODEL in "piper/en_US-lessac-medium" "piper/en_US-lessac-high" "piper/en_US-amy-medium" "piper/en_US-ryan-medium" "piper/en_GB-alan-medium" "piper/en_GB-cori-medium"; do
  SHORT_NAME=$(echo $MODEL | tr '/' '-')
  
  # Load this model
  load_tts_model "$MODEL"
  
  START=$(date +%s%3N)
  synth "$MODEL" "$MEDIUM" "default" "/tmp/piper-$SHORT_NAME.wav"
  END=$(date +%s%3N)
  DUR=$(wav_dur /tmp/piper-$SHORT_NAME.wav 2>/dev/null || echo "?")
  echo "$SHORT_NAME: $((END-START))ms | ${DUR}s"
  
  # Unload before next (avoids first-model bias)
  curl -s -X DELETE "$BASE/api/models/$(python3 -c "import urllib.parse; print(urllib.parse.quote('$MODEL', safe=''))")/artifacts" > /dev/null || true
done
# PASS: All 6 generate without error; US/GB accents audibly distinct
```

### Kokoro CPU (slow path validation)

```bash
install_provider "kokoro" 300
load_tts_model "kokoro"

START=$(date +%s%3N)
synth "kokoro" "$SHORT" "alloy" "/tmp/kokoro-cpu.wav"
END=$(date +%s%3N)
echo "Kokoro CPU: $((END-START))ms"
# PASS: Completes without error (time is informational — expected to be slow on CPU)
```

---

## Phase 4: Error Path Tests

```bash
# E1: Load model that hasn't had provider installed
curl -s -X POST $BASE/v1/audio/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"fish-speech-1.5"}' | python3 -m json.tool
# PASS: Returns {"error":"..."} with provider_missing code

# E2: Synthesize without loading model
curl -s -X POST $BASE/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts/1.7B-Base","input":"test","voice":"default"}' | python3 -m json.tool
# PASS: Returns {"error":"..."} not 500

# E3: Provider install job 404
curl -s $BASE/api/providers/install/fake-job-id-00000 | python3 -m json.tool
# PASS: Returns 404 with install_job_not_found
```

---

## Phase 5: Comparative Benchmark

```bash
echo "=== CPU TTS BENCHMARK ==="
echo "Input: $MEDIUM"
echo ""

declare -A TESTS=(
  ["pocket-tts/alba"]="pocket-tts|alba"
  ["piper/lessac-med"]="piper/en_US-lessac-medium|default"
  ["piper/amy-med"]="piper/en_US-amy-medium|default"
  ["piper/alan-gb"]="piper/en_GB-alan-medium|default"
  ["kokoro/cpu"]="kokoro|alloy"
)

printf "%-22s %10s %10s %12s\n" "Model" "Gen(ms)" "Audio(s)" "RT_factor"
for NAME in "${!TESTS[@]}"; do
  IFS='|' read -r MODEL VOICE <<< "${TESTS[$NAME]}"
  START=$(date +%s%3N)
  curl -s -X POST $BASE/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"input\":\"$MEDIUM\",\"voice\":\"$VOICE\",\"response_format\":\"wav\"}" \
    -o /tmp/bench-final-$NAME.wav 2>/dev/null
  END=$(date +%s%3N)
  GEN=$((END-START))
  DUR=$(wav_dur /tmp/bench-final-$NAME.wav 2>/dev/null || echo "0")
  FACTOR=$(python3 -c "print(round(${DUR} / (${GEN}/1000+0.001), 2))" 2>/dev/null || echo "?")
  printf "%-22s %10s %10s %12s\n" "$NAME" "${GEN}ms" "${DUR}s" "${FACTOR}x"
done
```

---

## Phase 6: Discord Voice Validation

After all local tests pass, Will sends each generated audio to Jeremy for ear test.
Files are sent as WAV attachments — Discord plays WAV inline.

```bash
# Convert to OGG for voice-skill parity (optional but preferred)
for WAV in /tmp/pocket-alba.wav /tmp/pocket-fantine.wav /tmp/piper-piper-en_US-lessac-medium.wav /tmp/piper-piper-en_GB-alan-medium.wav /tmp/kokoro-cpu.wav; do
  OGG="${WAV%.wav}.ogg"
  ffmpeg -i "$WAV" -c:a libopus "$OGG" -y -loglevel quiet 2>/dev/null && echo "✅ $OGG" || echo "⚠️ ffmpeg not available, sending WAV"
done
# Then use message tool to send each file
```

---

## Results Table

| Test | Model | Gen(ms) | Audio(s) | RT | Quality (JW) | Pass? |
|------|-------|---------|----------|-----|--------------|-------|
| STT-1 | whisper-tiny | — | — | — | — | — |
| STT-2 | whisper-tiny.en | — | — | — | — | — |
| STT-3 | whisper-base | — | — | — | — | — |
| STT-4 | whisper-base.en | — | — | — | — | — |
| STT-5 | whisper-small | — | — | — | — | — |
| STT-6 | moonshine/tiny | — | — | — | — | — |
| TTS-1 | pocket-tts (8 voices) | — | — | — | — | — |
| TTS-2 | pocket-tts latency | — | — | — | — | — |
| TTS-3 | pocket-tts stream | — | — | — | — | — |
| TTS-4 | piper (6 voices) | — | — | — | — | — |
| TTS-5 | kokoro/cpu | — | — | — | — | — |
| E1-E3 | Error paths | — | — | — | — | — |

---

## Pass Criteria

| Test | Target |
|------|--------|
| STT tiny | < 5s for 5s audio |
| STT base | < 10s for 5s audio |
| Pocket TTS short | < 3s |
| Pocket TTS RT factor | ≥ 2.0x (avg of 3 runs) |
| Piper any voice | < 10s |
| Kokoro CPU | Completes without error |
| All error paths | Return `{"error":"..."}` not 500 |
| All voice Discord samples | JW: audibly intelligible |

---

*Forge peer review applied. F → fixes applied for all 4 Critical and 5 High issues.*
