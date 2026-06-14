# Development and Docker Build Workflow

This workflow keeps normal development off the CUDA build path. Use the local machine for edits and fast tests, use CPU Docker only for smoke checks, and use a Windows Docker host when the CUDA image is ready to publish.

## 1. Local thin development

Use the repo directly. Do not build the CUDA image for routine edits.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

For API-only work, keep model preloading disabled so startup does not download weights:

```bash
OS_MAX_LOADED_MODELS=0 \
STT_PRELOAD_MODELS= \
TTS_PRELOAD_MODELS= \
uvicorn src.main:app --host 127.0.0.1 --port 8100
```

## 2. CPU Docker smoke testing

Use CPU Docker when you need to verify packaging, entrypoint behavior, Compose wiring, or health checks. This is still not tiny, but it avoids the CUDA layers.

```bash
docker compose -f docker-compose.cpu.yml build
docker compose -f docker-compose.cpu.yml up -d
curl -fsS http://localhost:8100/health
docker compose -f docker-compose.cpu.yml down
```

Keep caches mounted. The compose files already persist:

- Hugging Face models: `/home/openspeech/.cache/huggingface`
- Silero VAD: `/home/openspeech/.cache/silero-vad`
- TLS certs: `/var/lib/open-speech/certs`
- TTS cache: `/var/lib/open-speech/cache`

Do not change those to anonymous container paths unless you want rebuilds and restarts to redownload models.

## 3. Windows CUDA build/push box

When code is ready, build the GPU image on the Windows machine with Docker Desktop and enough free disk. Run these from a checkout of this repo.

```powershell
$Tag = "jwindsor1/open-speech:cuda-$(git rev-parse --short HEAD)"
docker build -f Dockerfile -t $Tag -t jwindsor1/open-speech:latest .
docker push $Tag
docker push jwindsor1/open-speech:latest
```

Optional provider/model bake arguments:

```powershell
docker build -f Dockerfile `
  --build-arg BAKED_PROVIDERS="kokoro,piper,pocket-tts" `
  --build-arg BAKED_TTS_MODELS="kokoro" `
  -t $Tag -t jwindsor1/open-speech:latest .
```

Use immutable tags for validation (`cuda-<gitsha>` or release tags). Move `latest` only after the GPU host passes a smoke test.

## 4. GPU host pull/run validation

On `kitchen-pc` or another NVIDIA Docker host:

```bash
docker pull jwindsor1/open-speech:cuda-<gitsha>
docker run --rm --gpus all -p 8100:8100 \
  -v open-speech-hf:/home/openspeech/.cache/huggingface \
  -v open-speech-vad:/home/openspeech/.cache/silero-vad \
  -v open-speech-certs:/var/lib/open-speech/certs \
  -v open-speech-cache:/var/lib/open-speech/cache \
  -e STT_DEVICE=cuda \
  -e STT_COMPUTE_TYPE=float16 \
  -e TTS_DEVICE=cuda \
  jwindsor1/open-speech:cuda-<gitsha>
```

In another shell:

```bash
curl -sk https://localhost:8100/health
```

If TLS is disabled with `OS_SSL_ENABLED=false`, use `http://localhost:8100/health` instead.

## 5. Tagging pattern

Current Dockerfiles use good layer ordering: heavy provider/runtime dependencies are installed before app source, so Docker layer cache helps when rebuilding on the same builder.

There is not currently a separate published `base`/`runtime` image in this repo. If builds become painful even on the Windows builder, split the Dockerfile later into:

- `jwindsor1/open-speech:runtime-cuda-<runtime-version>` rebuilt rarely for Python/CUDA/provider dependencies
- `jwindsor1/open-speech:cuda-<gitsha>` rebuilt often for app source

Until that split exists, treat it as planned/optional. Do not document or publish tags that the Dockerfile does not actually build.
