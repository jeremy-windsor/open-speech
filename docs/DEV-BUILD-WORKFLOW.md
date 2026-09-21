# Development and Docker Build Workflow

This workflow keeps normal harness development off the CUDA build path. Use the local machine for edits and fast tests, use CPU Docker only for smoke checks, and use a Windows Docker host when the CUDA image is ready to publish.

## 1. Local thin development

Use the repo directly. Do not build the CUDA image for routine edits.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The suite isolates the voice library, studio database, conversation and composer outputs, provider
data, and TTS cache in a temporary directory before importing the application. For a full run, use
an isolated Python environment and a disposable checkout as well: some tests exercise paths relative
to the current working directory, and the core runtime dependencies are needed even for mocked API
tests. For a committed revision:

```bash
test_root="$(mktemp -d /tmp/open-speech-tests-XXXXXX)"
python3 -m venv "$test_root/venv"
"$test_root/venv/bin/python" -m pip install -e '.[dev]'
mkdir "$test_root/checkout"
git archive HEAD | tar -x -C "$test_root/checkout"
(cd "$test_root/checkout" && PYTHONDONTWRITEBYTECODE=1 \
  "$test_root/venv/bin/python" -m pytest -q -p no:cacheprovider)
```

Review the result and remove the temporary directory when finished. This runs the committed tree;
copy any uncommitted changes into the disposable checkout if those changes are what you intend to
test. The acceptance matrix and live model commands are in [TTS-BACKENDS.md](TTS-BACKENDS.md#validating-the-harness).

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

When code is ready, run the repository suite on the Windows machine in a disposable container. This
mounts the checkout read-only and keeps the temporary test packages and test data outside the live
service. The core image supplies Python and its runtime dependencies; the shell entrypoint must be
overridden so Docker does not start the server instead of pytest.

```powershell
docker run --rm --gpus all --entrypoint sh `
  -v "${PWD}:/work:ro" -w /work -e PYTHONDONTWRITEBYTECODE=1 `
  jwindsor1/open-speech:latest -c `
  "/opt/venv/bin/python -m pip install -q --target /tmp/testdeps pytest==8.4.2 pytest-asyncio==0.24.0 httpx==0.27.2 && PYTHONPATH=/tmp/testdeps:/work /opt/venv/bin/python -m pytest -q -p no:cacheprovider tests"
```

Then build an immutable GPU image on that Windows host. Test this tag with the real Windows GPU
before tagging or pushing `latest`.

```powershell
$Tag = "jwindsor1/open-speech:cuda-$(git rev-parse --short HEAD)"
docker build -f Dockerfile -t $Tag .
```

For a disposable package and GPU check before promotion, start the immutable tag on a separate port:

```powershell
docker run -d --rm --name open-speech-canary --gpus all -p 8110:8100 `
  -e OS_SSL_ENABLED=false -e OS_WYOMING_ENABLED=false `
  -e STT_DEVICE=cuda -e STT_COMPUTE_TYPE=float16 -e TTS_DEVICE=cuda $Tag
```

Exercise English, Japanese, and Chinese Kokoro voices without installing packages or changing cache
ownership in that container. Check all 52 Kokoro presets, every advertised TTS model's load and
uncached WAV, the 14 STT models, and the long-form distilled English models with a recording over
four minutes. Use the existing `scripts/benchmark_tts.py --mode live` for Live Reader first-audio and
completion timing. Record failures and skips separately. When the canary is finished, stop that
disposable container with `docker stop open-speech-canary`. The live service's cached STT weights
may be needed for the full STT matrix when the isolated canary has no model cache or outbound access.

Preserve the old `latest` image under a unique rollback tag before promotion. After canary gates
pass, tag the tested image as `latest` and recreate only the core service with the GPU and Qwen
Compose overrides:

```powershell
docker tag jwindsor1/open-speech:latest jwindsor1/open-speech:rollback-OLDREV
docker tag $Tag jwindsor1/open-speech:latest
docker compose -f docker-compose.yml -f docker-compose.gpu.yml `
  -f docker-compose.qwen3.yml --profile qwen3 up -d `
  --no-build --no-deps --force-recreate open-speech
```

Replace `OLDREV` with the revision of the previous live image. Verify `/health`, catalog counts,
a short GPU STT request, Kokoro audio, Qwen worker availability, and Live Reader before treating
the deployment as current. Then publish the validated image tags:

```powershell
docker push $Tag
docker push jwindsor1/open-speech:latest
```

Optional provider/model bake arguments:

```powershell
docker build -f Dockerfile `
  --build-arg BAKED_PROVIDERS="kokoro,piper,pocket-tts" `
  --build-arg BAKED_TTS_MODELS="kokoro" `
  -t $Tag .
```

The Qwen3 canary is not a baked provider. Validate and build its isolated image separately:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.qwen3.yml --profile qwen3 config
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.qwen3.yml --profile qwen3 build qwen3
```

Do not add `qwen-tts`, its Transformers pin, or its Torch runtime to the core image. On an RTX 2070,
verify `sm_75`, float16, and SDPA from worker health before attempting generation. Test one Qwen model at
a time and repeat the Kokoro scripture benchmark after unloading it.

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

Use immutable `cuda-<gitsha>` tags for tested builds and update `latest` only after the same image passes
the GPU smoke test. Keep Docker's BuildKit cache and the named model volumes; deleting either turns an
ordinary source rebuild into a dependency or model download.
