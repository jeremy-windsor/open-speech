# Development

- [Install from source](#install-from-source), [Tests](#tests), [Smoke test](#smoke-test),
  [Building images](#building-images)

## Install from source

Open Speech supports **Python 3.12 only**. The server also runs the `ffmpeg` program, which uv doesn't
install, so put it on `PATH` first: `sudo apt install ffmpeg`, `brew install ffmpeg`, or
`winget install Gyan.FFmpeg`. Without it every mp3, opus, aac and flac answer fails with `500 ffmpeg not found`,
and mp3 is what an OpenAI app gets by default, what the web app's Speak and voice previews ask for, and what every
Reader part and Composer mix is made of. Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/jeremy-windsor/open-speech.git
cd open-speech
uv sync --frozen --extra cpu --extra all  # Locked CPU runtime and core TTS providers
# Add --extra dev for development, or select tts, piper, pocket, or supertonic individually.
```

For CUDA 12.8, install with this command instead, and set the devices in `.env` (below):

```bash
uv sync --frozen --extra cuda --extra all --no-install-package onnxruntime
```

The cuda extra brings `onnxruntime-gpu`, which provides the same `onnxruntime` Python package as the CPU build that
faster-whisper and Piper require; with both installed they overwrite each other's files, so ONNX Runtime can lose
its CUDA provider or break outright. `--no-install-package onnxruntime` leaves the CPU build out. For the same
reason, run commands in a CUDA install with `uv run --no-sync` (or set `UV_NO_SYNC=1` in the shell): a `uv run`
that syncs the environment installs the CPU build again.

The source defaults store data under `/home/openspeech/data`, the TTS cache under
`/var/lib/open-speech/cache`, and generated TLS certs under `/var/lib/open-speech/certs`. Those paths
suit the Docker image but usually are not writable on your own machine, so point them at a local
folder. The server does not read `.env` by itself; `uv run --env-file` loads it. Save this as `.env`
in the repository root:

```bash
OS_SSL_ENABLED=false
OS_STUDIO_DB_PATH=data/studio.db
OS_VOICE_LIBRARY_PATH=data/voices
OS_CONVERSATIONS_DIR=data/conversations
OS_COMPOSER_DIR=data/composer
TTS_CACHE_DIR=data/cache
READER_CACHE_DIR=data/reader
```

For a CUDA install (`--extra cuda`), add these lines too. Without them Whisper and every TTS model run on the CPU,
the source default:

```bash
STT_DEVICE=cuda
STT_COMPUTE_TYPE=float16
TTS_DEVICE=cuda
```

Batch job uploads and the batch job database go next to `OS_STUDIO_DB_PATH`. Every other setting is in
[Configuration](CONFIGURATION.md).

`uv run --env-file` reads `.env` by rules of its own ([Writing values in .env](CONFIGURATION.md#writing-values-in-env)):
single-quote a value that holds a `$`, a double quote, a blank or a backslash, as in `OS_API_KEY='Xk9$mP2q'` or
`OS_ASSISTANT_EXTRA_BODY='{"top_k": 20}'`, and write an apostrophe as `\'`, as in `OS_API_KEY=Xk9\'mP2q`. Otherwise
uv cuts the value, or stops reading `.env` at that line and leaves every setting below it unset.

Then start the server:

```bash
uv run --frozen --env-file .env python -m src.main
# A CUDA install: uv run --no-sync --env-file .env python -m src.main
```

It listens on `http://localhost:8100` (web UI at `/web`). To keep HTTPS, remove `OS_SSL_ENABLED=false`
and set both `OS_SSL_CERTFILE` and `OS_SSL_KEYFILE` to a writable cert/key path; Open Speech generates a
self-signed pair there if neither file exists yet. If only one of the two files exists, startup stops
instead of overwriting it. Generating the pair runs the `openssl` program, which must then be on `PATH` (Linux and
macOS have one; on Windows, Git for Windows ships one in its `usr\bin` folder), or startup stops with
`openssl not found`; a pair of your own needs no `openssl`.

At startup the server also loads `TTS_MODEL` (Kokoro by default) in the background when its provider
is installed. On the first start this downloads the model's weights. Set `TTS_PRELOAD_DEFAULT=false`
to load it on first use instead.

## Tests

Run the tests from the repository root, on Python 3.12. Keep the extras that
[Install from source](#install-from-source) used (for a CUDA install, `--extra cuda` in place of
`--extra cpu`, with `--no-install-package onnxruntime` on the sync and `uv run --no-sync` in place of
`uv run --frozen --extra ...`): uv's sync removes every package the extras it is given don't need, so leaving
out `--extra all` uninstalls Kokoro, Supertonic, Piper and Pocket-TTS.

```bash
uv sync --frozen --extra cpu --extra all --extra dev
uv run --frozen --extra cpu --extra all --extra dev python -m pytest -q -rs tests
```

The same commands work in PowerShell on Windows. Pass `tests` as the path, not `.`: the `./data` directory that Compose mounts can hold third-party
packages with test files of their own, which break collection.

The suite runs with a home folder and a Hugging Face cache of its own, under its temporary folder (`HOME`,
`USERPROFILE`, `XDG_CACHE_HOME`, `HF_HOME` and `HF_HUB_CACHE` are set for it, and `STT_MODEL_DIR` and
`HUGGINGFACE_HUB_CACHE` from your shell are dropped), so it never reads or deletes the models you downloaded.

Put Node.js and ffmpeg on `PATH` to exercise browser behavior and real audio encoding.
Node.js 22.18 or newer also runs the client-js checks in `tests/test_client_sdk.py`, which load
`client-js/src/index.ts` without a build step; older versions skip them.
Six Kokoro unit tests require the optional
Torch runtime. Windows skips the four symlink tests only when the account lacks symbolic-link privilege; Linux
still exercises them.
The JavaScript SDK's own tests (`client-js/tests/index.test.ts`) use the Node.js test runner: the
suite runs them through `tests/test_client_sdk.py`, and `node --test client-js/tests/index.test.ts`
runs them alone.
GitHub Actions (`.github/workflows/tests.yml`) runs the Python test suite on Python 3.12, with
Node.js 24, for pushes to `main` and for pull requests.

## Smoke test

To check a running server end to end the way an OpenAI app would use it, run the smoke test. It needs
only the Python standard library (Python 3.9 or newer), so it works without installing anything:

```bash
python scripts/smoke_openai_api.py --url https://localhost:8100 --insecure
```

It checks `/health` and `/v1/models`, synthesizes a Kokoro WAV through `/v1/audio/speech`, transcribes
it back, checks the `text`, `srt`, `vtt`, and `verbose_json` formats, and tries the OpenAI model names
`tts-1` and `whisper-1`. `--insecure` accepts the self-signed certificate. If `OS_API_KEY` is set in
your shell, the script sends it. The first run can take minutes while model weights download; exit
code `0` means every check passed.

## Building images

CPU and CUDA use the same Dockerfile and frozen `uv.lock`. Builds install only the selected
providers and download weights on first use into persistent model caches. `BAKED_TTS_MODELS`
can explicitly prefetch models. The UI ships in the image. `/health` reports its build revision.
After changing dependencies, run `uv lock` and export the pip-compatible core requirements
with `uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate --no-header -o requirements.lock`.

Build and validate an immutable CUDA tag on the GPU host:

```powershell
$Tag = "jwindsor1/open-speech:cuda-$(git rev-parse --short HEAD)"
docker build --build-arg DEVICE=cuda --build-arg REVISION=$(git rev-parse HEAD) -t $Tag .
docker run -d --rm --name open-speech-canary --gpus all -p 8110:8100 `
  -e OS_SSL_ENABLED=false -e OS_WYOMING_ENABLED=false `
  -e STT_DEVICE=cuda -e STT_COMPUTE_TYPE=float16 -e TTS_DEVICE=cuda $Tag
```

For CPU, build with `DEVICE=cpu`; `docker-compose.cpu.yml` is the CPU profile with smaller defaults.
Deploy a verified tag with `OPEN_SPEECH_IMAGE`; GPU Compose layers
`docker-compose.yml` and `docker-compose.gpu.yml`.

Each GPU voice worker image builds with `providers/` as its context and `<name>/Dockerfile` as its Dockerfile, so
it can copy the shared `providers/_common` code. Compose builds them with `up -d --build`
([GPU voice workers](CONFIGURATION.md#gpu-voice-workers)); to build one by hand, run
`docker build -f providers/chatterbox/Dockerfile providers` (or `qwen3`, `fish`, `breeze`). The Breeze image is
built for Turing (RTX 20) GPUs; `--build-arg BREEZE_CUDA_ARCHITECTURES=86` (RTX 30, or `89`, `120`) builds it for
your card.

Backend-specific conformance commands are in [TTS-BACKENDS.md](TTS-BACKENDS.md#validation).
