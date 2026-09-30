# Test hardening audit — 2026-09-29

The repository was pulled before review. Production fixes are in `fb68594` and
`67c4fde` on `codex/windows-speech-gates`. Tests use disposable application data,
bounded audio fixtures, fake inference boundaries, and isolated Python 3.12 environments.

## Confirmed defects and fixes

| Defect | Observable failure | Repair and regression evidence |
| --- | --- | --- |
| Composer PCM decoding | Averaging stereo integer channels changed the dtype before scaling; 8-bit unsigned silence also decoded above zero. Ordinary audio could clip severely. | Normalize signed PCM and center unsigned PCM before downmixing. Six PCM dtype/channel cases; four failed before repair. |
| Configured composer storage | Voice and conversation WAVs in configured directories outside default storage were rejected. | Admit the configured voice/conversation roots and actual composer output directory, preserving resolved-path containment. Three configured-root cases failed before repair; a symlink escape remains rejected. |
| Windows audio file ownership | ffmpeg, Whisper, and pyannote tried to reopen still-open temporary files. ffmpeg silently returned the original, unconverted audio; inference could fail with permission errors. | Use temporary directories and closed input handles. Keep Whisper's file until its lazy segment iterator completes. Fault-injection cases cover all three consumers and inference failure cleanup. The original ffmpeg implementation also failed the real stereo conversion test on native Windows: 24 kHz remained 24 kHz instead of becoming 16 kHz. |
| Studio render validation | Zero/negative sample rates, unsupported formats, and non-finite track values reached renderers. These could cause arithmetic/allocation failures or invalid audio. | Reject them at the request boundary with HTTP 422. Eight parameterized cases failed before repair and verify that rendering never starts. Preserve the existing behavior of clamping negative track offsets. |
| Conversation sample rates | Studio synthesis discarded the requested sample rate, then conversation rendering labeled native PCM with that rate. Duration and pitch changed. | Resample the assembled waveform using the selected model's native rate. Four duration/frequency cases, including unchanged rates; three failed before repair. |
| Host text encoding | Native Windows `/web` crashed decoding the UTF-8 HTML as CP1252. Unicode pronunciation dictionaries were also corrupted by the default encoding. | Read both explicitly as UTF-8. The web failure was observed in the full Windows suite, and three portable locale regressions failed before repair. |

Added 33 deterministic regression/boundary cases. Of these, 26 were observed failing
against the corresponding unrepaired implementations. No existing test was removed.

## Test harness corrections

- Composer output-path assertions now inspect `Path.parts`, retaining the directory
  and filename checks on both Windows and Linux.
- The two symlink tests skip only Windows error 1314 (missing symbolic-link privilege).
  Their containment and cleanup assertions still run on Linux.
- Session teardown closes import-time SQLite stores and collects abandoned test
  connections before removing test data. Native Windows previously reported locked
  database errors during interpreter cleanup.
- Python 3.12 is explicit in the README. An existing environment linked to a newer
  system interpreter can stop seeing its installed dependencies and is not a valid gate.

## Verification

Run from the repository root; Node.js and ffmpeg must be on `PATH`.

```bash
python -m pytest -q -rs -p no:cacheprovider
python -m pytest -q tests/test_audio_conversion.py tests/test_composer_unit.py \
  tests/test_studio_synthesis.py tests/test_studio_validation.py tests/test_text_encoding.py
node --check src/static/app.js
bun test client-js/tests
bun x --package typescript tsc -p client-js/tsconfig.json --noEmit
ruff check src tests --output-format concise --no-cache
git diff --check
```

| Gate | Result |
| --- | --- |
| Initial Linux suite without Node | 922 passed, 19 skipped, 34 warnings, 31.99 s |
| Final Linux suite with Node and ffmpeg | 972 passed, 2 skipped, 33 warnings, 24.35 s |
| JavaScript SDK | 4 passed; 11 assertions |
| JavaScript syntax / TypeScript checking | Passed |
| Dependency consistency | `uv pip check`: all installed packages compatible |
| Ruff | 216 pre-existing findings; comparison by file/rule/message found no new diagnostics |
| Native Windows suite with Node and ffmpeg | 970 passed, 4 skipped, 31 warnings, 25.24 s; exit 0 and clean SQLite teardown |
| Windows-hosted CUDA container, focused provider/regression suite | 46 passed, 0 skipped, 1 warning, 8.56 s |
| CUDA integration smoke | Health and web page passed; Kokoro synthesis, conversation resampling, Live Reader streaming, and Whisper transcription passed |

Linux used Python 3.12.14 and the declared development extras. Windows additionally
constrained runtime dependencies with `requirements.lock`. Linux's two skips require
Torch. Neither coverage tooling nor a coverage gate is configured; no coverage package
was added and no percentage is claimed.

### Windows and CUDA evidence

Validation ran through remote desktop on a Windows GPU machine, in a fresh temporary
clone of `67c4fde` and a managed Python 3.12.14 environment. The first native full run
found four failures: two unavailable symlink privileges, a Unix-only path assertion,
and the real `/web` encoding crash. The final run fixed the product and harness defects;
only the two Torch-dependent cases and two privilege-dependent symlink cases skipped.
An isolated reproduction of the original converter failed with the actual Windows
ffmpeg executable, confirming the file-sharing problem beyond the mocked boundary.

A separate, temporary container used the existing provider image with the reviewed
source and tests mounted read-only. It exposed only a loopback port, used cached model
weights offline, and did not replace the running application. CUDA was available on
an RTX 2070 SUPER with 8 GiB VRAM. The focused command was:

```bash
docker exec open-speech-audit python -m pytest -q -rs -p no:cacheprovider \
  tests/test_tts_kokoro.py tests/test_audio_conversion.py \
  tests/test_studio_synthesis.py tests/test_studio_validation.py \
  tests/test_text_encoding.py
```

The bounded integration check used a synthetic sentence and asserted:

- `/health` returned healthy and `/web` returned HTTP 200.
- Kokoro produced nonempty mono PCM WAV at 24 kHz, lasting 2.085 seconds.
- The same sentence rendered through a conversation at 16 kHz, lasting 2.084 seconds;
  the returned duration agreed with the WAV header.
- Live Reader emitted 46 valid PCM frames, totaling 109,668 bytes, and completed
  after the client acknowledged each frame.
- Faster Whisper base returned a nonempty English transcript of the generated WAV.

The temporary container was stopped and removed after verification. Existing speech
services remained running. This validates representative cached providers and the
patched source in the existing image; it does not validate a fresh image build or
constitute production deployment.

## Remaining gaps

1. No checked-in CI workflow runs the ordinary suite or JavaScript SDK on pushes.
   A Linux/Windows Python 3.12 matrix with Node and ffmpeg would catch these platform
   regressions automatically.
2. Ruff remains a failing repository-wide gate with 216 existing findings. This audit
   keeps that debt separate from behavioral fixes.
3. The JavaScript client build requires TypeScript, but its package does not declare
   that development dependency or a test script. The audit used temporary tooling;
   a clean client install alone does not provide a reproducible build/test setup.
4. `requirements.lock` pins direct core runtime packages, not all transitive or
   optional provider dependencies. The latest-compatible Linux run and constrained
   Windows run improve evidence but are not a complete reproducible provider lock.
5. Passing contract tests and representative GPU smoke tests do not qualify every
   registry model, language, voice, long-form recording, or cloning provider for
   quality, latency, and memory use. Keep per-model acceptance separate.
6. Actual microphone capture, speaker playback quality, and sustained interactive
   Live Reader behavior still need dedicated end-to-end acceptance. A generated WAV
   or WebSocket exchange alone does not establish those results.

Raw recordings, generated audio, private machine details, and test logs remain outside Git.
