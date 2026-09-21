# Windows GPU regression follow-up, 2026-09-20

This follow-up records the fixes and repeat gates after
[the initial model validation](VALIDATION-2026-09-20.md). All model inference and
the repository test suite ran in containers on the Windows RTX 2070 SUPER host.
The Linux laptop only sent requests, prepared disposable inputs, and edited
source. Private recordings, transcripts, and generated audio remain outside Git.

## Fixes

- `f2f92c6` corrected the two `unknown_model` response assertions, made
  bare-metal default tests independent of image environment variables, and
  fixed Piper mocks to work regardless of test import order.
- The registry no longer advertises two Piper variants whose official weight
  directories do not contain ONNX files. This leaves 28 Piper IDs.
- Distilled English small and medium models use faster-whisper's batched
  inference pipeline with 15-second VAD chunks for recordings over 30 seconds.
  The pipeline returns global segment timestamps. Short recordings and other
  models retain the normal inference path.
- `90d96ee` adds Misaki's Japanese and Chinese dependencies, downloads the
  full UniDic dictionary at image build time, and makes pre-baked Hugging Face
  cache files readable by the service account.
- `8c1d75a` moves the UniDic download after the existing Kokoro weight prefetch
  so a Windows CUDA rebuild can reuse the cached large weight layer. The
  `cuda-8c1d75a` build completed on the Windows host in about 90 seconds with
  that layer cached. An earlier build attempt stalled Docker Desktop; restarting
  its engine recovered the existing service and volumes before the final build.

## Windows gates

The full repository suite ran in a disposable Windows Docker container with
the checkout mounted read-only, temporary pytest packages, and bytecode/cache
writing disabled: **864 passed, 9 skipped, 0 failed**. The skips are retained
as skips, not counted as model inference passes. The full command is in
[the build workflow](DEV-BUILD-WORKFLOW.md#3-windows-cuda-buildpush-box).

An isolated Windows GPU canary with the `cuda-f2f92c6` image initially showed
two image packaging faults: Japanese needed the downloaded UniDic dictionary,
and the baked Kokoro voices were in a root-owned cache. After applying those
same repairs in the disposable canary, **52/52 Kokoro voices** produced valid,
non-silent mono PCM16 WAVs. English, Japanese, and Chinese voices were included.
The fresh, immutable `cuda-8c1d75a` image then passed **52/52 Kokoro voices**
without mounts, package installation, or manual cache permission changes.

A second isolated `cuda-8c1d75a` canary mounted the existing Windows Hugging
Face cache read-only and passed **14/14 advertised STT IDs** with real GPU
inference on the short reference clip. All **30/30 built-in TTS IDs** loaded and
returned non-silent, mono PCM16 WAV from uncached synthesis. The optional Qwen
worker was not connected to this disposable canary. The three long-form STT
checks used a 274.67-second recording:

| Model | Output characters | Final segment end | Inference time |
| --- | ---: | ---: | ---: |
| Faster Whisper large-v3 Turbo | 3,107 | 274.50 s | 8.787 s |
| Distil Whisper small.en | 3,060 | 274.67 s | 5.390 s |
| Distil Whisper medium.en | 2,980 | 274.67 s | 5.099 s |

The prior medium run ended near 67.76 seconds and returned only 403 characters.
The new run covers the recording's full duration. These numbers measure output
coverage and speed, not transcription accuracy.

After the canary gates, the tested image was promoted to the live Windows GPU
core with the existing Qwen worker and persistent volumes. The previous image
was retained under a rollback tag. Both containers reached healthy status and
the live catalog showed **14 STT and 31 TTS IDs**. Every one of the **31/31
advertised TTS IDs** loaded and produced a valid, non-silent WAV from uncached
one-shot synthesis. Every one also completed a separate **31/31 Live Reader**
WebSocket run with non-silent PCM and a first-audio timing. Representative live
first-audio measurements for the same short sentence were Kokoro 0.264 s,
Pocket-TTS 0.182 s, Piper lessac-medium 0.117 s, and Qwen CustomVoice 11.780 s.
These include client/network time and do not measure sustained playback.

The live four-provider conformance run with synthesis and rejection probes
recorded **21 pass, 0 fail, 23 skip**. Qwen audio was initially skipped by that
tool because it does not load a model before synthesis. After an explicit Qwen
load, its focused conformance run recorded **7 pass, 0 fail, 7 skip**, including
audio and invalid-control rejections. Skipped worker manifest, generation-limit,
and WebSocket checks were tested separately only where stated here.

A live Kokoro WebSocket cancellation returned `response.cancelled`, and a second
utterance in the same session completed with PCM. In this multi-segment run, the
cancelled response ID did not match the first audio frame's ID; that first
segment may already have completed before cancellation. This run proves session
recovery, but not cancellation of a specific in-flight segment. A short live
Turbo STT request returned timestamped text; Kokoro was then loaded again and
produced non-silent audio. The live `/health` endpoint returned HTTP 200.

The immutable image is running on the Windows host. A Docker Hub push of
`cuda-8c1d75a` was rejected with `insufficient_scope`, so neither the immutable
tag nor the new `latest` tag was published to the registry in this run. The
Windows-local tags and previous-image rollback tag remain available.

## Remaining acceptance work

- The unscripted long recording can prove coverage and timestamp extent, but
  has no audited reference transcript. A known-transcript >4-minute recording
  is still needed for a defensible human-speech word error rate. A clone WAV
  needs a listening check to assess similarity to the reference speaker. The
  earlier Windows run proved one Qwen Base clone request, but Base remains an
  opt-in worker model and was not repeated after this core-image update.
- Sustain Live Reader playback and exercise cancellation of a known active
  segment, saved-profile rendering, model switching, forced generation limits,
  incompatible manifests, and worker outage/abort behavior before declaring
  full feature acceptance. The repository suite covers some of these paths with
  fixtures; a controlled live worker outage would interrupt the Windows service.
