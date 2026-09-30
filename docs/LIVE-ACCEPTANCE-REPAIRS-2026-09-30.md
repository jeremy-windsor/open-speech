# Live acceptance repairs — 2026-09-30

All 12 confirmed failures from the [original audit](LIVE-ACCEPTANCE-2026-09-29.md)
have source repairs and focused verification. This run used real Piper and
faster-whisper inference, the official OpenAI Python SDK, HTTP/WebSocket calls,
FFmpeg decoding, and the browser. No new test source was retained.

[Measured results](LIVE-ACCEPTANCE-REPAIRS-2026-09-30.json) preserve output sizes,
sample counts, event names, transcripts, and separate regression results.

## Repairs and actual observations

| Original failure | Repair | Observed behavior |
| --- | --- | --- |
| Default official SDK Realtime handshake fails | Negotiate only an offered subprotocol; default connections use GA sessions | `openai==3.22.1` connects normally without a forced protocol |
| Nested GA audio settings ignored; legacy output events | Parse and echo nested input/output formats, voice, speed, transcription and turn detection; emit GA output lifecycle events | Real output and input transcription succeeded with PCM, PCMU and PCMA; output events use `response.output_audio.delta` |
| `stream_format` and voice objects rejected | Accept `audio`/`sse` and `{id: ...}`; normalize named-voice UUIDs | All six OpenAI speech formats decoded in both streaming modes; a real named voice UUID resolved its realization and synthesized WAV |
| Piper speaker IDs ignored | Preserve the requested speaker and pass its mapped `speaker_id` to Piper | LibriTTS speakers 3922 and 8699 passed actual IDs 0 and 1 into real synthesis; invalid speakers return 400 |
| Piper catalog ignores requested model | Read metadata for that model and serialize catalog access with its provider lock | LibriTTS lists 904 speakers; Lessac lists its own single voice while LibriTTS is resident |
| Piper artifact deletion targets the wrong repository | Delete only the selected ONNX/config snapshots and unused repository/shared blobs, with the provider lock held | Five selected paths removed; nine other-model files preserved; repeat deletion returns `not_found`; actual reload generated 131,518 PCM bytes |
| Native-rate headerless PCM | Resample speech PCM to 24 kHz; stream through a continuous resampler; invalidate old PCM cache identities | 25,600 native samples at 16 kHz became exactly 38,400 samples; 36,352 samples at 22.05 kHz became 39,567 samples, matching the 24 kHz target |
| Invalid audio returns 500 | Map the decoder's invalid-data rejection to a client error | Actual malformed transcription and translation requests both returned 400 |
| Project Realtime voice override ignored | Use the response voice override and validate through the synthesis router | Project client valid overrides produced audio; an invalid override produced zero audio, an error event and a failed terminal response |
| Unknown model lookup fabricates an object | Require a registered or actually advertised model | Official SDK unknown-model retrieval returned 404 |
| Unknown legacy pull returns 500 | Map invalid-model ValueError to 404 | Actual `/api/pull/does-not-exist` returned 404 |
| Composer failed render leaves spinner | Clear failure status, hide the old result, and restore the render button in `finally` | Browser missing-source request returned 400 and displayed `Render failed`; correcting the source rendered successfully and displayed `Done` |

The deletion checks also exercised multiple snapshot revisions, a blob retained
by another model, and regular-file caches used when symlinks are unavailable.
Thirty-six concurrent model/status/catalog polls succeeded during a real Piper
deletion; a following reload generated 130,560 PCM bytes. Local Piper status and
catalog reads retain its inference lock even though synthesis now accepts a model ID.
Deleting metadata alone does not label a model downloaded: an unloaded Lessac
High catalog query left its state `provider_installed` until weights are fetched.

## Additional dependency repairs

The stock denoising failure came from an absent `noisereduce` package. It is now
included in core dependencies and the pinned runtime requirements. With denoising
enabled, real tiny.en transcription returned:

> The quick brown fox jumps over the lazy dog. This is a real voice compatibility check.

A fresh install exposed another actual failure: faster-whisper 1.2.1 calls
`av.open(..., metadata_errors="ignore")`, which PyAV 19 removed. Core installation
now requires `av>=11,<19`; the pinned runtime uses the validated PyAV 18.1.0.
The [upstream changelog](https://github.com/PyAV-Org/PyAV/blob/main/CHANGELOG.rst)
records the removed argument.

Compressed streaming keeps codec-selected output rates; only speech PCM requests
24 kHz explicitly. Encoder failures now raise instead of silently finishing an
empty stream. Actual MP3, Opus, AAC, FLAC, WAV and PCM streams decoded successfully,
including the reconstructed audio from SSE events.

## Regression checks and limits

- Existing suite: **966 passed, 20 skipped, 1 deselected**. The excluded
  `tests/test_api.py::test_get_model` explicitly expects a fabricated `some-model`
  lookup to return 200. That expectation contradicts the requested 404 repair.
  No retained test source was created or changed.
- Changed Python files introduced **zero new lint findings**; existing findings
  decreased from 35 to 32. Diff whitespace checks and installed dependency checks passed.
- Runtime: isolated Linux CPU service, Python 3.12.14, Piper 1.4.2,
  faster-whisper 1.2.1, PyAV 18.1.0 and OpenAI SDK 3.22.1. Piper Lessac Low,
  Lessac Medium and LibriTTS-R performed actual inference; tiny.en performed actual ASR.
- This focused run did not repeat the original 50-model GPU matrix or qualify a
  freshly rebuilt release image, native Windows deployment, or production rollout.
  Production containers were not changed.
- No OpenAI cloud request was made. These are official-client compatibility
  results against the local service. SSE terminal events omit token usage because
  local providers do not report it; full OpenAI Realtime conversation/tool parity
  and hosted voice quality remain outside the application's audio-only contract.
- Diarization still needs the optional pyannote dependency and gated model access.
  No model agreement was accepted. The original simultaneous-residency/RAM limit
  and perceptual quality gates remain unqualified.
- Disposable callers, service, environment, model cache and application data were
  removed after verification. Ordinary pip/uv package download caches were preserved.
