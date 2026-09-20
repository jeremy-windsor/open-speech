# Windows GPU model validation, 2026-09-20

This is an observed run, not a claim that every advertised feature is production ready. The tested
Windows checkout was `22d93d7` (core image `sha256:e25d6907c8c5`, Qwen worker image
`sha256:808c96e272f1`) on an RTX 2070 SUPER with 8 GB VRAM. The Linux laptop made HTTPS requests
and prepared audio files; **all STT and TTS inference ran in the Windows Open Speech containers**.
The only model changes on Windows were loads/downloads and a temporary Qwen Base opt-in. The worker
was returned to `QWEN3_ENABLE_BASE=false`; the final catalog again omitted Base, `/health` returned
200, and Kokoro loaded on CUDA and produced non-silent audio. No personal recording or transcript was
added to the repository or voice library.

## Inputs and measurement limits

- A 14.55-second scripted recording with 47 written words supplied the STT reference. Each model
  returned one word fewer than the written script (2.13% word error rate after normalization). The
  exact spoken words were not independently audited, so this is a **script comparison**, not a
  certified accuracy score or a meaningful ranking among the 14 models.
- A 274.67-second unscripted recording tested long-form coverage and latency. It has no verified
  transcript, so text length and last timestamp are diagnostics, not accuracy scores.
- TTS used the same nine-word English sentence for every advertised ID. Each loadable ID received
  two uncached one-shot WAV requests and a Live Reader request. WAV checks required the advertised
  sample rate, mono PCM16, positive duration, and non-silent samples. Warm WAV is the second request;
  Live first PCM is time to the first Live Reader audio delta. Network transfer is included. These
  are short-sentence timings, not sustained reading or subjective quality scores.
- The 31 generated TTS WAVs were transcribed by the Windows Turbo STT model: 29 reproduced all nine
  words; Piper `en_US-arctic-medium` had one word difference and Qwen CustomVoice had two. This is
  a rough intelligibility proxy, not a listening test.

## STT: all 14 advertised IDs

Every ID loaded on CUDA and returned HTTP 200 with valid timestamped segments for both recordings.
The timings below exclude model load/download time. The first uncached large-v2 load took 169.309 s;
the short scripted inference times ranged from 0.670 to 9.320 s. The long-form text column is a
coverage signal only.

| Model | 14.55 s script | WER vs script | 274.67 s clip | Long text chars |
|---|---:|---:|---:|---:|
| `deepdml/faster-whisper-large-v3-turbo-ct2` | 1.182 s | 2.13% | 14.342 s | 3,107 |
| `Systran/faster-whisper-medium.en` | 1.551 s | 2.13% | 18.214 s | 3,056 |
| `Systran/faster-whisper-tiny` | 0.671 s | 2.13% | 8.211 s | 3,155 |
| `Systran/faster-whisper-base.en` | 0.806 s | 2.13% | 7.110 s | 3,112 |
| `Systran/faster-whisper-base` | 0.907 s | 2.13% | 7.355 s | 3,097 |
| `Systran/faster-whisper-small.en` | 1.025 s | 2.13% | 10.244 s | 3,064 |
| `Systran/faster-distil-whisper-large-v3` | 1.015 s | 2.13% | 6.880 s | 3,149 |
| `Systran/faster-whisper-large-v3` | 1.985 s | 2.13% | 40.848 s | 3,098 |
| `Systran/faster-whisper-large-v2` | 9.320 s | 2.13% | 32.003 s | 3,108 |
| `Systran/faster-distil-whisper-small.en` | 1.118 s | 2.13% | 6.615 s | 1,514 |
| `Systran/faster-whisper-small` | 1.050 s | 2.13% | 10.535 s | 3,082 |
| `Systran/faster-distil-whisper-medium.en` | 0.688 s | 2.13% | 6.577 s | 403 |
| `Systran/faster-whisper-medium` | 4.037 s | 2.13% | 18.590 s | 3,068 |
| `Systran/faster-whisper-tiny.en` | 0.670 s | 2.13% | 6.082 s | 3,080 |

The unchunked distilled medium result stopped at timestamp 67.76 s, despite a 274.67 s upload.
A second full-file request reproduced the cutoff. A separate late excerpt returned speech, so the
model can recognize that part of the recording. The unchunked distilled small result reached the
last timestamp but omitted substantial content. As an A/B diagnostic, splitting the original into
19 local 15-second WAVs and submitting **each chunk to the Windows models** gave 19/19 nonempty
responses: 3,056 characters in 11.255 s of summed inference for distilled medium, and 3,081
characters in 11.060 s for distilled small. This demonstrates a long-file path problem; a production
fix must handle overlap, sentence joins, and global timestamp offsets. The upstream
[Distil-Whisper model card](https://huggingface.co/distil-whisper/distil-medium.en) recommends
15-second chunking for long audio.

The Windows Turbo model also returned HTTP 200 for `json`, `text`, `srt`, `vtt`, and `verbose_json`
on the scripted recording; SRT/VTT had timestamps and VTT had its required header.

## TTS: every advertised ID

The normal manifest advertised 33 TTS IDs: Kokoro, Pocket-TTS, 30 Piper IDs, and Qwen CustomVoice.
**31/33 loaded, produced two valid WAVs, and completed Live Reader; two Piper IDs failed load.**
`cuda`/`cuda:0` below means inference used the Windows GPU. Piper and Pocket-TTS used the CPU on that
same Windows host. The table records observed warm one-shot completion and first Live Reader PCM, in
seconds; `—` means no inference succeeded.

| Model | Device | Result | Warm WAV | Live first PCM |
|---|---|---|---:|---:|
| `kokoro` | cuda | pass | 0.680 | 0.188 |
| `pocket-tts` | cpu | pass | 1.360 | 0.170 |
| `qwen3/0.6b-custom-voice` | cuda:0 | pass | 13.427 | 12.900 |
| `piper/en_US-lessac-low` | cpu | pass | 0.148 | 0.147 |
| `piper/en_US-lessac-medium` | cpu | pass | 0.130 | 0.120 |
| `piper/en_US-lessac-high` | cpu | pass | 0.423 | 0.418 |
| `piper/en_US-amy-medium` | cpu | pass | 0.197 | 0.149 |
| `piper/en_US-amy-high` | — | load failed | — | — |
| `piper/en_US-arctic-medium` | cpu | pass | 0.140 | 0.155 |
| `piper/en_US-bryce-medium` | cpu | pass | 0.205 | 0.167 |
| `piper/en_US-danny-low` | cpu | pass | 0.101 | 0.119 |
| `piper/en_US-hfc_female-medium` | cpu | pass | 0.126 | 0.112 |
| `piper/en_US-hfc_male-medium` | cpu | pass | 0.221 | 0.201 |
| `piper/en_US-joe-medium` | cpu | pass | 0.126 | 0.361 |
| `piper/en_US-john-medium` | cpu | pass | 0.153 | 0.138 |
| `piper/en_US-kathleen-low` | cpu | pass | 0.096 | 0.101 |
| `piper/en_US-kusal-medium` | cpu | pass | 0.136 | 0.119 |
| `piper/en_US-libritts_r-medium` | cpu | pass | 0.101 | 0.100 |
| `piper/en_US-ljspeech-medium` | cpu | pass | 0.160 | 0.134 |
| `piper/en_US-ljspeech-high` | cpu | pass | 0.593 | 0.526 |
| `piper/en_US-norman-medium` | cpu | pass | 0.152 | 0.114 |
| `piper/en_US-ryan-low` | cpu | pass | 0.083 | 0.083 |
| `piper/en_US-ryan-medium` | cpu | pass | 0.177 | 0.148 |
| `piper/en_US-ryan-high` | cpu | pass | 0.386 | 0.382 |
| `piper/en_GB-alan-low` | cpu | pass | 0.106 | 0.099 |
| `piper/en_GB-alan-medium` | cpu | pass | 0.148 | 0.138 |
| `piper/en_GB-cori-medium` | cpu | pass | 0.118 | 0.107 |
| `piper/en_GB-cori-high` | cpu | pass | 0.412 | 0.411 |
| `piper/en_GB-jenny_dioco-medium` | cpu | pass | 0.136 | 0.116 |
| `piper/en_GB-northern_english_male-medium` | cpu | pass | 0.206 | 0.175 |
| `piper/en_GB-semaine-medium` | cpu | pass | 0.171 | 0.136 |
| `piper/en_GB-southern_english_female-low` | cpu | pass | 0.136 | 0.092 |
| `piper/en_GB-southern_english_female-medium` | — | load failed | — | — |

The two failed Piper loads returned HTTP 400 `load_failed` after upstream 404s for the ONNX files.
The registry lists nonexistent variants: the official
[Amy directory](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy) has only low
and medium, while the
[Southern English female directory](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/southern_english_female)
has only low. Remove the invalid IDs from the registry/backend, or replace them with actual upstream
models, then recheck catalog and load behavior.

## Voices, clone, and conformance

The voice sweep requested every 69 selectable preset voices. Kokoro succeeded for 39/52: all five
Japanese and eight Chinese voices returned HTTP 500. A representative Japanese failure was
`No module named 'pyopenjtalk'`; Chinese failed with `No module named 'ordered_set'`. Pocket-TTS
succeeded for 8/8, and Qwen CustomVoice for 9/9. The Docker image installs the basic Kokoro extra,
while upstream [Misaki's optional dependencies](https://github.com/hexgrad/misaki/blob/main/pyproject.toml)
put those packages in its `ja` and `zh` extras. Add the language dependencies to the image, or
withhold those voices until the image can synthesize them; then rerun the language sweep.

Qwen Base was enabled briefly to test voice cloning with a 7.05-second PCM16 WAV excerpt and its
verified written transcript. It loaded on `cuda:0` in 62.724 s and returned a non-silent 2.833 s,
24 kHz mono PCM16 WAV in 20.299 s. Windows Turbo STT transcribed the clone to the requested
sentence. This proves one functional clone path, **not resemblance to the speaker**. The user must
listen to the private sample and decide whether it sounds like them. Base was then disabled again,
and the core catalog no longer advertised it.

The conformance report with rejection probes across all 33 normal TTS IDs recorded **105 pass,
0 fail, 229 skip**. Its metadata passes did not catch the missing Piper weights; skips are not
successful inference. The earlier isolated repository suite on the local checkout recorded
**865 passed, 2 failed, 2 skipped**. The two failures are stale `detail.code` expectations in
`tests/test_unified_api.py`; the actual central handler returns `error.code=unknown_model`.
The live Windows runtime has no `pytest` dependency. A separate disposable container on the same
Windows host used the unchanged core image, a read-only mount of the checkout fast-forwarded to
`46983cb`, and temporary `pytest`, `pytest-asyncio`, and
`httpx` packages. Its full suite recorded **844 passed, 16 failed, 9 skipped**. Failures clustered
in environment-compatibility tests that assumed bare-metal defaults, the test that assumed Wyoming
was disabled, and Piper mock tests. The GPU image supplies different default environment variables;
app registration also imports the real Piper package before the mock tests use
`sys.modules.setdefault`. A Piper load test passed when run alone in a fresh disposable container,
confirming test-order/environment sensitivity. The other two failures were the known
`unknown_model` assertions. The live containers were not modified, and the Windows suite is **not
green**.

## Regression gates to add or rerun

1. Correct the two unknown-model assertions. Make bare-metal-default tests explicitly clear image
   environment overrides, and make Piper mocks independent of module import order. Rerun the full
   suite in a disposable Windows container with read-only source and temporary test data. Keep
   test data outside the app's persistent voice/profile paths.
2. Fix the Piper catalog and Kokoro language dependencies, then repeat the failed model/voice
   cases and the full advertised-ID load, uncached WAV, and Live Reader matrix. Make a load failure
   a failed gate even when conformance metadata passes.
3. Implement long-form chunking for the affected distilled English STT models with overlapping
   windows, correct timestamp offsets, and sentence joins. Add a >4-minute known-transcript
   fixture and assert coverage and word error rate, not just HTTP 200 or a final timestamp.
4. Keep the short scripted STT fixture for format and latency smoke checks; get a recording whose
   exact spoken words have been audited before using WER to compare model accuracy. For TTS,
   compare samples by listening and rerun latency at cold and warm states and at longer lengths.
5. Exercise sustained Live Reader playback, cancellation/recovery, saved profiles, model-switch
   races, forced generation limits, incompatible manifests, and worker outage/abort handling.
   Schedule live outage drills because they interrupt the Windows service. Verify Kokoro restoration
   and a short STT request before promoting a new image.

Keep personal clips, reference audio, generated samples, raw transcripts, and host-specific test
results outside Git. The model and voice tables above contain only IDs, status, timing, and counts.
