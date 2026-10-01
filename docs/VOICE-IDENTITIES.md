# Named voices and reading presets

Named voices separate a voice's identity from the settings used to read with it.

- A **named voice** has a stable UUID and a name, such as Narrator.
- A **realization** specifies exactly one model, its provider voice or blend, and an
  optional saved reference recording. A voice can have one realization per exact model.
- A **reading preset** selects a voice/model and stores speed, effects, delivery
  instructions, and output format. Existing `/api/profiles` endpoints remain the preset API.

Having a Kokoro realization does not create a Qwen or Chatterbox clone. Adding another
realization is explicit, and using the same recording does not establish acoustic similarity.
The selected model must exist and support its controls; no provider or voice is substituted.

## Use in the web UI

1. In Speak, choose the provider and exact model, then a provider voice, blend, or saved
   cloning reference. Voice Lab's **Use in Speak** action supplies an existing recording.
2. Choose **Save named voice**. Enter a new name, or an existing voice's name to add a
   realization for another model or update the current version after confirmation.
3. Select that voice using **Named voice**, then set speed, effects, instructions, and format.
4. Choose **Save reading preset**. Multiple presets can use the same named voice.

In Settings, **Named voices** lists each voice's model versions and linked preset count.
Use **Edit** to load a version into Speak, **Rename** to change its label, or remove a
version or the whole voice. Updates apply to linked presets without changing their reading
controls. Removal keeps those presets as standalone recipes and preserves recordings.

Changing the provider voice, blend, or reference detaches the named-voice selection.
Changing speed or effects changes the reading recipe. Voices without a usable realization
remain visible but disabled for the selected model. Offline providers and missing references
retain their identity and realization records.

Speak, Live Reader, and Conversation resolve the exact model's realization. Conversation
also honors legacy blend strings, preset effects followed by turn effects, and instructions.
Missing saved conversation presets fail rather than falling back to the configured voice.
Effects are supported by complete HTTP synthesis and Conversation; Live Reader does not apply
the effects panel. Its model capabilities still determine whether it supports a given voice.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/voices/identities` | Create a named voice with `name` |
| GET | `/api/voices/identities?model=kokoro` | List identities and realization availability |
| GET | `/api/voices/identities?check_availability=false` | List saved voices without contacting providers |
| PATCH/DELETE | `/api/voices/identities/{id}` | Rename or remove an identity |
| POST | `/api/voices/identities/{id}/realizations` | Add `model`, `voice`, optional `reference_audio_id` |
| PUT/DELETE | `/api/voices/identities/{id}/realizations/{model}` | Update or remove an exact model version |
| POST/PUT | `/api/profiles[/{id}]` | Save a reading recipe; optional `voice_identity_id` and `instructions` |
| GET | `/api/profiles/{id}/resolve` | Resolve a preset's exact voice recipe |

Use the returned `voice` string (`voice:<UUID>`) with the ordinary speech request:

```json
{
  "model": "kokoro",
  "voice": "voice:<UUID returned by the identity API>",
  "input": "A short reading sample.",
  "speed": 1.2,
  "response_format": "wav"
}
```

The same string is accepted in Live Reader's `session.update`. HTTP synthesis resolves
the selected model's voice and stored reference before validating controls or using the cache.
Inline reference audio, design prompts, and replacement clone transcripts cannot override a
named voice. Clone requests remain excluded from the shared output cache.

These identity endpoints and voice IDs are Open Speech extensions. They do not implement
OpenAI's custom-voice creation or consent APIs.

## Existing installations

Startup adds tables and two profile columns to the existing Studio database. The one-time
migration links legacy profiles to named voices using their existing name and exact model.
Profile IDs, defaults, original blends, reference names, effects, speed, and timestamps remain
intact. Recordings are neither rewritten nor copied. Conflicting case variants with different
recipes remain standalone rather than overwriting a realization.

Default recipes and `TTS_VOICES_CONFIG` YAML templates are imported once into the same preset
catalog. Mapping-style YAML from `voice-presets.example.yml` and list-style YAML both work.
Existing recipes with matching names take precedence. Deleting an imported preset does not
recreate it at the next startup. Later YAML edits do not overwrite an imported user preset.
`/api/voice-presets` and `/api/profiles` now list the same persistent recipes.

Older clients can keep using provider voice strings and profile fields. Updating a linked
profile's voice/model/reference without an explicit identity ID detaches it into a standalone
recipe; changing only reading settings preserves the link.

## Speech formats and streaming

The [official speech API reference](https://developers.openai.com/api/reference/python/resources/audio/subresources/speech/methods/create)
defines `model`, `input`, `voice`, `instructions`, `response_format`, `speed`, and `stream_format`.
The [text-to-speech guide](https://developers.openai.com/api/docs/guides/text-to-speech) describes
the SDK streaming reader and raw 24 kHz, signed 16-bit little-endian PCM contract.

Open Speech accepts string voices and objects with `id`. A named-voice UUID in an object resolves
its explicit realization for the requested model. `stream_format: audio` streams audio bytes;
`stream_format: sse` emits base64 `speech.audio.delta` events followed by `speech.audio.done`.
The existing `?stream=true` extension remains supported. Speech PCM is resampled to 24 kHz;
provider-native rates remain available for WAV and internal audio processing.

Local providers do not expose OpenAI token usage counts, so the SSE terminal event omits usage.
