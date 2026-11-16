# Web app

Open **https://localhost:8100/web** and accept the self-signed certificate the first time. The web app works in
desktop and mobile browsers and installs as an app ([Install as an app](#iphone-lock-screen-and-home-screen)). It is
plain static files and native JavaScript modules: no build step, nothing from a CDN, and it runs under a
strict Content Security Policy. It talks only to the server it came from, so it works on a LAN without
internet access.

## Navigation

- **Wide screens** (960 px and up): a rail on the left groups the sections as **Listen** (Reader, Speak),
  **Capture** (Transcribe, Talk), **Create** (Voices, Studio) and **Manage** (History, Models, Settings).
- **Narrow screens**: a bottom bar holds Reader, Speak, Transcribe, Talk and **More**; More opens Voices, Studio,
  History, Models and Settings.
- The **status chip** (top bar on narrow screens, foot of the rail on wide ones) shows the speech model in use, a
  download or load in progress, and connection or API key problems. Select it for details.
- The first visit opens the Reader; after that the app opens on the section you used last.
- Every section has its own address, so Back, bookmarks and links work: `#reader`, `#reader/<document>`,
  `#speak`, `#speak/live`, `#transcribe`, `#talk`, `#voices`, `#voices/recordings`,
  `#voices/clone`, `#voices/design`, `#studio/conversations`, `#studio/composer`, `#history`, `#models/tts`,
  `#models/stt` and `#settings`. `#speak/live` opens Generate, with the address `#speak`, while Speak › Live is hidden (Settings
  › Developer). `#voices/named` and `#voices/presets` open `#voices`, and
  `#voicelab`, `#tts` and `#stt` open the matching section.
- Press `?` for the keyboard shortcuts. `g` then `r`, `s`, `t`, `a`, `v`, `d`, `h`, `m` or `,` goes to a section;
  Space or `k` plays and pauses. Settings › Shortcuts turns the single-key shortcuts off.

## Sections

- **Reader**: the **Library** of documents (EPUB, text, Markdown, HTML, or pasted text) and the reading view,
  which highlights the sentence being read. The player has play and pause, 15 s back and 30 s forward,
  previous and next paragraph, a chapter-wide scrubber, speeds from 0.5× to 3.5×, a sleep timer, the
  reading voice, and **Reading appearance** (text size, typeface, spacing, width, paper and highlight
  style). Start over and Mark as finished are in a document's menu. **Ask AI** in the reading view's top bar
  sends this sentence (the one being read, or the last one read), this paragraph, the selected paragraphs, the
  current chapter or the whole document to the AI, with the document's name and chapter and an optional question
  of your own. In the voice sheet, a model with no voices of its own and no named voice for it says so, with a
  link to Design a voice.
- **Speak**: type or open a `.txt` file, then **Generate** a take. The inspector sets the model, the voice
  (a picker with search, favorites and previews), a Kokoro blend (a visual editor), a reference recording
  for cloning models, speed, format, effects and instructions. Each take shows its first-audio time and
  real-time factor, plays and downloads, and its request can be copied as curl, Python or JavaScript. A take
  made from a description (a model that designs voices, with no voice chosen) has **Save as voice…** in its menu,
  which saves it as a designed named voice (at most 30 s); **Save named voice…** saves the newest such take.
  **Ask AI**, after Open .txt, sends the editor's text to the AI and speaks the answer. **Live**, a streaming
  test for developers, reads text as it arrives, the way an AI app streams its reply (see
  [Live Reader](API.md#live-reader-ws-v1audiospeechstream)); it is hidden until Settings › Developer › Show Speak ›
  Live turns it on.
- **Transcribe**: upload a file (with progress and Cancel) or record from the microphone. Transcripts list
  their segments with timestamps and export as TXT, SRT, VTT or JSON. **Ask AI**, beside Copy, sends the
  transcript to the AI.
- **Talk**: talk to the [AI assistant](AI-ASSISTANT.md) and hear it answer in a voice, over `/v1/realtime`.
  **Start talking** opens the microphone; speak, then pause. Your words, the waiting phrases, the thinking
  (dimmed and folded) and the answer appear in the conversation while the answer is read aloud. An answer
  that an error or the token limit cut off keeps what was said, with a note under it ("The answer was cut
  off: the AI reached its token limit."). **Type instead** sends a typed question (Enter sends, Shift+Enter starts a line, up to 8,000 characters), with the
  microphone on or off. Talking over an answer stops it; with speakers rather than headphones, turn off
  **Interrupt by talking**, so the answer's own sound doesn't stop it; with it off, the microphone pauses while the
  AI answers and listens again once the answer has played (Stop reply listens again at once). **Stop reply**, **Resume sound** and
  **New conversation** sit beside the main button. **Delete chat**, beside the conversation's heading, clears the
  chat: while Talk runs it starts a new conversation, so the AI forgets the old one; after Stop it only clears the
  screen, with Undo for 5 seconds. The AI preset, the reply voice ("Same as Speak" by default)
  and Interrupt by talking are remembered in this browser; one that no longer exists falls back to the
  default. The AI remembers the conversation while Talk stays connected: after a dropped connection Talk
  reconnects by itself and says the conversation started over. With saved assistants, **Assistant** shows which
  one answers (the preset's own, else the one in use); choosing another puts it in use everywhere, and the next
  turn uses it without starting the conversation over. With the assistant off, Talk says to set the
  AI server's address in Settings › AI assistant. To turn speech into text without the AI, use Transcribe.
- **Voices**: named voices and presets in one list, and recordings. A named voice is who speaks (one version per
  model); a preset is how it reads (speed, format, effects, instructions). Each named voice is a card with its
  versions and, under them, its presets and **New preset**; presets that use a plain voice, a blend or a recording
  are under **Presets without a named voice**. **Edit in Speak** on a version and **Edit** on a preset open it in
  Speak with an editing bar: **Update voice** or **Update preset** saves it, **Cancel** puts Speak back.
  **Clone a voice** is a guided three-step flow (Capture, Verify words, Test & save) with quality checks for
  duration, clipping, loudness, background noise and sample rate; it saves the recording (listed in Recordings)
  and, with **Also save as a named voice** (on by default), a named voice of the same name whose version for the
  Test with model clones it. **Design a voice** (Describe, Listen, Name & save) makes a named voice from a
  description with a model that can design one (Breeze TTS 2, Qwen3 1.7B VoiceDesign): choose the model and its
  options (Breeze: Guidance and Variation), generate an editable sample (at most 30 s) and listen; new words speak
  the same voice again, **Generate again** makes a new one. **Speak with** chooses the cloning models that get a
  version of it. A designed voice's card shows "Designed" with its model and description, **Play sample**, **Edit
  description** (the wizard filled in, ending with **Replace** or **Save as new voice**) and **Use in Speak**;
  deleting it deletes its sample, or moves it to Recordings when presets or other voices still use it.
  **Default in Speak** on a preset makes Speak open with it.
- **Studio**: conversations made of turns, rendered to WAV or MP3, and the **Composer**, which mixes tracks
  from rendered conversations and earlier mixes. A conversation's More actions menu has **Ask AI about the
  script…**.
- **History**: every speech and transcription entry the server kept, with search, the All, Speech and
  Transcripts filters, Load more, **Open in Speak** (the text with the settings it was spoken with), Copy
  text, Delete and Clear all. An entry's More actions menu has **Ask AI…**.
- **Models**: the TTS and STT models with their state, the server's configured STT default marked, and Load,
  Unload, Download and Delete files. An
  isolated worker that doesn't answer reads "Worker offline", not "Not installed". One that answers but
  can't run, such as a worker whose torch build has no code for the GPU, reads "Worker can't run" with the
  worker's reason.
- **Settings**: theme (System, Light or Dark) and Reduce motion; playback defaults (reading speed, skip
  intervals, what the previous and next buttons and the lock-screen track buttons move by, the sleep-timer
  fade, keeping the screen on); the API key; server status; the **AI assistant** (the saved assistants, its
  connection, AI presets with their Ask AI fields, Preview and Test; see [Settings › AI assistant](AI-ASSISTANT.md#settings--ai-assistant));
  history; shortcuts; Developer (Show Speak › Live).
- **Settings › AI assistant › Assistants**: each saved assistant is a named AI server connection, with its host,
  its model and **In use** on the one that answers. **Use** puts it in use, **Edit** opens it in the Connection
  form (with its **Name**), **Check** checks it as saved, **Delete** asks first. **Add assistant** opens the same
  form for a new one; what you leave as it is follows `.env`, and Check works before saving. With none saved, the
  form shows the `.env` connection and Save keeps it as the first assistant. A locked connection shows the `.env`
  connection read-only, with no list. An AI preset's **Assistant** is "The one in use" or a saved assistant.

**Ask AI** shows only while the AI assistant is on: in Transcribe (the transcript), the Reader (this sentence,
this paragraph, the selected paragraphs, the current chapter or the whole document; it starts with this sentence
once something has been read, or with a selection you just made), Speak (the editor's text, spoken by default), History
(an entry's text) and Studio (a conversation's script, a line per turn). It opens one sheet: what is sent and
its size, **Your question** (optional: quick picks "Explain this", "Summarize", "Define the hard words", or your own
words, up to 2,000 characters), **Ask it to** (the presets offered for that kind of input, by their action names), and **Then**:
Show the answer, Speak it in a voice, Save it as a document. With saved assistants, a line under **Ask it to** says which one
answers with that preset. A line says which server it is sent to; for a Reader part, another says that the document's name and
chapter go with it. The answer comes back as text, with its thinking folded, Play, Copy, Save as a document, Open in the Reader and
Send again; the AI's errors are said in plain words, with Try again. A whole book or another long input runs
as a job, with progress and Cancel, and its answers are saved in the Reader as a new document; a long text or
transcript is kept in the Reader too, where you can delete it. Closing the sheet leaves a job running, and a
notice says when it has finished (with Open) or failed. The sheet remembers, in this browser, the preset you last
chose for each kind of input, your answer choices, the voice and your last three questions; a preset or voice that is gone falls back to the
default. Each preset's action name, the inputs it is offered for and its first answer choice are set in Settings › AI
assistant. The sheet uses [`POST /api/assistant/ask`](AI-ASSISTANT.md#ask-ai-send-anything-to-the-ai).

The UI uses one name per idea:

| Term | Meaning |
|---|---|
| Document, Library | A book, article or pasted text in the Reader, and the list of them |
| Voice | One provider voice, such as `af_heart` |
| Blend | A weighted Kokoro mix, shown as `af_bella(2)+af_sky(1)` |
| Named voice, version | A saved voice identity, and its setup for one model ([Named voices](VOICE-IDENTITIES.md)) |
| Recording, reference recording | A clip and its transcript for cloning models |
| Preset, default preset | Saved delivery settings (`/api/profiles`) |
| Take | One result of Generate |
| Live | Speak's streaming test for developers (`/v1/audio/speech/stream`), shown by Settings › Developer |
| Talk | The view for talking to the AI assistant, by voice or typed (`/v1/realtime`) |
| Ask AI | Sending a transcript, a document, a selection or text to the AI with a preset (`POST /api/assistant/ask`) |
| Segment, interim text | A committed piece of a transcript, and the words still being recognized |
| Conversation, turn, render | A Studio script, its lines, and its audio |
| Mix, track | A Composer output, and one of its sources |

## Speech models, blends and presets

Speak opens with the default preset when one is set. Otherwise it opens on the model last picked in this
browser, or on the configured default (`TTS_MODEL`, normally Kokoro); an optional model that merely happens
to be loaded never takes its place. **Restore Kokoro** (named after the default) selects the default again
and loads it if needed.
When the chosen model is loaded, Generate sends its request at once; otherwise it loads the model first.
Loading from Speak or Models is an explicit load, which unloads the other loaded TTS models except the pinned
`TTS_MODEL` (see [Model residency](MODELS.md#model-residency)), so Speak asks before it switches.

A cloning model such as Chatterbox has no built-in voices: it speaks in the voice of a reference recording.
For such a model every voice picker says so, and named voices that have no version for that model are shown
last, disabled. Speak's picker also lists your recordings first, with **Clone a voice…** to make one. See
[Cloning models](CLONING-MODELS.md).

The blend editor only composes the voice string. The API contract is the plain `voice` field: the
UI builds `af_bella(2)+af_sky(1)`, and the API only sees `voice="af_bella(2)+af_sky(1)"`. Paste a blend
string into the editor to edit it, or copy it out.

**Save preset…** in Speak stores the model, voice or blend, speed, format, effects, reference recording and
instructions in the server-side Studio database, so presets survive browser storage clearing and container
replacement when the `/home/openspeech/data` volume is kept. Selecting a preset chip in Speak applies it.
In **Voices**, a preset's **Use in Speak** applies it, **Edit** opens it in Speak where **Update preset** saves
Speak's setup into it (**Save as new…** keeps it as another preset), and it can be set as the default, renamed or
deleted. A preset that can't be used with the current models says why and changes nothing. `TTS_VOICES_CONFIG` is a separate,
administrator-supplied YAML preset source.

## API key and WebSocket tickets

With `OS_API_KEY` set, enter the key in **Settings › Connection** (Save, Test and Clear are next to it). The
key stays in this browser's storage, and every request the page sends to its own server carries it as
`Authorization: Bearer KEY`. A browser cannot set headers on a WebSocket, so Live, live transcription and Talk
first ask `POST /api/auth/ws-ticket` for a one-time [ticket](API.md#authentication), a fresh one for every
connection and reconnect. The key never appears in a URL, in copied code (which reads
`$OPEN_SPEECH_API_KEY` or the environment instead) or anywhere on the page but the key field. A key the server refused is not sent
again until you enter another one, so a typo doesn't trip the [failed-auth lockout](SECURITY.md#api-key).

## iPhone: lock screen and Home Screen

- The Reader keeps playing on a locked iPhone. The lock screen shows the chapter as the track and the
  document as the album, with a chapter-wide scrubber, skip buttons, and previous and next track. Settings ›
  Playback › **Lock-screen previous and next** chooses whether those move by paragraph or by chapter.
- Safari needs one tap on Play before a page may play audio; after a reload the Reader asks for it.
- To install the app, open `/web` in Safari, tap **Share**, then **Add to Home Screen**. It opens full screen
  with its own icon. A Home Screen app may refuse a certificate the phone doesn't trust, so give Open
  Speech a trusted certificate first ([TLS certificates](SECURITY.md#tls-certificates)). The microphone (Transcribe,
  Talk, voice cloning) also needs HTTPS, or `localhost`.
- **Talk** with the microphone open may play the answer through the earpiece on iOS; headphones avoid it.
- **Keep the screen on while reading** (Settings › Playback) holds the screen awake where the browser
  supports it.

## Playback and other behavior

- The Reader keeps playing when you leave it. A **mini-player** on every other screen gives the controls back;
  its Stop button ends playback. Finishing a document ends the session, so the mini-player goes too.
- Starting any other audio stops Live rather than pausing it, because a paused Live session would hold up the
  Reader's rendering. Recording from the microphone pauses whatever is playing.
- History entries have no player, because the server keeps no audio for them.
- Speak streams MP3 takes without effects; other formats, and takes with effects, play when they are ready.
- The Live mode (Book, Responsive or Each word) is remembered across reloads.
- Saving a cloned recording opens Voices › Recordings with the new card; its **Use in Speak** button selects it
  in Speak.
- In Studio, turns move with up and down buttons. The Composer lists rendered conversations and earlier mixes as
  sources; recordings aren't listed, because the server exposes no file path for them.
