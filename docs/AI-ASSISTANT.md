# Talk to Your Own AI, or Give It a Book

This guide points Open Speech at an AI server you run, tries it in Settings, talks to it in the web app's Talk
view or through `examples/voice_chat.py`, asks it about a text with Ask AI, and sends it a book. The reference
for every setting, event and route is the README's [AI assistant](../README.md#ai-assistant) section.

Here Open Speech is the AI assistant's front end: it calls your AI server itself. If you have an AI agent of
your own that does its thinking and only needs Open Speech to hear and speak for it, see
[AI-AGENTS.md](AI-AGENTS.md) instead.

You need an OpenAI-compatible server that answers `POST /v1/chat/completions`, such as the llama.cpp
server, vLLM, SGLang or Ollama. The example below is a llama.cpp-style server on your network with the model
`your-model`. Your words and your books go to that server: a cloud API's URL sends them out of
your network.

## 1. Point it at the AI server

Either way works; Settings needs no restart.

- **Settings › AI assistant › Connection** in the web UI: enter the server URL, ending in `/v1`, and the
  model. Turn on **Send thinking as template variables** for a llama.cpp-style server. Press **Check**:
  it asks the server for its models without generating anything, and fills the model list. Then
  **Save connection**. The assistant is on at once.
- **`.env`**, then restart Open Speech:

  ```bash
  OS_ASSISTANT_URL=http://ai-server.lan:8080/v1
  OS_ASSISTANT_MODEL=your-model
  OS_ASSISTANT_TEMPLATE_KWARGS=true
  OS_ASSISTANT_REASONING_EFFORT=low
  ```

`OS_ASSISTANT_REASONING_EFFORT=low` keeps the thinking short, so voice replies come quickly. Some chat
templates think at length unless the effort arrives as a template variable, which is what
`OS_ASSISTANT_TEMPLATE_KWARGS=true` does. Leave that setting off for OpenAI's own API, which refuses the
field.

Voice turns stream the reply (`OS_ASSISTANT_STREAM=true`, the default): Open Speech starts speaking on the
first sentence while the AI is still writing the rest. A server that can't stream still works, because its
plain reply is read in one piece; `OS_ASSISTANT_STREAM=false` stops asking it to. If a proxy such as nginx or
Open WebUI sits between Open Speech and the AI server, it must pass the stream through unbuffered (for nginx,
`proxy_buffering off;`), or the whole answer arrives at the end.

In Docker, a server on the Docker host is `http://host.docker.internal:PORT/v1` (on Linux, add
`extra_hosts: ["host.docker.internal:host-gateway"]` to the `open-speech` service). A machine on your network
is reached by its name or IP address. If the server has a key, set it in the Connection area or in
`OS_ASSISTANT_API_KEY` (printable ASCII; a trailing line break is stripped; in `.env`, single-quote a key that
holds a `$`, as in `OS_ASSISTANT_API_KEY='Xk9$mP2q'`, which Compose and a source install's `uv run --env-file`
both read as written, while `$$` works under Compose only). It is sent only to that server,
and only with the certificate check it was set with: turning **Check TLS certificate** off for an https server
needs the key again in the same save.

On a network other people use, set `OS_API_KEY`, because anyone who can use the API can change the
connection and the presets. `OS_ASSISTANT_CONNECTION_LOCKED=true` keeps the connection in `.env` only.

To check it from a shell instead (add `-H "Authorization: Bearer $OS_API_KEY"` when the server has a key):

```bash
curl -sk https://localhost:8100/api/assistant
curl -sk -X POST https://localhost:8100/api/assistant/connection/check \
  -H "Content-Type: application/json" -d '{}'
```

The first says `"enabled": true` and the AI server's host. The second lists the server's models.

## 2. Try a preset

Settings › AI assistant › **Presets** shows the built-in `voice` and `book` presets. With `voice`
selected, type a message under **Test** and press Test. The reply, the thinking (folded) and the time it
took appear below; nothing is spoken or saved. Test uses the preset as it is on screen, so you can try a
change before you save it.

Things to tune:

- **Prompts:** the system prompt says how to answer; keep voice answers short.
- **Thinking:** the mode, the effort and the **Thinking budget (tokens)**, sent as `reasoning_budget_tokens`. A server
  that reads the field (llama.cpp-style) wraps up the thinking when it reaches the budget, and the model still
  writes a whole answer.
  Other servers ignore or refuse the field; Test shows whether yours takes it. The thinking is never spoken.
- **Limits:** **Max tokens** cuts the answer off mid-sentence without telling the model, and the thinking
  counts towards it. For short answers, say so in the system prompt ("Answer in 3 sentences.") and keep the
  limit as a safety net, or leave it at 0.
- **While waiting:** the phrases said while the AI works ("Hmm.", "OK, just a second."), and when. Press
  the play button beside a phrase to hear it in the voice chosen in Speak. Respell one that sounds odd
  ("Hmmm.", "Mm.").
- **Errors:** the sentence spoken for each kind of failure.

Press **Save**. **New** or **Duplicate** makes your own preset, such as "Short answers"; a client picks it
by its id.

## 3. Talk to it

### In the web app

Open **Talk** (in the rail under Capture, or More on a narrow screen; `g` then `a`). Press **Start talking**,
allow the microphone, and speak, then pause. The conversation shows your words, a waiting phrase when the AI
takes longer than 1.2 seconds, the thinking (dimmed and folded) and the answer, which is read aloud in the
reply voice a sentence at a time, starting while the AI is still writing. An answer that an error or the token
limit cut off keeps what was said, with a note under it ("The answer was cut off: the AI reached its token
limit."). Or type under **Type instead** and press Enter: that works with the microphone off.

- **AI preset** and **Reply voice** ("Same as Speak" by default) are under Preset and voice; the browser
  remembers them.
- **Talking over an answer** stops it and asks your new words; "mm-hmm" lets it go on. **Stop reply** stops it
  without a word. With speakers instead of headphones, turn off **Interrupt by talking**, so the answer's own
  sound doesn't stop it.
- **New conversation** makes the AI forget the one above. After a dropped connection Talk reconnects by itself
  and says the conversation started over.
- The microphone needs HTTPS, or `localhost`.

### From a terminal

Install the example dependencies (`sounddevice` needs PortAudio, as in
[LOCAL-LISTENER.md](LOCAL-LISTENER.md)). Keep the extras of the README's
[Installation (from source)](../README.md#installation-from-source) (for a CUDA install, `--extra cuda
--no-install-package onnxruntime` in place of `--extra cpu`, and `uv run --no-sync` in place of `uv run --frozen`,
as explained there): uv's sync is exact and removes what the extras it is given don't need, so leaving out `--extra all`
uninstalls Kokoro, Piper and Pocket-TTS.

```bash
uv sync --frozen --extra cpu --extra all --extra dev --extra examples
```

Then, **with headphones on**:

```bash
uv run --frozen python examples/voice_chat.py --url https://localhost:8100 --insecure --voice af_heart
```

Speak, then pause. You should see:

- `[you] ...`: what you said;
- `[waiting] Hmm.`: a phrase, when the AI takes longer than 1.2 seconds;
- `[thinking] ...`: the AI's thinking, dimmed, when the server sends it;
- `[ai] ...`: the reply, one line that grows a sentence at a time as it is spoken;
- `[cut off: assistant_reply_cut_off]`: the token limit stopped the reply after some of it was spoken (another
  code names another error).

Talk over the reply to stop it and ask something else; "mm-hmm" lets it go on. Useful options:

```bash
uv run --frozen python examples/voice_chat.py --preset short-answers
uv run --frozen python examples/voice_chat.py --voice "af_bella(2)+af_sky(1)"
uv run --frozen python examples/voice_chat.py --no-interrupt      # speakers, no headphones
uv run --frozen python examples/voice_chat.py --api-key "$OS_API_KEY"
uv run --frozen python examples/voice_chat.py --list-devices
uv run --frozen python examples/voice_chat.py --device 2 --output-device 4
```

Without headphones, the microphone hears the reply and the reply interrupts itself: `--no-interrupt`
stops that, but then you can't talk over a reply. The conversation lasts as long as the connection; run the
script again to start fresh.

A client of your own can also type: `conversation.item.create` with a user `input_text` message, then a
`response.create` with no text (README, [Talking to it](../README.md#talking-to-it), "Typed questions").

## 4. Ask it about a text, a transcript or a document

While the assistant is on, **Ask AI** appears in Transcribe (the transcript), the Reader (this sentence,
this paragraph, the selected paragraphs, the current chapter or the whole document), Speak (the editor's text), History (an entry's More
actions) and Studio (a conversation's script). Pick what it should do under **Ask it to**, then what to do with
the answer under **Then**: show it, speak it in a voice, or save it as a Reader document. Under what is sent, you can add a question of your own,
such as "Explain this" (the sheet offers a few, and your last three again). A part of a Reader document goes to the
AI with the document's name and chapter. A long input, such
as a whole book, runs as a job with progress and Cancel, and its answers are saved in the Reader. If you close
the sheet while it runs, a notice says when it has finished or failed. The sheet remembers your last choices and
questions in this browser.

Settings › AI assistant › Presets › **Ask AI** sets how each preset is offered there: its action name ("Answer"
for `voice`, "Rewrite for listening" for `book`), the inputs it is offered for, and its first answer choice.

The same from a shell, here a typed question answered aloud in `af_heart`:

```bash
curl -sk -X POST https://localhost:8100/api/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{"input": {"kind": "text", "text": "Why is the sky blue?"}, "output": {"speech": {"voice": "af_heart", "format": "wav"}}}'
```

The answer has the `reply` as text, its `thinking`, and `audio.base64`, the spoken reply as a WAV file.
`{"kind": "transcript", "history_id": "ID"}` asks about a transcript in History, and `{"kind": "document",
"document_id": "ID", "chapters": [0, 0]}` about a Reader document's first chapter, with the `book` preset by
default. An input longer than one piece of the preset's `chunk_chars` answers `202` with a job instead, which
you follow as in step 5.

Add a question of your own with `"question"`, and ask about one sentence with `"block"` and `"sentence"` (its
index in that block, from 0, as the Reader highlights it), here the second sentence of block 4 with the voice
preset:

```bash
curl -sk -X POST https://localhost:8100/api/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{"preset": "voice", "input": {"kind": "document", "document_id": "ID", "block": 4, "sentence": 1}, "question": "Explain this to me."}'
```

The AI gets the question, then a line such as `From "Moby-Dick", chapter "Loomings":`, then the sentence, each
after a blank line. Every document input names its document (and its chapter, when it has several), unless the
preset's template already uses `{title}` or `{chapter}`.

## 5. Send it a book

Import an EPUB or text file in the Reader's Library (or with `POST /v1/reader/documents`), and note its
id. Then start a job with the `book` preset, or your own:

```bash
curl -sk -X POST https://localhost:8100/v1/reader/documents/DOCUMENT_ID/assistant \
  -H "Content-Type: application/json" \
  -d '{"preset": "book", "overrides": {"user_template": "Retell this for a ten-year-old:\n\n{text}"}, "title": "For kids"}'
```

It answers `202` with the job's `id`. Follow it:

```bash
curl -sk https://localhost:8100/v1/reader/assistant-jobs/JOB_ID
```

Each chapter shows `pieces_done` of `pieces`, and its `text` once it's done. When the job is `done`,
`result_document_id` names a new document, "<title>: For kids", in the Reader's Library: play it like any
other, in any voice. `"chapters": [0, 0]` sends only the first chapter, which is a quick way to try a
prompt.

The answers are saved as they come, so a restart loses none: a job stopped by a restart ends `failed`
with `interrupted`, and starting the same job again sends only the missing pieces.
`DELETE /v1/reader/assistant-jobs/JOB_ID` cancels a job.

## Expected behavior

Pass:

- Settings › AI assistant says On, with your server's host and model, and Test answers.
- In Talk, your words appear, then the answer, which you hear; a slow answer gets a phrase first. The voice
  starts on the answer's first sentence, before the AI has finished writing it all.
- A voice turn in the script prints `[you]`, then one `[ai]` line that grows a sentence at a time as it is
  spoken, and you hear the reply.
- The server log has one `Realtime turn of session …` line per voice turn, with its timings in numbers
  (README, [What to expect for speed](../README.md#what-to-expect-for-speed)).
- Ask AI shows the answer, and plays it when you asked for speech.
- A book job reaches `done`, and its document plays in the Reader.

Fail:

- Test or a voice turn says "I couldn't reach the AI server." (`assistant_unreachable`): the URL is wrong,
  or the container can't reach that address (see the Docker note above). Check in the Connection area
  says the same.
- "The AI took too long to answer." (`assistant_timeout`): raise the preset's timeout, or lower the
  thinking.
- "The answer was cut off before it started." (`assistant_reply_cut_off`): the token limit stopped the
  model while it was still thinking; raise it, or lower the thinking (a lower effort, or a thinking budget
  well under the limit).
- "The AI server returned an error." (`assistant_http_error`): Test shows its HTTP status. A `401` means the
  key is missing or wrong; a server that refuses a field may need `max_completion_tokens`, or
  `OS_ASSISTANT_TEMPLATE_KWARGS` off.
- The voice only starts once the AI has finished, and the turn's timing line shows `ai_first_token_ms` close
  to `ai_done_ms`: a proxy between Open Speech and the AI server buffers the stream (README, Troubleshooting,
  [The AI assistant](../README.md#the-ai-assistant)). With `streamed=0` the server doesn't stream at all.
- An answer stops in the middle and says nothing more: the token limit (or a dropped connection) cut it off
  after its first sentences; the response is `incomplete` with the code. Raise the token limit, or ask for
  shorter answers in the prompt.
- Talk says "The AI assistant is off", or Ask AI doesn't appear: no URL is set or saved.
- The script prints "The server's AI assistant is off": no URL is set or saved, or TTS is off.
- The script stops with `[refused] close code 4001: Invalid or missing API key`: the server has `OS_API_KEY` set
  and the key is missing or wrong (pass `--api-key`). `close code 4004` means the server has
  `OS_REALTIME_ENABLED=false`.
