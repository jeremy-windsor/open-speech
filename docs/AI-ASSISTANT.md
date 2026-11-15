# Talk to Your Own AI, or Give It a Book

Open Speech can send what you say or type, a transcript, or a Reader document to your own AI server through
its OpenAI-compatible Chat Completions API (`POST {url}/chat/completions`), and speak the reply in any voice
or save it as a new Reader document. A voice turn streams the reply (`stream: true`) and starts speaking on
its first sentence; Ask AI and book jobs wait for the whole reply (`stream: false`). It is off until you give it
the server's address.

| Way in | How | What comes back |
|---|---|---|
| **Talk** | The web app's [Talk](WEB-APP.md#sections) view, or any `WS /v1/realtime` client, such as `examples/voice_chat.py`: speak, or type a question ([Talking to it](#talking-to-it)) | Short phrases such as "Hmm." while the AI works, then the spoken reply. Talking over it stops it. |
| **Ask AI** | The web app's [Ask AI](WEB-APP.md#sections), or `POST /api/assistant/ask` ([Ask AI](#ask-ai-send-anything-to-the-ai)), with typed text, a transcript, or a Reader document or part of one | The reply as text, and on request spoken (base64 audio) or saved as a Reader document. A long input runs as a book job |
| **A book** | Import an EPUB or text file into the Reader, then start a [book job](#book-jobs) on it | The AI's answer for each piece of the book, saved as a new Reader document you play like any other |

The thinking is never spoken. When the AI can't answer, Open Speech says so out loud.

Here Open Speech is the AI assistant's front end: it calls your AI server itself. If you have an AI agent of
your own that does its thinking and only needs Open Speech to hear and speak for it, leave the assistant off and see
[AI-AGENTS.md](AI-AGENTS.md) instead.

You need an OpenAI-compatible server that answers `POST /v1/chat/completions`, such as the llama.cpp
server, vLLM, SGLang or Ollama. The example below is a llama.cpp-style server on your network with the model
`your-model`. Your words and your books go to that server: a cloud API's URL sends them out of
your network.

- **Guide:** [1. Point it at the AI server](#1-point-it-at-the-ai-server), [2. Try a preset](#2-try-a-preset),
  [3. Talk to it](#3-talk-to-it), [4. Ask it about a text](#4-ask-it-about-a-text-a-transcript-or-a-document),
  [5. Send it a book](#5-send-it-a-book), [Expected behavior](#expected-behavior)
- **Reference:** [Settings, presets and the three layers](#settings-presets-and-the-three-layers),
  [Prompts](#prompts), [Thinking controls](#thinking-controls), [Waiting phrases](#waiting-phrases),
  [Errors, spoken](#errors-spoken), [Talking to it](#talking-to-it), [Book jobs](#book-jobs),
  [Ask AI](#ask-ai-send-anything-to-the-ai), [Settings › AI assistant](#settings--ai-assistant),
  [The connection](#the-connection), [Security and privacy](#security-and-privacy),
  [What to expect for speed](#what-to-expect-for-speed)

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
is reached by its name or IP address, as long as the container can resolve it. `docker-compose.yml` passes every
`OS_ASSISTANT_*` setting through from `.env`. If the server has a key, set it in the Connection area or in
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
[LOCAL-LISTENER.md](LOCAL-LISTENER.md)). Keep the extras of
[Install from source](DEVELOPMENT.md#install-from-source) (for a CUDA install, `--extra cuda
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
`response.create` with no text ([Talking to it](#talking-to-it), "Typed questions").

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
  ([What to expect for speed](#what-to-expect-for-speed)).
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
  to `ai_done_ms`: a proxy between Open Speech and the AI server buffers the stream
  ([Troubleshooting](TROUBLESHOOTING.md#the-ai-assistant)). With `streamed=0` the server doesn't stream at all.
- An answer stops in the middle and says nothing more: the token limit (or a dropped connection) cut it off
  after its first sentences; the response is `incomplete` with the code. Raise the token limit, or ask for
  shorter answers in the prompt.
- Talk says "The AI assistant is off", or Ask AI doesn't appear: no URL is set or saved.
- The script prints "The server's AI assistant is off": no URL is set or saved, or TTS is off.
- The script stops with `[refused] close code 4001: Invalid or missing API key`: the server has `OS_API_KEY` set
  and the key is missing or wrong (pass `--api-key`). `close code 4004` means the server has
  `OS_REALTIME_ENABLED=false`.

## Settings, presets and the three layers

Every field has three layers, with the same names everywhere: `.env` (`OS_ASSISTANT_*`, listed under
[Configuration](CONFIGURATION.md#server-settings)) gives the defaults, an AI preset overrides them, and one
request overrides the preset: a realtime `session.update`'s `assistant`, a book job's `overrides`, or an
`/api/assistant/ask` request's `overrides`. A field left empty falls back to the layer below.

The **connection** is apart: the URL, the API key, the default model, the TLS check, the template switch,
the token-field name, the extra body and Stream replies come from `.env` or from Settings › AI assistant › Connection
([The connection](#the-connection)), never from a preset or a session. A preset's own `model` still wins
over the default model.

Two **AI presets** are built in: `voice`, the default for `/v1/realtime` and for asking about text or a
transcript, and `book`, the default for book jobs and for asking about a document. You can edit both and add
your own, such as "Translate to Spanish", up to 50 in all.

| Field | Used by | Values |
|---|---|---|
| `name` | both | Up to 80 characters. A new preset's `id` is made from it; `voice` and `book` keep theirs |
| `label`, `inputs`, `output` | Ask AI | How the web app's Ask AI offers the preset, never sent to the AI and never overridden by a request: `label` is the action's name (up to 80 characters; default the preset's name, "Answer" for `voice`, "Rewrite for listening" for `book`), `inputs` the kinds of input it suits (`text`, `transcript`, `document`; `[]` is the default: `voice` `text` and `transcript`, `book` `document`, yours all three), and `output` its first answer choice (`text`, `speech` or `document`; default `speech` for `voice`, `document` for `book`, `text` for yours). A `PUT` that leaves one out keeps it, and `null` or an empty value is the default again |
| `model` | both | Up to 200 characters; empty uses the connection's model (`OS_ASSISTANT_MODEL`), and with none set the AI server picks |
| `system_prompt` | both | Up to 8,000 characters; empty uses `OS_ASSISTANT_SYSTEM_PROMPT` or the built-in voice prompt for a voice turn, and the built-in book prompt for a book job |
| `user_template` | both | Wraps your words or a piece of the book; must contain `{text}`; up to 8,000 characters; default `{text}` ([Prompts](#prompts)) |
| `thinking` | both | `default` (sends nothing), `on` or `off` ([Thinking controls](#thinking-controls)) |
| `reasoning_effort` | both | Empty (not sent), `none`, `minimal`, `low`, `medium`, `high` or `xhigh` |
| `thinking_budget` | both | 0 to 1,000,000 tokens of thinking, sent as `reasoning_budget_tokens`; 0 sends none ([Thinking controls](#thinking-controls)) |
| `show_thinking` | both | Send the AI's thinking to the client as text |
| `max_tokens` | both | 0 to 32,768; 0 sends no limit. It cuts the answer off ([Thinking controls](#thinking-controls)) |
| `temperature`, `top_p` | both | 0 to 2 and 0 to 1; empty sends none, so the AI server's own default applies |
| `timeout_s` | both | 5 to 3,600 seconds; empty uses `OS_ASSISTANT_TIMEOUT_S` for a voice turn and `OS_ASSISTANT_JOB_TIMEOUT_S` for a book job |
| `history_turns` | voice | 0 to 100 earlier exchanges sent with each turn |
| `fillers`, `filler_after_ms`, `filler_every_ms`, `filler_max` | voice | [Waiting phrases](#waiting-phrases) |
| `speak_errors`, `error_sentences` | voice | [Errors, spoken](#errors-spoken) |
| `chunk_chars` | book | 1,000 to 200,000: the most book text in one request |

Presets are kept in `assistant-presets.json` next to `OS_STUDIO_DB_PATH`. The file is written only when a
preset is saved, and a built-in you haven't changed isn't in it. A missing or unreadable file means only
the built-ins, with a warning in the log. Each preset in the API carries its effective values plus
`builtin`, `customized` and `inherited` (the fields that come from `.env` or the code). A `PUT` replaces
the editable fields: one left out or empty follows `.env` or the code again (except `label`, `inputs` and
`output`, which a `PUT` that leaves them out keeps), and the read-only fields may be sent back and are
ignored. `DELETE` on a built-in restores its defaults. Refusals: `422`
`validation_error` with the field in `param`, `404` `preset_not_found`, and `409` `preset_exists` or
`too_many_presets`.

## Prompts

The system prompt is the system message, and the user template wraps your words or a piece of the book
into the user message. The built-in prompts are:

- **voice:** "You are a helpful voice assistant. Your replies are read aloud. Answer in short, plain spoken
  sentences. Don't use Markdown, lists, tables, code, links or emoji."
- **book:** "You rework book text for listening, following the user's instructions. Reply only with plain
  prose paragraphs, because your reply is read aloud. Don't use Markdown, lists or headings, and don't
  comment on what you did."

A template must contain `{text}`. Book jobs also fill `{title}`, `{author}` and `{chapter}` (the chapter's
title). The built-in `voice` preset takes only `{text}`; your own presets may use all four, and a voice turn
leaves the other three empty. For example, a book preset with the template
`Summarize this part of "{chapter}":\n\n{text}` sends each piece with that instruction in front of it.
Rendering is one pass of literal replacement: other braces are kept as they are, and the text put in for
`{text}` is never searched for placeholders, so a book that contains `{title}` stays as written. An unknown
placeholder is refused when the preset is saved. In a realtime session, `instructions` replaces the
preset's system prompt.

## Thinking controls

Servers don't agree on how to switch or shorten thinking, so there are four controls. Each is sent only when
set, because some servers refuse fields they don't know.

| Control | What it sends | Who reads it |
|---|---|---|
| `thinking`: `on` or `off` | `"chat_template_kwargs": {"enable_thinking": true}` (or `false`) | Qwen3 and other models that can think or not, on vLLM, SGLang and the llama.cpp server |
| `reasoning_effort` | `"reasoning_effort"`; with `OS_ASSISTANT_TEMPLATE_KWARGS=true` also `chat_template_kwargs.reasoning_effort` | OpenAI's field; gpt-oss and similar reasoning models, and templates that read it |
| `thinking_budget`: 1 to 1,000,000 | `"reasoning_budget_tokens"`, the most tokens of thinking | Servers that read `reasoning_budget_tokens` (llama.cpp-style). Other servers ignore or refuse it; Settings' Test shows whether yours takes it |
| `OS_ASSISTANT_EXTRA_BODY` (connection only) | Whatever JSON object it holds, merged last; its `chat_template_kwargs` are merged key by key. The preset's thinking controls win where both set one: inside `chat_template_kwargs`, and at the top level a preset's `reasoning_effort` and `thinking_budget` over the extra body's `reasoning_effort` and `reasoning_budget_tokens`, which go only when the preset sends none. It can't set `messages` or `stream` | Anything else your server wants, such as Ollama's `think` or `top_k` |

The token limit goes as `max_tokens`, or as `max_completion_tokens` with
`OS_ASSISTANT_MAX_TOKENS_FIELD=max_completion_tokens`, for a server that follows OpenAI's newer API.

**A token limit cuts the answer off; a thinking budget doesn't.** `max_tokens` is a hard cut-off the model
isn't told about: an answer that reaches it stops mid-sentence (`finish_reason: "length"`), and on most
servers the thinking counts towards it. For short answers, ask for them in the prompt ("Answer in 3
sentences."), and keep the limit as a safety net. To shorten the thinking, set `thinking_budget`: when the
thinking reaches it, a server that reads `reasoning_budget_tokens` closes the thinking and the model writes a
complete answer, based on shorter thinking. How such a server treats a budget that doesn't fit inside
`max_tokens` is up to the server. By default neither is sent (`OS_ASSISTANT_THINKING_BUDGET=0`,
`OS_ASSISTANT_MAX_TOKENS=0`).

The thinking comes back in `message.reasoning_content`, `message.reasoning` or `<think>…</think>` at the
start of `content`; all three are read, and the answer is whatever is left. The thinking is never spoken,
and never sent back in the conversation history. With `show_thinking` it is sent to the client as text.
**The token limit includes the thinking** on most servers: a low `max_tokens` can stop the model while it
is still thinking, which is reported as `assistant_reply_cut_off` ("raise the token limit or lower the
thinking"), not as an empty reply.

**A streamed voice turn** (Stream replies, `OS_ASSISTANT_STREAM`, on by default) reads the same places as
they arrive: `delta.reasoning_content` (or `delta.reasoning`) and a leading `<think>…</think>` in
`delta.content` are the thinking, the rest of `content` the answer. The request also carries
`"stream_options": {"include_usage": true}`, so the server's last piece holds the token counts for the
[timing line](TROUBLESHOOTING.md#the-ai-assistant); a server that refuses that with HTTP 400 is asked once more without it,
and not again until the connection is saved or Open Speech restarts. The thinking streams first, and an
answer cut off by `max_tokens` (`finish_reason: "length"`) after some of it was spoken keeps what was said and
says nothing more ([Talking to it](#talking-to-it)). A chat template that opens `<think>` itself, so the reply
has a `</think>` but no `<think>`, can't be told apart from the answer while it streams: turn the server's
reasoning parser on (llama.cpp's `--reasoning-format`, vLLM's `--reasoning-parser`), or turn Stream replies
off.

## Waiting phrases

While a voice turn waits for the AI, Open Speech says short phrases, so you know it heard you.

| Preset field | Default | Meaning |
|---|---|---|
| `fillers` | `["Hmm.", "OK, just a second.", "Let me think about that."]` | Up to 10 phrases of up to 80 characters; an empty list turns them off |
| `filler_after_ms` | `1200` | No phrase if the reply comes sooner (0 to 30,000) |
| `filler_every_ms` | `4000` | The gap before the next phrase while it still waits (0 to 30,000) |
| `filler_max` | `3` | The most phrases in one turn (0 to 10) |

- They apply only to voice turns that go to the AI, not to book jobs or to word-for-word replies.
- They're chosen at random, never the same one twice in a row, and spoken in the session's voice. Each is
  made once per session, ahead of time after `session.update`, so the first one plays at once.
- Each phrase's audio follows an `open_speech.assistant.filler` event with its text. Phrases aren't part of
  the transcript, the history or `response.done`.
- A phrase that has started is finished, and the reply follows straight after it. No phrase starts once
  the reply has arrived. Talking over a phrase works as talking over a reply does.
- Some voices read "Hmm" oddly, for example letter by letter. Try your phrases with Preview in Settings, and
  respell them as they sound ("Hmmm.", "Mm.") or use the pronunciation dictionary.

## Errors, spoken

When the AI can't answer a voice turn, the preset's sentence for the error is spoken, unless
`speak_errors` is off. `error_sentences` maps a code to its sentence (up to 300 characters); a code left out
keeps the default.

| Code | When | Default sentence |
|---|---|---|
| `assistant_unreachable` | Connection refused, DNS or TLS failure | "I couldn't reach the AI server." |
| `assistant_timeout` | No whole reply within `timeout_s` | "The AI took too long to answer." |
| `assistant_http_error` | A status other than 200; only the status is reported, never the server's text | "The AI server returned an error." |
| `assistant_bad_response` | Not JSON, no `choices[0].message`, or a reply over 8 MiB | "The AI didn't send back an answer." |
| `assistant_empty_reply` | No answer left once the thinking is removed | "The AI didn't send back an answer." |
| `assistant_reply_cut_off` | An empty answer with `finish_reason: "length"`, or in a voice turn one cut off before its first sentence was complete (after that, [nothing more is said](#talking-to-it)) | "The answer was cut off before it started." |

## Talking to it

The web app's **Talk** view does all of this for you ([Sections](WEB-APP.md#sections)). A client of your own connects to
`wss://HOST:8100/v1/realtime` (no subprotocol gives a GA session) and sends a `session.update`:

```json
{"type": "session.update", "session": {
  "type": "realtime",
  "instructions": "Answer like a pirate.",
  "assistant": {"preset": "voice", "reasoning_effort": "low", "fillers": ["Mm."]},
  "audio": {
    "input": {"format": {"type": "audio/pcm", "rate": 24000},
              "turn_detection": {"type": "server_vad", "create_response": true}},
    "output": {"voice": "af_bella(2)+af_sky(1)"}}}}
```

- `assistant`: `preset` picks the AI preset (default `voice`). Every other key overrides that preset field
  for this session; an empty one is skipped, and `null` removes the override. `"assistant": null` goes back
  to the voice preset with no overrides. A value that isn't valid refuses the whole update (`error` with
  code `invalid_request`), and the session stays as it was.
- `instructions` is OpenAI's name for the system prompt, and replaces the preset's for this session.
- `session.created` and `session.updated` report both; `assistant` is `null` while the assistant is off,
  and also with `TTS_ENABLED=false`, because nothing could be spoken.
- `turn_detection.create_response` defaults to whether the assistant is on. As in OpenAI's API, `true`
  means "reply after each turn". A turn is one that server VAD ends: a client's own
  `input_audio_buffer.commit` starts no reply, as in OpenAI's API, and its `response.create` asks for one.
  Without the assistant the session reports `false`, even when asked for `true`.

A turn then goes like this:

1. Server VAD hears you stop, and faster-whisper writes down what you said:
   `conversation.item.input_audio_transcription.completed`.
2. Your words, wrapped in the preset's user template, join the conversation. With `create_response`, a
   response starts for a turn server VAD ended: `response.created`.
3. The AI is asked with `stream: true` (Stream replies, on by default). While it thinks,
   `open_speech.assistant.filler` and the phrase's audio, every few seconds, until the first sentence of its
   answer is ready.
4. `open_speech.assistant.thinking` with the thinking as text, once the answer has started, when the server
   sent some and `show_thinking` is on.
5. The answer is cut into sentences as they arrive (the Live Reader's sentence cutting), and each is spoken at
   once: a `response.output_audio_transcript.delta` with that sentence's text, then its audio
   (`response.output_audio.delta`), while the AI is still writing the rest. Legacy (`realtime` subprotocol)
   sessions get `response.audio_transcript.delta` and `response.audio.delta`. The deltas joined are the
   answer's text.
6. `response.output_audio_transcript.done` with the whole answer, and `response.done`.

With Stream replies off (`OS_ASSISTANT_STREAM=false`, or `"stream": false` saved through
[`PUT /api/assistant/connection`](#the-connection)), the AI's whole reply is awaited: the phrases play until it has all arrived, and one
`response.output_audio_transcript.delta` carries the whole text, before its audio. A server that answers a
streamed request with one plain JSON reply is read the same way, so it works either way.

More about a turn:

- **Manual turns.** Every transcript joins the conversation, also with `create_response: false` or manual
  commits; a `response.create` with no text then answers it, as in OpenAI's manual flow. A
  `response.create` that carries text (`instructions` or an `input_text` item) is still spoken word for
  word, with no AI call and no phrases. After a commit of your own, wait for its transcript (a non-blank
  `transcript` in `conversation.item.input_audio_transcription.completed`) before that `response.create`:
  after `input_audio_buffer_commit_empty`, a failed transcription or a blank one, nothing joined the
  conversation, and the AI would be asked about the conversation as it was. Words committed while a reply
  runs don't join it either, so send `response.cancel` first.
- **Typed questions.** Send `conversation.item.create` with `item: {type: "message", role: "user", content:
  [{type: "input_text", text}]}` (up to 8,000 characters; several `input_text` parts are joined with line
  breaks), then a `response.create` with no text. The item is acknowledged with `conversation.item.added` and
  `conversation.item.done` (`conversation.item.created` in legacy sessions), joins the conversation through
  the preset's user template, and is answered as a spoken turn is, waiting phrases, errors, cancel and
  barge-in included. It starts no reply by itself, even with `create_response`, so send the `response.create`
  once the item is acknowledged. `item.id` (1 to 32 characters) is echoed, or the server makes one;
  `previous_item_id` must be left out or `null`. Other items, roles and content are refused with an
  `invalid_request_error` (code `invalid_request`), as are blank text and text with a lone UTF-16 surrogate
  (half an emoji), and longer text gets `string_above_max_length`. A typed message while a reply runs gets
  `response_in_progress`: send `response.cancel` first (events are handled in order, so the three can go back
  to back). Without the assistant it gets `assistant_disabled`, and with `TTS_ENABLED=false` `tts_disabled`.
  A refused item isn't kept, so don't send the `response.create` after a refusal.
- **The reply** is cleaned as the Live Reader cleans text (Markdown removed, links read by their label,
  code blocks replaced with "Code block skipped.") and spoken in segments within the voice model's own
  limit, so an AI reply is never refused for its length.
- **Talking over it.** Real words over the wait or the reply cancel the AI request and the phrases, and
  become the next question; an acknowledgement such as "mm-hmm" lets it continue (see
  [Realtime Audio](API.md#realtime-audio)). `response.cancel` and closing the socket cancel it too, with a
  `response.done` whose status is `cancelled`. Each closes a streamed reply's connection, so the AI server can stop
  generating. The history keeps the answer as far as it was handed to the voice (the sentences spoken or
  being spoken; all of it once it had all arrived); an answer interrupted before its first sentence leaves the
  question unanswered in the history.
- **Errors.** An `AssistantError` before any of the answer was spoken (including a dropped stream, an error
  in it, or the token limit while it still thinks) gives an `error` event (type `server_error`, with the
  code), then the spoken sentence, then a `response.done` with status `failed` and the code in
  `status_details.error.code`. The conversation keeps only your question.
- **Cut off after it started.** An error or the token limit (`finish_reason: "length"`) after some of a
  streamed answer was spoken keeps what was said and says nothing more: not the sentence it was in the middle
  of, and no error sentence. `response.done` then has status `incomplete`, `status_details`
  `{"type": "incomplete", "error": {"type": "server_error", "code": ...}}` (such as `assistant_reply_cut_off`
  or `assistant_unreachable`), and the transcript and the history hold the part that was spoken. There is no
  `error` event.
- **The conversation** sends the last `history_turns` exchanges (the answers only, never the thinking),
  plus any unanswered question. It lasts as long as the connection: a reconnect starts fresh.
- **Voices.** Kokoro voices and blends work as they are, and named voices (`voice:<id>`) are resolved. A
  named voice that doesn't resolve refuses `session.update`; one whose version is a reference recording
  (a cloned voice) can't be used here.
- **Waiting is busy.** The time the AI takes doesn't count toward `OS_REALTIME_IDLE_TIMEOUT_S`. A client
  that only types sends nothing between questions; an empty `{"type": "session.update", "session": {}}`
  changes nothing and keeps the session open, as Talk does after 30 seconds of sending nothing.

To talk through it from a terminal, use `examples/voice_chat.py` (the `examples` extra, with PortAudio):

```bash
python examples/voice_chat.py --url https://HOST:8100 --insecure --voice af_heart --preset voice
```

It streams the microphone as 24 kHz PCM16, plays the reply, and prints your words, the phrases, the
thinking (dimmed) and the reply, on one `[ai]` line that grows a sentence at a time; a reply cut off after
some of it was spoken (a `response.done` with status `incomplete`) adds `[cut off: CODE]`, such as
`[cut off: assistant_reply_cut_off]` for the token limit. **Use headphones:** the script has no echo cancellation, so through
speakers the microphone hears the reply and the reply interrupts itself. Without headphones, add
`--no-interrupt`, which sends `interrupt_response: false`. `--api-key` defaults to the shell's
`OS_API_KEY`, and `--list-devices`, `--device` and `--output-device` choose the audio devices.

## Book jobs

Import the file into the Reader as usual (`POST /v1/reader/documents`: an EPUB, HTML, Markdown or text
file, or pasted text; there is no import from a URL), then start a job on the document:

```bash
curl -sk -X POST https://localhost:8100/v1/reader/documents/DOCUMENT_ID/assistant \
  -H "Content-Type: application/json" \
  -d '{"preset": "book", "overrides": {"user_template": "Summarize this part of \"{chapter}\":\n\n{text}"}, "chapters": [0, 2], "title": "Summary"}'
```

- **The body:** `preset` (default `book`), `overrides` (preset fields for this job; an empty one is
  skipped), `chapters` (`[first, last]`, the Reader's chapter indexes from 0, or `null` for all) and
  `title`. It answers `202` with the job. A bad field gets `422` with `param`; an unknown document `404`.
- **How it runs.** The settings are worked out once and kept in the job, so a later change to the preset
  doesn't change it. Each chapter is cut into pieces of at most `chunk_chars`, at block boundaries (a
  longer block at its sentences), and each piece is sent on its own, with no history. A chapter's opening
  headings aren't sent; the result keeps them as they are, so each chapter still opens with its title.
- **Saved as it goes.** Each answer is saved as soon as it comes back, so a restart loses none. An error is
  retried once; a second one stops the job as `failed` with the code in `error` (and `error_message`),
  keeping the finished pieces. Starting the same job again (the same document, and the same settings that
  reach the AI: model, prompts, thinking, effort, token limit and sampling) sends only what's missing.
- **One job at a time.** Later jobs wait as `queued`. A shutdown, or a crash found at the next start,
  leaves the job `failed` with the code `interrupted`; start it again to finish it.
- **The result** is a new Reader document named "<document title>: <your title, or the preset's name>",
  with one chapter per chapter sent and the answers as paragraphs (split at blank lines). The job's
  `result_document_id` names it. The thinking is kept in the job, not in the document.
- **The job** (`GET /v1/reader/assistant-jobs/{job_id}`) has `id`, `document_id`, `preset`, `title`,
  `chapters_requested`, `blocks_requested` (`[first, last]` for a job that
  [Ask AI](#ask-ai-send-anything-to-the-ai) started over a range of blocks, `null` otherwise), `status`
  (`queued`, `running`, `done`, `failed` or `cancelled`), `error`,
  `error_message`, `result_document_id`, `created`, `updated`, `settings` (the preset fields in effect,
  nothing server-only) and `chapters`: each with `index`, `title`, `status`, `pieces`, `pieces_done`, its
  `text` once done, its `thinking` when shown, and `error`. `GET /v1/reader/assistant-jobs` lists the jobs
  without their `settings` and `chapters`.
- **Canceling and deleting.** `DELETE /v1/reader/assistant-jobs/{job_id}` cancels a queued or running job;
  its finished pieces stay saved. Deleting the source document deletes its jobs and their pieces, and stops
  a running one; the result document stays.
- **Listening before the whole book is done.** A finished chapter's `text` is in the job straight away; any
  client can send it to the [Live Reader](API.md#live-reader-ws-v1audiospeechstream) and hear it at once.

The book-job routes answer `404` `assistant_disabled` while the assistant is off, and `503`
`assistant_not_started` if its client isn't running. The AI server is asked for one book at a time, but a
book job and a voice turn can still wait for each other at a server with one slot.

## Ask AI: send anything to the AI

`POST /api/assistant/ask` sends one input to the AI with a preset and gives the reply back. The web app's Ask AI
uses it ([Sections](WEB-APP.md#sections)).

```bash
curl -sk -X POST https://localhost:8100/api/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{"input": {"kind": "text", "text": "Why is the sky blue?"}, "output": {"speech": {"voice": "af_heart"}}}'
```

- **The input** is typed text (`{"kind": "text", "text": …}`), a transcript (`{"kind": "transcript", "text":
  …}`, or `"history_id"` for a history entry, read on the server), or a Reader document (`{"kind":
  "document", "document_id": …}`: all of it, or `"chapters": [first, last]`, `"blocks": [first, last]`, a
  selection's `"block_start"` and `"block_end"` (whole blocks are sent), or one sentence's `"block"` and
  `"sentence"`). The preset is `book` for a document and `voice` otherwise, and `overrides` takes its fields as
  Test does. Text may be up to `READER_MAX_DOCUMENT_CHARS` characters.
- **One sentence:** `"sentence"` is its index in block `"block"`, from 0, as the Reader reads and highlights the
  block (its `sentences` offsets in `GET /v1/reader/documents/{id}`; a block without any is one sentence). It is
  always answered at once.
- **Your own question:** `"question": "Explain this to me."` (up to 2,000 characters, not blank) goes with any
  input, before its text: the message is the question, a blank line, then the text (or, with a template other
  than `{text}`, the filled-in template). Every piece of a job gets it, and a different question sends every
  piece again.
- **The document's name and chapter** go with a document's text: its message starts with `From "<title>",
  chapter "<chapter>":` and a blank line (after the question, if any). The title is the Reader document's, which
  for an imported file is its file name unless the file gives a title. The chapter is named when the document has
  more than one and it isn't the document's title; each piece of a job names its own chapter (a piece never spans
  two). A preset template that uses `{title}` or `{chapter}` gets no such line, so nothing says it twice. Text and
  transcripts get none.
- **The reply** always comes back as text: `{reply, thinking, finish_reason, elapsed_ms, preset, input: {kind,
  chars}}`, and `input` also has `question_chars` with a question, and `block` and `sentence` for a sentence
  (never their text). `output.text` (`true` or `false`, `true` by default) says whether you want the text; the
  reply comes back as text whatever it says. A value that is not `true` or `false` is refused (`param`
  `output.text`), and so is `false` with neither `speech` nor `document`, which asks for no output (`param`
  `output`). `"output": {"speech": {…}}` also speaks it (`model`, `voice`, `format` and `speed`, by default
  `TTS_MODEL`, `TTS_VOICE`, `mp3` and `1.0`), cleaned as the Live Reader speaks, and returns the audio in base64
  (`audio: {format, content_type, base64, bytes}`): up to `TTS_MAX_INPUT_LENGTH` characters, and a longer reply
  comes back with `audio_error` (`speech_too_long`) instead. The speech settings are checked before the AI is
  asked: TTS off, or a model, voice or speed the TTS router refuses, is answered with `param` `output.speech`. `"output": {"document": {"title": …}}` saves it as a Reader
  document (`document_id`). Nothing else is kept on the server.
- **When the AI fails**, the answer is `502` (`504` for a timeout) with the error code and `elapsed_ms`, and,
  with speech asked and the preset's `speak_errors` on, the preset's sentence for it (`sentence`) and its
  audio.
- **A long input**, more than one piece of the preset's `chunk_chars`, runs as a book job instead: the answer
  is `202` with the `job`, which you follow (`GET /v1/reader/assistant-jobs/{id}`, each chapter's
  `pieces_done` of `pieces`) and cancel (`DELETE`) with the Reader's job routes, and whose answers become a new
  Reader document (`result_document_id`). A job is never spoken here; play its result in the Reader. A long text
  or transcript is kept in the Reader as a document first, so the job can resume and you can listen to it;
  the AI still gets the text exactly as it was sent, in pieces cut at blank lines (a longer paragraph at
  line breaks, then at sentences), each filling only `{text}`.
- **Refusals:** `422` `validation_error` with the field in `param` (every field outside these, `stream: true`,
  which is kept for a streamed reply later, and an input the Reader would need while `READER_ENABLED=false`),
  `404` `document_not_found` or `history_not_found`, and `503` `assistant_not_started`.

Presets also have a `label` (the action's name), `inputs` (the input kinds it suits) and `output` (its default
output) for Ask AI ([the presets](#settings-presets-and-the-three-layers)). The web app shows them; they never
reach the AI, and no request can override them.

## Settings › AI assistant

The web UI's Settings has an **AI assistant** section:

- **Status:** whether it is on, the AI server's host and the default model. Off, it says how to turn it
  on: enter the address under Connection, then Check and Save; when the connection is locked, set
  `OS_ASSISTANT_URL` in `.env` and restart. In the list of connection settings in words (shown when the
  Connection area can't be), "Certificate check: On (default)" means `OS_ASSISTANT_VERIFY_TLS` is unset, so
  the certificate is checked.
- **Connection:** the server URL, the model (a text field, with a list filled by Check), the API key, "Check
  TLS certificate", "Send thinking as template variables", "Stream replies", the token-limit field name and
  the extra body.
  - Each field says where its value comes from: from `.env`, saved here, or the default. A saved one has
    **Use .env**, which puts it back to `.env`.
  - The key is never shown, only "Set" or "Not set", with **Clear key**. It is sent only to the server it
    was saved for, so when the URL moves to another server (another scheme, host or port), the area says
    "Enter the key again for a new server", and saving without a key clears it. A change of path keeps it.
    It also stays with the certificate check: turning **Check TLS certificate** off for an https server
    needs the key again in the same save, which the key field says before you save; saving without it clears
    the key. Turning the check back on keeps the key.
  - **Check** asks the server for its models with the form as it is, unsaved, and generates nothing.
    **Save connection** sends only the fields you changed. **Revert to .env** forgets everything saved here,
    after asking.
  - Saving a URL turns the assistant on at once, with no restart. With
    `OS_ASSISTANT_CONNECTION_LOCKED=true` the area is a read-only list that points at `.env`.
- **Presets:** the picker (the built-ins first), **New**, **Duplicate** (a copy of the preset as it is on
  screen, unsaved changes included), and **Delete** for yours or **Restore defaults** for a built-in, each
  after a confirm. The fields are grouped as Ask AI, Prompts, Thinking, Limits, Sampling, While waiting and
  Errors; the voice preset hides the book's piece size, and the book preset the voice-only groups. **Ask AI**
  holds how the web app's Ask AI offers the preset, none of which reaches the AI: **Action name** (`label`),
  **First answer choice** (`output`) and **Offered for** (`inputs`: text, transcript, document).
  **Thinking** holds the mode, the effort and **Thinking budget (tokens)** (`thinking_budget`); **Limits**
  holds **Max tokens**, whose hint says that it cuts the answer off ([Thinking controls](#thinking-controls)).
  - **Save** sends the name, the preset's own fields and the ones you changed, so a field that comes from
    `.env` and that you didn't touch keeps following `.env`.
  - A refused field is marked with the server's reason. Unsaved changes are marked and can be discarded;
    switching presets asks first, and leaving the page warns.
- **Preview** speaks a phrase or an error sentence in the voice chosen in Speak. It can't use a reference
  recording chosen in Speak; choose a voice there first.
- **Test** sends one message ("Say hello in one sentence." unless you type one) with the preset as it is on
  screen, saved or not, and shows the reply, the thinking (folded), the time taken and the finish reason,
  or the failure in plain words. The reply isn't spoken, and nothing is saved. A field you emptied is tried
  as its `.env` or built-in default, as it would be once saved.

## The connection

`GET /api/assistant/connection` answers `{enabled, locked, url, model, api_key_set, api_key_tls_checked,
verify_tls, template_kwargs, max_tokens_field, extra_body, stream, sources}`; `sources` maps each of the eight
fields to `env`, `saved` or `default`. The key itself is never in an answer, a log, a preset or a job.
`api_key_tls_checked` is read-only: `true` or `false` says whether the key in use was set with the certificate
check on (for the `.env` key, the value of `OS_ASSISTANT_VERIFY_TLS`), and it is `null` when no key is in use. A
`PUT` that sends it back is not refused, and changes nothing.

- **`PUT`** takes any of the eight fields and saves each one sent, so send only what changes: `url` (`""`
  saves "off"), `api_key` (left out keeps the key, `""` clears it; stripped, then printable ASCII of up to
  4,096 characters), `model` (up to 200 characters), `verify_tls`, `template_kwargs` and `stream` (`true` or
  `false`; `stream` is Stream replies, `OS_ASSISTANT_STREAM`),
  `max_tokens_field` (`max_tokens` or `max_completion_tokens`), and `extra_body` (an object, or JSON text, up to
  16,000 characters, nesting objects and lists at most 32 levels deep, with no `NaN` or `Infinity`, which JSON
  doesn't have). `null` for any field forgets its saved value, so it follows `.env` again. It answers the new
  view; a bad field gets `422` with `param`, and nothing is saved.
  Saving rebuilds the shared client: a request already on its way finishes on the old one.
- **`DELETE`** forgets the saved connection, so every field follows `.env` again.
- **`POST /api/assistant/connection/check`** takes the same fields as an unsaved candidate over the
  connection in use (`{}` checks the one in use) and makes one `GET {url}/models`, within 10 seconds. It
  answers `200` `{reachable: true, status, models, listed, elapsed_ms}`: `models` holds only the model ids of
  an OpenAI-style list (`listed`), and another HTTP status gives `models: []`. A server it can't reach gets
  `502` `assistant_unreachable`, or `504` `assistant_timeout`, with `reachable: false`; no URL gets `422`
  with `param` `url`.
- **The key stays with its server.** A key is sent only to the origin (scheme, host and port, with 80 and
  443 filled in) of the URL it was saved for, and `OS_ASSISTANT_API_KEY` only to `OS_ASSISTANT_URL`'s
  origin. A `PUT` that moves the URL to another origin without `api_key` clears the saved key
  (`api_key_set: false`); a change of path, such as `/v1` to `/api/v1`, keeps it. A check of another origin
  sends no key unless its body has `api_key`.
- **The key stays with its certificate check.** Whether the certificate check was on when a key was saved
  is kept beside it (shown as `api_key_tls_checked`). A key saved with the check on isn't sent to an https URL
  once the check
  is off: a `PUT` that turns `verify_tls` off for an https URL without `api_key` clears the saved key
  (`api_key_set: false`), so give the key again in the same `PUT`, and a check with `verify_tls: false` sends
  no stored key. `OS_ASSISTANT_API_KEY` counts as set with `OS_ASSISTANT_VERIFY_TLS`, so
  `OS_ASSISTANT_VERIFY_TLS=false` with a key in `.env` keeps working, but turning the check off in Settings
  withholds the `.env` key from an https URL. Turning the check back on keeps a key, and an http URL, which
  has no certificate, isn't affected.
- **The key's own rule:** it is stripped (so a trailing line break or tab from a secret file isn't sent), and
  must then be printable ASCII of up to 4,096 characters. `OS_ASSISTANT_API_KEY` that breaks it stops startup
  with `OS_ASSISTANT_API_KEY must be printable ASCII of up to 4096 characters`, never quoting the key; a saved
  one gets `422` naming `api_key`. No error text holds the key.
- **Guards, for every URL from `.env` or saved:** `http` or `https` with a valid host and port (a host such
  as `192.168.1.300`, which isn't an address, is refused); no user name or password, query or fragment;
  redirects aren't followed; only a `200`'s body is read, up to 8 MiB; and an error names only the HTTP
  status, never the server's text, or says that the server's answer isn't valid HTTP. `OS_ASSISTANT_URL`
  that breaks one stops startup with a clear message, as does an `OS_ASSISTANT_EXTRA_BODY` that isn't a JSON
  object, nests deeper than 32 levels or holds `NaN` or `Infinity`. A startup error names the setting and the
  reason, never its value.
- **Locked.** With `OS_ASSISTANT_CONNECTION_LOCKED=true` the saved file is ignored, `PUT` and `DELETE`
  answer `403` `connection_locked`, and so does a check whose body sets any field.

`GET /api/assistant` reports `enabled`, the AI server's `host` (no path, query or key), the default
`model`, `server_settings` (for each connection setting, whether it differs from its default, never its
value), `locked`, and `sources` keyed by the `.env` names. `POST /api/assistant/test` takes
`{preset, text?, overrides?}` and answers `{reply, thinking, finish_reason, elapsed_ms}`, or `502` (`504`
for a timeout) with the error code and `elapsed_ms`. Its `overrides` are the preset as it is on screen: a
field sent as `null` or empty is tried as its `.env` or built-in default, and a field left out keeps the
saved value.

## Security and privacy

- **Your words and your books go to the AI server.** With a server on your network they stay there; a
  cloud API's URL sends them out.
- **Anyone who can use the API can change the presets and the connection**, and `/check` makes the server
  call any http or https address its network can reach (answering only the status and model ids). Set
  `OS_API_KEY` when others can reach the server, or `OS_ASSISTANT_CONNECTION_LOCKED=true` to keep the
  connection in `.env`.
- **The saved key** is kept in plain text in `assistant-connection.json` next to `OS_STUDIO_DB_PATH` (the
  `./data` volume under Compose), and is never returned or logged.
- **What is kept.** A voice conversation lasts as long as its connection. Ask AI keeps nothing on the server
  unless you save the answer, except for a long input: it runs as a book job, a long text or transcript is
  kept in the Reader as a document, and the answers are saved as a new one, both of which you can delete.
- **The AI's reply is only spoken or saved as text.** There are no tools and nothing it can run, so a
  malicious reply or book can only make the AI say something.
- **Limits:** `OS_STREAM_MAX_CONNECTIONS` caps realtime sessions, book jobs run one at a time, the preset
  fields have the caps above, and a reply is read up to 8 MiB.

## What to expect for speed

A voice turn's delay is roughly the end-of-turn silence (`silence_duration_ms`, 500 ms by default), plus
the faster-whisper decode, plus the AI's time for its thinking and the first sentence of its answer, plus
the voice model's first audio. With Stream replies off it is the AI's whole reply instead of its first
sentence. The AI part dominates. Low thinking keeps it short, keep voice replies short through the prompt,
and the waiting phrases fill the silence from 1.2 seconds on. Book jobs have no one waiting on each piece,
so more thinking is fine there. An AI server on another machine also leaves this machine's GPU to
faster-whisper and the voice model.

Each voice turn that goes to the AI logs one INFO line of its timings, in numbers only (never a word said,
answered or thought, and never the key):

```
Realtime turn of session sess_…: silence_ms=500 stt_ms=640 ai_first_token_ms=180 ai_first_answer_ms=1450 ai_done_ms=2300 first_audio_ms=1620 reply_chars=212 tokens_in=812 tokens_out=96 streamed=1
```

| Field | Meaning |
|---|---|
| `silence_ms` | The end-of-turn silence server VAD waited for; `-` for a typed question or a client's own commit |
| `stt_ms` | How long faster-whisper took on the turn; `-` for a typed question |
| `ai_first_token_ms` | From the request to the AI's first piece, thinking or answer |
| `ai_first_answer_ms` | From the request to the first piece of the answer: the gap from `ai_first_token_ms` is the thinking |
| `ai_done_ms` | From the request to the reply's end (or its failure) |
| `first_audio_ms` | From the request to the reply's first audio sent (not a waiting phrase's): what is left after `ai_first_answer_ms` is the voice's share (the first sentence completing, then the voice model) |
| `reply_chars` | The answer's length as spoken, in characters |
| `tokens_in`, `tokens_out` | The server's token counts, when it sends them (`stream_options.include_usage`) |
| `streamed` | `1` when the reply arrived as a stream, `0` when it came whole |

`-` marks what wasn't measured or reached. What you heard waiting is about `silence_ms + stt_ms +
first_audio_ms`. When `ai_first_answer_ms` is close to `ai_done_ms` on a streamed turn, the answer arrived in one
go: see [The AI assistant](TROUBLESHOOTING.md#the-ai-assistant) under Troubleshooting.
