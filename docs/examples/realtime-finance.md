You talk to the model with your voice, and it talks back — in its **own generated voice**. No
"press to transcribe", no typing. Real speech-to-speech. And when you ask it something it can't just
*say* off the top of its head — "how am I doing on my savings goal?" — it quietly hands the hard part
to a second agent and reads you the answer once the numbers come back.

That handoff is the whole point of this example. It's the pattern you'll reach for again and again:
keep a friendly, fast voice on the front, and let a normal [`Agent`][pydantic_ai.Agent] do the
careful thinking behind it.

Demonstrates:

- [realtime sessions](../realtime.md) — streaming audio in and out over one connection
- [delegating to a text agent](../realtime.md#delegating-to-a-text-agent) — the voice can't do
  structured output, so it asks one that can
- [background tools](../realtime.md#background-tools) — the model keeps chatting while a slow,
  multi-step analysis runs, and the result streams in when it's ready
- [tools](../tools.md) and [structured output](../output.md) — tool results are typed Pydantic
  models, rendered as cards in the chat

## How it fits together

Here's the mental model — it's smaller than it sounds:

```
🎙  you speak ──▶ OpenAI Realtime (the "voice")
                      │  "let me check that…"  (speaks immediately)
                      ▼
                 ask_analyst / run_deep_analysis      ← tools on the voice agent
                      │
                      ▼
                 supervisor Agent (a normal text model)
                      │  returns typed Widgets
                      ▼
🔊  voice reads the answer  +  📇 cards appear in the browser
```

The voice agent's job is to be *responsive* — to acknowledge you out loud right away. The supervisor's
job is to be *correct*. Realtime models aren't great at multi-step reasoning and can't return
structured output, so we don't ask them to. We give the voice a single tool that delegates to a
supervisor whose tool outputs are typed `Widget`s, and the browser draws those as cards.

!!! tip "Why two agents?"
    It's tempting to make one model do everything. But a realtime voice model and a frontier text
    model are good at *different* things. Letting each do what it's best at — the voice stays snappy,
    the supervisor stays accurate — is what makes the experience feel good. This split is the single
    most useful idea in the example.

The supervisor delegates in two flavors, and you can hear the difference:

- **Synchronous** — `ask_analyst` answers a quick question. The voice waits, then speaks.
- **Background** — `run_deep_analysis` and `plan_savings_goal` are slower, so they run in the
  background. The model keeps talking to you while the work happens, and the result cards stream in
  when they're done. Just say "run a deep analysis on my spending" and keep chatting.

## Running the Example

This one needs a single **`OPENAI_API_KEY`** with [OpenAI Realtime](https://platform.openai.com/docs/guides/realtime)
access — it powers both the voice and the supervisor. Put it in a `.env` file at the repo root:

```dotenv
OPENAI_API_KEY=sk-...
```

!!! info "Prefer Gemini? You can switch right in the UI."
    There's a provider dropdown in the browser — pick **Gemini** instead of OpenAI and the page asks
    the server for a Gemini voice. For that you'll also want a `GOOGLE_API_KEY` (or Vertex AI; see the
    [camera example](realtime-camera.md#using-a-work-google-cloud-account-vertex-ai)). The supervisor
    stays on OpenAI either way.

With [dependencies installed and your key set](./setup.md#usage), start the server:

```bash
uvicorn pydantic_ai_examples.realtime_finance.app:app
```

Then open <http://localhost:8000>, tap **Start**, allow the microphone, and just *talk*. Try:

- "How much did I spend on groceries last month?"
- "Am I on track for my savings goal?"
- "Run a deep analysis on my spending and walk me through it."

!!! note "It needs your microphone, so it needs a secure context"
    Browsers only allow microphone access on `localhost` or over HTTPS. On the same machine
    `localhost` is fine. To try it from your phone, expose it over HTTPS with a quick tunnel — the
    [camera example](realtime-camera.md#use-it-from-your-phone-https) shows exactly how.

!!! tip "Want to see the whole conversation?"
    Set a `LOGFIRE_WRITE_TOKEN` and the session shows up in [Logfire](../logfire.md) as a span tree:
    the realtime session, each tool call (including the delegated supervisor run), and cumulative
    token usage. It's the fastest way to understand what the model actually did.

## Example Code

The server — it bridges the browser WebSocket to a realtime session, forwarding mic audio in and
audio/cards out:

```snippet {path="/examples/pydantic_ai_examples/realtime_finance/app.py"}```

The voice agent and its tools — this is where the delegation and background dispatch live:

```snippet {path="/examples/pydantic_ai_examples/realtime_finance/voice.py"}```

The supervisor — an ordinary [`Agent`][pydantic_ai.Agent] whose tools return typed widgets:

```snippet {path="/examples/pydantic_ai_examples/realtime_finance/supervisor.py"}```

The typed results the supervisor returns, which the browser renders as cards:

```snippet {path="/examples/pydantic_ai_examples/realtime_finance/widgets.py"}```

And the browser: it captures the microphone as PCM, plays the model's audio back, and draws the
cards. (It's plain HTML and JavaScript — no build step.)

```snippet {path="/examples/pydantic_ai_examples/realtime_finance/index.html"}```
