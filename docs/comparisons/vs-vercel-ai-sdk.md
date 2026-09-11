# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK is the wire to a React UI: streaming, tool cards, approval in the browser. No
Python library matches that, ours included.

If the agent lives in TypeScript, use it. If the agent lives in Python, keep the AI SDK in the
browser and put Pydantic AI behind it (we speak their protocol). `abortSignal` aborts the request.
Ours raises `RunCancelled` holding the conversation.

## Side by side

| | Vercel AI SDK 7.0.97 | Pydantic AI 2.42 |
|---|---|---|
| Language | TypeScript | Python |
| UI stream | What it was built for | Adapters, including their protocol |
| Stop | `abortSignal` | `RunCancelled` with history |
| Structured output | Chosen for you | You pick the transport |
| Crash recovery | Not first-party | Six engines wrap the agent |
| Test offline | Subclass their spec | `TestModel` / `FunctionModel` |

## FAQ

**Both?** Yes. That's a common split.

**Drop-in?** No. Different language.

---

*`ai` 7.0.97, Pydantic AI 2.42. No durable-agent export in v7.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
