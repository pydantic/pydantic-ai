# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK is the wire to a React UI: streaming, tool cards, approval in the browser. Pydantic
AI is a Python agent. [`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter] speaks their
protocol, so the browser can stay theirs.

`abortSignal` aborts the request. Ours raises `RunCancelled` holding the conversation.

## Side by side

| | Vercel AI SDK | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| UI stream | What it was built for | [`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter] |
| Stop | `abortSignal` | A stop signal; you get the messages back |
| Structured output | `generateObject` / `Output` | You pick the transport |
| Crash recovery | Not in `ai`; `@ai-sdk/workflow` is a sibling | The same agent, inside Temporal, DBOS, or Prefect |
| Test offline | `MockLanguageModel` from `ai/test` | A fake model you script; no API key |

## FAQ

**The agent lives in TypeScript?** Use the AI SDK. This page is for a Python agent behind that UI.

**Both?** Yes. That's a common split.

**Drop-in?** No. Different language.
