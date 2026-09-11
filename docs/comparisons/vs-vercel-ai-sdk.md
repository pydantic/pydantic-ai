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
| Test offline | `MockLanguageModelV4` from `ai/test` | A fake model you script; no API key |

## FAQ

**Can the React UI stay?** Yes. [`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter] speaks
their protocol. The agent is Python.

**Can a coding agent sit behind that UI?** Yes. [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/)
on the Python agent. The adapter still speaks their protocol.
