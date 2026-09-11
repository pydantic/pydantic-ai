# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK is the wire to a React UI: streaming, tool cards, approval in the browser. No
Python library matches that, ours included.

If the agent lives in TypeScript, use it. If the agent lives in Python, keep the AI SDK in the
browser and put Pydantic AI behind it ([`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter]
speaks their protocol). `abortSignal` aborts the request. Ours raises `RunCancelled` holding the
conversation.

## Side by side

| | Vercel AI SDK `ai` 7.0.97 | Pydantic AI 2.42 |
|---|---|---|
| Language | TypeScript | Python |
| UI stream | What it was built for | [`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter] |
| Stop | `abortSignal` | `RunCancelled` with history |
| Structured output | `generateObject` / `Output` | You pick the transport |
| Crash recovery | Not in `ai`; `@ai-sdk/workflow` is a sibling | Six engines wrap the agent |
| Test offline | `MockLanguageModel` from `ai/test` | `TestModel` / `FunctionModel` |

## FAQ

**Both?** Yes. That's a common split.

**Drop-in?** No. Different language.

---

*`ai` 7.0.97 tarball unpacked. `ToolLoopAgent`, `abortSignal`, `uploadSkill`, and `ai/test`
`MockLanguageModel` are in the package. `WorkflowAgent` is documented for `@ai-sdk/workflow`, not
exported from `ai`. Pydantic AI 2.42.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
