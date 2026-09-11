# Pydantic AI vs OpenAI Agents SDK

The OpenAI Agents SDK is OpenAI's agent library: sessions, guardrails, handoffs, and new OpenAI
features first. `after_turn` cancel is a stop mode we don't have.

Pydantic AI runs on any model, including OpenAI. The fork is what you hold after you stop, and which
engine recovers a crashed run.

## Side by side

| | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
| Models | OpenAI first; others via LiteLLM | Any provider, `FallbackModel` |
| Stop | `cancel('immediate')` or `cancel('after_turn')` on a stream | A stop signal; you get the messages back |
| Continuity | A session | A message list you store |
| Crash recovery | Not first-party | The same agent, inside Temporal, DBOS, or Prefect |
| Trusted state | `TContext`, not sent to the LLM | A typed object your tools read; the model never sees it |
| Guardrails | Input, output, tool tripwires | Capabilities |
| Test offline | `ScriptedModel` in `agents.testing`; no `ALLOW_MODEL_REQUESTS` | A fake model you script; no API key |
| Tracing | Their dashboard; zero `gen_ai.*` | OpenTelemetry GenAI names, when you turn them on |

## Stop, keep the conversation, continue

A support agent has filed a refund. The customer closes the chat. You want the work, and a way to
finish it when they come back.

Their streamed `cancel()` ends the iteration. What you keep is a session, or whatever you collected
while it streamed. Ours raises [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] holding the
messages. The next `run` continues from that list. Storage is yours.

A `CancellationToken` cannot be passed through Temporal, DBOS, or Prefect; cancel that workflow
instead. Cancelling from inside a tool leaves the last model tool call unexecuted.

Durability is the same split. They remember conversations as sessions. The same
[`Agent`][pydantic_ai.Agent] you write for a local run is what Temporal, DBOS, or Prefect resumes
after a crash. Their SDK has a `temporal` extra so it can detect a Temporal workflow; that is not a
wrapper that replays the agent.

## FAQ

**Can I keep using OpenAI models?** Yes, including the Responses API, and any other provider with
`FallbackModel`.

**The user closed the tab.** You get the conversation back as
[`RunCancelled`][pydantic_ai.exceptions.RunCancelled]. The next `run` continues from that list.
Storage is yours.
