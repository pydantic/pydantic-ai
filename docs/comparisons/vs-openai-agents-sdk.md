# Pydantic AI vs OpenAI Agents SDK

The OpenAI Agents SDK is OpenAI's agent library: sessions, guardrails, handoffs, and new OpenAI
features first. `after_turn` cancel is a stop mode we don't have.

Pydantic AI runs on any model, including OpenAI. The fork is what you hold after you stop, and which
engine recovers a crashed run.

## Side by side

| | OpenAI Agents SDK 0.22.2 | Pydantic AI 2.42 |
|---|---|---|
| Models | OpenAI first; others via LiteLLM | Any provider, `FallbackModel` |
| Stop | `cancel('immediate')` or `cancel('after_turn')` on a stream | `RunCancelled` with history |
| Continuity | A session | A message list you store |
| Crash recovery | Not first-party | Six engines wrap the agent |
| Trusted state | `TContext`, not sent to the LLM | `deps_type` plus `RunContext` |
| Guardrails | Input, output, tool tripwires | Capabilities |
| Test offline | `ScriptedModel` in `agents.testing`; no `ALLOW_MODEL_REQUESTS` | `TestModel` / `FunctionModel`; `ALLOW_MODEL_REQUESTS = False` |
| Tracing | Their dashboard; zero `gen_ai.*` | OpenTelemetry GenAI conventions, when enabled |

## Stop, keep the conversation, continue

A support agent has filed a refund. The customer closes the chat. You want the work, and a way to
finish it when they come back.

Their streamed `cancel()` ends the iteration. What you keep is a session, or whatever you collected
while it streamed. Ours raises [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] holding the
messages. The next `run` continues from that list. Storage is yours.

```python {title="cancel_then_resume.py"}
from pydantic_ai import Agent, CancellationToken, RunCancelled, RunContext

token = CancellationToken()
agent = Agent('openai:gpt-5.6-luna')


@agent.tool_plain
def file_refund(item: str) -> str:
    return f'{item}: refund filed'


@agent.tool
def notify_customer(ctx: RunContext, item: str) -> str:
    token.cancel()
    return f'{item}: customer emailed'


try:
    agent.run_sync('Refund the duplicate invoice charge.', cancellation_token=token)
except RunCancelled as cancelled:
    history = cancelled.all_messages()
    print('work kept while stopped:', 'refund filed' in str(history))
    #> work kept while stopped: True

resumed = agent.run_sync(message_history=history)
print('after resuming:', resumed.output)
#> after resuming: Refund filed and the customer was told.
```

A `CancellationToken` cannot be passed through Temporal, DBOS, or Prefect; cancel that workflow
instead. Cancelling from inside a tool leaves the last model tool call unexecuted.

Durability is the same split. They remember conversations as sessions. We wrap the agent in an engine
you already run (`TemporalDurability()` still needs the worker and workflow). Their SDK has a
`temporal` extra so it can detect a Temporal workflow; that is not a durability wrapper that replays
the agent.

## FAQ

**Can I use OpenAI models?** Yes, including the Responses API. Choosing us isn't choosing against OpenAI.

**Drop-in?** No. Guardrails become capabilities. Handoffs become an agent used as a tool. Sessions
become history you store.

---

*openai-agents 0.22.2, installed. `RunResultStreaming.cancel(mode='immediate'|'after_turn')` is on
the streamed result, not `Runner`. `RunContextWrapper` documents that `TContext` is not passed to the
LLM. `ScriptedModel` lives in `agents.testing`. Zero `gen_ai.` strings in the package. Pydantic AI 2.42.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
