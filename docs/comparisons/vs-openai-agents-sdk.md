# Pydantic AI vs OpenAI Agents SDK

The OpenAI Agents SDK is OpenAI's own agent library. You build an `Agent` and hand it to a `Runner`,
and the pieces around it are shaped like the OpenAI platform: conversations persist as **sessions**,
safety checks are **guardrails**, and passing work to another agent is a **handoff**. It is small,
well documented, and it gets new OpenAI features first.

It has grown a lot too. Version 0.22.2 puts guardrails on input, on output, and now on tool calls
going both ways. Handoffs can filter and nest history. Streamed runs stop with
`cancel(mode='immediate')` or `cancel(mode='after_turn')`, and we don't have an equivalent of
`after_turn`. Structured output is clean: hand it an `output_type` and it validates the model's
JSON into your type, no forced tool call.

So what's actually left to argue about? Two things. Where your agent is allowed to run, and what
you're holding after you stop it.

## Stopping a run without losing it

Say a customer support agent has already filed a refund request, and then the customer closes the
chat. You want to stop, keep what happened, and pick it up when they come back.

In the OpenAI SDK you stop a streamed run and the iteration ends. What you keep afterwards depends on
where the conversation lives: if you use a session, that's your record; if not, it's whatever you
collected as it streamed. Sessions are a small protocol — `get_items`, `add_items`, `pop_item`,
`clear_session` — and the storage behind them ships as separate packages.

In Pydantic AI, stopping produces a value. The run raises `RunCancelled`, that exception carries the
whole conversation, and passing it to the next run continues from there. Nothing is stored anywhere
unless you store it.

```python {title="cancel_then_resume.py"}
"""Stopping keeps the work: the exception carries the conversation, and the
next run continues from it."""

from pydantic_ai import Agent, CancellationToken, RunCancelled, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

calls = 0


async def model(messages, info):
    global calls
    calls += 1
    if calls == 1:
        return ModelResponse(parts=[ToolCallPart('file_refund', {'item': 'invoice'})])
    if calls == 2:
        return ModelResponse(parts=[ToolCallPart('notify_customer', {'item': 'invoice'})])
    return ModelResponse(parts=[TextPart('Refund filed and the customer was told.')])


token = CancellationToken()
agent = Agent(FunctionModel(model))


@agent.tool_plain
def file_refund(item: str) -> str:
    return f'{item}: refund filed'


@agent.tool
def notify_customer(ctx: RunContext, item: str) -> str:
    token.cancel()  # e.g. the customer closed the chat just as we got here
    return f'{item}: customer emailed'


try:
    agent.run_sync('Refund the duplicate invoice charge.', cancellation_token=token)
except RunCancelled as cancelled:
    history = cancelled.all_messages()
    print('work kept while stopped:', 'refund filed' in str(history))
    #> work kept while stopped: True

resumed = agent.run_sync(message_history=history)  # no new prompt: the run is mid-flight
print('after resuming:', resumed.output)
#> after resuming: Refund filed and the customer was told.
```



The completed work is in the history, and the second run finishes the job. There is no session
service in the middle, so the history is yours to put wherever you already put things.

One thing worth knowing before you rely on it: cancelling from inside a tool leaves the model's last
tool call unexecuted, so a resume drops it. Cancelling with a token instead marks the history
interrupted, which is cleaner.

## Where the agent can run

The other difference is durability, and it's less about features than about who owns the loop.

A Pydantic AI run is an ordinary coroutine, so a durable engine can wrap the agent object without
changing it. Adding `TemporalDurability()` to its capabilities gives you Temporal's retries and
crash recovery; DBOS and Prefect
wrappers ship in the same repository, and Restate, Kitaru, and Airflow adapters live in those
projects. If your company already runs one of those, that's the one you use.

The OpenAI SDK has no first-party equivalent, though a Temporal contrib package exists — which is
worth saying, because it proves the idea isn't impossible there. Their durability story is sessions,
and sessions remember conversations, not executions. If the process dies halfway through a run,
a session tells you what was said, not what was half-done.

## Side by side

| | OpenAI Agents SDK 0.22.2 | Pydantic AI 2.42 |
|---|---|---|
| Models | OpenAI first; others through LiteLLM or a custom `Model` | Any provider directly, with `FallbackModel` for failover |
| Trusted state | `TContext` travels with the run | `deps_type`, a separate argument tools read and the model never sees |
| Stopping a run | `cancel('immediate')` or `cancel('after_turn')` on a streamed run | `CancellationToken` from any thread, or `ctx.cancel()` inside a tool; ends in `RunCancelled` holding the history |
| Picking it back up | A session, or `previous_response_id` | Pass the history to the next run; storage is yours |
| Safety checks | Guardrails on input, output, and tool calls, with tripwires | Capabilities, which bundle tools, instructions, settings and hooks together and can load on demand |
| Handing work over | `handoff()` registers a tool named `transfer_to_<agent>` | An agent used as a tool, or a capability |
| Crash recovery | Not first-party; a Temporal contrib exists | Six engines wrap the agent: Temporal, DBOS, Prefect, Restate, Kitaru, Airflow |
| Testing offline | Write your own `Model`; there's no test model included | `TestModel` and `FunctionModel` ship with it; `ALLOW_MODEL_REQUESTS = False` blocks real calls |
| Evals | A separate product | `pydantic-evals` runs in your test suite using the agent's own types |
| Tracing | Their dashboard, or OpenInference spans under its own attribute names; zero `gen_ai.*` | OpenTelemetry GenAI semantic conventions (36 `gen_ai.*` attributes) — your existing dashboards read them |

## Choose the OpenAI SDK when

- You are building on OpenAI and want their newest features the week they land.
- Their hosted sessions and tracing dashboard are things you'd rather not build or run.
- `after_turn` is the stop you want: let the current turn finish, then halt.
- You want the smallest possible amount of code between you and the Responses API.

## Choose Pydantic AI when

- You want the same agent to run on Claude, Gemini, and GPT without a rewrite.
- Credentials and customer identity must sit where the model can't reach them.
- You need crash recovery from an engine your company already operates.
- You want the agent's tests to run offline in CI like the rest of your test suite.

## FAQ

**Can I use OpenAI models with Pydantic AI?**
Yes, including the Responses API, and that's how most people run it. Choosing us isn't choosing
against OpenAI.

**Is it a drop-in replacement?**
No. Tools and prompts carry over almost unchanged. Guardrails become capabilities or plain validation,
handoffs become an agent called as a tool, and sessions become message history you store yourself.

**What does the OpenAI SDK do better?**
New OpenAI features arrive there first, `after_turn` is a real stop mode we don't have, and their
hosted tracing works the moment you install it.

---

*Checked against openai-agents 0.22.2 and Pydantic AI 2.42 on 2026-09-10. The OpenAI SDK facts come from
reading the installed package — method signatures, exported guardrail types, and `handoff()` parameters. The
Pydantic AI example is run by this repository's test suite on every commit, so its output is what it printed.
The `gen_ai.*` counts are distinct semantic-convention attribute names found in each installed package's
source; ours were also captured from a live run through a plain OpenTelemetry exporter. We recheck this page's
version pins and behaviour claims each time Pydantic AI ships a minor release; if something here has gone
stale, [tell us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
