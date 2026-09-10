# Pydantic AI vs OpenAI Agents SDK

You're choosing a Python agent framework and you're down to
[Pydantic AI](../agent.md) and the OpenAI Agents SDK. This page is the tiebreaker — the answer first, then code
you can run in seconds.

## Pydantic AI fits if you need

- one extension noun — a capability (tools + instructions + settings + hooks, deferrable, serializable)
- a **deps boundary** the model cannot cross
- **cancellation that resumes**: stop the run, keep the history, continue as an ordinary run
- durability by wrapping, offline tests, or a choice of providers

## Why the answers differ

Their framework organizes the agent into platform-shaped categories (guardrails, handoffs, sessions); ours has one typed unit that carries the same concerns. Their resume is a session on their platform; ours is an exception that carries the history to the next run.

## See it work

Say you need a stop button that doesn't destroy the conversation.

In the OpenAI Agents SDK, streamed runs can stop (`cancel(mode='immediate'|'after_turn')`), and resume lives in platform sessions (0.17.3).

Your side, runs offline:

```python {title="cancel_then_resume.py"}
"""Cancellation, then resume: the exception carries the work; the next run continues.

A stop gesture interrupts the in-flight model request. The run ends in
RunCancelled; everything completed before it — the record_a result — is
preserved in the exception's history. Pass that history to the next run:
the remaining work executes and the run completes.
"""
import asyncio
import threading
import time

from pydantic_ai import Agent, CancellationToken, RunCancelled
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

calls = 0


async def model(messages, info):
    global calls
    calls += 1
    if calls == 1:
        return ModelResponse(parts=[ToolCallPart('record_a', {'item': 'invoice'})])
    if calls == 2:
        await asyncio.sleep(3600)  # in-flight request, awaiting the model
    if calls == 3:
        return ModelResponse(parts=[ToolCallPart('record_b', {'item': 'receipt'})])
    return ModelResponse(parts=[TextPart('run resumed and completed')])


agent = Agent(FunctionModel(model), deps_type=None)


@agent.tool
def record_a(ctx, item: str) -> str:
    return f'a:{item}:done'


@agent.tool
def record_b(ctx, item: str) -> str:
    return f'b:{item}:done'


def main():
    token = CancellationToken()

    def stop_gesture():
        time.sleep(0.3)
        token.cancel()

    threading.Thread(target=stop_gesture, daemon=True).start()
    try:
        agent.run_sync('start', cancellation_token=token)
        raise SystemExit('BUG: first run completed')
    except RunCancelled as exc:
        history = exc.all_messages()
        preserved = 'invoice' in str(history)
        print(f'first run cancelled; completed work preserved: {preserved}')

    resumed = agent.run_sync('continue', message_history=history)
    print('resumed run output:', resumed.output)
    assert preserved
    assert resumed.output == 'run resumed and completed'


main()


```

```text
first run cancelled; completed work preserved: True
resumed run output: run resumed and completed
```

**Notice:** Here the stop is an exception that carries the history. The next run is ordinary code — and the completed work is already in it.

## The details

| What you get | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
|---|---|---|
| Extension model | Separate categories: **guardrails** (functions), **handoffs** (tools named `transfer_to_<name>`), hooks | **One noun**: a capability bundles tools + instructions + settings + hooks, orderable, deferrable, serializable into `AgentSpec` |
| Trusted state | `TContext` flows through the loop | `deps_type` — the model cannot choose or see it |
| Cancellation | Streamed-run `cancel(mode='immediate'\|'after_turn')` | Typed: `CancellationToken` (thread-safe, multi-run), `ctx.cancel()`, `RunCancelled` carrying resumable history |
| Resume | Sessions / `previous_response_id` — platform continuity | The exception carries history; resume is a normal run (proven below) |
| Durability | Engine-side adapters (the Temporal contrib exists) | First-party wraps on the public interface — Temporal, DBOS, Prefect, Restate, Kitaru, Airflow |
| Output | Plain JSON validated into typed models via `tools=[]` | Output transports: text, tool, native, structured — wire semantics are yours |
| Events | Run items — platform-shaped | Typed event stream (part/tool/result/final); capabilities can transform it |
| Offline tests | Pluggable `Model`, no first-party test model | `TestModel` / `FunctionModel` drive the whole pipeline deterministically |

## If this answer doesn't fit you

If you're all-in on the OpenAI platform — sessions, Responses continuity, their tracing — their SDK is the natural layer, and honestly, `after_turn` is a nice stop. We won't pretend otherwise. Our claim is narrower: if the loop has to be yours, these are the seams the platform shape won't give you — and OpenAI models work with us either way.

---

---

*Versions: openai-agents 0.17.3; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
