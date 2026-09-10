# Pydantic AI vs OpenAI Agents SDK

**OpenAI Agents SDK, at its best:** the first-party SDK for the platform most of the industry runs
on — guardrails, handoffs, sessions, structured outputs, and a streamed run you can cancel with
`mode='after_turn'`.

**Pydantic AI, at its best:** one extension noun instead of categories — a capability carries tools,
instructions, settings, and hooks together — plus a typed deps boundary, durable wraps, and
cancellation that ends in a catchable, resumable exception.

*Verified against `openai-agents 0.17.3` (2026-09-10). Pydantic AI claims below are self-contained
scripts — offline, no API keys — re-executed by this repository's test suite.*

## Quick comparison

| What you get | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
| Extension model | Separate categories: **guardrails** (functions), **handoffs** (tools named `transfer_to_<name>`), hooks | **One noun**: a capability bundles tools + instructions + settings + hooks, orderable, deferrable, serializable into `AgentSpec` |
| Trusted state | `TContext` flows through the loop | `deps_type` — the model cannot choose or see it |
| Cancellation | Streamed-run `cancel(mode='immediate'\|'after_turn')` | Typed: `CancellationToken` (thread-safe, multi-run), `ctx.cancel()`, `RunCancelled` carrying resumable history |
| Resume | Sessions / `previous_response_id` — platform continuity | The exception carries history; resume is a normal run (proven below) |
| Durability | Engine-side adapters (the Temporal contrib exists) | First-party wraps on the public interface — Temporal, DBOS, Prefect, Restate, Kitaru, Airflow |
| Output | Plain JSON validated into typed models via `tools=[]` | Output transports: text, tool, native, structured — wire semantics are yours |
| Events | Run items — platform-shaped | Typed event stream (part/tool/result/final); capabilities can transform it |
| Offline tests | Pluggable `Model`, no first-party test model | `TestModel` / `FunctionModel` drive the whole pipeline deterministically |

## Prove it yourself

Their resume is a session on their platform. Ours is luggage: cancel the run, keep the history,
resume as an ordinary run that completes the remaining work.

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

The note that matters: the stop interrupt hit the *in-flight model request*, so history ends marked
`interrupted` and the next run repairs it automatically. Cancelling *from inside a tool* leaves the
pending tool call unexecuted and resume needs that trailing call dropped — a real difference between
the two interrupt points, documented in the research notes.

## Key differences

**Their best:** the platform is the product — Responses API continuity, sessions, memory, tracing,
and `after_turn` is a genuine turn-granularity grace on streamed runs.

**Ours:** the seams are typed and local. One capability noun pays for itself across all four of
their categories (their handoffs are tools with a reserved name; our non-tool capabilities hide
tools until loaded). The deps boundary, a cancellation that is a catchable exception, and durability
wraps on the public interface all exist because the run is a normal coroutine — nothing platform-
shaped hides inside it.

## When to choose OpenAI Agents SDK

You are all-in on the OpenAI platform — Responses API, sessions, platform tracing, and their
streaming UI behavior are your product. `after_turn` covers your stop-grace case.

## When to choose Pydantic AI

You want the run in your process with typed seams: a deps boundary, cancellation you can resume,
durability by wrapping, offline tests, and one extension noun. You can still use OpenAI models — the
[`openai` provider](../models/openai.md) is first-class.

## Summary

Their categories work because their platform is one system. Ours compose because they're one noun.
Their resume is a session; ours is an exception you pass to the next run.

*OpenAI SDK behavior pinned to 0.17.3; probe records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI behavior
verified on 2.42.0, 2026-09-10.*