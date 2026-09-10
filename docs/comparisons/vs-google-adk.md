# Pydantic AI vs Google ADK

**Google ADK** is a full-surface agent framework from a cloud vendor — `LlmAgent` config,
workflows with a builder, in-memory services, code executors (including subprocess interpreters via
anyio), and A2A for interop.

**Pydantic AI** is a typed loop with cancellation as a first-class, resumable outcome —
plus capabilities, deps, budgets, evals, and durability wraps.

*Verified against Google ADK 2026-09 (adk-venv install; docs-grounded where noted). Pydantic AI
claims below are self-contained scripts — offline, no API keys — re-executed by this repository's
test suite.*

## Quick comparison

| What you get | Google ADK | Pydantic AI |
|---|---|---|
| Foundation | Deeply anyio-internal (task groups, `fail_after`, interceptors) — arguably the closest to us in the anyio world | asyncio-native with the same anyio primitives at the seams |
| Cancellation | **No user cancellation API**: `def cancel` exists only in the A2A executor; the runner surfaces nothing | Typed: `CancellationToken` (thread-safe, one token = many runs — proven below), `ctx.cancel()`, catchable `RunCancelled` |
| Extension | Agent/Workflow classes + code executors | Capabilities: one unit, deferrable, serializable, event-stream-aware |
| Trusted state | Handlers receive app context | `deps_type` boundary — the model cannot choose or see it |
| Durable | Workflows; checkpointing is your responsibility | Six engine wraps on the public interface (Temporal/DBOS/Prefect/Restate/Kitaru/Airflow) |
| Evals | `evaluation` module exists | Typed datasets + evaluators, CI-runnable offline |

## Prove it yourself

They have the primitives (anyio task groups) but no user-facing cancel. Here one token stops three
concurrent runs:

```python {title="one_token_many_runs.py"}
"""One stop gesture, many runs: a CancellationToken governs every run it was
given to, and cancelling the token cancels all of them.

Google ADK's runner exposes no user cancellation API (grep-verified,
2026-09-10); here the same token stops three concurrent runs at once.
"""
import asyncio

from pydantic_ai import Agent, CancellationToken, RunCancelled
from pydantic_ai.models.function import FunctionModel

CONCURRENT = 3

async def hang(messages, info):
    await asyncio.sleep(3600)  # in-flight until cancelled

async def main():
    token = CancellationToken()
    agent = Agent(FunctionModel(hang))
    tasks = [asyncio.create_task(agent.run('r', cancellation_token=token)) for _ in range(CONCURRENT)]
    await asyncio.sleep(0.1)
    token.cancel()  # one gesture
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f'runs cancelled by one token: {sum(isinstance(r, RunCancelled) for r in results)}/{CONCURRENT}')
    assert all(isinstance(r, RunCancelled) for r in results)

asyncio.run(main())
```

```text
runs cancelled by one token: 3/3
```


The cancellation machinery is ours because the loop is ours — attribution between your cancel and an
external `CancelledError` (external wins), plus the resumable history, are part of the same seam.

## Key differences

**Google ADK.** the vendor surface is real — workflows, code executors, service layers, A2A — and
their anyio depth is genuinely close to ours.

**Pydantic AI.** on top of the same primitives we ship cancellation as a product: typed, thread-safe,
multi-run, resumable. ADK's own anyio task groups could do it — but the API isn't shipped.

## When to choose Google ADK

You want a vendor-maintained full-surface framework — workflows builder, code executors, A2A — and
cancellation is not your product concern (or you're already on the Google stack).

## When to choose Pydantic AI

A stop gesture, a quota check, or a human approval must interrupt the run cleanly — and resume with
history intact. One token cancelling 3 concurrent runs is the small version of that.

## Summary

Their anyio internals are infrastructure; our cancellation is an outcome. 3/3.

*Google ADK behavior docs/install-grounded 2026-09; records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI verified on
2.42.0, 2026-09-10.*