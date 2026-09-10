# Pydantic AI vs Google ADK

You're choosing a Python agent framework and you're down to
[Pydantic AI](../agent.md) and Google ADK. This page is the tiebreaker — the answer first, then code
you can run in seconds.

## Pydantic AI fits if you need

- a **stop gesture that interrupts cleanly** — from a thread, a tool, or one token governing many runs
- interruption that **resumes with history intact**
- cancellation as a typed outcome, plus deps, budgets, and evals

## Why the answers differ

ADK runs deep anyio — the primitives exist — but its runner exposes no user cancellation API that we could find. We ship cancellation as a product on the same primitives: typed, thread-safe, resumable. One token cancelling three concurrent runs is the small version.

## See it work

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

## The details

| What you get | Google ADK | Pydantic AI |
|---|---|---|
|---|---|---|
| Foundation | Deeply anyio-internal (task groups, `fail_after`, interceptors) — arguably the closest to us in the anyio world | asyncio-native with the same anyio primitives at the seams |
| Cancellation | **No user cancellation API**: `def cancel` exists only in the A2A executor; the runner surfaces nothing | Typed: `CancellationToken` (thread-safe, one token = many runs — proven below), `ctx.cancel()`, catchable `RunCancelled` |
| Extension | Agent/Workflow classes + code executors | Capabilities: one unit, deferrable, serializable, event-stream-aware |
| Trusted state | Handlers receive app context | `deps_type` boundary — the model cannot choose or see it |
| Durable | Workflows; checkpointing is your responsibility | Six engine wraps on the public interface (Temporal/DBOS/Prefect/Restate/Kitaru/Airflow) |
| Evals | `evaluation` module exists | Typed datasets + evaluators, CI-runnable offline |

## If this answer doesn't fit you

If you want a vendor-maintained, full-surface framework on the Google stack — workflows builder, code executors, A2A — ADK is a deliberate choice, and we respect it. Our only claim is the one gap we verified: its runner exposes no user cancellation surface. If a stop gesture matters to you, that's this page's whole point.

---

---

*Versions: Google ADK (install/docs); Pydantic AI 2.42.0 — 2026-09. Snippets re-executed by this repository's tests.*
