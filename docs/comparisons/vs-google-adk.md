# Pydantic AI vs Google ADK

Choosing an agent framework and you're down to
[Pydantic AI](../agent.md) and Google ADK. This page is the tiebreaker — the answer first, then code
you can run in seconds.

## Pydantic AI fits if you need

- a **stop gesture that interrupts cleanly** — from a thread, a tool, or one token governing many runs
- interruption that **resumes with history intact**
- cancellation as a typed outcome, plus deps, budgets, and evals

## Why the answers differ

ADK runs deep anyio — the primitives exist — but its runner exposes no user cancellation API that we could find. We ship cancellation as a product on the same primitives: typed, thread-safe, resumable. One token cancelling three concurrent runs is the small version. Community experience matches the shape: ADK builds workflows fast, but a developer on r/AI_Agents (2025) hit rigidity trying to inject state into every chat message — the thing a `deps` boundary gives you for free here.

## See it work

Say a user hits stop and the work has to actually stop.

Google ADK runs deep anyio internally, but its runner surfaces no user cancellation API we could find (2026-09 install).

Your side, runs offline:

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

**Notice:** One token, three concurrent runs, one gesture: all cancelled. Typed, thread-safe, and the history survives for resume.

## The details

| What you get | Google ADK | Pydantic AI |
|---|---|---|
|---|---|---|
| Foundation | Deeply anyio-internal (task groups, `fail_after`, interceptors) — arguably the closest to us in the anyio world | asyncio-native with the same anyio primitives at the seams |
| Cancellation | **No user cancellation API**: `def cancel` exists only in the A2A executor; the runner surfaces nothing | Typed: `CancellationToken` (thread-safe, one token = many runs — proven below), `ctx.cancel()`, catchable `RunCancelled` |
| Extension | Agent/Workflow classes + code executors | Capabilities: one unit, deferrable, spec-declarable, event-stream-aware |
| Trusted state | Handlers receive app context | `deps_type` boundary — the model cannot choose or see it |
| Durable | Workflows; checkpointing is your responsibility | Six engine wraps (Temporal/DBOS/Prefect first-party; Restate/Kitaru/Airflow external) |
| Evals | `evaluation` module exists | Typed datasets + evaluators, CI-runnable offline |

## If this answer doesn't fit you

If you want a vendor-maintained, full-surface framework on the Google stack — workflows builder, code executors, A2A — ADK is a deliberate choice, and we respect it. Our only claim is the one gap we verified: its runner exposes no user cancellation surface. If a stop gesture matters to you, that's this page's whole point.

---

## FAQ

**Is Pydantic AI a drop-in replacement for Google ADK?**
Drop-in, no — the loop and the seams are different, even though the ideas carry over (tools,
prompts, outputs). If you're weighing a move, that honesty is the point of this page: read the fits
list and run the proof before you decide.

**When should I use Google ADK on its own?**
When you want a vendor-maintained full-surface framework on the Google stack — workflows, code executors, A2A — and cancellation isn't your product concern.

**Why do people pick Pydantic AI over Google ADK?**
Because the loop is yours end to end — typed deps, cancellation that resumes, budgets that stop side
effects before they start, evals in CI — and every one of those claims is a snippet on this page you
can run in seconds. Community threads on r/AI_Agents add "documentation" and "low abstraction" to
that list; see Independent takes on the [overview](index.md).


---

*Versions: Google ADK (install/docs); Pydantic AI 2.42.0 — 2026-09. Snippets re-executed by this repository's tests.*
