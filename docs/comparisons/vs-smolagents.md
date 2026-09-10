# Pydantic AI vs smolagents

You're choosing a Python agent framework and you're down to
[Pydantic AI](../agent.md) and smolagents. This page is the tiebreaker — the answer first, then code
you can run in seconds.

## Pydantic AI fits if you need

- **structured async**: parallel tool calls, budgets, cancellation, event streams
- a **deps boundary** — bounding what the model knows, not just where code runs
- **evals in CI**, and seams that hold for every provider, not just Python-executing models

## Why the answers differ

Their loop executes the code the model writes inside a sandbox; ours bounds the model itself with typed deps, and the loop is structured enough to limit, cancel, and observe. Different problems — the proof shows the concurrency their sync loop cannot have.

## See it work

Say you want parallel work, not a loop that serializes.

smolagents runs sync-only — zero `asyncio`/`anyio` reference in 1.26.0.

Your side, runs offline:

```python {title="parallel_tool_calls.py"}
"""Parallel tool calls in one turn.

The model asks for three slow calls in one response; the async loop runs
them concurrently. Wall time tracks the slowest, not the sum.
"""
import asyncio
import time

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

started: list[str] = []


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart('slow', {'name': 'a', 'ms': 250}),
                ToolCallPart('slow', {'name': 'b', 'ms': 250}),
                ToolCallPart('slow', {'name': 'c', 'ms': 250}),
            ]
        )
    return ModelResponse(parts=[TextPart('done')])


agent = Agent(FunctionModel(model))


@agent.tool
async def slow(ctx, name: str, ms: int) -> str:
    await asyncio.sleep(ms / 1000)
    started.append(name)
    return f'{name}:done'


async def main():
    t0 = time.perf_counter()
    with capture_run_messages() as msgs:
        await agent.run('run the three jobs')
    wall = time.perf_counter() - t0
    seen = {p.content for m in msgs for p in m.parts if hasattr(p, 'content') and 'done' in str(getattr(p, 'content', ''))}
    print(f'completed in parallel: {sorted(seen)}')
    print(f'wall time: {wall:.2f}s (the three calls would take ~0.75s one after another)')
    assert {'a:done', 'b:done', 'c:done'} <= seen
    assert wall < 0.7


asyncio.run(main())


```

```text
completed in parallel: ['a:done', 'b:done', 'c:done', 'done']
wall time: 0.26s (the three calls would take ~0.75s one after another)
```

**Notice:** Three 250 ms calls in one response finished together in ~0.26 s. Structured async is the prerequisite for budgets and cancellation that actually work.

## The details

| What you get | smolagents | Pydantic AI |
|---|---|---|
|---|---|---|
| Runtime | Sync-only — the loop owns your thread; zero `asyncio`/`anyio` reference (grep-verified) | Async-first; concurrent tool calls run in parallel (proven below) |
| Model interaction | Writes and executes Python in a sandbox | Calls typed tools with validated arguments |
| Isolation | Code sandbox: `import os` and `open()` are forbidden; 11-module allowlist | Typed deps — the model cannot choose or see trusted state; tools hold the boundary |
| Stopping | Only `final_answer` (model-driven) | Typed cancellation: `ctx.cancel()`, thread-safe token, catchable `RunCancelled` with resumable history |
| Extension | Tools + the Model ABC (a real seam) | Capabilities: tools + instructions + hooks, deferrable, serializable |
| Evals | Not first-party | Typed datasets + evaluators, CI-runnable offline |

## If this answer doesn't fit you

If your agent's whole job is writing and running Python, smolagents is the smallest thing that does it, and its sandboxing story is real. We won't pretend we're the minimal code-executor. We're built for what wraps around it — budgets, cancellation, checks that run in CI. Different axes; this page shows ours.

---

---

*Versions: smolagents 1.26.0; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
