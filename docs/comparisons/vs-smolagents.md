# Pydantic AI vs smolagents

**smolagents, at its best:** the smallest loop that runs agents — the model writes Python, a
sandboxed executor runs it (with an allowlist of 11 stdlib modules, Docker/E2B/Modal/Blaxel options
for real isolation), and the loop ends when the model calls `final_answer`.

**Pydantic AI, at its best:** a structured, async-first loop with a typed deps boundary, exact
budgets, typed cancellation, and evals — the same seams for any provider.

*Verified against `smolagents 1.26.0` (2026-09-10). Pydantic AI claims below are self-contained
scripts — offline, no API keys — re-executed by this repository's test suite.*

## Quick comparison

| What you get | smolagents | Pydantic AI |
|---|---|---|
| Runtime | Sync-only — the loop owns your thread; zero `asyncio`/`anyio` reference (grep-verified) | Async-first; concurrent tool calls run in parallel (proven below) |
| Model interaction | Writes and executes Python in a sandbox | Calls typed tools with validated arguments |
| Isolation | Code sandbox: `import os` and `open()` are forbidden; 11-module allowlist | Typed deps — the model cannot choose or see trusted state; tools hold the boundary |
| Stopping | Only `final_answer` (model-driven) | Typed cancellation: `ctx.cancel()`, thread-safe token, catchable `RunCancelled` with resumable history |
| Extension | Tools + the Model ABC (a real seam) | Capabilities: tools + instructions + hooks, deferrable, serializable |
| Evals | Not first-party | Typed datasets + evaluators, CI-runnable offline |

## Prove it yourself

Their loop is sequential by construction. Ours runs a model's batch of tool calls concurrently —
three 250 ms calls finish in ~0.26 s, not ~0.75 s:

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

Concurrency is not the point by itself — it's why budgets, cancellation, and event streams work at
all: the loop is structured, so it can be limited, interrupted, and observed.

## Key differences

**Their best:** agentic code execution with real sandboxing options is a genuinely different model —
and their allowlist is a sane default for it.

**Ours:** the boundary is about *what the model is allowed to know and do*, not only where code runs.
A typed deps boundary, budgets that halt before side effects, resumable cancellation, and evals in
CI all exist because the loop is structured async — and they hold for every provider, not just
Python-executing models.

## When to choose smolagents

Your agent's job is writing and running Python, and you want the smallest thing that does that —
with their sandboxing story.

## When to choose Pydantic AI

You want a production loop: parallel execution, exact budgets, typed cancellation, offline tests —
and the model's reach bounded by deps, not just a sandbox wall.

## Summary

Their loop is a sandbox for the code the model writes; ours is a structured loop that bounds the
model itself. Three 250 ms calls finished in 0.26 s.

*smolagents behavior pinned to 1.26.0 (installed, probed); records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI verified
on 2.42.0, 2026-09-10.*