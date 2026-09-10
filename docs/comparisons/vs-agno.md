# Pydantic AI vs Agno

**Agno** is a batteries-included Python framework with a bundled runtime and the AgentOS
hosted platform — plus toolsets, teams, and evals in the modern 3.x rewrite.

**Pydantic AI** is a library, not a runtime — the same agent runs sync, async, or
node-by-node wherever your application already runs, with typed seams and engines you choose.

*Verified against `agno 3.0.x` (2026-09-10). Pydantic AI claims below are self-contained scripts —
offline, no API keys — re-executed by this repository's test suite.*

## Quick comparison

| What you get | Agno | Pydantic AI |
|---|---|---|
| Runtime | Bundled runtime + AgentOS hosted platform | Runtime-agnostic: sync, async, or driven node-by-node (proven below) |
| Tools | Curated toolsets; shell tool wraps host `subprocess` (requires confirmation) | Typed tools + toolsets per run + capabilities; deps boundary |
| Multi-agent | Agent teams | Plain async orchestration; graph builder |
| Evals | Included | Typed datasets + evaluators, CI-runnable offline |
| Durable | Your infrastructure / platform | Six engine wraps on the public interface |

## Prove it yourself

No runtime to adopt — the same agent gives the same result however your app drives it:

```python {title="runtime_agnostic.py"}
"""No bundled runtime to adopt: the same agent runs sync, async, and driven
node-by-node with iter() - whichever shape your application already uses."""
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart


async def model(messages, info):
    return ModelResponse(parts=[TextPart('same result')])


agent = Agent(FunctionModel(model))


async def via_iter():
    async with agent.iter('q') as run:
        async for _ in run:
            pass
    return run.result.output


sync_result = agent.run_sync('q').output
async_result = asyncio.run(agent.run('q')).output
iter_result = asyncio.run(via_iter())

print(f'sync={sync_result!r} async={async_result!r} iter={iter_result!r}')
assert sync_result == async_result == iter_result
```

```text
sync='same result' async='same result' iter='same result'
```

## Key differences

**Agno.** batteries and a hosted platform are real — AgentOS, teams, and bundled tools lower
the start cost.

**Pydantic AI.** the loop stays in your process, so every production seam applies — typed deps, budgets,
cancellation with resumable history, offline tests — and durability is chosen by wrapping, not by
adopting a runtime.

## When to choose Agno

You want the all-in-one plus a hosted runtime and team orchestration out of the box.

## When to choose Pydantic AI

Your agent runs where your app runs — and you want the seams that make it production-safe regardless
of where that is.

## Summary

Their value is the runtime they add; ours is that there is nothing to add. The agent returned
result' sync, async, and iterated.

*Agno behavior pinned to 3.0.x (installed, probed); records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI verified
on 2.42.0, 2026-09-10.*