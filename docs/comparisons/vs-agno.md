# Pydantic AI vs Agno

You're choosing a Python agent framework and have narrowed it to [Pydantic AI](../agent.md) and Agno.
This page makes the call — and lets you check the evidence yourself: every snippet runs offline,
no API keys.

## Pydantic AI fits if you need

- an agent that **runs where your app runs** — sync, async, or iterated, same result
- the production seams: deps, budgets, resumable cancellation, offline tests
- durability by **choosing an engine** rather than adopting a runtime

## Why the answers differ

Their value is the runtime they add; ours is that there is nothing to add. The same agent, three driving styles, one result.

## See it work

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

## The details

| What you get | Agno | Pydantic AI |
|---|---|---|
|---|---|---|
| Runtime | Bundled runtime + AgentOS hosted platform | Runtime-agnostic: sync, async, or driven node-by-node (proven below) |
| Tools | Curated toolsets; shell tool wraps host `subprocess` (requires confirmation) | Typed tools + toolsets per run + capabilities; deps boundary |
| Multi-agent | Agent teams | Plain async orchestration; graph builder |
| Evals | Included | Typed datasets + evaluators, CI-runnable offline |
| Durable | Your infrastructure / platform | Six engine wraps on the public interface |

## If this answer doesn't fit you

If all-in-one plus a hosted runtime (AgentOS) and team orchestration is the product you want, Agno genuinely ships that. We chose the opposite trade: nothing to adopt, run where your app runs. This page is that trade, demonstrated.

---

---

*Versions: agno 3.0.x; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
