# Pydantic AI vs CrewAI

You're choosing a Python agent framework and have narrowed it to [Pydantic AI](../agent.md) and CrewAI.
This page makes the call — and lets you check the evidence yourself: every snippet runs offline,
no API keys.

## Pydantic AI fits if you need

- orchestration with **branches, joins, or retries you can see and control**
- **typed seams at every stage** — fan-out returns ints, sums flow into deps
- the production checklist on top: cancellation, budgets, offline tests

## Why the answers differ

Their crew is a DSL the framework interprets; ours is ordinary async code with a type at every seam. Same multi-agent shape, far more visibility.

## See it work

```python {title="chain_as_code.py"}
"""A crew is code: chained and fanned-out agents with types between them.

Two branches each extract an int (the tool's signature and return are
typed); the branches are gathered; the typed sum flows into a downstream
agent's deps. No role DSL - just ordinary async code with a type at every
seam.
"""
import asyncio
import re

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart


async def model_num(messages, info):
    if len(messages) == 1:
        value = int(re.search(r'(\d+)', str(messages[0])).group(1))
        return ModelResponse(parts=[ToolCallPart('num', {'v': value})])
    return ModelResponse(parts=[TextPart('done')])


num_agent = Agent(FunctionModel(model_num))


@num_agent.tool
def num(ctx, v: int) -> int:
    return v


async def model_txt(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('describe', {'prefix': 'total'})])
    return ModelResponse(parts=[TextPart('done')])


txt_agent = Agent(FunctionModel(model_txt), deps_type=int)


@txt_agent.tool
def describe(ctx, prefix: str) -> str:
    return f'{prefix}:{ctx.deps}'


async def branch(value: int) -> int:
    with capture_run_messages() as msgs:
        await num_agent.run(f'get {value}')
    returns = [p.content for m in msgs for p in m.parts if isinstance(p, ToolReturnPart)]
    assert returns, 'no tool return captured'
    return int(returns[0])  # the typed tool return is the seam


async def main():
    a, b = await asyncio.gather(branch(21), branch(22))
    total = a + b
    with capture_run_messages() as msgs:
        await txt_agent.run('finish', deps=total)
    used = 'total:43' in str(msgs)
    print(f'typed chain: branch(a)={a!r}, branch(b)={b!r}, sum={total!r}')
    print(f'downstream agent received the typed sum as deps: {used}')
    assert isinstance(a, int) and isinstance(b, int)
    assert used


asyncio.run(main())

```

```text
typed chain: branch(a)=21, branch(b)=22, sum=43
downstream agent received the typed sum as deps: True
```

## The details

| What you get | CrewAI | Pydantic AI |
|---|---|---|
|---|---|---|
| Orchestration | A DSL: `Agent(role=..., goal=..., backstory=...)`, `Crew(process=...)` | Plain async code: chain agents, fan out with `asyncio.gather`, branch with normal control flow |
| Typed seams | `inputs` dicts; output models via Pydantic | `deps_type` boundary; typed tool signatures and returns at every seam (proven below) |
| Loop access | A `kickoff()` you run; stopping = kill your thread | Typed cancellation + resumable history; `iter()` drives the loop node-by-node |
| Memory/knowledge | Built-in memory + `StringKnowledgeSource` | Your memory: deps + history processors are yours to wire |
| Evals | Not first-party in the core loop | Typed datasets + evaluators, CI-runnable offline |

## If this answer doesn't fit you

If crew-of-roles with memory and knowledge out of the box is exactly your shape, CrewAI's DSL compresses it nicely, and we're not going to pretend this page replaces that. Ours is for when the orchestration needs to be code you can actually read and review — branches, joins, retries — or when you want a type at every seam.

---

---

*Versions: crewai 1.15.21; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
