# Pydantic AI vs CrewAI

**CrewAI, at its best:** the orchestration DSL — agents with roles, goals, and backstories, crews
run by `process='sequential'` or `'hierarchical'`, plus knowledge sources and memory.

**Pydantic AI, at its best:** orchestration-as-code — chained and fanned-out agents with a type at
every seam, plus the production checklist (deps, budgets, cancellation, evals).

*Verified against `crewai 1.15.21` (2026-09-10). Pydantic AI claims below are self-contained scripts
— offline, no API keys — re-executed by this repository's test suite.*

## Quick comparison

| What you get | CrewAI | Pydantic AI |
|---|---|---|
| Orchestration | A DSL: `Agent(role=..., goal=..., backstory=...)`, `Crew(process=...)` | Plain async code: chain agents, fan out with `asyncio.gather`, branch with normal control flow |
| Typed seams | `inputs` dicts; output models via Pydantic | `deps_type` boundary; typed tool signatures and returns at every seam (proven below) |
| Loop access | A `kickoff()` you run; stopping = kill your thread | Typed cancellation + resumable history; `iter()` drives the loop node-by-node |
| Memory/knowledge | Built-in memory + `StringKnowledgeSource` | Your memory: deps + history processors are yours to wire |
| Evals | Not first-party in the core loop | Typed datasets + evaluators, CI-runnable offline |

## Prove it yourself

A "crew" is a fan-out, a join, and a downstream agent — as ordinary async code with a type between
each step:

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

The orchestration is the code — visible, debuggable, and subject to your normal review — not a DSL
the framework interprets.

## Key differences

**Their best:** the role DSL compresses common multi-agent shapes into a few lines, and memory +
knowledge are included — real batteries for the crew-of-specialists pattern.

**Ours:** orchestration-as-code keeps every step typed and inspectable (the fan-out returns ints, the
sum flows into deps), and the production checklist applies: the same runs are cancellable,
budgeted, and testable offline.

## When to choose CrewAI

You want the crew-of-roles pattern with memory and knowledge shipped — and your orchestration fits
their sequential/hierarchical processes.

## When to choose Pydantic AI

Your multi-agent flow has branches, joins, or retries you need to see and control — or you want
typed seams and the production checklist on top of the same fan-out shape.

## Summary

Their crew is a DSL the framework interprets; ours is code with types at the seams. The fan-out
returned ints 21 and 22; the downstream agent received the sum (43) as deps.

*CrewAI behavior pinned to 1.15.21 (installed, construction-probed); records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI verified
on 2.42.0, 2026-09-10.*