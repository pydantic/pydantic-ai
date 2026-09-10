# Pydantic AI vs Pi

**Pi** is a shipped coding agent — a polished CLI (TUI, skills, memory, provider
catalog) built on the same harness primitives, with a deliberate no-sandbox security stance.

**Pydantic AI / pydantic-ai-harness** is the harness as a library — `CodeMode` (Monty),
`FileSystem`, `Shell`, subagents, compaction, skills, memory, and ACP, each a replaceable
capability, on the typed, durable loop.

*Verified against pi (installed CLI) and pydantic-harness @ `1ad638f8` (2026-09-10). Pydantic AI
claims below are self-contained scripts — offline, no API keys — re-executed by this repository's
test suite.*

## Quick comparison

| What you get | Pi | Pydantic AI |
|---|---|---|
| Shape | A CLI you run | A library you build into your product |
| The loop | Shipped, in the app | Yours: drive it, wrap it, cancel it (proven below) |
| Extensions | Skills (frontmatter), your config | Capabilities: replaceable and composable, same units as any agent |
| Security | No-sandbox stance, documented | Sandbox/harness isolation is a composable choice (Harness) |
| Durable | — | Six engine wraps on the public interface |

## Prove it yourself

The loop that a CLI ships is an object in your process when you need it to be:

```python {title="in_process_loop.py"}
"""The loop is an object in your process — not a harness you configure.

No subprocess, no CLI contract: the run is a value you drive node by node,
and everything runs in your pid. You keep your code, your exits, your
libraries around the loop; nothing is spawned to run it.
"""
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

pid = os.getpid()


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('twice', {'n': 21})])
    return ModelResponse(parts=[TextPart('42')])


agent = Agent(FunctionModel(model))


@agent.tool
def twice(ctx, n: int) -> int:
    return n * 2


async def main():
    nodes = []
    async with agent.iter('what is 21*2?') as run:
        async for node in run:
            nodes.append(type(node).__name__)
            await asyncio.sleep(0)  # ordinary Python between nodes
    print('nodes:', ' -> '.join(nodes))
    print('the loop ran in your own process:', pid == os.getpid())


asyncio.run(main())
```

```text
nodes: UserPromptNode -> ModelRequestNode -> CallToolsNode -> ModelRequestNode -> CallToolsNode -> End
the loop ran in your own process: True
```

Pi is a product built on these primitives; the harness is the primitives themselves, ran anywhere.

## Key differences

**Pi.** pi is a genuinely good product — TUI, skills, memory, provider compatibility table,
a no-sandbox stance stated plainly.

**Pydantic AI.** the same affordances as a rewirable, typed library on a durable loop. Any capability a CLI
hides is one you can replace; any step it takes is one you can observe, cancel, or make durable.

## When to choose Pi

You want a working coding agent today, CLI-first, and you'll take the product's decisions.

## When to choose Pydantic AI (or pydantic-ai-harness)

You are building the product: the loop in your process, harness capabilities as libraries, durable
execution under it, and the same primitives across agents and coding.

## Summary

Pi is the harness shipped as a CLI; Pydantic AI is the harness as a library. The loop ran in your
process: True.

*Pi behavior pinned to installed CLI; harness pinned to `1ad638f8`; records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI verified
on 2.42.0, 2026-09-10.*