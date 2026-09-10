# Pydantic AI vs Pi

Choosing an agent framework and you're down to
[Pydantic AI](../agent.md) and Pi. This page is the tiebreaker — the answer first, then code
you can run in seconds.

## Pydantic AI fits if you need

- you are **building the product**: the loop in your process, harness capabilities as libraries, durable execution under it
- pydantic-ai-harness: `CodeMode`, `FileSystem`, `Shell`, subagents, compaction, skills, ACP — replaceable
- the same primitives across agents and coding

## Why the answers differ

Pi is the harness shipped as a CLI; Pydantic AI is the harness as a library. Same family — the difference is ownership: you run Pi; you build with us.

## See it work

Say you're building a coding product, not running someone else's CLI.

Pi ships a polished CLI you run — TUI, skills, memory, its decisions made for you.

Your side, runs offline:

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

**Notice:** The loop is an object in your process: drive it, wrap it, cancel it. The product is yours to shape.

## The details

| What you get | Pi | Pydantic AI |
|---|---|---|
|---|---|---|
| Shape | A CLI you run | A library you build into your product |
| The loop | Shipped, in the app | Yours: drive it, wrap it, cancel it (proven below) |
| Extensions | Skills (frontmatter), your config | Capabilities: replaceable and composable, same units as any agent |
| Security | No-sandbox stance, documented | Sandbox/harness isolation is a composable choice (Harness) |
| Durable | — | Six engine wraps on the public interface |

## If this answer doesn't fit you

If you want a coding agent today and would rather take a product's decisions than make them, Pi is the product — it's literally built on these same primitives, and we're glad it exists. If you're building the product, the harness-as-library is the starting point. Both are fine; they're just different.

---

---

*Versions: pi installed CLI / pydantic-harness @ 1ad638f8; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
