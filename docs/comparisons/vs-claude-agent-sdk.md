# Pydantic AI vs Claude Agent SDK

You're choosing a Python agent framework and have narrowed it to [Pydantic AI](../agent.md) and Claude Agent SDK.
This page makes the call — and lets you check the evidence yourself: every snippet runs offline,
no API keys.

## Pydantic AI fits if you need

- the loop **in your process** — typed, cancellable, testable, wrap-able
- a **deps boundary** and budgets that halt before side effects
- **offline tests** that need no subprocess
- Claude models available, plus your choice of any other provider
- durability by wrapping (a subprocess cannot be wrapped)

## Why the answers differ

Theirs is a harness you configure — skills, hooks, permissions for their process. Ours is a value you drive; the proof below runs the loop node by node inside your own PID. The difference is who owns the process.

## See it work

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

## The details

| What you get | Claude Agent SDK | Pydantic AI |
|---|---|---|
|---|---|---|
| Runtime | A `claude` **subprocess** driven over a JSON protocol | The run is a value in your process (proven below) |
| Extension | Config surface — hooks, plugins, skills for *their* harness | Typed capabilities in your code, deferrable + serializable |
| Trusted state | Environment / context handed to the harness | `deps_type` — a boundary the model cannot cross |
| Cancellation | Stop = kill the subprocess | Typed: `CancellationToken` (thread-safe), `ctx.cancel()`, catchable `RunCancelled` with resumable history |
| History | Resume by session id (the CLI's state) | Typed, repairable history you can pass between runs |
| Offline tests | Their harness; stub-level control is theirs to expose | `TestModel` / `FunctionModel` drive the whole pipeline |
| Events | JSON protocol: `SystemMessage` → `AssistantMessage` → `ResultMessage` | Typed stream (part/tool/result/final), transformable by capabilities |
| Cascade | Prompt caching, Claude models, harness default behaviors | You can use Claude models too ([`anthropic` provider](../models/anthropic.md)) — but the loop is yours |

## If this answer doesn't fit you

Easy one: if what you actually need is Claude Code inside your product — the real harness, its skills, its hooks, its defaults — that's theirs, and we can't argue with it. No library gives you Claude Code better than Claude's own SDK does. We can only argue the other direction: if the loop needs to live in your code, typed and testable, no harness gives you that either. Claude models work fine here too.

---

---

*Versions: claude-agent-sdk 0.2.87; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
