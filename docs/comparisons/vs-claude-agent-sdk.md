# Pydantic AI vs Claude Agent SDK

**Claude Agent SDK** is the harness that powers Claude Code as a library — you configure
the `claude` process (skills, hooks, permissions, substitutions) and drive it over a subprocess
protocol.

**Pydantic AI** is the loop is an object in *your* process — typed, testable,
cancellable, wrap-able — and the same seams (deps, capabilities, cancellation, evals) that work for
any provider.

*Verified against `claude-agent-sdk 0.2.87` / `claude` CLI (2026-09-10). Pydantic AI claims below are
self-contained scripts — offline, no API keys — re-executed by this repository's test suite.*

## Quick comparison

| What you get | Claude Agent SDK | Pydantic AI |
|---|---|---|
| Runtime | A `claude` **subprocess** driven over a JSON protocol | The run is a value in your process (proven below) |
| Extension | Config surface — hooks, plugins, skills for *their* harness | Typed capabilities in your code, deferrable + serializable |
| Trusted state | Environment / context handed to the harness | `deps_type` — a boundary the model cannot cross |
| Cancellation | Stop = kill the subprocess | Typed: `CancellationToken` (thread-safe), `ctx.cancel()`, catchable `RunCancelled` with resumable history |
| History | Resume by session id (the CLI's state) | Typed, repairable history you can pass between runs |
| Offline tests | Their harness; stub-level control is theirs to expose | `TestModel` / `FunctionModel` drive the whole pipeline |
| Events | JSON protocol: `SystemMessage` → `AssistantMessage` → `ResultMessage` | Typed stream (part/tool/result/final), transformable by capabilities |
| Cascade | Prompt caching, Claude models, harness default behaviors | You can use Claude models too ([`anthropic` provider](../models/anthropic.md)) — but the loop is yours |

## Prove it yourself

Two claims, one script. The run has node structure you can drive — and it executes in your pid, not
in a child process you configure and kill.

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

That's the whole difference in one word: **ownership**. Theirs is a harness you configure; ours is a
value you drive. You keep your signals, your exits, your profilers, your process supervision around
the loop — nothing is spawned to run the agent.

## Key differences

**Claude Agent SDK.** the product is real — Claude Code's skills, hooks, and subprocess isolation are
battle-tested behaviors, and prompt caching with their models is a first-party advantage.

**Pydantic AI.** everything meaningful about the run is inspectable and changeable from your code: typed
deps, budgets that halt before side effects, cancellation that is a catchable, resumable exception,
evals in CI, and durability by wrapping — none of which exist for a loop you can't reach into.

## When to choose Claude Agent SDK

Your agent *is* the Claude Code process: you want its skills, hooks, and default behaviors, or you
are building for Studio/IDE surfaces and want the subprocess isolation it gives you.

## When to choose Pydantic AI

You want the loop in your process with typed seams and your choice of provider, models included.
The harness boundaries exist for exactly this reason: a CLI is great until you need to read the run.

## Summary

They ship a harness you configure; we ship a loop you drive. Same family of models available either
way — the difference is who owns the process.

*Claude SDK behavior pinned to 0.2.87; probe records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI behavior
verified on 2.42.0, 2026-09-10.*