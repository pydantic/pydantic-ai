# Pydantic AI vs Pi

Choosing an agent framework and you're down to [Pydantic AI](../agent.md) and Pi? Pi is the coding
agent from the pi (earendil) project — and yes, it's a library too: the same package that ships the
CLI exports an embeddable core (`createAgentSession`, `createAgentSessionRuntime`,
`createCodingTools`, extensions — verified in 0.85.1). So this page is not "CLI vs library"; both
sides embed. The real differences are the language, and what sits underneath the loop.

## Pydantic AI fits if you need

- the loop **in your own process**, typed and cancellable, on a general framework — not a session
  runtime built for a coding CLI
- a **Python** stack (Pi's core is TypeScript/Node)
- coding capabilities as **composeable units** — `CodeMode` (Monty), `FileSystem`, `Shell`,
  subagents, compaction, skills, memory, ACP — each replaceable, on the same capability model as
  every other agent
- what a shipped product needs underneath: typed deps, budgets, resumable cancellation, **evals in
  CI**, specs, and **durable execution** wraps (Temporal, DBOS, Prefect, Restate, Kitaru, Airflow)

## Why the answers differ

Pi's core exists to power a coding agent: sessions, compaction, code tools, extensions — opinionated
and well-built, in TypeScript. Ours is the general framework, with the coding edition (the harness)
composed from replaceable capabilities on the same typed loop. Same idea, different center of
gravity: theirs is the coding session; ours is the loop and everything you can hang off it.

## See it work

Say you're building a product on an agent core, and the loop should run inside your own process.

In Pi, the core is TypeScript — `createAgentSession({...})` plus `createCodingTools()` from the
same published package (0.85.1); what you embed is the session engine the CLI uses.

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

**Notice:** both embed. Here the loop is Python, node by node, in your PID — and underneath it sit
the seams a shipped product keeps needing: cancellation that resumes, budgets, evals, durable
engine wraps. That's the difference the CLI-shaped core doesn't give you by itself.

## The details

| What you get | Pi | Pydantic AI |
|---|---|---|
| Language | TypeScript/Node | Python 3.10+ |
| Embeddable core | `createAgentSession` + coding tools + extensions (0.85.1) | pydantic-ai-harness: capability library |
| Loop underneath | Their session runtime | pydantic-ai: typed deps, budgets, typed cancellation |
| Coding capabilities | Opinionated session + tools + compaction | `CodeMode`, `FileSystem`, `Shell`, skills, memory, ACP — replaceable |
| Durability | Sessions/compaction (their model) | Engine wraps: Temporal, DBOS, Prefect, Restate, Kitaru, Airflow |
| Evals / specs | Not first-party there | Typed datasets + evaluators in CI; `AgentSpec` |
| Product | CLI (TUI/print/RPC) + core, same package | Harness-as-library docs; sandbox isolation a composable choice |

## If this answer doesn't fit you

If your product is TypeScript — or you want Pi's opinionated session-and-compaction stack as its
foundation, exactly as it powers the CLI — Pi's core is a real path, and the package's docs
(`docs/`, `examples/`) show embedding it. If you're building on Python, or you want the general
framework underneath the coding layer (typed, cancellable, budgeted, durable, evals included) with
coding capabilities composed in, that's the harness. Both directions are fine; they're just
different centers of gravity.

---

*Versions: pi 0.85.1 (exports verified), pydantic-harness @ 1ad638f8; Pydantic AI 2.42.0 — 2026-09-10.
Snippets re-executed by this repository's tests.*
