---
name: pydantic-ai-harness
description: Extend Pydantic AI agents with batteries-included capabilities from pydantic-ai-harness -- Code Mode (collapse many tool calls into one sandboxed Python execution), a filesystem and shell, sub-agents, planning, context compaction, and more. Use when the user mentions pydantic-ai-harness, CodeMode, Monty, code mode, or tool sandboxing, when they want first-party filesystem/shell/sub-agent/planning/compaction capabilities for a Pydantic AI agent, when they want an agent to run agent-written Python, or when a Pydantic AI agent would benefit from orchestrating multiple tool calls in a single sandboxed script.
license: MIT
compatibility: Requires Python 3.10+ and pydantic-ai-slim>=2.1.0
metadata:
  version: "0.1.0"
  author: pydantic
---

# Building with Pydantic AI Harness

Pydantic AI Harness is the official capability library for Pydantic AI. Capabilities that need model or
framework support -- and those fundamental to every agent -- live in core `pydantic-ai`; optional,
batteries-included capabilities live here. Both are composed onto an agent through the same
`capabilities=[...]` API.

This skill covers the capabilities shipped by `pydantic-ai-harness`. For the core framework -- agents,
tools, structured output, hooks, and testing -- use the `building-pydantic-ai-agents` skill instead.

## When to Use This Skill

Invoke this skill when:
- The user mentions `pydantic-ai-harness`, `CodeMode`, code mode, or the Monty sandbox
- An agent makes many sequential tool calls that could collapse into one sandboxed Python execution
- The user wants the model to write Python that loops, branches, aggregates, or parallelizes tool calls with `asyncio.gather`
- The user asks to sandbox or constrain the code an agent runs

Do **not** use this skill for:
- Core Pydantic AI usage -- building agents, adding tools, structured output, streaming, or testing (use `building-pydantic-ai-agents`)
- Capabilities that ship in core `pydantic-ai`, such as web search, tool search, and thinking
- The Pydantic validation library on its own (`pydantic`/`BaseModel` without agents)

## Supported Capabilities

`CodeMode` has a full reference below; it is the flagship capability and the one this skill goes deep on.
The rest ship today and each has its own README with API and examples.

**Stable** (imported from `pydantic_ai_harness`):

| Capability | Description | Reference |
|---|---|---|
| `CodeMode` | Wraps eligible tools into a single sandboxed `run_code` tool so the model orchestrates them in Python | [Code Mode](./references/CODE-MODE.md) |
| `FileSystem` | Read, write, edit, and search files under a root directory, with traversal prevention | [README](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/filesystem) |
| `Shell` | Run commands in a subprocess with allowlists, a default denylist, timeouts, and env masking | [README](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/shell) |
| `ManagedPrompt` | Back an agent's instructions with a Logfire-managed prompt | [README](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire) |

**Experimental** (imported from `pydantic_ai_harness.experimental`, may change in any release): `SubAgents`,
`Planning`, a compaction family (`SlidingWindow`, `SummarizingCompaction`, ...), `OverflowingToolOutput`,
`RepoContext`, `StepPersistence`, `PyaiDocs`, `RuntimeAuthoring`, and an ACP server adapter.

The full, current list with links and status is in the
[capability matrix](https://github.com/pydantic/pydantic-ai-harness#capability-matrix).

## Install

```bash
uv add pydantic-ai-harness
```

Each capability declares its own extra. Code Mode needs the Monty sandbox:

```bash
uv add "pydantic-ai-harness[codemode]"   # `code-mode` is also accepted as an alias
```

Requires Python 3.10+ and `pydantic-ai-slim>=2.1.0`.

## Quick Start

A harness capability is added to the agent like any other. Here `CodeMode` wraps an MCP server's tools into
a single `run_code` tool that the model drives with Python.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP  # MCP ships in core pydantic-ai

from pydantic_ai_harness import CodeMode

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        # native=False routes the MCP tools through a local toolset so CodeMode can wrap them.
        # Without it, providers with native MCP run the tools server-side and bypass the sandbox.
        MCP('https://hn.caseyjhand.com/mcp', native=False),
        CodeMode(),
    ],
)

result = agent.run_sync(
    'Across the top and best Hacker News feeds, find the most-discussed story with at '
    'least 100 points and summarize its comment thread in one paragraph.'
)
print(result.output)
#> The most-discussed story clearing 100 points is ...
```

Instead of one model round-trip per tool call, the model writes a single Python script that fetches both
feeds with `asyncio.gather`, dedupes and ranks them in plain Python, and pulls the winning thread --
collapsing many calls into one `run_code`.

## Key Practices

- **Confirm a harness capability is actually needed.** If core Pydantic AI tools and capabilities are enough, use the `building-pydantic-ai-agents` skill instead -- don't reach for the harness by default.
- **Read the reference before writing code.** Each capability has its own configuration, constraints, and gotchas -- load the linked reference (e.g. [Code Mode](./references/CODE-MODE.md)) first.
- **Install the capability's extra.** Importing `CodeMode` without `pydantic-ai-harness[codemode]` raises an `ImportError`; the Monty sandbox is an optional dependency.

## Common Gotchas

- **`native=True` tools bypass `CodeMode`.** Provider-native MCP servers and web search execute server-side, so `run_code` never sees them. Construct them with `native=False` to keep them local and wrappable.
- **The Monty sandbox is a Python subset.** No class definitions, no third-party imports, and only a small stdlib allowlist -- read [Code Mode](./references/CODE-MODE.md#sandbox-restrictions) before debugging generated code that fails to run.
- **`CodeMode` needs its extra.** Install `pydantic-ai-harness[codemode]`, not the bare package.
