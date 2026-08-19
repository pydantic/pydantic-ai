---
title: Pydantic AI Harness
description: "Your agent's favorite harness, built on Pydantic AI: 30+ capabilities and complete agents assembled from them, from a coding agent to your own custom stack."
---

# Pydantic AI Harness

*Your agent's favorite harness, built on Pydantic AI*

[![Join Slack](https://img.shields.io/badge/Slack-Join%20Slack-4A154B?logo=slack)](https://logfire.pydantic.dev/docs/join-slack/)

**Pydantic AI Harness** is the official [capability](/ai/capabilities/overview/) and harness library for [Pydantic AI](/ai/). Every Pydantic AI agent already has a light harness: the typed agent loop, [any model](/ai/models/overview/), your own tools, structured output. For simple agents that's enough. But set an agent loose on complex, long-running work (fix a codebase, research a question, run for hours unattended) and what it needs around the model grows: a [workspace](filesystem.md) to act in, a [plan](planning.md) it keeps current, [memory](memory.md) that carries across sessions, [sub-agents](subagents.md) to hand work to, [context management](compaction.md) that holds up in hour ten, and [durable execution](/ai/capabilities/durable_execution/overview/) that survives a restart. **Pydantic AI Harness** ships that harness.

Everything here is one primitive: a [capability](/ai/capabilities/overview/), a self-contained unit of agent behavior you add to `capabilities=[...]` on any agent. There are [30+ of them](#capabilities), and complete agents like [Coder](coder.md) and [Researcher](researcher.md) are themselves capabilities combined: they come apart the way they went together. Snap on a single block, compose your own stack, or start from the whole coding agent and take it apart later.

## Quick start

Install with [`uv`](https://docs.astral.sh/uv/):

```bash
uv add "pydantic-ai-harness[anthropic]"
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder()])

result = agent.run_sync('Find out why tests/test_parser.py fails and fix the bug it caught.')
print(result.output)
#> Found it: `parse()` returned None on empty input instead of raising. Fixed in src/parser.py; tests pass now.
```

That's a complete [coding agent](coder.md): [workspace-rooted file access](filesystem.md), [allowlisted shell](shell.md), [repo orientation](repo-context.md), [planning](planning.md), a read-only [explorer sub-agent](subagents.md), and [context management](compaction.md) that survives long sessions, and it runs anywhere a Pydantic AI agent runs. [`agent.to_cli_sync()`](/ai/cli/) opens it as a chat in your terminal, [`agent.to_web()`](/ai/web/) in the browser, and [`Coder`](coder.md)'s exported [`coder_agent`](coder.md#api-reference) runs without writing a file at all, combined with [`clai`](/ai/cli/) (the Pydantic AI CLI) and [`uvx`](https://docs.astral.sh/uv/guides/tools/):

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-fable-5
```

Every model works: swap the string for [any provider's](/ai/models/overview/). Need more? Add capabilities to the list; here's the same coder on `gpt-5.6-sol`, with web search and cross-session memory:

```bash
uv add "pydantic-ai-slim[openai]"
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai_harness import Coder, Memory
from pydantic_ai_harness.memory import FileStore

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[
        Coder(),
        WebSearch(),  # look up docs and error messages on the web
        Memory(FileStore('.agent-memory')),  # remembers across sessions
    ],
)
```

[Skills](skills.md) (your `SKILL.md` procedures, loaded on demand; point it at a `skills/` directory and add the `skills` extra), [Web Fetch](/ai/capabilities/web-fetch/), [Guardrails](guardrails.md), and [Dynamic Workflow](dynamic-workflow.md) slot in the same way; the [Coder page](coder.md#not-included-by-default) lists what pairs well.

## No magic: it's capabilities all the way down

`Coder` is not a framework inside the framework; it's a [`CombinedCapability`](/ai/capabilities/custom/) bundling the same blocks you can use directly. This is the exact agent the [Coder page](coder.md)'s exported `coder_agent` gives you, written out block by block:

<!-- Keep this blown-out example in sync across docs/coder.md, docs/index.md, README.md, pydantic_ai_harness/coder/README.md, and examples/coding_agent.py. -->

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import (
    ClearToolResults,
    FileSystem,
    LLM_API_KEY_ENV_PATTERNS,
    Planning,
    RepoContext,
    Shell,
    SubAgent,
    SubAgents,
    ToolOutputLimits,
    WarnNearLimits,
)

allowed_commands = [
    'git', 'rg', 'grep', 'find', 'ls', 'cat', 'sed', 'head', 'tail',
    'python', 'uv', 'pytest', 'ruff', 'make',
]

explorer = SubAgent(
    Agent(
        name='explorer',
        description='Explore the codebase and answer questions without modifying anything',
        instructions='Answer with concrete paths and evidence.',
        capabilities=[
            FileSystem('.', read_only=True),
            RepoContext(workspace_dir=Path('.')),
        ],
    )
)

agent = Agent(
    'anthropic:claude-fable-5',
    name='coder',
    instructions='You are a coding agent built on Pydantic AI.',
    capabilities=[
        FileSystem('.'),  # read/write/edit/search, path-traversal safe
        Shell(  # allowlisted commands, LLM API keys stripped from their environment
            cwd='.',
            allowed_commands=allowed_commands,
            denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
        ),
        RepoContext(workspace_dir=Path('.')),  # loads AGENTS.md/CLAUDE.md + repo structure
        Planning(),  # structured task plans the model maintains
        SubAgents(agents=[explorer], agent_folders=None),  # delegate exploration off the main context
        ClearToolResults(max_fraction=0.7),  # clears old tool results near the limit
        WarnNearLimits(max_context_fraction=0.9),  # warns the model before it hits limits
        ToolOutputLimits(),  # bounds oversized tool results
    ],
)
```

Start from the harness and remove what you don't want, or start from the blocks and build up; both are first-class. Constructor arguments (working directory, command allowlist, window sizes) thread through to the underlying capabilities.

## Capabilities

Every capability is a self-contained unit you drop into `capabilities=[...]`, and they all compose, with each other and with your own. Some come with [`pydantic-ai`](/ai/) itself, the rest with this package; the **Package** column says which. 50+ in all, grouped by what they give your agent:

### Harnesses

Complete agent stacks as regular combined capabilities: one import gives you a working agent, and you can take either apart into the blocks below.

| Harness | Package | What it provides |
|---|---|---|
| [Coder](coder.md) | Harness | A complete coding-agent stack: files, shell, repo context, planning, a read-only explorer sub-agent, and context controls |
| [Researcher](researcher.md) | Harness | A complete web-research stack: search, page fetching, a delegated sub-researcher, and bounded tool output |

### Execution environments

The workspace the agent acts in: the files it edits and the commands it runs, local or isolated.

| Capability | Package | What it does |
|---|---|---|
| [FileSystem](filesystem.md) | Harness | Read, write, edit, search files under a root; path-traversal and symlink safe, secrets read-only |
| [Shell](shell.md) | Harness | Command execution with allowlists, denylists, timeouts, and credential-stripping |
| [Modal Sandbox](modal-sandbox.md) | Harness | Commands and files in an isolated [Modal](https://modal.com) cloud sandbox |

### Tools & native abilities

Connections to systems outside the agent's workspace, and abilities the provider executes natively.

| Capability | Package | What it does |
|---|---|---|
| [MCP](/ai/capabilities/mcp/) | Core | Connect any MCP server's tools; local by default, provider-native connectors opt-in |
| [Image Generation](/ai/capabilities/image-generation/) | Core | Generate and edit images; provider-native where supported, sub-agent fallback elsewhere |
| [StackOne](stackone.md) | Harness | Act on linked SaaS accounts (HRIS, ATS, CRM, …) via [StackOne](https://www.stackone.com) |
| [LocalStack](localstack.md) | Harness | An emulated AWS environment with AWS CLI tools |
| [Macroscope](macroscope.md) | Harness | Run a local [Macroscope](https://docs.macroscope.com/cli) code review and hand the findings to the agent |

### Web & research

Finding and reading things on the open web.

| Capability | Package | What it does |
|---|---|---|
| [Web Search](/ai/capabilities/web-search/) | Core | Provider-native search where available, local DuckDuckGo fallback everywhere |
| [Web Fetch](/ai/capabilities/web-fetch/) | Core | Fetch and read URLs, native or local |
| [X Search](/ai/capabilities/x-search/) | Core | Search X; native on xAI, subagent fallback elsewhere |
| [Exa Search](exa-search.md) | Harness | Web research via [Exa](https://exa.ai): excerpted search, full-page reads, opt-in cited deep search |
| [Exa Agent](exa-search.md) | Harness | Delegate open-ended research to the Exa Agent API |
| [You.com Search](youdotcom.md) | Harness | Web search and page reads via [You.com](https://you.com): query-relevant excerpts or full-page markdown |
| [You.com Research](youdotcom.md) | Harness | Cited answers and multi-step research via the You.com Answer, Research, and Finance Research APIs |
| [Browser Use](browser-use.md) | Harness | Hand web tasks to an autonomous [browser-use](https://github.com/browser-use/browser-use) agent driving a real browser |

### Reasoning, planning & delegation

How the agent thinks and divides the work.

| Capability | Package | What it does |
|---|---|---|
| [Thinking](/ai/capabilities/thinking/) | Core | Provider-adaptive extended thinking at configurable effort |
| [Planning](planning.md) | Harness | Model-owned task plans with a cache-safe live reminder |
| [Subagents](subagents.md) | Harness | Delegate self-contained tasks to named child agents |
| [Dynamic Workflow](dynamic-workflow.md) | Harness | The model orchestrates sub-agents from one Python script: fan-out, chain, vote in a single tool call, with hard `max_agent_calls` budgets |
| [Advisor](advisor.md) | Harness | Let an executor consult a stronger model mid-run |

### Context management

How the agent spends its context window: the difference between an agent that degrades over a long run and one that doesn't, and between paying for tokens N times or once.

| Capability | Package | What it does |
|---|---|---|
| [Code Mode](code-mode.md) | Harness | The model writes one Python script that calls many tools inside a [Monty](https://github.com/pydantic/monty) sandbox: one round-trip instead of N, and intermediate results never enter the context window. The answer to tool-call token bloat |
| [Tool Search](/ai/capabilities/tool-search/) | Core | Load tool definitions on demand instead of carrying hundreds in every prompt |
| [Compaction](/ai/capabilities/compaction/) | Core | Provider-native compaction on OpenAI and Anthropic; the provider summarizes history server-side |
| [Compaction](compaction.md) | Harness | Model-agnostic strategies: tool-result clearing, sliding-window trimming, LLM summarization, tiered; all window-relative, with live usage reporting |
| [Tool Output Limits](tool-output-limits.md) | Harness | Truncate, spill to a queryable file, or summarize oversized tool returns at the source |
| [Warn On Cache Busts](warn-on-cache-busts.md) | Harness | Detect prompt-cache prefix collapses between requests, from the provider's own numbers |

### Knowledge & memory

What the agent knows and remembers, loaded when relevant instead of carried in every prompt.

| Capability | Package | What it does |
|---|---|---|
| [Memory](memory.md) | Harness | A persistent, namespaced notebook: bounded prompt injection, on-demand search; in-memory/file/Postgres stores |
| [Conversation Search](conversation-search.md) | Harness | BM25 search over stored history, including turns compaction dropped |
| [Skills](skills.md) | Harness | Load [Agent Skill](/ai/capabilities/on-demand/) (`SKILL.md`) instructions on demand |
| [Repo Context](repo-context.md) | Harness | Start runs oriented: `AGENTS.md`/`CLAUDE.md` + repository structure |
| [Pydantic AI Docs](pydantic-ai-docs.md) | Harness | On-demand Pydantic AI documentation lookup |

### Control & safety

Bounding what the agent may do, and keeping it on-instructions.

| Capability | Package | What it does |
|---|---|---|
| [Guardrails](guardrails.md) | Harness | Validate/block/redact user input, tool calls, tool results, and output, including secret masking and parallel async guards |
| [Prompt Injection Defender](prompt-injection-defender.md) | Harness | Classify local tool results for indirect prompt injection and optionally withhold high-risk results |
| [Spend Limits](spend.md) | Harness | Cross-window USD/token budgets and per-response cost tracking, per model and per tenant |
| [Tool approval](/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval) | Core | Flag tool calls that need human approval before they run |
| [Handle Deferred Tool Calls](/ai/capabilities/handle-deferred-tool-calls/) | Core | Resolve approval-deferred tool calls programmatically |
| [System Reminders](system-reminders.md) | Harness | Cache-safe re-injection of guidance mid-run to counter instruction fade |

### Self-extension

| Capability | Package | What it does |
|---|---|---|
| [Capability Creation](capability-creation.md) | Harness | The agent writes, validates, and persists *new capabilities* during a run, loaded on the next run: self-extension with typed, inspectable units instead of arbitrary code |

### Execution runtime

Outside the loop: how runs persist, survive failures, and get observed and configured in production.

| Capability | Package | What it does |
|---|---|---|
| [Durable execution](/ai/capabilities/durable_execution/overview/) | Core | Runs that survive restarts and failures on [Temporal](/ai/capabilities/durable_execution/temporal/), [DBOS](/ai/capabilities/durable_execution/dbos/), or [Prefect](/ai/capabilities/durable_execution/prefect/), with [Restate](/ai/capabilities/durable_execution/restate/), [Kitaru](/ai/capabilities/durable_execution/kitaru/), and [Airflow](/ai/capabilities/durable_execution/airflow/) integrations |
| [Step Persistence](step-persistence.md) | Harness | Save, restore, resume (`continue_run`), and fork (`fork_run`) runs; file/SQLite/Mongo backends |
| [Instrumentation](/ai/capabilities/instrumentation/) | Core | OpenTelemetry GenAI spans for every model and tool call; the raw material for [Logfire](https://pydantic.dev/logfire) traces |
| [Managed Prompt](managed-prompt.md) | Harness | Back instructions with a [Logfire](https://pydantic.dev/logfire)-managed prompt; version and roll out without redeploying |
| [Thread Executor](/ai/capabilities/thread-executor/) | Core | Run sync tools on a shared thread pool |

Core also ships loop-customization capabilities for production servers: [Select Model](/ai/capabilities/select-model/), [Resolve Model ID](/ai/capabilities/resolve-model-id/), [Prepare Tools / Prepare Output Tools](/ai/capabilities/prepare-tools/), [Prefix Tools](/ai/capabilities/prefix-tools/), [Set Tool Metadata](/ai/capabilities/set-tool-metadata/), [Include Tool Return Schemas](/ai/capabilities/include-tool-return-schemas/), [Process History](/ai/capabilities/process-history/), [Process Event Stream](/ai/capabilities/process-event-stream/), [Reinject System Prompt](/ai/capabilities/reinject-system-prompt/), and [Raise Content Filter Error](/ai/capabilities/raise-content-filter-error/).

And the agent plugs into any interface: [ACP](acp.md) *(experimental, Harness)* serves it to editors like Zed over the [Agent Client Protocol](https://agentclientprotocol.com), and core ships the [web chat UI](/ai/web/), [CLI](/ai/cli/), [frontend adapters](/ai/ui/overview/) (AG-UI, Vercel AI), and [realtime voice](/ai/realtime/overview/).

Community packages extend the same capability system further; see [third-party capabilities](/ai/capabilities/third-party/).

## When do you need the Harness?

"Harness" is the field's term for everything around the model that turns it into an agent: the loop, the tools, the context management. Reach for this package when your agent should *do* more than core's lean harness covers: touch files, run code, browse, remember, delegate, or stay coherent through hours-long runs. The boundary between the packages is mechanical, not a maturity tier: core ships the capabilities that require model or framework support (provider-native tools like [image generation](/ai/capabilities/image-generation/), provider APIs like [compaction](/ai/capabilities/compaction/), deep loop integration like [tool search](/ai/capabilities/tool-search/), and fundamentals like [thinking](/ai/capabilities/thinking/), [MCP](/ai/capabilities/mcp/), and [web search](/ai/capabilities/web-search/)) and the Harness ships everything else, as a separate package so capabilities can iterate at the speed the field moves while Pydantic AI itself stays lean.

## Installation

```bash
uv add pydantic-ai-harness
```

This installs [`pydantic-ai-slim`](/ai/install/) with it, so it works on its own; you don't need to install Pydantic AI separately. Model providers and the CLI come via extras that pass through to Pydantic AI: `pydantic-ai-harness[anthropic]`, `[cli]`. Some capabilities need their own extra for optional dependencies; each capability's page gives its exact install line. Requires Python 3.10+.

New to Pydantic AI itself? Start with [its docs](/ai/): the agent you mount these capabilities on is defined there.

## Observability

Everything the harness does is observable: core's [Instrumentation](/ai/capabilities/instrumentation/) capability (or `logfire.instrument_pydantic_ai()`) emits a full trace of every run: every model call and tool call, with token and cost tracking. It's standard OpenTelemetry, so any OTLP backend works; [Logfire](https://pydantic.dev/logfire) is the easiest way to see it during development.

## Build your own

[Capabilities](/ai/capabilities/custom/) are the primary extension point for Pydantic AI, and every capability in this library doubles as a worked example. Publishing a standalone package? Use the `pydantic-ai-<name>` naming convention; see [Publishing capability packages](/ai/guides/extensibility/#publishing-capability-packages).

## Version policy

Pydantic AI Harness uses **0.x versioning**, and that's a statement about API stability, not maturity: these capabilities are tested end-to-end and meant for production use, but their APIs may still move between minor releases (0.1 -> 0.2): renamed parameters, changed defaults, restructured APIs, always with deprecation warnings where practical. Patch releases will not intentionally break existing behavior, and every breaking change is documented in release notes with migration guidance your agent can follow. Keeping the Harness a separate package from [Pydantic AI](https://github.com/pydantic/pydantic-ai), which has a [stricter version policy](/ai/project/version-policy/), is what lets capabilities iterate at the speed the field moves.

## Pydantic AI references

- [Capabilities](/ai/capabilities/overview/): what capabilities are, built-in capabilities, building your own
- [Hooks](/ai/core-concepts/hooks/): lifecycle hooks reference, ordering, error handling
- [Extensibility](/ai/guides/extensibility/): publishing packages, third-party ecosystem
- [Toolsets](/ai/tools-toolsets/toolsets/): building tools for capabilities
- [API reference](/ai/api/pydantic-ai/capabilities/): full API docs
