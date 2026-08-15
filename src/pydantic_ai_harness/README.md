# Pydantic AI Harness

[![CI](https://github.com/pydantic/pydantic-ai-harness/actions/workflows/main.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai-harness/actions/workflows/main.yml?query=branch%3Amain)
[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-harness.svg)](https://pypi.python.org/pypi/pydantic-ai-harness)
[![versions](https://img.shields.io/pypi/pyversions/pydantic-ai-harness.svg)](https://github.com/pydantic/pydantic-ai-harness)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai-harness.svg)](https://github.com/pydantic/pydantic-ai-harness/blob/main/LICENSE)
[![Join Slack](https://img.shields.io/badge/Slack-Join%20Slack-4A154B?logo=slack)](https://logfire.pydantic.dev/docs/join-slack/)

**Your agent's favorite harness, built on Pydantic AI**

---

**Pydantic AI Harness** is the official [capability](https://ai.pydantic.dev/capabilities/overview/) and harness library for [Pydantic AI](https://ai.pydantic.dev/). Every Pydantic AI agent already has a light harness: the typed agent loop, [any model](https://ai.pydantic.dev/models/), your own tools, structured output. For simple agents that's enough. But set an agent loose on complex, long-running work (fix a codebase, research a question, run for hours unattended) and what it needs around the model grows: a [workspace](pydantic_ai_harness/filesystem/) to act in, a [plan](pydantic_ai_harness/planning/) it keeps current, [memory](pydantic_ai_harness/memory/) that carries across sessions, [sub-agents](pydantic_ai_harness/subagents/) to hand work to, [context management](pydantic_ai_harness/compaction/) that holds up in hour ten, and [durable execution](https://ai.pydantic.dev/capabilities/durable_execution/overview/) that survives a restart. **Pydantic AI Harness** ships that harness.

Everything here is one primitive: a [capability](https://ai.pydantic.dev/capabilities/), a self-contained unit of agent behavior you add to `capabilities=[...]` on any agent. There are [30+ of them](#capabilities), and complete agents like [Coder](pydantic_ai_harness/coder/) and [Researcher](pydantic_ai_harness/researcher/) are themselves capabilities combined: they come apart the way they went together. Snap on a single block, compose your own stack, or start from the whole coding agent and take it apart later.

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

That's a complete [coding agent](pydantic_ai_harness/coder/): [workspace-rooted file access](pydantic_ai_harness/filesystem/), [allowlisted shell](pydantic_ai_harness/shell/), [repo orientation](pydantic_ai_harness/repo_context/), [planning](pydantic_ai_harness/planning/), a read-only [explorer sub-agent](pydantic_ai_harness/subagents/), and [context management](pydantic_ai_harness/compaction/) that survives long sessions, and it runs anywhere a Pydantic AI agent runs. [`agent.to_cli_sync()`](https://ai.pydantic.dev/cli/) opens it as a chat in your terminal, [`agent.to_web()`](https://ai.pydantic.dev/web/) in the browser, and [`Coder`](pydantic_ai_harness/coder/)'s exported `coder_agent` runs without writing a file at all, combined with [`clai`](https://ai.pydantic.dev/cli/) (the Pydantic AI CLI) and [`uvx`](https://docs.astral.sh/uv/guides/tools/):

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-fable-5
```

Every model works: swap the string for [any provider's](https://ai.pydantic.dev/models/). Need more? Add capabilities to the list; here's the same coder on `gpt-5.6-sol`, with web search and cross-session memory:

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

[Skills](pydantic_ai_harness/skills/) (your `SKILL.md` procedures, loaded on demand; point it at a `skills/` directory and add the `skills` extra), [Web Fetch](https://ai.pydantic.dev/capabilities/web-fetch/), [Guardrails](pydantic_ai_harness/guardrails/), and [Dynamic Workflow](pydantic_ai_harness/dynamic_workflow/) slot in the same way; the [Coder README](pydantic_ai_harness/coder/) lists what pairs well.

## No magic: it's capabilities all the way down

`Coder` is not a framework inside the framework; it's a [`CombinedCapability`](https://ai.pydantic.dev/capabilities/custom/) bundling the same blocks you can use directly. This is the exact agent the exported [`coder_agent`](pydantic_ai_harness/coder/) gives you, written out block by block:

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

Every capability is a self-contained unit you drop into `capabilities=[...]`, and they all compose, with each other and with your own. Some come with [`pydantic-ai`](https://github.com/pydantic/pydantic-ai) itself, the rest with this package; the **Package** column says which. 50+ in all, grouped by what they give your agent:

### Harnesses

Complete agent stacks as regular combined capabilities: one import gives you a working agent, and you can take either apart into the blocks below.

| Harness | Package | What it provides |
|---|---|---|
| [Coder](pydantic_ai_harness/coder/) | Harness | A complete coding-agent stack: files, shell, repo context, planning, a read-only explorer sub-agent, and context controls |
| [Researcher](pydantic_ai_harness/researcher/) | Harness | A complete web-research stack: search, page fetching, a delegated sub-researcher, and bounded tool output |

### Execution environments

The workspace the agent acts in: the files it edits and the commands it runs, local or isolated.

| Capability | Package | What it does |
|---|---|---|
| [FileSystem](pydantic_ai_harness/filesystem/) | Harness | Read, write, edit, search files under a root; path-traversal and symlink safe, secrets read-only |
| [Shell](pydantic_ai_harness/shell/) | Harness | Command execution with allowlists, denylists, timeouts, and credential-stripping |
| [Modal Sandbox](pydantic_ai_harness/modal_sandbox/) | Harness | Commands and files in an isolated [Modal](https://modal.com) cloud sandbox |

### Tools & native abilities

Connections to systems outside the agent's workspace, and abilities the provider executes natively.

| Capability | Package | What it does |
|---|---|---|
| [MCP](https://ai.pydantic.dev/capabilities/mcp/) | Core | Connect any MCP server's tools; local by default, provider-native connectors opt-in |
| [Image Generation](https://ai.pydantic.dev/capabilities/image-generation/) | Core | Generate and edit images; provider-native where supported, sub-agent fallback elsewhere |
| [StackOne](pydantic_ai_harness/stackone/) | Harness | Act on linked SaaS accounts (HRIS, ATS, CRM, …) via [StackOne](https://www.stackone.com) |
| [LocalStack](pydantic_ai_harness/localstack/) | Harness | An emulated AWS environment with AWS CLI tools |
| [Macroscope](pydantic_ai_harness/macroscope/) | Harness | Run a local [Macroscope](https://docs.macroscope.com/cli) code review and hand the findings to the agent |

### Web & research

Finding and reading things on the open web.

| Capability | Package | What it does |
|---|---|---|
| [Web Search](https://ai.pydantic.dev/capabilities/web-search/) | Core | Provider-native search where available, local DuckDuckGo fallback everywhere |
| [Web Fetch](https://ai.pydantic.dev/capabilities/web-fetch/) | Core | Fetch and read URLs, native or local |
| [X Search](https://ai.pydantic.dev/capabilities/x-search/) | Core | Search X; native on xAI, subagent fallback elsewhere |
| [Exa Search](pydantic_ai_harness/exa/) | Harness | Web research via [Exa](https://exa.ai): excerpted search, full-page reads, opt-in cited deep search |
| [Exa Agent](pydantic_ai_harness/exa/) | Harness | Delegate open-ended research to the Exa Agent API |
| [Browser Use](pydantic_ai_harness/browser_use/) | Harness | Hand web tasks to an autonomous [browser-use](https://github.com/browser-use/browser-use) agent driving a real browser |

### Reasoning, planning & delegation

How the agent thinks and divides the work.

| Capability | Package | What it does |
|---|---|---|
| [Thinking](https://ai.pydantic.dev/capabilities/thinking/) | Core | Provider-adaptive extended thinking at configurable effort |
| [Planning](pydantic_ai_harness/planning/) | Harness | Model-owned task plans with a cache-safe live reminder |
| [Subagents](pydantic_ai_harness/subagents/) | Harness | Delegate self-contained tasks to named child agents |
| [Dynamic Workflow](pydantic_ai_harness/dynamic_workflow/) | Harness | The model orchestrates sub-agents from one Python script: fan-out, chain, vote in a single tool call, with hard `max_agent_calls` budgets |
| [Advisor](pydantic_ai_harness/advisor/) | Harness | Let an executor consult a stronger model mid-run |

### Context management

How the agent spends its context window: the difference between an agent that degrades over a long run and one that doesn't, and between paying for tokens N times or once.

| Capability | Package | What it does |
|---|---|---|
| [Code Mode](pydantic_ai_harness/code_mode/) | Harness | The model writes one Python script that calls many tools inside a [Monty](https://github.com/pydantic/monty) sandbox: one round-trip instead of N, and intermediate results never enter the context window. The answer to tool-call token bloat |
| [Tool Search](https://ai.pydantic.dev/capabilities/tool-search/) | Core | Load tool definitions on demand instead of carrying hundreds in every prompt |
| [Compaction](https://ai.pydantic.dev/capabilities/compaction/) | Core | Provider-native compaction on OpenAI and Anthropic; the provider summarizes history server-side |
| [Compaction](pydantic_ai_harness/compaction/) | Harness | Model-agnostic strategies: tool-result clearing, sliding-window trimming, LLM summarization, tiered; all window-relative, with live usage reporting |
| [Tool Output Limits](pydantic_ai_harness/tool_output_limits/) | Harness | Truncate, spill to a queryable file, or summarize oversized tool returns at the source |
| [Warn On Cache Busts](pydantic_ai_harness/warn_on_cache_busts/) | Harness | Detect prompt-cache prefix collapses between requests, from the provider's own numbers |

### Knowledge & memory

What the agent knows and remembers, loaded when relevant instead of carried in every prompt.

| Capability | Package | What it does |
|---|---|---|
| [Memory](pydantic_ai_harness/memory/) | Harness | A persistent, namespaced notebook: bounded prompt injection, on-demand search; in-memory/file/Postgres stores |
| [Conversation Search](pydantic_ai_harness/conversation_search/) | Harness | BM25 search over stored history, including turns compaction dropped |
| [Skills](pydantic_ai_harness/skills/) | Harness | Load [Agent Skill](https://ai.pydantic.dev/capabilities/on-demand/) (`SKILL.md`) instructions on demand |
| [Repo Context](pydantic_ai_harness/repo_context/) | Harness | Start runs oriented: `AGENTS.md`/`CLAUDE.md` + repository structure |
| [Pydantic AI Docs](pydantic_ai_harness/pydantic_ai_docs/) | Harness | On-demand Pydantic AI documentation lookup |

### Control & safety

Bounding what the agent may do, and keeping it on-instructions.

| Capability | Package | What it does |
|---|---|---|
| [Guardrails](pydantic_ai_harness/guardrails/) | Harness | Validate/block/redact user input, tool calls, tool results, and output, including secret masking and parallel async guards |
| [Spend Limits](pydantic_ai_harness/spend/) | Harness | Cross-window USD/token budgets and per-response cost tracking, per model and per tenant |
| [Tool approval](https://ai.pydantic.dev/deferred-tools#human-in-the-loop-tool-approval) | Core | Flag tool calls that need human approval before they run |
| [Handle Deferred Tool Calls](https://ai.pydantic.dev/capabilities/handle-deferred-tool-calls/) | Core | Resolve approval-deferred tool calls programmatically |
| [System Reminders](pydantic_ai_harness/system_reminders/) | Harness | Cache-safe re-injection of guidance mid-run to counter instruction fade |

### Self-extension

| Capability | Package | What it does |
|---|---|---|
| [Capability Creation](pydantic_ai_harness/capability_creation/) | Harness | The agent writes, validates, and persists *new capabilities* during a run, loaded on the next run: self-extension with typed, inspectable units instead of arbitrary code |

### Execution runtime

Outside the loop: how runs persist, survive failures, and get observed and configured in production.

| Capability | Package | What it does |
|---|---|---|
| [Durable execution](https://ai.pydantic.dev/capabilities/durable_execution/overview/) | Core | Runs that survive restarts and failures on [Temporal](https://ai.pydantic.dev/capabilities/durable_execution/temporal/), [DBOS](https://ai.pydantic.dev/capabilities/durable_execution/dbos/), or [Prefect](https://ai.pydantic.dev/capabilities/durable_execution/prefect/), with [Restate](https://ai.pydantic.dev/capabilities/durable_execution/restate/), [Kitaru](https://ai.pydantic.dev/capabilities/durable_execution/kitaru/), and [Airflow](https://ai.pydantic.dev/capabilities/durable_execution/airflow/) integrations |
| [Step Persistence](pydantic_ai_harness/step_persistence/) | Harness | Save, restore, resume (`continue_run`), and fork (`fork_run`) runs; file/SQLite/Mongo backends |
| [Instrumentation](https://ai.pydantic.dev/capabilities/instrumentation/) | Core | OpenTelemetry GenAI spans for every model and tool call; the raw material for [Logfire](https://pydantic.dev/logfire) traces |
| [Managed Prompt](pydantic_ai_harness/logfire/) | Harness | Back instructions with a [Logfire](https://pydantic.dev/logfire)-managed prompt; version and roll out without redeploying |
| [Thread Executor](https://ai.pydantic.dev/capabilities/thread-executor/) | Core | Run sync tools on a shared thread pool |

Core also ships loop-customization capabilities for production servers: [Select Model](https://ai.pydantic.dev/capabilities/select-model/), [Resolve Model ID](https://ai.pydantic.dev/capabilities/resolve-model-id/), [Prepare Tools / Prepare Output Tools](https://ai.pydantic.dev/capabilities/prepare-tools/), [Prefix Tools](https://ai.pydantic.dev/capabilities/prefix-tools/), [Set Tool Metadata](https://ai.pydantic.dev/capabilities/set-tool-metadata/), [Include Tool Return Schemas](https://ai.pydantic.dev/capabilities/include-tool-return-schemas/), [Process History](https://ai.pydantic.dev/capabilities/process-history/), [Process Event Stream](https://ai.pydantic.dev/capabilities/process-event-stream/), [Reinject System Prompt](https://ai.pydantic.dev/capabilities/reinject-system-prompt/), and [Raise Content Filter Error](https://ai.pydantic.dev/capabilities/raise-content-filter-error/).

And the agent plugs into any interface: [ACP](pydantic_ai_harness/experimental/acp/) *(experimental, Harness)* serves it to editors like Zed over the [Agent Client Protocol](https://agentclientprotocol.com), and core ships the [web chat UI](https://ai.pydantic.dev/web/), [CLI](https://ai.pydantic.dev/cli/), [frontend adapters](https://ai.pydantic.dev/ui/) (AG-UI, Vercel AI), and [realtime voice](https://ai.pydantic.dev/realtime/).

Community packages extend the same capability system further; see [third-party capabilities](https://ai.pydantic.dev/capabilities/third-party/).

## Composing from blocks

A research agent from regular capabilities -- this is literally [`Researcher`](pydantic_ai_harness/researcher/)'s composition, minus its short default instructions:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai_harness import SubAgent, SubAgents, ToolOutputLimits

sub_researcher = SubAgent(
    Agent(
        name='researcher',
        description='Research a focused sub-question on the web and report back with findings and source links',
        capabilities=[WebSearch(local=True), WebFetch(local=True), ToolOutputLimits()],
    )
)

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        WebSearch(local=True),  # native provider search, DuckDuckGo fallback elsewhere
        WebFetch(local=True),  # read the pages behind the results, native or local
        SubAgents(agents=[sub_researcher], agent_folders=None),
        ToolOutputLimits(),  # fetched pages don't flood the context
    ],
)

result = agent.run_sync('What changed in the top three Python agent frameworks this month? Cite sources.')
print(result.output)
#> ...
```

Everything is observable: `logfire.instrument_pydantic_ai()` gives you [a full trace of every run](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946): every model call and tool call, with token and cost tracking. It's standard OpenTelemetry, so any OTLP backend works; [Logfire](https://pydantic.dev/logfire) is the easiest way to see it during development.

## When do you need the Harness?

"Harness" is the field's term for everything around the model that turns it into an agent: the loop, the tools, the context management. Reach for this package when your agent should *do* more than core's lean harness covers: touch files, run code, browse, remember, delegate, or stay coherent through hours-long runs. The boundary between the packages is mechanical, not a maturity tier: core ships the capabilities that require model or framework support (provider-native tools like [image generation](https://ai.pydantic.dev/capabilities/image-generation/), provider APIs like [compaction](https://ai.pydantic.dev/capabilities/compaction/), deep loop integration like [tool search](https://ai.pydantic.dev/capabilities/tool-search/), and fundamentals like [thinking](https://ai.pydantic.dev/capabilities/thinking/), [MCP](https://ai.pydantic.dev/capabilities/mcp/), and [web search](https://ai.pydantic.dev/capabilities/web-search/)) and the Harness ships everything else, as a separate package so capabilities can iterate at the speed the field moves while Pydantic AI itself stays lean.

## Installation

```bash
uv add pydantic-ai-harness
```

This installs [`pydantic-ai-slim`](https://ai.pydantic.dev/install/) with it, so it works on its own; you don't need to install Pydantic AI separately. Model providers and the CLI come via extras that pass through to Pydantic AI: `pydantic-ai-harness[anthropic]`, `[cli]`. Some capabilities need their own extra for optional dependencies; each capability's page gives its exact install line. Requires Python 3.10+.

## Build your own

[Capabilities](https://ai.pydantic.dev/capabilities/#building-custom-capabilities) are the primary extension point for Pydantic AI, and every capability in this repo doubles as a worked example. Publishing a standalone package? Use the `pydantic-ai-<name>` naming convention; see [Publishing capability packages](https://ai.pydantic.dev/extensibility/#publishing-capability-packages).

## Contributing

We welcome capability contributions:

1. **Start with an issue.** [Open a capability request](https://github.com/pydantic/pydantic-ai-harness/issues/new?template=capability-request.yml) so we can discuss approach and priority before code is written.
2. **Then open a PR** and link the issue. We review based on community interest; upvotes on both count.
3. **Don't chase green CI.** Get the approach working and let us know; we may push to your branch or follow up, and you'll be credited as the original author. (See the [Pydantic AI contributing guide](https://github.com/pydantic/pydantic-ai/blob/main/CONTRIBUTING.md).)

> **Note**: PRs that modify `pyproject.toml` or `uv.lock` from non-team members are auto-closed by CI to prevent supply chain risk. If you need a new dependency, [open an issue](https://github.com/pydantic/pydantic-ai-harness/issues/new).

### Development

```bash
make install   # install dependencies
make format    # ruff format
make lint      # ruff check
make typecheck # pyright strict
make test      # pytest
make testcov   # pytest with 100% branch coverage
```

## Version policy

Pydantic AI Harness uses **0.x versioning**, and that's a statement about API stability, not maturity: these capabilities are tested end-to-end and meant for production use, but their APIs may still move between minor releases (0.1 -> 0.2): renamed parameters, changed defaults, restructured APIs, always with deprecation warnings where practical. Patch releases will not intentionally break existing behavior, and every breaking change is documented in release notes with migration guidance your agent can follow. Keeping the Harness a separate package from [Pydantic AI](https://github.com/pydantic/pydantic-ai), which has a [stricter version policy](https://ai.pydantic.dev/version-policy/), is what lets capabilities iterate at the speed the field moves.

## Part of the Pydantic Stack

Everything you need to ship production-grade AI agents:

- [Pydantic AI](https://pydantic.dev/pydantic-ai?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai-harness): the type-safe agent framework
- [Pydantic Logfire](https://pydantic.dev/logfire?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai-harness): AI-first, full-stack observability
- [Logfire AI Gateway](https://pydantic.dev/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai-harness): unified LLM proxy
- [Pydantic Evals](https://ai.pydantic.dev/evals/): evaluate any Python function, agents included
- [genai-prices](https://github.com/pydantic/genai-prices): model pricing data, kept current

## License

MIT; see [LICENSE](LICENSE).
