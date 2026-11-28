# Pydantic AI Harness

[![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-harness.svg)](https://pypi.python.org/pypi/pydantic-ai-harness)
[![versions](https://img.shields.io/pypi/pyversions/pydantic-ai-harness.svg)](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai.svg)](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/LICENSE)
[![Join Slack](https://img.shields.io/badge/Slack-Join%20Slack-4A154B?logo=slack)](https://logfire.pydantic.dev/docs/join-slack/)

**Your agent's favorite harness, built on Pydantic AI**

---

**Pydantic AI Harness** is the official [capability](https://ai.pydantic.dev/capabilities/overview/) and harness library for [Pydantic AI](https://ai.pydantic.dev/). Every Pydantic AI agent already has a light harness: the typed agent loop, [any model](https://ai.pydantic.dev/models/), your own tools, structured output. For simple agents that's enough. But set an agent loose on complex, long-running work (fix a codebase, research a question, run for hours unattended) and what it needs around the model grows: a [workspace](pydantic_ai_harness/file_system/) to act in, a [plan](pydantic_ai_harness/planning/) it keeps current, [memory](pydantic_ai_harness/memory/) that carries across sessions, [sub-agents](pydantic_ai_harness/sub_agents/) to hand work to, [context management](pydantic_ai_harness/compaction/) for long runs, and [durable execution](https://ai.pydantic.dev/capabilities/durable_execution/overview/) that survives a restart. **Pydantic AI Harness** ships that harness.

Everything here is one primitive: a [capability](https://ai.pydantic.dev/capabilities/), a self-contained unit of agent behavior you add to `capabilities=[...]` on any agent. There are [100+ of them](#capabilities), and complete agents like [Coder](pydantic_ai_harness/coder/) and [Researcher](pydantic_ai_harness/researcher/) are themselves combinations of capabilities. Use a single capability, compose your own set, or start from the whole coding agent and swap parts out.

## Pydantic AI, the harness SDK, `Coder`, and Pax

An agent on its own is a loop calling a model and tools. A **harness** is everything around that loop that makes it hold up on real work: the workspace it acts in, the tools, context management, recovery, guardrails, and the interface you steer it through. Four layers carry that idea; start with the one that matches what you are doing:

| Layer | What it is | Start here when |
|---|---|---|
| [Pydantic AI](https://pydantic.dev/docs/ai/) (`pydantic-ai`) | The agent framework: the agent loop, models from every major provider, tools, structured output, and the capability system | You are building an agent, or an LLM feature in your own application |
| [Pydantic AI Harness](#capabilities) (`pydantic-ai-harness`) | The SDK you build a harness of your own from: workspaces, file and shell tools, memory, sub-agents, context management, and ready-made agents like `Coder` | Your agent needs to edit files, run commands, remember, delegate, or keep working through long runs |
| [`Coder`](pydantic_ai_harness/coder/) | A ready-made coding agent in one capability: file and shell tools, task delegation, repo instructions and context management. You supply the workspace it runs in, and any guardrails | You want a coding agent in your own code, to run, extend or swap parts of |
| [Pax](https://pydantic.dev/docs/ai/pax/) (`pydantic-ai-pax`) | The TUI in front of `Coder`, and a terminal interface for any Pydantic AI agent; everything in it is a plugin, the coding agent included | You want to use a coding agent rather than build one, or run your own agent in a terminal |

Each package installs the one it is built on: `pydantic-ai-harness` installs `pydantic-ai-slim`, and `pydantic-ai-pax` installs `pydantic-ai-harness`. `Coder` supplies the tools, context management and recovery; you choose the workspace it runs in and any guardrails, and Pax is the interface in front of it. That is why either works without the other: build on `Coder` and never open Pax, or point Pax at an agent that is entirely your own. Neither is [`clai`](https://pydantic.dev/docs/ai/integrations/cli/), the legacy CLI that ships with Pydantic AI.

## Quick start

uv:

```bash
uv add "pydantic-ai-harness[coder,anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[coder,anthropic]"
```

<!-- Keep this blown-out example in sync across docs/coder.md, docs/index.md, README.md, pydantic_ai_harness/coder/README.md, and examples/coding_agent.py. -->

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder import Coder

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[LocalWorkspace('.'), Coder()],
)
result = agent.run_sync('Find out why tests/test_parser.py fails and fix the bug it caught.')
print(result.output)
```

`LocalWorkspace('.')` is where the agent works: its file tools and commands run on your machine, in this directory. It is not a sandbox, so commands can reach anything you can. To run the same agent in an isolated cloud machine, swap it for a sandbox capability (Modal, E2B, Render, Sprites, Daytona, or boxd); see [Workspaces](#workspaces).

With [Modal](pydantic_ai_harness/modal_sandbox/), for example:

```python
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[ModalSandbox(), Coder()])
```

Coder provides six tools: `read_file`, `write_file`, `edit_file`, `list_files`, `grep`, and `shell`, plus `delegate_task` to hand a sub-task to a fresh run of the same agent, repository context, and context controls. Shell commands are unrestricted and can persist beyond individual runs. Default instructions guide autonomous investigation, editing, and verification; pass `instructions=` to add your own guidance.

```bash
uv tool install pydantic-ai-pax
pax
```

Point it at your own agent with `pax --agent mymodule:agent`, or at the bundled export with `pax --agent pydantic_ai_harness.coder:coder_agent -m anthropic:claude-opus-5-5`, which is the example above without a model.

Every model works: swap the string for [any provider's](https://ai.pydantic.dev/models/). Need more? Add capabilities to the list; here's the same coder on `gpt-6.1-sol`, with web search and cross-session memory:

uv:

```bash
uv add "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-slim[openai]"
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace, WebSearch
from pydantic_ai_harness import Coder, Memory
from pydantic_ai_harness.memory import FileMemoryStore

agent = Agent(
    'openai:gpt-6.1-sol',
    capabilities=[
        LocalWorkspace('.'),
        Coder(),
        WebSearch(),  # look up docs and error messages on the web
        Memory(FileMemoryStore('.agent-memory')),  # remembers across sessions
    ],
)
```

[Skills](pydantic_ai_harness/skills/) (your `SKILL.md` procedures, loaded on demand; point it at a `skills/` directory and add the `skills` extra), [Web Fetch](https://ai.pydantic.dev/capabilities/web-fetch/), [Guardrails](pydantic_ai_harness/guardrails/), and [Dynamic Workflow](pydantic_ai_harness/dynamic_workflow/) slot in the same way.

## Built from capabilities

`Coder` is a regular combined capability: [`FileSystem`](pydantic_ai_harness/file_system/) with five of its tools and content hashes off, [`Shell`](pydantic_ai_harness/shell/) with its persistent `shell` tool and no allowlist, [`RepoContext`](pydantic_ai_harness/repo_context/), [`SubAgents`](pydantic_ai_harness/sub_agents/) delegating to the agent itself, [summarizing compaction](pydantic_ai_harness/compaction/), and a bounded [`ToolOutputLimits`](pydantic_ai_harness/tool_output_limits/), plus its default instructions and JSON argument repair. Use it whole, or build the same agent from those capabilities to change any setting; the [Coder page](pydantic_ai_harness/coder/) lists the exact configuration, tool signatures, delegation, and the persistent shell lifecycle.

## Workspaces

A [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/) is where the agent's files and commands live: your machine with `LocalWorkspace`, or an isolated sandbox. Harness capabilities never pick one for you. Attach one, or the run fails at its start and tells you what to attach.

[Coder](pydantic_ai_harness/coder/), [FileSystem](pydantic_ai_harness/file_system/), [Shell](pydantic_ai_harness/shell/), [Repo Context](pydantic_ai_harness/repo_context/), and [Macroscope](pydantic_ai_harness/macroscope/) work in the run's workspace, starting in its working directory. To work in a subdirectory, set it on the workspace: `LocalWorkspace('./repo')`. To continue in the same files from a later run or another agent, see [Sharing a workspace](pydantic_ai_harness/coder/#sharing-a-workspace).

Skills, SubAgents, ToolOutputLimits, and Memory's `FileMemoryStore` use the run's workspace for their own files too. To keep those files on your machine while the agent works in a sandbox, point them at a local location:

```python
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import Memory, ToolOutputLimits
from pydantic_ai_harness.memory import FileMemoryStore
from pydantic_ai_harness.tool_output_limits import LocalFileOverflowStore

memory = Memory(FileMemoryStore('.', workspace=LocalWorkspaceBackend('/var/lib/myapp/memory')))  # notes on this machine
limits = ToolOutputLimits(store=LocalFileOverflowStore())  # spills in this machine's temp directory
```

By default, harness files (Shell's background job logs, tool-output spills) go in `.pydantic-ai-harness/` in the workspace's working directory, which is git-ignored. When the workspace user cannot write to the working directory, Shell keeps its job files in a directory private to that user under the workspace's temp directory (`$TMPDIR` in its environment, or `/tmp`) instead.

Step Persistence's file and SQLite stores, Media's disk and SQLite stores, and Code Mode mounts use paths on the machine running the agent, not the run's workspace.

`FileSystem`'s `root_dir` limits only the file tools, not `Shell` commands.

Upgrading from an earlier release? The [Coder page](pydantic_ai_harness/coder/#upgrading) lists what changed.

## From problem to capability

Look up a capability by the problem it solves. The Capabilities section below lists all of them by what they give your agent.

| If your problem is... | Capability | How it works |
|---|---|---|
| The model's output must pass a check, and bad output should be retried or blocked | [`OutputGuardrail`](pydantic_ai_harness/guardrails/) | Its `guard` returns `GuardrailResult.retry(...)` to send the output back to the model, or `False` to raise `OutputBlocked`. `retry` needs `run()`: under `run_stream()` it raises `UnexpectedModelBehavior`. |
| One tool result is flooding the context window | [`ToolOutputLimits`](pydantic_ai_harness/tool_output_limits/) | By default, a result of 10,000 characters or more spills to a file in the run's workspace, and the model gets a preview plus a handle to read the rest. If the write fails, the result is truncated to 4,000 characters. Without a workspace, pass `store=LocalFileOverflowStore()`. |
| The agent's files and commands must start in a specific directory | Core [`LocalWorkspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/), or a sandbox | Set the directory on the workspace: `LocalWorkspace('./repo')`, or `working_dir=` on a sandbox. Every capability that works in the workspace starts there; `Shell(cwd=)` and `FileSystem(cwd=)` warn and are ignored. In your own tool, `ctx.workspace.run(..., cwd='subdir')` starts one command elsewhere. |
| A run, a day, or a tenant needs a spend ceiling | Core [`UsageLimits`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits), or [`SpendLimits`](pydantic_ai_harness/spend_limits/) | For one run, pass `usage_limits=UsageLimits(cost_limit=Decimal('5'))`: the run raises `UsageLimitExceeded` after the response that crosses the limit. For a day or month, per tenant, or across worker processes, `SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')])` refuses the next request once a window is spent. |
| The model hits a usage or spend limit before it answers | [`ReportRemainingBudget`](pydantic_ai_harness/report_remaining_budget/) | `ReportRemainingBudget(spend_limits=spend)` tells the model on every request how many requests, tool calls, tokens and dollars the run's `UsageLimits` and the `SpendLimits` budgets have left, so it can answer while it still can. |
| The model should get only a few calls to an expensive tool per run | [`ToolBudgets`](pydantic_ai_harness/tool_budgets/) | `ToolBudgets(per_tool={'web_search': 3}, refuse_repeats={'web_search'})` answers the fourth call, or an exact repeat, with a tool result saying the budget is used, and the run continues. `UsageLimits.tool_calls_limit` stays the hard ceiling. |
| The model repeats the same tool call without making progress | [`DetectLoops`](pydantic_ai_harness/detect_loops/) | The default `action='nudge'` adds a system prompt asking the model to change approach; `action='error'` stops the run with `LoopDetectedError`. |
| A tool raises an exception and the whole run ends | [`ReportToolErrors`](pydantic_ai_harness/report_tool_errors/) | Sends the exception to the model as a failed tool result naming the tool and the exception type, so the model can try another way. `format_error=` replaces the text, for example to include the exception's message. `ModelRetry`, `ToolFailed` and approvals keep their meaning, and `UsageLimits.request_limit` stops a model that keeps calling a failing tool. |
| The model sends malformed JSON in a tool call | [`RepairToolArguments`](pydantic_ai_harness/repair_tool_arguments/) | Runs `json-repair` on the arguments before validation, fixing trailing commas, single-quoted keys and truncated JSON; valid arguments pass through unchanged. Import it from `pydantic_ai_harness.repair_tool_arguments` and install the `repair-tool-arguments` extra. `Coder` includes `RepairToolArguments`, so its `coder` extra installs `json-repair` too. |

## Capabilities

Every capability is a self-contained unit you drop into `capabilities=[...]`, and they all compose, with each other and with your own. Some come with [`pydantic-ai`](https://github.com/pydantic/pydantic-ai) itself, the rest with this package; the **Package** column says which. 100+ in all, grouped by what they give your agent:

### Harnesses

Complete agent stacks as regular combined capabilities: one import gives you a working agent, and you can take either apart into the blocks below.

| Harness | Package | What it provides |
|---|---|---|
| [Coder](pydantic_ai_harness/coder/) | Harness | Six coding tools, persistent shell commands, delegation to itself, autonomous guidance, and context controls |
| [Researcher](pydantic_ai_harness/researcher/) | Harness | A complete web-research stack: search, page fetching, a delegated sub-researcher, and bounded tool output |

### Workspaces

Where the agent runs commands and keeps files: this machine, a container, or a sandbox. [Compare them](https://pydantic.dev/docs/ai/core-concepts/workspace/#supported-workspaces).

| Capability | Package | What it does |
|---|---|---|
| [Local Workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/) | Core | Commands and files in a directory on this machine; no isolation. Adds no tools: pair it with the [Files & shell](#files--shell) tools or your own |
| [Docker Container](pydantic_ai_harness/docker_container/) | Harness | Commands and files in a local [Docker](https://www.docker.com/) or Podman container |
| [Smol Sandbox](pydantic_ai_harness/smol_sandbox/) | Harness | Commands and files in a [smolvm](https://github.com/smol-machines/smolvm) microVM on your own machine; no cloud account |
| [BoxLite Sandbox](pydantic_ai_harness/boxlite_sandbox/) | Harness | Commands and files in a [BoxLite](https://github.com/boxlite-ai/boxlite) micro-VM on your own machine, with an optional egress allow-list; no cloud account |
| [OpenSandbox](pydantic_ai_harness/opensandbox_sandbox/) | Harness | Commands and files in a sandbox on your own [OpenSandbox](https://open-sandbox.ai) server, on Docker or Kubernetes |
| [Kubernetes Sandbox](pydantic_ai_harness/kubernetes_sandbox/) | Harness | Commands and files in a sandbox pod from a Kubernetes [agent-sandbox](https://agent-sandbox.sigs.k8s.io) warm pool |
| [Modal Sandbox](pydantic_ai_harness/modal_sandbox/) | Harness | Commands and files in an isolated [Modal](https://modal.com) cloud sandbox |
| [E2B Sandbox](pydantic_ai_harness/e2b_sandbox/) | Harness | Commands and files in an isolated [E2B](https://e2b.dev) cloud sandbox |
| [Daytona Sandbox](pydantic_ai_harness/daytona_sandbox/) | Harness | Commands and files in an isolated [Daytona](https://www.daytona.io) cloud sandbox |
| [Render Sandbox](pydantic_ai_harness/render_sandbox/) | Harness | Commands and files in an isolated [Render](https://render.com/docs/sandboxes) sandbox |
| [Sprites Sandbox](pydantic_ai_harness/sprites_sandbox/) | Harness | Commands and files in a persistent [Fly.io Sprite](https://sprites.dev) |
| [boxd Sandbox](pydantic_ai_harness/boxd_sandbox/) | Harness | Commands and files in an isolated [boxd](https://boxd.sh) cloud machine |
| [SSH Workspace](pydantic_ai_harness/ssh_workspace/) | Harness | Commands and files on a remote host over your `ssh` client, as your SSH user; not sandboxed there |
| [Bubblewrapped Workspace](pydantic_ai_harness/bubblewrap_sandbox/) | Harness | Wraps another workspace capability so its commands run in a Linux [bubblewrap](https://github.com/containers/bubblewrap) wrapper on that workspace's host (best-effort convenience, not a security boundary) |

### Files & shell

The tools the model uses in the run's workspace, whichever one it is.

| Capability | Package | What it does |
|---|---|---|
| [FileSystem](pydantic_ai_harness/file_system/) | Harness | Read, write, edit, list, and search files under a root in the run's workspace, with opt-in ripgrep tools; path-traversal checked, secrets read-only |
| [Smart Search](pydantic_ai_harness/smart_search/) | Harness | Code search by behavior in the run's workspace: plain-English queries, ranked excerpts, judged by any model (a decision model over System One, or TypeSafe's Jev, recommended) |
| [Shell](pydantic_ai_harness/shell/) | Harness | Command execution in the run's workspace with allowlists, denylists, timeouts, credential-stripping, and opt-in commands that outlive the run |
| [Announce Working Directory](pydantic_ai_harness/announce_working_directory/) | Harness | Names the workspace's working directory in the instructions, once, for your own tools as well as `FileSystem` and `Shell` |

### Tools & native abilities

Abilities the provider executes natively, and connections to any MCP server.

| Capability | Package | What it does |
|---|---|---|
| [MCP](https://ai.pydantic.dev/capabilities/mcp/) | Core | Connect any MCP server's tools; local by default, provider-native connectors opt-in |
| [Image Generation](https://ai.pydantic.dev/capabilities/image-generation/) | Core | Generate and edit images; provider-native where supported, sub-agent fallback elsewhere |

### Integrations

Tools from third-party services, most through the service's own hosted MCP server.

| Capability | Package | What it does |
|---|---|---|
| [GitHub](pydantic_ai_harness/github/) | Harness | Read and change GitHub repositories, issues, pull requests, and other accessible resources. |
| [Linear](pydantic_ai_harness/linear/) | Harness | Read and change Linear issues, projects, teams, and comments. |
| [Macroscope](pydantic_ai_harness/macroscope/) | Harness | Run a local [Macroscope](https://docs.macroscope.com/cli) code review and hand the findings to the agent |
| [LocalStack](pydantic_ai_harness/localstack/) | Harness | An emulated AWS environment with AWS CLI tools |
| [Google Workspace](pydantic_ai_harness/google_workspace/) | Harness | Use Gmail, Calendar, Drive, and other Google Workspace tools. |
| [Notion](pydantic_ai_harness/notion/) | Harness | Search and change Notion workspace content. |
| [Slack](pydantic_ai_harness/slack/) | Harness | Give an agent Slack messages, channels, and canvas tools. |
| [Day AI](pydantic_ai_harness/day_ai/) | Harness | Search and update CRM records and meeting context through [Day AI](https://day.ai)'s hosted MCP server |
| [Grain](pydantic_ai_harness/grain/) | Harness | Search and read meetings, transcripts, and notes, and create clips and tag meetings, through [Grain](https://grain.com)'s hosted MCP server |
| [Pylon](pydantic_ai_harness/pylon/) | Harness | Work with support issues, accounts, and contacts through [Pylon](https://www.usepylon.com)'s hosted MCP server |
| [StackOne](pydantic_ai_harness/stackone/) | Harness | Act on linked SaaS accounts (HRIS, ATS, CRM, …) via [StackOne](https://www.stackone.com) |
| [PostHog](pydantic_ai_harness/posthog/) | Harness | Query product analytics and manage feature flags, experiments, and dashboards through [PostHog](https://posthog.com)'s hosted MCP server |
| [Logfire MCP](pydantic_ai_harness/logfire_mcp/) | Harness | Query Logfire telemetry and manage observability resources. |
| [Ordinal](pydantic_ai_harness/ordinal/) | Harness | Draft, schedule, and analyze social posts through [Ordinal](https://www.tryordinal.com)'s hosted MCP server |

### Web & research

Finding and reading things on the open web.

| Capability | Package | What it does |
|---|---|---|
| [Web Search](https://ai.pydantic.dev/capabilities/web-search/) | Core | Provider-native search where available, local DuckDuckGo fallback everywhere |
| [Web Fetch](https://ai.pydantic.dev/capabilities/web-fetch/) | Core | Fetch and read URLs, native or local |
| [X Search](https://ai.pydantic.dev/capabilities/x-search/) | Core | Search X; native on xAI, subagent fallback elsewhere |
| [Exa Search](pydantic_ai_harness/exa/) | Harness | Web research via [Exa](https://exa.ai): excerpted search, full-page reads, opt-in cited deep search |
| [Exa Agent](pydantic_ai_harness/exa/) | Harness | Delegate open-ended research to the Exa Agent API |
| [You.com Search](pydantic_ai_harness/youdotcom/) | Harness | Web search and page reads via [You.com](https://you.com): query-relevant excerpts or full-page markdown |
| [You.com Research](pydantic_ai_harness/youdotcom/) | Harness | Cited answers and multi-step research via the You.com Answer, Research, and Finance Research APIs |
| [Keenable Search](pydantic_ai_harness/keenable/) | Harness | Web research via [Keenable](https://keenable.ai): excerpted search and full-page reads, keyless and with no extra to install |
| [Browser Use](pydantic_ai_harness/browser_use/) | Harness | Hand web tasks to an autonomous [browser-use](https://github.com/browser-use/browser-use) agent driving a real browser |
| [CDP Browser](pydantic_ai_harness/cdp_browser/) | Harness | Drive a throwaway Chrome tab with raw DevTools Protocol commands, best from Code Mode |
| [Playwright Browser](pydantic_ai_harness/playwright/) | Harness | Drive a real Chromium page yourself: navigate, click, type, read, and inspect what the page did |
| [Computer Use](pydantic_ai_harness/computer_use/) | Harness | See and control the desktop of the run's workspace with screenshots, mouse, and keyboard |

### Reasoning, planning & delegation

How the agent thinks and divides the work.

| Capability | Package | What it does |
|---|---|---|
| [Thinking](https://ai.pydantic.dev/capabilities/thinking/) | Core | Provider-adaptive extended thinking at configurable effort |
| [Planning](pydantic_ai_harness/planning/) | Harness | Model-owned task plans with a cache-safe live reminder |
| [Subagents](pydantic_ai_harness/sub_agents/) | Harness | Delegate self-contained tasks to named child agents |
| [Peers](pydantic_ai_harness/peers/) | Harness | Discover and message other agent sessions without controlling their runs |
| [Dynamic Workflow](pydantic_ai_harness/dynamic_workflow/) | Harness | The model orchestrates sub-agents from one Python script: fan-out, chain, vote in a single tool call, with hard `max_agent_calls` budgets |
| [Advisor](pydantic_ai_harness/advisor/) | Harness | Let an executor consult a stronger model mid-run |
| [Background Tools](pydantic_ai_harness/background_tools/) | Harness | Run selected tools concurrently; results arrive as follow-up messages |

### Context management

How the agent spends its context window over a long run, and how often it pays for the same tokens.

| Capability | Package | What it does |
|---|---|---|
| [Code Mode](pydantic_ai_harness/code_mode/) | Harness | The model writes one Python script that calls many tools inside a [Monty](https://github.com/pydantic/monty) sandbox: one round-trip instead of N, and intermediate results never enter the context window |
| [IPython Code Mode](pydantic_ai_harness/ipython_code_mode/) | Harness | Code Mode with a persistent [IPython](https://ipython.org/) kernel in place of the Monty sandbox: full CPython and any installed package, with state kept between calls. For local agents only; not a sandbox |
| [Tool Bridge](pydantic_ai_harness/tool_bridge/) | Harness | Keep the tool list the model sees fixed behind `tool_search`, `tool_describe` and `tool_call`, so the prompt cache holds however many tools sit behind them |
| [Tool Search](https://ai.pydantic.dev/capabilities/tool-search/) | Core | Load tool definitions on demand instead of carrying hundreds in every prompt |
| [Compaction](https://ai.pydantic.dev/capabilities/compaction/) | Core | Provider-native compaction on OpenAI and Anthropic; the provider summarizes history server-side |
| [Compaction](pydantic_ai_harness/compaction/) | Harness | Model-agnostic strategies: tool-result clearing, sliding-window trimming, LLM summarization, tiered; all window-relative, with live usage reporting |
| [Tool Output Limits](pydantic_ai_harness/tool_output_limits/) | Harness | Truncate, spill to a queryable file, or summarize oversized tool returns at the source |
| [Warn On Cache Busts](pydantic_ai_harness/warn_on_cache_busts/) | Harness | Detect prompt-cache prefix collapses between requests, from the provider's own numbers |
| [Label Cross Model History](pydantic_ai_harness/label_cross_model_history/) | Harness | Tell the model when earlier assistant turns came from another model, after a model switch or handoff |

### Knowledge & memory

What the agent knows and remembers, loaded when relevant instead of carried in every prompt.

| Capability | Package | What it does |
|---|---|---|
| [Memory](pydantic_ai_harness/memory/) | Harness | A persistent, namespaced notebook: bounded prompt injection, on-demand search; in-memory/file/Postgres stores |
| [Pixeltable](pydantic_ai_harness/pixeltable/) | Harness | Read-only, allowlisted tools over Pixeltable tables and embedding indexes, plus a Pixeltable `MemoryStore` |
| [Conversation Search](pydantic_ai_harness/conversation_search/) | Harness | BM25 search over stored history, including turns compaction dropped |
| [Skills](pydantic_ai_harness/skills/) | Harness | Load [Agent Skill](https://ai.pydantic.dev/capabilities/on-demand/) (`SKILL.md`) instructions on demand |
| [Repo Context](pydantic_ai_harness/repo_context/) | Harness | Start runs oriented: `AGENTS.md`/`CLAUDE.md` + repository structure |
| [Report Stale Files](pydantic_ai_harness/report_stale_files/) | Harness | Tell the model when a file it read or wrote has since changed or been deleted in the workspace |
| [Pydantic AI Docs](pydantic_ai_harness/pydantic_ai_docs/) | Harness | On-demand Pydantic AI documentation lookup |
| [Current Time](pydantic_ai_harness/current_time/) | Harness | Today's date, or the current hour, in a time zone you pick, without breaking the prompt cache |

### Control & safety

Bounding what the agent may do, and keeping it on-instructions.

| Capability | Package | What it does |
|---|---|---|
| [Repair Tool Arguments](pydantic_ai_harness/repair_tool_arguments/) | Harness | Repair malformed JSON tool arguments before schema validation. |
| [Guardrails](pydantic_ai_harness/guardrails/) | Harness | Validate/block/redact user input, tool calls, tool results, and output, including secret masking and parallel async guards |
| [Goal](pydantic_ai_harness/goal/) | Harness | Require a caller-defined completion check before an unattended run ends |
| [Prompt Injection Defender](pydantic_ai_harness/prompt_injection_defender/) | Harness | Classify local tool results for indirect prompt injection and optionally withhold high-risk results |
| [Artifact Scanning](pydantic_ai_harness/artifact_scanning/) | Harness | Scan memory files and skills for prompt injection, credential exfiltration, and hidden content before they are stored or loaded |
| [Spend Limits](pydantic_ai_harness/spend_limits/) | Harness | Cross-window USD/token budgets and per-response cost tracking, per model and per tenant |
| [Ask User](pydantic_ai_harness/ask_user/) | Harness | Let the model ask the user multiple-choice questions mid-run; you supply the answerer (terminal, web, test) |
| [Deny Unanswered](pydantic_ai_harness/deny_unanswered/) | Harness | Deny approval and question calls that nobody answered by the end of an unattended run, in parent and child agents |
| [Tool approval](https://ai.pydantic.dev/deferred-tools#human-in-the-loop-tool-approval) | Core | Flag tool calls that need human approval before they run |
| [Tool Call Judge](pydantic_ai_harness/tool_call_judge/) | Harness | A second model answers one risk question per tool call and blocks the call before the tool function runs |
| [Approval Reviewer](pydantic_ai_harness/approval_reviewer/) | Harness | A second model answers one risk question per call waiting for approval, approves the low-risk ones, and leaves the rest to a person |
| [Permissions](pydantic_ai_harness/permissions/) | Harness | Ask a person before file writes and commands, by permission mode: full access, auto (asks before risky commands), or read-only; allow, ask about, or deny tool calls and shell commands by rule |
| [Handle Deferred Tool Calls](https://ai.pydantic.dev/capabilities/handle-deferred-tool-calls/) | Core | Resolve approvals, answers and external results on deferred tool calls inside the run, in place or by waiting for them, durably on Temporal and AWS Lambda |
| [System Reminders](pydantic_ai_harness/system_reminders/) | Harness | Cache-safe re-injection of guidance mid-run to counter instruction fade |
| [Detect Loops](pydantic_ai_harness/detect_loops/) | Harness | Notice repeated tool calls, results, alternations, or responses, or a response repeating itself as it streams, and nudge the model, stop the run, or call your handler |
| [Report Tool Errors](pydantic_ai_harness/report_tool_errors/) | Harness | Send an exception a tool raises to the model as a failed tool result, so the run continues |
| [Tool Budgets](pydantic_ai_harness/tool_budgets/) | Harness | Cap calls per tool and per run, and refuse exact repeats, with a tool result the model can act on instead of an exception |
| [Report Remaining Budget](pydantic_ai_harness/report_remaining_budget/) | Harness | Tell the model on every request how many requests, tool calls, tokens and dollars its usage limits and spend budgets have left |
| [Trajectory Judge](pydantic_ai_harness/trajectory_judge/) | Harness | A second model reviews the live run every N requests over a sliding token window and steers it mid-run |
| [Refine Output](pydantic_ai_harness/refine_output/) | Harness | Send the agent back to review and revise its output before the run ends |
| [Require Verification](pydantic_ai_harness/require_verification/) | Harness | Send the agent back to run its checks when it tries to finish after editing code without a passing check since its last edit |
| [Wrap Up At Limits](pydantic_ai_harness/wrap_up_at_limits/) | Harness | End the run with a final answer, not `UsageLimitExceeded`, when a usage limit would stop it |

### Self-extension

| Capability | Package | What it does |
|---|---|---|
| [Capability Creation](pydantic_ai_harness/capability_creation/) | Harness | The agent writes, validates, and persists *new capabilities* during a run; the host loads them into a later run via `load_active(workspace)` and `capabilities=` |

### Execution runtime

Outside the loop: how runs persist, survive failures, and get observed and configured in production.

| Capability | Package | What it does |
|---|---|---|
| [Durable execution](https://ai.pydantic.dev/capabilities/durable_execution/overview/) | Core | Runs that survive restarts and failures, on nine engines: [Temporal](https://ai.pydantic.dev/capabilities/durable_execution/temporal/), [DBOS](https://ai.pydantic.dev/capabilities/durable_execution/dbos/), [Prefect](https://ai.pydantic.dev/capabilities/durable_execution/prefect/), [Restate](https://ai.pydantic.dev/capabilities/durable_execution/restate/), [Airflow](https://ai.pydantic.dev/capabilities/durable_execution/airflow/), and the Harness engines AWS Lambda, Absurd, pgtask, and Azure Durable Task below. See the [workspace guide](https://pydantic.dev/docs/ai/core-concepts/workspace/#durable-execution) for Coder, Shell, and FileSystem |
| [Temporal durability](pydantic_ai_harness/temporal/) | Harness | Run model requests, tool calls, and MCP communication as Temporal activities; the code lives in `pydantic_ai.durable_exec.temporal` until v3 |
| [DBOS durability](pydantic_ai_harness/dbos/) | Harness | Run model requests and MCP communication as DBOS steps; the code lives in `pydantic_ai.durable_exec.dbos` until v3 |
| [Prefect durability](pydantic_ai_harness/prefect/) | Harness | Run model requests, tool calls, and MCP communication as Prefect tasks; the code lives in `pydantic_ai.durable_exec.prefect` until v3 |
| [AWS Lambda durability](pydantic_ai_harness/aws_lambda/) | Harness | Checkpoint model requests and tool calls into AWS Lambda durable function steps |
| [Absurd durability](pydantic_ai_harness/absurd/) | Harness | Checkpoint model requests, MCP calls, and tool calls into Absurd steps on PostgreSQL |
| [pgtask durability](pydantic_ai_harness/pgtask/) | Harness | Checkpoint model requests, MCP calls, and tool calls into pgtask steps on PostgreSQL |
| [Azure Durable Task durability](pydantic_ai_harness/azure_durable_task/) | Harness | Run model requests, tool calls and durable operations as activities in Durable Task Scheduler orchestrations |
| [Render Workflows](pydantic_ai_harness/render/) | Harness | Run model requests and tool calls as chained task runs with separate retries, timeouts, and compute settings; retrying the task that runs the agent can repeat completed work |
| [Step Persistence](pydantic_ai_harness/step_persistence/) | Harness | Save, restore, resume (`continue_run`), and fork (`fork_run`) runs; file/SQLite/Mongo backends |
| [Model Fallback](https://ai.pydantic.dev/capabilities/model-fallback/) | Core | Fall back to other models when a model request fails or its response is rejected, within the run's own request step |
| [Model Router](pydantic_ai_harness/model_router/) | Harness | Let another model pick the run's model from a named menu, once per run or per step, with a declared default |
| [Instrumentation](https://ai.pydantic.dev/capabilities/instrumentation/) | Core | OpenTelemetry GenAI spans for every model and tool call; the raw material for [Logfire](https://pydantic.dev/logfire) traces |
| [Managed Prompt](pydantic_ai_harness/logfire/) | Harness | Back instructions with a [Logfire](https://pydantic.dev/logfire)-managed prompt; version and roll out without redeploying |
| [Thread Executor](https://ai.pydantic.dev/capabilities/thread-executor/) | Core | Run sync tools on a shared thread pool |

Core also ships loop-customization capabilities for production servers: [Select Model](https://ai.pydantic.dev/capabilities/select-model/), [Resolve Model ID](https://ai.pydantic.dev/capabilities/resolve-model-id/), [Prepare Tools / Prepare Output Tools](https://ai.pydantic.dev/capabilities/prepare-tools/), [Prefix Tools](https://ai.pydantic.dev/capabilities/prefix-tools/), [Set Tool Metadata](https://ai.pydantic.dev/capabilities/set-tool-metadata/), [Include Tool Return Schemas](https://ai.pydantic.dev/capabilities/include-tool-return-schemas/), [Process History](https://ai.pydantic.dev/capabilities/process-history/), [Process Event Stream](https://ai.pydantic.dev/capabilities/process-event-stream/), [Reinject System Prompt](https://ai.pydantic.dev/capabilities/reinject-system-prompt/), and [Raise Content Filter Error](https://ai.pydantic.dev/capabilities/raise-content-filter-error/).

And the agent plugs into any interface: [ACP](pydantic_ai_harness/experimental/acp/) *(experimental, Harness)* serves it to editors like Zed over the [Agent Client Protocol](https://agentclientprotocol.com), and core ships the [web chat UI](https://ai.pydantic.dev/web/), [CLI](https://ai.pydantic.dev/cli/), [frontend adapters](https://ai.pydantic.dev/ui/) (AG-UI, Vercel AI), and [realtime voice](https://ai.pydantic.dev/realtime/).

Community packages extend the same capability system further; see [third-party capabilities](https://ai.pydantic.dev/capabilities/third-party/).

## Composing from blocks

A research agent from regular capabilities -- this is [`Researcher`](pydantic_ai_harness/researcher/)'s composition, minus its short default instructions:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai_harness import SubAgent, SubAgents, ToolOutputLimits
from pydantic_ai_harness.tool_output_limits import LocalFileOverflowStore

sub_researcher = SubAgent(
    Agent(
        name='researcher',
        description='Research a focused sub-question on the web and report back with findings and source links',
        capabilities=[WebSearch(local=True), WebFetch(local=True), ToolOutputLimits(store=LocalFileOverflowStore())],
    )
)

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        WebSearch(local=True),  # native provider search, DuckDuckGo fallback elsewhere
        WebFetch(local=True),  # read the pages behind the results, native or local
        SubAgents(agents=[sub_researcher], agent_folders=None),
        ToolOutputLimits(store=LocalFileOverflowStore()),  # fetched pages don't flood the context
    ],
)

result = agent.run_sync('What changed in the top three Python agent frameworks this month? Cite sources.')
print(result.output)
#> ...
```

`logfire.instrument_pydantic_ai()` records [a full trace of every run](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946): every model call and tool call, with token and cost tracking. It's standard OpenTelemetry, so any OTLP backend works; run `uvx logfire-cli wizard --task pydantic-ai` once to [set up Logfire](https://pydantic.dev/docs/ai/integrations/logfire/#using-logfire).

## When do you need the Harness?

"Harness" is the field's term for everything around the model that turns it into an agent: the loop, the tools, the context management. Reach for this package when your agent should *do* more than core's lean harness covers: touch files, run code, browse, remember, delegate, or stay coherent through hours-long runs. The boundary between the packages is mechanical, not a maturity tier: core ships the capabilities that require model or framework support (provider-native tools like [image generation](https://ai.pydantic.dev/capabilities/image-generation/), provider APIs like [compaction](https://ai.pydantic.dev/capabilities/compaction/), deep loop integration like [tool search](https://ai.pydantic.dev/capabilities/tool-search/), and fundamentals like [thinking](https://ai.pydantic.dev/capabilities/thinking/), [MCP](https://ai.pydantic.dev/capabilities/mcp/), and [web search](https://ai.pydantic.dev/capabilities/web-search/)) and the Harness ships everything else, as a separate package so capabilities can iterate at the speed the field moves while Pydantic AI itself stays lean.

## Installation

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

This installs [`pydantic-ai-slim`](https://ai.pydantic.dev/install/) with it, so it works on its own; you don't need to install Pydantic AI separately. Model providers and the CLI come via extras that pass through to Pydantic AI: `pydantic-ai-harness[anthropic]`, `[cli]`. Some capabilities need their own extra for optional dependencies; each capability's page gives its exact install line. Requires Python 3.11+.

## Build your own

[Capabilities](https://ai.pydantic.dev/capabilities/#building-custom-capabilities) are the primary extension point for Pydantic AI, and every capability in this repo doubles as a worked example. Publishing a standalone package? Use the `pydantic-ai-<name>` naming convention; see [Publishing capability packages](https://ai.pydantic.dev/extensibility/#publishing-capability-packages).

## Contributing

We welcome capability contributions:

1. **Start with an issue.** [Open a feature request](https://github.com/pydantic/pydantic-ai/issues/new?template=feature-request.yaml) so we can discuss approach and priority before code is written.
2. **Then open a PR** and link the issue. We review based on community interest; upvotes on both count.
3. **Don't chase green CI.** Get the approach working and let us know; we may push to your branch or follow up, and you'll be credited as the original author. (See the [Pydantic AI contributing guide](https://github.com/pydantic/pydantic-ai/blob/main/CONTRIBUTING.md).)


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

Pydantic AI Harness uses **0.x versioning** to signal API stability, not maturity: the capabilities are tested end-to-end and meant for production use, but their APIs may change between minor releases (0.1 -> 0.2) through renamed parameters, changed defaults, or restructured APIs, with deprecation warnings where practical. Patch releases will not intentionally break existing behavior, and every breaking change is documented in release notes with migration guidance your agent can follow. Keeping the Harness a separate package from [Pydantic AI](https://github.com/pydantic/pydantic-ai), which has a [stricter version policy](https://ai.pydantic.dev/version-policy/), lets capabilities change faster than core.

On 0.x, each Harness release ships alongside the Pydantic AI release with the same minor and patch number (Harness 0.51.0 ships with Pydantic AI 2.51.0) and pins that exact `pydantic-ai-slim` version, so upgrading the Harness upgrades Pydantic AI with it and every installable pair has been tested together.

### After 1.0

From 1.0, the Harness follows the rules of Pydantic AI's [version policy](https://ai.pydantic.dev/version-policy/): no intentional breaking changes in minor releases, and deprecated functionality is not removed until the next major version. The changes that policy does not consider breaking, such as bug fixes for code that relied on undocumented behavior, OpenTelemetry span attributes, and `__repr__` output, are not breaking for the Harness either. Neither is a change that follows a third-party service or SDK a capability integrates with, when that service removes or changes what the capability relied on. Capabilities marked experimental on their page (currently [ACP](pydantic_ai_harness/experimental/acp/)) are excluded until they graduate.

A capability's behavior includes what it sends to the model, so stability covers more than signatures:

- **Defaults**: a minor release does not change a default parameter value in a way that changes what an existing agent does, such as enabling a tool, lowering a limit, or switching a fallback. New behavior arrives behind a new parameter, and a default changes only in a major release.
- **Stored history**: tool names, argument schemas, and the other records a capability writes into message history and reads back on later runs, such as metadata keys and framing tags, persist in histories that you may store and resume. A minor release can add optional tool arguments but does not rename or remove a tool, an argument, or a record a capability relies on.
- **Prompt text**: the wording of instructions, tool descriptions, and other text a capability sends to the model can change in a minor release, since improving how models use a capability is part of maintaining it. Release notes list these changes. Where a capability accepts its own instructions or descriptions as parameters, passing them keeps your wording fixed.

From 1.0, Harness version numbers no longer mirror Pydantic AI's, and a Harness major release does not require a Pydantic AI major release. Every Harness release still pins the exact `pydantic-ai-slim` release it was tested with, and moving that pin to a new Pydantic AI major version is itself a Harness major release, so upgrading the Harness within a major version does not cross a Pydantic AI major. The pin is listed in the package metadata on PyPI, and the release notes name the Pydantic AI release each Harness release ships with.

## Part of the Pydantic Stack

Related Pydantic projects:

- [Pydantic AI](https://pydantic.dev/pydantic-ai?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai-harness): the type-safe agent framework
- [Pydantic Logfire](https://pydantic.dev/logfire?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai-harness): AI-first, full-stack observability
- [Logfire AI Gateway](https://pydantic.dev/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai-harness): unified LLM proxy
- [Pydantic Evals](https://ai.pydantic.dev/evals/): evaluate any Python function, agents included
- [genai-prices](https://github.com/pydantic/genai-prices): model pricing data, kept current

## License

MIT; see [LICENSE](LICENSE).
