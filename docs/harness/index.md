---
title: Pydantic AI Harness
description: "Pydantic AI Harness is the official capability library for Pydantic AI: a coding agent, file and shell tools, memory, subagents, and context management."
---

# Pydantic AI Harness

*Your agent's favorite harness, built on Pydantic AI*

[![Join Slack](https://img.shields.io/badge/Slack-Join%20Slack-4A154B?logo=slack)](https://logfire.pydantic.dev/docs/join-slack/)

**Pydantic AI Harness** is the official [capability](../capabilities/overview.md) and harness library for [Pydantic AI](../index.md). Every Pydantic AI agent already has a light harness: the typed agent loop, [any model](../models/overview.md), your own tools, structured output. For simple agents that's enough. But set an agent loose on complex, long-running work (fix a codebase, research a question, run for hours unattended) and what it needs around the model grows: a [workspace](filesystem.md) to act in, a [plan](planning.md) it keeps current, [memory](memory.md) that carries across sessions, [sub-agents](subagents.md) to hand work to, [context management](compaction.md) for long runs, and [durable execution](../durable_execution/overview.md) that survives a restart. **Pydantic AI Harness** ships that harness.

Everything here is one primitive: a [capability](../capabilities/overview.md), a self-contained unit of agent behavior you add to `capabilities=[...]` on any agent. There are [100+ of them](#capabilities), and complete agents like [Coder](coder.md) and [Researcher](researcher.md) are themselves combinations of capabilities. Use a single capability, compose your own set, or start from the whole coding agent and swap parts out.

## Pydantic AI, the Harness, and Pax

Three packages, each built on the one before it. Start with the one that matches what you are doing:

| Package | What it is | Start here when |
|---|---|---|
| [Pydantic AI](../index.md) (`pydantic-ai`) | The agent framework: the agent loop, models from every major provider, tools, structured output, and the capability system | You are building an agent, or an LLM feature in your own application |
| [Pydantic AI Harness](#capabilities) (`pydantic-ai-harness`) | Capabilities for a Pydantic AI agent: workspaces, file and shell tools, memory, sub-agents, context management, and complete agents like `Coder` | Your agent needs to edit files, run commands, remember, delegate, or keep working through long runs |
| [Pax](clai2.md) (`pydantic-ai-pax`) | A coding agent for your terminal, run as `pax`, built on the Harness's `Coder` and extended with plugins | You want to use a coding agent rather than build one, or chat with your own agent in a terminal |

Each package installs the one it is built on: `pydantic-ai-harness` installs `pydantic-ai-slim`, and `pydantic-ai-pax` installs `pydantic-ai-harness`. Pax is not the [`clai` chat CLI](../cli.md) that ships with Pydantic AI.

## Quick start

```bash
pip/uv-add "pydantic-ai-harness[coder,anthropic]"
```

<!-- Keep this blown-out example in sync across docs/coder.md, docs/index.md, README.md, pydantic_ai_harness/coder/README.md, and examples/coding_agent.py. -->

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder import Coder

agent = Agent(
    '{{ anthropic:large }}',
    capabilities=[LocalWorkspace('.'), Coder()],
)
result = agent.run_sync('Find out why tests/test_parser.py fails and fix the bug it caught.')
print(result.output)
```

`LocalWorkspace('.')` is where the agent works: its file tools and commands run on your machine, in this directory. It is not a sandbox, so commands can reach anything you can. To run the same agent in an isolated cloud machine, swap it for a sandbox capability (Modal, E2B, Render, Sprites, Daytona, or boxd); see [Workspaces](#workspaces).

With [Modal](modal-sandbox.md), for example:

```python
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent('{{ anthropic:large }}', capabilities=[ModalSandbox(), Coder()])
```

Coder provides six tools: `read_file`, `write_file`, `edit_file`, `list_files`, `grep`, and `shell`, plus `delegate_task` to hand a sub-task to a fresh run of the same agent, repository context, and context controls. Shell commands are unrestricted and can persist beyond individual runs. Default instructions guide autonomous investigation, editing, and verification; pass `instructions=` to add your own guidance.

```bash
uv tool install pydantic-ai-pax
pax
```

Point it at your own agent with `pax --agent mymodule:agent`, or at the bundled export with `pax --agent pydantic_ai_harness.coder:coder_agent -m {{ anthropic:large }}`, which is the example above without a model.

Every model works: swap the string for [any provider's](../models/overview.md). Need more? Add capabilities to the list; here's the same coder on `{{ openai:medium.name }}`, with web search and cross-session memory:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace, WebSearch
from pydantic_ai_harness import Coder, Memory
from pydantic_ai_harness.memory import FileMemoryStore

agent = Agent(
    '{{ openai:medium }}',
    capabilities=[
        LocalWorkspace('.'),
        Coder(),
        WebSearch(),  # look up docs and error messages on the web
        Memory(FileMemoryStore('.agent-memory')),  # remembers across sessions
    ],
)
```

[Skills](skills.md) (your `SKILL.md` procedures, loaded on demand; point it at a `skills/` directory and add the `skills` extra), [Web Fetch](../capabilities/web-fetch.md), [Guardrails](guardrails.md), and [Dynamic Workflow](dynamic-workflow.md) slot in the same way.

## Built from capabilities

`Coder` is a regular combined capability: [`FileSystem`](filesystem.md) with five of its tools and content hashes off, [`Shell`](shell.md) with its persistent `shell` tool and no allowlist, [`RepoContext`](repo-context.md), [`SubAgents`](subagents.md) delegating to the agent itself, [`ClearToolResults`](compaction.md), and a bounded [`ToolOutputLimits`](tool-output-limits.md), plus its default instructions and JSON argument repair. Use it whole, or build the same agent from those capabilities to change any setting; the [Coder page](coder.md) lists the exact configuration, tool signatures, delegation, and the persistent shell lifecycle.

On a remote sandbox, `Coder` loads repo instructions at run start and may create the sandbox before the first model tool call. Use `Coder(repo_context=False)` for lazy creation.

## Workspaces

A [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/) is where the agent's files and commands live: your machine with `LocalWorkspace`, or an isolated sandbox. Harness capabilities never pick one for you. Attach one, or the run fails at its start and tells you what to attach.

[Coder](coder.md), [FileSystem](filesystem.md), [Shell](shell.md), [Repo Context](repo-context.md), and [Macroscope](macroscope.md) work in the run's workspace, starting in its working directory. To work in a subdirectory, set it on the workspace: `LocalWorkspace('./repo')`. To continue in the same files from a later run or another agent, see [Sharing a workspace](coder.md#sharing-a-workspace).

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

Upgrading from an earlier release? The [Coder page](coder.md#upgrading) lists what changed.

## From problem to capability

Look up a capability by the problem it solves. The Capabilities section below lists all of them by what they give your agent.

| If your problem is... | Capability | How it works |
|---|---|---|
| The model's output must pass a check, and bad output should be retried or blocked | [`OutputGuardrail`](guardrails.md) | Its `guard` returns `GuardrailResult.retry(...)` to send the output back to the model, or `False` to raise `OutputBlocked`. `retry` needs `run()`: under `run_stream()` it raises `UnexpectedModelBehavior`. |
| One tool result is flooding the context window | [`ToolOutputLimits`](tool-output-limits.md) | By default, a result of 10,000 characters or more spills to a file in the run's workspace, and the model gets a preview plus a handle to read the rest. If the write fails, the result is truncated to 4,000 characters. Without a workspace, pass `store=LocalFileOverflowStore()`. |
| The agent's files and commands must start in a specific directory | Core [`LocalWorkspace`](../workspace.md), or a sandbox | Set the directory on the workspace: `LocalWorkspace('./repo')`, or `working_dir=` on a sandbox. Every capability that works in the workspace starts there; `Shell(cwd=)` and `FileSystem(cwd=)` warn and are ignored. In your own tool, `ctx.workspace.run(..., cwd='subdir')` starts one command elsewhere. |
| A run, a day, or a tenant needs a spend ceiling | Core [`UsageLimits`](../agent.md#usage-limits), or [`SpendLimits`](spend.md) | For one run, pass `usage_limits=UsageLimits(cost_limit=Decimal('5'))`: the run raises `UsageLimitExceeded` after the response that crosses the limit. For a day or month, per tenant, or across worker processes, `SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')])` refuses the next request once a window is spent. |
| The model hits a usage or spend limit before it answers | [`ReportRemainingBudget`](report-remaining-budget.md) | `ReportRemainingBudget(spend_limits=spend)` tells the model on every request how many requests, tool calls, tokens and dollars the run's `UsageLimits` and the `SpendLimits` budgets have left, so it can answer while it still can. |
| The model should get only a few calls to an expensive tool per run | [`ToolBudgets`](tool-budgets.md) | `ToolBudgets(per_tool={'web_search': 3}, refuse_repeats={'web_search'})` answers the fourth call, or an exact repeat, with a tool result saying the budget is used, and the run continues. `UsageLimits.tool_calls_limit` stays the hard ceiling. |
| The model repeats the same tool call without making progress | [`DetectLoops`](detect-loops.md) | The default `action='nudge'` adds a system prompt asking the model to change approach; `action='error'` stops the run with `LoopDetectedError`. |
| A tool raises an exception and the whole run ends | [`ReportToolErrors`](report-tool-errors.md) | Sends the exception to the model as a failed tool result naming the tool and the exception type, so the model can try another way. `format_error=` replaces the text, for example to include the exception's message. `ModelRetry`, `ToolFailed` and approvals keep their meaning, and `UsageLimits.request_limit` stops a model that keeps calling a failing tool. |
| The model sends malformed JSON in a tool call | [`RepairToolArguments`](repair-tool-arguments.md) | Runs `json-repair` on the arguments before validation, fixing trailing commas, single-quoted keys and truncated JSON; valid arguments pass through unchanged. Import it from `pydantic_ai_harness.repair_tool_arguments` and install the `repair-tool-arguments` extra. `Coder` includes `RepairToolArguments`, so its `coder` extra installs `json-repair` too. |

## Capabilities

Every capability is a self-contained unit you drop into `capabilities=[...]`, and they all compose, with each other and with your own. Some come with [`pydantic-ai`](../index.md) itself, the rest with this package; the **Package** column says which. 100+ in all, grouped by what they give your agent:

### Harnesses

Complete agent stacks as regular combined capabilities: one import gives you a working agent, and you can take either apart into the blocks below.

| Harness | Package | What it provides |
|---|---|---|
| [Coder](coder.md) | Harness | Six coding tools, persistent shell commands, delegation to itself, autonomous guidance, and context controls |
| [Researcher](researcher.md) | Harness | A complete web-research stack: search, page fetching, a delegated sub-researcher, and bounded tool output |

### Workspaces

<span id="execution-environments"></span>

Where the agent runs commands and keeps files: this machine, a container, or a sandbox. [Compare them](../workspace.md#supported-workspaces).

| Capability | Package | What it does |
|---|---|---|
| [Local Workspace](../workspace.md) | Core | Commands and files in a directory on this machine; no isolation. Adds no tools: pair it with the [Files & shell](#files-shell) tools or your own |
| [Docker Container](docker-container.md) | Harness | Commands and files in a local [Docker](https://www.docker.com/) or Podman container |
| [Smol Sandbox](smol-sandbox.md) | Harness | Commands and files in a [smolvm](https://github.com/smol-machines/smolvm) microVM on your own machine; no cloud account |
| [BoxLite Sandbox](boxlite-sandbox.md) | Harness | Commands and files in a [BoxLite](https://github.com/boxlite-ai/boxlite) micro-VM on your own machine, with an optional egress allow-list; no cloud account |
| [OpenSandbox](opensandbox-sandbox.md) | Harness | Commands and files in a sandbox on your own [OpenSandbox](https://open-sandbox.ai) server, on Docker or Kubernetes |
| [Kubernetes Sandbox](kubernetes-sandbox.md) | Harness | Commands and files in a sandbox pod from a Kubernetes [agent-sandbox](https://agent-sandbox.sigs.k8s.io) warm pool |
| [Modal Sandbox](modal-sandbox.md) | Harness | Commands and files in an isolated [Modal](https://modal.com) cloud sandbox |
| [E2B Sandbox](e2b-sandbox.md) | Harness | Commands and files in an isolated [E2B](https://e2b.dev) cloud sandbox |
| [Daytona Sandbox](daytona-sandbox.md) | Harness | Commands and files in an isolated [Daytona](https://www.daytona.io) cloud sandbox |
| [Render Sandbox](render-sandbox.md) | Harness | Commands and files in an isolated [Render](https://render.com/docs/sandboxes) sandbox |
| [Sprites Sandbox](sprites-sandbox.md) | Harness | Commands and files in a persistent [Fly.io Sprite](https://sprites.dev) |
| [boxd Sandbox](boxd-sandbox.md) | Harness | Commands and files in an isolated [boxd](https://boxd.sh) cloud machine |
| [SSH Workspace](ssh-workspace.md) | Harness | Commands and files on a remote host over your `ssh` client, as your SSH user; not sandboxed there |
| [Bubblewrapped Workspace](bubblewrap-sandbox.md) | Harness | Wraps another workspace capability so its commands run in a Linux [bubblewrap](https://github.com/containers/bubblewrap) wrapper on that workspace's host (best-effort convenience, not a security boundary) |

### Files & shell {#files-shell}

The tools the model uses in the run's workspace, whichever one it is.

| Capability | Package | What it does |
|---|---|---|
| [FileSystem](filesystem.md) | Harness | Read, write, edit, list, and search files under a root in the run's workspace, with opt-in ripgrep tools; path-traversal checked, secrets read-only |
| [Smart Search](smart-search.md) | Harness | Code search by behavior in the run's workspace: plain-English queries, ranked excerpts, judged by any model (a decision model over System One, or TypeSafe's Jev, recommended) |
| [Shell](shell.md) | Harness | Command execution in the run's workspace with allowlists, denylists, timeouts, credential-stripping, and opt-in commands that outlive the run |

### Tools & native abilities

Abilities the provider executes natively, and connections to any MCP server.

| Capability | Package | What it does |
|---|---|---|
| [MCP](../capabilities/mcp.md) | Core | Connect any MCP server's tools; local by default, provider-native connectors opt-in |
| [Image Generation](../capabilities/image-generation.md) | Core | Generate and edit images; provider-native where supported, sub-agent fallback elsewhere |

### Integrations

Tools from third-party services, most through the service's own hosted MCP server. The [Integrations](integrations.md) page groups them by what they connect to.

| Capability | Package | What it does |
|---|---|---|
| [GitHub](github.md) | Harness | Read and change GitHub repositories, issues, pull requests, and other accessible resources. |
| [Linear](linear.md) | Harness | Read and change Linear issues, projects, teams, and comments. |
| [Macroscope](macroscope.md) | Harness | Run a local [Macroscope](https://docs.macroscope.com/cli) code review and hand the findings to the agent |
| [LocalStack](localstack.md) | Harness | An emulated AWS environment with AWS CLI tools |
| [Google Workspace](google-workspace.md) | Harness | Use Gmail, Calendar, Drive, and other Google Workspace tools. |
| [Notion](notion.md) | Harness | Search and change Notion workspace content. |
| [Slack](slack.md) | Harness | Give an agent Slack messages, channels, and canvas tools. |
| [Day AI](day-ai.md) | Harness | Search and update CRM records and meeting context through [Day AI](https://day.ai)'s hosted MCP server |
| [Grain](grain.md) | Harness | Search and read meetings, transcripts, and notes, and create clips and tag meetings, through [Grain](https://grain.com)'s hosted MCP server |
| [Pylon](pylon.md) | Harness | Work with support issues, accounts, and contacts through [Pylon](https://www.usepylon.com)'s hosted MCP server |
| [StackOne](stackone.md) | Harness | Act on linked SaaS accounts (HRIS, ATS, CRM, …) via [StackOne](https://www.stackone.com) |
| [PostHog](posthog.md) | Harness | Query product analytics and manage feature flags, experiments, and dashboards through [PostHog](https://posthog.com)'s hosted MCP server |
| [Logfire MCP](logfire-mcp.md) | Harness | Query Logfire telemetry and manage observability resources. |
| [Ordinal](ordinal.md) | Harness | Draft, schedule, and analyze social posts through [Ordinal](https://www.tryordinal.com)'s hosted MCP server |

### Web & research

Finding and reading things on the open web.

| Capability | Package | What it does |
|---|---|---|
| [Web Search](../capabilities/web-search.md) | Core | Provider-native search where available, local DuckDuckGo fallback everywhere |
| [Web Fetch](../capabilities/web-fetch.md) | Core | Fetch and read URLs, native or local |
| [X Search](../capabilities/x-search.md) | Core | Search X; native on xAI, subagent fallback elsewhere |
| [Exa Search](exa-search.md) | Harness | Web research via [Exa](https://exa.ai): excerpted search, full-page reads, opt-in cited deep search |
| [Exa Agent](exa-search.md) | Harness | Delegate open-ended research to the Exa Agent API |
| [You.com Search](youdotcom.md) | Harness | Web search and page reads via [You.com](https://you.com): query-relevant excerpts or full-page markdown |
| [You.com Research](youdotcom.md) | Harness | Cited answers and multi-step research via the You.com Answer, Research, and Finance Research APIs |
| [Keenable Search](keenable-search.md) | Harness | Web research via [Keenable](https://keenable.ai): excerpted search and full-page reads, keyless and with no extra to install |
| [Browser Use](browser-use.md) | Harness | Hand web tasks to an autonomous [browser-use](https://github.com/browser-use/browser-use) agent driving a real browser |
| [CDP Browser](cdp-browser.md) | Harness | Drive a throwaway Chrome tab with raw DevTools Protocol commands, best from Code Mode |
| [Playwright Browser](playwright.md) | Harness | Drive a real Chromium page yourself: navigate, click, type, read, and inspect what the page did |
| [Computer Use](computer-use.md) | Harness | See and control the desktop of the run's workspace with screenshots, mouse, and keyboard |

### Reasoning, planning & delegation

How the agent thinks and divides the work.

| Capability | Package | What it does |
|---|---|---|
| [Thinking](../capabilities/thinking.md) | Core | Provider-adaptive extended thinking at configurable effort |
| [Planning](planning.md) | Harness | Model-owned task plans with a cache-safe live reminder |
| [Subagents](subagents.md) | Harness | Delegate self-contained tasks to named child agents |
| [Peers](peers.md) | Harness | Discover and message other agent sessions without controlling their runs |
| [Dynamic Workflow](dynamic-workflow.md) | Harness | The model orchestrates sub-agents from one Python script: fan-out, chain, vote in a single tool call, with hard `max_agent_calls` budgets |
| [Advisor](advisor.md) | Harness | Let an executor consult a stronger model mid-run |
| [Background Tools](background-tools.md) | Harness | Run selected tools concurrently; results arrive as follow-up messages |

### Context management

How the agent spends its context window over a long run, and how often it pays for the same tokens.

| Capability | Package | What it does |
|---|---|---|
| [Code Mode](code-mode.md) | Harness | The model writes one Python script that calls many tools inside a [Monty](https://github.com/pydantic/monty) sandbox: one round-trip instead of N, and intermediate results never enter the context window |
| [IPython Code Mode](ipython-code-mode.md) | Harness | Code Mode with a persistent [IPython](https://ipython.org/) kernel in place of the Monty sandbox: full CPython and any installed package, with state kept between calls. For local agents only; not a sandbox |
| [Tool Bridge](tool-bridge.md) | Harness | Keep the tool list the model sees fixed behind `tool_search`, `tool_describe` and `tool_call`, so the prompt cache holds however many tools sit behind them |
| [Tool Search](../capabilities/tool-search.md) | Core | Load tool definitions on demand instead of carrying hundreds in every prompt |
| [Compaction](../capabilities/compaction.md) | Core | Provider-native compaction on OpenAI and Anthropic; the provider summarizes history server-side |
| [Compaction](compaction.md) | Harness | Model-agnostic strategies: tool-result clearing, sliding-window trimming, LLM summarization, tiered; all window-relative, with live usage reporting |
| [Tool Output Limits](tool-output-limits.md) | Harness | Truncate, spill to a queryable file, or summarize oversized tool returns at the source |
| [Warn On Cache Busts](warn-on-cache-busts.md) | Harness | Detect prompt-cache prefix collapses between requests, from the provider's own numbers |
| [Label Cross Model History](label-cross-model-history.md) | Harness | Tell the model when earlier assistant turns came from another model, after a model switch or handoff |

### Knowledge & memory

What the agent knows and remembers, loaded when relevant instead of carried in every prompt.

| Capability | Package | What it does |
|---|---|---|
| [Memory](memory.md) | Harness | A persistent, namespaced notebook: bounded prompt injection, on-demand search; in-memory/file/Postgres stores |
| [Pixeltable](pixeltable.md) | Harness | Read-only, allowlisted tools over Pixeltable tables and embedding indexes, plus a Pixeltable `MemoryStore` |
| [Conversation Search](conversation-search.md) | Harness | BM25 search over stored history, including turns compaction dropped |
| [Skills](skills.md) | Harness | Load [Agent Skill](../capabilities/on-demand.md) (`SKILL.md`) instructions on demand |
| [Repo Context](repo-context.md) | Harness | Start runs oriented: `AGENTS.md`/`CLAUDE.md` + repository structure |
| [Report Stale Files](report-stale-files.md) | Harness | Tell the model when a file it read or wrote has since changed or been deleted in the workspace |
| [Pydantic AI Docs](pydantic-ai-docs.md) | Harness | On-demand Pydantic AI documentation lookup |
| [Current Time](current-time.md) | Harness | Today's date, or the current hour, in a time zone you pick, without breaking the prompt cache |

### Control & safety

Bounding what the agent may do, and keeping it on-instructions.

| Capability | Package | What it does |
|---|---|---|
| [Repair Tool Arguments](repair-tool-arguments.md) | Harness | Repair malformed JSON tool arguments before schema validation. |
| [Guardrails](guardrails.md) | Harness | Validate/block/redact user input, tool calls, tool results, and output, including secret masking and parallel async guards |
| [Goal](goal.md) | Harness | Require a caller-defined completion check before an unattended run ends |
| [Prompt Injection Defender](prompt-injection-defender.md) | Harness | Classify local tool results for indirect prompt injection and optionally withhold high-risk results |
| [Artifact Scanning](artifact-scanning.md) | Harness | Scan memory files and skills for prompt injection, credential exfiltration, and hidden content before they are stored or loaded |
| [Spend Limits](spend.md) | Harness | Cross-window USD/token budgets and per-response cost tracking, per model and per tenant |
| [Ask User](ask-user.md) | Harness | Let the model ask the user multiple-choice questions mid-run; you supply the answerer (terminal, web, test) |
| [Deny Unanswered](deny-unanswered.md) | Harness | Deny approval and question calls that nobody answered by the end of an unattended run, in parent and child agents |
| [Tool approval](../deferred-tools.md#human-in-the-loop-tool-approval) | Core | Flag tool calls that need human approval before they run |
| [Tool Call Judge](tool-call-judge.md) | Harness | A second model answers one risk question per tool call and blocks the call before the tool function runs |
| [Approval Reviewer](approval-reviewer.md) | Harness | A second model answers one risk question per call waiting for approval, approves the low-risk ones, and leaves the rest to a person |
| [Permissions](permissions.md) | Harness | Ask a person before file writes and commands, by permission mode: full access, auto (asks before risky commands), or read-only; allow, ask about, or deny tool calls and shell commands by rule |
| [Handle Deferred Tool Calls](../capabilities/handle-deferred-tool-calls.md) | Core | Resolve approvals, answers and external results on deferred tool calls inside the run, in place or by waiting for them, durably on Temporal, DBOS and AWS Lambda |
| [System Reminders](system-reminders.md) | Harness | Cache-safe re-injection of guidance mid-run to counter instruction fade |
| [Detect Loops](detect-loops.md) | Harness | Notice repeated tool calls, results, alternations, or responses, or a response repeating itself as it streams, and nudge the model, stop the run, or call your handler |
| [Report Tool Errors](report-tool-errors.md) | Harness | Send an exception a tool raises to the model as a failed tool result, so the run continues |
| [Tool Budgets](tool-budgets.md) | Harness | Cap calls per tool and per run, and refuse exact repeats, with a tool result the model can act on instead of an exception |
| [Report Remaining Budget](report-remaining-budget.md) | Harness | Tell the model on every request how many requests, tool calls, tokens and dollars its usage limits and spend budgets have left |
| [Trajectory Judge](trajectory-judge.md) | Harness | A second model reviews the live run every N requests over a sliding token window and steers it mid-run |
| [Refine Output](refine-output.md) | Harness | Send the agent back to review and revise its output before the run ends |
| [Require Verification](require-verification.md) | Harness | Send the agent back to run its checks when it tries to finish after editing code without a passing check since its last edit |
| [Wrap Up At Limits](wrap-up-at-limits.md) | Harness | End the run with a final answer, not `UsageLimitExceeded`, when a usage limit would stop it |

### Self-extension

| Capability | Package | What it does |
|---|---|---|
| [Capability Creation](capability-creation.md) | Harness | The agent writes, validates, and persists *new capabilities* during a run; the host loads them into a later run via `load_active(workspace)` and `capabilities=` |

### Execution runtime

Outside the loop: how runs persist, survive failures, and get observed and configured in production.

| Capability | Package | What it does |
|---|---|---|
| [Durable execution](../durable_execution/overview.md) | Core | Runs that survive restarts and failures, on nine engines: [Temporal](../durable_execution/temporal.md), [DBOS](../durable_execution/dbos.md), [Prefect](../durable_execution/prefect.md), [Restate](../durable_execution/restate.md), [Airflow](../durable_execution/airflow.md), and the Harness engines AWS Lambda, Absurd, pgtask, and Azure Durable Task below. See the [workspace guide](../workspace.md#durable-execution) for Coder, Shell, and FileSystem |
| [AWS Lambda durability](aws-lambda.md) | Harness | Checkpoint model requests and tool calls into AWS Lambda durable function steps |
| [Absurd durability](absurd.md) | Harness | Checkpoint model requests, MCP calls, and tool calls into Absurd steps on PostgreSQL |
| [pgtask durability](pgtask.md) | Harness | Checkpoint model requests, MCP calls, and tool calls into pgtask steps on PostgreSQL |
| [Azure Durable Task durability](azure-durable-task.md) | Harness | Run model requests, tool calls and durable operations as activities in Durable Task Scheduler orchestrations |
| [Render Workflows](render-workflows.md) | Harness | Run model requests and tool calls as chained task runs with separate retries, timeouts, and compute settings; retrying the task that runs the agent can repeat completed work |
| [Step Persistence](step-persistence.md) | Harness | Save, restore, resume (`continue_run`), and fork (`fork_run`) runs; file/SQLite/Mongo backends |
| [Model Fallback](../capabilities/model-fallback.md) | Core | Fall back to other models when a model request fails or its response is rejected, within the run's own request step |
| [Model Router](model-router.md) | Harness | Let another model pick the run's model from a named menu, once per run or per step, with a declared default |
| [Instrumentation](../capabilities/instrumentation.md) | Core | OpenTelemetry GenAI spans for every model and tool call; the raw material for [Logfire](https://pydantic.dev/logfire) traces |
| [Managed Prompt](managed-prompt.md) | Harness | Back instructions with a [Logfire](https://pydantic.dev/logfire)-managed prompt; version and roll out without redeploying |
| [Thread Executor](../capabilities/thread-executor.md) | Core | Run sync tools on a shared thread pool |

Core also ships loop-customization capabilities for production servers: [Select Model](../capabilities/select-model.md), [Resolve Model ID](../capabilities/resolve-model-id.md), [Prepare Tools / Prepare Output Tools](../capabilities/prepare-tools.md), [Prefix Tools](../capabilities/prefix-tools.md), [Set Tool Metadata](../capabilities/set-tool-metadata.md), [Include Tool Return Schemas](../capabilities/include-tool-return-schemas.md), [Process History](../capabilities/process-history.md), [Process Event Stream](../capabilities/process-event-stream.md), [Reinject System Prompt](../capabilities/reinject-system-prompt.md), and [Raise Content Filter Error](../capabilities/raise-content-filter-error.md).

And the agent plugs into any interface: [ACP](acp.md) *(experimental, Harness)* serves it to editors like Zed over the [Agent Client Protocol](https://agentclientprotocol.com), and core ships the [web chat UI](../web.md), [CLI](../cli.md), [frontend adapters](../ui/overview.md) (AG-UI, Vercel AI), and [realtime voice](../realtime/overview.md).

Community packages extend the same capability system further; see [third-party capabilities](../capabilities/third-party.md).

## When do you need the Harness?

"Harness" is the field's term for everything around the model that turns it into an agent: the loop, the tools, the context management. Reach for this package when your agent should *do* more than core's lean harness covers: touch files, run code, browse, remember, delegate, or stay coherent through hours-long runs. The boundary between the packages is mechanical, not a maturity tier: core ships the capabilities that require model or framework support (provider-native tools like [image generation](../capabilities/image-generation.md), provider APIs like [compaction](../capabilities/compaction.md), deep loop integration like [tool search](../capabilities/tool-search.md), and fundamentals like [thinking](../capabilities/thinking.md), [MCP](../capabilities/mcp.md), and [web search](../capabilities/web-search.md)) and the Harness ships everything else, as a separate package so capabilities can iterate at the speed the field moves while Pydantic AI itself stays lean.

## Installation

```bash
pip/uv-add pydantic-ai-harness
```

This installs [`pydantic-ai-slim`](../install.md) with it, so it works on its own; you don't need to install Pydantic AI separately. Model providers and the CLI come via extras that pass through to Pydantic AI: `pydantic-ai-harness[anthropic]`, `[cli]`. Some capabilities need their own extra for optional dependencies; each capability's page gives its exact install line. Requires Python 3.11+.

New to Pydantic AI itself? Start with [its docs](../index.md): the agent you mount these capabilities on is defined there.

## Observability

Core's [Instrumentation](../capabilities/instrumentation.md) capability (or `logfire.instrument_pydantic_ai()`) emits a full trace of every run: every model call and tool call, with token and cost tracking. It's standard OpenTelemetry, so any OTLP backend works; run `uvx logfire-cli wizard --task pydantic-ai` once to [set up Logfire](../logfire.md#using-logfire).

## Build your own

[Capabilities](../capabilities/custom.md) are the primary extension point for Pydantic AI, and every capability in this library doubles as a worked example. Publishing a standalone package? Use the `pydantic-ai-<name>` naming convention; see [Publishing capability packages](../extensibility.md#publishing-capability-packages).

## Version policy

Pydantic AI Harness uses **0.x versioning** to signal API stability, not maturity: the capabilities are tested end-to-end and meant for production use, but their APIs may change between minor releases (0.1 -> 0.2) through renamed parameters, changed defaults, or restructured APIs, with deprecation warnings where practical. Patch releases will not intentionally break existing behavior, and every breaking change is documented in release notes with migration guidance your agent can follow. Keeping the Harness a separate package from [Pydantic AI](https://github.com/pydantic/pydantic-ai), which has a [stricter version policy](../version-policy.md), lets capabilities change faster than core.

On 0.x, each Harness release ships alongside the Pydantic AI release with the same minor and patch number (Harness 0.51.0 ships with Pydantic AI 2.51.0) and pins that exact `pydantic-ai-slim` version, so upgrading the Harness upgrades Pydantic AI with it and every installable pair has been tested together.

### After 1.0

From 1.0, the Harness follows the rules of Pydantic AI's [version policy](../version-policy.md): no intentional breaking changes in minor releases, and deprecated functionality is not removed until the next major version. The changes that policy does not consider breaking, such as bug fixes for code that relied on undocumented behavior, OpenTelemetry span attributes, and `__repr__` output, are not breaking for the Harness either. Neither is a change that follows a third-party service or SDK a capability integrates with, when that service removes or changes what the capability relied on. Capabilities marked experimental on their page (currently [ACP](acp.md)) are excluded until they graduate.

A capability's behavior includes what it sends to the model, so stability covers more than signatures:

- **Defaults**: a minor release does not change a default parameter value in a way that changes what an existing agent does, such as enabling a tool, lowering a limit, or switching a fallback. New behavior arrives behind a new parameter, and a default changes only in a major release.
- **Stored history**: tool names, argument schemas, and the other records a capability writes into message history and reads back on later runs, such as metadata keys and framing tags, persist in histories that you may store and resume. A minor release can add optional tool arguments but does not rename or remove a tool, an argument, or a record a capability relies on.
- **Prompt text**: the wording of instructions, tool descriptions, and other text a capability sends to the model can change in a minor release, since improving how models use a capability is part of maintaining it. Release notes list these changes. Where a capability accepts its own instructions or descriptions as parameters, passing them keeps your wording fixed.

From 1.0, Harness version numbers no longer mirror Pydantic AI's, and a Harness major release does not require a Pydantic AI major release. Every Harness release still pins the exact `pydantic-ai-slim` release it was tested with, and moving that pin to a new Pydantic AI major version is itself a Harness major release, so upgrading the Harness within a major version does not cross a Pydantic AI major. The pin is listed in the package metadata on PyPI, and the release notes name the Pydantic AI release each Harness release ships with.

## Further reading

- [The harness thesis](https://pydantic.dev/articles/the-harness-thesis): why long-horizon agents need a harness that makes their work observable, governable and reusable
- [What makes a good harness](https://pydantic.dev/articles/what-makes-a-good-harness): getting the right text to the model at the right moment, with capabilities, code mode and just-in-time tools

## Pydantic AI references

- [Capabilities](../capabilities/overview.md): what capabilities are, built-in capabilities, building your own
- [Hooks](../hooks.md): lifecycle hooks reference, ordering, error handling
- [Extensibility](../extensibility.md): publishing packages, third-party ecosystem
- [Toolsets](../toolsets.md): building tools for capabilities
- [API reference](../api/capabilities.md): full API docs
