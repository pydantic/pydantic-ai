---
title: Pydantic AI Harness
description: The official capability library for Pydantic AI -- pick-and-choose batteries that turn your agent into a coding agent, research assistant, or anything else.
---

# Pydantic AI Harness

**The batteries for your [Pydantic AI](/ai/) agent.**

Pydantic AI's [capabilities](/ai/core-concepts/capabilities/) and [hooks](/ai/core-concepts/hooks/) API is how you give an agent its harness -- bundles of tools, lifecycle hooks, instructions, and model settings that extend what the agent can do without any framework changes.

**Pydantic AI Harness** is the official capability library for Pydantic AI, maintained by the [Pydantic AI](https://github.com/pydantic/pydantic-ai) team. Pydantic AI core ships the capabilities that require model or framework support, plus the ones fundamental to every agent -- [web search](/ai/core-concepts/capabilities/#provider-adaptive-tools), [tool search](/ai/tools-toolsets/deferred-tools/), [thinking](/ai/core-concepts/capabilities/#thinking). Everything else lives here: standalone building blocks you pick and choose to turn your agent into a coding agent, a research assistant, or anything else. This is also where new capabilities start -- as they stabilize and prove themselves broadly essential, they can graduate into core.

## What goes where?

Pydantic AI core ships the agent loop, model providers, the capabilities/hooks abstraction, and two kinds of capabilities:

- **Capabilities that require model or framework support** -- anything backed by provider native tools (like [image generation](/ai/core-concepts/capabilities/#provider-adaptive-tools)), provider-specific APIs (like compaction via the OpenAI or Anthropic APIs), or deep agent graph integration. These go hand-in-hand with model class code and need to ship together.
- **Capabilities that are fundamental to the agent experience** -- things nearly every agent benefits from, like [web search](/ai/core-concepts/capabilities/#provider-adaptive-tools), [tool search](/ai/tools-toolsets/deferred-tools/), and [thinking](/ai/core-concepts/capabilities/#thinking). These feel like qualities of the agent itself, not accessories.

**Pydantic AI Harness** is where everything else lives: standalone capabilities that make specific categories of agents powerful, or that are still finding their final shape. Context management, memory, guardrails, file system access, code execution, multi-agent orchestration -- these are the building blocks you pick and choose based on what your agent needs to do.

The harness is also where new capabilities *start*. It ships as a separate package so capabilities can iterate faster without the strict backward-compatibility requirements of core. As a capability stabilizes and proves itself broadly essential, it can graduate into core -- [code mode](code-mode.md) is an early candidate.

Many capabilities benefit from a "fall up" pattern: they typically start as a local implementation that works with every model, then gain provider-native support that uses the provider's built-in API when available -- auto-switching between the two. This is how [web search](/ai/core-concepts/capabilities/#provider-adaptive-tools), [web fetch](/ai/core-concepts/capabilities/#provider-adaptive-tools), and [image generation](/ai/core-concepts/capabilities/#provider-adaptive-tools) already work in core, and the same approach is coming for skills, code mode, and context compaction.

## Installation

```bash
uv add pydantic-ai-harness
```

Some capabilities need an extra to pull in their optional dependencies:

```bash
uv add "pydantic-ai-harness[codemode]"          # Code Mode (adds the Monty sandbox)
uv add "pydantic-ai-harness[dynamic-workflow]"  # Dynamic Workflow (adds the Monty sandbox)
uv add "pydantic-ai-harness[logfire]"           # Managed Prompt (Logfire-managed prompts)
uv add "pydantic-ai-harness[acp]"               # ACP (Agent Client Protocol SDK)
```

The `code-mode` extra is also supported as an alias for `codemode`.

Requires Python 3.10+ and `pydantic-ai-slim>=2.1.0`.

## Quick start

Install the harness alongside the Pydantic AI extras this example uses:

```bash
uv add "pydantic-ai-slim[anthropic,mcp,duckduckgo,logfire]" "pydantic-ai-harness[code-mode]"
```

```python
import logfire
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, WebSearch
from pydantic_ai_harness import CodeMode

# See https://pydantic.dev/docs/ai/integrations/logfire/ for setup details.
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[
        # Wraps every tool into a single run_code tool, sandboxed by Monty
        # (https://github.com/pydantic/monty -- pulled in by the [code-mode] extra).
        # The model writes Python that calls multiple tools with loops, conditionals,
        # asyncio.gather, and local filtering -- one model round-trip for N tool calls.
        CodeMode(),
        # Connect to any MCP server -- here, the open-source Hacker News server
        # (https://github.com/cyanheads/hn-mcp-server). native=False forces the
        # local MCP toolset so CodeMode can wrap the tools; without it,
        # providers that natively support MCP server connectors execute the tools
        # server-side and bypass the sandbox.
        MCP('https://hn.caseyjhand.com/mcp', native=False),
        # Provider-adaptive web search; native=False routes through the local
        # DuckDuckGo fallback (the [duckduckgo] extra above) so CodeMode can batch
        # web searches alongside the HN calls in a single run_code.
        WebSearch(native=False),
    ],
)

result = agent.run_sync(
    "Across the top, best, and 'show HN' Hacker News feeds, find the most-discussed "
    "story with at least 100 points. Pull its comment thread, its submitter's profile, "
    "and any web coverage. Summarize what you find in one paragraph."
)
print(result.output)
"""
The most-discussed HN story across top/best/show clearing 100 points is "Vibe coding
and agentic engineering are getting closer than I'd like" by Simon Willison (748 points,
853 comments, on the Best feed), submitted by long-time HNer e12e. The piece argues
that the two modes Willison once kept mentally separate -- throwaway "vibe coding" and
disciplined "agentic engineering" -- are blurring, since agents like Claude Code now
reliably handle non-trivial tasks like "build a JSON API endpoint that runs a SQL query"
with tests and docs on the first pass. The HN thread is unusually substantive, with
commenters debating whether LLMs created or merely *exposed* sloppy engineering
practices and warning of a "normalization of deviance" as engineers stop reviewing diffs.
"""
```

[![Logfire trace from the Quick start run](images/quick-start-trace.png)](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946)

**[See this run as a public Logfire trace ->](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946)** Each `run_code` span fans out into the tool calls the model issued from inside the sandbox -- it's the easiest way to understand what code mode actually did.

## Capabilities

Each capability is a self-contained battery you drop into an agent's `capabilities=[...]` list. They compose with each other and with Pydantic AI's [built-in capabilities](/ai/core-concepts/capabilities/).

| Capability | What it does | Extra |
|---|---|---|
| [Code Mode](code-mode.md) | Wraps the agent's tools into a single `run_code` tool, sandboxed by [Monty](https://github.com/pydantic/monty). The model writes Python that calls the tools as functions -- with loops, conditionals, `asyncio.gather`, and local filtering -- collapsing N tool calls into one model round-trip. | `codemode` |
| [FileSystem](filesystem.md) | Sandboxed file access scoped to a root directory: read, write, edit, search, and find files. Rejects path traversal above the root, resolves symlinks before authorizing, and keeps `.git/`, `.env`, key files, and secrets read-only by default. | -- |
| [Shell](shell.md) | Command execution in a subprocess rooted at a working directory, gated by allowlists, denylists, timeouts, and optional environment-variable stripping (including a preset for common LLM provider credentials). | -- |
| [Context](context.md) | Auto-loads repo context -- `CLAUDE.md`/`AGENTS.md` and repository structure -- so the agent starts a run already oriented in the project. | -- |
| [Pydantic AI Docs](pydantic-ai-docs.md) | An on-demand `read_pyai_docs` tool that pulls Pydantic AI documentation into the run when the agent needs it, instead of preloading it. | -- |
| [Compaction](compaction.md) | Keeps a run within token limits: sliding-window trimming, LLM-powered summarization of older messages, and warnings before the context or iteration ceiling is hit. | -- |
| [Overflowing Tool Output](overflowing-tool-output.md) | Reduces an oversized tool return when it is produced -- truncate, spill to a queryable file, or summarize -- so a large payload does not persist in history and get re-sent every request. | -- |
| [Step Persistence](step-persistence.md) | Saves and restores full conversation state; snapshot, resume (`continue_run`), and fork (`fork_run`) a run. | -- |
| [Media](media.md) | Offloads large `BinaryContent` to content-addressed stores (local or S3) so big media does not bloat message history. | -- |
| [Subagents](subagents.md) | Delegates subtasks to specialized child agents through a delegate tool. | -- |
| [Dynamic Workflow](dynamic-workflow.md) | Orchestrates sub-agents from a model-written Python script -- fan-out, chaining, and voting in a single tool call. | `dynamic-workflow` |
| [Planning](planning.md) | Breaks a complex task into a structured plan before execution and tracks progress against it. | -- |
| [Memory](memory.md) | Gives an agent a persistent, namespaced notebook with bounded prompt injection, on-demand search, and concurrency-safe stores. | -- |
| [Runtime Authoring](runtime-authoring.md) | Lets an agent author, validate, and load real capabilities at runtime. | -- |
| [Guardrails](guardrails.md) | Validates user input before a run starts and model output after it completes -- block or redact, with structured results. | -- |
| [Managed Prompt](managed-prompt.md) | Backs an agent's instructions with a [Logfire-managed prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/), so you can version, label, and roll out prompt changes from the Logfire UI without redeploying -- with a code default that keeps the agent working when no remote value is available. | `logfire` |
| [ACP](acp.md) *(experimental)* | Serves an agent to editors (Zed, etc.) over the [Agent Client Protocol](https://agentclientprotocol.com) -- streamed text, diff-rendered edits, and tool approval. | `acp` |

Most capabilities are stable within the [version policy](#version-policy) below. [ACP](acp.md) is the exception -- it is still experimental, imported from `pydantic_ai_harness.experimental.acp`, and may change or be removed in a future release.

## Build your own

[Capabilities](/ai/core-concepts/capabilities/#building-custom-capabilities) are the primary extension point for Pydantic AI. Any of the capabilities in this library can serve as a reference for building your own.

Publishing as a standalone package? Use the `pydantic-ai-<name>` naming convention -- see [Publishing capability packages](/ai/guides/extensibility/#publishing-capability-packages).

## Version policy

Pydantic AI Harness uses **0.x versioning** to signal that APIs are still stabilizing. During 0.x, minor releases (0.1 -> 0.2) may include breaking changes -- renamed parameters, changed defaults, restructured APIs -- while patch releases (0.1.0 -> 0.1.1) will not intentionally break existing behavior. All breaking changes are documented in release notes with migration guidance. This is why the harness is a separate package from [Pydantic AI](https://github.com/pydantic/pydantic-ai), which has a [stricter version policy](/ai/project/version-policy/). As the core capabilities stabilize, the library will move toward 1.0 with matching stability guarantees.

## Pydantic AI references

- [Capabilities](/ai/core-concepts/capabilities/) -- what capabilities are, built-in capabilities, building your own
- [Hooks](/ai/core-concepts/hooks/) -- lifecycle hooks reference, ordering, error handling
- [Extensibility](/ai/guides/extensibility/) -- publishing packages, third-party ecosystem
- [Toolsets](/ai/tools-toolsets/toolsets/) -- building tools for capabilities
- [API reference](/ai/api/pydantic-ai/capabilities/) -- full API docs
