# Pydantic Harness

[![CI](https://github.com/pydantic/pydantic-harness/actions/workflows/main.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-harness/actions/workflows/main.yml?query=branch%3Amain)
[![PyPI](https://img.shields.io/pypi/v/pydantic-harness.svg)](https://pypi.python.org/pypi/pydantic-harness)
[![versions](https://img.shields.io/pypi/pyversions/pydantic-harness.svg)](https://github.com/pydantic/pydantic-harness)
[![license](https://img.shields.io/github/license/pydantic/pydantic-harness.svg)](https://github.com/pydantic/pydantic-harness/blob/main/LICENSE)

**The batteries for your [Pydantic AI](https://ai.pydantic.dev/) agent.**

---

Pydantic AI's [capabilities](https://ai.pydantic.dev/capabilities/) and [hooks](https://ai.pydantic.dev/hooks/) API is how you give an agent its harness -- bundles of tools, lifecycle hooks, instructions, and model settings that extend what the agent can do without any framework changes.

**Pydantic Harness** is the official capability library for Pydantic AI, maintained by the [Pydantic AI](https://github.com/pydantic/pydantic-ai) team. Pydantic AI core ships capabilities that require model or framework support, and capabilities fundamental to every agent -- [web search](https://ai.pydantic.dev/capabilities/#provider-adaptive-tools), [tool search](https://ai.pydantic.dev/deferred-tools/), [thinking](https://ai.pydantic.dev/capabilities/#thinking). Everything else lives here: standalone building blocks you pick and choose to turn your agent into a coding agent, a research assistant, or anything else. This is also where new capabilities start -- as they stabilize and prove themselves broadly essential, they can graduate into core.

The [capability matrix](#capability-matrix) tracks where we are. [Tell us what to prioritize.](#help-us-prioritize)

## Installation

```bash
uv add pydantic-harness
```

Extras for specific capabilities:

```bash
uv add "pydantic-harness[code-mode]"   # CodeMode (adds the Monty sandbox)
```

Requires Python 3.10+ and `pydantic-ai-slim>=1.76.0`.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP  # from the core pydantic-ai package
from pydantic_harness import CodeMode

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        MCP('https://api.githubcopilot.com/mcp/'),
        CodeMode(),
    ],
)

result = agent.run_sync('Rank the open PRs on pydantic/pydantic-harness by thumbs-up reactions. Which 5 should we merge first?')
```

[`MCP`](https://ai.pydantic.dev/capabilities/#provider-adaptive-tools) (from the core `pydantic-ai` package) connects your agent to any MCP server -- here, [GitHub's official MCP server](https://github.com/github/github-mcp-server).

[`CodeMode`](code_mode/) wraps all tools into a single `run_code` tool powered by our [Monty](https://github.com/pydantic/monty) sandbox, so the model can orchestrate multiple tool calls with Python code instead of one model round-trip per call.

## Capability matrix

We studied the leading coding agents and AI harnesses -- [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [OpenClaw](https://github.com/open-claw/open-claw), [Goose](https://block.github.io/goose/), [Hermes](https://github.com/NousResearch/hermes-agent), [OpenAI Codex](https://openai.com/index/codex/), [SWE-agent](https://github.com/SWE-agent/SWE-agent), [OpenHands](https://github.com/All-Hands-AI/OpenHands), and [Mastra](https://mastra.ai/) -- to map every capability area that matters for production agents. Each one is tracked as an [issue](https://github.com/pydantic/pydantic-harness/issues) in this repo.

**Vote on whatever is linked in the Status column** -- PRs if we're actively building it, issues if it's planned -- to help us decide what to work on next.

| Category | Capability | Description | Status | Community&nbsp;alternatives |
|---|---|---|---|---|
| **Tools &&nbsp;execution** | **Code mode** | Sandboxed Python execution via [Monty](https://github.com/pydantic/monty) -- one `run_code` call replaces N tool calls | :white_check_mark: [Docs](code_mode/) | |
| | **Tool search** | Progressive tool discovery for large tool sets | :white_check_mark: [Pydantic&nbsp;AI](https://ai.pydantic.dev/deferred-tools/) | |
| | **File system** | Read, write, edit, search files with path traversal prevention | :construction: [PR&nbsp;#139](https://github.com/pydantic/pydantic-harness/pull/139) | [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) (vstorm&#8209;co) |
| | **Shell** | Execute commands with allowlists, denylists, and timeouts | :construction: [PR&nbsp;#139](https://github.com/pydantic/pydantic-harness/pull/139) | [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) (vstorm&#8209;co) |
| | **Repo context injection** | Auto-load CLAUDE.md/AGENTS.md and repo structure | :construction: [PR&nbsp;#154](https://github.com/pydantic/pydantic-harness/pull/154) | [pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents) (vstorm&#8209;co) |
| | **Verification loop** | Run tests after edits, auto-fix failures | :construction: [PR&nbsp;#160](https://github.com/pydantic/pydantic-harness/pull/160) | |
| **Context management** | **Sliding window** | Trim conversation history to stay within token limits | :construction: [PR&nbsp;#140](https://github.com/pydantic/pydantic-harness/pull/140) | [summarization-pydantic-ai](https://github.com/vstorm-co/summarization-pydantic-ai) (vstorm&#8209;co) |
| | **Context compaction** | LLM-powered summarization of older messages | :construction: [PR&nbsp;#140](https://github.com/pydantic/pydantic-harness/pull/140) | [summarization-pydantic-ai](https://github.com/vstorm-co/summarization-pydantic-ai) (vstorm&#8209;co) |
| | **Limit warnings** | Warn agent before hitting context/iteration limits | :construction: [PR&nbsp;#140](https://github.com/pydantic/pydantic-harness/pull/140) | [summarization-pydantic-ai](https://github.com/vstorm-co/summarization-pydantic-ai) (vstorm&#8209;co) |
| | **Tool output management** | Truncate, summarize, or spill large tool outputs | :construction: [PR&nbsp;#131](https://github.com/pydantic/pydantic-harness/pull/131) | |
| | **System reminders** | Inject periodic reminders to counteract instruction drift | :construction: [PR&nbsp;#135](https://github.com/pydantic/pydantic-harness/pull/135) | |
| **Memory &&nbsp;persistence** | **Memory** | Persistent key-value memory across sessions | :construction: [PR&nbsp;#137](https://github.com/pydantic/pydantic-harness/pull/137) | [memv](https://github.com/vstorm-co/memv) (vstorm&#8209;co) |
| | **Session persistence** | Save and restore full conversation state | :construction: [PR&nbsp;#153](https://github.com/pydantic/pydantic-harness/pull/153) | |
| **Agent orchestration** | **Sub-agents** | Delegate subtasks to specialized child agents | :construction: [PR&nbsp;#138](https://github.com/pydantic/pydantic-harness/pull/138) | [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai) (vstorm&#8209;co) |
| | **Skills** | Progressive tool loading -- search, activate, deactivate | :construction: [PR&nbsp;#133](https://github.com/pydantic/pydantic-harness/pull/133) | [pydantic-ai-skills](https://github.com/DougTrajano/pydantic-ai-skills) (DougTrajano) |
| | **Planning** | Break complex tasks into structured plans before execution | :construction: [PR&nbsp;#136](https://github.com/pydantic/pydantic-harness/pull/136) | |
| | **Task tracking** | Track tasks, subtasks, and dependencies | :memo: [#65](https://github.com/pydantic/pydantic-harness/issues/65) | [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) (vstorm&#8209;co) |
| **Safety &&nbsp;guardrails** | **Input guardrails** | Validate user input before the agent run starts | :construction: [PR&nbsp;#134](https://github.com/pydantic/pydantic-harness/pull/134) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Output guardrails** | Validate model output after the run completes | :construction: [PR&nbsp;#134](https://github.com/pydantic/pydantic-harness/pull/134) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Cost/token budgets** | Enforce token and cost limits per run | :construction: [PR&nbsp;#134](https://github.com/pydantic/pydantic-harness/pull/134) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Tool access control** | Block tools or require approval before execution | :construction: [PR&nbsp;#134](https://github.com/pydantic/pydantic-harness/pull/134) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Async guardrails** | Run validation concurrently with model requests | :construction: [PR&nbsp;#134](https://github.com/pydantic/pydantic-harness/pull/134) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Secret masking** | Detect and redact secrets in agent I/O | :construction: [PR&nbsp;#157](https://github.com/pydantic/pydantic-harness/pull/157) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Approval workflows** | Require human approval for sensitive operations | :construction: [PR&nbsp;#156](https://github.com/pydantic/pydantic-harness/pull/156) | |
| | **Tool budget** | Limit total tool calls or cost per run | :construction: [PR&nbsp;#161](https://github.com/pydantic/pydantic-harness/pull/161) | |
| **Reliability** | **Stuck loop detection** | Detect and break out of repetitive agent loops | :construction: [PR&nbsp;#130](https://github.com/pydantic/pydantic-harness/pull/130) | |
| | **Tool error recovery** | Retry failed tool calls with backoff and budget | :construction: [PR&nbsp;#158](https://github.com/pydantic/pydantic-harness/pull/158) | |
| | **Tool orphan repair** | Fix orphaned tool calls in conversation history | :construction: [PR&nbsp;#132](https://github.com/pydantic/pydantic-harness/pull/132) | |
| **Reasoning** | **Adaptive reasoning** | Adjust thinking effort based on task complexity | :construction: [PR&nbsp;#155](https://github.com/pydantic/pydantic-harness/pull/155) | |
| | **Current time** | Inject current date/time into system prompt | :construction: [PR&nbsp;#159](https://github.com/pydantic/pydantic-harness/pull/159) | |

> Packages by [vstorm-co](https://github.com/vstorm-co) are endorsed by the Pydantic AI team. We're working with them to upstream some of their implementations into this repo.

## Help us prioritize

**Vote on whatever is linked in the Status column above.** If there's a PR, vote on the PR -- it means we're actively building it. If there's only an issue, vote on the issue.

Want something that's not on the list? [Open a capability request](https://github.com/pydantic/pydantic-harness/issues/new?template=capability-request.yml).

## Build your own

[Capabilities](https://ai.pydantic.dev/capabilities/#building-custom-capabilities) are the primary extension point for Pydantic AI. Any of the existing capabilities in this repo can serve as a reference for building your own.

**Publishing as a standalone package?** Use the `pydantic-ai-<name>` naming convention. See [Publishing capability packages](https://ai.pydantic.dev/extensibility/#publishing-capability-packages).

## Contributing

We welcome capability contributions. Here's how:

1. **Start with an issue.** [Open a capability request](https://github.com/pydantic/pydantic-harness/issues/new?template=capability-request.yml) describing the behavior you want. This lets us discuss the approach and priority before code is written -- we can close an approach without closing the problem.
2. **Then open a PR.** Once the issue exists, you're welcome to open a PR with an implementation. Link the issue in your PR. We review based on community interest -- upvotes on both the issue and PR count.
3. **Don't chase green CI.** Get the approach working, then let us know. We'll take it from there -- we may push to your branch, rewrite, or open a follow-up PR. You'll be credited as the original author. (See the [Pydantic AI contributing guide](https://github.com/pydantic/pydantic-ai/blob/main/CONTRIBUTING.md).)

> **Note**: PRs that modify `pyproject.toml` or `uv.lock` from non-team members are auto-closed by CI to prevent supply chain risk. If you need a new dependency, [open an issue](https://github.com/pydantic/pydantic-harness/issues/new).

### Development

```bash
make install   # install dependencies
make format    # ruff format
make lint      # ruff check
make typecheck # pyright strict
make test      # pytest
make testcov   # pytest with 100% branch coverage
```

## Pydantic AI references

- [Capabilities](https://ai.pydantic.dev/capabilities/) -- what capabilities are, built-in capabilities, building your own
- [Hooks](https://ai.pydantic.dev/hooks/) -- lifecycle hooks reference, ordering, error handling
- [Extensibility](https://ai.pydantic.dev/extensibility/) -- publishing packages, third-party ecosystem
- [Toolsets](https://ai.pydantic.dev/toolsets/) -- building tools for capabilities
- [API reference](https://ai.pydantic.dev/api/capabilities/) -- full API docs

## License

MIT -- see [LICENSE](LICENSE).
