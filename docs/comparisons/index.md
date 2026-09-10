# Pydantic AI vs other agent frameworks

Pydantic AI is the de facto, Pythonic way to write agents in Python. These pages compare it against
the agent frameworks people actually reach for — **their best vs our best, today**, with every claim
runnable on your laptop and every version pinned.

We ship **primitives, not decisions**: typed, composable seams (capabilities, events, dependencies,
durability engines) that you wire yourself. We don't pick your architecture. Most comparisons below
prove that in under 25 lines of code you can run offline — no API keys.

## The comparisons

| Framework | What you'll see |
|---|---|
| [LangChain & LangGraph](langchain-langgraph.md) | Graph DSL vs plain async; interrupt (node replay) vs cancellation; middleware vs capabilities |
| [OpenAI Agents SDK](openai-agents-sdk.md) | One capability noun vs guardrail/handoff categories; typed deps; durable wraps |
| [Claude Agent SDK](claude-agent-sdk.md) | Config surface for a subprocess harness vs a typed, in-process loop |
| [smolagents](smolagents.md) | Sync-only loop vs structured async; sandbox limits vs a typed boundary |
| [CrewAI](crewai.md) | Role/process orchestration vs orchestration-as-code |
| [Google ADK](google-adk.md) | anyio-deep internals without a user cancellation API |
| [AG2](ag2.md) | Checkpointed task state machines and envelope cancellation |
| [Vercel AI SDK](vercel-ai-sdk.md) | The TS ecosystem norm (AbortSignal) vs typed Python cancellation |
| [Mastra](mastra.md) | TS/Node processors architecture vs typed capabilities |
| [Agno](agno.md) | A bundled runtime vs a library with runtime choice |
| [Pi](pi-coding-agent.md) | A shipped CLI vs the harness as a library |

## The flagship

[The production agent](production-agents.md) — the checklist production agents actually need, each
row demonstrated by a runnable snippet: a deps boundary the model can't cross, cancellation as a
typed resumable outcome, budgets that halt *before* side effects, self-repairing history, specs that
fail at load, evals in CI.

## Method

- Every claim is either **runnable here** (offline, deterministic, ~5 seconds) or carries a pinned
  version of the framework it describes.
- Competitor pages state their best fairly; our gaps are in the same tone as theirs.
- Page sources and probe records live in the [`pydantic-ai-notes`](https://github.com/pydantic/pydantic-ai-notes)
  repository's framework-comparison series; versions last verified 2026-09-10.