# Pydantic AI vs other agent frameworks

You're choosing an agent framework, and you've likely got a shortlist. This series is how you decide.

- **Narrowed it to two?** Pick that page below — each one starts with the answer, then proves it
  with code you can run offline in seconds, no API keys.
- **Deciding broadly?** Start with [the production checklist](production-agents.md) — what a shipped
  agent actually requires, demonstrated row by row. Hold every framework you're considering to that
  list, including us.

The thread through all of it: we ship **primitives, not decisions** — typed, composable seams
(dependencies, capabilities, events, durable engines) that you wire yourself. We don't pick your
architecture, and we don't expect you to take our word; every claim on these pages runs on your
laptop. Framework versions are documented on each page.

## The comparisons

| Framework | What you'll see |
|---|---|
| [vs LangChain & LangGraph](vs-langchain-langgraph.md) | Graph DSL vs plain async; interrupt (node replay) vs cancellation; middleware vs capabilities |
| [vs OpenAI Agents SDK](vs-openai-agents-sdk.md) | One capability noun vs guardrail/handoff categories; typed deps; durable wraps |
| [vs Claude Agent SDK](vs-claude-agent-sdk.md) | Config surface for a subprocess harness vs a typed, in-process loop |
| [vs smolagents](vs-smolagents.md) | Sync-only loop vs structured async; sandbox limits vs a typed boundary |
| [vs CrewAI](vs-crewai.md) | Role/process orchestration vs orchestration-as-code |
| [vs Google ADK](vs-google-adk.md) | anyio-deep internals without a user cancellation API |
| [vs AG2](vs-ag2.md) | Checkpointed task state machines and envelope cancellation |
| [vs Vercel AI SDK](vs-vercel-ai-sdk.md) | The TS ecosystem norm (AbortSignal) vs typed Python cancellation |
| [vs Mastra](vs-mastra.md) | TS/Node processors architecture vs typed capabilities |
| [vs Agno](vs-agno.md) | A bundled runtime vs a library with runtime choice |
| [vs Pi](vs-pi.md) | A shipped CLI vs the harness as a library |

## The flagship

[The production agent](production-agents.md) — the checklist production agents actually need, each
row demonstrated by a runnable snippet: a deps boundary the model can't cross, cancellation as a
typed resumable outcome, budgets that halt *before* side effects, self-repairing history, specs that
fail at load, evals in CI.

## Method

- Every claim on a page is either a **self-contained code block on that page** (offline, no API
  keys, deterministic — the repository's test suite executes each one) or carries a pinned version
  of the framework it describes.
- Competitor pages state each framework's capabilities factually; our gaps are stated in the same
tone as theirs.
- Page sources and probe records live in the [`pydantic-ai-notes`](https://github.com/pydantic/pydantic-ai-notes)
  repository's framework-comparison series; versions last verified 2026-09-10.