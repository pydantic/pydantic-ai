# Pydantic AI vs other agent frameworks

Pydantic AI is the de facto, Pythonic way to write agents in Python. These pages compare it against
the agent frameworks people actually reach for. Each page compares what the frameworks
actually ship, with every claim runnable on your laptop and framework versions documented on the page.

We ship **primitives, not decisions**: typed, composable seams (capabilities, events, dependencies,
durability engines) that you wire yourself. We don't pick your architecture. Most comparisons below
prove that in under 25 lines of code you can run offline — no API keys.

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