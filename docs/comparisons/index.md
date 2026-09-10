# Pydantic AI vs other agent frameworks

If you're here, you likely want to know how we stack up against the other agent frameworks
demanding your attention. Lucky for you, we've done that for you :)

We have comparisons with all of the agent frameworks our users compare us with. Each one starts
with the answer, then proves it with code you can run yourself in seconds — no API keys, no "trust
us, it's fine." Framework versions are documented on each page.

**How to use this:**

- **Narrowed it down to two?** Jump to that page. The answer comes first.
- **Still deciding broadly?** Start with [the production checklist](production-agents.md) — what a
  shipped agent actually needs, row by row — and hold every framework you're considering to it.
  Including us.

One thing we do believe: we ship **primitives, not decisions** — typed, composable seams
(dependencies, capabilities, events, durable engines) that you wire yourself. We don't pick your
architecture; that's your job.

And we trust agents a little more than the old playbook did: for example, a run is allowed to end
itself when that's right — see the [cancellation row](production-agents.md#3-cancellation-is-a-typed-resumable-outcome)
on the checklist. Every knob is still yours; we just ship the one that lets the loop say it's done.

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

## The fine print

- Every snippet on these pages is a self-contained block that runs offline and deterministically;
the repository's test suite executes each one, so what you see printed is what the code prints today.
- Where we state something another framework does or doesn't do, there's a pinned version and a
verification record behind it (in [`pydantic-ai-notes`](https://github.com/pydantic/pydantic-ai-notes)).
- We're not perfect: pages say so where we're not the answer.