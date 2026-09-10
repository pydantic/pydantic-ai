# Pydantic AI vs other agent frameworks

If you're here, you likely want to know how we stack up against the other agent frameworks
demanding your attention. Lucky for you, we've done that for you :)

We have comparisons with all of the agent frameworks our users compare us with. Each one starts
with the answer, then proves it with code you can run yourself in seconds — no API keys, no "trust
us, it's fine." Framework versions are documented on each page.

**How to use this:**

- **Narrowed it down to two?** Jump to that page. The answer comes first.
- **Still deciding broadly?** Scan the landscape below, then start with [the production
  checklist](production-agents.md) — what a shipped agent actually needs, row by row — and hold
  every framework you're considering to it. Including us.
- **Reading benchmark claims about us elsewhere?** [Tradeoffs, translated](under-the-hood.md) —
  we take the speed/memory/lines-of-code comparisons head-on and show you the switch behind them.

One thing we do believe: we ship **primitives, not decisions** — typed, composable seams
(dependencies, capabilities, events, durable engines) that you wire yourself. We don't pick your
architecture; that's your job.

And we trust agents a little more than the old playbook did: for example, a run is allowed to end
itself when that's right — see the [cancellation
row](production-agents.md#3-cancellation-is-a-typed-resumable-outcome) on the checklist. Every knob
is still yours; we just ship the one that lets the loop say it's done.

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
| [vs Pi](vs-pi.md) | A TS coding agent with an embeddable core vs the harness as composeable, replaceable capabilities (Python, typed) |

## The landscape at a glance

What each framework genuinely does, and where it stops. Every "where" is argued with a pinned
version and runnable proof on that framework's page — we don't expect you to take the table's word.

| Framework | What it does well | Where it stops |
|---|---|---|
| LangChain & LangGraph | The largest ecosystem in agent tooling; checkpointed workflows | `interrupt()` resumes by re-running the node; no typed deps boundary; nothing you'd call cancellation |
| OpenAI Agents SDK | Platform continuity — sessions, Responses, tracing; `after_turn` stop | Extension split across guardrails/handoffs/hooks; resume is a platform session, not history you own |
| Claude Agent SDK | The actual Claude Code harness — skills, hooks, subprocess isolation | The loop is a `claude` subprocess you configure; stopping means killing it |
| smolagents | The minimal code-exec agent, with real sandboxing options | Sync-only; bounds the code the model writes, not what it can know |
| CrewAI | Crew-of-roles pattern with memory and knowledge out of the box | Orchestration is a DSL; stopping a kickoff means killing the thread |
| Google ADK | Vendor full-surface: workflows builder, code executors, A2A | The runner exposes no user cancellation API (we looked); no typed deps boundary |
| AG2 | Checkpointed `Task` state machine plus a spec/protocol surface (ACP, A2A, live) | Specs aren't validated against your types at load; cancel is an envelope with no in-flight abort |
| Vercel AI SDK | The TS ecosystem default: streaming UI, tool loops, provider adapters | JavaScript only; the result's shape is your handler's job |
| Mastra | TS all-in-one: agents, workflows, observability | Extension is spread across surfaces; JavaScript only |
| Agno | Batteries plus a hosted runtime (AgentOS) and team orchestration | A runtime to adopt; the loop isn't yours by default |
| Pi | A coding agent: polished CLI plus an embeddable TS core | The center is the coding session; underneath sits their runtime, not a typed framework with durable wraps and evals |

## When the other framework is the right call

We'll say it so you don't have to wonder — it's the only way the rest of this page is worth
trusting:

- Your app is **all-in on the OpenAI platform** — their SDK is the natural layer.
- You want **Claude Code inside your product** — their harness, full stop.
- You build in **TypeScript** — Vercel or Mastra are the ecosystem defaults.
- Your agent's whole job is **writing and running Python** — smolagents is the minimal fit.
- **Crew-of-roles** with memory out of the box — CrewAI's DSL is built for it.
- A **vendor-maintained full-surface** framework on the Google stack — ADK.
- A **checkpointed, protocol-spanning** system (ACP/A2A/live) — AG2's state machine.
- A **coding agent in TypeScript** — Pi's core was built for exactly that.
- You want a **hosted runtime** and out-of-the-box team orchestration — Agno.
- Your product **lives in the LangChain ecosystem** — that ecosystem is a real thing to build on.

And the one nobody else will print: if your agent is a prompt wrapper — no tools, no state, no
memory — none of us are the right fit. A logging library will serve you better, and that's fine.

## The flagship

[The production agent](production-agents.md) — the checklist production agents actually need, each
row demonstrated by a runnable snippet: a deps boundary the model can't cross, cancellation as a
typed resumable outcome, budgets that halt *before* side effects, self-repairing history, specs
that fail at load, evals in CI.

## The takeaway

Every framework makes a fine demo — that's why you're still deciding. The differences only show up
when the agent ships: state the model can't reach, budgets that stop side effects before they
start, a stop button that leaves your conversation intact, evals in CI, a loop you can drive node
by node. That's the checklist, and it's the bar we hold ourselves to first.

One summary claim we'll stand behind, because each proof is two clicks away: of the frameworks
above, ours is the one where the loop stays yours end to end — and six durable engines wrap the
same agent without changing it.

## Independent takes

We cite people we don't control; that's the point.

- **Developer experience**: an independent 90-day, five-framework benchmark, reported in
  [Speakeasy's framework comparison](https://www.speakeasy.com) (2026-03), scored Pydantic AI 8/10 —
  the highest of the five — against 5/10 for LangChain, and credited type validation with catching
  23 bugs during development that would have reached production.
- **Cost**: the same report measured a 90-day build at $390 total with Pydantic AI (zero license
  fees) versus $1,088 with CrewAI.
- **Community**: the "which framework" threads on [r/AI_Agents](https://www.reddit.com/r/AI_Agents/)
  name our documentation and our low abstraction as the reasons ("easy to make custom behaviours").
- **And the criticism worth hearing from us first**: heavy generics make for noisy tracebacks — a
  repeated community gripe. We read it, we're working on the developer experience, and we'd rather
  you meet that complaint here than in production.

## The fine print

- Every snippet on these pages is a self-contained block that runs offline and deterministically;
the repository's test suite executes each one, so what you see printed is what the code prints today.
- Where we state something another framework does or doesn't do, there's a pinned version and a
verification record behind it (in [`pydantic-ai-notes`](https://github.com/pydantic/pydantic-ai-notes)).
- We're not perfect: pages say so where we're not the answer.