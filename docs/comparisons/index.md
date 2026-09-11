# Pydantic AI vs other agent frameworks

You're picking an agent framework. These pages are what we'd want to read if we were the ones picking.

Every page gives the other framework its best shot first, in its own terms, before we say a word about
ourselves.

Anything we say about someone else's software, we installed it and read it. The bottom of each page
tells you which version, and how we checked.

Anything we say about ours comes with a script you can copy. CI runs every one of them on every
commit, so what's printed on the page is what the code printed today, not what it printed when
somebody wrote the page.

Where we're the wrong answer, we say so. Want the bad news first? It's
[all in one place](production-agents.md#where-were-not-the-answer).

## Start here

**Down to two frameworks?** Go straight to the page.

| | |
|---|---|
| [LangChain and LangGraph](vs-langchain-langgraph.md) | Why a LangChain agent is always a LangGraph graph, where each framework lets you pause, and Deep Agents against our harness |
| [OpenAI Agents SDK](vs-openai-agents-sdk.md) | Stopping a run and keeping it, and where the agent is allowed to run |
| [Claude Agent SDK](vs-claude-agent-sdk.md) | A subprocess you configure against a loop in your own process |
| [CrewAI](vs-crewai.md) | Roles and tasks against ordinary async code, and what you can see while it runs |
| [smolagents](vs-smolagents.md) | Writing code instead of calling tools, and what being synchronous costs |
| [Google ADK](vs-google-adk.md) | Gemini-native breadth, and what happens when someone hits stop |
| [AG2](vs-ag2.md) | Two typed frameworks that landed on similar ideas, and where durability comes from |
| [Agno](vs-agno.md) | A runtime you deploy against a library you embed |
| [Mastra](vs-mastra.md) | The TypeScript all-in-one, and what crosses the language line |
| [Vercel AI SDK](vs-vercel-ai-sdk.md) | Which language your agent lives in, and how to run both |
| [Pi](vs-pi.md) | A finished coding agent against a set of parts |

**Still deciding broadly?** Read [what a production agent needs](production-agents.md). It's the list
we'd hold any framework to, us included, and every item has a runnable proof under it.

**Reading benchmark claims about us?** [Under the hood](under-the-hood.md) takes the speed and
lines-of-code comparisons apart and shows what's actually being measured.

## The short version

People came to us for strong primitives: a typed agent, real validation, and nothing sitting between
you and the model that you didn't ask for. We still really value that. We just give you bigger badder
meaner blocks to work with now.

A capability is one object. It carries tools, instructions, model settings, lifecycle hooks and
event-stream handling, and it can wait to show up until the model actually asks for it. Sixty-three
hooks, with a `before_`, `after_`, `wrap_` and error handler at every stage of a run.

What we don't want is for you to build an entire world around Pydantic AI. We want to help you get
your work done and then get out of your way, with as little friction as we can manage, while still
giving you eyes on what the agent is doing. Your agent goes inside the application you already have.

Six things come out of that, and every one has a proof you can run in a few seconds:

- **Traces your existing tools already understand.** When instrumentation is enabled, we emit the
  OpenTelemetry [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/),
  36 distinct `gen_ai.*` attributes, so the agent can appear in the dashboards your vendor already
  ships. Google ADK does this too. LangChain, the OpenAI Agents SDK, Agno and smolagents emit zero.
- **A ceiling in money, not tokens.** `cost_limit` is priced by
  [genai-prices](https://github.com/pydantic/genai-prices) across 41 providers and 1,646 models. It is
  checked after each response (the first request can exceed it); pair it with `request_limit`. No
  other framework can stop a run on spend across providers.
- **The whole loop runs offline.** `TestModel` calls your tools with no scripting. Every fake chat
  model in `langchain-core` raises `NotImplementedError` on `bind_tools`, so none of them can test an
  agent.
- **Stopping gives you the conversation back.** Cancellation raises, the exception carries the
  history, and resuming is a normal run.
- **Crash recovery from an engine you already operate:** Temporal, DBOS and Prefect in-tree, and
  Restate, Kitaru, and Apache Airflow through integrations those projects maintain themselves.
- **Trusted state is a typed dependency API.** `deps_type` plus `RunContext` is in the agent's type,
  so tools read credentials and the model never sees them. Other frameworks also keep local context
  off the wire; ours is the one your type checker knows the shape of.

## Where each one is strong

Every "where it stops" below is argued properly on that framework's page, against a version we
installed.

| Framework | Strongest at | Where it stops |
|---|---|---|
| LangChain and LangGraph | An integration catalogue far larger than ours; checkpointed workflows and time travel; Deep Agents as a shipped harness | Pausing anywhere their middleware doesn't already pause means `interrupt()`, which replays the enclosing node's work; stopping a run needs their experimental v3 stream; trusted state isn't separate from the conversation |
| OpenAI Agents SDK | OpenAI features first; hosted sessions and tracing; `after_turn` stops at a turn boundary | Continuity is a session rather than history you own; no first-party crash recovery |
| Claude Agent SDK | Claude Code's behaviour, immediately, including permissions and rewind | The loop is a `claude` subprocess; Anthropic only; tests are integration tests |
| CrewAI | A role-and-task vocabulary that gets a multi-agent demo running quickly; memory and knowledge included; plenty of tutorial material | Orchestration is a DSL; no stop method on a crew; limits are per agent rather than per run |
| smolagents | The smallest way to let a model write and run code, with honest sandboxing | Synchronous, so tool calls don't overlap; stopping leaves an error, not a resumable run |
| Google ADK | Gemini, Vertex, A2A, and Google's tooling with no glue | No cancellation API anywhere in the runner; trusted state isn't separate |
| AG2 | Durable checkpointed tasks with no infrastructure to run; broad first-party protocols | The AutoGen-era API is gone at 1.0; specs aren't validated against your types at load |
| Agno | A deployable agent service with auth, roles, and a UI, out of the box | Shell and Python tools run on the host by default; spans carry no `gen_ai.*` attributes |
| Mastra | TypeScript all-in-one: workflows, memory, evals, a real dev experience | JavaScript only; tracing and deployment lean on their platform |
| Vercel AI SDK | Streaming state shared with a React front end, which is what it was built for | JavaScript only; no first-party crash recovery |
| Pi | A finished coding agent that you can also embed | TypeScript; deliberately no sandbox of its own; changing its behaviour means forking |

## What other people say

There is one write-up people quote at us often: a [five-framework
comparison](https://nextbuild.co/blog/ai-agent-frameworks-benchmarked-pydanticai) from the agency
NextBuild, which scored Pydantic AI 8/10 for developer experience against 5/10 for LangChain. We're
not going to lean on it. It's one team's account of one project rather than a benchmark anyone can
re-run; Mastra scored above us on that same measure at 9/10; and its headline cost figure is mostly a
subscription line item, the $1,088 it attributes to CrewAI is $898 of Pro-tier licensing plus $190 of
infrastructure, which says something about pricing pages and nothing about the frameworks.

If you find that quoted somewhere as proof that Pydantic AI is cheaper, it isn't proof, and it isn't
ours.

The complaint we hear most about us is real and unresolved: heavy use of generics makes for noisy
tracebacks. We're working on it, and we'd rather you heard it here.

---

*All pages checked on 2026-09-10 against Pydantic AI 2.42. Framework versions are named at the bottom of each
page, along with how the claims were checked. We recheck every page against current releases each time
Pydantic AI ships a minor version; if something here has gone stale, please [open an
issue](https://github.com/pydantic/pydantic-ai/issues/new) and we'll fix it.*
