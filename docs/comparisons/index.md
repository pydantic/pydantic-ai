# Pydantic AI vs other agent frameworks

You're picking an agent framework. These pages are what we'd want to read if we were the ones picking.

Every other library on this list made a bet about what an agent is: a graph, a session, a subprocess,
a crew, a Google runner, a Task envelope, a deployable OS, a TypeScript platform, a React stream, a
finished coding agent. Ours is a typed Python value you put in the application you already have.
Pause, durability, a coding harness, and tests attach to that value instead of changing its shape.

Anything we say about someone else's software, we installed it and read it. The bottom of each page
tells you which version, and how we checked. Anything we say about ours comes with a script you can
copy. CI runs every one of them on every commit.

Where we're the wrong answer, we say so.
[All in one place](production-agents.md#where-were-not-the-answer).

## Start here

**Down to two frameworks?** Go straight to the page.

| | The bet | The fork |
|---|---|---|
| [LangChain and LangGraph](vs-langchain-langgraph.md) | An agent is a graph | Pause without replaying the node |
| [OpenAI Agents SDK](vs-openai-agents-sdk.md) | An agent is a session on their platform | Stop and still own the conversation |
| [Claude Agent SDK](vs-claude-agent-sdk.md) | An agent is the `claude` CLI | The loop in your process, on any model |
| [CrewAI](vs-crewai.md) | An agent is a role on a crew | Branches as ordinary async code |
| [smolagents](vs-smolagents.md) | The model writes Python | Async tools that overlap, and a stop that resumes |
| [Google ADK](vs-google-adk.md) | An agent is a Gemini runner | A stop button that leaves you a value |
| [AG2](vs-ag2.md) | Durability lives in a Task | Durability from an engine you already run |
| [Agno](vs-agno.md) | An agent is a service you deploy | A library you embed |
| [Mastra](vs-mastra.md) | TypeScript all-in-one | Python, next to the rest of your stack |
| [Vercel AI SDK](vs-vercel-ai-sdk.md) | The wire to a React UI | Python behind that wire, or stay in TypeScript |
| [Pi](vs-pi.md) | A finished coding agent | The parts it's made of, on the same agent object |

**Still deciding broadly?** Read [what a production agent needs](production-agents.md). It's the list
we'd hold any framework to, us included, and every item has a runnable proof under it.

**Reading benchmark claims about us?** [Under the hood](under-the-hood.md) takes the speed and
lines-of-code comparisons apart and shows what's actually being measured.

## Where each one is strong

Every "where it stops" below is argued on that framework's page, against a version we installed.

| Framework | Strongest at | Where it stops |
|---|---|---|
| LangChain and LangGraph | An integration catalogue far larger than ours; checkpointed workflows and time travel; Deep Agents as a shipped harness | `interrupt()` inside a node replays that node's work; `abort()` exists only on the experimental v3 stream |
| OpenAI Agents SDK | OpenAI features first; hosted sessions and tracing; `after_turn` stops at a turn boundary | Continuity is a session rather than history you own; no first-party crash recovery |
| Claude Agent SDK | Claude Code's behaviour, immediately, including permissions and rewind | The loop is a `claude` subprocess; Anthropic only; tests are integration tests |
| CrewAI | A role-and-task vocabulary that gets a multi-agent demo running quickly; memory and knowledge included | Orchestration is a DSL; no stop method on a crew; limits are per agent rather than per run |
| smolagents | The smallest way to let a model write and run code, with honest sandboxing | Synchronous, so tool calls don't overlap; stopping leaves an error, not a resumable run |
| Google ADK | Gemini, Vertex, A2A, and Google's tooling with no glue | No cancellation API on `Runner` or `LlmAgent` |
| AG2 | Durable checkpointed tasks with no infrastructure to run; broad first-party protocols | The AutoGen-era API is gone at 1.0; durability is a Task envelope, not your engine |
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
