# Pydantic AI vs other agent frameworks

Every other library on this list made a bet about what an agent is. Ours is a typed Python value you
put in the application you already have. Pause, durability, a coding harness, and tests attach to
that value instead of changing its shape.

Anything we say about someone else's software, we installed it and read it. The bottom of each page
names the version. Anything we say about ours comes with a script CI runs on every commit.

| Framework | Their bet | The fork |
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

Where we're the wrong answer, once:
[all in one place](production-agents.md#where-were-not-the-answer).

The production checklist, with a runnable proof under every item:
[what a production agent needs](production-agents.md).

Speed and lines-of-code claims:
[under the hood](under-the-hood.md).

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
