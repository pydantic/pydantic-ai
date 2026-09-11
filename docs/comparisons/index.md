# Comparisons

Picking an agent framework is mostly a question of what you want to own and what you want handed to you. These pages put Pydantic AI side by side with the frameworks people ask us about most, one page each, with every cell checked against the other framework's released package rather than its docs.

- [vs LangChain & LangGraph](vs-langchain-langgraph.md)
- [vs Claude Agent SDK](vs-claude-agent-sdk.md)
- [vs Vercel AI SDK](vs-vercel-ai-sdk.md)
- [vs OpenAI Agents SDK](vs-openai-agents-sdk.md)
- [vs Google ADK](vs-google-adk.md)
- [vs Mastra](vs-mastra.md)
- [vs LiveKit Agents](vs-livekit.md)
- [vs Pi](vs-pi.md)
- [vs Agno](vs-agno.md)
- [vs CrewAI](vs-crewai.md)

## How to read the tables

**"Yes" means the framework ships the capability itself.** Where a framework only forwards a tool that the model provider hosts and executes, the cell reads "Via provider tools": you get that capability from the provider, on the providers that offer it, not from the framework.

**"Build your own harness"** means you own the agent loop in your own process, and tools, system prompt and session are data you pass into it. A *harness* is the system around the model that orchestrates execution, tools and context; a framework whose loop and built-in tools live in a separate binary it drives for you is a "No" on this row, however good that loop is.

**Our column covers Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) together.** The Harness is a separate package of ready-made capabilities (memory, guardrails, sandboxes, browsers, sub-agents) for the same [`Agent`][pydantic_ai.Agent], the way LangChain's coding harness ships in `deepagents` rather than in `langchain`. One difference worth knowing before you depend on it: the Harness uses 0.x versioning and its own docs say APIs may move between minor releases, while the library itself follows a [stricter version policy](../version-policy.md).
