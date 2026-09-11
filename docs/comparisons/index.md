# Comparisons

Pydantic AI is an agent SDK: a typed [`Agent`][pydantic_ai.Agent] you build with, where harnesses like
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) are capabilities on the same object — take ours
apart or build your own. These pages compare it, honestly, with the frameworks you're weighing it
against.

The list mixes a few different kinds of thing, and it's worth knowing which one you're holding:

- **Agent SDKs** — libraries you build an agent with:
  [LangChain & LangGraph](vs-langchain-langgraph.md), [Vercel AI SDK](vs-vercel-ai-sdk.md) and
  [Mastra](vs-mastra.md) (TypeScript), [Agno](vs-agno.md), [CrewAI](vs-crewai.md). Pydantic AI is one
  of these.
- **Vendor SDKs** — first-party libraries that surface one vendor's features first:
  [OpenAI Agents SDK](vs-openai-agents-sdk.md), [Google ADK](vs-google-adk.md). Pydantic AI treats
  every provider the same, in [one interface](../models/overview.md).
- **Standalone harnesses** — a finished agent you drive or extend, not an SDK you build with:
  [Pi](vs-pi.md), and the [Claude Agent SDK](vs-claude-agent-sdk.md), which is Claude Code as a library.
  The Claude Agent SDK is not an agent SDK the way LangChain is: it's one harness or nothing — you can
  drive it, but you can't build harnesses with it. The Pydantic AI equivalent,
  [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/), is a capability you can take apart, because
  the SDK under it is the product.
- **A realtime runtime** — [LiveKit Agents](vs-livekit.md): WebRTC rooms, telephony, STT/LLM/TTS
  pipelines. Pydantic AI [speaks](../realtime/overview.md) on the same call, from the same agent.

These are different categories with overlapping features, so the tables on each page start with what
the framework *is* — language, providers, durability, observability, whether you can build your own
harness — and only then get to the feature checklist.

[Install Pydantic AI](../install.md).
