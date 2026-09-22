# Migrate to Pydantic AI

If you have an application built with another agent framework, your coding agent can use a migration skill bundled with Pydantic AI. Each skill helps the agent understand the source framework, choose the right Pydantic AI components, and port one working application path at a time.

The skills preserve behavior rather than translating API names one for one. They separate responsibilities between:

- [Pydantic AI](agent.md) for agents, models, tools, outputs, messages, and streaming;
- [Pydantic Graph](graph.md) for explicit application control flow;
- [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) for optional capabilities such as skills, memory, subagents, planning, and sandboxed tools;
- [Pydantic Evals](evals.md) for evaluation datasets and checks;
- your application for authentication, storage, APIs, UI, deployment, and other product infrastructure.

## Supported source frameworks

| Source framework | Migration skill |
|---|---|
| [LangChain and LangGraph](comparisons/vs-langchain-langgraph.md) | `migrating-langchain-to-pydantic-ai` |
| [Agno](comparisons/vs-agno.md) | `migrating-agno-to-pydantic-ai` |
| [Claude Agent SDK](comparisons/vs-claude-agent-sdk.md) | `migrating-claude-agent-sdk-to-pydantic-ai` |
| [Google Agent Development Kit](comparisons/vs-google-adk.md) | `migrating-google-adk-to-pydantic-ai` |
| [Mastra](comparisons/vs-mastra.md) | `migrating-mastra-to-pydantic-ai` |
| [OpenAI Agents SDK](comparisons/vs-openai-agents-sdk.md) | `migrating-openai-agents-sdk-to-pydantic-ai` |
| [Pi](comparisons/vs-pi.md) | `migrating-pi-to-pydantic-ai` |
| [Vercel AI SDK and Eve](comparisons/vs-vercel-ai-sdk.md) | `migrating-vercel-ai-sdk-and-eve-to-pydantic-ai` |

## Use a migration skill

Install the skills bundled with your project dependencies:

```bash
uvx library-skills --all
```

See [Coding Agent Skills](coding-agent-skills.md#library-skills) for installation details, including Claude Code support.

Then open your existing application in your coding agent and name the matching skill in your request. For example:

```text
Use the migrating-openai-agents-sdk-to-pydantic-ai skill to migrate this application
to Pydantic AI. Preserve its existing API and behavior, and verify the port with tests.
```

The coding agent should first trace a real request through the source application and record what callers observe. It can then migrate the smallest complete path, test the old and new behavior at the same boundary, and explain any difference that cannot be preserved exactly. Remove the old framework only after every required path has been verified.
