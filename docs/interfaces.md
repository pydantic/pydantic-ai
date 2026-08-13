# Interfaces

One agent definition, every surface. A Pydantic AI [agent](agent.md) is plain Python with no interface baked in — so the same agent can run headless inside your backend, chat in a terminal, serve a web UI, power your own frontend, live inside an editor, or hold a spoken conversation. This page is the map; each surface has its own page with the details.

| Surface | What it looks like | Where |
|---|---|---|
| **Your code** | `result = await agent.run('...')` — an ordinary awaitable in any Python function, with [typed output](output.md) | [Running agents](agent.md#running-agents) |
| **Terminal** | `agent.to_cli_sync()`, or `clai -a mymodule:agent` against any agent you can import | [CLI](cli.md) |
| **Web chat** | A built-in browser chat for any agent — `clai web` or `agent.to_web()` | [Web Chat UI](web.md) |
| **Your frontend** | Stream agent runs to your own UI over the [AG-UI](ui/ag-ui.md) or [Vercel AI](ui/vercel-ai.md) protocols — including Vercel's `useChat` React hooks | [UI Event Streams](ui/overview.md) |
| **Editors** | Serve an agent to Zed and other editors over the Agent Client Protocol | [ACP](https://pydantic.dev/docs/ai/harness/acp/) (Harness, experimental) |
| **Voice** | The same agent, tools, and observability over a live audio session — voice is just another frontend | [Realtime](realtime/overview.md) |

A few things fall out of interfaces being decoupled from agents:

- **You don't choose up front.** The agent you prototype in a script is the agent you ship behind an API, and the one you hand to teammates as a CLI — including complete agents from the [Harness](https://pydantic.dev/docs/ai/harness/), like `clai -a pydantic_ai_harness.coder:coder_agent`.
- **Interfaces stack.** The same deployed agent can serve the web UI for your team and the AG-UI stream for your product at once; a [realtime session](realtime/overview.md) can hand its history to a text run and back.
- **Human-in-the-loop is interface-agnostic.** [Deferred tools and approval](deferred-tools.md#human-in-the-loop-tool-approval) surface wherever the agent runs — approval prompts in the CLI, approval UI events in your frontend.

More surfaces are on the way — follow the [roadmap discussions](https://github.com/pydantic/pydantic-ai/issues) for messaging channels and API endpoints.
