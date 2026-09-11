# Pydantic AI vs Agno

Agno ships a library *and* **AgentOS**, a FastAPI runtime with auth, a UI, and storage. If you want
an agent service on Friday, that's the product. We don't ship one.

The library does not require AgentOS: `Agent(name='x')` constructs, `agno.agent.agent` never imports
`agno.os`. Pydantic AI is only the library half. You put the agent in the app you already run.

Shell and Python tools default to host `subprocess` and in-process `exec`. Gate them with
`requires_confirmation_tools`. Ours: `CodeMode` in Monty, `requires_approval=True` on any tool.

## Side by side

| | Agno 3.0.9 | Pydantic AI 2.42 |
|---|---|---|
| Deploy | Optional AgentOS (UI, auth, roles) | Nothing to deploy |
| Stop | `cancel_run(run_id)` | `CancellationToken` / `RunCancelled` |
| Shell / code | Host `subprocess` / `exec` by default | Monty / Modal, plus approval |
| Memory | Built in | You wire it |
| Crash recovery | AgentOS durable API | Six engines wrap the agent |
| Tracing | OpenInference; zero `gen_ai.*` | OpenTelemetry GenAI conventions, when enabled |

## FAQ

**Faster to construct?** Microseconds either way. A model call is the number.
See [under the hood](under-the-hood.md).

**AgentOS equivalent?** Your FastAPI app, a UI adapter, your collector. More work, your stack.

---

*agno 3.0.9, Pydantic AI 2.42. Tool defaults from the installed package; AgentOS from their docs.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
