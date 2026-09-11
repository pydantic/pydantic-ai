# Pydantic AI vs Agno

Agno is a library plus **AgentOS**, a FastAPI runtime with auth, a UI, and storage. Pydantic AI is
only the library: an agent you put in the application you already run.

## Side by side

| | Agno | Pydantic AI |
|---|---|---|
| What you run | Optional AgentOS (UI, auth, roles) | The agent, in your existing app |
| Stop | `Agent.cancel_run(run_id)` | A stop signal; you get the messages back |
| Shell / code | Host `subprocess` / `exec` by default | Sandboxed (`Monty` / Modal), plus approval on any tool |
| Memory | Built in | You wire it |
| Crash recovery | AgentOS persistence | The same agent, inside Temporal, DBOS, or Prefect |
| Tracing | OpenInference, not `gen_ai.*` | OpenTelemetry GenAI names, when you turn them on |

## Host by default

`ShellTools` runs `subprocess.run` on the machine. `PythonTools` runs `exec`. Their own docstring
calls that an RCE sink if the agent is prompt-injected. You can require a human click with
`ShellTools(requires_confirmation_tools=["run_shell_command"])`.

Ours: `CodeMode` executes model-written Python inside [Monty](https://github.com/pydantic/monty).
Any tool can take `requires_approval=True`. Untrusted work goes in Modal or a container.

AgentOS is not a tax on the library: `Agent(name='x')` constructs, and `agno.agent.agent` never
imports `agno.os`.

## FAQ

**Do I have to take AgentOS?** No. The library runs without it.

**Faster to construct?** Microseconds either way. A model call is the number.
See [under the hood](under-the-hood.md).
