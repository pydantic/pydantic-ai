# Pydantic AI vs Agno

Agno is a library plus **AgentOS**, a FastAPI runtime with auth, a UI, and storage. Pydantic AI is
only the library: an agent you put in the application you already run.

## Side by side

| | Agno | Pydantic AI |
|---|---|---|
| What you run | Optional AgentOS (UI, auth, roles) | The agent, in your existing app |
| Stop | `Agent.cancel_run(run_id)` | A stop signal; you get the messages back |
| Shell / code | Host `subprocess` / `exec` by default; CodeMode is host too; E2B, Daytona, Superserve as tools | `Coder()` (host, allowlist) or `CodeMode` (Monty); Modal / a container for untrusted work |
| Memory | Built in (`MemoryManager`) | You wire it |
| Crash recovery | Agent `db` / `checkpoint` | The same agent, inside Temporal, DBOS, or Prefect |
| Tracing | OpenInference, not `gen_ai.*` | OpenTelemetry GenAI names, when you turn them on |

## Host by default

`ShellTools` runs `subprocess.run` on the machine. `PythonTools` runs `exec`. Their `CodeMode` is an
IPython kernel in the same process; their own module docstring says it is not a sandbox. You can
require a human click with `ShellTools(requires_confirmation_tools=["run_shell_command"])`.

They also ship sandbox toolkits: `E2BTools`, `DaytonaTools`, and `SuperserveTools` (Firecracker).
Those are extra tools you attach, not the default for `ShellTools` / `PythonTools` / `CodeMode`.

Ours: [`CodeMode`](https://pydantic.dev/docs/ai/harness/code-mode/) executes model-written Python
inside [Monty](https://github.com/pydantic/monty). [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/)
is host by default; the allowlist is a guardrail, not a VM. Untrusted work goes in Modal or a
container. Any tool can take `requires_approval=True`.

AgentOS is not a tax on the library: `Agent(name='x')` constructs, and `agno.agent.agent` never
imports `agno.os`.

## FAQ

**Does the agent need its own service?** No. It goes in the app you already run.

**Can I still get a UI and a coding harness?** Yes. [`to_web()`][pydantic_ai.agent.Agent.to_web],
[`to_cli_sync()`](../cli.md), and
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/).
