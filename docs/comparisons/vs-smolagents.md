# Pydantic AI vs smolagents

smolagents asks the model to write Python and runs it. Pydantic AI does that too:
[`CodeMode`](https://pydantic.dev/docs/ai/harness/code-mode/) in the harness, inside
[Monty](https://github.com/pydantic/monty). The default is async tool calls.

Their loop is synchronous: `CodeAgent.run()` owns the thread. Their default sandbox is a restricted
interpreter (no `os`, no `open`); Docker and friends are the real isolation.

## Side by side

| | smolagents | Pydantic AI |
|---|---|---|
| How the model acts | Writes Python | Tool calls, or `CodeMode` (write Python in Monty) |
| Async | No | Yes |
| Stop | `interrupt()` raises `AgentError` between steps | A stop signal; you get the messages back |
| Sandbox | Restricted interpreter; escalate to Docker/E2B/Modal | Monty / Modal in the harness |
| Crash recovery | None in core | The same agent, inside Temporal, DBOS, or Prefect |
| Test offline | Subclass `Model` | A fake model you script; no API key |

## FAQ

**Write-code instead of tools?** `CodeMode` in the harness, inside Monty.

**Is their sandbox safe?** For accidents, the defaults are honest. For an adversarial prompt, they
tell you to use Docker.
