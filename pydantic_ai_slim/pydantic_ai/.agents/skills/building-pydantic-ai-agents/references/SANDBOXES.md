# Sandboxes

Use a sandbox when tools need an execution environment for commands or files. Pydantic AI attaches a sandbox to a run through the structural [`Sandbox`][pydantic_ai.sandbox.Sandbox] protocol; it does not decide which command or file tools the model may call.

## Attach and expose

The caller owns a sandbox passed directly to a run:

```python
from pydantic_ai import Agent, LocalSandbox, RunContext, UserError

agent = Agent('openai:gpt-5.2')


@agent.tool
async def execute(ctx: RunContext[None], command: str) -> str:
    sandbox = ctx.sandbox
    if sandbox is None:
        raise UserError('No sandbox is attached to this run.')
    result = await sandbox.run(command, shell=True, timeout=30)
    return result.stdout if result.exit_code == 0 else f'[exit {result.exit_code}] {result.stderr}'


async def main() -> None:
    async with LocalSandbox() as sandbox:
        await agent.run('Create and run hello.py.', sandbox=sandbox)
```

`LocalSandbox` runs on the host and isolates nothing. Use it only for trusted development and tests; use a container, VM, or remote implementation for untrusted code. Keep approval, command restrictions, output limits, and path policy in the tool layer.

## Ownership

- The caller of `run(sandbox=...)` owns the sandbox: create it before the run, tear it down after (typically an `async with` around the run).
- The handle is available on `ctx.sandbox` for the whole run; sharing one sandbox across runs is just passing the same handle to each run.

## Durable execution

Live sandbox handles do not cross durable boundaries:

- Temporal workflows reject `sandbox=`. Carry a serializable `{provider, sandbox_id}` reference and re-open it inside an activity.
- DBOS durable `run` and `run_sync` reject `sandbox=`. Re-open by reference inside a tool decorated with `@DBOS.step()`.
- Prefect includes provider-qualified sandbox identity in tool-task cache keys, but the caller still owns lifecycle.

Keep credentials worker-side, make create/open operations idempotent, and use a server-side TTL because terminated workflows do not run cleanup.

See the full [sandbox guide](https://ai.pydantic.dev/sandbox/) for the protocol, lifecycle rules, and implementation contracts.
