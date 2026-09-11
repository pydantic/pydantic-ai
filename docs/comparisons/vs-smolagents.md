# Pydantic AI vs smolagents

smolagents asks the model to write Python and runs it. The default sandbox is a restricted
interpreter (no `os`, no `open`); Docker and friends are the real isolation. The loop is
synchronous: `CodeAgent.run()` owns the thread.

Pydantic AI is async tool calls. `CodeMode` in the harness is the write-Python path, inside
[Monty](https://github.com/pydantic/monty).

## Side by side

| | smolagents 1.26.0 | Pydantic AI 2.42 |
|---|---|---|
| How the model acts | Writes Python | Tool calls; `CodeMode` if you want code |
| Async | No | Yes |
| Stop | Flag between steps | `RunCancelled` with history |
| Sandbox | Restricted interpreter; escalate to Docker/E2B/Modal | Monty / Modal in the harness |
| Crash recovery | None in core | Six engines wrap the agent |
| Test offline | Subclass `Model` | `TestModel` / `FunctionModel` |

## Tool calls that overlap

Three independent lookups start together:

```python {title="parallel_tool_calls.py"}
import asyncio

from pydantic_ai import Agent, RunContext

events: list[str] = []
agent = Agent('openai:gpt-5.6-luna')


@agent.tool
async def slow_lookup(ctx: RunContext, name: str) -> str:
    events.append(f'start:{name}')
    await asyncio.sleep(0.01)
    events.append(f'end:{name}')
    return f'{name}:done'


async def main():
    await agent.run('Run the warehouse lookups for A, B, and C.')
    print('first three events:', events[:3])
    #> first three events: ['start:a', 'start:b', 'start:c']
```

smolagents' `interrupt()` is a flag between steps, from another thread, and you get an error, not a
resumable history.

## FAQ

**Write-code instead of tools?** `CodeMode` in the harness, inside Monty.

**Is their sandbox safe?** For accidents, the defaults are honest. For an adversarial prompt, they
tell you to use Docker.

---

*smolagents 1.26.0, Pydantic AI 2.42. `import os` / `open(...)` errors and `CodeAgent.run` being sync
come from the installed package.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
