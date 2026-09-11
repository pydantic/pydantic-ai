# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK is Claude Code as a library: it spawns the `claude` CLI
(`SubprocessCLITransport`) and you configure it in data. You get that program immediately, including
permissions, rewind, and MCP.

Pydantic AI plus [pydantic-ai-harness](https://pydantic.dev/docs/ai/harness/)
[`Coder`](https://pydantic.dev/docs/ai/harness/coder/) is a coding agent in *your* process, on any
model, as a capability you can take apart.

## Side by side

| | Claude Agent SDK | Pydantic AI |
|---|---|---|
| Where the loop runs | `claude` subprocess | Your process |
| Models | Claude (Anthropic, Bedrock, Vertex, Foundry) | Any provider |
| Coding agent | The CLI, immediately | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/), or the blocks it bundles |
| Tools | `allowed_tools: list[str]` the CLI owns | Your functions, plus harness tools |
| Stop | `interrupt()`, streaming only | A stop signal; you get the messages back |
| Continuity | Sessions, `rewind_files` | A message list you store |
| Crash recovery | The chat, not the half-finished tool | The same agent, inside Temporal, DBOS, or Prefect |
| Test offline | Launch the CLI | A fake model you script; no API key |

## Your process, not theirs

Claude's tools are strings (`allowed_tools=['Read', 'Glob']`). Hooks are JSON events. That's the
interface you get to a loop running in another process.

`Coder()` is a combined capability on a normal [`Agent`][pydantic_ai.Agent]. Files, shell, planning,
and sub-agents run in this process. You add your own tools, typed deps, and approval gates on the
same object:

```python {title="coder_in_process.py" test="skip" lint="skip"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import Coder


@dataclass
class Support:
    orders: dict[str, str]


agent = Agent(
    'openai:gpt-5.6-luna',  # not locked to Anthropic
    deps_type=Support,
    capabilities=[Coder()],  # files, shell, planning, sub-agents
)


@agent.tool
def lookup_order(ctx: RunContext[Support], order_id: str) -> str:
    """Your function. A debugger stops here. The Claude CLI cannot see this."""
    return ctx.deps.orders[order_id]
```

`ClaudeSDKClient.interrupt()` is a control request over the transport, streaming mode only. Ours is a
stop signal you pass into the run, or `ctx.cancel()` from inside a tool. Either way you get the
messages back as [`RunCancelled`][pydantic_ai.exceptions.RunCancelled], not an empty abort.

A Claude session is the chat. If the process dies while a tool is still running, you can reopen the
conversation; that tool does not run again as a durable step. In Pydantic AI you keep a normal
[`Agent`][pydantic_ai.Agent]. Attach Temporal, DBOS, or Prefect and that same object is what the
worker runs.

## FAQ

**Can I get a coding agent without spawning `claude`?** Yes.
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) in your process, on any model, including
Claude.

**Can I add my own tools next to the coding harness?** Yes. Same [`Agent`][pydantic_ai.Agent], same
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/). A debugger stops in your function.
