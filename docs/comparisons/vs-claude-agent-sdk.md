# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK is Claude Code as a library: it spawns the `claude` CLI
(`SubprocessCLITransport`) and you configure it in data. You get that program immediately, including
permissions, rewind, and MCP.

Pydantic AI plus [pydantic-ai-harness](https://pydantic.dev/docs/ai/harness/)
[`Coder`](https://pydantic.dev/docs/ai/harness/coder/) is a coding agent in *your* process, on any
model, as a capability you can take apart.

## Side by side

| | Claude Agent SDK 0.2.152 | Pydantic AI 2.42 + harness |
|---|---|---|
| Where the loop runs | `claude` subprocess | Your process |
| Models | Anthropic | Any provider |
| Coding agent | The CLI, immediately | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/), or the blocks it bundles |
| Tools | Strings the CLI owns | Your functions, plus harness tools |
| Stop | `interrupt()`, streaming only | `CancellationToken`, `RunCancelled` |
| Continuity | Sessions, rewind | History you own |
| Crash recovery | Not a session | Six engines wrap the agent |
| Test offline | Launch the CLI | `TestModel` / `FunctionModel` |

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
`CancellationToken` or `ctx.cancel()` from inside a tool.

Sessions resume a conversation. They don't recover a half-finished run. Wrap the same agent in
Temporal, DBOS, or Prefect if you need the work to restart.

## FAQ

**Can I use Claude models?** Yes, directly. This page is about whose process the loop lives in.

**Is `Coder()` Claude Code?** No. It's the parts, on your agent. More assembly, more yours.

---

*claude-agent-sdk 0.2.152, Pydantic AI 2.42. `Coder()` matches the [Harness Coder docs](https://pydantic.dev/docs/ai/harness/coder/);
this repo does not install the harness, so that snippet is not executed in CI.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
