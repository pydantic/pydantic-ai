# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK gives you Claude Code from Python. That's not a figure of speech: the library
finds the `claude` binary on your machine and runs it as a child process, then talks to it over its
own protocol. Everything the CLI can do — editing files, running commands, skills, subagents, hooks,
checkpoints, session forking, MCP servers, the permission prompts — you get, because it is the same
program.

For building something Claude-Code-shaped, that is the shortest path there is, and version 0.2.152 has
grown useful things since: `max_budget_usd` caps what a run may spend, and permission handling,
session resumption, and hook events are all configurable.

Pydantic AI runs the loop in your process, on any model, with your own Python functions as tools. The
two are genuinely different tools, and the honest way to choose is to notice which side of a line your
project is on.

## The line: whose process is it

Configuring the Claude SDK means describing an agent in data. Tools are strings —
`allowed_tools=['Read', 'Glob']`. Subagents are dictionaries. Hooks are JSON events. That's a
reasonable interface to a program running elsewhere, and it's the only interface available, because
your Python isn't where the loop lives.

In Pydantic AI the loop is in your process, so a tool is a function you wrote, with your types, your
imports, and a debugger that stops inside it:

```python {title="in_process_loop.py"}
"""The loop runs in your process, so a tool is just your function."""

import os

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

ran_in: list[int] = []


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('where_am_i', {})])
    return ModelResponse(parts=[TextPart('In the same process that called me.')])


agent = Agent(FunctionModel(model))


@agent.tool_plain
def where_am_i() -> str:
    """Report which process this tool is executing in."""
    ran_in.append(os.getpid())
    return 'checked'


result = agent.run_sync('which process runs the tools?')
print('the tool ran in this process:', ran_in == [os.getpid()])
#> the tool ran in this process: True
print(result.output)
#> In the same process that called me.
assert ran_in == [os.getpid()]
```



Same process id — no subprocess, no protocol between you and your own tools. In practice that's what
decides several things at once: your tools can hold a database connection, your tests can run without
launching anything, and you can step through a tool call in a debugger.

It also means you can test the whole loop offline. `TestModel` and `FunctionModel` script the model's
behaviour deterministically, and `ALLOW_MODEL_REQUESTS = False` turns any accidental real API call into
an error. Testing a Claude SDK agent means launching the CLI and letting it talk to Anthropic, which is
an integration test whether you wanted one or not.

## Sessions remember conversations; engines remember work

The Claude SDK's continuity story is sessions: resume by id, fork one, rewind to a checkpoint. That's
good for a conversation you want to pick back up.

It's not crash recovery. If the process dies halfway through a long run, the session tells you what
was said, not which of the six things the agent was doing had finished. Pydantic AI runs can be
wrapped by a durable engine — Temporal, DBOS, Prefect, Restate, Kitaru, or Airflow — which restarts the
work rather than the transcript, and the wrapping doesn't change the agent.

## Side by side

| | Claude Agent SDK 0.2.152 | Pydantic AI 2.42 |
|---|---|---|
| Where the loop runs | The `claude` CLI, as a child process | Your process |
| Models | Anthropic | Any provider, with `FallbackModel` for failover |
| Tools | Named in strings; the CLI owns them | Your Python functions, with types and validation |
| Subagents and hooks | Configuration dictionaries and JSON events | Capabilities and typed hooks in your code |
| Trusted state | Nothing typed; configuration and environment | `deps_type`, read by tools, invisible to the model |
| Coding-agent features | Everything Claude Code has, immediately | Composable pieces in the harness: filesystem, shell, `CodeMode`, subagents, skills, memory |
| Spend limits | `max_budget_usd` for the run | `UsageLimits` on requests, tool calls and tokens, checked before the next call |
| Stopping a run | Kill the subprocess | `CancellationToken`, `ctx.cancel()`, `RunCancelled` with resumable history |
| Continuity | Sessions: resume, fork, rewind | Message history you own and store |
| Crash recovery | Not the same thing as a session | Six engines wrap the agent object |
| Testing offline | Launch the CLI; it's an integration test | `TestModel` and `FunctionModel`, no network |

## Choose the Claude Agent SDK when

- You want Claude Code's behaviour and you want it today.
- Anthropic is your model and that isn't going to change.
- Its permission prompts, checkpoints, and rewind are features you'd otherwise have to build.
- A subprocess is fine, and configuring in data suits you better than writing the loop.

## Choose Pydantic AI when

- The agent has to run inside your service, holding your connections.
- You need the same agent on more than one provider.
- Your tools deserve real types, real validation, and a debugger.
- You want offline deterministic tests, spend ceilings, crash recovery, and evals in CI.

## FAQ

**Can I use Claude models with Pydantic AI?**
Yes, directly, including the Anthropic-hosted tools. This isn't about which model you use.

**Can Pydantic AI build a coding agent?**
Yes, through [pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness), which supplies a
filesystem, a shell, subagents, skills, memory, compaction, and `CodeMode` as pieces you assemble.
It's a library of parts rather than a finished product, so it's more work than pointing the Claude SDK
at a directory — and more yours afterwards.

**Is one more secure?**
They protect different things. The Claude SDK isolates by process and has a mature permission
interface. Pydantic AI keeps trusted state out of the model's reach entirely, pauses on any tool marked
`requires_approval=True`, and runs model-written code in a sandbox through the harness.

---

*Checked against claude-agent-sdk 0.2.152 and Pydantic AI 2.42 on 2026-09-10. The subprocess behaviour
and the `ClaudeAgentOptions` fields, including `max_budget_usd`, come from reading the installed
package. The Pydantic AI example is executed by this repository's test suite.*
