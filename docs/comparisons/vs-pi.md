# Pydantic AI vs Pi

Pi is a coding agent you run in a terminal, and it's also a library: the same package that ships the
CLI exports an embeddable core, so you can drive the agent from your own TypeScript. It has skills
discovered from `SKILL.md` files, automatic conversation compaction, a strict line-delimited JSON
protocol for driving it programmatically, and a clear-eyed security position — its own documentation
says Pi ships no sandbox and that real isolation has to come from a container or a VM.

The closest thing on our side isn't Pydantic AI by itself. It's Pydantic AI plus
[pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness), which is where the
coding-agent pieces live: a filesystem, a shell, subagents, skills, memory, compaction, planning, and
`CodeMode` for running model-written Python inside the [Monty](https://github.com/pydantic/monty)
sandbox.

So this is a comparison between a finished product you can also embed, and a set of parts you
assemble. Both are legitimate. Which one you want depends on how much of the agent you need to change.

## A product you configure, or parts you assemble

Pi has made the decisions. What the loop does, how compaction works, when skills load, how the
conversation is stored — all settled, all good defaults, and you get a working coding agent
immediately. Configuration is how you influence it.

The harness has made almost none. Every piece is a capability you add, replace, or leave out, and they
sit on the same agent object as everything else, which means the coding pieces compose with ordinary
agent features instead of living in a separate world:

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



The loop runs in your process, so a tool is a Python function with your types and your imports, and you
can put a breakpoint in it.

The trade is honest in both directions. If you want a coding agent, Pi is running today and the harness
is an afternoon of assembly. If you want a coding agent that does something Pi didn't anticipate —
different compaction, a different filesystem, an approval gate on one specific action, a spend ceiling
per customer — that's a capability on our side and a fork on theirs.

## Language, and where isolation lives

Pi's core is TypeScript and Node. Ours is Python. For most teams that settles it before any feature
comparison starts.

On isolation, Pi's position is that it isn't the sandbox — you run it in a container, mount things
read-only, and give it minimal credentials. That's a defensible design and they say it plainly.

Ours puts more of the boundary in the library: `CodeMode` runs model-written code in Monty instead of
your interpreter, `ModalSandbox` gives the agent an isolated cloud container, any tool can be marked
`requires_approval=True` so the run pauses and hands you the pending call, and `deps_type` keeps
credentials somewhere the model can't see them at all. You should still run the thing in a container.
The difference is how much survives when you don't.

## Side by side

| | Pi | Pydantic AI + harness 2.42 |
|---|---|---|
| Language | TypeScript and Node | Python |
| What you get | A working coding agent, plus an embeddable core | Parts you assemble onto any agent |
| Changing behaviour | Configuration, extensions, skills | Any piece is a capability you swap |
| Isolation | Deliberately none; run it in a container | `CodeMode` in Monty, `ModalSandbox`, per-tool approval, plus your container |
| Trusted state | Environment and configuration | `deps_type`, read by tools, invisible to the model |
| Skills | `SKILL.md` files with progressive disclosure | Skills as a capability, alongside the rest |
| Driving it programmatically | Line-delimited JSON over stdin and stdout | Ordinary Python function calls |
| Models | Configurable provider and model | Any provider, with `FallbackModel` for failover |
| Crash recovery | Session resume | Six engines wrap the agent object |
| Testing offline | Run the agent | `TestModel` and `FunctionModel`, no network |

## Choose Pi when

- You want a good coding agent now and don't intend to change how it works.
- TypeScript is your language.
- Its skills and compaction behaviour already match how you want to work.
- You're happy to provide isolation at the container level, which you should be doing anyway.

## Choose Pydantic AI and the harness when

- The coding agent is part of a larger Python application, not a tool you run.
- You need to replace or add behaviour that a configuration flag doesn't cover.
- You want spend ceilings, approval gates, resumable cancellation, and evals from the same framework.
- Credentials must sit where the model can't reach them.

## FAQ

**Is the harness a Claude Code or Pi competitor?**
Not as a product. It's a library of the pieces those products are made of. If you want a finished
coding agent, use a finished coding agent.

**How much assembly is it really?**
A useful agent with a filesystem, a shell, and subagents is a short file. Matching a mature CLI's
behaviour — its compaction, its permission prompts, its polish — is considerably more.

**What does Pi do better?**
Being finished. And its security documentation is more direct about its own limits than most projects
manage.

---

*Pi behaviour described here comes from its CLI and its own documentation, checked at version 0.85.1; we did
not run its embeddable core. The Pydantic AI example is executed by this repository's test suite on every
commit. Pydantic AI 2.42, checked 2026-09-10. We recheck this page's version pins and behaviour claims each
time Pydantic AI ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
