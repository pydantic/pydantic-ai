# Pydantic AI vs CrewAI

CrewAI asks you to describe a team. Each agent gets a role, a goal, and a backstory; each unit of work
is a `Task`; a `Crew` runs them in order or puts one agent in charge of the others. It reads well and
it demos well, and if the work you're modelling really is a sequence of handoffs between specialists,
that vocabulary does a lot of the thinking for you. It also has a large body of tutorial material,
which matters more than it sounds: when you get stuck at 2am, someone has often already written up
your exact problem.

Version 1.15.21 is more than the role DSL people remember. Crews, flows, and agents can checkpoint
automatically and restart from a checkpoint, agents carry limits for iterations, wall-clock time,
tokens, and requests per minute, and there are first-party modules for memory, knowledge, MCP, skills,
and agent-to-agent messaging.

Pydantic AI has no crew and no roles. Multi-agent work is just Python: call an agent, branch on the
answer, run two at once with `asyncio.gather`, feed one result into the next. That's a worse fit than
CrewAI when the work genuinely is a list of tasks, and a better one when it isn't.

## When the shape of the work isn't a list of tasks

A crew is a good fit when the work really is a sequence of steps, or a manager delegating. It gets
harder when the work has a branch in it — different follow-up depending on what came back, two things
in parallel joined at the end, a retry on one branch only. In CrewAI those live inside the process
mode you picked and the framework decides how they run.

In Pydantic AI they're just code, so they look like the rest of your application and your reviewer
reads them the same way:

```python {title="chain_as_code.py"}
"""A crew is code: chained and fanned-out agents with types between them.

Two branches each extract an int (the tool's signature and return are
typed); the branches are gathered; the typed sum flows into a downstream
agent's dependencies. No roles, no process mode - ordinary async code with a
type at every step.
"""
import asyncio
import re

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel


async def model_num(messages, info):
    if len(messages) == 1:
        value = int(re.search(r'(\d+)', str(messages[0])).group(1))
        return ModelResponse(parts=[ToolCallPart('num', {'v': value})])
    return ModelResponse(parts=[TextPart('done')])


num_agent = Agent(FunctionModel(model_num))


@num_agent.tool
def num(ctx, v: int) -> int:
    return v


async def model_txt(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('describe', {'prefix': 'total'})])
    return ModelResponse(parts=[TextPart('done')])


txt_agent = Agent(FunctionModel(model_txt), deps_type=int)


@txt_agent.tool
def describe(ctx, prefix: str) -> str:
    return f'{prefix}:{ctx.deps}'


async def branch(value: int) -> int:
    with capture_run_messages() as msgs:
        await num_agent.run(f'get {value}')
    returns = [p.content for m in msgs for p in m.parts if isinstance(p, ToolReturnPart)]
    assert returns, 'no tool return captured'
    return int(returns[0])  # the tool's typed return is what the next step gets


async def main():
    a, b = await asyncio.gather(branch(21), branch(22))
    total = a + b
    with capture_run_messages() as msgs:
        await txt_agent.run('finish', deps=total)
    used = 'total:43' in str(msgs)
    print(f'typed chain: branch(a)={a!r}, branch(b)={b!r}, sum={total!r}')
    #> typed chain: branch(a)=21, branch(b)=22, sum=43
    print(f'downstream agent received the typed sum as deps: {used}')
    #> downstream agent received the typed sum as deps: True
    assert isinstance(a, int) and isinstance(b, int)
    assert used


```


Two branches ran at the same time, each returned an `int`, the sum went into the next agent as
dependencies instead of as text in a prompt. Nothing here is a framework concept — it's `asyncio`,
a couple of functions, and type hints.

## What you can see while it runs

Pydantic AI emits a typed stream of events as the run happens — the model starting to speak, each tool
call and its result, the final answer — and you consume it with a normal `async for`. It also emits
OpenTelemetry, so the run shows up wherever your other traces go. CrewAI's equivalent is its event bus
plus the third-party observability integrations it documents, which is a different shape: you subscribe
to a bus instead of iterating the run.

Budgets are the other half of that. `UsageLimits` caps model requests, tool calls, and tokens, and the
check happens *before* the next request goes out, so a runaway loop stops instead of being noticed on
the bill. CrewAI's limits are per agent — `max_iter`, `max_execution_time`, `max_tokens`, `max_rpm` —
which cover a lot but aren't a ceiling on the whole crew.

## Side by side

| | CrewAI 1.15.21 | Pydantic AI 2.42 |
|---|---|---|
| How you describe work | Roles, goals, backstories, tasks, and a process mode | Ordinary async Python: call, branch, gather |
| Trusted state | Crew inputs and values captured in tools | `deps_type`, a separate argument tools read and the model never sees |
| Budgets | Per agent: iterations, time, tokens, requests per minute — no money limit | Per run: requests, tool calls, tokens, and `cost_limit` in USD, priced across 41 providers by `genai-prices`, checked before the next request |
| Crash recovery | Built-in checkpoints on crews, flows, and agents | Six engines wrap the agent: Temporal, DBOS, Prefect, Restate, Kitaru, Airflow |
| Stopping a run | No stop or cancel method on `Crew` | `CancellationToken` from another thread, or `ctx.cancel()` in a tool; the history survives and resumes |
| Watching it work | Events and their platform | Typed event stream plus OpenTelemetry |
| Memory and knowledge | Built in, including knowledge sources | Bring your own, wired through dependencies and history processors |
| Testing offline | Needs a live model for a real run | `TestModel` and `FunctionModel` drive the whole loop with no network |
| Evals | `crewai.experimental.evaluation`: goal alignment, reasoning efficiency, an experiment runner | `pydantic-evals` in your test suite, using the agent's own types |

## Choose CrewAI when

- The work genuinely is a team of specialists doing steps in order, and the role framing helps you
  think about it.
- You want memory, knowledge sources, and RAG without assembling them.
- You're moving fast and the tutorial library is worth real time to you.
- You want their managed platform to run and watch it.

## Choose Pydantic AI when

- The orchestration has branches, joins, and retries you want to read in a diff.
- Credentials and customer identity must sit where the model can't reach them.
- You need a hard ceiling on spend for the whole run, not per agent.
- You want the agent's tests to run offline in CI alongside everything else.

## FAQ

**Is Pydantic AI a drop-in replacement?**
No, and the port isn't mechanical. Tools carry over; roles and tasks become functions and control
flow. People usually find the crew was three or four ordinary steps.

**Does Pydantic AI have anything like a Crew?**
No, deliberately. Multi-agent patterns — an agent as a tool, a router, a parallel fan-out — are
documented as code you write instead of a class you configure.

**What does CrewAI do better?**
Getting to a working multi-agent demo, by some distance, and its memory and knowledge batteries are
more complete out of the box than ours.

---

*Checked against crewai 1.15.21 and Pydantic AI 2.42 on 2026-09-10. The CrewAI facts come from reading the
installed package — `Crew` and `Agent` fields, `kickoff` parameters, checkpoint configuration, and the absence
of any cancel method. Its runtime behaviour needs a live model and was not run. The Pydantic AI example is
executed by this repository's test suite on every commit. We recheck this page's version pins and behaviour
claims each time Pydantic AI ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
