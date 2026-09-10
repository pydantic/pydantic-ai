# Pydantic AI vs Agno

Agno is two products that ship together. There's the library — `Agent`, `Team`, toolkits, memory,
knowledge, guardrails — and there's **AgentOS**, a runtime you deploy: a FastAPI application with
prebuilt endpoints for sessions, memory, knowledge and evals, a control-plane UI, JWT auth with
role-based access, storage, and background runs. If what you want is an agent service running by
Friday, that combination is hard to beat, and nothing in Pydantic AI competes with it directly.

Pydantic AI is only the library half. There's no runtime to deploy, no control plane, and no UI. You
put the agent inside whatever you already run.

That's the whole comparison, really — but it has two consequences worth spelling out.

## Nothing to adopt

An Agno agent is built for AgentOS, and its shape follows from that: a large keyword constructor, an
implied session and storage story, and a deployment target. It's coherent, and if you're deploying
AgentOS it's exactly right.

A Pydantic AI agent has no assumed home. The same object runs blocking, runs async, or gets driven a
step at a time inside a loop you control:

```python {title="runtime_agnostic.py"}
"""No bundled runtime to adopt: the same agent runs sync, async, and driven
node-by-node with iter() - whichever shape your application already uses."""
import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel


async def model(messages, info):
    return ModelResponse(parts=[TextPart('same result')])


agent = Agent(FunctionModel(model))


async def via_iter():
    async with agent.iter('q') as run:
        async for _ in run:
            pass
    return run.result.output


sync_result = agent.run_sync('q').output
async_result = asyncio.run(agent.run('q')).output
iter_result = asyncio.run(via_iter())

print(f'sync={sync_result!r} async={async_result!r} iter={iter_result!r}')
#> sync='same result' async='same result' iter='same result'
assert sync_result == async_result == iter_result


```


Same agent, same answer, three shapes — which matters when the agent has to live inside a Django view,
a Celery task, a Lambda handler, or a websocket server you already have.

For crash recovery, the same idea applies: rather than a durable API belonging to the runtime, a
durable engine is a capability you add. `capabilities=[TemporalDurability()]` is the whole change, and DBOS, Prefect,
Restate, Kitaru, and Airflow have equivalents. You use whichever your company already runs.

## What the tools are allowed to do

Agno positions itself for coding agents and ships shell, file, and Python tools to match. Their
defaults are worth understanding before you turn them on, and Agno documents them honestly — the shell
tool's own docstring says the command "is executed directly on the host OS" and tells you to gate it
with `requires_confirmation_tools=["run_shell_command"]`.

At 3.0.9: `run_shell_command` runs `subprocess` on the host by default. `PythonTools` runs
model-written code with `exec` in your process and includes tools that install packages with pip. File
and Python tools do contain paths by default — `restrict_to_base_dir=True`, with `..`, absolute paths,
and symlink escapes rejected — which is a real protection and recently added.

Pydantic AI's plain tools are just your functions, so there's nothing to sandbox. When you do want the
model executing code, the harness gives you `CodeMode`, which runs it inside the
[Monty](https://github.com/pydantic/monty) sandbox, and `ModalSandbox`, which gives the agent an
isolated cloud container instead of your host. Approval before a risky tool runs is built into the
framework: mark it `requires_approval=True` and the run pauses and hands you the pending call.

## Side by side

| | Agno 3.0.9 | Pydantic AI 2.42 |
|---|---|---|
| What you deploy | AgentOS: a runtime with endpoints, UI, auth, roles, storage | Nothing; the agent goes inside your app |
| Agent shape | One large constructor built around the runtime | A typed value that runs sync, async, or step by step |
| Trusted state | Session state and values captured in tools | `deps_type`, read by tools, invisible to the model |
| Stopping a run | `cancel_run(run_id)` | `CancellationToken` across runs, `ctx.cancel()` in a tool, `RunCancelled` with resumable history |
| Shell and code tools | Host `subprocess` and in-process `exec` by default, with warnings and opt-in confirmation | `CodeMode` in Monty, `ModalSandbox` for containers, `requires_approval=True` on any tool |
| Crash recovery | AgentOS durable API | Six engines wrap the agent object; you pick |
| Structured output | `output_schema` — note that `output_model` means the parser model | `output_type`, with explicit control over how it goes over the wire |
| Memory and knowledge | Built in and well developed | Bring your own; ours is thinner |
| Evals | Eval classes stored in AgentOS | `pydantic-evals` in your test suite, using the agent's own types |
| Tracing | Their control plane | OpenTelemetry to wherever you send everything else |

## Choose Agno when

- You want a deployable agent service with auth, roles, and a UI, and you don't want to build it.
- Their memory and knowledge features match what you need — they're more complete than ours.
- A team-of-agents abstraction fits your problem and you'd rather configure than code it.
- Running one more service is fine, and having it be theirs is a plus.

## Choose Pydantic AI when

- The agent has to live inside an application you already have.
- Credentials and identity must sit where the model can't reach them.
- You want model-written code in a sandbox rather than on the host, and approval gates in the
  framework rather than in a tool's configuration.
- You want crash recovery from an engine you already operate.

## FAQ

**Is Agno faster?**
It constructs agents faster, and you'll see benchmarks about that. Construction happens once and takes
microseconds either way; a single model call takes hundreds of milliseconds. It's not the number to
choose on.

**Can I get something like AgentOS with Pydantic AI?**
Not out of the box. You'd put the agent behind your own FastAPI app, use one of the UI adapters, and
send telemetry to Logfire or your own collector. That's more work, and it's your stack afterwards.

**What does Agno do better?**
Time to a running, authenticated, observable agent service. If that's the job, they've built the thing
and we haven't.

---

*Checked against agno 3.0.9 and Pydantic AI 2.42 on 2026-09-10. The tool behaviour comes from reading
the installed package: `ShellTools.run_shell_command` and its docstring, `PythonTools`, and the
`restrict_to_base_dir` default. AgentOS claims are from Agno's documentation, not run. The Pydantic AI
example is executed by this repository's test suite.*
