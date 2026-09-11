# Pydantic AI vs Agno

Agno is two products that ship together. There's the library, `Agent`, `Team`, toolkits, memory,
knowledge, guardrails, and there's **AgentOS**, a runtime you deploy: a FastAPI application with
prebuilt endpoints for sessions, memory, knowledge and evals, a control-plane UI, JWT auth with
role-based access, storage, and background runs. If what you want is an agent service running by
Friday, that combination gets you there, and we don't ship anything that competes with it directly.

Pydantic AI is only the library half. There's no runtime to deploy, no control plane, and no UI. You
put the agent inside whatever you already run.

## Do you have to take AgentOS?

No, and we want to be straight about that, because plenty of comparisons get it wrong. `Agent(name='x')`
constructs fine with no database and no AgentOS, `agno/agent/agent.py` never imports `agno.os`, and
`agent.run()` works. AgentOS sits next to the library. It isn't a tax on using it.

What does follow from AgentOS is the shape of the agent: a large keyword constructor, an implied
session and storage story, and a deployment target it was designed against. That's coherent, and if
you are deploying AgentOS it's exactly right.

A Pydantic AI agent has no assumed home. The same object runs blocking, runs async, or gets driven a
step at a time inside a loop you control, which is what you want when the agent has to live inside a
Django view, a Celery task, a Lambda handler, or a websocket server you already have.

For crash recovery, same idea: instead of a durable API that belongs to the runtime, a durable engine
is a capability you add. `capabilities=[TemporalDurability()]` is how you attach it; you still need
that engine's worker and workflow (or the DBOS/Prefect equivalent). `agent.run()` is not durable just
because the capability is present. You use whichever engine your company already runs.

## What the tools are allowed to do

Agno positions itself for coding agents and ships shell, file, and Python tools to match. Their
defaults are worth understanding before you turn them on, and Agno documents them: the shell
tool's own docstring says the command "is executed directly on the host OS" and tells you to gate it
with `requires_confirmation_tools=["run_shell_command"]`.

At 3.0.9: `run_shell_command` runs `subprocess` on the host by default. `PythonTools` runs
model-written code with `exec` in your process and includes tools that install packages with pip. File
and Python tools do contain paths by default, `restrict_to_base_dir=True`, with `..`, absolute paths,
and symlink escapes rejected, which is a real protection and recently added.

Pydantic AI's plain tools are just your functions, so there's nothing to sandbox. When you do want the
model executing code, the harness gives you `CodeMode`, which runs it inside the
[Monty](https://github.com/pydantic/monty) sandbox, and `ModalSandbox`, which gives the agent an
isolated cloud container instead of your host. Approval before a risky tool runs is built into the
framework: mark it `requires_approval=True`, include [`DeferredToolRequests`][pydantic_ai.DeferredToolRequests]
in `output_type`, and the run pauses and hands you the pending call.

## Side by side

| | Agno 3.0.9 | Pydantic AI 2.42 |
|---|---|---|
| What you can deploy | AgentOS if you want it: endpoints, UI, auth, roles, storage. Optional, not required | Nothing to deploy; the agent goes inside the app you already have |
| Agent shape | One large constructor, designed against AgentOS | A typed value that runs sync, async, or step by step |
| Trusted state | Session state and values captured in tools | `deps_type`, read by tools, invisible to the model |
| Stopping a run | `cancel_run(run_id)` | `CancellationToken` across runs, `ctx.cancel()` in a tool, `RunCancelled` with resumable history |
| Shell and code tools | Host `subprocess` and in-process `exec` by default, with warnings and opt-in confirmation | `CodeMode` in Monty, `ModalSandbox` for containers, `requires_approval=True` on any tool |
| Crash recovery | AgentOS durable API | Temporal, DBOS and Prefect in-tree; Restate, Kitaru, and Airflow through integrations those projects maintain |
| Structured output | `output_schema`, note that `output_model` means the parser model | `output_type`, with explicit control over how it goes over the wire |
| Memory and knowledge | Built in and well developed | Bring your own; ours is thinner |
| Evals | `AccuracyEval`, `ReliabilityEval`, `PerformanceEval`, agent-as-judge, importable without AgentOS | `pydantic-evals` in your test suite, using the agent's own types |
| Tracing | OpenTelemetry spans via OpenInference, under `llm.*` and `openinference.*` names; zero `gen_ai.*` attributes | OpenTelemetry GenAI semantic conventions when instrumentation is enabled |

## FAQ

**Is Agno faster?**
It constructs agents faster, and you'll see benchmarks about that. Construction happens once and takes
microseconds either way; a single model call takes hundreds of milliseconds. It's not the number to
choose on. [Under the hood](under-the-hood.md) takes those comparisons apart.

**Can I get something like AgentOS with Pydantic AI?**
Not out of the box. You'd put the agent behind your own FastAPI app, use one of the UI adapters, and
send telemetry to Logfire or your own collector. That's more work, and it's your stack afterwards.

---

*Checked against agno 3.0.9 and Pydantic AI 2.42 on 2026-09-10. The tool behaviour comes from reading the
installed package: `ShellTools.run_shell_command` and its docstring, `PythonTools`, and the
`restrict_to_base_dir` default. `Agent(name='x')` constructs; `agno.agent.agent` does not import `agno.os`.
AgentOS claims are from Agno's documentation, not run. We recheck this page's version pins and behaviour
claims each time Pydantic AI ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
