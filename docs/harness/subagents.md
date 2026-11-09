---
title: Subagents
description: "Let a Pydantic AI agent delegate tasks to named subagents through one delegate_task tool, with per-subagent budgets, model choice, and Markdown agent files."
---

# Subagents

Let an agent delegate self-contained tasks to named child agents, each running in its own context.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/subagents/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

![The parent model calls delegate_task. SubAgents checks the agent name and its max_calls budget, then runs the delegate in a new run with the parent's deps, usage and workspace, but not its conversation. The output, a steering message, or a retry comes back.](../img/harness/subagents.svg)

## The problem

One agent that does everything accumulates many tools and a long context. Specialized sub-agents keep each context focused, but hand-wired delegation needs a tool per agent, forwarded deps, threaded usage limits, and a prompt listing the delegates.

## The solution

[`SubAgents`][pydantic_ai_harness.subagents.SubAgents] takes [`SubAgent`][pydantic_ai_harness.subagents.SubAgent] entries and exposes one `delegate_task(agent_name, task)` tool. Each delegation runs the sub-agent in its own run, with its own message history, and returns its output to the parent. The system prompt lists the sub-agents, so the listing stays in the cached prefix; a change between runs of a conversation is [appended](../agent.md#appending-instruction-changes) instead of rewriting the prefix.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgent, SubAgents

researcher = Agent('{{ anthropic:large }}', name='researcher', description='Researches a topic')
writer = Agent('{{ anthropic:large }}', name='writer', description='Turns notes into prose')

orchestrator = Agent(
    '{{ anthropic:large }}',
    capabilities=[SubAgents(agents=[SubAgent(researcher), SubAgent(writer)])],
)

result = orchestrator.run_sync('Research the history of TLS and write a one-paragraph summary.')
print(result.output)
```

A delegate's name, used by the parent model and the prompt listing, is the agent's `name` or a `SubAgent(name=...)` override. Duplicate names are an error, and an agent with neither is rejected.

## The tool

| Tool | Purpose |
|---|---|
| `delegate_task(agent_name, task)` | Run the named sub-agent on a self-contained task and return its output. |

- The sub-agent has its own message history, so `task` must be self-contained.
- An unknown `agent_name` raises [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], so the model can correct itself.
- The parent receives `str(result.output)`.
- A `models` menu adds a `model` argument (see below).

## Deps, usage, tools, and capabilities

- **Deps are forwarded.** Every sub-agent is an `AbstractAgent[AgentDepsT, Any]` and gets the parent run's `deps`.
- **The workspace is inherited** unless `workspace_resolver` selects another directory; parent read-only restrictions still apply.
- **Usage is shared by default**, so a parent `usage_limits` applies across the whole agent tree. `forward_usage=False` gives each sub-agent run its own accounting.
- **The parent's tools are not passed on.** A sub-agent runs with its own tools. `include_self=True` delegates to an agent with all of the parent's tools and capabilities (see below).
- **`shared_capabilities`** are applied to every sub-agent run, e.g. a common guardrail, memory, or planning capability.
- **`event_stream_handler`** is forwarded to each sub-agent run with the sub-agent's own [`RunContext`][pydantic_ai.tools.RunContext], so its model-streaming and tool events reach the caller.

`inherit_tools=True` is deprecated and emits a `HarnessDeprecationWarning`. It adds the parent agent's own `tools=`/`toolsets=` tools, minus the delegate tool, to each sub-agent run, but not tools from the parent's capabilities. Instead, bind the tools a sub-agent needs to its `Agent`, use `include_self=True`, or give disk-loaded agents tools with a `tool_resolver`.

## Delegating to the agent itself

With `include_self=True`, the roster also lists the running agent as `self`: a fresh run of `RunContext.agent` on the parent run's model (or the chosen `models` option), without the parent conversation. Everything bound to the `Agent` registers again in the child, so a guardrail, approval gate, or audit hook next to `SubAgents` sees the delegate's tool calls, not only `delegate_task`. [`Coder`](coder.md) does this by default.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgents

agent = Agent(
    '{{ anthropic:large }}',
    capabilities=[SubAgents(include_self=True)],
)
```

- **Only what is bound to the `Agent` carries over**, not capabilities, toolsets, instructions, or model settings passed to `run()`. Passing `SubAgents(include_self=True)` itself to `run()` raises a [`UserError`][pydantic_ai.exceptions.UserError] at run start.
- **Delegation depth is capped.** The delegate has `delegate_task` too. `max_depth` (default `3`, counting the top-level run) lets the top-level run delegate and its delegates delegate once more; a run at the limit gets neither `delegate_task` nor the listing. It applies to every delegation through `SubAgents`, including explicit rosters.
- **Delegates inherit everything, including what may not suit a sub-task**, such as an `AskUser` capability that prompts the user from inside a delegation. The delegate returns the agent's own `output_type`, rendered with `str()`; a run that ends with [`DeferredToolRequests`](../deferred-tools.md) or [`PendingModelResponse`](../batch.md#ending-a-run-while-it-waits) returns a steering message instead (see [Failure handling](#failure-handling)).
- `inherit_tools` does not apply to `self`. With `include_self=True`, the name `self` is reserved: an explicit delegate with that name is an error, and a disk definition with it is skipped with a warning.
- **On Temporal, `agent_folders` is not supported**: disk sub-agents add a toolset at run time, which Temporal refuses, so pass `agent_folders=None`. When `self` is the only delegate, it runs in the workflow, so register a `Model` instance from `models` with `TemporalDurability(models=...)`. Otherwise delegations run in an activity without access to the parent run's model, so each delegate needs its own model or a `models` option; see [delegation on Temporal](../durable_execution/temporal.md#harness-workspace-capabilities).

## Per-delegate run controls

Each `SubAgent` carries its own budgets. Unset controls use the `SubAgents` defaults.

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import SubAgent, SubAgents

reproducer = Agent('{{ anthropic:large }}', instructions='Reproduce the reported bug from a minimal script.')
librarian = Agent('{{ anthropic:large }}', instructions='Find relevant docs, issues, and prior art.')

orchestrator = Agent(
    '{{ anthropic:large }}',
    capabilities=[
        SubAgents(
            agents=[
                SubAgent(reproducer, usage_limits=UsageLimits(request_limit=35), timeout_seconds=600, max_calls=1),
                SubAgent(librarian, usage_limits=UsageLimits(request_limit=18), timeout_seconds=300, max_calls=2),
            ]
        )
    ],
)
```

| Field | Effect |
|---|---|
| `models` | The menu keys this delegate may run on; the first is its default. See [Per-delegation model selection](#per-delegation-model-selection). |
| `usage_limits` | A request/token budget for one delegation, on the child's own accounting (not the parent's or siblings'). With `forward_usage=True`, the child's usage is then added to the parent's. Reaching it is soft, not a run-stopping `UsageLimitExceeded`. |
| `timeout_seconds` | A wall-clock budget for one delegation. On expiry the child run is cancelled and the parent gets a steering message; the child's `event_stream_handler` gets no terminal event. |
| `max_calls` | Delegations to this sub-agent per parent run; beyond it, a budget-exhausted message without running the child. Counted per `run_id` and cleared when the run ends, so each parent run and tree level budgets independently. |
| `on_failure` | Replaces the default steering message for this delegate's soft outcomes, and also makes child failures soft. |
| `contain_errors` | Return an unexpected crash to the parent as a bounded `ModelRetry` instead of aborting the parent run. Unset inherits `SubAgents(contain_errors=...)` (default off). |

## Per-delegation model selection

The orchestrator knows how hard a task is when it writes the brief, so let it pick the model: a `models` menu gives `delegate_task` a `model` argument naming one of its keys.

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness import SubAgent, SubAgents
from pydantic_ai_harness.subagents import ModelOption

reviewer = Agent('{{ anthropic:large }}', name='reviewer', description='Reviews a diff')
linter = Agent('{{ anthropic:large }}', name='linter', description='Runs the linter and reports failures')

orchestrator = Agent(
    '{{ anthropic:large }}',
    capabilities=[
        SubAgents(
            agents=[SubAgent(reviewer), SubAgent(linter, models=['fast'])],
            models={
                'fast': '{{ anthropic:small }}',
                'standard': '{{ anthropic:medium }}',
                'deep': ModelOption(
                    '{{ anthropic:large }}',
                    description='hard reasoning, multi-file changes',
                    settings=ModelSettings(thinking='xhigh'),
                ),
            },
        )
    ],
)
```

- **Off by default.** Without a menu, the tool schema has no `model` argument.
- **The keys are the interface.** The system prompt lists each key with its model and description, and the schema offers them as an enum. Name them for the job (`'fast'`, `'deep'`), not the vendor.
- **A [`ModelOption`][pydantic_ai_harness.subagents.ModelOption]** adds a routing hint and `ModelSettings`, so one key can mean "same model, more thinking". Its settings merge over the sub-agent's own `model_settings`.
- **Resolution order:** the key the parent passed, else the delegate's first allowed key, else the delegate's own model, else the parent run's model.
- **A delegate can be restricted.** With `SubAgent(linter, models=['fast'])`, the first listed key is its default and other keys are refused with a `ModelRetry`. The listing shows `- linter: Runs the linter and reports failures (models: fast)`. A key the menu does not define is a `ValueError` at construction.
- **A rejected key costs nothing.** It returns a `ModelRetry` listing the valid options before `max_calls` is charged.

## Per-delegation working directory

Set `workspace_resolver` to offer `delegate_task(working_directory=...)`. The async callback receives the parent `RunContext` and a `DelegationWorkspaceRequest` with the delegate name and requested directory, validates them, and returns the child's `Workspace`. Without it, the argument is omitted and children inherit the parent's workspace.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.subagents import DelegationWorkspaceRequest, SubAgent, SubAgents

workspaces: dict[str, Workspace] = {'/repos/policy': Workspace(LocalWorkspaceBackend('/repos/policy'))}


async def resolve_workspace(ctx: RunContext[object], request: DelegationWorkspaceRequest) -> Workspace:
    if request.agent_name != 'reviewer' or request.directory not in workspaces:
        raise ModelRetry('Choose the reviewer and a configured directory.')
    return workspaces[request.directory]


reviewer = Agent(name='reviewer', deps_type=object, capabilities=[FileSystem(read_only=True)])
delegation = SubAgents(
    agents=[SubAgent(reviewer, read_only=True)],
    workspace_resolver=resolve_workspace,
)
```

The child's relative paths and repository instructions use the returned workspace. Parent and `SubAgent(read_only=True)` restrictions still apply, but arbitrary Python tools can bypass the workspace API, so give the delegate only trusted read-only tools. The resolver must keep any other backend restrictions the application requires.

The caller owns the returned workspace and keeps it alive until background children finish. Managed tasks save the resolved directory in `DelegationTask.working_directory`; a resume without the argument resolves it again (so the resolver must accept its own resolved spelling), and a different directory is rejected. Older tasks without this field inherit the parent workspace. Remote and durable delegations reject directory overrides. Disk agent discovery is unaffected; `SubAgents.workspace` controls that.

## Failure handling

A *soft outcome* returns a steering message as a normal tool result, so the parent model decides what to do next instead of re-delegating at once, as a `ModelRetry` invites. Timeouts, reached `usage_limits`, and exhausted `max_calls` are always soft; `on_failure` replaces their default message.

A child run that ends with `DeferredToolRequests` (its `output_type` includes it and a tool needed approval or was deferred) is soft too, since the parent can't approve the child's calls or resume it, and re-delegating would likely stop the same way. The tool returns a steering message saying which sub-agent stopped and why, or `on_failure`. A sub-agent that resolves its own deferred calls, e.g. with [`HandleDeferredToolCalls`](../capabilities/handle-deferred-tool-calls.md), returns its output as usual. A run ending with `PendingModelResponse` (its `output_type` includes it and a model request was left waiting on the provider, e.g. in a batch) also returns a steering message, including on a remote worker, and the request is cancelled, best effort: a [batch](../batch.md) is only cancelled once none of its other requests is waiting.

A *soft model error* in the child (`ModelRetry`, or `UnexpectedModelBehavior`, e.g. exhausted retries) becomes a parent `ModelRetry` by default: the parent sees `Sub-agent '<name>' failed: ...` and can re-delegate. `tool_retries` (default `2`; `None` inherits the parent agent's default) bounds consecutive delegate failures before the parent run aborts; a successful delegation resets the count. With `on_failure`, the error returns that message as a normal tool result instead.

Hard errors stop the whole run. A `UsageLimitExceeded` from a child *without* its own `usage_limits` (so it shares the parent's accounting) means the whole tree is out of budget, so it propagates.

An *unexpected crash* (any other exception, such as a provider `ModelAPIError`/`FallbackExceptionGroup` or a tool's `ValueError`) aborts the parent run by default. `contain_errors=True` (per delegate, or as the `SubAgents` default) returns it as a bounded `ModelRetry` instead, and stays loud: the message carries the exception (`Sub-agent '<name>' crashed: ...`), it is logged through `logging`, and `tool_retries` still turns consecutive crashes into an abort. Unlike `on_failure`, it never returns a soft result, so a bug is not reported as success. Never contained: cancellation, a shared `UsageLimitExceeded`, pydantic-ai control-flow signals (`CallDeferred`, `ApprovalRequired`, `Skip*`), and `UserError`. `asyncio.CancelledError` propagates as-is; a child's own `RunContext.cancel()` (`RunCancelled`) leaves the tool uncontained, and pydantic-ai isolates it as a failed `delegate_task` return, not a retry that invites re-delegation.

## Events

`SubAgents` emits typed capability events in the `sub_agents` namespace, so a host can show delegations and outcomes without parsing the tool's arguments and result:

| Event | Dispatch | When | Payload |
|---|---|---|---|
| [`DelegationStartEvent`][pydantic_ai_harness.subagents.DelegationStartEvent] | stream | the delegation passed every check and the child run is about to start | `agent_name`, `task`, `truncated`, `model` (the menu key, or `None`), `inherits_tools` |
| [`DelegationEndEvent`][pydantic_ai_harness.subagents.DelegationEndEvent] | stream | the delegation settled into what the parent receives | `agent_name`, `outcome`, `output`, `truncated`, `usage`, `duration_seconds` |

Both are notifications. A start and its end share the `tool_call_id` core stamps on every event, which pairs them under parallel delegation.

`outcome` is `ok`, `timeout`, `budget` (the child's own `usage_limits`), `failed` (a soft model error, or a run ending with `DeferredToolRequests` or `PendingModelResponse`), or `contained` (a crash `contain_errors` caught). `output` is what the tool returns (child output, steering message, or retry text), emitted before any `ToolGuardrail` result guard screens it; the screened text is the `ToolReturnPart` in core's `FunctionToolResultEvent`. `usage` is the child's own `RunUsage` with separate accounting (`usage_limits` set, or `forward_usage=False`), else `None`.

Nothing is emitted for a delegation refused before the child runs (an unknown sub-agent, a model key off the menu, an exhausted `max_calls`); the tool result says why. An exception escaping the tool (a shared usage limit, an uncontained crash, a cancellation) emits no end event. A `SubAgentToolset` registered directly in `Agent(toolsets=[...])` emits nothing. As with any capability event, a listener that raises aborts the parent run.

`task` and `output` are cut at `MAX_EVENT_TEXT_CHARS` (4096) with a `truncated` flag, so one verbose delegation can't flood a persisted or forwarded stream. [Task events](#task-events) and `AgentCreatedEvent` are not cut.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai_harness.subagents import DelegationEndEvent, SubAgent, SubAgents


class ReportDelegations(AbstractCapability):
    @on_event(DelegationEndEvent)
    async def on_delegation_end(self, ctx, event: DelegationEndEvent) -> None:
        print(f'{event.agent_name}: {event.outcome} in {event.duration_seconds:.1f}s')


researcher = Agent('{{ anthropic:large }}', name='researcher')
agent = Agent(
    '{{ anthropic:large }}',
    capabilities=[SubAgents(agents=[SubAgent(researcher)]), ReportDelegations()],
)
```

To stream the child run's model output, pass an `event_stream_handler`. See [capability events](../capabilities/overview.md#capability-events) for how `@on_event` works.

Plain delegation adds no OpenTelemetry spans: the child's core run spans nest under the parent's tool-call span. Task control records its own decisions, which core's spans don't show, as zero-duration spans on the delegating run's tracer, with `sub_agents.`-prefixed attributes including `sub_agents.agent_name` (only as content for a created agent, whose name the model chose) and, for a task, `sub_agents.task_id`:

| Span | When | Attributes |
| --- | --- | --- |
| `sub_agents vetoed` | a listener vetoed a task, message, answer, cancellation, or created agent | `sub_agents.event` (the vetoed event's kind), `sub_agents.veto_reason` |
| `sub_agents task backgrounded` | a task moved to the background | `sub_agents.reason`: `requested`, `auto`, or `question` |
| `sub_agents question answered` | a listener answered a task's question, not the model | `sub_agents.question`, `sub_agents.answer` |
| `sub_agents task refused` | `max_running_tasks` refused a delegation | `sub_agents.running_tasks`, `sub_agents.max_running_tasks` |
| `sub_agents task not accepted` | no worker picked a remote task up within `accept_timeout` | `sub_agents.accept_timeout` |
| `sub_agents remote task stopped` | a remote task ended without its worker's report, and its worker was told to stop it | |

Veto reasons, questions, answers, and a vetoed `create_agent` call's name are attached only when instrumentation includes content. A `sub_agents wait for tasks` span, with `sub_agents.running_tasks`, covers a top-level run waiting for its background tasks.

## Task control

By default a delegation finishes inside its `delegate_task` call. With `task_control=TaskControl()`, delegations become tasks the model can background, answer, steer, cancel, and resume.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgent, SubAgents
from pydantic_ai_harness.subagents import TaskControl

researcher = Agent('{{ anthropic:large }}', name='researcher', description='Researches a topic')

orchestrator = Agent(
    '{{ anthropic:large }}',
    capabilities=[SubAgents(agents=[SubAgent(researcher)], task_control=TaskControl(auto_background_after=60))],
)
```

| Tool | Purpose |
|---|---|
| `delegate_task(agent_name, task, background=False, resume=None)` | Start a task, or resume a finished one by task ID. `background=True` returns a receipt right away. |
| `list_tasks()` | The tasks this run started and the tasks they started, with status and IDs (the 50 most relevant). |
| `message_task(task_id, message)` | Steer a running task; it reads the message, as sent, at its next model request. Its instructions say its delegating agent may message it. |
| `answer_task(task_id, answer)` | Answer the question a task is waiting on. |
| `cancel_task(task_id, force=False)` | Stop a task and the tasks it started, before its next model request, or right away with `force=True`. |

- **Reports and questions arrive as messages**: `<task_output>` and `<task_question>` data on the user channel, which the task-control instructions mark as untrusted. A top-level run whose model finishes while background tasks run waits for them, and their reports become its next request.
- **Questions.** Each sub-agent gets an `ask_parent(question)` tool. Asking moves the task to the background, since the delegating model can only answer after `delegate_task` returns. Turn it off with `SubAgent(can_ask_questions=False)`. A task that can't run in the background (a run-owned non-local workspace) gets no `ask_parent`.
- **Limits**, all off by default. `TaskControl(auto_background_after=...)` backgrounds a task still running after that many seconds. `TaskControl(max_running_tasks=...)` caps a conversation's concurrent tasks across its whole delegation tree. `SubAgent(resumable=False)` makes a delegate one-shot (see `one_shot` under [Managed delegation sessions](#managed-delegation-sessions)).
- **Who controls a task.** Only the run that started a task can message, answer, or resume it. Any run above it can cancel it.
- **Each run gets an owner.** Without a bound `DelegationTasks`, `SubAgents` opens one per run and cancels its still-running tasks when the run ends; delegations follow [Managed delegation sessions](#managed-delegation-sessions), including for `usage_limits`. Another agent's run started by one of the run's tools is not a task and doesn't share the owner. Bind your own owner to keep, persist, and observe tasks across runs.
- Only one `SubAgents` on an agent can set `task_control`. Task control, and `dynamic_agents` within a bound `DelegationTasks` owner, are not supported under durable execution: a run under a durability capability raises a `UserError`. Deprecated wrapper agents such as `TemporalAgent` give a run no durability capability, so the check can't catch them; don't use task control or `dynamic_agents` with them.
- On Temporal and DBOS, `dynamic_agents` keeps the roster in the workflow and runs each delegation other than to `self` in its own durable unit (on the parent run's model unless the created agent has one), so recovery rebuilds the created agents and reuses each result. The `factory` reruns on replay, so keep it deterministic. Engines that record each tool call in its own unit would lose the roster on replay, so they raise a `UserError`. Under any durability capability, `SubAgents` needs an `id` (set by default); `id=None` raises a `UserError`.
- Task control and `DelegationTasks` run tasks as asyncio tasks, so AnyIO's Trio backend is not supported.

### Task events

With task control, every task step is a typed `sub_agents` capability event in the stream of the run that started the task, so a nested task's events reach its own parent's run. A bound `DelegationTasks` observer also receives them as `DelegationTaskEvent(event=...)`, even after the delegating run ended, with the task run's own events in place of `TaskProgressEvent`.

| Event | When | A listener can |
|---|---|---|
| `TaskAssignedEvent`, `TaskFollowUpEvent` (`TaskRequestEvent` matches both) | a task is about to start, or to resume | rewrite `prompt`, or `veto()` it |
| `TaskStartedEvent` | the task's run is starting, locally or on a worker | |
| `TaskBackgroundedEvent` | the task moved to the background; `reason` is `requested`, `auto`, or `question` | |
| `TaskProgressEvent` | an event of the task's own run, with `TaskControl(forward_events=True)` | |
| `TaskQuestionEvent` | the task asked a question | set `answer`, so the model is not asked |
| `TaskAnswerEvent` | an answer is about to reach the task | rewrite `answer`, or `veto()` it |
| `TaskMessageEvent` | a steering message is about to reach the task | rewrite `message`, or `veto()` it |
| `TaskCancelRequestEvent`, `TaskForceCancelEvent` | a cancellation is about to reach the task | `veto()` it |
| `TaskRetryEvent` | a worker is running a failed remote task again | |
| `TaskCompletedEvent`, `TaskFailedEvent`, `TaskCancelledEvent` | the task settled | rewrite `output` of a completed task |

These events and `AgentCreatedEvent` are not cut at `MAX_EVENT_TEXT_CHARS`: each is the message a transport carries, so it arrives whole. A listener that persists or forwards them bounds them itself.

Events listeners can change are dispatched immediately, so their decisions take effect; a vetoed message is not sent, and the model is told why. A bound `DelegationTasks` sends `DelegationStartEvent` and `DelegationEndEvent` only to its observer; the run's own owner sends them to the run.

```python
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai_harness.subagents import TaskAssignedEvent, TaskQuestionEvent


class DelegationPolicy(AbstractCapability):
    @on_event(TaskAssignedEvent)
    async def check_task(self, ctx, event: TaskAssignedEvent) -> None:
        if 'production' in event.prompt:
            event.veto('ask the user before touching production')

    @on_event(TaskQuestionEvent)
    async def answer_known(self, ctx, event: TaskQuestionEvent) -> None:
        if event.question == 'Which branch?':
            event.answer = 'main'
```

### Remote sub-agents

A [`SubAgentWorker`][pydantic_ai_harness.subagents.SubAgentWorker] serves agents to delegating agents in other processes, which list them by name as [`RemoteSubAgent`][pydantic_ai_harness.subagents.RemoteSubAgent]s and reach them through a [`SubAgentTransport`][pydantic_ai_harness.subagents.SubAgentTransport]:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness.subagents import RemoteSubAgent, SubAgents, SubAgentWorker, TaskControl

transport = ...  # a SubAgentTransport both processes reach


# In the process that hosts the agent:
async def serve():
    researcher = Agent('{{ anthropic:large }}', name='researcher')
    await SubAgentWorker(agents={'researcher': researcher}, transport=transport, deps=None).serve()


# In the delegating process:
orchestrator = Agent(
    '{{ anthropic:large }}',
    capabilities=[
        SubAgents(
            task_control=TaskControl(
                remote_agents=[RemoteSubAgent(name='researcher', description='Researches a topic')],
                transport=transport,
            )
        )
    ],
)
```

- **A transport is a set of mailboxes.** Implement `send`, `receive`, and optionally `release` over anything that moves bytes (a WebSocket, a Redis list), with `serialize_message` and `deserialize_message` at the boundary. Messages are `SubAgentMessage` events. `InMemoryTransport` serves an in-process worker.
- **The transport is the trust boundary.** Anyone who can send to it can run the served agents, with their tools and `deps`, and read the results. Put authentication, authorization, and network binding in your transport. Worker output and errors reach the delegating model like a local delegate's: tool results in the foreground, framed reports in the background.
- **A transport reports a lost worker.** A stopping worker sends `TaskFailedEvent` for its running tasks. If it dies without doing so, the delegating run waits until cancelled, unless `receive` raises when the worker is gone; the protocol has no heartbeats.
- **The worker runs the agent** with its own model, tools, and `deps`, which never cross the transport, and without per-delegate run controls, which stay with the delegating `SubAgents`. It applies the delegating run's usage limits to the task's usage and continues the delegating run's trace. The usage it reports is added to the delegating run's, unless `forward_usage=False`.
- **A task nobody picks up fails.** The worker sends `TaskStartedEvent` when it starts a task. A task no worker picks up within `TaskControl.accept_timeout` seconds (default `30`) fails, and one that stops without its worker's report (e.g. its delegating run ended) is force-cancelled on the worker.
- `SubAgentWorker(retries=...)` reruns a failed task from its prompt, with a `TaskRetryEvent`. The worker keeps up to `max_conversations` task conversations (default `1000`) for resuming, so serve each remote agent name from one worker: a follow-up reaching another worker fails.
- A remote agent runs on its worker's model, so `delegate_task` refuses a `model` for it, and its delegations emit no `DelegationStartEvent` or `DelegationEndEvent`.

### Agents the model creates

With `dynamic_agents`, the model gets a `create_agent(name, description, instructions, model=None)` tool and can delegate to the agents it creates until the run ends, with or without task control.

```python
from pydantic_ai_harness.subagents import DynamicAgents, SubAgents

SubAgents(models={'fast': '{{ anthropic:small }}'}, dynamic_agents=DynamicAgents(models=['fast'], max_agents=3))
```

- A created agent is an `Agent` with the model-written instructions and no tools, running on the `models` key it was created with (or the delegating run's model when `DynamicAgents.models` is empty). It has no `name` or `description`, since tracing records those even without content; the model delegates to it by its `create_agent` name. Created agents aren't in the cache-stable roster, so the model never sees the description; only a custom `DynamicAgents(factory=...)` reads it, from the [`DynamicAgentSpec`][pydantic_ai_harness.subagents.DynamicAgentSpec] it builds the agent from.
- An `AgentCreatedEvent` is dispatched just before the agent is created: a listener can rewrite its `instructions` (and its `description`), or `veto()` it.
- `max_agents` (default `5`) caps how many agents one run creates. Only one `SubAgents` on an agent can set `dynamic_agents`.

Task control, remote agents, and created agents are a port of [`subagents-pydantic-ai`](https://github.com/vstorm-co/subagents-pydantic-ai), a community package by [Vstorm](https://vstorm.co/) (MIT licensed).

## Managed delegation sessions

Without task control, `SubAgents` delegates in the foreground only, unless the caller opens and binds a `DelegationTasks` owner. The owner adds `background` and `resume` to the tool schema, gives each child a stable conversation ID, and owns every worker (an asyncio task) until shutdown, with no extra agent loop.

`DelegationReports` on the parent run delivers settled background reports through core's queued-message API: the child's output in a `UserPromptPart`, framed as `<task_output task_id="..." agent_name="...">` data (plus `outcome="..."` when the task didn't succeed), like a background tool's `<tool_result>`. Model-written output never enters system voice, and a closing tag inside it can't end the frame. Nothing precedes the frame; the launch receipt already said a report would follow.

Reports default to `priority='when_idle'`, so an active parent finishes its current work first. For an idle continuation, use `DelegationReports(..., priority='asap')` with `agent.run(None, ...)`: pending reports enter the first model request without a synthetic user prompt. The host schedules wake-ups between runs. A report or question arriving after a run read its last queued messages waits for the conversation's next run.

`is_task_report(part)` recognizes a report or a task's question (framed as `<task_question task_id="..." agent_name="...">`), e.g. to keep it out of a transcript of what the user typed. As with `BackgroundTools` completions, the metadata is never sent to the model, and the Vercel AI and AG-UI conversions drop it, so a round-tripped history holds plain user text that `is_task_report` doesn't match. `message_history` is trusted as the caller's own, metadata included: load untrusted client history with a UI adapter, which never puts client metadata on message content, or through `sanitize_messages`, which resets `TextContent.metadata`.

```python {test="skip"}
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.subagents import DelegationReports, DelegationTasks, SubAgents

agent = Agent('{{ anthropic:medium }}', capabilities=[SubAgents(include_self=True)])
tasks = DelegationTasks(directory=Path('.task-history'))


async def converse():
    async with tasks.opened():
        with tasks.bind():
            result = await agent.run(
                'Delegate an independent investigation in the background.',
                conversation_id='review',
                capabilities=[DelegationReports(tasks, conversation_id='review')],
            )
            print(result.output)
            # Keep this owner open across subsequent parent turns.
```

`opened()` drains workers on exit. A task still running when its process exited loads as cancelled on the next open; if it was waiting for an answer, its output names the question, so the parent can resume it with the answer. Keep workspace and plugin resources alive outside that scope. Detached execution is refused for run-owned non-local workspaces, but foreground delegation works. Inside the owner, a `SubAgents` at the default `max_depth` uses the owner's (default `4`: the main run plus three child layers); a non-default one takes precedence.

`restore(task)` adds a finished `DelegationTask` from elsewhere, such as another agent's session, so the parent can resume it by ID; called before `opened()`, it loads on open. If a task with that ID has since run under a different `run_id`, it is kept and `restore` returns `False`.

`background(task_id)` releases a foreground waiter without restarting the child. `await cancel(task_id)` stops and drains the child and its descendants; the model can't resume it until the application calls `await allow_resume(task_id)`. `one_shot` agent names never resume; `SubAgent(resumable=False)` does the same where the delegate is declared, also under the owner `SubAgents(task_control=...)` opens. A resume keeps the child ID and history, with a new run ID and the current direct parent.

A child waits for its descendants and consumes their reports before its output settles. Reports go to the direct parent; an idle parent gets them on its next explicitly started run. Delivery is acknowledged only when core emits `EnqueuedMessagesEvent`, and acknowledgements are persisted. With task control, `ask_parent` questions route the same way: `questions(conversation_id=...)` lists children whose question awaits delivery, and the observer is notified when one starts waiting, so the application can start the idle parent's next run to answer it.

`steer(task_id, text)` queues a user message on a running child's run with `priority='asap'`: the child reads it at its next model request, and one about to finish continues instead. It returns `False` when the child isn't running, so the application can offer a resume. A message sent before the child's run starts waits for it; one sent as its final response settles may arrive too late.

An observer receives `DelegationTaskEvent`: the task identity and an optional correlated child stream event, which also carries the streaming model's `model_name` and `context_window`, so a UI can show the child's context usage. Managed start/end events carry `task_id` and `parent_id`. Managed cancellation and uncontained exceptions produce terminal outcomes; unmanaged behavior is unchanged.

Metadata and final/interrupted histories are saved atomically under `directory`. `step_store` checkpoints through `StepPersistence` to recover a process-killed child's latest frontier. Loading an interrupted record never executes its tools, so inspect partial effects before an explicit resume, which answers tool calls the interruption left open with an `interrupted` tool return.

Managed children with `forward_usage=True` share live usage accounting and parent ceilings. A per-child budget becomes an absolute ceiling at launch, which concurrent sibling spend may reach earlier; no child can bypass the parent's budget. Unmanaged per-child accounting is unchanged.

The owner's `agents`, `aliases`, and `instructions` extend the roster and guidance only inside the bound scope, and only for agents with `SubAgents`. `SubAgent(read_only=True)` wraps the child's workspace in `ReadOnlyWorkspace`; give that agent only trusted read-only capabilities too, since arbitrary Python tools can bypass the workspace API. CLAI's Explore and Plan specialists use filesystem readers and expose no shell, code execution, or parent plugin tools.

Core's agent, model, and tool spans provide execution telemetry, and task, parent, and child run IDs provide correlation. The owner adds no logging exporter.

## Discovery

The system prompt lists each sub-agent (via `get_instructions`) with its agent's `description` or a `SubAgent(description=...)` override, or by name alone when it has neither.

## Loading sub-agents from disk

With `agent_folders` set, every Claude-style `*.md` and Codex-style `*.toml` file in those folders becomes a sub-agent alongside `agents`, with no `Agent` code. Files are read in sorted filename order, and missing folders are skipped.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgents

orchestrator = Agent(
    '{{ anthropic:large }}',
    capabilities=[SubAgents(agent_folders='agents')],  # .agents/agents/, .claude/agents/, .codex/agents/
)
```

Definitions are read at the start of each run from the run's [workspace](../workspace.md), so a sandbox's agent files are found and your home directory is not read. For definitions shipped with your application, pass `workspace=LocalWorkspaceBackend('/app')`.

- A folder-name `str` (conventionally `'agents'`) loads `.agents/<name>/`, `.claude/<name>/`, then `.codex/<name>/` under the workspace's working directory, so a workspace that uses `.agents/` for something else (such as skills) still loads agents from the others. A run without a workspace skips them.
- A sequence of workspace paths, absolute or relative to the working directory, loads exactly those folders, in order. A run without a workspace fails at its start.
- `None`, the default, disables disk loading.

Convention discovery warns once when it skips a `.agents/` or `.claude/` folder in the current or home directory that the workspace does not reach, naming the folder and the `workspace=` that reads it.

### Definition format

A Claude-style Markdown definition with optional frontmatter:

```markdown
---
name: researcher
description: Researches a topic and reports findings
tools: Read, Grep
---
You research topics. Report your findings, each with a source.
```

- `name` is the delegate name; it defaults to the filename stem.
- `description` drives the prompt listing, and the body becomes the agent's instructions.
- `tools` (or `allowed-tools`) is a comma-separated string or a YAML block list; see [Tools](#tools).
- `model` and `color` are ignored: the model is inherited from the parent (see below), and `color` has no Pydantic AI equivalent.

A small, dependency-free parser reads only those keys (`pyyaml` is not a harness dependency); full YAML frontmatter is not supported.

A Codex-style standalone TOML definition, parsed with the standard-library `tomllib` module:

```toml
name = "reviewer"
description = "Reviews code for bugs"
developer_instructions = "Inspect the code and report findings. Do not edit files."
tools = ["Read", "Grep"]
```

- `name`, `description`, and `developer_instructions` are required nonempty strings.
- `tools` (or `allowed-tools`, not both) is optional: a list of strings or a comma-separated string.
- `model`, `effort`, `model_reasoning_effort`, and `color` are ignored with a warning; use `agent_overrides` for models and effort.
- Any other key, including sandbox or permission settings, skips the file with a warning rather than silently granting broader tools. The older `[agents.<name>] config_file` layout is not supported.

Nothing in a definition file is executed. A malformed or invalid file is skipped with a warning, without blocking the others.

### Models and effort

Disk agents inherit the parent run's model. `agent_overrides`, keyed by agent name, sets a model and a thinking/effort level:

```python
from pydantic_ai_harness import SubAgents
from pydantic_ai_harness.subagents import AgentOverride

SubAgents(
    agent_folders='agents',
    agent_overrides={'researcher': AgentOverride(model='{{ anthropic:large }}', effort='high')},
)
```

With `effort` unset, the disk agent adds no thinking setting, so the model's defaults apply. An explicit value, including `False` or `'minimal'`, is passed through unchanged as Pydantic AI's `ModelSettings.thinking`.

`MINIMUM_EFFORT_FLOOR` and `clamp_effort(level, floor=...)` remain importable for compatibility but are deprecated and unused by `SubAgents`. Pass `AgentOverride(effort=...)` for an explicit level, or apply an application-specific floor outside the capability.

### Tools

Without a `tool_resolver`, a disk agent gets no tools and its `tools` list is ignored. A `tool_resolver` receives each tool name (so it can honor entries like `Bash(git:*)`) and returns the toolsets that provide it, or `None` for an unknown name, which is skipped with a warning.

```python {names="defined"}
from collections.abc import Sequence

from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness import SubAgents

TOOLSETS: dict[str, Sequence[AgentToolset[object]]] = {}  # your tool name -> toolsets mapping


def resolve(tool_name: str) -> Sequence[AgentToolset[object]] | None:
    return TOOLSETS.get(tool_name)


SubAgents(agent_folders='agents', tool_resolver=resolve)
```

### Precedence

When a name appears in several sources, the higher-precedence one wins and the others are skipped with a warning: `agents` first; then `.agents/`, `.claude/`, `.codex/` for convention discovery, or earlier folders first for a path sequence. A duplicate name within `agents` is still an error.

## Configuration

```python {names="defined"}
from pydantic_ai_harness import SubAgents

SubAgents(
    agents=(),  # SubAgent entries: each agent with its run controls
    models={},  # per-delegation model menu: key -> model or ModelOption (off when empty)
    agent_folders=None,  # folder name ('agents'), workspace paths, or None
    agent_overrides={},  # disk agent name -> AgentOverride (model, effort)
    tool_resolver=None,  # tool name -> toolsets, for disk agents
    forward_usage=True,  # share the parent's usage with sub-agent runs
    inherit_tools=False,  # deprecated: use include_self=True, or tool_resolver for disk agents
    shared_capabilities=(),  # applied to every sub-agent run
    event_stream_handler=None,  # forwarded to each sub-agent run
    tool_name='delegate_task',
    tool_retries=2,  # retries after a sub-agent error before aborting; None inherits the agent default
    contain_errors=False,  # default for SubAgent.contain_errors
    workspace=None,  # WorkspaceBackend to read agent_folders from instead of the run's workspace
    workspace_resolver=None,  # async (ctx, DelegationWorkspaceRequest) -> Workspace
    include_self=False,  # also list the running agent as the delegate `self`
    max_depth=3,  # delegation levels, counting the top-level run
    task_control=None,  # TaskControl: background tasks, task-control tools, questions
    dynamic_agents=None,  # DynamicAgents: the model creates sub-agents with `create_agent`
)
```

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgent

agent = Agent('{{ anthropic:large }}')

SubAgent(
    agent,  # the child agent to run
    name=None,  # delegate name; defaults to the agent's `name`
    description=None,  # prompt-listing description; defaults to the agent's `description`
    models=None,  # menu keys this delegate may run on; the first is its default
    usage_limits=None,  # per-delegation request/token budget (own accounting)
    timeout_seconds=None,  # per-delegation wall-clock budget
    max_calls=None,  # delegations to this sub-agent per parent run
    on_failure=None,  # steering message for soft outcomes
    contain_errors=None,  # contain an unexpected crash; None inherits the SubAgents default
    read_only=False,  # run the child in a ReadOnlyWorkspace
    resumable=True,  # a finished task can be resumed, with a task owner
    can_ask_questions=True,  # give the sub-agent `ask_parent`, with task_control
)
```

```python {names="defined"}
from pydantic_ai_harness.subagents import TaskControl

TaskControl(
    remote_agents=(),  # RemoteSubAgent entries a SubAgentWorker serves
    transport=None,  # SubAgentTransport reaching that worker, required with remote_agents
    accept_timeout=30.0,  # seconds a remote task waits for a worker to pick it up
    auto_background_after=None,  # seconds before a delegation's task moves to the background
    max_running_tasks=None,  # tasks a conversation may run at once, across its delegation tree
    forward_events=False,  # forward the events of task runs as TaskProgressEvent
)
```

`SubAgents` holds live `Agent` instances, so it is not serializable via the [agent spec](../agent-spec.md): `get_serialization_name()` returns `None`.

## Notes

- Sub-agents can have their own `SubAgents`, forming a tree. Each `SubAgents` stops offering delegation once the run is at its own `max_depth`, so a delegate with a higher limit can go deeper than its parent's. To bound the whole tree, share `usage` (the default) and set `usage_limits` on the top-level run.
- Delegations the model issues in parallel run as independent sub-agent runs.

## Further reading

- [Pydantic AI capabilities](../capabilities/overview.md)
- [Multi-agent applications](../multi-agent-applications.md)

## API reference

::: pydantic_ai_harness.subagents.SubAgents

::: pydantic_ai_harness.subagents.SubAgent

::: pydantic_ai_harness.subagents.ModelOption

::: pydantic_ai_harness.subagents.AgentOverride

::: pydantic_ai_harness.subagents.DelegationStartEvent

::: pydantic_ai_harness.subagents.DelegationEndEvent

::: pydantic_ai_harness.subagents.TaskControl

::: pydantic_ai_harness.subagents.RemoteSubAgent

::: pydantic_ai_harness.subagents.SubAgentWorker

::: pydantic_ai_harness.subagents.SubAgentTransport

::: pydantic_ai_harness.subagents.InMemoryTransport

::: pydantic_ai_harness.subagents.serialize_message

::: pydantic_ai_harness.subagents.deserialize_message

::: pydantic_ai_harness.subagents.DynamicAgents

::: pydantic_ai_harness.subagents.DynamicAgentSpec

::: pydantic_ai_harness.subagents.DynamicAgentFactory

::: pydantic_ai_harness.subagents.SubAgentMessage

::: pydantic_ai_harness.subagents.ParentMessage

::: pydantic_ai_harness.subagents.ChildMessage

::: pydantic_ai_harness.subagents.Recipient

::: pydantic_ai_harness.subagents.TaskSettledEvent

::: pydantic_ai_harness.subagents.BackgroundReason

::: pydantic_ai_harness.subagents.TaskRequestEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.TaskAssignedEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.TaskFollowUpEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.TaskStartedEvent

::: pydantic_ai_harness.subagents.TaskBackgroundedEvent

::: pydantic_ai_harness.subagents.TaskProgressEvent

::: pydantic_ai_harness.subagents.TaskQuestionEvent

::: pydantic_ai_harness.subagents.TaskAnswerEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.TaskMessageEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.TaskCancelRequestEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.TaskForceCancelEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.TaskRetryEvent

::: pydantic_ai_harness.subagents.TaskCompletedEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.TaskFailedEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.TaskCancelledEvent
    options:
        inherited_members: true

::: pydantic_ai_harness.subagents.AgentCreatedEvent
    options:
        inherited_members: true
