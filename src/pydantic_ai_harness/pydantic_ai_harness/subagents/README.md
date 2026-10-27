# Subagents

Let an agent delegate self-contained tasks to named child agents.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/subagents/)

## The problem

A single agent that does everything accumulates a large tool set and a long context. Splitting the work across specialized sub-agents keeps each context focused, but wiring up delegation by hand means writing a tool per agent, forwarding deps, threading usage limits, and telling the model what it can delegate to.

## The solution

`SubAgents` takes a sequence of `SubAgent` entries and exposes a single `delegate_task(agent_name, task)` tool. Each delegation runs the chosen sub-agent in its own run -- with its own message history, so it never sees the parent conversation -- and returns its output to the parent. The available sub-agents are listed in the system prompt, so the listing stays in the cached prefix; if it changes between runs of the same conversation, the change is [appended](https://pydantic.dev/docs/ai/core-concepts/agent/#appending-instruction-changes) instead of rewriting the prefix.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgent, SubAgents

researcher = Agent('anthropic:claude-opus-5-5', name='researcher', description='Researches a topic and reports findings')
writer = Agent('anthropic:claude-opus-5-5', name='writer', description='Turns notes into polished prose')

orchestrator = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[SubAgents(agents=[SubAgent(researcher), SubAgent(writer)])],
)

result = orchestrator.run_sync('Research the history of TLS and write a one-paragraph summary.')
print(result.output)
```

A delegate's name -- how the parent model refers to it, and how it is listed in the prompt -- is the agent's own `name`, or a `SubAgent(name=...)` override. Two delegates resolving to the same name is an error, and an agent with no name and no override is rejected.

## The tool

| Tool | Purpose |
|---|---|
| `delegate_task(agent_name, task)` | Run the named sub-agent on a self-contained task and return its output. |

- The sub-agent runs with its own message history, so `task` must be self-contained.
- An unknown `agent_name` raises `ModelRetry`, so the model can correct itself.
- The result returned to the parent is `str(result.output)`.
- With a `models` menu configured, the tool takes an extra `model` argument (see below).

## Deps, usage, tools, and capabilities

- **Deps are forwarded.** The parent run's `deps` are passed to each sub-agent, so sub-agents share the parent's `AgentDepsT` (enforced by the type signature -- every sub-agent is an `AbstractAgent[AgentDepsT, Any]`).
- Delegated agents inherit the parent's workspace unless `workspace_resolver` selects another directory.
  Parent read-only restrictions still apply.
- **Usage is shared by default.** The parent's `usage` is passed to each sub-agent run, so token usage aggregates and a parent `usage_limits` applies across the whole agent tree. Set `forward_usage=False` to give each sub-agent run its own accounting.
- **The parent's tools are not passed on.** A sub-agent runs with its own tools. To delegate to an agent that has all of the parent's tools and capabilities, use `include_self=True` (see "Delegating to the agent itself" below).
- **Capabilities can be shared.** `shared_capabilities` are applied to every sub-agent run -- e.g. give all sub-agents a common guardrail, memory, or planning capability without rebuilding each `Agent`.
- **Sub-agent events can be streamed.** Pass an `event_stream_handler` and it's forwarded to each sub-agent run, so the sub-agent's model-streaming and tool events surface to the caller (the handler receives the sub-agent's own `RunContext`).

`inherit_tools=True` is deprecated and emits a `HarnessDeprecationWarning`. It adds the parent agent's own tools (registered via `tools=` or `toolsets=`) to each sub-agent run, minus the delegate tool, but not the tools contributed by the parent's capabilities. Bind the tools an explicit sub-agent needs directly to its `Agent`, use `include_self=True` to delegate to a fresh run of the agent with everything bound to it, or use a `tool_resolver` to give disk-loaded agents tools.

## Delegating to the agent itself

With `include_self=True`, the roster also lists the running agent itself, as `self`. A delegation to `self` starts a fresh run of that agent (`RunContext.agent`) on the parent run's model (or the `models` option the parent picks), with no parent conversation. Because the child is the same `Agent`, it has every capability, toolset, and instruction bound to it, and they register again in the child run: a guardrail, approval gate, or audit hook bound next to `SubAgents` sees the tool calls the delegate makes, not only the `delegate_task` call. This is what [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) uses by default.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgents

agent = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[SubAgents(include_self=True)],
)
```

- **Only what is bound to the `Agent` carries over.** Capabilities, toolsets, instructions, and model settings passed to the parent's `run()` are not part of the agent, so the delegate does not get them. Passing `SubAgents(include_self=True)` itself to `run()` raises a `UserError` when the run starts, since the delegate would come up without it.
- **Delegation depth is capped.** The delegate carries `delegate_task` too, so `max_depth` (default `3`, counting the top-level run) bounds the tree: the top-level run delegates, its delegates delegate once more, and a run at the limit gets neither `delegate_task` nor the sub-agent listing. The limit applies to every delegation through `SubAgents`, including explicit rosters.
- **Delegates inherit everything, including what may not suit a sub-task.** An `AskUser` capability bound to the agent can prompt the user from inside a delegation, and the delegate returns the agent's own `output_type`, rendered with `str()` (a delegate run that ends with [`DeferredToolRequests`](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) or [`PendingModelResponse`](https://pydantic.dev/docs/ai/core-concepts/batch/#ending-a-run-while-it-waits) returns a steering message instead, as described under failure handling).
- The deprecated `inherit_tools` does not apply to `self`, whose tools are already the parent's. The name `self` is reserved: an explicit delegate with that name is an error, and a disk definition with that name is skipped with a warning.
- **On Temporal, delegate to `self` only.** Sub-agents read from `agent_folders` add a toolset at run time, which Temporal refuses, so pass `SubAgents(include_self=True, agent_folders=None)`; see [delegation on Temporal](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#harness-workspace-capabilities).

## Per-delegate run controls

Each `SubAgent` carries its own budgets, so one delegate's controls do not touch the others. A `SubAgent` with no controls set runs with the `SubAgents` defaults.

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import SubAgent, SubAgents

reproducer = Agent('anthropic:claude-opus-5-5', instructions='Reproduce the reported bug from a minimal script.')
librarian = Agent('anthropic:claude-opus-5-5', instructions='Find relevant docs, issues, and prior art.')

orchestrator = Agent(
    'anthropic:claude-opus-5-5',
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
| `models` | Which keys of the `SubAgents` model menu this delegate may run on, and which one it runs on by default: the first key listed. See "Per-delegation model selection" below. |
| `usage_limits` | A request/token budget for one delegation. The child runs with its own usage accounting, so the budget counts only that child's requests and tokens (not the parent's or siblings'). With `forward_usage=True`, the child's usage is added to the parent's usage after the delegation. Reaching the budget is a soft outcome (see below), not a run-stopping `UsageLimitExceeded`. |
| `timeout_seconds` | A wall-clock budget for one delegation. When the child exceeds it, its run is cancelled and the parent gets a soft steering message instead of hanging on the child. The cancelled child's `event_stream_handler` (if any) stops receiving events without a terminal event. |
| `max_calls` | The maximum number of delegations to this sub-agent per parent run. Once reached, further delegations return a soft budget-exhausted message without running the child. Counts are scoped to one `Agent.run` (a `run_id`) and cleared when it ends, so each parent run and each level of a nested tree budgets independently. |
| `on_failure` | A steering message returned to the parent for any soft degradation of this delegate, in place of the built-in default. Setting it also makes child failures soft (see below). |
| `contain_errors` | Whether an unexpected crash in this delegate is caught and returned to the parent as a bounded `ModelRetry` instead of aborting the parent run (see below). Unset inherits the `SubAgents(contain_errors=...)` default (off). |

## Per-delegation model selection

The orchestrator knows how hard a task is at the moment it writes the brief, so it is the right place to decide which model runs it. Configure a `models` menu and the delegate tool gains a `model` argument that names one of its keys.

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness import SubAgent, SubAgents
from pydantic_ai_harness.subagents import ModelOption

reviewer = Agent('anthropic:claude-opus-5-5', name='reviewer', description='Reviews a diff')
linter = Agent('anthropic:claude-opus-5-5', name='linter', description='Runs the linter and reports failures')

orchestrator = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        SubAgents(
            agents=[SubAgent(reviewer), SubAgent(linter, models=['fast'])],
            models={
                'fast': 'anthropic:claude-haiku-4-5',
                'standard': 'anthropic:claude-sonnet-5',
                'deep': ModelOption(
                    'anthropic:claude-opus-5-5',
                    description='hard reasoning, multi-file changes',
                    settings=ModelSettings(thinking='xhigh'),
                ),
            },
        )
    ],
)
```

- **Off by default.** With no `models` menu the `model` argument is not in the tool schema at all, and every delegation runs exactly as it did before.
- **The keys are the interface.** They are listed in the system prompt with each entry's model and description, and the tool schema offers them as an enum, so the model picks from the menu instead of inventing a model name. Name them for the job (`'fast'`, `'deep'`), not for the vendor.
- **An entry is a model or a `ModelOption`.** `ModelOption(model, description=..., settings=...)` adds a routing hint and per-option `ModelSettings`, so one key can mean "same model, more thinking". Those settings merge over the sub-agent's own `model_settings`, which keeps the parts the option does not set.
- **Resolution order.** The key the parent passed, else the delegate's first allowed key (see below), else the delegate's own model, else the parent run's model.
- **A delegate can be restricted.** `SubAgent(linter, models=['fast'])` pins that delegate to `fast`: the first listed key is what it runs on when the parent passes no `model`, and the others are refused with a `ModelRetry`. The restriction is rendered in the prompt listing (`- linter: Runs the linter (models: fast)`). Restricting to a key the menu does not define is a `ValueError` at construction.
- **A rejected key costs nothing.** An unknown or unavailable key comes back as a `ModelRetry` listing the valid options, before the delegate's `max_calls` budget is charged.

## Per-delegation working directory

Set `workspace_resolver` to offer `delegate_task(working_directory=...)`. The async callback receives
the parent `RunContext` and a `DelegationWorkspaceRequest` with the delegate name and requested
directory, and returns a `Workspace`. The host
validates the request and decides which delegates may use another workspace. Without this option,
the tool omits `working_directory` and children inherit the parent's workspace.

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

The child runs with the returned workspace, so its relative file paths and repository instructions
use that directory. Read-only restrictions from the parent or `SubAgent(read_only=True)` still apply.
Also limit the delegate to trusted read-only tools: arbitrary Python tools can bypass the workspace
API. The resolver must preserve any additional backend restrictions the application requires.

The caller owns the returned workspace and keeps its resources alive until background children
finish. Managed tasks record the resolved directory in `DelegationTask.working_directory`. On resume,
omitting the argument resolves that saved directory again; changing it is rejected. The resolver must
therefore accept its own resolved directory spelling. Older tasks without this field inherit the
parent workspace. Remote and durable delegations reject directory overrides. This option does not
change where disk agent definitions are discovered: `SubAgents.workspace` controls that separately.

## Failure handling

A *soft outcome* returns a steering message to the parent as a normal tool result, so its model reads the message and decides what to do next (rather than immediately re-delegating, which a `ModelRetry` invites). A timeout, a reached `usage_limits` budget, and an exhausted `max_calls` budget are always soft. When `on_failure` is set, the message it carries replaces the built-in default for these outcomes.

A sub-agent run that ends with `DeferredToolRequests` (its `output_type` includes it and a tool needed approval or was deferred) is a soft outcome too. The parent cannot approve the child's tool calls or resume its run, and delegating the same task again would most likely stop the same way, so instead of returning the marker as output the delegation returns a steering message naming the sub-agent and why it stopped, or the `on_failure` message when that is set. A sub-agent that resolves its own deferred calls, for example with [`HandleDeferredToolCalls`](https://pydantic.dev/docs/ai/capabilities/handle-deferred-tool-calls/), never ends this way and returns its output as usual. A run that ends with `PendingModelResponse` (its `output_type` includes it and a model request was left waiting on the provider, for example in a batch) returns a steering message the same way, including on a remote worker, and the request is cancelled, since nobody will check on it later. Cancelling is best effort, and a [batch](https://pydantic.dev/docs/ai/core-concepts/batch/) is only cancelled once no other request in it is still waiting.

A sub-agent run that fails with a *soft model error* (`ModelRetry`, `UnexpectedModelBehavior`, e.g. it exhausted its own retries) is, by default, converted into a `ModelRetry` for the parent -- so the parent's model sees `Sub-agent '<name>' failed: ...` and can react by re-delegating. The delegate tool defaults to `tool_retries=2`, so the parent aborts only after that many consecutive delegate failures; the counter resets after any successful delegation. Raise `tool_retries` to tolerate a flakier sub-agent, or set `None` to inherit the parent agent's default tool retries. Set `on_failure` for a delegate to make its failures soft instead: the child error returns the `on_failure` message as a normal tool result.

Hard errors propagate to stop the whole run. A `UsageLimitExceeded` from a child that has *no* per-delegate `usage_limits` (so it shares the parent's accounting) means the whole tree is out of budget and propagates; a child reaching its *own* `usage_limits` is soft, as above.

An *unexpected crash* -- any other exception the child raises, such as a provider `ModelAPIError`/`FallbackExceptionGroup` or a plain `ValueError` from a bad tool argument -- propagates by default and aborts the parent run. Set `contain_errors=True` (per delegate, or as the `SubAgents` default) to catch it and return it to the parent as a bounded `ModelRetry` instead, so one delegate crash cannot kill the whole run. Containment stays loud: the exception rides the retry message (`Sub-agent '<name>' crashed: ...`), it is logged via the standard `logging` module, and `tool_retries` still bounds consecutive crashes into an abort. This is orthogonal to `on_failure` -- a contained crash always raises the loud retry, never the soft `on_failure` return, so a genuine bug is never masked as success. Cancellation, a shared `UsageLimitExceeded`, pydantic-ai control-flow signals (`CallDeferred`, `ApprovalRequired`, the `Skip*` signals), and `UserError` bypass containment regardless of `contain_errors`. Cancellation covers both kinds: external cancellation (`asyncio.CancelledError`) propagates as-is, and a child's own first-party cancellation (`RunContext.cancel()` inside the child, raising `RunCancelled`) leaves the delegate tool uncontained, after which pydantic-ai isolates it as a failed `delegate_task` return the parent model can react to, not a crash retry that invites re-delegation.

## Events

`SubAgents` emits typed capability events in the `sub_agents` namespace so a host can show a delegation as it runs, and how it ended, without parsing the delegate tool's arguments and result:

| Event | Dispatch | When | Payload |
|---|---|---|---|
| `DelegationStartEvent` | stream | a delegation passed every check and the child run is about to start | `agent_name`, `task`, `truncated`, `model` (the menu key, or `None`), `inherits_tools` |
| `DelegationEndEvent` | stream | the delegation settled into what the parent receives | `agent_name`, `outcome`, `output`, `truncated`, `usage`, `duration_seconds` |

Both are notifications. One delegation is one `delegate_task` call, so a start and its end share the `tool_call_id` core stamps on every event; that is how a subscriber pairs them when the model delegates in parallel.

`outcome` follows the failure handling above: `ok` (the child's output went back to the parent), `timeout`, `budget` (the child's own `usage_limits`), `failed` (a soft model error, returned as `on_failure` or raised as a `ModelRetry`, or a child run that ended with `DeferredToolRequests` or `PendingModelResponse`, returned as a steering message), or `contained` (a crash `contain_errors` caught). `output` is what the delegate tool hands back to the parent in each case: the child's output, the steering message, or the retry text. It is emitted from inside the tool, before any `ToolGuardrail` result guard screens that text for the model; a host that needs the screened version reads the `ToolReturnPart` in core's `FunctionToolResultEvent`. `usage` is the child's own `RunUsage` when it has separate accounting (`usage_limits` set, or `forward_usage=False`) and `None` when it accrues into the parent's usage, where its share is not separable.

A delegation refused before the child runs (an unknown sub-agent, a model key off the menu, an exhausted `max_calls` budget) emits nothing; the tool result says why. An exception that propagates out of the delegate tool (a shared usage limit, an uncontained crash, a cancellation) ends without an end event. A `SubAgentToolset` registered directly in `Agent(toolsets=[...])` has no owning capability and emits nothing. As with any capability event, a listener that raises aborts the parent run.

Their `task` and `output` are cut at `MAX_EVENT_TEXT_CHARS` (4096) with a `truncated` flag, so these two events cannot flood a persisted or forwarded event stream with one verbose delegation. The [task events](#task-events) and `AgentCreatedEvent` are not cut.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai_harness.subagents import DelegationEndEvent, SubAgent, SubAgents


class ReportDelegations(AbstractCapability):
    @on_event(DelegationEndEvent)
    async def on_delegation_end(self, ctx, event: DelegationEndEvent) -> None:
        print(f'{event.agent_name}: {event.outcome} in {event.duration_seconds:.1f}s')


researcher = Agent('anthropic:claude-opus-5-5', name='researcher')
agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[SubAgents(agents=[SubAgent(researcher)]), ReportDelegations()],
)
```

Nested model streaming from the child run is not an event concern; pass an `event_stream_handler` for that.

`SubAgents` emits no OpenTelemetry spans for plain delegation: the child run is a core agent run with its own spans nested under the parent's tool-call span, and the events above carry the outcome a trace would only show as an exception or a tool result.

Task control records its own decisions, which core's spans don't show, as zero-duration spans on the delegating run's tracer, with `sub_agents.`-prefixed attributes, including `sub_agents.agent_name` (for a created agent, whose name the model chose, only as content) and, for a task, `sub_agents.task_id`:

| Span | When | Attributes |
| --- | --- | --- |
| `sub_agents vetoed` | a listener vetoed a task, message, answer, cancellation, or created agent | `sub_agents.event` (the vetoed event's kind), `sub_agents.veto_reason` |
| `sub_agents task backgrounded` | a task moved to the background | `sub_agents.reason`: `requested`, `auto`, or `question` |
| `sub_agents question answered` | a listener answered a task's question, not the model | `sub_agents.question`, `sub_agents.answer` |
| `sub_agents task refused` | `max_running_tasks` refused a delegation | `sub_agents.running_tasks`, `sub_agents.max_running_tasks` |
| `sub_agents task not accepted` | no worker picked a remote task up within `accept_timeout` | `sub_agents.accept_timeout` |
| `sub_agents remote task stopped` | a remote task ended without its worker's report, and its worker was told to stop it | |

A veto reason, question, or answer, and the name a vetoed `create_agent` call asked for, are attached only when the run's instrumentation includes content. While a top-level run waits for its background tasks, a `sub_agents wait for tasks` span covers the wait, with `sub_agents.running_tasks`.

## Task control

By default a delegation finishes inside the `delegate_task` call that started it. With `task_control=TaskControl()`, delegations become tasks the model manages: they can run in the background, ask questions, be steered or cancelled, and be resumed.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgent, SubAgents
from pydantic_ai_harness.subagents import TaskControl

researcher = Agent('anthropic:claude-opus-5-5', name='researcher', description='Researches a topic and reports findings')

orchestrator = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[SubAgents(agents=[SubAgent(researcher)], task_control=TaskControl(auto_background_after=60))],
)
```

| Tool | Purpose |
|---|---|
| `delegate_task(agent_name, task, background=False, resume=None)` | Start a task, or resume a finished one with its task ID. `background=True` returns a receipt right away. |
| `list_tasks()` | The tasks this run started and the tasks they started, with status and IDs (the 50 most relevant). |
| `message_task(task_id, message)` | Steer a running task; it reads the message, as sent, at its next model request. A task's instructions tell it its delegating agent may send it messages. |
| `answer_task(task_id, answer)` | Answer the question a task is waiting on. |
| `cancel_task(task_id, force=False)` | Stop a task before its next model request, or right away with `force=True`, together with the tasks it started. |

- **Reports and questions arrive as messages.** A background task's report is `<task_output>` data, and a task's question `<task_question>` data, on the user channel; the task-control instructions tell the model both are untrusted task data. A top-level run whose model ends while its background tasks still run waits for them, so their reports become its next request.
- **Questions.** Each sub-agent gets an `ask_parent(question)` tool. A question moves its task to the background, because the delegating model can only answer once `delegate_task` returns, and reaches that model as a message. Turn it off per delegate with `SubAgent(can_ask_questions=False)`. A task that can't run in the background (a run-owned workspace that isn't local) gets no `ask_parent`.
- **Limits.** `TaskControl(auto_background_after=...)` moves a task still running after that many seconds to the background. `TaskControl(max_running_tasks=...)` caps how many tasks a conversation runs at once, across its whole delegation tree. `SubAgent(resumable=False)` makes a delegate one-shot (see [Managed delegation sessions](#managed-delegation-sessions) for `one_shot`). All are off by default.
- **Who controls a task.** Only the run that started a task can message, answer, or resume it. Any run above it can cancel it.
- **Each run gets an owner.** When no `DelegationTasks` is bound, `SubAgents` opens one for the run and cancels the tasks still running when the run ends. Delegations then follow the owner's rules in [Managed delegation sessions](#managed-delegation-sessions), including how a delegate's `usage_limits` apply. Another agent's run that one of the run's tools starts is not a task, and doesn't share the owner. Bind your own owner to keep tasks across runs, persist them, and observe them.
- Only one `SubAgents` on an agent can set `task_control`. Task control and `dynamic_agents` are not supported under durable execution: a run that uses them under a durability capability raises a `UserError`. The deprecated wrapper agents, such as `TemporalAgent`, give a run no durability capability, so this check can't catch them; don't use either feature with them.
- Task control and `DelegationTasks` run tasks as asyncio tasks, so they need the asyncio event loop: AnyIO's Trio backend is not supported.

### Task events

With task control, every step of a task is a typed capability event in the `sub_agents` namespace. A task's events go into the event stream of the run that started it, so a nested task's events reach its own parent's run. A bound `DelegationTasks` observer also receives them as `DelegationTaskEvent(event=...)`, including those of a task that settles after its delegating run ended. In place of `TaskProgressEvent`, it receives the task run's own events, as it always did.

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

Unlike `DelegationStartEvent` and `DelegationEndEvent`, these events are not cut at `MAX_EVENT_TEXT_CHARS`: each is the message itself, which a transport carries to the other side, so a prompt, question, answer, or output arrives whole. Nor is `AgentCreatedEvent`. A listener that persists or forwards them bounds them itself.

Events listeners can change are dispatched immediately, so their decisions take effect. A vetoed message is not sent, and the model is told why. With a bound `DelegationTasks`, `DelegationStartEvent` and `DelegationEndEvent` go to its observer only, as in [Managed delegation sessions](#managed-delegation-sessions); with the run's own owner, they go to the run as before.

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

A `SubAgentWorker` serves agents to delegating agents in other processes. The delegating side lists them as `RemoteSubAgent`s, by name, and reaches them through a `SubAgentTransport`:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness.subagents import RemoteSubAgent, SubAgents, SubAgentWorker, TaskControl

transport = ...  # a SubAgentTransport both processes reach


# In the process that hosts the agent:
async def serve():
    researcher = Agent('anthropic:claude-opus-5-5', name='researcher')
    await SubAgentWorker(agents={'researcher': researcher}, transport=transport, deps=None).serve()


# In the delegating process:
orchestrator = Agent(
    'anthropic:claude-opus-5-5',
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

- **A transport is a set of mailboxes.** Implement `send`, `receive`, and optionally `release` over a WebSocket, a Redis list, or anything else that moves bytes, using `serialize_message` and `deserialize_message` at the boundary. Every message is one of the `SubAgentMessage` events above. `InMemoryTransport` serves a worker in the same process.
- **The transport is the trust boundary.** Anyone who can send to it can run the served agents, with their tools and `deps`, and read what they return. Put authentication, authorization, and network binding in your transport. A worker's output and errors reach the delegating model like a local delegate's: in the foreground as tool results, in the background as framed reports.
- **A transport reports a lost worker.** A worker that stops sends `TaskFailedEvent` for the tasks it was running. One that dies without doing so can't: the delegating run then waits for its report until the run is cancelled, unless `receive` raises when the worker is gone. Heartbeats are not part of the protocol.
- **The worker runs the agent** with its own model, tools, and `deps`, which never cross the transport. It takes the agents it serves by name, without per-delegate run controls: those belong to the delegating side's `SubAgents`. It applies the delegating run's usage limits to the task's own usage, and continues the delegating run's trace. The usage it reports with the settled event is added to the delegating run's usage, unless `forward_usage=False`.
- **A task nobody picks up fails.** The worker sends `TaskStartedEvent` when it starts a task. A remote task that no worker picks up within `TaskControl.accept_timeout` seconds (default `30`) fails, and a remote task that stops without its worker's report, for example because its delegating run ended, is force-cancelled on the worker.
- `SubAgentWorker(retries=...)` runs a failed task again from its prompt, with a `TaskRetryEvent`. The worker keeps up to `max_conversations` task conversations (default `1000`) so they can be resumed. The conversation stays in that worker, so serve each remote agent name from one worker: a follow-up that reaches another worker fails, since it has no conversation to resume.
- A remote agent runs on its worker's model, so `delegate_task` refuses a `model` for it, and its delegations emit no `DelegationStartEvent` or `DelegationEndEvent`.

### Agents the model creates

With `dynamic_agents`, the model gets a `create_agent(name, description, instructions, model=None)` tool and can delegate to the agents it creates until the run ends. This works with or without task control.

```python
from pydantic_ai_harness.subagents import DynamicAgents, SubAgents

SubAgents(models={'fast': 'anthropic:claude-haiku-4-5'}, dynamic_agents=DynamicAgents(models=['fast'], max_agents=3))
```

- A created agent is an `Agent` with the model-written instructions and no tools, on the `models` key it was created with, or on the delegating run's model when `DynamicAgents.models` is empty. It has no `name` or `description` of its own, since tracing records those even without content. The model delegates to it by the name it gave in `create_agent`. The description goes only to a custom factory, as `DynamicAgentSpec.description`, and is never shown to the model: created agents aren't added to the cache-stable roster. Pass `DynamicAgents(factory=...)` to build it yourself from the `DynamicAgentSpec`.
- An `AgentCreatedEvent` is dispatched immediately before the agent is created: a listener can rewrite its `instructions` (and its `description`, which only a custom factory reads), or `veto()` it.
- `max_agents` (default `5`) caps how many agents one run creates. Only one `SubAgents` on an agent can set `dynamic_agents`.

Task control, remote agents, and created agents are a port of [`subagents-pydantic-ai`](https://github.com/vstorm-co/subagents-pydantic-ai), a community package by [Vstorm](https://vstorm.co/) (MIT licensed).

## Discovery

The sub-agents are listed in the system prompt via `get_instructions`, using each agent's `description` (or a `SubAgent(description=...)` override). A sub-agent with no description is listed by name alone.

## Loading sub-agents from disk

A repo's agent definitions can become delegates without writing any `Agent` code. With `agent_folders` set, every Claude-style `*.md` file and Codex-style `*.toml` file under those folders is loaded as a sub-agent, alongside the explicitly-passed `agents`. Files are read in sorted filename order.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgents

orchestrator = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[SubAgents(agent_folders='agents')],  # loads .agents/agents/ from the run's workspace
)
```

Definitions are read at the start of every run from the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), so a sandbox's agent files are found and nothing is read from your home directory. To read them from somewhere else, such as definitions that ship with your application, pass `workspace=LocalWorkspaceBackend('/app')`.

`agent_folders` controls which folders are read:

- A folder-name `str` (`'agents'` is the conventional layout): load from `.agents/<name>/`, `.claude/<name>/`, and `.codex/<name>/` under the workspace's working directory, in that order, so a workspace that uses `.agents/` for something else (such as skills) still loads agents from the others. A run without a workspace skips them.
- A sequence of workspace paths, absolute or relative to the working directory, loads from exactly those folders, in order.
- `None`, the default, disables disk loading, exposing only the explicitly-passed `agents`.

Earlier releases loaded the conventional folders by default. Pass `agent_folders='agents'` to keep doing so.

Until this release, the folders were read from this machine, including the home folder `~/.agents/agents/`. A run without a workspace now fails at its start when given a path sequence. Convention discovery warns once when it skips a folder in the current or home directory that the workspace does not reach, naming the folder and the `workspace=` that reads it.

### Definition format

A definition is a Claude-style markdown file with optional frontmatter, or a Codex-style TOML file. Markdown:

```markdown
---
name: researcher
description: Researches a topic and reports findings
tools: Read, Grep
---
You research topics. Report your findings, each with a source.
```

- `name` is the delegate name (how the parent refers to it and how it is listed). It falls back to the filename stem when absent.
- `description` drives the prompt listing.
- The markdown body becomes the agent's instructions.
- `tools` (or `allowed-tools`) is a comma-separated string or a YAML block list. See "Tools" below.
- `model` and `color` are ignored: the model is inherited from the parent (see below), and `color` has no pyai equivalent.

Frontmatter is read by a small, dependency-free parser limited to those keys (`pyyaml` is not a harness dependency). Full YAML frontmatter is not supported.

A Codex-style standalone TOML file:

```toml
name = "reviewer"
description = "Reviews code for bugs"
developer_instructions = "Inspect the code and report findings. Do not edit files."
tools = ["Read", "Grep"]
```

- `name`, `description`, and `developer_instructions` are required nonempty strings.
- `tools` (or `allowed-tools`) is optional: a list of strings or a comma-separated string, not both keys.
- `model`, `effort`, `model_reasoning_effort`, and `color` are ignored with a warning; use `agent_overrides` for models and effort.
- Any other key, including sandbox or permission settings, skips that file with a warning rather than silently granting broader tools. The older `[agents.<name>] config_file` layout is not supported.
- TOML files are parsed with the standard-library `tomllib` module.

Nothing in a definition file is executed. A malformed or invalid file is skipped with a warning without blocking the others.

### Models and effort

Disk agents inherit the parent run's model by default. Per agent, the caller can override the model and set a thinking/effort level via `agent_overrides`, keyed by the agent's name:

```python
from pydantic_ai_harness import SubAgents
from pydantic_ai_harness.subagents import AgentOverride

SubAgents(
    agent_folders='agents',
    agent_overrides={'researcher': AgentOverride(model='anthropic:claude-opus-5-5', effort='high')},
)
```

When `effort` is unset, the disk agent adds no thinking setting, so the inherited model's defaults apply. An explicit value, including `False` or `'minimal'`, is passed through unchanged via Pydantic AI's `ModelSettings.thinking`.

`MINIMUM_EFFORT_FLOOR` and `clamp_effort(level, floor=...)` remain importable for compatibility but are deprecated. `SubAgents` no longer uses them. Pass `AgentOverride(effort=...)` when a disk agent needs an explicit level, or apply an application-specific floor outside the capability.

### Tools

Without a `tool_resolver`, a disk agent gets no tools and its `tools` frontmatter is ignored. To map the frontmatter tool names to toolsets, pass a `tool_resolver`: it receives each tool name (so it can honor entries like `Bash(git:*)`) and returns the toolsets that provide it, or `None` for an unknown name, which is skipped with a warning.

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

When the same name appears in more than one source, the higher-precedence one wins and the others are skipped with a warning: explicitly-passed `agents` first; for convention discovery, the workspace's `.agents/` folder before its `.claude/` folder; and for an explicit path sequence, earlier folders before later ones. A duplicate name within the explicitly-passed `agents` list is still an error.

## Configuration

```python {names="defined"}
from pydantic_ai_harness import SubAgents

SubAgents(
    agents=(),             # Sequence[SubAgent[AgentDepsT]] -- each pairs an agent with its run controls
    models={},             # Mapping[str, Model | str | ModelOption] -- per-delegation model menu (off when empty)
    agent_folders=None,    # folder-name str ('agents' is conventional) | Sequence[str | Path] workspace paths | None
    agent_overrides={},    # Mapping[str, AgentOverride] -- per-disk-agent model/effort override
    tool_resolver=None,    # Callable[[str], Sequence[AgentToolset[object]] | None] -- disk-agent tool mapping
    forward_usage=True,    # share the parent's usage with sub-agent runs
    inherit_tools=False,   # deprecated: use include_self=True, or tool_resolver for disk agents
    shared_capabilities=(),# capabilities applied to every sub-agent run
    event_stream_handler=None,  # forwarded to each sub-agent run to stream its events
    tool_name='delegate_task',
    tool_retries=2,        # extra delegate-tool attempts after a sub-agent error before aborting (None inherits the agent default)
    contain_errors=False,  # default for SubAgent.contain_errors: contain an unexpected crash as a bounded retry
    workspace=None,        # WorkspaceBackend to read agent_folders from instead of the run's workspace
    workspace_resolver=None,  # async (ctx, request: DelegationWorkspaceRequest) -> Workspace
    include_self=False,    # also list the running agent itself as the delegate `self`
    max_depth=3,           # delegation levels, counting the top-level run
    task_control=None,     # TaskControl -- background tasks, task-control tools, questions, and task events
    dynamic_agents=None,   # DynamicAgents -- let the model create sub-agents with `create_agent`
)
```

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgent

agent = Agent('anthropic:claude-opus-5-5')

SubAgent(
    agent,                 # AbstractAgent[AgentDepsT, Any] -- the child agent to run
    name=None,             # delegate name; defaults to the agent's own `name`
    description=None,      # prompt-listing description; defaults to the agent's own `description`
    models=None,           # Sequence[str] -- menu keys this delegate may run on; the first is its default
    usage_limits=None,     # per-delegation request/token budget (isolated accounting)
    timeout_seconds=None,  # per-delegation wall-clock budget
    max_calls=None,        # max delegations to this sub-agent per parent run
    on_failure=None,       # steering message for soft degradations of this delegate
    contain_errors=None,   # contain an unexpected crash as a bounded retry; None inherits the SubAgents default
    resumable=True,        # whether a finished task can be resumed, with a task owner
    can_ask_questions=True,  # give the sub-agent `ask_parent`, with task_control
)
```

```python {names="defined"}
from pydantic_ai_harness.subagents import TaskControl

TaskControl(
    remote_agents=(),      # Sequence[RemoteSubAgent] -- agents a SubAgentWorker serves
    transport=None,        # SubAgentTransport reaching that worker, required with remote_agents
    accept_timeout=30.0,   # seconds a remote task waits for a worker to pick it up
    auto_background_after=None,  # seconds a delegation waits before its task moves to the background
    max_running_tasks=None,      # tasks a conversation may run at once, across its delegation tree
    forward_events=False,  # forward the events of task runs as TaskProgressEvent
)
```

`SubAgents` is not serializable via the agent spec (it holds live `Agent` instances), so `get_serialization_name()` returns `None`.

## Notes

- Sub-agents can themselves have `SubAgents`, forming a tree. Each `SubAgents` stops offering delegation once the run is at its own `max_depth`, so a delegate with a higher limit can go deeper than its parent's. Share `usage` (the default) and set a `usage_limits` on the top-level run to bound the whole tree.
- Delegations the model issues in parallel run as independent sub-agent runs.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Multi-agent applications](https://ai.pydantic.dev/multi-agent-applications/)

## Managed delegation sessions

`SubAgents` keeps its existing foreground-only behavior unless a caller explicitly
opens and binds `DelegationTasks`. This owner adds `background` and `resume` to the
delegate tool's schema, gives every child a stable conversation ID, and owns every
worker, an asyncio task, until shutdown. Use `DelegationReports` on the parent run to deliver settled
background reports through core's queued-message API. Each report is the child's
output in a `UserPromptPart`, framed as `<task_output task_id="..." agent_name="...">`
data, with `outcome="..."` added when the task didn't succeed, like a background
tool's `<tool_result>`. The output was written by a model, so it never enters
system voice, and a closing tag inside it can't end the frame early. Nothing
precedes the frame: the acceptance receipt the parent got at launch already said a
report would follow. Use `is_task_report(part)` to
recognize the output part, or a task's question, which arrives the same way framed as
`<task_question task_id="..." agent_name="...">`, for example to keep it out of a transcript of what the
user typed. Like `BackgroundTools` completions, the metadata is never sent to the
model, but the Vercel AI and AG-UI message conversions omit it: a history that
round-trips through them returns the report or question as plain user text, and
`is_task_report` no longer matches it. History passed in `message_history` is
trusted as the caller's own, this metadata included. Load history from an untrusted
client with a UI adapter, which never puts client metadata on message content;
`sanitize_messages` becomes a
safe path too once [#10943](https://github.com/pydantic/pydantic-ai/pull/10943) lands; until then it keeps
`TextContent.metadata`. No extra
agent loop is implemented. Reports default to `priority='when_idle'`
so active parents finish their current work first. A host that starts an idle
continuation should use `DelegationReports(..., priority='asap')` and
`agent.run(None, ...)`: pending reports then enter the first model request, with
no synthetic user prompt. The host owns wake-up scheduling between runs. A report or question that arrives after a
run has read its last queued messages is kept for the conversation's next run.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.subagents import DelegationReports, DelegationTasks, SubAgents

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[SubAgents(include_self=True)])
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

`opened()` drains workers on exit. A task still running when its process exited is
loaded as cancelled when the owner next opens; if it was waiting for an answer, its output
names the question, so the parent can resume it with the answer. Keep workspace and plugin
resources alive outside that scope. Detached execution is refused for run-owned non-local workspaces;
foreground delegation still works. `max_depth=4` counts the main run and allows
three child layers. An explicitly configured non-default `SubAgents.max_depth`
still takes precedence. Ordinary `SubAgents` retains its original depth default.

`restore(task)` adds a finished `DelegationTask` made elsewhere, such as one imported
from another agent's session, so the parent can resume it by ID. It works before
`opened()` too; the owner loads the task when it opens. A task with the same ID that
has run since, under a different `run_id`, is kept rather than replaced, and
`restore` returns `False`.

`background(task_id)` releases a foreground waiter without restarting the child.
`await cancel(task_id)` stops and drains that child and its descendants. A user stop
blocks model-requested resume until the application explicitly calls
`await allow_resume(task_id)`. `one_shot` names never resume. `SubAgent(resumable=False)`
does the same for one delegate, also with the owner `SubAgents(task_control=...)` opens: use
`one_shot` when the application binds the owner and decides by agent name, and `resumable`
where the delegate is declared. Either one makes the task one-shot. A resume uses the same
child ID and its independent history, a new run ID, and the current direct parent.
A child waits for its own descendants and consumes their reports before its final
output settles. Reports are routed to the direct parent; an idle parent receives
pending reports on its next explicitly started run. Enqueue delivery is acknowledged
only when core emits `EnqueuedMessagesEvent`, and acknowledgements are persisted.
With task control, a child's `ask_parent` question is routed the same way:
`questions(conversation_id=...)` lists the children whose question awaits delivery, and
the observer is notified when one starts waiting, so an application can start its idle
parent's next run to answer it.

`steer(task_id, text)` sends a running child a user message on its own run's queue
with `priority='asap'`: the child reads it at its next model request, and one about to
finish goes on with it instead. It returns `False` when the child is not running, so an
application can offer to resume it. A message sent before the child's run starts waits
for it; one sent as the child's final response settles may arrive too late for it to read.

An observer receives `DelegationTaskEvent`, with the task identity and an optional
correlated child stream event. Events from the child's own stream also carry the
`model_name` and `context_window` of the model streaming that request, so a UI can
show the child's context usage. Managed start/end events carry `task_id` and
`parent_id`. Managed cancellation and uncontained exceptions produce terminal
outcomes; unmanaged events and exception propagation keep their original contract.
Metadata and final/interrupted histories are atomically saved under `directory`.
Pass `step_store` to checkpoint through `StepPersistence` and recover a process-killed
child's latest frontier. Loading an interrupted record never executes its tools.
Inspect possible partial effects before an explicit resume. A resume closes out tool
calls the interruption left unanswered with an `interrupted` tool return, so the child
sees they did not complete.

Managed children with `forward_usage=True` share live usage accounting and inherit
parent ceilings. A per-child budget is converted to an absolute ceiling at launch;
concurrent sibling spend may reach that ceiling earlier, but cannot bypass the
parent's budget. The ordinary unmanaged per-child accounting contract is unchanged.

`agents`, `aliases`, and `instructions` extend the roster and guidance only inside
the bound scope; they do not add delegation to an agent without `SubAgents`.
`SubAgent(read_only=True)` wraps the child's workspace in `ReadOnlyWorkspace`.
Also give that agent only trusted read-only capabilities: arbitrary Python tools
can bypass the workspace API. CLAI's Explore and Plan specialists use filesystem
readers and expose no shell, code execution, or parent plugin tools.

Core's agent/model/tool spans provide execution telemetry; task IDs, parent IDs,
and child run IDs provide correlation. This owner adds no logging exporter.
