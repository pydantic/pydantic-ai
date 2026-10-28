"""Sub-agent toolset: a single delegate tool that runs named child agents."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Coroutine, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field as dataclass_field, replace
from functools import partial
from typing import Any, Generic, TypeVar
from uuid import uuid4

import anyio
from typing_extensions import TypeIs

from pydantic_ai import AgentStreamEvent, CapabilityEvent
from pydantic_ai.agent import AbstractAgent, AgentRunResult, EventStreamHandler
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, HookTimeoutError
from pydantic_ai.durable_exec import DurableRunContext
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    RunCancelled,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import repair_messages
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, ObjectJsonSchema, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

# Private import: pydantic-ai has no public way to tell capability-contributed
# toolsets apart from the agent's own in `agent.toolsets`.
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace
from pydantic_ai_harness._sub_agent_output import (
    ServingModelRecorder,
    cancel_pending_output,
    resumable_output_reason,
    unresumable_output_message,
)
from pydantic_ai_harness.subagents._control import (
    DynamicAgents,
    DynamicAgentSpec,
    LocalTaskLink,
    RemoteSubAgent,
    TaskControl,
    TaskTelemetry,
    default_factory,
    reporting_settlement,
    run_owned,
    run_remote,
    task_group,
    validate_agent_name,
)
from pydantic_ai_harness.subagents._events import (
    AgentCreatedEvent,
    DelegationEndEvent,
    DelegationOutcome,
    DelegationStartEvent,
    ParentMessage,
    TaskAnswerEvent,
    TaskAssignedEvent,
    TaskBackgroundedEvent,
    TaskCancelRequestEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskFollowUpEvent,
    TaskForceCancelEvent,
    TaskMessageEvent,
    TaskProgressEvent,
    TaskRequestEvent,
    TaskStartedEvent,
    bounded_text,
)
from pydantic_ai_harness.subagents._models import ModelOption, validate_restriction
from pydantic_ai_harness.subagents._tasks import (
    NOT_BACKGROUNDABLE,
    RESUME_DIRECTORY_MISMATCH,
    DelegationReports,
    DelegationTask,
    DelegationTasks,
)

logger = logging.getLogger(__name__)

SELF_AGENT_NAME = 'self'
"""The name a delegate to the running agent itself is listed and called by (`SubAgents.include_self`)."""

DEFAULT_MAX_DEPTH = 3
"""Default `SubAgents.max_depth`: the top-level run, its delegates, and theirs."""

_depth: ContextVar[int] = ContextVar('pydantic_ai_harness.subagents.depth', default=1)
"""How deep in a delegation tree the current run is, counting the top-level run as 1.

Set around each child run, so every run started inside a delegation -- including a delegation
of a delegation -- reads its own level, and sibling delegations running concurrently in one
parent step each read the level their parent set.
"""


@contextmanager
def delegation_level(level: int) -> Generator[None]:
    """Run what this block starts at `level` of a delegation tree, as a worker does for a remote task."""
    token = _depth.set(level)
    try:
        yield
    finally:
        _depth.reset(token)


def at_max_depth(max_depth: int) -> bool:
    """Whether the current run is as deep in its delegation tree as `max_depth` allows, so it may not delegate."""
    owner = DelegationTasks.current()
    limit = owner.max_depth if owner is not None and max_depth == DEFAULT_MAX_DEPTH else max_depth
    return _depth.get() >= limit


_MODEL_ARG = 'model'
"""Name of the delegate tool's model-selection argument, shared by the function
signature and the schema rewrite that shapes it to the configured menu."""
_MAX_LISTED_TASKS = 50
_PREVIEW_CHARS = 80
CONTROL_TOOL_NAMES = frozenset({'list_tasks', 'message_task', 'answer_task', 'cancel_task'})
"""The tools `SubAgents(task_control=...)` adds next to the delegate tool."""
CREATE_AGENT_TOOL_NAME = 'create_agent'


@dataclass(kw_only=True, frozen=True)
class DelegationWorkspaceRequest:
    """A delegate's requested working directory, to be resolved by the application."""

    agent_name: str
    """The delegate whose workspace is being selected."""

    directory: str
    """The requested path, or the saved resolved path when resuming without a new path."""


WorkspaceResolver = Callable[[RunContext[AgentDepsT], DelegationWorkspaceRequest], Awaitable[Workspace]]
"""Resolve a delegate's requested working directory to a caller-owned workspace."""


def _backgroundable(workspace: Workspace) -> bool:
    """Whether a delegation from this run can outlive the tool call: only one in the run's own local workspace can."""
    return not workspace.attached or isinstance(workspace.backend, LocalWorkspaceBackend)


def _vetoed_reply(what: str, event: ParentMessage | AgentCreatedEvent) -> str:
    """What the model is told when a listener vetoed `event`: `what` happened, and why."""
    return f'{what}: {event.veto_reason or "a policy refused it"}.'


_EventT = TypeVar('_EventT', bound=CapabilityEvent)


def _is_request_response_model(model: object) -> TypeIs[Model]:
    """Narrow a model without losing its provider client type."""
    return isinstance(model, Model)


# Signals that must always reach the parent run, even when a delegate has
# `contain_errors` on. Containing the first five would break the agent graph
# (deferred/approval/skip control-flow); a `UserError` is a setup bug that no
# retry can fix, so masking it into a retry only delays and obscures it.
# First-party cancellation (`RunContext.cancel()` in the child) raises
# `RunCancelled`, a plain `Exception`, so it must be listed here or containment
# would mislabel a deliberate stop as a crash and retry it. Once it escapes the
# delegate tool, pydantic-ai isolates it as a failed tool return in the parent
# (pydantic/pydantic-ai#7199), the same outcome as the uncontained path.
# External cancellation (`asyncio.CancelledError`, a `BaseException`) is out of
# `except Exception`'s reach already, and a shared `UsageLimitExceeded` has its
# own clause.
_ALWAYS_PROPAGATE: tuple[type[Exception], ...] = (
    CallDeferred,
    ApprovalRequired,
    SkipModelRequest,
    SkipToolValidation,
    SkipToolExecution,
    UserError,
    RunCancelled,
)


@dataclass(frozen=True)
class SubAgent(Generic[AgentDepsT]):
    """One delegate: a child agent plus its per-delegate run controls.

    Pass a sequence of these as `SubAgents(agents=[...])`. The delegate's name --
    how the parent model refers to it, and how it is listed in the system prompt --
    is `name` when set, otherwise the agent's own `name`. An agent with neither is
    rejected by `SubAgents`.

    Every control below is optional; an unset field leaves the corresponding
    behaviour at the `SubAgents` default.
    """

    agent: AbstractAgent[AgentDepsT, Any]
    """The agent that runs when this delegate is invoked."""

    name: str | None = None
    """Name the parent model uses to delegate to this agent. Defaults to the
    agent's own `name` when unset."""

    description: str | None = None
    """Description for the system-prompt listing. Defaults to the agent's own
    `description` when unset; a delegate with neither is listed by name alone."""

    models: Sequence[str] | None = None
    """Which of `SubAgents.models` this delegate may run on, as menu keys, and
    which one it runs on by default: the first key listed. Leave it unset to let
    the parent pick any configured option and to fall back to the delegate's own
    model when it picks none. Set it to pin a delegate to one option
    (`models=['fast']`) or to bound an expensive delegate to a subset. Naming a key
    the menu does not define is an error."""

    usage_limits: UsageLimits | None = None
    """Request/token budget for one delegation. When set, the child runs with
    its own usage accounting so the budget counts only the child's own requests
    and tokens (not the parent's or siblings'). When `forward_usage=True`, that
    usage is added to the parent's usage after the delegation. Hitting this budget
    is a soft outcome (steering message), not a run-stopping `UsageLimitExceeded`."""

    timeout_seconds: float | None = None
    """Wall-clock budget for one delegation. When the child exceeds it, the run
    is cancelled and the parent gets a soft steering message instead of hanging
    on the child."""

    max_calls: int | None = None
    """Maximum number of delegations to this sub-agent per parent run. Once
    reached, further delegations return a soft budget-exhausted message without
    running the child."""

    on_failure: str | None = None
    """Steering message returned to the parent for any soft degradation of this
    delegate (timeout, child failure, a child run ending with `DeferredToolRequests` or `PendingModelResponse`,
    usage budget reached, call budget exhausted), in place of the built-in default. Setting it also makes child
    failures soft: a child error returns this message as a normal tool result
    instead of raising a parent `ModelRetry`."""

    contain_errors: bool | None = None
    """Whether an unexpected sub-agent crash is contained instead of aborting the
    parent run. When `True`, an exception the child raises that is not an expected
    soft degradation (a provider `ModelAPIError`/`FallbackExceptionGroup`, a plain
    `ValueError` from a bad tool argument, etc.) is caught and returned to the parent
    as a bounded `ModelRetry`, so one delegate crash cannot kill the whole run. It
    stays loud: the exception rides the retry message and is logged, and
    `tool_retries` still bounds consecutive crashes into an abort. Cancellation, a
    shared usage-limit, pydantic-ai control-flow signals, and `UserError` always
    propagate regardless. Unset inherits `SubAgents.contain_errors` (default off).
    Orthogonal to `on_failure`, which only sets the message for expected soft
    degradations; a contained crash always raises the loud `ModelRetry`."""

    read_only: bool = dataclass_field(default=False, kw_only=True)
    """Run in a `ReadOnlyWorkspace`; supply only trusted read-only tools as well."""

    resumable: bool = dataclass_field(default=True, kw_only=True)
    """Whether a finished task can be resumed, within a `DelegationTasks` owner or with `SubAgents.task_control` set.
    Set this to `False` for a one-shot agent. `DelegationTasks(one_shot=...)` does the same by name, for an owner
    the application binds."""

    can_ask_questions: bool = dataclass_field(default=True, kw_only=True)
    """Whether the sub-agent gets an `ask_parent` tool to ask the delegating agent a question, with
    `SubAgents.task_control` set. Not offered when the task can't run in the background."""

    @property
    def resolved_name(self) -> str | None:
        """The delegate's name: `name` if set, else the agent's own `name`."""
        return self.name or self.agent.name


@dataclass(frozen=True, kw_only=True)
class _Ended:
    """How one delegation settled, before the parent hears about it."""

    outcome: DelegationOutcome
    output: str
    """What the parent model receives: the child's output, or a steering or retry message."""
    cause: Exception | None = None
    """Set when `output` is raised to the parent as a `ModelRetry` rather than returned."""


async def _send_to_owner(
    owner: DelegationTasks, record: DelegationTask, event: DelegationStartEvent | DelegationEndEvent
) -> None:
    """Send a managed delegation's start or end to the application's observer.

    An owner the run opened for itself has no observer, so the event goes to the delegating run instead.
    """
    if run_owned(owner):
        await owner.publish(record, event)
    else:
        await owner.notify(record, event)


def _emits_events(ctx: RunContext[AgentDepsT]) -> bool:
    """Whether the running tool belongs to a capability, so `ctx.emit` accepts a capability event.

    A `SubAgentToolset` registered directly in `Agent(toolsets=[...])` has no owning
    capability, and core refuses capability events from it; it emits nothing.
    """
    tool_name = ctx.tool_name
    if tool_name is None:
        return False
    tool_def = ctx.tools.get(tool_name)
    return tool_def is not None and tool_def.capability_id is not None


def _is_capability_contributed(toolset: AbstractToolset[AgentDepsT]) -> bool:
    """Whether `toolset`'s tree contains a `CapabilityOwnedToolset`."""
    found = False

    def visit(node: AbstractToolset[AgentDepsT]) -> None:
        nonlocal found
        if isinstance(node, CapabilityOwnedToolset):
            found = True

    toolset.apply(visit)
    return found


def _resumable_here(owner: DelegationTasks, resume: str, *, conversation_id: str, agent_name: str) -> DelegationTask:
    """The task `resume` names, if the delegating run, and not another task, started it.

    Raises:
        ModelRetry: the task can't be resumed, or another task started it.
    """
    previous = owner.check_resume(resume, conversation_id=conversation_id, agent_name=agent_name)
    if previous.parent_id != DelegationTasks.child_id():
        raise ModelRetry(f'Task {resume} was started by another task; only the agent that started it can resume it.')
    return previous


def _conversation_id(ctx: RunContext[Any], owner: DelegationTasks) -> str:
    """The conversation the current run's tasks belong to: a child's tasks belong to its parent's."""
    parent_id = DelegationTasks.child_id()
    if parent_id is not None:
        return owner.records[parent_id].conversation_id
    return ctx.conversation_id or ctx.run_id or ''


def _descendants(ctx: RunContext[Any], owner: DelegationTasks) -> list[DelegationTask]:
    """The tasks the current run started, and the tasks they started in turn, oldest first."""
    ancestor = DelegationTasks.child_id()
    conversation_id = _conversation_id(ctx, owner)

    def descends(record: DelegationTask) -> bool:
        parent = record.parent_id
        while parent is not None and parent != ancestor:
            parent = owner.records[parent].parent_id
        return parent == ancestor and record.id != ancestor

    return [r for r in owner.records.values() if r.conversation_id == conversation_id and descends(r)]


def _task_line(owner: DelegationTasks, record: DelegationTask) -> str:
    details = ['running' if record.status == 'running' else f'finished: {record.outcome}']
    if record.background:
        details.append('background')
    if record.parent_id != DelegationTasks.child_id():
        details.append(f'started by task {record.parent_id}')
    details.append('resumable' if record.resumable and not record.user_stopped else 'not resumable')
    line = f'- {record.id} ({record.agent_name}, {", ".join(details)}): {_preview(record.prompt)!r}'
    if (question := owner.pending_question(record.id)) is not None:
        # The question came whole as its own message; this is a reminder of which one it is.
        line += f'\n  Waiting for an answer to: {_preview(question[1])!r}'
    return line


def _preview(text: str) -> str:
    return text if len(text) <= _PREVIEW_CHARS else f'{text[:_PREVIEW_CHARS]}...'


class SubAgentToolset(FunctionToolset[AgentDepsT]):
    """Exposes one delegate tool that dispatches a task to a named sub-agent.

    Each delegation runs the child agent in a fresh run with its own message
    history, so the sub-agent never sees the parent conversation. The parent's
    `deps` are forwarded; its `usage` is shared when enabled; its tools are
    inherited when enabled; any `shared_capabilities` are applied to every
    sub-agent run; and sub-agent events are streamed to `event_stream_handler`
    when one is set. Per-delegate run controls come from each `SubAgent`.

    When a `models` menu is configured, the delegate tool takes an extra `model`
    argument carrying a menu key, so the parent routes each delegation to the
    model that fits the task. Without a menu the argument is not offered at all.
    """

    def __init__(
        self,
        *,
        agents: Mapping[str, SubAgent[AgentDepsT]],
        forward_usage: bool,
        inherit_tools: bool,
        shared_capabilities: Sequence[AgentCapability[AgentDepsT]],
        event_stream_handler: EventStreamHandler[AgentDepsT] | None,
        tool_name: str,
        tool_retries: int | None,
        contain_errors: bool,
        call_counts: dict[str, dict[str, int]],
        models: Mapping[str, ModelOption] | None = None,
        include_self: bool = False,
        max_depth: int = DEFAULT_MAX_DEPTH,
        id: str | None = None,
        task_control: TaskControl | None = None,
        dynamic_agents: DynamicAgents[AgentDepsT] | None = None,
        created: dict[str, dict[str, SubAgent[AgentDepsT]]] | None = None,
        workspace_resolver: WorkspaceResolver[AgentDepsT] | None = None,
    ) -> None:
        super().__init__(id=id)
        if dynamic_agents is not None and created is None:
            raise ValueError(
                '`dynamic_agents` needs a `created` mapping that is cleared as each run ends, as `SubAgents` does.'
            )
        self._task_control = task_control
        control = task_control if task_control is not None else TaskControl()
        self._remote: dict[str, RemoteSubAgent] = {remote.name: remote for remote in control.remote_agents}
        self._transport = control.transport
        self._accept_timeout = control.accept_timeout
        self._auto_background_after = control.auto_background_after
        self._max_running_tasks = control.max_running_tasks
        self._forward_events = control.forward_events
        self._dynamic_agents = dynamic_agents
        # Agents created with `create_agent`, keyed by run_id then name. Shared with the capability, which
        # clears each run's entry in wrap_run.
        self._created: dict[str, dict[str, SubAgent[AgentDepsT]]] = created if created is not None else {}
        # Names a `create_agent` call holds while listeners see it, keyed by run_id: parallel calls see them taken.
        self._creating: dict[str, set[str]] = {}
        self._agents: dict[str, SubAgent[AgentDepsT]] = dict(agents)
        self._forward_usage = forward_usage
        self._inherit_tools = inherit_tools
        self._shared_capabilities = list(shared_capabilities)
        self._event_stream_handler = event_stream_handler
        self._tool_name = tool_name
        self._contain_errors = contain_errors
        self._workspace_resolver = workspace_resolver
        self._models: dict[str, ModelOption] = dict(models or {})
        self._include_self = include_self
        self._max_depth = max_depth
        for name, sub_agent in self._agents.items():
            validate_restriction(name, sub_agent.models, self._models)
        # Run-scoped delegation counts, keyed by run_id then sub-agent name.
        # Shared with the capability, which clears each run's entry in wrap_run.
        self._call_counts = call_counts
        # A delegation to the running agent alone, on its own model, runs the child on the same
        # durable agent, so under Temporal it runs in the workflow, where the child's model requests
        # and tools become activities; an activity cannot run an agent (`ctx.model` is unavailable there).
        # Any other delegate would make its model requests in workflow code, so it stays an activity.
        self_only = include_self and not self._agents and not self._models
        self.add_function(
            self.delegate_task,
            name=tool_name,
            retries=tool_retries,
            prepare=self._prepare_delegate,
            metadata={'temporal': False} if self_only else None,
        )
        if task_control is not None:
            self.add_function(self.list_tasks, prepare=self._prepare_control)
            self.add_function(self.message_task, prepare=self._prepare_control)
            self.add_function(self.answer_task, prepare=self._prepare_control)
            self.add_function(self.cancel_task, prepare=self._prepare_control)
        if dynamic_agents is not None:
            self.add_function(self.create_agent, name=CREATE_AGENT_TOOL_NAME, prepare=self._prepare_control)

    def _prepare_control(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
        """Offer a task-control or `create_agent` tool only where the delegate tool is offered."""
        return None if at_max_depth(self._max_depth) else tool_def

    def _prepare_delegate(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
        """Shape the delegate tool's `model` argument to the configured menu, or hide it at `max_depth`.

        With no menu the argument is dropped from the schema, so a capability that
        does not opt in exposes exactly the tool it did before. With a menu the
        argument becomes an enum of its keys, so the model picks from the offered
        options rather than inventing a model name. The result depends only on
        static configuration and on the run's delegation depth, which does not change during
        a run, so the tool schema stays cache-stable.
        """
        if at_max_depth(self._max_depth):
            return None
        schema: ObjectJsonSchema = {**tool_def.parameters_json_schema}
        properties: dict[str, object] = {**schema.get('properties', {})}
        if self._models:
            properties[_MODEL_ARG] = {
                'type': 'string',
                'enum': list(self._models),
                'description': (
                    'Which model to run the sub-agent on, as one of the keys listed in the instructions. '
                    "Omit it to use the sub-agent's default model."
                ),
            }
        else:
            properties.pop(_MODEL_ARG, None)
        if self._workspace_resolver is None:
            properties.pop('working_directory', None)
        if DelegationTasks.current() is None:
            # `SubAgents(task_control=...)` opens an owner for each run, around the tool calls.
            properties.pop('background', None)
            properties.pop('resume', None)
        schema['properties'] = properties
        return replace(tool_def, parameters_json_schema=schema)

    def _inherited_toolsets(self, ctx: RunContext[AgentDepsT]) -> list[AbstractToolset[AgentDepsT]] | None:
        """The parent agent's own toolsets, excluding capability-contributed ones.

        Capability toolsets are bound to capability instances registered in the
        parent run; carrying them into the sub-agent's run (where their owner is
        not registered) fails `CapabilityOwnedToolset`'s ownership resolution, and
        the tools would arrive without the hooks and instructions that make them
        work. Use `shared_capabilities` to share a capability with sub-agents.
        The delegate tool itself is also filtered out by name, so delegation
        cannot recurse. When this toolset was registered via the `SubAgents`
        capability the capability filter already drops it; the name filter covers
        direct registration in `Agent(toolsets=[...])`, where nothing wraps it in
        `CapabilityOwnedToolset`.
        """
        agent = ctx.agent
        if agent is None:  # pragma: no cover - the running agent is always set during a run
            return None
        # Capability toolsets surface as `CombinedToolset(CapabilityOwnedToolset(...))`
        # entries, so ownership is detected by walking each tree. Only core's capability
        # assembly constructs `CapabilityOwnedToolset`, so a tree containing one is
        # capability-contributed in its entirety.
        return [
            toolset.filtered(lambda _ctx, tool_def: tool_def.name != self._tool_name)
            for toolset in agent.toolsets
            if not _is_capability_contributed(toolset)
        ]

    def _call_budget_message(
        self, ctx: RunContext[AgentDepsT], agent_name: str, sub_agent: SubAgent[AgentDepsT]
    ) -> str | None:
        """Charge this run's delegation to `agent_name` against `max_calls`, and steer the model away when it is over.

        Runs synchronously before any await, so concurrent delegations in one run
        count without a lock.
        """
        if sub_agent.max_calls is None:
            return None
        counts = self._call_counts.setdefault(ctx.run_id or '', {})
        counts[agent_name] = counts.get(agent_name, 0) + 1
        if counts[agent_name] <= sub_agent.max_calls:
            return None
        return self._steer(
            sub_agent.on_failure,
            f'Delegate budget for {agent_name!r} is exhausted for this run '
            f'({sub_agent.max_calls} call(s)). Synthesize from existing evidence and '
            f'choose the next action; do not delegate to {agent_name!r} again.',
        )

    def _resolve_model_key(self, agent_name: str, sub_agent: SubAgent[AgentDepsT], key: str | None) -> str | None:
        """The menu key one delegation runs on, or `None` to leave the model as it was.

        An unset `key` falls back to the delegate's own first allowed option, and
        to no option at all when the delegate allows the whole menu.

        Raises:
            ModelRetry: the key is not on the menu, or not one this delegate allows.
        """
        allowed = sub_agent.models
        if key is None:
            return allowed[0] if allowed else None
        if key not in self._models:
            available = ', '.join(self._models) or '(none configured)'
            raise ModelRetry(f'Unknown model {key!r}. Available models: {available}.')
        if allowed is not None and not allowed:
            raise ModelRetry(f'Sub-agent {agent_name!r} runs on your own model: omit `model`.')
        if allowed is not None and key not in allowed:
            raise ModelRetry(
                f'Sub-agent {agent_name!r} cannot run on model {key!r}. Available models for it: {", ".join(allowed)}.'
            )
        return key

    async def delegate_task(
        self,
        ctx: RunContext[AgentDepsT],
        agent_name: str,
        task: str,
        model: str | None = None,
        background: bool = False,
        resume: str | None = None,
        *,
        working_directory: str | None = None,
    ) -> str:
        """Delegate a self-contained task to a named sub-agent and return its result.

        The sub-agent runs in its own fresh context and does not see this
        conversation, so `task` must contain everything it needs.

        Args:
            ctx: The run context (provides the parent's deps, usage, and tools).
            agent_name: Name of the sub-agent to run. Must be one of the agents
                listed in the instructions.
            task: The complete, self-contained instruction for the sub-agent.
            model: Which model to run the sub-agent on, as one of the model keys
                listed in the instructions. Omit it to use the sub-agent's default
                model. Only offered when a model menu is configured.
            background: Return an acceptance receipt while the child runs, within a task owner.
            resume: Continue the saved child with this ID, within a task owner.
            working_directory: Start the child in this directory using the configured workspace resolver.
                On resume, omit it to keep the saved directory, or supply the same resolved directory.
        """
        owner = DelegationTasks.current()
        if owner is not None:
            agent_name = owner.aliases.get(agent_name, agent_name)
        elif background or resume is not None:
            raise ModelRetry('Background execution and resume require an open `DelegationTasks` owner')
        if self._task_control is not None and owner is not None:
            return await self._delegate_controlled(
                ctx,
                owner,
                agent_name,
                task,
                model=model,
                background=background,
                resume=resume,
                working_directory=working_directory,
            )
        sub_agent = self._resolve_agent(ctx, agent_name)

        if owner is not None and resume is not None and model is None and resume in owner.records:
            model = owner.records[resume].model

        if owner is not None and resume is not None and run_owned(owner):
            # In the tree of a task-control run, plain delegations keep its rule on who resumes a task.
            _resumable_here(owner, resume, conversation_id=_conversation_id(ctx, owner), agent_name=agent_name)

        # Resolved before the call budget is charged, so a bad model key costs nothing.
        key = self._resolve_model_key(agent_name, sub_agent, model)
        previous = (
            owner.check_resume(resume, conversation_id=_conversation_id(ctx, owner), agent_name=agent_name)
            if owner is not None and resume is not None
            else None
        )
        workspace, directory = await self._delegation_workspace(ctx, agent_name, working_directory, previous)

        if (exhausted := self._call_budget_message(ctx, agent_name, sub_agent)) is not None:
            return exhausted
        if owner is not None:

            async def run(record: DelegationTask) -> str:
                return await self._run_delegation(
                    ctx, agent_name, sub_agent, task=task, key=key, record=record, workspace=workspace
                )

            return await owner.delegate(
                agent_name=agent_name,
                prompt=task,
                conversation_id=ctx.conversation_id or ctx.run_id or '',
                model=key,
                background=background,
                resume=resume,
                run=run,
                backgroundable=_backgroundable(workspace),
                resumable=sub_agent.resumable,
                working_directory=directory,
            )
        return await self._run_delegation(ctx, agent_name, sub_agent, task=task, key=key, workspace=workspace)

    async def _delegation_workspace(
        self,
        ctx: RunContext[AgentDepsT],
        agent_name: str,
        requested: str | None,
        previous: DelegationTask | None,
    ) -> tuple[Workspace, str | None]:
        if previous is not None:
            if requested is not None and previous.working_directory is None:
                raise ModelRetry(RESUME_DIRECTORY_MISMATCH)
            requested = requested if requested is not None else previous.working_directory
        if requested is None:
            return ctx.workspace, None
        if self._workspace_resolver is None:
            raise ModelRetry('`working_directory` requires `SubAgents(workspace_resolver=...)`.')
        if isinstance(ctx, DurableRunContext) or ctx.in_durable_context or agent_name in self._remote:
            raise ModelRetry('`working_directory` is not supported for durable or remote delegations.')
        workspace = await self._workspace_resolver(
            ctx, DelegationWorkspaceRequest(agent_name=agent_name, directory=requested)
        )
        directory = await workspace.working_dir()
        if previous is not None and directory != previous.working_directory:
            # Reject before charging max_calls or publishing task-start events.
            raise ModelRetry(RESUME_DIRECTORY_MISMATCH)
        return (ReadOnlyWorkspace(workspace) if ctx.workspace.read_only else workspace), directory

    def _resolve_agent(self, ctx: RunContext[AgentDepsT], agent_name: str) -> SubAgent[AgentDepsT]:
        """The delegate `agent_name` names, with the running agent resolved at call time for `include_self`.

        Raises:
            ModelRetry: no delegate has that name.
        """
        if self._include_self and agent_name == SELF_AGENT_NAME:
            agent = ctx.agent
            if agent is None:  # pragma: no cover - the running agent is always set during a run
                raise UserError('Delegating to the running agent requires `RunContext.agent`.')
            return SubAgent(agent, name=SELF_AGENT_NAME)
        owner = DelegationTasks.current()
        if owner is not None and agent_name in owner.agents:
            return owner.agents[agent_name]
        # Read only when agents can be created: a durable activity's context may not carry `run_id`.
        created = self._created.get(ctx.run_id or '', {}) if self._dynamic_agents is not None else {}
        sub_agent = self._agents.get(agent_name) or created.get(agent_name)
        if sub_agent is None:
            names = [*self._agents, *self._remote, *created]
            if self._include_self:
                names.append(SELF_AGENT_NAME)
            raise ModelRetry(f'Unknown sub-agent {agent_name!r}. Available sub-agents: {", ".join(sorted(names))}.')
        return sub_agent

    async def _delegate_controlled(
        self,
        ctx: RunContext[AgentDepsT],
        owner: DelegationTasks,
        agent_name: str,
        task: str,
        *,
        model: str | None,
        background: bool,
        resume: str | None,
        working_directory: str | None,
    ) -> str:
        """Delegate with `SubAgents(task_control=...)`: announce the task, then hand it to the owner."""
        parent_id = DelegationTasks.child_id()
        conversation_id = _conversation_id(ctx, owner)
        previous = None
        if resume is not None:
            previous = _resumable_here(owner, resume, conversation_id=conversation_id, agent_name=agent_name)
            if model is None:
                model = previous.model
        workspace, directory = await self._delegation_workspace(ctx, agent_name, working_directory, previous)
        backgroundable = _backgroundable(workspace)
        if background and not backgroundable:
            raise ModelRetry(NOT_BACKGROUNDABLE)
        remote = self._remote.get(agent_name)
        sub_agent: SubAgent[AgentDepsT] | None = None
        key: str | None = None
        if remote is not None:
            if model is not None:
                raise ModelRetry(f"Sub-agent {agent_name!r} runs on its worker's model; omit `model`.")
            resumable, can_ask = remote.resumable, remote.can_ask_questions
        else:
            sub_agent = self._resolve_agent(ctx, agent_name)
            key = self._resolve_model_key(agent_name, sub_agent, model)
            if (exhausted := self._call_budget_message(ctx, agent_name, sub_agent)) is not None:
                return exhausted
            resumable, can_ask = sub_agent.resumable, sub_agent.can_ask_questions
        limit = self._max_running_tasks
        if limit is not None and (running := owner.running_count(conversation_id)) >= limit:
            self._telemetry(ctx).decision(
                'task refused', agent_name=agent_name, attributes={'running_tasks': running, 'max_running_tasks': limit}
            )
            raise ModelRetry(
                f'{running} tasks are already running, the most allowed at once. Wait for a report, or cancel a '
                'task, before starting another.'
            )
        task_id = resume or uuid4().hex
        start_type = TaskAssignedEvent if previous is None else TaskFollowUpEvent
        # The task counts as running while listeners see it, so concurrent delegations can't overrun the limit.
        with owner.starting(conversation_id):
            # The observer gets the start once the task runs, so a vetoed start reaches only listeners.
            start = await self._announce(
                ctx,
                owner,
                None,
                start_type(
                    task_id=task_id,
                    agent_name=agent_name,
                    prompt=task,
                    parent_task_id=parent_id,
                    depth=_depth.get() + 1,
                    # The delegating model can only answer from the background.
                    can_ask_questions=can_ask and backgroundable,
                    forward_events=self._forward_events,
                    usage_limits=ctx.usage_limits,
                ),
            )
        if start.vetoed:
            return _vetoed_reply('The task was not sent', start)

        async def run(record: DelegationTask) -> str:
            await owner.notify(record, start)
            if remote is not None:
                assert self._transport is not None, '`SubAgents` requires a `transport` for `remote_agents`.'
                return await run_remote(
                    owner=owner,
                    record=record,
                    start=start,
                    transport=self._transport,
                    accept_timeout=self._accept_timeout,
                    usage=ctx.usage if self._forward_usage else None,
                    telemetry=self._telemetry(ctx),
                )
            assert sub_agent is not None
            return await self._run_controlled(ctx, owner, record, sub_agent, key=key, start=start, workspace=workspace)

        after = self._auto_background_after
        async with task_group() as group:
            if after is not None and not background and backgroundable:
                group.start_soon(self._background_after, ctx, owner, task_id, after)
            try:
                output = await owner.delegate(
                    agent_name=agent_name,
                    prompt=start.prompt,
                    conversation_id=conversation_id,
                    model=key,
                    background=background,
                    resume=resume,
                    run=run,
                    backgroundable=backgroundable,
                    task_id=task_id,
                    resumable=resumable,
                    working_directory=directory,
                )
            finally:
                # The timer only counts while this call waits for the task.
                group.cancel_scope.cancel()
        record = owner.records[task_id]
        if background and record.status == 'running':
            await self._announce(
                ctx, owner, record, TaskBackgroundedEvent(task_id=task_id, agent_name=agent_name, reason='requested')
            )
        return output

    async def _background_after(
        self, ctx: RunContext[AgentDepsT], owner: DelegationTasks, task_id: str, delay: float
    ) -> None:
        await anyio.sleep(delay)
        record = owner.records[task_id]
        # The delegation tool call stops waiting, and this with it, once the task settles or moves to the background.
        if record.status == 'running' and not record.background:  # pragma: no branch
            event = TaskBackgroundedEvent(task_id=task_id, agent_name=record.agent_name, reason='auto')
            await self._announce(ctx, owner, record, event)
            owner.background(task_id)

    async def _run_controlled(
        self,
        ctx: RunContext[AgentDepsT],
        owner: DelegationTasks,
        record: DelegationTask,
        sub_agent: SubAgent[AgentDepsT],
        *,
        key: str | None,
        start: TaskRequestEvent,
        workspace: Workspace,
    ) -> str:
        """Run a local task and report how it settled as task events."""
        link = LocalTaskLink(
            owner=owner, record=record, can_ask=start.can_ask_questions, telemetry=self._telemetry(ctx)
        )
        await owner.publish(record, TaskStartedEvent(task_id=record.id, agent_name=record.agent_name))
        task_id, agent_name = record.id, record.agent_name
        async with reporting_settlement(owner, record):
            output = await self._run_delegation(
                ctx,
                agent_name,
                sub_agent,
                task=record.prompt,
                key=key,
                record=record,
                extra_capabilities=[link],
                workspace=workspace,
            )
        if record.outcome == 'ok':
            completed = TaskCompletedEvent(task_id=task_id, agent_name=agent_name, output=output)
            return (await owner.publish(record, completed)).output
        await owner.publish(record, TaskFailedEvent(task_id=task_id, agent_name=agent_name, error=output))
        return output

    async def _announce(
        self, ctx: RunContext[AgentDepsT], owner: DelegationTasks, record: DelegationTask | None, event: _EventT
    ) -> _EventT:
        """Emit a task event from one of this toolset's tool calls, then send it to the observer."""
        if _emits_events(ctx):
            await ctx.emit(event)
        if isinstance(event, ParentMessage) and event.vetoed:
            self._telemetry(ctx).vetoed(event)
        elif isinstance(event, TaskBackgroundedEvent):
            self._telemetry(ctx).backgrounded(event)
        if record is not None:
            await owner.notify(record, event)
        return event

    def _telemetry(self, ctx: RunContext[AgentDepsT]) -> TaskTelemetry:
        # Read only when agents can be created: a durable activity's context may not carry `run_id`.
        created = self._created.get(ctx.run_id or '', {}) if self._dynamic_agents is not None else {}
        return TaskTelemetry.of(ctx, model_named=frozenset(created))

    def _owner(self) -> DelegationTasks:
        owner = DelegationTasks.current()
        assert owner is not None, '`SubAgents(task_control=...)` opens or binds an owner around every run.'
        return owner

    def _visible(self, ctx: RunContext[AgentDepsT], owner: DelegationTasks, task_id: str) -> DelegationTask:
        """A task this run started, directly or through the tasks it started."""
        record = owner.records.get(task_id)
        if record is None or record not in _descendants(ctx, owner):
            raise ModelRetry(f'Unknown task {task_id!r}. Call `list_tasks` to see your tasks.')
        return record

    async def list_tasks(self, ctx: RunContext[AgentDepsT]) -> str:
        """List the tasks you started, and the tasks they started in turn, with their status and IDs."""
        owner = self._owner()
        visible = _descendants(ctx, owner)
        if not visible:
            return 'You have not started any tasks.'
        running = [record for record in visible if record.status == 'running']
        finished = [record for record in visible if record.status != 'running']
        # Running tasks first, the newest of them when there are too many, then the newest finished ones.
        listed_running = running[-_MAX_LISTED_TASKS:]
        room = _MAX_LISTED_TASKS - len(listed_running)
        listed_finished = finished[-room:] if room else []
        shown = {record.id for record in [*listed_running, *listed_finished]}
        lines = [_task_line(owner, record) for record in visible if record.id in shown]
        omitted = [
            f'{count} {kind} task{"s" if count != 1 else ""}'
            for count, kind in (
                (len(running) - len(listed_running), 'older running'),
                # With no room left, the finished tasks left out may be newer than those shown.
                (len(finished) - len(listed_finished), 'older finished' if listed_finished else 'finished'),
            )
            if count
        ]
        if omitted:
            lines.append(f'({" and ".join(omitted)} not shown.)')
        return '\n'.join(lines)

    async def message_task(self, ctx: RunContext[AgentDepsT], task_id: str, message: str) -> str:
        """Send a message to a running task you started, to steer it. It reads the message at its next model request.

        Args:
            ctx: The run context.
            task_id: The task's ID.
            message: What to tell it.
        """
        owner = self._owner()
        record = self._visible(ctx, owner, task_id)
        if record.parent_id != DelegationTasks.child_id():
            raise ModelRetry(f'Task {task_id} is steered by the task that started it, {record.parent_id}, not you.')
        if record.status != 'running':
            raise ModelRetry(
                f'Task {task_id} has finished ({record.outcome}), and messages only reach running tasks. '
                'To continue it, delegate again with `resume`.'
            )
        event = await self._announce(
            ctx, owner, record, TaskMessageEvent(task_id=task_id, agent_name=record.agent_name, message=message)
        )
        if event.vetoed:
            return _vetoed_reply('The message was not sent', event)
        if record.agent_name in self._remote:
            assert self._transport is not None
            await self._transport.send(event)
        elif not owner.steer(task_id, event.message):
            return f'Task {task_id} finished before the message reached it.'
        return f'Message sent to task {task_id}.'

    async def answer_task(self, ctx: RunContext[AgentDepsT], task_id: str, answer: str) -> str:
        """Answer the question a task you started is waiting on.

        Args:
            ctx: The run context.
            task_id: The task's ID.
            answer: Your answer.
        """
        owner = self._owner()
        record = self._visible(ctx, owner, task_id)
        if record.parent_id != DelegationTasks.child_id():
            raise ModelRetry(f'Task {task_id} asks the task that started it, {record.parent_id}, not you.')
        question = owner.pending_question(task_id)
        if question is None:
            raise ModelRetry(f'Task {task_id} is not waiting for an answer.')
        event = await self._announce(
            ctx,
            owner,
            record,
            TaskAnswerEvent(task_id=task_id, agent_name=record.agent_name, question_id=question[0], answer=answer),
        )
        if event.vetoed:
            return _vetoed_reply('The answer was not sent', event)
        if not owner.answer(task_id, question_id=question[0], answer=event.answer):
            return f'Task {task_id} is no longer waiting for that answer.'
        return f'Answer sent to task {task_id}.'

    async def cancel_task(self, ctx: RunContext[AgentDepsT], task_id: str, force: bool = False) -> str:
        """Cancel a running task, and every task it started.

        Args:
            ctx: The run context.
            task_id: The task's ID.
            force: Stop it immediately, interrupting its current step. By default it stops before its next
                model request.
        """
        owner = self._owner()
        record = self._visible(ctx, owner, task_id)
        if record.status != 'running':
            return f'Task {task_id} already finished ({record.outcome}); there is nothing to cancel.'
        event_type = TaskForceCancelEvent if force else TaskCancelRequestEvent
        event = await self._announce(ctx, owner, record, event_type(task_id=task_id, agent_name=record.agent_name))
        if event.vetoed:
            return _vetoed_reply('The cancellation was not sent', event)
        if force:
            await owner.cancel(task_id, user=False)
        elif record.agent_name in self._remote:
            assert self._transport is not None
            await self._transport.send(event)
        else:
            owner.request_stop(task_id)
        reporter = (
            'you' if record.parent_id == DelegationTasks.child_id() else f'task {record.parent_id}, which started it'
        )
        resumable = 'It can be resumed afterwards.' if record.resumable else 'It is one-shot and cannot be resumed.'
        return f'Cancelling task {task_id}. Its report goes to {reporter}. {resumable}'

    async def create_agent(
        self, ctx: RunContext[AgentDepsT], name: str, description: str, instructions: str, model: str | None = None
    ) -> str:
        """Create a new sub-agent you can delegate tasks to for the rest of this run.

        Args:
            ctx: The run context.
            name: A unique name: 1 to 64 letters, digits, `_` or `-`.
            description: What the sub-agent is for.
            instructions: Its instructions: its role, how to work, and what to return.
            model: Which model to run it on, as one of the model keys listed in the instructions. Omit it to
                use the default.
        """
        dynamic = self._dynamic_agents
        assert dynamic is not None, '`create_agent` is only offered with `dynamic_agents` set.'
        if (problem := validate_agent_name(name)) is not None:
            raise ModelRetry(problem)
        run_key = ctx.run_id or ''
        created = self._created.setdefault(run_key, {})
        reserved = self._creating.get(run_key, set())
        taken = {*self._agents, *self._remote, *created, *reserved}
        if self._include_self:
            taken.add(SELF_AGENT_NAME)
        if (owner := DelegationTasks.current()) is not None:
            # They resolve first, so a created namesake would be unreachable.
            taken.update(owner.agents, owner.aliases)
        if name in taken:
            raise ModelRetry(f'A sub-agent named {name!r} already exists.')
        if len(created) + len(reserved) >= dynamic.max_agents:
            raise ModelRetry(f'You already created {dynamic.max_agents} sub-agents, the most allowed in one run.')
        if dynamic.models:
            if model is None:
                model = dynamic.models[0]
            elif model not in dynamic.models:
                raise ModelRetry(f'Model {model!r} is not allowed. Choose one of: {", ".join(dynamic.models)}.')
        elif model is not None:
            raise ModelRetry('You cannot choose a model; omit `model` to use your own.')
        event = AgentCreatedEvent(name=name, description=description, instructions=instructions, model=model)
        # Reserved before listeners see it, so a parallel call cannot take the name or the last place meanwhile.
        creating = self._creating.setdefault(run_key, set())
        creating.add(name)
        try:
            if _emits_events(ctx):
                await ctx.emit(event)
            if event.vetoed:
                self._telemetry(ctx).vetoed(event)
                return _vetoed_reply('The sub-agent was not created', event)
            spec = DynamicAgentSpec(
                name=name, description=event.description, instructions=event.instructions, model=model
            )
            agent = (dynamic.factory or default_factory)(spec)
            # Without a model, it runs on the delegating run's: `()` allows no menu key either.
            created[name] = SubAgent(
                agent, name=name, description=event.description, models=() if model is None else [model]
            )
        finally:
            creating.discard(name)
            if not creating:
                del self._creating[run_key]
        return f'Created sub-agent {name!r}. Delegate tasks to it with `{self._tool_name}`.'

    async def _run_delegation(
        self,
        ctx: RunContext[AgentDepsT],
        agent_name: str,
        sub_agent: SubAgent[AgentDepsT],
        *,
        task: str,
        key: str | None,
        workspace: Workspace,
        record: DelegationTask | None = None,
        extra_capabilities: Sequence[AbstractCapability[Any]] = (),
    ) -> str:
        """Run one accepted delegation, announcing its start and how it ended."""
        # A delegation to the running agent already carries the parent's tools, so inheriting them
        # again would register every tool twice.
        is_self = self._include_self and agent_name == SELF_AGENT_NAME
        inherit_tools = self._inherit_tools and not is_self
        # Announced before the child coroutine exists, so an emit that does not return
        # (a cancellation landing on the await) leaves no never-awaited coroutine behind.
        owner = DelegationTasks.current() if record is not None else None
        emits = owner is not None or _emits_events(ctx)

        async def emit(event: DelegationStartEvent | DelegationEndEvent) -> None:
            if owner is not None and record is not None:
                event.task_id, event.parent_id = record.id, record.parent_id
                await _send_to_owner(owner, record, event)
            # Only called when `emits` holds or an owner exists, and an owner always comes with a record.
            elif emits:  # pragma: no branch
                await ctx.emit(event)

        if emits:
            text, truncated = bounded_text(task)
            await emit(
                DelegationStartEvent(
                    agent_name=agent_name, task=text, truncated=truncated, model=key, inherits_tools=inherit_tools
                )
            )
        started = time.perf_counter()

        toolsets = self._inherited_toolsets(ctx) if inherit_tools else None
        capabilities = list(self._shared_capabilities)
        if owner is not None and record is not None:
            capabilities.append(DelegationReports(owner, conversation_id=record.conversation_id, task_id=record.id))
            capabilities.extend(owner.persistence_capabilities())
        serving_model = ServingModelRecorder()
        capabilities.append(serving_model)
        capabilities.extend(extra_capabilities)
        usage_limits: UsageLimits | None
        if sub_agent.usage_limits is not None:
            # Isolated accounting so the per-child budget counts only this child.
            own_budget = True
            child_usage = RunUsage()
            usage_limits = sub_agent.usage_limits
            if owner is not None and self._forward_usage:
                child_usage = None
                usage_limits = _managed_limits(ctx.usage, parent=ctx.usage_limits, child=sub_agent.usage_limits)
        else:
            own_budget = False
            child_usage = None if self._forward_usage else RunUsage()
            usage_limits = ctx.usage_limits if owner is not None else None

        # A selected menu option decides the model and how it runs. Without one, a
        # sub-agent with no model of its own (e.g. one loaded from disk) inherits the
        # parent run's model, and one that brought its own keeps it. The running agent runs on
        # the parent run's model, which may be a run-level override of the agent's own.
        run_model: Model | KnownModelName | str | None
        settings: ModelSettings | None
        if key is not None:
            option = self._models[key]
            run_model = option.model
            settings = option.settings
        else:
            # `ctx.model` is an `AbstractModel`; only a request-response `Model` can drive a
            # sub-agent run. When the parent run uses something else (a realtime model), fall
            # back to `None` so the sub-agent uses its own default rather than being handed a
            # model it cannot run with.
            ctx_model = ctx.model
            run_model = (
                ctx_model
                if (is_self or sub_agent.agent.model is None) and _is_request_response_model(ctx_model)
                else None
            )
            settings = None
        run = sub_agent.agent.run(
            task,
            deps=ctx.deps,
            model=run_model,
            model_settings=settings,
            usage=ctx.usage if child_usage is None else child_usage,
            usage_limits=usage_limits,
            toolsets=toolsets,
            capabilities=capabilities,
            workspace=ReadOnlyWorkspace(workspace) if sub_agent.read_only else workspace,
            event_stream_handler=partial(self._stream_child, owner=owner, record=record)
            if record is not None
            else self._event_stream_handler,
            # A child stopped mid-tool-call (process exit, cancellation) saved calls without results;
            # the resumed run adds a new prompt, so close them out as interrupted first.
            message_history=repair_messages(record.messages) if record is not None else None,
            conversation_id=record.id if record is not None else None,
            run_id=record.run_id if record is not None else None,
        )
        token = _depth.set(_depth.get() + 1)
        try:
            ended = await self._settle(agent_name, sub_agent, run, own_budget=own_budget, serving_model=serving_model)
        except BaseException as exc:
            outcome: DelegationOutcome = (
                'cancelled' if isinstance(exc, (asyncio.CancelledError, RunCancelled)) else 'error'
            )
            if owner is not None:
                with anyio.move_on_after(5, shield=True):
                    await emit(
                        DelegationEndEvent(
                            agent_name=agent_name,
                            outcome=outcome,
                            output=str(exc),
                            truncated=False,
                            usage=child_usage,
                            duration_seconds=time.perf_counter() - started,
                        )
                    )
            raise
        finally:
            _depth.reset(token)
            if child_usage is not None and self._forward_usage:
                ctx.usage.incr(child_usage)
        if record is not None:
            record.outcome = ended.outcome
        if emits:
            text, truncated = bounded_text(ended.output)
            await emit(
                DelegationEndEvent(
                    agent_name=agent_name,
                    outcome=ended.outcome,
                    output=text,
                    truncated=truncated,
                    usage=child_usage,
                    duration_seconds=time.perf_counter() - started,
                )
            )
        if ended.cause is not None:
            raise ModelRetry(ended.output) from ended.cause
        return ended.output

    async def _stream_child(
        self,
        child_ctx: RunContext[AgentDepsT],
        events: AsyncIterable[AgentStreamEvent],
        *,
        owner: DelegationTasks | None,
        record: DelegationTask | None,
    ) -> None:
        async def observed() -> AsyncIterable[AgentStreamEvent]:
            async for event in events:
                # Installed only for a record, and a record exists only under the owner that `current` returns.
                if owner is not None and record is not None:  # pragma: no branch
                    record.messages = child_ctx.messages
                    # The model streaming this request, so an observer can show the child's context usage.
                    model = child_ctx.model
                    await owner.notify(record, event, model_name=model.model_name, context_window=model.context_window)
                    if self._forward_events:
                        progress = TaskProgressEvent(task_id=record.id, agent_name=record.agent_name, event=event)
                        await owner.publish(record, progress, observe=False)
                yield event

        if self._event_stream_handler is not None:
            await self._event_stream_handler(child_ctx, observed())
        else:
            async for _ in observed():
                pass

    async def _settle(
        self,
        agent_name: str,
        sub_agent: SubAgent[AgentDepsT],
        run: Coroutine[Any, Any, AgentRunResult[Any]],
        *,
        own_budget: bool,
        serving_model: ServingModelRecorder,
    ) -> _Ended:
        """Await the child and turn what happened into what the parent receives.

        Soft outcomes (timeout, own usage budget, a child run that ended with
        `DeferredToolRequests` or `PendingModelResponse`) become a steering message the parent reads as a normal
        tool result; a child's soft failure or contained crash becomes a `ModelRetry`
        the caller raises. Everything else propagates and aborts the delegation.
        """
        timeout = sub_agent.timeout_seconds
        try:
            result = await (asyncio.wait_for(run, timeout) if timeout is not None else run)
        except TimeoutError as exc:
            if timeout is None or isinstance(exc, HookTimeoutError):
                # The child itself timed out: a hook overran its own budget, or no
                # delegation budget is set at all. That is a child crash, so the
                # crash handlers decide what the parent sees.
                return self._crash_outcome(agent_name, sub_agent, exc)
            return _Ended(
                outcome='timeout',
                output=self._steer(
                    sub_agent.on_failure,
                    f'Sub-agent {agent_name!r} exceeded its {timeout}s time budget. '
                    f'Treat this as a recoverable observation and decide from existing evidence.',
                ),
            )
        except UsageLimitExceeded:
            if own_budget:
                return _Ended(
                    outcome='budget',
                    output=self._steer(
                        sub_agent.on_failure,
                        f'Sub-agent {agent_name!r} reached its usage budget. '
                        f'Treat this as a recoverable observation and decide from existing evidence.',
                    ),
                )
            # A shared/parent usage limit means the whole tree is out of budget.
            raise
        except (ModelRetry, UnexpectedModelBehavior) as exc:
            if sub_agent.on_failure is not None:
                return _Ended(outcome='failed', output=sub_agent.on_failure)
            # Soft sub-agent failures come back to the parent as a retry it can react to.
            return _Ended(outcome='failed', output=f'Sub-agent {agent_name!r} failed: {exc}', cause=exc)
        except _ALWAYS_PROPAGATE:
            raise
        except Exception as exc:
            return self._crash_outcome(agent_name, sub_agent, exc)
        if (reason := resumable_output_reason(result.output)) is not None:
            # The parent cannot resume the child run, and delegating the same task again would most
            # likely stop the same way, so this is a soft outcome rather than a retry. Nobody will check
            # on a request the child left pending, so it's cancelled.
            await cancel_pending_output(result.output, serving_model)
            return _Ended(
                outcome='failed',
                output=self._steer(
                    sub_agent.on_failure,
                    unresumable_output_message(agent_name, reason),
                ),
            )
        return _Ended(outcome='ok', output=str(result.output))

    def _crash_outcome(self, agent_name: str, sub_agent: SubAgent[AgentDepsT], exc: Exception) -> _Ended:
        """Contain an unexpected child crash, or let it abort the parent."""
        contain = sub_agent.contain_errors if sub_agent.contain_errors is not None else self._contain_errors
        if not contain:
            raise exc
        # Contain the crash so it cannot abort the parent, but keep it loud: the
        # exception rides the retry message and is logged, and `tool_retries`
        # bounds consecutive crashes into an abort.
        logger.warning('Contained crash from sub-agent %r', agent_name, exc_info=exc)
        return _Ended(
            outcome='contained',
            output=(
                f'Sub-agent {agent_name!r} crashed: {type(exc).__name__}: {exc}. '
                f'Treat this as a recoverable failure and decide from existing evidence.'
            ),
            cause=exc,
        )

    @staticmethod
    def _steer(on_failure: str | None, default: str) -> str:
        """A soft steering message: the delegate's `on_failure` override, else `default`."""
        if on_failure is not None:
            return on_failure
        return default


def _managed_limits(usage: RunUsage, *, parent: UsageLimits | None, child: UsageLimits) -> UsageLimits:
    """Cap shared accounting by both budgets, including concurrent sibling spend.

    Managed children never hide usage in a private counter until completion. A child's
    relative budget becomes an absolute ceiling at launch; concurrent spend can reach it
    earlier, but cannot bypass the parent's ceiling.
    """
    parent = parent or UsageLimits(request_limit=None)

    def limit(parent_limit: int | None, child_limit: int | None, baseline: int = 0) -> int | None:
        if child_limit is None:
            return parent_limit
        absolute = baseline + child_limit
        return absolute if parent_limit is None else min(parent_limit, absolute)

    cost = parent.cost_limit
    if child.cost_limit is not None:
        absolute_cost = (usage.cost or 0) + child.cost_limit
        cost = absolute_cost if cost is None else min(cost, absolute_cost)
    return UsageLimits(
        request_limit=limit(parent.request_limit, child.request_limit, usage.requests),
        tool_calls_limit=limit(parent.tool_calls_limit, child.tool_calls_limit, usage.tool_calls),
        input_tokens_limit=limit(parent.input_tokens_limit, child.input_tokens_limit, usage.input_tokens),
        output_tokens_limit=limit(parent.output_tokens_limit, child.output_tokens_limit, usage.output_tokens),
        total_tokens_limit=limit(parent.total_tokens_limit, child.total_tokens_limit, usage.total_tokens),
        per_request_input_tokens_limit=limit(
            parent.per_request_input_tokens_limit, child.per_request_input_tokens_limit
        ),
        cost_limit=cost,
        count_tokens_before_request=parent.count_tokens_before_request or child.count_tokens_before_request,
    )
