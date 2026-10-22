"""Task control for `SubAgents(task_control=True)`: the child side of a task, remote tasks, and created agents.

The design of task control -- typed task messages that listeners can rewrite or veto, transports and
workers for agents in other processes, and agents the model creates -- is a port of Vstorm's
[subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai) (MIT licensed, copyright
(c) 2026 Vstorm), rewritten on `DelegationTasks`.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias
from uuid import uuid4

import anyio
import anyio.abc
from opentelemetry import propagate
from opentelemetry.trace import Tracer

from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.subagents._events import (
    AgentCreatedEvent,
    ParentMessage,
    TaskAnswerEvent,
    TaskBackgroundedEvent,
    TaskCancelledEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskForceCancelEvent,
    TaskProgressEvent,
    TaskQuestionEvent,
    TaskRequestEvent,
    TaskSettledEvent,
    TaskStartedEvent,
)
from pydantic_ai_harness.subagents._tasks import CANCELLED_ANSWER, DelegationTask, DelegationTasks

if TYPE_CHECKING:
    from pydantic_ai_harness.subagents._transport import SubAgentTransport

_logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r'[A-Za-z0-9_-]{1,64}')

STEERING_PREFIX = 'Message from the agent that delegated this task to you:'
"""Starts a steering message the delegating model sent with `message_task`."""

STOPPED_OUTPUT = 'Task stopped; no successful result is available.'

RUN_OWNER: ContextVar[tuple[str, DelegationTasks] | None] = ContextVar(
    'pydantic_ai_harness.subagents.run_owner', default=None
)
"""The run ID and owner of the run that opened a `DelegationTasks` of its own, for its task-control `SubAgents`."""


@dataclass(frozen=True, kw_only=True)
class TaskTelemetry:
    """Records task control's own decisions as zero-duration spans on the delegating run's tracer.

    Core's spans show the tool calls; these show what a listener or a limit decided about them. Text
    that can quote a prompt, such as a veto reason, a question, or an answer, is attached only when the
    run's instrumentation includes content.
    """

    tracer: Tracer
    include_content: bool

    @classmethod
    def of(cls, ctx: RunContext[AgentDepsT]) -> TaskTelemetry:
        return cls(tracer=ctx.tracer, include_content=ctx.trace_include_content)

    def decision(
        self,
        name: str,
        *,
        agent_name: str | None,
        task_id: str | None = None,
        attributes: Mapping[str, str | float] | None = None,
        content: Mapping[str, str] | None = None,
    ) -> None:
        """Record the span `sub_agents {name}`, with `sub_agents.`-prefixed attributes."""
        span = self.tracer.start_span(f'sub_agents {name}')
        if span.is_recording():
            if agent_name is not None:
                span.set_attribute('sub_agents.agent_name', agent_name)
            if task_id is not None:
                span.set_attribute('sub_agents.task_id', task_id)
            for key, value in (attributes or {}).items():
                span.set_attribute(f'sub_agents.{key}', value)
            if self.include_content:
                for key, value in (content or {}).items():
                    span.set_attribute(f'sub_agents.{key}', value)
        span.end()

    def vetoed(self, event: ParentMessage | AgentCreatedEvent) -> None:
        """Record that a listener vetoed `event`."""
        reason = {'veto_reason': event.veto_reason} if event.veto_reason is not None else None
        if isinstance(event, AgentCreatedEvent):
            # The model chose this name, so it is content.
            content = {'agent_name': event.name, **(reason or {})}
            self.decision('vetoed', agent_name=None, attributes={'event': event.kind}, content=content)
        else:
            self.decision(
                'vetoed',
                agent_name=event.agent_name,
                task_id=event.task_id,
                attributes={'event': event.kind},
                content=reason,
            )

    def backgrounded(self, event: TaskBackgroundedEvent) -> None:
        """Record that a task moved to the background, and why."""
        self.decision(
            'task backgrounded', agent_name=event.agent_name, task_id=event.task_id, attributes={'reason': event.reason}
        )


def run_owned(owner: DelegationTasks) -> bool:
    """Whether `owner` was opened by a run for itself, rather than bound by the application."""
    entry = RUN_OWNER.get()
    return entry is not None and entry[1] is owner


def validate_agent_name(name: str) -> str | None:
    """Why `name` can't name a remote or created sub-agent, or `None` if it can."""
    if _NAME_RE.fullmatch(name):
        return None
    return f'Sub-agent name {name!r} must be 1 to 64 letters, digits, `_` or `-`.'


@asynccontextmanager
async def task_group() -> AsyncGenerator[anyio.abc.TaskGroup]:
    """An anyio task group that raises a single error as is, rather than in an `ExceptionGroup`."""
    try:
        async with anyio.create_task_group() as group:
            yield group
    except BaseExceptionGroup as errors:
        if len(errors.exceptions) == 1 and not isinstance(errors.exceptions[0], BaseExceptionGroup):
            raise errors.exceptions[0] from None
        raise  # pragma: no cover - two of the group's tasks failing at once


@dataclass(frozen=True, kw_only=True)
class RemoteSubAgent:
    """A sub-agent that a `SubAgentWorker` in another process serves, reached through `SubAgents.transport`.

    The worker runs the agent with its own model, tools and deps; this side only knows its name and
    what it is for.
    """

    name: str
    """The name the model delegates to, and the worker serves it under. 1 to 64 letters, digits, `_` or `-`."""

    description: str | None = None
    """What the agent is good at, for the system-prompt listing."""

    resumable: bool = True
    """Whether a finished task can be resumed, continuing the conversation the worker keeps."""

    can_ask_questions: bool = True
    """Whether the agent gets an `ask_parent` tool to ask the delegating agent a question."""

    def __post_init__(self) -> None:
        if (problem := validate_agent_name(self.name)) is not None:
            raise ValueError(problem)


@dataclass(kw_only=True)
class DynamicAgentSpec:
    """What the model asked for when it called `create_agent`."""

    name: str
    """The new sub-agent's name."""

    description: str
    """What the sub-agent is for."""

    instructions: str
    """The sub-agent's instructions, written by the model."""

    model: str | None
    """The `SubAgents.models` key the sub-agent runs on, or `None` for the delegating run's model."""


DynamicAgentFactory: TypeAlias = 'Callable[[DynamicAgentSpec], AbstractAgent[AgentDepsT, Any]]'
"""Builds the agent for a sub-agent the model created with `create_agent`."""


@dataclass(kw_only=True)
class DynamicAgents(Generic[AgentDepsT]):
    """Lets the model create sub-agents during a run, with a `create_agent` tool.

    A created sub-agent is delegated to like any other, by the run that created it, until that run ends.
    """

    models: Sequence[str] = ()
    """The `SubAgents.models` keys the model may run a created sub-agent on; the first is the default.
    Empty: created sub-agents run on the delegating run's model."""

    max_agents: int = 5
    """How many sub-agents one run may create."""

    factory: DynamicAgentFactory[AgentDepsT] | None = None
    """Builds the agent. The default is an `Agent` with the requested instructions and no tools, on the
    model `SubAgents` routes the delegation to."""


def default_factory(spec: DynamicAgentSpec) -> AbstractAgent[object, str]:
    """An agent with the requested instructions and no tools or model of its own."""
    return Agent(deps_type=object, name=spec.name, description=spec.description, instructions=spec.instructions)


class TaskLink(AbstractCapability[Any], ABC):
    """Joins a sub-agent's run to its delegating run: an `ask_parent` tool, and stopping when asked to."""

    def __init__(self, *, can_ask: bool) -> None:
        self.can_ask = can_ask
        self._asking = anyio.Lock()

    def get_toolset(self) -> AgentToolset[Any] | None:
        if not self.can_ask:
            return None
        toolset = FunctionToolset[Any]()
        toolset.add_function(self.ask_parent, takes_ctx=False, name='ask_parent')
        return toolset

    async def ask_parent(self, question: str) -> str:
        """Ask the agent that delegated this task to you a question, and wait for its answer.

        Only ask when you can't reasonably continue without the answer.

        Args:
            question: The question, with the context the other agent needs to answer it.
        """
        # The delegating side tracks one question per task, so questions take turns.
        async with self._asking:
            if self.stopping():
                return CANCELLED_ANSWER
            return await self.ask(uuid4().hex[:12], question)

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if self.stopping():
            ctx.cancel()
            # The cancellation reaches the task driving the run at its next checkpoint; wait for it here,
            # so the request is never sent.
            await anyio.sleep_forever()
        return request_context

    @abstractmethod
    def stopping(self) -> bool:
        """Whether the delegating run asked the task to stop."""
        raise NotImplementedError

    @abstractmethod
    async def ask(self, question_id: str, question: str) -> str:
        """Put `question` to the delegating run and return its answer."""
        raise NotImplementedError


class LocalTaskLink(TaskLink):
    """The link of a task running in the delegating run's process."""

    def __init__(
        self, *, owner: DelegationTasks, record: DelegationTask, can_ask: bool, telemetry: TaskTelemetry
    ) -> None:
        super().__init__(can_ask=can_ask)
        self.owner = owner
        self.record = record
        self.telemetry = telemetry

    def stopping(self) -> bool:
        return self.owner.stop_requested(self.record.id)

    async def ask(self, question_id: str, question: str) -> str:
        record = self.record
        event = TaskQuestionEvent(
            task_id=record.id, agent_name=record.agent_name, question_id=question_id, question=question
        )
        answer = await put_question(self.owner, record, event, self.telemetry)
        return answer.answer


async def put_question(
    owner: DelegationTasks, record: DelegationTask, event: TaskQuestionEvent, telemetry: TaskTelemetry
) -> TaskAnswerEvent:
    """Get a task's question answered: by a listener of the delegating run, or else by its model."""
    event = await owner.publish(record, event)
    if event.answer is not None:
        answer = await owner.publish(
            record,
            TaskAnswerEvent(
                task_id=record.id, agent_name=record.agent_name, question_id=event.question_id, answer=event.answer
            ),
        )
        if not answer.vetoed:
            telemetry.decision(
                'question answered',
                agent_name=record.agent_name,
                task_id=record.id,
                content={'question': event.question, 'answer': answer.answer},
            )
            return answer
        telemetry.vetoed(answer)
    if record.status == 'running' and not record.background:
        # The delegating model can only answer once the delegation tool call waiting for the task returns.
        backgrounded = TaskBackgroundedEvent(task_id=record.id, agent_name=record.agent_name, reason='question')
        await owner.publish(record, backgrounded)
        telemetry.backgrounded(backgrounded)
        owner.background(record.id)
    text = await owner.ask(record.id, question_id=event.question_id, question=event.question)
    return TaskAnswerEvent(task_id=record.id, agent_name=record.agent_name, question_id=event.question_id, answer=text)


async def run_remote(
    *,
    owner: DelegationTasks,
    record: DelegationTask,
    start: TaskRequestEvent,
    transport: SubAgentTransport,
    accept_timeout: float | None,
    usage: RunUsage | None,
    telemetry: TaskTelemetry,
) -> str:
    """Send a task to the worker serving its agent, and relay what it sends back until it settles."""
    mailbox = f'task:{record.id}:parent'
    settled: TaskSettledEvent | None = None
    try:
        try:
            settled = await _relay(owner, record, start, transport, mailbox, accept_timeout, telemetry)
        except asyncio.CancelledError:
            forced = not owner.stop_requested(record.id)
            with anyio.move_on_after(5, shield=True):
                await owner.publish(
                    record, TaskCancelledEvent(task_id=record.id, agent_name=record.agent_name, forced=forced)
                )
            raise
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            await owner.publish(record, TaskFailedEvent(task_id=record.id, agent_name=record.agent_name, error=error))
            raise
        if settled is None:
            assert accept_timeout is not None, 'only a timeout ends the wait for a worker'
            telemetry.decision(
                'task not accepted',
                agent_name=record.agent_name,
                task_id=record.id,
                attributes={'accept_timeout': accept_timeout},
            )
            error = (
                f'No worker picked up the task within {accept_timeout}s, so sub-agent {record.agent_name!r} '
                'is unavailable. Do the task another way.'
            )
            await owner.publish(record, TaskFailedEvent(task_id=record.id, agent_name=record.agent_name, error=error))
            record.outcome = 'failed'
            return error
        return await _settle_remote(owner, record, settled, usage)
    finally:
        # What ended the task matters more to the delegating run than a failed cleanup, so cleanup errors are logged.
        with anyio.CancelScope(shield=True):
            if settled is None:
                # The task is ending without its worker's report, so stop it rather than leave it running there.
                # This is teardown, not a decision listeners could veto.
                telemetry.decision('remote task stopped', agent_name=record.agent_name, task_id=record.id)
                try:
                    await transport.send(TaskForceCancelEvent(task_id=record.id, agent_name=record.agent_name))
                except Exception:
                    _logger.exception('Could not stop remote sub-agent task %s on its worker.', record.id)
            try:
                await transport.release(mailbox)
            except Exception:
                _logger.exception('Could not release the mailbox of remote sub-agent task %s.', record.id)


async def _relay(
    owner: DelegationTasks,
    record: DelegationTask,
    start: TaskRequestEvent,
    transport: SubAgentTransport,
    mailbox: str,
    accept_timeout: float | None,
    telemetry: TaskTelemetry,
) -> TaskSettledEvent | None:
    """Send the task, then pass on what its worker sends until its report, or `None` if no worker picks it up."""
    propagate.inject(start.trace_context)
    await transport.send(start)
    try:
        with anyio.fail_after(accept_timeout):
            message = await transport.receive(mailbox)
    except TimeoutError:
        return None
    async with task_group() as group:
        while not isinstance(message, TaskSettledEvent):
            if isinstance(message, TaskQuestionEvent):
                group.start_soon(_answer_remote, owner, record, message, transport, telemetry)
            elif isinstance(message, TaskProgressEvent):
                await owner.publish(record, message, observe=False)
                await owner.notify(record, message.event)
            else:
                await owner.publish(record, message)
            message = await transport.receive(mailbox)
        group.cancel_scope.cancel()
    return message


async def _answer_remote(
    owner: DelegationTasks,
    record: DelegationTask,
    question: TaskQuestionEvent,
    transport: SubAgentTransport,
    telemetry: TaskTelemetry,
) -> None:
    await transport.send(await put_question(owner, record, question, telemetry))


async def _settle_remote(
    owner: DelegationTasks, record: DelegationTask, message: TaskSettledEvent, usage: RunUsage | None
) -> str:
    if message.usage is not None and usage is not None:
        # usage-attribution: a remote sub-agent's run reports its usage on its worker's own span
        usage.incr(message.usage)
    message = await owner.publish(record, message)
    if isinstance(message, TaskCompletedEvent):
        return message.output
    if isinstance(message, TaskCancelledEvent):
        record.outcome = 'cancelled'
        return STOPPED_OUTPUT
    assert isinstance(message, TaskFailedEvent), f'Unexpected report {message.kind!r}.'
    record.outcome = 'failed'
    return message.error


def started(record: DelegationTask) -> TaskStartedEvent:
    return TaskStartedEvent(task_id=record.id, agent_name=record.agent_name)
