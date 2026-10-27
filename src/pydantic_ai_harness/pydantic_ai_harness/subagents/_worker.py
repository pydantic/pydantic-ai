"""The worker side of a remote task: runs the sub-agent and talks to the delegating run over a transport."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic

import anyio
import anyio.abc
from opentelemetry import context as otel_context, propagate

from pydantic_ai import AgentStreamEvent, CancellationToken, RunContext
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import WrapRunHandler
from pydantic_ai.exceptions import RunCancelled, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness._sub_agent_output import (
    ServingModelRecorder,
    cancel_pending_output,
    resumable_output_reason,
    unresumable_output_message,
)
from pydantic_ai_harness.subagents._control import TaskLink, task_group, validate_agent_name
from pydantic_ai_harness.subagents._events import (
    TaskAnswerEvent,
    TaskCancelledEvent,
    TaskCancelRequestEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskFollowUpEvent,
    TaskForceCancelEvent,
    TaskMessageEvent,
    TaskProgressEvent,
    TaskQuestionEvent,
    TaskRequestEvent,
    TaskRetryEvent,
    TaskSettledEvent,
    TaskStartedEvent,
    agent_mailbox,
    task_mailbox,
)
from pydantic_ai_harness.subagents._tasks import CANCELLED_ANSWER, enqueue_while_open
from pydantic_ai_harness.subagents._toolset import delegation_level
from pydantic_ai_harness.subagents._transport import SubAgentTransport

_STOP_REPORT_SECONDS = 5.0
"""How long a stopping worker tries to report each task it was running."""

_logger = logging.getLogger(__name__)


def _steer(ctx: RunContext[Any], message: str) -> None:
    """Queue the delegating agent's message on the task's run, as it was sent.

    A run that has read its last queued messages drops it, as a local task does.
    """
    enqueue_while_open(ctx, UserPromptPart(message), priority='asap')


class _WorkerLink(TaskLink):
    """Joins a served sub-agent's run to its delegating run, over the transport."""

    def __init__(self, *, start: TaskRequestEvent, transport: SubAgentTransport) -> None:
        super().__init__(can_ask=start.can_ask_questions)
        self.start = start
        self.transport = transport
        self.token = CancellationToken()
        self.forced = False
        self._ctx: RunContext[Any] | None = None
        self._stop_requested = False
        self._pending_messages: list[str] = []
        self._answers: dict[str, tuple[anyio.Event, list[str]]] = {}

    def stopping(self) -> bool:
        return self._stop_requested

    async def ask(self, question_id: str, question: str) -> str:
        answered = anyio.Event()
        answer: list[str] = []
        self._answers[question_id] = (answered, answer)
        await self.transport.send(
            TaskQuestionEvent(
                task_id=self.start.task_id, agent_name=self.start.agent_name, question_id=question_id, question=question
            )
        )
        await answered.wait()
        return answer[0]

    async def wrap_run(self, ctx: RunContext[Any], *, handler: WrapRunHandler) -> Any:
        self._ctx = ctx
        for message in self._pending_messages:
            _steer(ctx, message)
        self._pending_messages.clear()
        try:
            return await handler()
        finally:
            self._ctx = None

    async def receive(self) -> None:
        """Handle the messages the delegating run sends the running task, until cancelled."""
        mailbox = task_mailbox(self.start.task_id, 'worker')
        while True:
            message = await self.transport.receive(mailbox)
            if isinstance(message, TaskMessageEvent):
                if self._ctx is None:
                    self._pending_messages.append(message.message)
                else:
                    _steer(self._ctx, message.message)
            elif isinstance(message, TaskAnswerEvent):
                if (waiter := self._answers.pop(message.question_id, None)) is not None:
                    answered, answer = waiter
                    answer.append(message.answer)
                    answered.set()
            elif isinstance(message, TaskCancelRequestEvent):
                self._stop_requested = True
                if self._ctx is None:
                    self.token.cancel()
                # A task waiting for an answer makes no model request to stop before, so stop waiting.
                for answered, answer in self._answers.values():
                    answer.append(CANCELLED_ANSWER)
                    answered.set()
                self._answers.clear()
            else:
                assert isinstance(message, TaskForceCancelEvent), f'Unexpected message {message.kind!r}.'
                self.forced = True
                self.token.cancel()


class _ForwardingWorkerLink(_WorkerLink):
    """A worker link that also forwards every event of the sub-agent's run to the delegating run."""

    async def on_event(self, ctx: RunContext[Any], *, event: AgentStreamEvent) -> None:
        await self.transport.send(
            TaskProgressEvent(task_id=self.start.task_id, agent_name=self.start.agent_name, event=event)
        )


@dataclass(kw_only=True)
class SubAgentWorker(Generic[AgentDepsT]):
    """Serves sub-agents to delegating agents in other processes, over a transport.

    Run `serve` in the process that hosts the agents. The delegating agent's `SubAgents` lists them as
    `RemoteSubAgent`s with the same names, and reaches them through the same transport.

    The worker keeps the conversations of tasks, up to `max_conversations`, so a delegating run can
    resume them.
    """

    agents: Mapping[str, AbstractAgent[AgentDepsT, object]]
    """The agents this worker serves, by the name delegating agents reach them by.

    Run controls such as usage limits or timeouts belong to the agent, or to the delegating run's
    `SubAgents`, which applies its own to the task.
    """

    transport: SubAgentTransport
    """The transport the delegating agents send tasks over."""

    deps: AgentDepsT
    """The deps every task's run gets. Deps never cross the transport."""

    retries: int = 0
    """How many times to run a task again when its run raises an error, starting over from its prompt.

    Cancellation, usage limits, and `UserError` are not retried.
    """

    max_conversations: int = 1000
    """How many task conversations to keep for resuming; the least recently used is forgotten first."""

    _histories: dict[str, list[ModelMessage]] = field(
        default_factory=dict[str, list[ModelMessage]], init=False, repr=False
    )

    def __post_init__(self) -> None:
        for name in self.agents:
            if (problem := validate_agent_name(name)) is not None:
                raise ValueError(problem)
        if self.retries < 0:
            raise ValueError(f'`retries` must be zero or more, got {self.retries}.')
        if self.max_conversations < 0:
            raise ValueError(f'`max_conversations` must be zero or more, got {self.max_conversations}.')

    async def serve(self) -> None:
        """Receive and run tasks for every agent, until cancelled."""
        async with anyio.create_task_group() as group:
            for name, agent in self.agents.items():
                group.start_soon(self._serve_agent, group, name, agent)

    async def _serve_agent(
        self, group: anyio.abc.TaskGroup, name: str, agent: AbstractAgent[AgentDepsT, object]
    ) -> None:
        while True:
            start = await self.transport.receive(agent_mailbox(name))
            if isinstance(start, TaskRequestEvent):
                group.start_soon(self._run, agent, start)
            else:
                _logger.warning(
                    'Sub-agent worker for %r skipped a %r message, which does not start a task.', name, start.kind
                )

    async def _run(self, agent: AbstractAgent[AgentDepsT, object], start: TaskRequestEvent) -> None:
        context_token = otel_context.attach(propagate.extract(start.trace_context))
        try:
            await self._run_task(agent, start)
        except Exception as e:
            # One task's failure, from a misconfigured sub-agent or the transport, doesn't stop the worker,
            # which serves other tasks too. The delegating run is told if the transport still can.
            if isinstance(e, UserError):
                error = str(e)
            else:
                _logger.exception('Sub-agent task %s failed on its worker.', start.task_id)
                error = f'{type(e).__name__}: {e}'
            try:
                await self.transport.send(
                    TaskFailedEvent(task_id=start.task_id, agent_name=start.agent_name, error=error)
                )
            except Exception:
                _logger.exception('Could not report the failure of sub-agent task %s.', start.task_id)
        finally:
            otel_context.detach(context_token)
            while len(self._histories) > self.max_conversations:
                del self._histories[next(iter(self._histories))]

    async def _run_task(self, agent: AbstractAgent[AgentDepsT, object], start: TaskRequestEvent) -> None:
        link = (_ForwardingWorkerLink if start.forward_events else _WorkerLink)(start=start, transport=self.transport)
        settled: TaskSettledEvent | None = None
        try:
            try:
                await self.transport.send(TaskStartedEvent(task_id=start.task_id, agent_name=start.agent_name))
                async with task_group() as group:
                    group.start_soon(link.receive)
                    try:
                        with delegation_level(start.depth):
                            settled = await self._attempts(agent, start, link)
                    finally:
                        group.cancel_scope.cancel()
            except anyio.get_cancelled_exc_class():
                # The worker is stopping. Its delegating run would otherwise wait for this task forever, so it
                # gets the task's report if it settled, or else a failure.
                report = settled or TaskFailedEvent(
                    task_id=start.task_id,
                    agent_name=start.agent_name,
                    error='The worker serving this task stopped before it finished.',
                )
                with anyio.CancelScope(shield=True), anyio.move_on_after(_STOP_REPORT_SECONDS) as deadline:
                    try:
                        await self.transport.send(report)
                    except Exception:
                        _logger.exception('Could not report sub-agent task %s as its worker stopped.', start.task_id)
                if deadline.cancelled_caught:
                    _logger.error('Timed out reporting sub-agent task %s as its worker stopped.', start.task_id)
                raise
            assert settled is not None, 'the attempts returned, so the task settled'
            # Sent even if the worker stops meanwhile, which then waits for it, so the report is neither lost
            # nor followed by a failure. Bounded, so a transport that blocks can't keep the worker from stopping.
            with anyio.move_on_after(_STOP_REPORT_SECONDS, shield=True) as deadline:
                await self.transport.send(settled)
            if deadline.cancelled_caught:
                _logger.error('Timed out sending the report of sub-agent task %s.', start.task_id)
        finally:
            with anyio.move_on_after(_STOP_REPORT_SECONDS, shield=True) as deadline:
                try:
                    await self.transport.release(task_mailbox(start.task_id, 'worker'))
                except Exception:
                    # The task was reported, or is reported as failed: this is no failure of the task.
                    _logger.exception('Could not release the mailbox of sub-agent task %s.', start.task_id)
            if deadline.cancelled_caught:
                _logger.error('Timed out releasing the mailbox of sub-agent task %s.', start.task_id)

    async def _attempts(
        self, agent: AbstractAgent[AgentDepsT, object], start: TaskRequestEvent, link: _WorkerLink
    ) -> TaskSettledEvent:
        task_id, agent_name = start.task_id, start.agent_name
        history: list[ModelMessage] | None = None
        if isinstance(start, TaskFollowUpEvent):
            history = self._histories.get(task_id)
            if history is None:
                return TaskFailedEvent(
                    task_id=task_id, agent_name=agent_name, error='This worker has no conversation to resume.'
                )
        # Every attempt adds to one usage, so the usage limits and the reported usage cover them all.
        usage = RunUsage()
        attempt = 0
        while True:
            attempt += 1
            serving_model = ServingModelRecorder()
            try:
                result = await agent.run(
                    start.prompt,
                    message_history=history,
                    deps=self.deps,
                    usage=usage,
                    usage_limits=start.usage_limits,
                    capabilities=[link, serving_model],
                    cancellation_token=link.token,
                )
            except RunCancelled as e:
                self._keep(task_id, e.all_messages())
                return TaskCancelledEvent(task_id=task_id, agent_name=agent_name, forced=link.forced, usage=usage)
            except UserError:
                raise
            except UsageLimitExceeded as e:
                return TaskFailedEvent(task_id=task_id, agent_name=agent_name, error=str(e), usage=usage)
            except Exception as e:
                error = f'{type(e).__name__}: {e}'
                if attempt > self.retries:
                    return TaskFailedEvent(task_id=task_id, agent_name=agent_name, error=error, usage=usage)
                await self.transport.send(
                    TaskRetryEvent(task_id=task_id, agent_name=agent_name, attempt=attempt + 1, error=error)
                )
                continue
            self._keep(task_id, result.all_messages())
            if (reason := resumable_output_reason(result.output)) is not None:
                # The delegating run can't check on a request the run left pending, so it's cancelled.
                await cancel_pending_output(result.output, serving_model)
                return TaskFailedEvent(
                    task_id=task_id,
                    agent_name=agent_name,
                    error=unresumable_output_message(agent_name, reason),
                    usage=usage,
                )
            return TaskCompletedEvent(task_id=task_id, agent_name=agent_name, output=str(result.output), usage=usage)

    def _keep(self, task_id: str, messages: list[ModelMessage]) -> None:
        # Re-inserted, so the dict stays ordered from least to most recently used.
        self._histories.pop(task_id, None)
        self._histories[task_id] = messages
