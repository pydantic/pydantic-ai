"""Live ownership of a conversation across multiple agent runs."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, overload

import anyio
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import TypedDict, Unpack

from . import _instructions, _utils, messages, models, result, usage as _usage
from ._cancel import CancellationToken
from ._enqueue import EnqueueContent, PendingMessage, PendingMessagePriority
from ._session import SessionRuntime, bind_session
from .agent.abstract import (
    AbstractAgent,
    AgentMetadata,
    AgentModelSettings,
    AgentRetries,
    AgentRunEvents,
    EventStreamHandler,
    RunOutputDataT,
)
from .agent.wrapper import WrapperAgent
from .capabilities import AgentCapability
from .conversation import Conversation
from .exceptions import UserError
from .output import OutputDataT, OutputSpec
from .run import AgentRun, AgentRunResult
from .tools import AgentDepsT, DeferredToolResults
from .toolsets import AbstractToolset
from .workspaces import WorkspaceBackend, WorkspaceRef

if TYPE_CHECKING:
    from .agent.spec import AgentSpec

__all__ = ('AgentSession', 'SessionState', 'SessionStateTypeAdapter')


@dataclass(kw_only=True)
class SessionState:
    """Portable session checkpoint; it never contains connections, tasks, or dependencies.

    A checkpoint captured during a run is marked with `active_run_id`. Opening it directly is
    rejected: a partial history is not proof that retrying an external effect is safe.
    """

    __pydantic_config__ = ConfigDict(defer_build=True)

    conversation: Conversation = field(default_factory=Conversation)
    """Messages, accumulated usage, conversation identity, and deferred tool requests."""
    pending: list[PendingMessage] = field(default_factory=list[PendingMessage])
    """Input not yet delivered into a run's history, preserving enqueue identity and priority."""
    active_run_id: str | None = None
    """Unfinished work at capture time. This is not an instruction to retry that work."""


SessionStateTypeAdapter = TypeAdapter(SessionState)
"""Adapter for serializing and validating a session checkpoint."""


class _RunOptions(TypedDict, Generic[AgentDepsT], total=False):
    """Shared forwarding options; dependency defaults are resolved by the session, not the agent."""

    conversation: Conversation | None
    message_history: Sequence[messages.ModelMessage] | None
    deferred_tool_results: DeferredToolResults | None
    conversation_id: str | None
    run_id: str | None
    model: models.Model | models.KnownModelName | str | None
    model_settings: AgentModelSettings[AgentDepsT] | None
    usage_limits: _usage.UsageLimits | None
    cancellation_token: CancellationToken | None
    usage: _usage.RunUsage | None
    metadata: AgentMetadata[AgentDepsT] | None
    retries: int | AgentRetries | None
    infer_name: bool
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None
    capabilities: Sequence[AgentCapability[AgentDepsT]] | None
    workspace: WorkspaceBackend | WorkspaceRef | Literal['new'] | None
    spec: dict[str, Any] | AgentSpec | None


class _InstructedRunOptions(_RunOptions[AgentDepsT], total=False):
    instructions: _instructions.AgentInstructions[AgentDepsT]


class AgentSession(WrapperAgent[AgentDepsT, OutputDataT]):
    """A conversation's live owner, containing sequential runs and their shared resources.

    Create with [`Agent.session`][pydantic_ai.agent.Agent.session] and enter with `async with`.
    The usual `run`, `iter`, `run_stream`, and `run_stream_events` interfaces retain their run
    semantics. History, usage, and undelivered input belong to this session instead of the caller.
    """

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        conversation: Conversation | None = None,
        state: SessionState | None = None,
        deps: AgentDepsT = None,
        model: models.Model | models.KnownModelName | str | None = None,
    ) -> None:
        super().__init__(agent)
        if state is not None and conversation is not None:
            raise UserError('Pass either `state` or `conversation`, not both.')
        if state is not None and state.active_run_id is not None:
            raise UserError('The session checkpoint contains an unfinished run; reconcile it before resuming.')
        self._runtime = SessionRuntime(
            state.conversation if state is not None else conversation or Conversation(),
            state.pending if state is not None else None,
        )
        self._deps = deps
        self._model = model
        self._stack = AsyncExitStack()
        self._entered = False
        self._used = False
        self._run_scope: anyio.CancelScope | None = None
        self._run_finished: anyio.Event | None = None

    async def __aenter__(self) -> Self:
        if self._used:
            raise UserError('An agent session may only be entered once; create another from its `state`.')
        self._used = True
        group = await self._stack.enter_async_context(anyio.create_task_group())
        self._runtime.resources.bind_group(group)
        self._entered = True
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        self._entered = False
        self._runtime.close()
        # Runs started by a streaming handle or another task must finish teardown before the
        # resources they use close. The run task, not this caller, exits its cancel scope.
        if self._run_scope is not None:
            self._run_scope.cancel()
            assert self._run_finished is not None
            with anyio.CancelScope(shield=True):
                await self._run_finished.wait()
        self._runtime.resources.close()
        try:
            return await self._stack.__aexit__(*args)
        except BaseExceptionGroup as group:
            if len(group.exceptions) == 1:
                raise group.exceptions[0] from None
            raise

    @property
    def conversation(self) -> Conversation:
        """A detached snapshot of the conversation, including the current run's history."""
        return self._runtime.snapshot()[0]

    @property
    def state(self) -> SessionState:
        """A detached checkpoint of conversation, pending input, and unfinished work."""
        conversation, pending, active_run_id = self._runtime.snapshot()
        return SessionState(conversation=conversation, pending=pending, active_run_id=active_run_id)

    def enqueue(self, *content: EnqueueContent, priority: PendingMessagePriority = 'asap') -> str | None:
        """Submit input to the active run or retain it for the next run while idle.

        This does not start a run. As with `AgentRun.enqueue`, delivery is at a request boundary,
        not native mid-response steering. An empty call is a no-op.
        """
        pending = PendingMessage.from_content(*content, priority=priority)
        if pending is None:
            return None
        self._runtime.enqueue(pending)
        return pending.enqueue_id

    def cancel(self) -> None:
        """Cancel the current run without closing the session. Does nothing while idle.

        Cancellation uses the same controller and teardown semantics as `AgentRun.cancel()`.
        It does not undo completed tools or imply that generated audio was played.
        """
        self._runtime.cancel()

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AgentRunResult[Any]:
        """Use the session dependencies unless this run explicitly overrides them."""
        return await super().run(
            user_prompt,
            output_type=output_type,
            deps=deps if _utils.is_set(deps) else self._deps,
            event_stream_handler=event_stream_handler,
            **kwargs,
        )

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AgentRunResult[Any]:
        """Use the session dependencies unless this run explicitly overrides them."""
        return super().run_sync(
            user_prompt,
            output_type=output_type,
            deps=deps if _utils.is_set(deps) else self._deps,
            event_stream_handler=event_stream_handler,
            **kwargs,
        )

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    def run_stream(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, Any]]:
        """Use the session dependencies unless this run explicitly overrides them."""
        return super().run_stream(
            user_prompt,
            output_type=output_type,
            deps=deps if _utils.is_set(deps) else self._deps,
            event_stream_handler=event_stream_handler,
            **kwargs,
        )

    @overload
    def run_stream_sync(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_RunOptions[AgentDepsT]],
    ) -> result.StreamedRunResultSync[AgentDepsT, OutputDataT]: ...

    @overload
    def run_stream_sync(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_RunOptions[AgentDepsT]],
    ) -> result.StreamedRunResultSync[AgentDepsT, RunOutputDataT]: ...

    def run_stream_sync(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_RunOptions[AgentDepsT]],
    ) -> result.StreamedRunResultSync[AgentDepsT, Any]:
        """Use the session dependencies unless this run explicitly overrides them."""
        return super().run_stream_sync(
            user_prompt,
            output_type=output_type,
            deps=deps if _utils.is_set(deps) else self._deps,
            event_stream_handler=event_stream_handler,
            **kwargs,
        )

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AbstractAsyncContextManager[AgentRunEvents[OutputDataT]]: ...

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AbstractAsyncContextManager[AgentRunEvents[RunOutputDataT]]: ...

    def run_stream_events(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AbstractAsyncContextManager[AgentRunEvents[Any]]:
        """Use the session dependencies unless this run explicitly overrides them."""
        return super().run_stream_events(
            user_prompt,
            output_type=output_type,
            deps=deps if _utils.is_set(deps) else self._deps,
            **kwargs,
        )

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: None = None,
        conversation: Conversation | None = None,
        message_history: Sequence[messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        cancellation_token: CancellationToken | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        workspace: WorkspaceBackend | WorkspaceRef | Literal['new'] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        conversation: Conversation | None = None,
        message_history: Sequence[messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        cancellation_token: CancellationToken | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        workspace: WorkspaceBackend | WorkspaceRef | Literal['new'] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[Any] | None = None,
        conversation: Conversation | None = None,
        message_history: Sequence[messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        cancellation_token: CancellationToken | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        workspace: WorkspaceBackend | WorkspaceRef | Literal['new'] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AsyncGenerator[AgentRun[AgentDepsT, Any]]:
        """Execute a run with this session's state; other arguments behave as on `Agent.iter`.

        `conversation`, `message_history`, `conversation_id`, and `usage` must be supplied when
        creating the session, not per run. Only one run may write to a session at a time.
        """
        if not self._entered:
            raise UserError('Enter the agent session with `async with` before starting a run.')
        if any(value is not None for value in (conversation, message_history, conversation_id, usage)):
            raise UserError('The session owns `conversation`, `message_history`, `conversation_id`, and `usage`.')
        self._runtime.claim()
        finished = self._run_finished = anyio.Event()
        final_conversation: Conversation | None = None
        error: BaseException | None = None
        try:
            with anyio.CancelScope() as scope:
                self._run_scope = scope
                try:
                    target = self.wrapped
                    while isinstance(target, WrapperAgent):
                        target = target.wrapped
                    with bind_session(target, self._runtime):
                        async with self.wrapped.iter(
                            user_prompt,
                            output_type=output_type,
                            conversation=self._runtime.conversation,
                            deferred_tool_results=deferred_tool_results,
                            run_id=run_id,
                            model=model if model is not None else self._model,
                            instructions=instructions,
                            deps=deps if _utils.is_set(deps) else self._deps,
                            model_settings=model_settings,
                            usage_limits=usage_limits,
                            cancellation_token=cancellation_token,
                            metadata=metadata,
                            retries=retries,
                            infer_name=infer_name,
                            toolsets=toolsets,
                            capabilities=capabilities,
                            workspace=workspace,
                            spec=spec,
                        ) as run:
                            self._runtime.require_attached()
                            yield run
                        if run.result is not None:
                            final_conversation = run.result.conversation
                except BaseException as exc:
                    error = exc
                    raise
            # Closing a session cancels the run's scope; do not swallow that cancellation and
            # pretend the caller received a successful result.
            if error is not None:
                raise error
        finally:
            self._runtime.release(final_conversation)
            self._run_scope = None
            self._run_finished = None
            finished.set()
