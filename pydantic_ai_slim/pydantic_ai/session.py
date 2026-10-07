"""Live ownership of a conversation across multiple agent runs."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, overload

import anyio
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import TypedDict, Unpack

from . import _instructions, _operations, _utils, messages, models, result, usage as _usage
from ._cancel import CancellationToken
from ._enqueue import EnqueueContent, PendingMessage, PendingMessagePriority
from ._operations import ToolOperation as ToolOperation
from ._session import SessionRuntime, bind_session
from .agent.abstract import (
    AbstractAgent,
    AgentMetadata,
    AgentModelSettings,
    AgentRealtime,
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
    from .realtime import AudioRetention, KnownRealtimeModelName, RealtimeModel, RealtimeModelSettings, RealtimeSession
    from .realtime._persistent import RealtimeAttachment
    from .realtime._run import RealtimeRun
    from .realtime.model import RealtimeProviderSession

__all__ = ('AgentSession', 'RealtimeAgentSession', 'SessionState', 'SessionStateTypeAdapter', 'ToolOperation')


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
    operations: list[ToolOperation] = field(default_factory=list[ToolOperation])
    """Tool execution and result-delivery facts; never an instruction to repeat an effect."""
    active_run_id: str | None = None
    """Unfinished work at capture time. This is not an instruction to retry that work."""

    def recover(
        self,
        *,
        tool_results: Mapping[str, messages.ModelRequest] | None = None,
        deliveries: Mapping[str, Literal['ready', 'committed']] | None = None,
        abandon_run: bool = False,
    ) -> SessionState:
        """Reconcile a detached checkpoint before opening a new session.

        Keys are `ToolOperation.operation_id`, not provider tool-call IDs. Supply normalized
        `tool_results` only after verifying an unknown external outcome; this never runs a tool.
        Each request must contain one matching return/retry and optional user content.
        `deliveries` explicitly authorizes resending an existing result (`ready`), or records
        external confirmation (`committed`). A lost send is never silently made ready.

        An active checkpoint additionally requires `abandon_run=True`: the application must first
        stop its old driver and fence any other writers. This abandons unfinished model generation
        and closes unanswered calls, not external effects. All unknown tool outcomes still need
        results. Usage after the checkpoint cannot be recovered here. Durable engines should
        normally replay their own recorded operations rather than abandon an active workflow.

        The original checkpoint is unchanged, including if reconciliation fails.
        """
        if self.active_run_id is not None and not abandon_run:
            raise UserError('The checkpoint contains an unfinished run; stop its owner and pass `abandon_run=True`.')
        recovered = deepcopy(self)
        operations = {op.operation_id: op for op in recovered.operations}
        if len(operations) != len(recovered.operations):
            raise UserError('The checkpoint contains duplicate tool operation IDs.')
        for operation_id, request in (tool_results or {}).items():
            operation = _get_operation(operations, operation_id)
            request = deepcopy(request)
            returns = [p for p in request.parts if isinstance(p, (messages.ToolReturnPart, messages.RetryPromptPart))]
            if (
                len(returns) != 1
                or returns[0].tool_call_id != operation.call.tool_call_id
                or returns[0].tool_name != operation.call.tool_name
                or any(
                    not isinstance(p, (messages.ToolReturnPart, messages.RetryPromptPart, messages.UserPromptPart))
                    for p in request.parts
                )
            ):
                raise UserError(f'Recovery result must answer only tool operation {operation_id!r}.')
            request = replace(
                request,
                run_id=operation.run_id,
                conversation_id=recovered.conversation.conversation_id,
                state='complete',
            )
            _operations.apply(operations, operation_id, _operations.ReconcileTool([request]))
        for operation_id, status in (deliveries or {}).items():
            _get_operation(operations, operation_id)
            _operations.apply(operations, operation_id, _operations.ReconcileDelivery(status))
        recovered.operations = list(operations.values())
        _require_settled_operations(recovered.operations)
        for operation in recovered.operations:
            if operation.execution == 'completed' and (
                operation.operation_id in (tool_results or {})
                or self.active_run_id is not None
                and any(request.run_id == self.active_run_id for request in operation.result)
            ):
                _restore_operation_result(recovered.conversation, operation)
        if self.active_run_id is not None:
            recovered.conversation.messages = messages.repair_messages(
                recovered.conversation.messages,
                repair_last_response=recovered.conversation.deferred_tool_requests is None,
            )
            if recovered.conversation.messages:
                last = recovered.conversation.messages[-1]
                recovered.conversation.messages[-1] = replace(last, state='interrupted')
        recovered.active_run_id = None
        return recovered


def _get_operation(operations: Mapping[str, ToolOperation], operation_id: str) -> ToolOperation:
    try:
        return operations[operation_id]
    except KeyError:
        raise UserError(f'Unknown tool operation {operation_id!r}.') from None


def _require_settled_operations(operations: Sequence[ToolOperation]) -> None:
    for operation in operations:
        if operation.execution in ('running', 'interrupted') or operation.delivery in (
            'sending',
            'sent',
            'accepted',
            'uncertain',
        ):
            raise UserError(f'Tool operation {operation.operation_id!r} is unresolved; use `state.recover()` first.')


def _restore_operation_result(conversation: Conversation, operation: ToolOperation) -> None:
    # Restore beside the originating response, not at the end of unrelated later history.
    history = conversation.messages
    indices = [
        index
        for index, response in enumerate(history)
        if isinstance(response, messages.ModelResponse)
        and response.run_id == operation.run_id
        and (operation.response_timestamp is None or response.timestamp == operation.response_timestamp)
        and operation.call_index < len(response.tool_calls)
        and response.tool_calls[operation.call_index].tool_call_id == operation.call.tool_call_id
        and response.tool_calls[operation.call_index].tool_name == operation.call.tool_name
    ]
    if len(indices) != 1:
        raise UserError(
            f'The conversation cannot unambiguously locate the call for operation {operation.operation_id!r}.'
        )
    index = indices[0]
    following = history[index + 1] if index + 1 < len(history) else None
    existing = (
        list(following.parts) if isinstance(following, messages.ModelRequest) else list[messages.ModelRequestPart]()
    )
    restored_parts: list[messages.ModelRequestPart] = []
    for request in operation.result:
        assert isinstance(request, messages.ModelRequest)
        restored_parts.extend(request.parts)
    for part in existing:
        if (
            isinstance(part, (messages.ToolReturnPart, messages.RetryPromptPart))
            and part.tool_call_id == operation.call.tool_call_id
        ):
            if any(
                isinstance(restored, (messages.ToolReturnPart, messages.RetryPromptPart))
                and restored.timestamp == part.timestamp
                and restored.tool_name == part.tool_name
                for restored in restored_parts
            ):
                # Already assembled completion must not duplicate multimodal user content.
                return
            raise UserError(f'History already has a different result for operation {operation.operation_id!r}.')
    if isinstance(following, messages.ModelRequest):
        history[index + 1] = replace(following, parts=[*existing, *restored_parts])
    else:
        history[index + 1 : index + 1] = deepcopy(operation.result)


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


class _RealtimeMediaOptions(TypedDict, total=False):
    audio_retention: AudioRetention
    handle_barge_in: bool
    retain_images_every_n: int
    retain_images_max: int | None
    retain_audio_max_seconds: float | None
    provider_session: RealtimeProviderSession | None


class _RealtimeOptions(TypedDict, Generic[AgentDepsT], total=False):
    model_settings: RealtimeModelSettings | None
    instructions: _instructions.AgentInstructions[AgentDepsT]
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None
    capabilities: Sequence[AgentCapability[AgentDepsT]] | None
    usage: _usage.RunUsage | None
    usage_limits: _usage.UsageLimits | None
    metadata: AgentMetadata[AgentDepsT] | None
    conversation_id: str | None
    run_id: str | None
    message_history: Sequence[messages.ModelMessage] | None


class _RealtimeSessionOptions(_RealtimeOptions[AgentDepsT], total=False):
    audio_retention: AudioRetention
    handle_barge_in: bool
    retain_images_every_n: int
    retain_images_max: int | None
    retain_audio_max_seconds: float | None
    provider_session: RealtimeProviderSession | None


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
        if state is not None:
            _require_settled_operations(state.operations)
            self._runtime.operations.update((op.operation_id, deepcopy(op)) for op in state.operations)
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
        try:
            try:
                await self._cancel_active_run()
                if self._runtime.realtime is not None:
                    await self._runtime.realtime.close()
            finally:
                self._runtime.resources.close()
        except BaseException as exc:
            args = (type(exc), exc, exc.__traceback__)
            # Exit the task group even when connection shutdown fails. Passing the failure into
            # the stack preserves its normal cancellation and multi-error aggregation semantics.
            await self._exit_stack(*args)
            raise
        return await self._exit_stack(*args)

    async def _exit_stack(self, *args: Any) -> bool | None:
        try:
            return await self._stack.__aexit__(*args)
        except BaseExceptionGroup as group:
            if len(group.exceptions) == 1:
                raise group.exceptions[0] from None
            raise

    async def _cancel_active_run(self) -> None:
        # Runs started by a streaming handle or another task must finish teardown before the
        # resources they use close. The run task, not this caller, exits its cancel scope.
        if self._run_scope is not None:
            self._run_scope.cancel()
            assert self._run_finished is not None
            with anyio.CancelScope(shield=True):
                await self._run_finished.wait()

    @property
    def conversation(self) -> Conversation:
        """A detached snapshot of the conversation, including the current run's history."""
        return self._runtime.snapshot()[0]

    @property
    def state(self) -> SessionState:
        """A detached checkpoint of conversation, pending input, and unfinished work."""
        conversation, pending, active_run_id = self._runtime.snapshot()
        return SessionState(
            conversation=conversation,
            pending=pending,
            active_run_id=active_run_id,
            operations=deepcopy(list(self._runtime.operations.values())),
        )

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
        self._prepare_options(kwargs)
        async with self._execution():
            result = await self.wrapped.run(
                user_prompt,
                output_type=output_type,
                deps=deps if _utils.is_set(deps) else self._deps,
                event_stream_handler=event_stream_handler,
                **kwargs,
            )
            self._runtime.require_attached()
            self._runtime.record_result(result.conversation)
            return result

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

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AsyncGenerator[result.StreamedRunResult[AgentDepsT, Any]]:
        """Use the session dependencies unless this run explicitly overrides them."""
        self._prepare_options(kwargs)
        async with self._execution():
            async with self.wrapped.run_stream(
                user_prompt,
                output_type=output_type,
                deps=deps if _utils.is_set(deps) else self._deps,
                event_stream_handler=event_stream_handler,
                **kwargs,
            ) as streamed:
                self._runtime.require_attached()
                yield streamed

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

    @asynccontextmanager
    async def run_stream_events(
        self,
        user_prompt: str | Sequence[messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        **kwargs: Unpack[_InstructedRunOptions[AgentDepsT]],
    ) -> AsyncGenerator[AgentRunEvents[Any]]:
        """Use the session dependencies unless this run explicitly overrides them."""
        self._prepare_options(kwargs)
        async with self._execution():
            async with self.wrapped.run_stream_events(
                user_prompt,
                output_type=output_type,
                deps=deps if _utils.is_set(deps) else self._deps,
                **kwargs,
            ) as events:
                yield events
            if events.result is not None:
                self._runtime.require_attached()
                self._runtime.record_result(events.result.conversation)

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
        if any(value is not None for value in (conversation, message_history, conversation_id, usage)):
            raise UserError('The session owns `conversation`, `message_history`, `conversation_id`, and `usage`.')
        async with self._execution():
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
                self._runtime.record_result(run.result.conversation)

    def realtime(
        self,
        model: RealtimeModel | KnownRealtimeModelName | str,
        *,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        conversation: Conversation | None = None,
        **kwargs: Unpack[_RealtimeOptions[AgentDepsT]],
    ) -> AgentRealtime[AgentDepsT]:
        """Bind a live interaction to this session's exclusive conversation owner.

        The existing `.session()` attachment is one Run. Its history and pending input are
        returned to this owner before another ordinary or live Run can begin.
        """
        if conversation is not None or any(
            kwargs.get(key) is not None for key in ('message_history', 'conversation_id', 'usage')
        ):
            raise UserError('The session owns `conversation`, `message_history`, `conversation_id`, and `usage`.')
        return super().realtime(model, deps=deps if _utils.is_set(deps) else self._deps, **kwargs)

    @asynccontextmanager
    async def _open_realtime_session(
        self,
        model: RealtimeModel | KnownRealtimeModelName | str,
        *,
        deps: AgentDepsT = None,
        **kwargs: Unpack[_RealtimeSessionOptions[AgentDepsT]],
    ) -> AsyncGenerator[RealtimeSession]:
        if any(kwargs.get(key) is not None for key in ('message_history', 'conversation_id', 'usage')):
            raise UserError('The session owns `message_history`, `conversation_id`, and `usage`.')
        async with self._execution(realtime=True):
            conversation = self._runtime.conversation
            if conversation.deferred_tool_requests is not None:
                raise UserError("Resolve the conversation's deferred tools before starting a realtime run.")
            kwargs['message_history'] = conversation.messages
            kwargs['conversation_id'] = conversation.conversation_id
            kwargs['usage'] = conversation.usage
            async with self.wrapped._open_realtime_session(model, deps=deps, **kwargs) as live:
                self._runtime.require_attached()
                yield live
            self._runtime.record_result(live.result.conversation if live.result is not None else live.conversation)

    def _prepare_options(self, options: _RunOptions[AgentDepsT]) -> None:
        if any(options.get(key) is not None for key in ('conversation', 'message_history', 'conversation_id', 'usage')):
            raise UserError('The session owns `conversation`, `message_history`, `conversation_id`, and `usage`.')
        options['conversation'] = self._runtime.conversation
        if options.get('model') is None:
            options['model'] = self._model

    @asynccontextmanager
    async def _execution(self, *, realtime: bool = False) -> AsyncGenerator[None]:
        if not self._entered:
            raise UserError('Enter the agent session with `async with` before starting a run.')
        self._runtime.claim(realtime=realtime)
        finished = self._run_finished = anyio.Event()
        error: BaseException | None = None
        try:
            with anyio.CancelScope() as scope:
                self._run_scope = scope
                try:
                    target = self.wrapped
                    while isinstance(target, WrapperAgent):
                        target = target.wrapped
                    with bind_session(target, self._runtime):
                        yield
                except BaseException as exc:
                    error = exc
                    raise
            # Closing a session cancels the run's scope; do not turn cancellation into success.
            if error is not None:
                raise error
        finally:
            self._runtime.release()
            self._run_scope = None
            self._run_finished = None
            finished.set()


class RealtimeAgentSession(Generic[AgentDepsT]):
    """A realtime attachment to an `AgentSession`, containing sequential, independent runs.

    Created by `session.realtime(model).connect()`. The first non-short-circuited run opens the
    connection. Normal run exit waits for outstanding replies, tools and input transcripts;
    connection exit closes the transport. Stop submitting audio before leaving a run. Apply an
    application deadline when the provider might never deliver its final transcript.

    Dependencies, metadata, hooks and tool resources are resolved afresh per run. Wire settings,
    instructions and advertised tool schemas must remain unchanged on this connection.
    """

    def __init__(self, definition: AgentRealtime[AgentDepsT], **media: Unpack[_RealtimeMediaOptions]) -> None:
        owner = definition._agent  # pyright: ignore[reportPrivateUsage]
        if not isinstance(owner, AgentSession):
            raise UserError('Use `agent.session()` before `session.realtime(...).connect()`.')
        self._owner = owner
        self._definition = definition
        self._media = media
        self._attachment: RealtimeAttachment | None = None
        self._used = False

    async def __aenter__(self) -> Self:
        from .realtime._persistent import RealtimeAttachment

        if self._used:
            raise UserError('A realtime attachment may only be entered once.')
        if not self._owner._entered:  # pyright: ignore[reportPrivateUsage]
            raise UserError('Enter the agent session before opening a realtime connection.')
        runtime = self._owner._runtime  # pyright: ignore[reportPrivateUsage]
        if runtime.realtime is not None:
            raise UserError('This agent session already owns a realtime connection.')
        runtime.claim(realtime=True)
        runtime.release()
        group = runtime.resources._group  # pyright: ignore[reportPrivateUsage]
        assert group is not None
        self._attachment = runtime.realtime = RealtimeAttachment(group)
        self._used = True
        return self

    async def __aexit__(self, *args: Any) -> None:
        assert self._attachment is not None
        self._attachment.closed = True
        try:
            await self._owner._cancel_active_run()  # pyright: ignore[reportPrivateUsage]
            await self._attachment.close()
        finally:
            self._owner._runtime.realtime = None  # pyright: ignore[reportPrivateUsage]

    @asynccontextmanager
    async def run(
        self,
        *,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        metadata: AgentMetadata[AgentDepsT] | None | _utils.Unset = _utils.UNSET,
        usage_limits: _usage.UsageLimits | None | _utils.Unset = _utils.UNSET,
        run_id: str | None = None,
    ) -> AsyncGenerator[RealtimeRun]:
        """Execute a fresh run on this attachment without reconnecting.

        Normal exit drains pending work. Cancellation or an exceptional exit closes the connection;
        create another attachment rather than reusing an uncertain provider execution frontier.
        """
        from .realtime._run import RealtimeRun

        if self._attachment is None or self._attachment.closed:
            raise UserError('Enter the realtime connection before starting a run; it must not be closed.')
        definition = copy(self._definition)
        if _utils.is_set(deps):
            definition._deps = deps  # pyright: ignore[reportPrivateUsage]
        if _utils.is_set(metadata):
            definition._metadata = metadata  # pyright: ignore[reportPrivateUsage]
        if _utils.is_set(usage_limits):
            definition._usage_limits = usage_limits  # pyright: ignore[reportPrivateUsage]
        definition._run_id = run_id  # pyright: ignore[reportPrivateUsage]
        handle: RealtimeRun | None = None
        try:
            async with definition._session(**self._media) as session:  # pyright: ignore[reportPrivateUsage]
                handle = session._run.handle or RealtimeRun(session)  # pyright: ignore[reportPrivateUsage]
                yield handle
        finally:
            if handle is not None:
                handle._capture()  # pyright: ignore[reportPrivateUsage]
