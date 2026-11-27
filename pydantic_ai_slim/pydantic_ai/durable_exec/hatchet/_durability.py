from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, cast
from weakref import WeakKeyDictionary

import anyio
from hatchet_sdk import DurableContext, Hatchet, NonRetryableError, Standalone
from hatchet_sdk.engine_version import MinEngineVersion
from pydantic import JsonValue, TypeAdapter
from pydantic_core import to_json

from pydantic_ai._agent_graph import set_agent_graph_sleep
from pydantic_ai._cancel import provide_run_deadline
from pydantic_ai._utils import aclose_if_supported
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering, WrapRunHandler
from pydantic_ai.durable_exec._base import (
    MODEL_RESPONSE_STREAM_EVENT_TYPES,
    BaseDurabilityCapability,
    conversation_id_from_run_id,
)
from pydantic_ai.durable_exec._operation import (
    DurableOperationId,
    ToolsetCallToolId,
    ToolsetKind,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import RoleBasedOperationConfig
from pydantic_ai.durable_exec._utils import has_retry_on_model_errors
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelQuotaExceededError,
    ModelRequestRejectedError,
    ModelRequestTimeout,
    RunCancelled,
    RunTimedOut,
    UserError,
)
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.tool_manager import (
    ToolManager,
    _parallel_execution_mode_ctx_var,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._clock import durable_clock
from ._event_stream import HatchetAgentRunEvents, HatchetEventStream, publish_event, task_event_stream_handler
from ._operation_backend import (
    HatchetOperationBackend,
    HatchetTaskConfig,
    TaskOutput,
    validate_task_config,
)
from ._run_context import HatchetRunContext, deserialize_run_context
from ._spec import HATCHET_ENGINE_SPEC
from ._transports import HatchetParameterTransports, TaskInput

if TYPE_CHECKING:
    from hatchet_sdk.features.streams import StreamEvent

_DEFAULT_NON_RETRYABLE_MODEL_ERRORS: tuple[type[ModelAPIError], ...] = (
    ModelRequestRejectedError,
    ModelQuotaExceededError,
    ModelRequestTimeout,
)
"""The model errors another attempt of the same request can't fix, which Hatchet doesn't retry by default."""

_DEFAULT_TASK_CONFIG = HatchetTaskConfig(execution_timeout=timedelta(seconds=60))

_runs_by_context: WeakKeyDictionary[DurableContext, int] = WeakKeyDictionary()
"""How many agent runs each durable task has started, which tells its runs' default IDs apart."""

_published_event_counts: WeakKeyDictionary[DurableContext, dict[str | None, int]] = WeakKeyDictionary()
"""How many events each run has published from the durable task, across all of its nodes' event streams."""

_run_start_times: WeakKeyDictionary[DurableContext, dict[str | None, datetime]] = WeakKeyDictionary()
"""The start time `get_run_start_time` recorded for each run, which `wrap_run` starts the run's durable clock at."""


@dataclass(init=False, kw_only=True)
class HatchetDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by running its model requests and tool calls as Hatchet tasks.

    When the agent runs inside a Hatchet [durable task](https://docs.hatchet.run/home/durable-execution),
    each model request, MCP communication and tool call is spawned as a child task, whose result the durable
    task's event log records: a durable task that's replayed after a crash, an eviction or a retry gets the
    results back without making the requests again. Outside a durable task, the capability is transparent.

    The tasks are registered on `hatchet` when the capability is attached to an agent. Pass
    [`workflows`][pydantic_ai.durable_exec.hatchet.HatchetDurability.workflows] to the worker
    alongside the durable task that runs the agent.

    Example:
        ```python {test="skip"}
        from hatchet_sdk import DurableContext, Hatchet

        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.hatchet import HatchetDurability

        hatchet = Hatchet()
        durability = HatchetDurability(hatchet)
        agent = Agent('openai:gpt-5.6-sol', name='my_agent', capabilities=[durability])


        @hatchet.durable_task(name='answer', input_validator=dict[str, str])
        async def answer(input: dict[str, str], ctx: DurableContext) -> dict[str, str]:
            result = await agent.run(input['question'])
            return {'answer': result.output}


        worker = hatchet.worker('agent-worker', workflows=[answer, *durability.workflows])
        ```
    """

    engine_spec = HATCHET_ENGINE_SPEC

    hatchet: Hatchet
    """The Hatchet client the capability registers its tasks on."""

    run_context_type: type[HatchetRunContext[AgentDepsT]]
    """The `HatchetRunContext` subclass used to serialize/deserialize the run context."""

    def __init__(
        self,
        hatchet: Hatchet,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        deps_type: type[AgentDepsT] | None = None,
        task_config: HatchetTaskConfig | None = None,
        model_task_config: HatchetTaskConfig | None = None,
        event_stream_handler_task_config: HatchetTaskConfig | None = None,
        toolset_task_config: Mapping[str, HatchetTaskConfig] | None = None,
        run_context_type: type[HatchetRunContext[AgentDepsT]] = HatchetRunContext[AgentDepsT],
        non_retryable_model_errors: Sequence[type[ModelAPIError]] = _DEFAULT_NON_RETRYABLE_MODEL_ERRORS,
        event_stream: bool | HatchetEventStream = False,
    ) -> None:
        """Create a `HatchetDurability` capability.

        The agent's model, name, and toolsets are discovered when the capability is attached to an agent.

        Args:
            hatchet: The Hatchet client to register the agent's tasks on.
            models: Additional models keyed by ID for runtime model switching. The agent's primary model is
                always registered as `'default'`. A `Model` instance can't be serialized into a task, so a
                run-time model instance has to be registered here and referenced by key; model name strings
                cross as written and are built on the worker.
            event_stream_handler: Optional event stream handler. Model events are handled live inside the
                model request tasks, and other events in a task per event.
            name: Unique agent name used in the Hatchet task names. Defaults to the agent's `name`.
            deps_type: The type of the agent's dependencies, used to serialize them into tasks as JSON.
                Defaults to the agent's own `deps_type`.
            task_config: Base Hatchet task config for all tasks. Defaults to a 60-second
                `execution_timeout` and no retries.
            model_task_config: Task config merged on top of the base for model request tasks.
            event_stream_handler_task_config: Task config merged on top of the base for event stream
                handler tasks.
            toolset_task_config: Per-toolset task configs keyed by toolset ID, merged on top of the base.
            run_context_type: The `HatchetRunContext` subclass used to serialize the run context into tasks.
            non_retryable_model_errors: Model error classes that fail a model task without Hatchet retrying
                it, whatever its `retries` say, subclasses included. The durable task still receives the error
                as its own class. Defaults to
                [`ModelRequestRejectedError`][pydantic_ai.exceptions.ModelRequestRejectedError],
                [`ModelQuotaExceededError`][pydantic_ai.exceptions.ModelQuotaExceededError] and
                [`ModelRequestTimeout`][pydantic_ai.exceptions.ModelRequestTimeout]; `()` retries them all.
            event_stream: `True`, or a [`HatchetEventStream`][pydantic_ai.durable_exec.hatchet.HatchetEventStream],
                to publish the run's events to a Hatchet durable stream topic, by default one per durable task and agent, so a
                consumer outside the durable task can observe them as they happen with
                [`stream_agent_events()`][pydantic_ai.durable_exec.hatchet.HatchetDurability.stream_agent_events].
                Model events are published live from the model request task, so tokens stream through.
                Orthogonal to `event_stream_handler`: when both are set, the handler still sees every event.

        Note:
            A tool's `'hatchet'` metadata can override the run options of its task (`priority`, `sticky`,
            `desired_worker_labels` and `additional_metadata`), or be `False` to run an async function tool
            directly in the durable task instead of in a task of its own:

            ```python {test="skip" lint="skip"}
            @my_toolset.tool(metadata={'hatchet': {'desired_worker_labels': [DesiredWorkerLabel(key='gpu', value='true')]}})
            async def my_gpu_tool(...): ...
            ```
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self.hatchet = hatchet
        self.run_context_type = run_context_type
        self._deps_type = deps_type
        self._non_retryable_model_errors = tuple(non_retryable_model_errors)
        base = {**_DEFAULT_TASK_CONFIG, **validate_task_config(task_config or {}, '`task_config`')}
        self._task_config = cast(HatchetTaskConfig, base)
        self._model_task_config = cast(
            HatchetTaskConfig, {**base, **validate_task_config(model_task_config or {}, '`model_task_config`')}
        )
        self._event_task_config = cast(
            HatchetTaskConfig,
            {
                **base,
                **validate_task_config(event_stream_handler_task_config or {}, '`event_stream_handler_task_config`'),
            },
        )
        self._toolset_task_config = {
            toolset_id: validate_task_config(config, f'`toolset_task_config[{toolset_id!r}]`')
            for toolset_id, config in (toolset_task_config or {}).items()
        }
        self._operation_backend: HatchetOperationBackend | None = None
        self._deps_adapter: TypeAdapter[Any] | None = None
        # Shared with the bound copies `for_agent()` makes, so `workflows` on the instance the user holds lists
        # the tasks of every agent it's attached to.
        self._registered_tasks: list[Standalone[TaskInput, TaskOutput]] = []
        self._bound_agent_names: list[str] = []
        self._retry_on_model_errors = False
        self._event_stream = (
            HatchetEventStream() if event_stream is True else event_stream if event_stream is not False else None
        )

    @property
    def in_durable_context(self) -> bool:
        return isinstance(self.hatchet.current_context, DurableContext)

    def _durable_context(self) -> DurableContext:
        ctx = self.hatchet.current_context
        assert isinstance(ctx, DurableContext)
        return ctx

    def _check_bindable(self) -> None:
        if self.in_durable_context:
            raise UserError(
                'An agent with `HatchetDurability` must be constructed outside of a Hatchet durable task, '
                'so its tasks can be registered with the worker before the durable task runs. '
                'Construct the agent at module level (or in worker setup code) and reference it from the task.'
            )

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        if self.name not in self._bound_agent_names:
            self._bound_agent_names.append(self.name)
        if self._deps_type is None:
            self._deps_type = cast('type[AgentDepsT]', agent.deps_type)
        self._deps_adapter = TypeAdapter(self._deps_type)
        non_retryable_model_errors = self._non_retryable_model_errors
        self._retry_on_model_errors = has_retry_on_model_errors(agent.root_capability)
        if self._retry_on_model_errors:
            # `RetryOnModelErrors` retries model errors in the durable task, so the model task doesn't retry them too:
            # its `retries` are left to crashes and timeouts, which never reach a hook.
            if non_retryable_model_errors != _DEFAULT_NON_RETRYABLE_MODEL_ERRORS:
                warnings.warn(
                    '`non_retryable_model_errors` is ignored because the agent has `RetryOnModelErrors`: that '
                    'capability does the retrying of model errors, so the model request task treats every model '
                    'error as non-retryable. Configure which errors are retried with `RetryOnModelErrors(retry_on=...)` '
                    'instead.',
                    UserWarning,
                    stacklevel=2,
                )
            non_retryable_model_errors = (ModelAPIError,)
        backend = HatchetOperationBackend(
            self.hatchet,
            runtime=self,
            agent_name=self.name,
            config=RoleBasedOperationConfig[HatchetTaskConfig](
                model=self._model_task_config,
                event=self._event_task_config,
                capability=self._task_config,
                tool=self._task_config,
                resolve_tool=self._resolve_tool_config,
            ),
            parameter_transports=HatchetParameterTransports(self),
            non_retryable_model_errors=non_retryable_model_errors,
        )
        self._operation_backend = backend
        # Bind events first, as the durable model and toolset wrappers capture this operation.
        if self._event_stream_handler is not None:
            self._bound_event_operation = self._bind_event_operation(backend)
        default_model = self._models_by_id.get('default')
        model_name = self.default_model_id or (default_model.model_id if default_model is not None else 'default')
        self._bound_model_operations = self._bind_model_operations(backend, model_id=None, model_name=model_name)
        self._register_toolsets(agent)

    def get_durable_operation_backend(self) -> HatchetOperationBackend:
        backend = self._operation_backend
        if backend is None:
            raise UserError('`HatchetDurability` must be attached to an agent before its tasks are used.')
        return backend

    @property
    def workflows(self) -> list[Standalone[TaskInput, TaskOutput]]:
        """The Hatchet tasks the capability registered, to pass to `hatchet.worker(workflows=...)`.

        A worker that runs the durable task running the agent needs them, or another worker on the same
        Hatchet tenant does: a child task runs on whichever worker serves it.
        """
        return list(self._registered_tasks)

    def _register_task(self, task: Standalone[TaskInput, TaskOutput]) -> None:
        self._registered_tasks.append(task)

    # --- Configuration ---

    def _toolset_config(self, toolset_id: str) -> HatchetTaskConfig:
        return cast(HatchetTaskConfig, {**self._task_config, **self._toolset_task_config.get(toolset_id, {})})

    def _toolset_operation_config(self, kind: ToolsetKind, toolset_id: str) -> HatchetTaskConfig:
        return self._toolset_config(toolset_id)

    def _resolve_tool_config(
        self, operation_id: DurableOperationId, tool: object | None, name: str
    ) -> HatchetTaskConfig | Literal[False]:
        base = self._toolset_config(cast(Any, operation_id).toolset_id)
        typed_tool = cast(ToolsetTool[Any] | None, tool)
        metadata = typed_tool.tool_def.metadata if typed_tool is not None else None
        tool_config = metadata.get('hatchet') if metadata is not None else None
        if tool_config is None:
            return base
        if tool_config is False:
            is_tool_operation = isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId)
            if is_tool_operation and operation_id.toolset_kind != 'function':
                raise UserError(
                    f'Hatchet task config for tool {name!r} has been explicitly set to `False`, but only function '
                    'tools can run outside a task: MCP and dynamic toolset tools may perform I/O.'
                )
            if not isinstance(tool, FunctionToolsetTool) or not tool.is_async:
                raise UserError(
                    f'Hatchet task config for tool {name!r} has been explicitly set to `False`, but non-async '
                    'tools run in threads, which a durable task does not replay. Make the tool function async.'
                )
            return False
        if not isinstance(tool_config, Mapping):
            raise UserError(f"Tool {name!r} has invalid `'hatchet'` metadata: expected a dict or `False`.")
        return cast(
            HatchetTaskConfig,
            {**base, **validate_task_config(cast(Mapping[str, object], tool_config), f'Tool {name!r} metadata')},
        )

    # --- Transport runtime ---

    def serialize_operation_run_context(self, ctx: RunContext[Any]) -> dict[str, Any]:
        return self.run_context_type.serialize_run_context(ctx)

    def deserialize_operation_run_context(self, serialized: dict[str, Any], deps: Any) -> RunContext[AgentDepsT]:
        return deserialize_run_context(self.run_context_type, serialized, deps=deps, agent=self._agent)

    def dump_deps(self, deps: Any) -> JsonValue:
        assert self._deps_adapter is not None
        return cast(JsonValue, self._deps_adapter.dump_python(deps, mode='json'))

    def load_deps(self, payload: JsonValue) -> Any:
        assert self._deps_adapter is not None
        # Validated as the JSON it was dumped as, which a strict model accepts where it would reject the Python form.
        return self._deps_adapter.validate_json(to_json(payload))

    # --- Capability hooks ---

    def get_default_run_id(self) -> str | None:
        if not self.in_durable_context:
            return None
        ctx = self._durable_context()
        # A replayed durable task starts its agent runs in the same order, so they get the same IDs.
        index = _runs_by_context.get(ctx, 0)
        _runs_by_context[ctx] = index + 1
        return f'{ctx.task_run_id}:{index}'

    def get_default_conversation_id(self, run_id: str) -> str | None:
        return conversation_id_from_run_id(run_id) if self.in_durable_context else None

    async def get_run_start_time(self, run_id: str) -> datetime | None:
        if not self.in_durable_context:
            return None
        durable_ctx = self._durable_context()
        start = await durable_ctx.aio_now()
        _run_start_times.setdefault(durable_ctx, {})[run_id] = start
        return start

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Make the run replay-safe: durable sleeps for the agent graph's waits, and message times from a durable clock.

        A replayed durable task spawns its child tasks again and Hatchet checks each against the recorded one,
        input included, so the run must stamp its messages with the times it stamped the first time.
        """
        if not self.in_durable_context:
            return await handler()
        if (
            not self._retry_on_model_errors
            and ctx.root_capability is not None
            and has_retry_on_model_errors(ctx.root_capability)
        ):
            # The model task's retries are set when the agent is bound, so it would retry model errors too.
            raise UserError(
                'In a Hatchet durable task, `RetryOnModelErrors` must be one of the capabilities the agent is built '
                'with, so the model request task leaves retrying model errors to it. Add it to the agent directly '
                'rather than through a `DynamicCapability`.'
            )
        durable_ctx = self._durable_context()
        if not durable_ctx._supports_durable_eviction:  # pyright: ignore[reportPrivateUsage]
            # Such an engine spawns child tasks as plain runs, outside the event log, so a replay runs them again.
            raise UserError(
                f'Hatchet engine {durable_ctx._engine_version or "of unknown version"} records no child tasks in the '  # pyright: ignore[reportPrivateUsage]
                f'event log of a durable task, so it cannot replay an agent run. Upgrade it to '
                f'{MinEngineVersion.DURABLE_EVICTION} or later.'
            )

        async def durable_sleep(seconds: float) -> None:
            # Hatchet takes seconds to wake a durable task even from a zero-length sleep, which waits for nothing.
            if seconds <= 0:
                return
            await durable_ctx.aio_sleep_for(timedelta(seconds=seconds))

        # Recorded before the run spawns anything, so a replay starts its clock at the same time. A run with a
        # `timeout=` recorded it already, to count the deadline from.
        start = _run_start_times.get(durable_ctx, {}).pop(ctx.run_id, None) or await durable_ctx.aio_now()
        # Parallel tool calls emit their events once all of them are done, in call order: a replay hands their results
        # back in another order than they finished, and event IDs and event handler tasks are numbered by position.
        mode = 'sequential' if _parallel_execution_mode_ctx_var.get() == 'sequential' else 'parallel_ordered_events'
        with (
            set_agent_graph_sleep(durable_sleep),
            durable_clock(start),
            ToolManager.parallel_execution_mode(mode),
        ):
            return await super().wrap_run(ctx, handler=handler)

    @contextmanager
    def _tool_run_context_scope(self, ctx: RunContext[AgentDepsT]) -> Generator[RunContext[AgentDepsT]]:
        with super()._tool_run_context_scope(ctx) as durable_ctx:
            if ctx.deadline is None or self.in_durable_context:
                # In the durable task, the run already provides its deadline, on its replay-safe clock.
                yield durable_ctx
                return
            # A child task doesn't share the durable task's context, so an agent a tool runs here inherits the run's
            # deadline from the serialized context instead.
            with provide_run_deadline(ctx.deadline):
                yield durable_ctx

    # --- Durable streams ---

    def _run_topic(self) -> str | None:
        """The run's durable stream topic, inside a durable task with `event_stream` set."""
        if (config := self._event_stream) is None or not self.in_durable_context:
            return None
        return config.topic(self.name, self._durable_context().workflow_run_id)

    @property
    def has_wrap_run_event_stream(self) -> bool:
        # A topic is a reason to stream on its own, but only inside a durable task, where there is one.
        return super().has_wrap_run_event_stream or self._run_topic() is not None

    def _model_stream_event_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        """Publish the live model stream from inside the model request task that produces it."""
        handler = super()._model_stream_event_handler()
        if (config := self._event_stream) is None:
            return handler
        return task_event_stream_handler(self.hatchet, config, self.name, handler=handler)

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Publish the run's durable-task-side events to its topic."""
        events = super().wrap_run_event_stream(ctx, stream=stream)
        config = self._event_stream
        topic = self._run_topic()
        if config is None or topic is None:
            async for event in events:
                yield event
            return

        # Model events were published by the task that produced them; these are their copies. A replayed durable
        # task publishes the rest again, in the same order, so their IDs say which ones a consumer already has.
        # Each node of the run streams its events through here, so the count is kept per run, not per stream.
        counts = _published_event_counts.setdefault(self._durable_context(), {})
        try:
            async for event in events:
                if not isinstance(event, MODEL_RESPONSE_STREAM_EVENT_TYPES) and (
                    config.events is None or config.events(event)
                ):
                    index = counts.get(ctx.run_id, 0)
                    counts[ctx.run_id] = index + 1
                    await publish_event(self.hatchet, config, topic, message_id=f'{ctx.run_id}:{index}', event=event)
                yield event
        finally:
            await aclose_if_supported(events)

    def _companion_capabilities(self) -> list[AbstractCapability[AgentDepsT]]:
        """Pair with the terminal-event publisher when publishing events; it wraps outside the base's companions."""
        companions = super()._companion_capabilities()
        if self._event_stream is None:
            return companions
        return [_TerminalEventPublisher(self), *companions]

    async def _publish_terminal_event(self, ctx: RunContext[AgentDepsT], result: AgentRunResult[Any]) -> None:
        config = self._event_stream
        topic = self._run_topic()
        if config is not None and topic is not None:
            await publish_event(
                self.hatchet, config, topic, message_id=f'{ctx.run_id}:result', event=AgentRunResultEvent(result)
            )

    async def _publish_terminal_error(
        self, ctx: RunContext[AgentDepsT], error_name: str, *, retryable: bool = True
    ) -> None:
        config = self._event_stream
        topic = self._run_topic()
        if config is not None and topic is not None:
            durable_ctx = self._durable_context()
            if retryable and durable_ctx.attempt_number < durable_ctx.max_attempts:
                # Hatchet retries the durable task if the error fails it, and the retry's run publishes the outcome
                # under the same IDs, so an error here would end subscribers' iteration before a result that may come.
                return
            # Only the exception type: a consumer may show the stream to end users, and the message can carry a
            # provider's error body or a tool's internals. The durable task's own result has the full error.
            await publish_event(self.hatchet, config, topic, message_id=f'{ctx.run_id}:result', error=error_name)

    def stream_agent_events(
        self,
        workflow_run_id: str,
        *,
        output_type: type[OutputDataT] = cast('type[Any]', Any),
        cursor: str | None = None,
    ) -> HatchetAgentRunEvents[OutputDataT]:
        """Subscribe to the events a durable agent run publishes with `event_stream` set.

        This is a durable [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events]: the events
        arrive typed and in order, ending with the [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent]
        that carries the run's result, so they can be handed to a [`UIAdapter`][pydantic_ai.ui.UIAdapter]. The
        topic keeps its messages after the run, so a consumer can subscribe late, or again.

        A run that times out or cancels itself ends the iterator like any failed run. A run that fails while the
        durable task has retries left publishes no error, as the retry's run publishes the outcome. If the durable
        task catches that error instead, the iterator goes on to the task's next agent run, if any. A durable task that Hatchet
        cancels publishes no terminal event, because Hatchet stops a task it evicts the same way and an evicted task's
        run continues once it's restored. A consumer that has to notice a cancelled run checks the workflow run's
        status, e.g. with `hatchet.runs.aio_get_status(workflow_run_id)`.

        Args:
            workflow_run_id: The workflow run ID of the durable task running the agent.
            output_type: The agent's output type, so the terminal event's result is decoded into it.
            cursor: The `cursor` of an earlier iterator, to resume right after the last event it read. The new
                iterator doesn't know which events the earlier one yielded, so an event a replayed durable task
                published again after that cursor is yielded again.
        """
        config = self._event_stream
        if config is None:
            raise UserError(
                'This `HatchetDurability` has no `event_stream`, so its runs publish no events to subscribe to. '
                'Pass `event_stream=True` to publish them.'
            )
        topic = config.topic(self._stream_agent_name(), workflow_run_id)

        def subscribe(after: str | None) -> AsyncIterator[StreamEvent]:
            return self.hatchet.streams.aio_subscribe(topic, namespace=config.namespace, cursor=after)

        return HatchetAgentRunEvents(subscribe, output_type=output_type, cursor=cursor)

    def _stream_agent_name(self) -> str:
        """The agent whose topic `stream_agent_events` reads: this capability's, once it's attached to one agent."""
        if self.name:
            return self.name
        if len(self._bound_agent_names) == 1:
            return self._bound_agent_names[0]
        if not self._bound_agent_names:
            raise UserError('`HatchetDurability` must be attached to an agent before its events can be streamed.')
        raise UserError(
            f"This `HatchetDurability` is attached to agents {self._bound_agent_names!r}, so it can't tell whose "
            'events to stream. Use a `HatchetDurability` per agent, or pass `name=` to it.'
        )


class _TerminalEventPublisher(AbstractCapability[AgentDepsT]):
    """Publishes the run's terminal event from the `outermost` tier, once every capability has shaped the result.

    `HatchetDurability` is `innermost`, so publishing from its own hooks would see a result other capabilities still
    transform, or a run they still fail.
    """

    def __init__(self, durability: HatchetDurability[AgentDepsT]) -> None:
        self._durability = durability

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Publish the run's result once every `after_run` has returned, or end subscribers' iteration if it fails.

        Wrapping the run sees a failure from any capability's hook, including an `after_run` that raises, which
        `on_run_error` doesn't, so a run is never published as both a success and a failure. Subscribers need the
        failure too, as no result event will come: a topic outlives its runs. A run's own cancellation, by its `timeout=` or `RunContext.cancel()`, reaches it as
        a `CancelledError` and ends the run with `RunTimedOut` or `RunCancelled`, so it's published as that. Hatchet
        cancelling or evicting the durable task isn't the run's own, and publishes nothing.
        """
        try:
            result = await handler()
        except Exception as error:
            await self._durability._publish_terminal_error(  # pyright: ignore[reportPrivateUsage]
                ctx, type(error).__name__, retryable=not isinstance(error, NonRetryableError)
            )
            raise
        except asyncio.CancelledError:
            cancellation = ctx._cancellation  # pyright: ignore[reportPrivateUsage]
            if cancellation is not None and cancellation.cancel_requested:
                error_name = RunTimedOut.__name__ if cancellation.timed_out else RunCancelled.__name__
                # Shielded, as the cancellation would otherwise interrupt the publish.
                with anyio.CancelScope(shield=True):
                    await self._durability._publish_terminal_error(ctx, error_name)  # pyright: ignore[reportPrivateUsage]
            raise
        try:
            await self._durability._publish_terminal_event(ctx, result)  # pyright: ignore[reportPrivateUsage]
        except Exception as error:
            # E.g. an output that can't be serialized: no result event will come, so end subscribers' iteration.
            await self._durability._publish_terminal_error(ctx, type(error).__name__)  # pyright: ignore[reportPrivateUsage]
            raise
        return result
