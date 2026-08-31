"""Instrumentation capability for OpenTelemetry/Logfire tracing of agent runs."""

from __future__ import annotations

import sys
import warnings
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar

from opentelemetry.baggage import get_all as _otel_get_all_baggage, set_baggage as _otel_set_baggage
from opentelemetry.context import attach as _otel_attach, detach as _otel_detach
from opentelemetry.trace import StatusCode
from pydantic_core import ValidationError, to_json

from pydantic_ai._instrumentation import (
    DEFAULT_INSTRUMENTATION_VERSION,
    InstrumentationNames,
    MessageJsonCache,
    get_agent_run_baggage_attributes,
    get_instructions,
    has_stale_message_json,
    open_model_request_span,
    redact_binary_content,
    safe_to_json,
    serialize_any,
    time_to_first_chunk_ctx,
)
from pydantic_ai._run_context import get_run_state_key
from pydantic_ai._utils import UNSET, Unset
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    MessageHistoryMutatedWarning,
    ModelRetry,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, ToolCallPart, tool_return_ta
from pydantic_ai.tools import ToolDefinition

from .abstract import (
    AbstractCapability,
    CapabilityOrdering,
    RawToolArgs,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapOutputProcessHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from pydantic_ai._run_context import RunContext, RunPreparationContext
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.output import OutputContext
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.tools import AgentDepsT
    from pydantic_ai.usage import RunUsage


def _default_settings() -> InstrumentationSettings:
    """Lazy import to avoid loading the OTel SDK eagerly at module import time."""
    from pydantic_ai.models.instrumented import InstrumentationSettings

    return InstrumentationSettings()


@dataclass
class _RunState:
    new_message_index: int
    run_span: Span
    baggage_at_span_start: Mapping[str, object]
    """OTel baggage active when the run span opened, i.e. what a baggage span processor already
    recorded on it. `wrap_run` mirrors what has been attached since (see
    `_tag_run_span_with_late_baggage`)."""
    metadata: dict[str, Any] | None = None
    last_result: AgentRunResult[Any] | None = None
    last_messages: list[ModelMessage] | None = None
    last_model_request_parameters: ModelRequestParameters | None = None
    last_formatted_instructions: str | None | Unset = UNSET
    variable_instructions: bool = False
    message_json_cache: MessageJsonCache = field(default_factory=MessageJsonCache)
    """Per-run cache of input messages' serialized OTel JSON fragments (see `MessageJsonCache`)."""


@dataclass
class Instrumentation(AbstractCapability[Any]):
    """Capability that instruments agent runs with OpenTelemetry/Logfire tracing.

    When added to an agent via `capabilities=[Instrumentation(...)]`, this capability
    creates OpenTelemetry spans for the agent run, model requests, and tool executions.

    An `Instrumentation` capability materialized only during `for_run` can instrument model
    requests and tools, but cannot create the agent-run span because the whole-run hook has
    already been entered. Configure it on the agent-level capability tree for full tracing.

    Other capabilities can add attributes to these spans using the OpenTelemetry API
    (`opentelemetry.trace.get_current_span().set_attribute(key, value)`).
    """

    _safe_at_runtime: ClassVar[bool] = True
    """Workflow-side only — no toolsets, native tools, or model wrapping introduced — so safe
    to attach per-run even when a durability capability is bound. Internal flag read by the
    bundled durable-execution integrations.
    """

    settings: InstrumentationSettings = field(default_factory=lambda: _default_settings())
    """OTel/Logfire instrumentation settings. Defaults to `InstrumentationSettings()`,
    which uses the global `TracerProvider` (typically configured by `logfire.configure()`)."""

    _runs: dict[object, _RunState] = field(default_factory=dict[object, _RunState], repr=False, init=False)
    """Per-run state shared by the agent-level instance and its `for_run` copies."""
    # Resolved from `self.settings.version` whenever `__post_init__` runs, including on
    # the per-run copy created by `dataclasses.replace`.
    _instrumentation_names: InstrumentationNames = field(
        default_factory=lambda: InstrumentationNames.for_version(DEFAULT_INSTRUMENTATION_VERSION),
        repr=False,
        init=False,
    )
    _fallback_message_json_cache: MessageJsonCache | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self._instrumentation_names = InstrumentationNames.for_version(self.settings.version)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def from_spec(cls, **kwargs: Any) -> Instrumentation:
        """Build an `Instrumentation` capability from a YAML/JSON spec.

        Accepts the serializable subset of [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]
        kwargs (`include_binary_content`, `include_content`, `version`,
        `use_aggregated_usage_attribute_names`). The OTel `tracer_provider` and `meter_provider`
        fields can't be expressed in YAML and default to the global providers (typically configured
        via `logfire.configure()`).

        YAML form:

            capabilities:
              - Instrumentation: {}                # default settings
              - Instrumentation:
                  version: 2
                  include_content: false
        """
        from pydantic_ai.models.instrumented import InstrumentationSettings

        return cls(settings=InstrumentationSettings(**kwargs))

    async def for_run(self, ctx: RunContext[Any]) -> Instrumentation:
        """Return a fresh copy for per-run state isolation and record the resolved run context."""
        inst = replace(self)
        inst._runs = self._runs
        inst._record_run_context(ctx)
        return inst

    # ------------------------------------------------------------------
    # wrap_entire_run — agent run span
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def wrap_entire_run(self, ctx: RunPreparationContext[AgentDepsT]) -> AsyncGenerator[None]:
        """Keep resource setup, execution, recovery, and teardown inside the agent-run span.

        A realtime session needs no guard here: it dispatches only the four run hooks, never
        `wrap_entire_run`, so it never opens a second run span alongside the session's own
        canonical `invoke_agent` span. See the capability-owned span direction documented in
        `realtime/_session.py`.
        """
        settings = self.settings
        names = self._instrumentation_names
        agent_name = ctx.agent.name or 'agent'

        span_attributes: dict[str, Any] = {
            'model_name': ctx.model.model_name if ctx.model else 'no-model',
            'agent_name': agent_name,
            'gen_ai.agent.name': agent_name,
            'gen_ai.agent.call.id': ctx.run_id,
            'gen_ai.conversation.id': ctx.conversation_id,
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': f'{agent_name} run',
        }

        rendered = ctx.agent.render_description(ctx.deps)
        if rendered is not None:
            span_attributes['gen_ai.agent.description'] = rendered

        with settings.tracer.start_as_current_span(
            names.get_agent_run_span_name(agent_name),
            attributes=span_attributes,
        ) as span:
            otel_ctx = _otel_set_baggage('gen_ai.agent.name', agent_name)
            otel_ctx = _otel_set_baggage('gen_ai.agent.call.id', ctx.run_id, context=otel_ctx)
            otel_ctx = _otel_set_baggage('gen_ai.conversation.id', ctx.conversation_id, context=otel_ctx)
            token = _otel_attach(otel_ctx)
            run_state = _RunState(
                new_message_index=len(ctx.messages),
                run_span=span,
                baggage_at_span_start=_otel_get_all_baggage(),
            )
            self._runs[get_run_state_key(ctx)] = run_state
            try:
                yield
            finally:
                _otel_detach(token)
                self._runs.pop(get_run_state_key(ctx), None)
                if span.is_recording():
                    # Best effort: this runs while the exit stack unwinds, where a raised
                    # exception would mask the run's own error. Telemetry must never do that.
                    active_error = sys.exc_info()[1]
                    try:
                        result = run_state.last_result
                        if result is not None:
                            message_history, metadata = result.all_messages(), result.metadata
                        else:
                            message_history, metadata = run_state.last_messages or ctx.messages, run_state.metadata
                        span.set_attributes(
                            self._run_span_end_attributes(ctx.usage, run_state, message_history, metadata)
                        )
                    except Exception as attribute_error:
                        if active_error is None:
                            warnings.warn(
                                f'Failed to record agent run span attributes: {attribute_error!r}',
                                RuntimeWarning,
                                stacklevel=1,
                            )
                    if active_error is None and run_state.last_result is not None:
                        # One O(history) pass per run: turn any silent staleness the per-request
                        # fragment cache may have recorded into a loud signal. Skipped when the run
                        # errored: with warnings configured as errors, warning here in the `finally`
                        # would displace the propagating run exception.
                        if run_state.message_json_cache and has_stale_message_json(
                            settings, run_state.last_result.all_messages(), run_state.message_json_cache
                        ):
                            warnings.warn(
                                'In-place mutation of messages already in the history was detected during this run: '
                                "the `gen_ai.input.messages` attribute recorded on the run's model request spans may "
                                'not match the messages actually sent to the model. Mutating history messages in '
                                'place is not supported; build new message or part objects instead, e.g. via a '
                                'history processor.',
                                MessageHistoryMutatedWarning,
                            )

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Mirror onto the run span the baggage that was attached before this point in the chain.

        A capability that publishes OTel baggage around the run — a resolved prompt variant, a
        tenant id — expects every span of the run to carry it, which is what the span processor
        that copies baggage onto spans when they start does. The run span opens in
        `wrap_entire_run`, ahead of run preparation and the whole `wrap_run` chain, so by then
        there is nothing for the processor to copy and the run span would be the only span of the
        run missing the value.

        This is the seam the run span itself used to open at, so the entries mirrored here are
        exactly the ones the processor used to record: whatever run preparation and the
        capabilities that wrap this one have attached, and not the baggage of the capabilities
        this one wraps, which is attached further in.
        """
        run_state = self._run_state(ctx)
        if run_state is not None:
            self._tag_run_span_with_late_baggage(run_state)
        return await handler()

    def _tag_run_span_with_late_baggage(self, run_state: _RunState) -> None:
        """Set the baggage attached since the run span opened as attributes on it.

        Only entries added (or changed) since then are written: the ones already there were
        recorded by the processor itself, including whatever conflict handling it applies.
        """
        span = run_state.run_span
        if not span.is_recording():
            return
        at_start = run_state.baggage_at_span_start
        attributes = {
            key: value
            for key, value in _otel_get_all_baggage().items()
            # Non-string baggage isn't a valid attribute value; the processor skips it too.
            if isinstance(value, str) and at_start.get(key) != value
        }
        if attributes:
            span.set_attributes(attributes)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Record the resolved model once run assembly is complete."""
        self._record_run_context(ctx)

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Run last (outermost capability, reversed dispatch) to capture replacements and recoveries."""
        run_state = self._run_state(ctx)
        if run_state is None:
            return result
        run_state.last_result = result
        span = run_state.run_span
        if self.settings.include_content and span.is_recording():
            span.set_attribute(
                'final_result',
                result.output
                if isinstance(result.output, str)
                else safe_to_json(serialize_any(redact_binary_content(result.output, self.settings))).decode(),
            )
        return result

    def _run_span_end_attributes(
        self,
        usage: RunUsage,
        run_state: _RunState,
        message_history: list[ModelMessage],
        metadata: dict[str, Any] | None,
    ) -> dict[str, str | int | float | bool]:
        """Compute the end-of-run span attributes."""
        settings = self.settings
        new_message_index = run_state.new_message_index

        last_instructions = get_instructions(message_history, run_state.last_model_request_parameters)
        attrs: dict[str, Any] = {
            'pydantic_ai.all_messages': safe_to_json(
                settings.messages_to_otel_messages(list(message_history))
            ).decode(),
            **settings.system_instructions_attributes(last_instructions),
        }

        if new_message_index > 0:
            attrs['pydantic_ai.new_message_index'] = new_message_index

        if run_state.variable_instructions:
            attrs['pydantic_ai.variable_instructions'] = True

        if metadata is not None:
            attrs['metadata'] = safe_to_json(serialize_any(redact_binary_content(metadata, settings))).decode()

        usage_attrs = settings.aggregated_usage_attributes(usage)

        return {
            **usage_attrs,
            **attrs,
            'logfire.json_schema': to_json(
                {
                    'type': 'object',
                    'properties': {
                        **{k: {'type': 'array'} if isinstance(v, str) else {} for k, v in attrs.items()},
                        'final_result': {'type': 'object'},
                    },
                }
            ).decode(),
        }

    def _run_state(self, ctx: RunContext[Any]) -> _RunState | None:
        return self._runs.get(get_run_state_key(ctx))

    def _record_run_context(self, ctx: RunContext[Any]) -> None:
        run_state = self._run_state(ctx)
        if run_state is None:
            return
        run_state.metadata = ctx.metadata
        run_state.run_span.set_attribute('model_name', ctx.model.model_name)

    # ------------------------------------------------------------------
    # wrap_model_request — model request span
    # ------------------------------------------------------------------

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        # Track the latest messages so _run_span_end_attributes has them on error paths
        # (ctx.messages may be stale because UserPromptNode replaces the list reference).
        run_state = self._run_state(ctx)
        if run_state is not None:
            run_state.last_messages = request_context.messages

        if run_state is not None:
            message_json_cache = run_state.message_json_cache
        else:
            message_json_cache = self._fallback_message_json_cache = (
                self._fallback_message_json_cache or MessageJsonCache()
            )
        with open_model_request_span(self.settings, request_context, message_json_cache=message_json_cache) as (
            finish,
            prepared_request_context,
        ):
            if run_state is not None:
                # Stash for `_run_span_end_attributes`: feeding the parameters into
                # `get_instructions` lets it use the canonical `instruction_parts` source
                # (which includes prompted-output template instructions and is properly sorted)
                # instead of falling back to reading `ModelRequest.instructions` from history.
                run_state.last_model_request_parameters = prepared_request_context.model_request_parameters

                # Track whether the fully formatted instructions (including prompted-output schemas) vary across requests.
                # This does an apples-to-apples comparison of the final payload sent to the model.
                current_instructions = get_instructions(
                    request_context.messages, prepared_request_context.model_request_parameters
                )
                if not isinstance(run_state.last_formatted_instructions, Unset):
                    if current_instructions != run_state.last_formatted_instructions:
                        run_state.variable_instructions = True
                run_state.last_formatted_instructions = current_instructions

            response = await handler(request_context)
            # For streaming requests, the agent graph's handler reports TTFT through
            # `time_to_first_chunk_ctx` (set in the same task, so the value is visible here);
            # for non-streaming requests this reads the `None` default.
            finish(response, time_to_first_chunk=time_to_first_chunk_ctx.get())
            return response

    # ------------------------------------------------------------------
    # wrap_tool_execute — tool execution span
    # ------------------------------------------------------------------

    async def on_tool_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        error: ValidationError | ModelRetry,
    ) -> ValidatedToolArgs:
        """Emit an error span for a tool call whose argument validation failed.

        Runs only after every other capability has declined to recover the error, so a
        recovered validation failure produces no span. The span keeps the `execute_tool`
        operation name so tracing backends group it with other tool spans, and sets
        `pydantic_ai.tool.failure_stage: 'validation'` to distinguish it from execution
        failures.

        With content capture enabled, the span records the retry prompt built from the
        error as the tool result. That is the exact message the model receives when the
        agent loop handles the failure; raw-mode callers (e.g. sandboxed dispatch via
        `handle_call(wrap_validation_errors=False)`) surface the raw exception to the
        calling code instead, and the recorded prompt is just the rendered description
        of the failure.
        """
        names = self._instrumentation_names
        attributes = self._tool_span_attributes(call)
        # The tool never ran: keep the `execute_tool` operation name so backends find the
        # span, but say so in the message and mark the failure stage for querying.
        attributes['logfire.msg'] = f'invalid tool call: {call.tool_name}'
        attributes[names.tool_failure_stage_attr] = 'validation'
        with self.settings.tracer.start_as_current_span(
            names.get_tool_span_name(call.tool_name),
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            if self.settings.include_content and span.is_recording():
                retry = RetryPromptPart.from_error(error, tool_name=call.tool_name, tool_call_id=call.tool_call_id)
                span.set_attribute(names.tool_result_attr, retry.model_response())
                span.record_exception(error, escaped=True)
            else:
                # Validation errors may contain rejected arguments, so omit their message and
                # stack trace when content capture is disabled. Execution spans keep their
                # existing exception recording behavior. The type formatting must match what
                # the OTel SDK's `Span.record_exception` would have produced for this error.
                error_type = type(error)
                type_name = (
                    f'{error_type.__module__}.{error_type.__qualname__}'
                    if error_type.__module__ != 'builtins'
                    else error_type.__qualname__
                )
                span.add_event(
                    'exception',
                    attributes={'exception.type': type_name, 'exception.escaped': True},
                )
            span.set_status(StatusCode.ERROR)
        raise error

    def _tool_span_attributes(self, call: ToolCallPart) -> dict[str, Any]:
        """Build the span attributes shared by `wrap_tool_execute` and `wrap_output_process`.

        Both spans use `gen_ai.operation.name='execute_tool'` and the same `gen_ai.tool.*`
        attributes — they only differ in how the result is serialized and which exceptions
        are special-cased, which stays in the call-site `try/except`.
        """
        names = self._instrumentation_names
        include_content = self.settings.include_content
        return {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': call.tool_name,
            'gen_ai.tool.call.id': call.tool_call_id,
            **({names.tool_arguments_attr: call.args_as_json_str()} if include_content else {}),
            **get_agent_run_baggage_attributes(),
            'logfire.msg': f'running tool: {call.tool_name}',
            'logfire.json_schema': to_json(
                {
                    'type': 'object',
                    'properties': {
                        **(
                            {
                                names.tool_arguments_attr: {'type': 'object'},
                                names.tool_result_attr: {'type': 'object'},
                            }
                            if include_content
                            else {}
                        ),
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                }
            ).decode(),
        }

    async def _run_tool_span(
        self,
        *,
        span_name: str,
        attributes: dict[str, Any],
        action: Callable[[], Awaitable[Any]],
        serialize_result: Callable[[Any], str],
        handle_tool_control_flow: bool = False,
    ) -> Any:
        """Open a `gen_ai`-flavoured tool/output span around `action`.

        Records the serialized result on success (when `include_content` is enabled and
        the span is recording), records the exception and sets status `ERROR` on failure.

        When `handle_tool_control_flow` is True, the helper additionally special-cases
        `CallDeferred`/`ApprovalRequired` (deferrals are control flow, not errors) and
        records `ToolRetryError`'s retry prompt as the tool result before re-raising.
        Output-function spans leave that flag off — `ToolRetryError` is treated as a
        plain error there because the retry prompt is recorded on the surrounding
        request/agent spans, and `CallDeferred`/`ApprovalRequired` never reach output
        processing.
        """
        settings = self.settings
        names = self._instrumentation_names
        include_content = settings.include_content

        with settings.tracer.start_as_current_span(
            span_name,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await action()
            except (CallDeferred, ApprovalRequired) as exc:
                if not handle_tool_control_flow:
                    span.record_exception(exc, escaped=True)
                    span.set_status(StatusCode.ERROR)
                    raise
                # Deferrals are control flow, not errors: capture the deferral name (and
                # metadata when available) as span attributes, and only mark the span
                # ERROR for older instrumentation versions that expected that shape.
                span.set_attribute(names.tool_deferral_name_attr, type(exc).__name__)
                if include_content and span.is_recording() and exc.metadata is not None:
                    redacted_metadata = redact_binary_content(exc.metadata, settings)
                    try:
                        metadata_str = to_json(redacted_metadata).decode()
                    except (TypeError, ValueError):
                        metadata_str = repr(redacted_metadata)
                    span.set_attribute(names.tool_deferral_metadata_attr, metadata_str)
                if settings.version < 5:
                    span.record_exception(exc, escaped=True)
                    span.set_status(StatusCode.ERROR)
                raise
            except ToolRetryError as e:
                if handle_tool_control_flow and include_content and span.is_recording():
                    # Tool retries are surfaced as model-visible errors; record the prompt
                    # the model will see as the tool result before re-raising.
                    span.set_attribute(names.tool_result_attr, e.tool_retry.model_response())
                span.record_exception(e, escaped=True)
                span.set_status(StatusCode.ERROR)
                raise
            except ToolFailedError as e:
                if handle_tool_control_flow and include_content and span.is_recording():
                    span.set_attribute(names.tool_result_attr, e.tool_failed.model_response_str(wrap_if_error=False))
                span.record_exception(e, escaped=True)
                span.set_status(StatusCode.ERROR)
                raise
            except BaseException as e:
                span.record_exception(e, escaped=True)
                span.set_status(StatusCode.ERROR)
                raise

            if include_content and span.is_recording():
                span.set_attribute(
                    names.tool_result_attr,
                    result if isinstance(result, str) else serialize_result(result),
                )

        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        attributes = self._tool_span_attributes(call)
        if ctx.realtime:
            # Realtime spans all carry this marker (see `docs/realtime/observability.md`) so
            # backends can recognize the session tree; the tool span is shared with classic runs,
            # which stay unmarked.
            attributes['pydantic_ai.realtime'] = True
        return await self._run_tool_span(
            span_name=self._instrumentation_names.get_tool_span_name(call.tool_name),
            attributes=attributes,
            action=lambda: handler(args),
            serialize_result=lambda value: tool_return_ta.dump_json(
                redact_binary_content(value, self.settings)
            ).decode(),
            handle_tool_control_flow=True,
        )

    # ------------------------------------------------------------------
    # wrap_output_process — output tool execution span (tool-mode only)
    # ------------------------------------------------------------------

    async def wrap_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        handler: WrapOutputProcessHandler,
    ) -> Any:
        """Emit a span for output-function execution.

        Output processing for plain validation (no function) is not span-worthy — the
        validated value is the model's response itself, no user code ran. We open a
        span only when an output function will execute, regardless of whether the
        output arrived via a tool call. The span name reflects the function (or tool
        name when the function name is unavailable, e.g. union processors).
        """
        if not output_context.has_function:
            return await handler(output)

        names = self._instrumentation_names
        include_content = self.settings.include_content
        tool_call = output_context.tool_call
        # Tool-mode output: the registered tool name (e.g. `final_result`) is what the
        # model called, so use it as the span target. For non-tool output, fall back to
        # the function name (when known) or a generic placeholder.
        span_target = tool_call.tool_name if tool_call else (output_context.function_name or 'output_function')

        attributes: dict[str, Any] = {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': span_target,
            **get_agent_run_baggage_attributes(),
            'logfire.msg': f'running output function: {span_target}',
        }
        if tool_call is not None and tool_call.tool_call_id:
            attributes['gen_ai.tool.call.id'] = tool_call.tool_call_id
        if include_content:
            attributes[names.tool_arguments_attr] = safe_to_json(redact_binary_content(output, self.settings)).decode()

        attributes['logfire.json_schema'] = to_json(
            {
                'type': 'object',
                'properties': {
                    **(
                        {
                            names.tool_arguments_attr: {'type': 'object'},
                            names.tool_result_attr: {'type': 'object'},
                        }
                        if include_content
                        else {}
                    ),
                    'gen_ai.tool.name': {},
                    **({'gen_ai.tool.call.id': {}} if tool_call is not None and tool_call.tool_call_id else {}),
                },
            }
        ).decode()

        return await self._run_tool_span(
            span_name=names.get_output_tool_span_name(span_target),
            attributes=attributes,
            action=lambda: handler(output),
            serialize_result=lambda value: safe_to_json(
                serialize_any(redact_binary_content(value, self.settings))
            ).decode(),
        )
