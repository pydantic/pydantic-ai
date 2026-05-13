"""Instrumentation capability for OpenTelemetry/Logfire tracing of agent runs."""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from opentelemetry.baggage import set_baggage as _otel_set_baggage
from opentelemetry.context import attach as _otel_attach, detach as _otel_detach
from opentelemetry.trace import SpanKind, StatusCode
from pydantic_core import to_json

from pydantic_ai._instrumentation import (
    GEN_AI_REQUEST_MODEL_ATTRIBUTE,
    GEN_AI_SYSTEM_ATTRIBUTE,
    MODEL_SETTING_ATTRIBUTES,
    CostCalculationFailedWarning,
    InstrumentationNames,
    annotate_tool_call_otel_metadata,
    build_tool_definitions,
    event_to_dict,
    get_agent_run_baggage_attributes,
    get_instructions,
    model_attributes,
    model_request_parameters_attributes,
    serialize_any,
)
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ToolRetryError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, tool_return_ta
from pydantic_ai.tools import ToolDefinition

from .abstract import (
    AbstractCapability,
    CapabilityOrdering,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapOutputProcessHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
)

if TYPE_CHECKING:
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.output import OutputContext
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.tools import AgentDepsT


def _default_settings() -> InstrumentationSettings:
    """Lazy import to avoid loading the OTel SDK eagerly at module import time."""
    from pydantic_ai.models.instrumented import InstrumentationSettings

    return InstrumentationSettings()


@dataclass
class Instrumentation(AbstractCapability[Any]):
    """Capability that instruments agent runs with OpenTelemetry/Logfire tracing.

    When added to an agent via `capabilities=[Instrumentation(...)]`, this capability
    creates OpenTelemetry spans for the agent run, model requests, and tool executions.

    Other capabilities can add attributes to these spans using either the OpenTelemetry API
    (`opentelemetry.trace.get_current_span().set_attribute(key, value)`) or the Logfire SDK
    (`logfire.current_span().set_attribute(key, value)`).
    """

    settings: InstrumentationSettings = field(default_factory=lambda: _default_settings())
    """OTel/Logfire instrumentation settings. Defaults to `InstrumentationSettings()`,
    which uses the global `TracerProvider`/`LoggerProvider` (typically configured by
    `logfire.configure()`)."""

    # Per-run state (set in `for_run`, mutated by `wrap_model_request`). `for_run`
    # returns a shallow copy via `replace(self)` for per-run isolation. These fields
    # are updated as the run progresses and assume sequential model requests within
    # a run — if the agent loop ever issues concurrent model requests, accesses to
    # these fields would race.
    _agent_name: str = field(default='agent', repr=False, init=False)
    _new_message_index: int = field(default=0, repr=False, init=False)
    _last_messages: list[ModelMessage] | None = field(default=None, repr=False, init=False)
    _last_model_request_parameters: ModelRequestParameters | None = field(default=None, repr=False, init=False)
    # Resolved once from `self.settings.version` in `__post_init__` and preserved across
    # `dataclasses.replace` calls in `for_run` (which only touches init=True fields).
    _instrumentation_names: InstrumentationNames = field(
        default_factory=lambda: InstrumentationNames.for_version(2), repr=False, init=False
    )

    def __post_init__(self) -> None:
        self._instrumentation_names = InstrumentationNames.for_version(self.settings.version)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def from_spec(cls, **kwargs: Any) -> Instrumentation:
        """Build an `Instrumentation` capability from a YAML/JSON spec.

        Accepts the serializable subset of [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]
        kwargs (`include_binary_content`, `include_content`, `version`, `event_mode`,
        `use_aggregated_usage_attribute_names`). The OTel `tracer_provider`, `meter_provider`,
        and `logger_provider` fields can't be expressed in YAML and default to the global
        providers (typically configured via `logfire.configure()`).

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
        """Return a fresh copy for per-run state isolation."""
        inst = replace(self)
        inst._agent_name = (ctx.agent.name if ctx.agent else None) or 'agent'
        inst._new_message_index = len(ctx.messages)
        return inst

    # ------------------------------------------------------------------
    # wrap_run — agent run span
    # ------------------------------------------------------------------

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        settings = self.settings
        names = self._instrumentation_names
        agent_name = self._agent_name

        span_attributes: dict[str, Any] = {
            'model_name': ctx.model.model_name if ctx.model else 'no-model',
            'agent_name': agent_name,
            'gen_ai.agent.name': agent_name,
            'gen_ai.agent.call.id': ctx.run_id or '',
            'gen_ai.conversation.id': ctx.conversation_id or '',
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': f'{agent_name} run',
        }

        if ctx.agent is not None:  # pragma: no branch
            rendered = ctx.agent.render_description(ctx.deps)
            if rendered is not None:
                span_attributes['gen_ai.agent.description'] = rendered

        with settings.tracer.start_as_current_span(
            names.get_agent_run_span_name(agent_name),
            attributes=span_attributes,
        ) as span:
            otel_ctx = _otel_set_baggage('gen_ai.agent.name', agent_name)
            otel_ctx = _otel_set_baggage('gen_ai.agent.call.id', ctx.run_id or '', context=otel_ctx)
            otel_ctx = _otel_set_baggage('gen_ai.conversation.id', ctx.conversation_id or '', context=otel_ctx)
            token = _otel_attach(otel_ctx)
            result: AgentRunResult[Any] | None = None
            try:
                result = await handler()

                if settings.include_content and span.is_recording():
                    span.set_attribute(
                        'final_result',
                        (
                            result.output
                            if isinstance(result.output, str)
                            else to_json(serialize_any(result.output)).decode()
                        ),
                    )

                return result
            finally:
                _otel_detach(token)
                if span.is_recording():
                    # Get current messages and metadata from the result (which holds the up-to-date state).
                    # ctx.messages/ctx.metadata may be stale because the run state is mutated during execution.
                    if result is not None:
                        message_history = result.all_messages()
                        metadata = result.metadata
                    else:
                        # On error, use the last messages seen during model requests.
                        message_history = self._last_messages or ctx.messages
                        metadata = ctx.metadata
                    span.set_attributes(self._run_span_end_attributes(ctx, message_history, metadata))

    def _run_span_end_attributes(
        self,
        ctx: RunContext[Any],
        message_history: list[ModelMessage],
        metadata: dict[str, Any] | None,
    ) -> dict[str, str | int | float | bool]:
        """Compute the end-of-run span attributes."""
        settings = self.settings
        new_message_index = self._new_message_index

        if settings.version == 1:
            attrs: dict[str, Any] = {
                'all_messages_events': to_json(
                    [event_to_dict(e) for e in settings.messages_to_otel_events(message_history)]
                ).decode()
            }
        else:
            last_instructions = get_instructions(message_history, self._last_model_request_parameters)
            attrs = {
                'pydantic_ai.all_messages': to_json(settings.messages_to_otel_messages(list(message_history))).decode(),
                **settings.system_instructions_attributes(last_instructions),
            }

            if new_message_index > 0:
                attrs['pydantic_ai.new_message_index'] = new_message_index

            if any(
                (isinstance(m, ModelRequest) and m.instructions is not None and m.instructions != last_instructions)
                for m in message_history[new_message_index:]
            ):
                attrs['pydantic_ai.variable_instructions'] = True

        if metadata is not None:
            attrs['metadata'] = to_json(serialize_any(metadata)).decode()

        usage_attrs = (
            {
                k.replace('gen_ai.usage.', 'gen_ai.aggregated_usage.', 1): v
                for k, v in ctx.usage.opentelemetry_attributes().items()
            }
            if settings.use_aggregated_usage_attribute_names
            else ctx.usage.opentelemetry_attributes()
        )

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
        settings = self.settings
        model = request_context.model

        # Track the latest messages so _run_span_end_attributes has them on error paths
        # (ctx.messages may be stale because UserPromptNode replaces the list reference).
        self._last_messages = request_context.messages

        prepared_settings, prepared_parameters = model.prepare_request(
            request_context.model_settings,
            request_context.model_request_parameters,
        )
        # Stash for `_run_span_end_attributes`: feeding the parameters into
        # `get_instructions` lets it use the canonical `instruction_parts` source
        # (which includes prompted-output template instructions and is properly sorted)
        # instead of falling back to reading `ModelRequest.instructions` from history.
        self._last_model_request_parameters = prepared_parameters

        operation = 'chat'
        span_name = f'{operation} {model.model_name}'
        attributes: dict[str, Any] = {
            'gen_ai.operation.name': operation,
            **model_attributes(model),
            **model_request_parameters_attributes(prepared_parameters),
            **get_agent_run_baggage_attributes(),
            'logfire.json_schema': to_json(
                {
                    'type': 'object',
                    'properties': {'model_request_parameters': {'type': 'object'}},
                }
            ).decode(),
        }

        tool_definitions = build_tool_definitions(prepared_parameters)
        if tool_definitions:
            attributes['gen_ai.tool.definitions'] = to_json(tool_definitions).decode()

        if prepared_settings:
            for key in MODEL_SETTING_ATTRIBUTES:
                if isinstance(value := prepared_settings.get(key), float | int):
                    attributes[f'gen_ai.request.{key}'] = value

        record_metrics: Callable[[], None] | None = None
        try:
            with settings.tracer.start_as_current_span(span_name, attributes=attributes, kind=SpanKind.CLIENT) as span:
                # `finish` is a closure rather than inline so we can (a) set result attributes
                # inside the `with span:` block — they attach to the span — and (b) call the
                # captured `record_metrics` in the outer `finally` AFTER the span closes,
                # so observability backends that aggregate metrics from span attributes
                # don't double-count.
                def finish(response: ModelResponse) -> None:
                    nonlocal record_metrics

                    # FallbackModel updates these span attributes via get_current_span().
                    attributes.update(getattr(span, 'attributes', {}))
                    request_model = attributes[GEN_AI_REQUEST_MODEL_ATTRIBUTE]
                    system = cast(str, attributes[GEN_AI_SYSTEM_ATTRIBUTE])

                    response_model = response.model_name or request_model
                    price_calculation = None

                    def _record_metrics() -> None:
                        metric_attributes = {
                            'gen_ai.provider.name': system,
                            'gen_ai.system': system,
                            'gen_ai.operation.name': operation,
                            'gen_ai.request.model': request_model,
                            'gen_ai.response.model': response_model,
                        }
                        settings.record_metrics(response, price_calculation, metric_attributes)

                    record_metrics = _record_metrics

                    # Compute cost before the `is_recording()` gate so `_record_metrics`
                    # always emits cost data, even when the span is dropped by sampling.
                    try:
                        price_calculation = response.cost()
                    except LookupError:
                        pass
                    except Exception as e:  # pragma: no cover — safety net for unexpected genai_prices errors
                        warnings.warn(
                            f'Failed to get cost from response: {type(e).__name__}: {e}',
                            CostCalculationFailedWarning,
                        )

                    if not span.is_recording():
                        return

                    settings.handle_messages(request_context.messages, response, system, span, prepared_parameters)

                    attributes_to_set: dict[str, Any] = {
                        **response.usage.opentelemetry_attributes(),
                        'gen_ai.response.model': response_model,
                    }
                    if price_calculation is not None:
                        attributes_to_set['operation.cost'] = float(price_calculation.total_price)

                    if response.provider_response_id is not None:
                        attributes_to_set['gen_ai.response.id'] = response.provider_response_id
                    if response.finish_reason is not None:
                        attributes_to_set['gen_ai.response.finish_reasons'] = [response.finish_reason]
                    span.set_attributes(attributes_to_set)
                    span.update_name(f'{operation} {request_model}')

                response = await handler(request_context)
                annotate_tool_call_otel_metadata(response, prepared_parameters)
                finish(response)
                return response
        finally:
            if record_metrics:
                record_metrics()

    # ------------------------------------------------------------------
    # wrap_tool_execute — tool execution span
    # ------------------------------------------------------------------

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
                    try:
                        metadata_str = to_json(exc.metadata).decode()
                    except (TypeError, ValueError):
                        metadata_str = repr(exc.metadata)
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
        return await self._run_tool_span(
            span_name=self._instrumentation_names.get_tool_span_name(call.tool_name),
            attributes=self._tool_span_attributes(call),
            action=lambda: handler(args),
            serialize_result=lambda value: tool_return_ta.dump_json(value).decode(),
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
            attributes[names.tool_arguments_attr] = to_json(output).decode()

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
            serialize_result=lambda value: to_json(serialize_any(value)).decode(),
        )
