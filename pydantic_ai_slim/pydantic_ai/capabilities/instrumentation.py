"""Instrumentation capability for OpenTelemetry/Logfire tracing of agent runs."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from opentelemetry.baggage import set_baggage as _otel_set_baggage
from opentelemetry.context import attach as _otel_attach, detach as _otel_detach
from opentelemetry.trace import SpanKind, StatusCode

from pydantic_ai._instrumentation import InstrumentationNames, get_agent_run_baggage_attributes
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ToolRetryError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
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
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.output import OutputContext
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.tools import AgentDepsT


@dataclass
class Instrumentation(AbstractCapability[Any]):
    """Capability that instruments agent runs with OpenTelemetry/Logfire tracing.

    When added to an agent (either explicitly via `capabilities=[Instrumentation(...)]`
    or implicitly via `Agent(instrument=True)`), this capability creates OpenTelemetry
    spans for the agent run, model requests, and tool executions.

    Other capabilities can add attributes to these spans using the standard OpenTelemetry
    API: `opentelemetry.trace.get_current_span().set_attribute(key, value)`.
    """

    settings: InstrumentationSettings

    # Per-run state (set in for_run, not user-provided)
    _agent_name: str = field(default='agent', repr=False, init=False)
    _new_message_index: int = field(default=0, repr=False, init=False)
    _last_messages: list[ModelMessage] | None = field(default=None, repr=False, init=False)
    _instrumentation_names: InstrumentationNames = field(
        default_factory=lambda: InstrumentationNames.for_version(2), repr=False, init=False
    )

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # InstrumentationSettings takes non-serializable providers (TracerProvider, etc.),
        # so this capability is not constructible from YAML/JSON specs.
        return None

    async def for_run(self, ctx: RunContext[Any]) -> Instrumentation:
        """Return a fresh copy for per-run state isolation."""
        inst = replace(self)
        inst._agent_name = (ctx.agent.name if ctx.agent else None) or 'agent'
        inst._new_message_index = len(ctx.messages)
        inst._instrumentation_names = InstrumentationNames.for_version(self.settings.version)
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
        from pydantic_ai.models.instrumented import InstrumentedModel

        settings = self.settings
        names = self._instrumentation_names
        agent_name = self._agent_name

        span_attributes: dict[str, Any] = {
            'model_name': ctx.model.model_name if ctx.model else 'no-model',
            'agent_name': agent_name,
            'gen_ai.agent.name': agent_name,
            'gen_ai.agent.call.id': ctx.run_id or '',
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
                            else json.dumps(InstrumentedModel.serialize_any(result.output))
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
                        metadata = result._state.metadata  # pyright: ignore[reportPrivateUsage]
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
        from pydantic_ai.messages import ModelRequest
        from pydantic_ai.models.instrumented import InstrumentedModel

        settings = self.settings
        new_message_index = self._new_message_index

        if settings.version == 1:
            attrs: dict[str, Any] = {
                'all_messages_events': json.dumps(
                    [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(message_history)]
                )
            }
        else:
            last_instructions = InstrumentedModel._get_instructions(message_history)  # pyright: ignore[reportPrivateUsage]
            attrs = {
                'pydantic_ai.all_messages': json.dumps(settings.messages_to_otel_messages(list(message_history))),
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
            attrs['metadata'] = json.dumps(InstrumentedModel.serialize_any(metadata))

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
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {
                        **{k: {'type': 'array'} if isinstance(v, str) else {} for k, v in attrs.items()},
                        'final_result': {'type': 'object'},
                    },
                }
            ),
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
        from pydantic_ai.models.instrumented import (
            GEN_AI_REQUEST_MODEL_ATTRIBUTE,
            GEN_AI_SYSTEM_ATTRIBUTE,
            MODEL_SETTING_ATTRIBUTES,
            CostCalculationFailedWarning,
            InstrumentedModel,
            build_tool_definitions,
        )

        settings = self.settings
        model = request_context.model

        # Unwrap InstrumentedModel to prevent double-spanning
        if isinstance(model, InstrumentedModel):
            model = model.wrapped
            request_context = replace(request_context, model=model)

        # Track the latest messages so _run_span_end_attributes has them on error paths
        # (ctx.messages may be stale because UserPromptNode replaces the list reference).
        self._last_messages = request_context.messages

        prepared_settings, prepared_parameters = model.prepare_request(
            request_context.model_settings,
            request_context.model_request_parameters,
        )

        operation = 'chat'
        span_name = f'{operation} {model.model_name}'
        attributes: dict[str, Any] = {
            'gen_ai.operation.name': operation,
            **InstrumentedModel.model_attributes(model),
            **InstrumentedModel.model_request_parameters_attributes(prepared_parameters),
            **get_agent_run_baggage_attributes(),
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {'model_request_parameters': {'type': 'object'}},
                }
            ),
        }

        tool_definitions = build_tool_definitions(prepared_parameters)
        if tool_definitions:
            attributes['gen_ai.tool.definitions'] = json.dumps(tool_definitions)

        if prepared_settings:
            for key in MODEL_SETTING_ATTRIBUTES:
                if isinstance(value := prepared_settings.get(key), float | int):
                    attributes[f'gen_ai.request.{key}'] = value

        record_metrics: Callable[[], None] | None = None
        try:
            with settings.tracer.start_as_current_span(span_name, attributes=attributes, kind=SpanKind.CLIENT) as span:

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

                    if not span.is_recording():
                        return

                    settings.handle_messages(request_context.messages, response, system, span, prepared_parameters)

                    attributes_to_set: dict[str, Any] = {
                        **response.usage.opentelemetry_attributes(),
                        'gen_ai.response.model': response_model,
                    }
                    try:
                        price_calculation = response.cost()
                    except LookupError:
                        pass
                    except Exception as e:  # pragma: no cover — safety net for unexpected genai_prices errors
                        warnings.warn(
                            f'Failed to get cost from response: {type(e).__name__}: {e}',
                            CostCalculationFailedWarning,
                        )
                    else:
                        attributes_to_set['operation.cost'] = float(price_calculation.total_price)

                    if response.provider_response_id is not None:
                        attributes_to_set['gen_ai.response.id'] = response.provider_response_id
                    if response.finish_reason is not None:
                        attributes_to_set['gen_ai.response.finish_reasons'] = [response.finish_reason]
                    span.set_attributes(attributes_to_set)
                    span.update_name(f'{operation} {request_model}')

                response = await handler(request_context)
                finish(response)
                return response
        finally:
            if record_metrics:
                record_metrics()

    # ------------------------------------------------------------------
    # wrap_tool_execute — tool execution span
    # ------------------------------------------------------------------

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        settings = self.settings
        names = self._instrumentation_names
        include_content = settings.include_content

        span_attributes: dict[str, Any] = {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': call.tool_name,
            'gen_ai.tool.call.id': call.tool_call_id,
            **({names.tool_arguments_attr: call.args_as_json_str()} if include_content else {}),
            **get_agent_run_baggage_attributes(),
            'logfire.msg': f'running tool: {call.tool_name}',
            'logfire.json_schema': json.dumps(
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
            ),
        }

        with settings.tracer.start_as_current_span(
            names.get_tool_span_name(call.tool_name),
            attributes=span_attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                tool_result = await handler(args)
                if include_content and span.is_recording():
                    from pydantic_ai import messages as _messages

                    span.set_attribute(
                        names.tool_result_attr,
                        tool_result
                        if isinstance(tool_result, str)
                        else _messages.tool_return_ta.dump_json(tool_result).decode(),
                    )
            except (CallDeferred, ApprovalRequired) as exc:
                span.set_attribute(names.tool_deferral_name_attr, type(exc).__name__)
                if include_content and span.is_recording() and exc.metadata is not None:
                    try:
                        metadata_str = json.dumps(exc.metadata)
                    except (TypeError, ValueError):
                        metadata_str = repr(exc.metadata)
                    span.set_attribute(names.tool_deferral_metadata_attr, metadata_str)
                if settings.version < 5:
                    span.record_exception(exc, escaped=True)
                    span.set_status(StatusCode.ERROR)
                raise
            except ToolRetryError as e:
                part = e.tool_retry
                if include_content and span.is_recording():
                    span.set_attribute(names.tool_result_attr, part.model_response())
                span.record_exception(e, escaped=True)
                span.set_status(StatusCode.ERROR)
                raise
            except BaseException as e:
                span.record_exception(e, escaped=True)
                span.set_status(StatusCode.ERROR)
                raise

        return tool_result

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
        """Emit an `execute_tool` span for tool-mode output processing.

        Output tools no longer go through `wrap_tool_execute`, so this hook restores
        the outer "execute_tool {tool_name}" span that wraps the validated final-result
        tool call. Output-function spans (emitted by `execute_output_function`) nest
        inside this when present.

        For non-tool output (text, native, prompted, image), no span is created here —
        function execution still gets its own span via `execute_output_function`, and
        plain text/image passthrough has nothing meaningful to span.
        """
        tool_call = output_context.tool_call
        if tool_call is None:
            return await handler(output)

        from pydantic_ai.models.instrumented import InstrumentedModel

        settings = self.settings
        names = self._instrumentation_names
        include_content = settings.include_content

        span_attributes: dict[str, Any] = {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': tool_call.tool_name,
            'gen_ai.tool.call.id': tool_call.tool_call_id,
            **({names.tool_arguments_attr: tool_call.args_as_json_str()} if include_content else {}),
            **get_agent_run_baggage_attributes(),
            'logfire.msg': f'running tool: {tool_call.tool_name}',
            'logfire.json_schema': json.dumps(
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
            ),
        }

        with settings.tracer.start_as_current_span(
            names.get_tool_span_name(tool_call.tool_name),
            attributes=span_attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await handler(output)
            except BaseException as e:
                # `do_process` (passed as `handler` by `run_output_process_hooks`) raises
                # `ModelRetry` (or other exceptions) — `ToolRetryError` is only synthesized
                # by `run_output_process_hooks`'s outer wrapper, after this span has closed.
                span.record_exception(e, escaped=True)
                span.set_status(StatusCode.ERROR)
                raise
            if include_content and span.is_recording():
                span.set_attribute(
                    names.tool_result_attr,
                    result if isinstance(result, str) else json.dumps(InstrumentedModel.serialize_any(result)),
                )

        return result
