"""Private OpenTelemetry support for realtime sessions.

`SessionInstrumentation` owns the OTel state and span construction for one
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession]: the session-wide `invoke_agent`/`realtime`
span, the per-response `chat` spans, and the instantaneous lifecycle spans. The session hands it the
static session metadata once at construction and delegates all span work here, keeping the (long)
session module about conversation assembly rather than telemetry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pydantic_core
from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, StatusCode, set_span_in_context

from pydantic_graph._utils import get_traceparent

from .._instrumentation import (
    InstrumentationNames,
    annotate_tool_call_otel_metadata,
    build_tool_definitions,
    model_metric_attributes,
    model_request_parameters_attributes,
    provider_attributes,
    redact_binary_content,
    response_attributes,
    response_price_calculation,
    safe_to_json,
    serialize_any,
)
from ..messages import ModelMessage, ModelResponse
from ..models.instrumented import InstrumentationSettings

if TYPE_CHECKING:
    from ..models import ModelRequestParameters
    from ..usage import RunUsage
    from .settings import RealtimeModelSettings

_REALTIME_SPAN_ATTRIBUTE = 'pydantic_ai.realtime'


class SessionInstrumentation:
    """Own a realtime session's OTel spans and the static metadata they are built from."""

    def __init__(
        self,
        settings: InstrumentationSettings | None,
        *,
        agent_name: str | None = None,
        agent_description: str | None = None,
        model_name: str | None = None,
        provider_name: str | None = None,
        provider_url: str | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
        model_settings: RealtimeModelSettings | None = None,
        output_type: Literal['speech', 'text'] = 'speech',
    ) -> None:
        self.settings = settings
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.model_name = model_name
        self.provider_name = provider_name
        self.provider_url = provider_url
        self.conversation_id = conversation_id
        self.run_id = run_id
        self.instructions = instructions
        self.metadata = metadata
        self.model_request_parameters = model_request_parameters
        self.model_settings = model_settings
        # The semconv `gen_ai.output.type` value: `'speech'` for spoken audio (the enum's term for
        # voice output), `'text'` for text-only. Starts from the configured output modality and is
        # updated from the content the provider actually emits (see `set_output_type`).
        self.output_type: Literal['speech', 'text'] = output_type

        # The session span is deliberately not made current in the owner's task. Child spans receive
        # this explicit context directly, or through the pump task's same-task attach/detach pair.
        self.context: Context | None = None
        self.session_span: Span | None = None
        self._session_span_attributes: dict[str, Any] | None = None
        # The `chat {model}` span for the response currently being assembled (see `ensure_chat_span`).
        self.chat_span: Span | None = None
        # The `speak {model}` span covering how long the model is actually audible (see
        # `start_playback_span`). Only a sideband reports playback, so it stays `None` elsewhere.
        self.playback_span: Span | None = None

    def start_session_span(self) -> None:
        """Open the session-wide span, the realtime analog of the classic agent-run span.

        The semconv operation-name enum has no realtime/speech value (nor do the OTel-native voice
        frameworks — LiveKit, Pipecat — emit one), so the session reports as an `invoke_agent`
        invocation like the classic agent-run span, with `gen_ai.output.type` (`speech`/`text`)
        marking the modality. `agent_name` defaults to `'agent'` like the classic span so an unnamed
        agent's session still carries the attribute that backends group runs by (e.g. Logfire's Runs
        view). No-op when instrumentation is disabled.
        """
        settings = self.settings
        if settings is None:
            return
        agent_name = self.agent_name or 'agent'
        names = InstrumentationNames.for_version(settings.version)
        attributes: dict[str, Any] = {
            'gen_ai.operation.name': 'invoke_agent',
            'gen_ai.output.type': self.output_type,
            # Both the semconv (`gen_ai.agent.name`) and legacy (`agent_name`) keys, matching the
            # classic run span, so backends that group runs by either recognize the session as a run.
            'gen_ai.agent.name': agent_name,
            'agent_name': agent_name,
            # An explicit marker so a backend can tell a realtime session (and its `chat` turns) apart
            # from a classic run: the semconv has no realtime operation, and `gen_ai.output.type` only
            # distinguishes audio from text, not realtime from a classic run that happens to be text.
            _REALTIME_SPAN_ATTRIBUTE: True,
            # Display the session as `<agent> realtime`, mirroring the classic run span's `<agent> run`
            # message, so it reads as the realtime variant of an agent run regardless of span name.
            'logfire.msg': f'{agent_name} realtime',
        }
        if self.model_name:
            # Match the classic agent-run span, which reports the model under the plain `model_name`
            # key (not `gen_ai.request.model`, which it keeps on its child `chat` spans only). The
            # realtime `chat`/turn spans likewise carry `gen_ai.request.model`.
            attributes['model_name'] = self.model_name
        if self.provider_name:
            # Provider/server attributes (`gen_ai.provider.name`, the deprecated `gen_ai.system`, and
            # `server.address`) so the session span identifies the provider, like the `chat` spans.
            attributes.update(provider_attributes(self.provider_name, self.provider_url))
        if self.agent_description:
            attributes['gen_ai.agent.description'] = self.agent_description
        if self.conversation_id:
            # Match the classic agent-run span's key (see `capabilities/instrumentation.py`) so a
            # realtime session can be correlated with other runs sharing the conversation id.
            attributes['gen_ai.conversation.id'] = self.conversation_id
        if self.run_id:
            attributes['gen_ai.agent.call.id'] = self.run_id
        # `model_request_parameters` / `model_settings` are sent once at connect (not per turn), so this
        # session span is their honest scope. They're also duplicated onto each per-turn span so
        # Logfire's per-step rendering (native tools, tool definitions) fires there too; see
        # `_request_config_attributes`.
        attributes.update(self._request_config_attributes(settings))
        # Follow the configured instrumentation version's agent-run naming: the semconv
        # `invoke_agent {name}` when that version is active (v3+), otherwise a bare `realtime`
        # operation name (the classic v2 span name is likewise a bare `agent run`).
        if names.agent_run_span_name == 'invoke_agent':
            span_name = names.get_agent_run_span_name(agent_name)
        else:
            span_name = 'realtime'
        parent_context = otel_context.get_current()
        span = settings.tracer.start_span(
            span_name,
            context=parent_context,
            attributes=attributes,
            kind=SpanKind.CLIENT,
        )
        self.session_span = span
        self.context = set_span_in_context(span, parent_context)
        self._session_span_attributes = attributes

    def end_session_span(
        self,
        error: BaseException | None,
        *,
        usage: RunUsage,
        messages: list[ModelMessage],
        new_message_index: int | None,
        final_result: str | None,
        audio_chunks_dropped: int,
        transcript_items_dropped: int,
    ) -> str | None:
        """Finalize and end the session span, returning its traceparent (for `AgentRunResult`).

        Attaches cumulative usage, run context, and the conversation to the span, mirroring the
        classic agent-run span's end-of-run contract. No-op (returning `None`) when instrumentation
        is disabled or no span was started.
        """
        settings = self.settings
        span = self.session_span
        if settings is None or span is None:
            return None
        if error is not None:
            self.record_error(span, error)
        # Report cumulative usage under `gen_ai.aggregated_usage.*` (mirroring the classic agent-run
        # span) so backends that sum span attributes don't double-count it against the per-turn `chat`
        # spans, which carry each response's usage under `gen_ai.usage.*`. Shared with the classic span.
        attributes: dict[str, Any] = {
            **settings.aggregated_usage_attributes(usage),
            **settings.system_instructions_attributes(self.instructions),
            'pydantic_ai.audio_chunks_dropped': audio_chunks_dropped,
            'pydantic_ai.transcript_items_dropped': transcript_items_dropped,
        }
        schema_properties: dict[str, Any] = {}
        if 'gen_ai.system_instructions' in attributes:
            schema_properties['gen_ai.system_instructions'] = {'type': 'array'}
        # Mirror the classic agent-run span's end-of-run contract (the `Instrumentation`
        # capability's `_run_span_end_attributes`): the full conversation — seeded history included —
        # under `pydantic_ai.all_messages`, with `pydantic_ai.new_message_index` marking where this
        # session's messages begin. Emitted regardless of `include_content`: `otel_message_parts`
        # redacts part *content* when it is disabled, leaving the conversation structure. The
        # `logfire.json_schema` entry marks the attribute as a JSON array so the Logfire UI
        # deserializes and renders it as a conversation rather than as a string.
        if messages:
            attributes['pydantic_ai.all_messages'] = safe_to_json(settings.messages_to_otel_messages(messages)).decode()
            if new_message_index is not None:
                attributes['pydantic_ai.new_message_index'] = new_message_index
            schema_properties['pydantic_ai.all_messages'] = {'type': 'array'}
        if self.metadata is not None:
            # Redact binary payloads under `include_binary_content=False`, matching the classic
            # run span's metadata attribute (the `Instrumentation` capability).
            attributes['metadata'] = safe_to_json(
                serialize_any(redact_binary_content(self.metadata, settings))
            ).decode()
            schema_properties['metadata'] = {}
        # Declare the session-wide `model_request_parameters` / `model_settings` blobs (set at start by
        # `_request_config_attributes`) as objects here, since this rebuilds the span's `logfire.json_schema`.
        schema_properties.update(self._request_config_schema_properties(settings))
        # Mirror the classic run span's `final_result` (set by the `Instrumentation` capability): a
        # realtime session has no single output, so use the most recent assistant reply's text, which the
        # Logfire UI renders as the run's final response. Gated on `include_content` like the classic span.
        if settings.include_content and final_result is not None:
            attributes['final_result'] = final_result
        if schema_properties:
            attributes['logfire.json_schema'] = pydantic_core.to_json(
                {'type': 'object', 'properties': schema_properties}
            ).decode()
        span.set_attributes(attributes)
        traceparent = get_traceparent(span) or None
        span.end()
        self.session_span = None
        self.context = None
        self._session_span_attributes = None
        return traceparent

    def ensure_chat_span(self) -> None:
        """Open a `chat {model}` span for the assistant response now being assembled, if not already open.

        A realtime turn isn't a single request/response, so the honest lifetime of a `chat` span is one
        assistant `ModelResponse`: it opens when that response's first content arrives (the first
        assistant part or tool call) and closes in `_finalize_response`. Tool calls split a turn into
        multiple responses (mirroring a classic run), so each response gets its own span. The span is
        deliberately *not* entered as the current span: `execute_tool` spans run after the response is
        finalized and stay siblings under the session span, matching the classic agent-run tree.

        The session-wide request config (`model_request_parameters`, `model_settings`,
        `gen_ai.tool.definitions`) is duplicated here from the session span via
        `_request_config_attributes`, matching where the classic `chat` span (`open_model_request_span`)
        carries it so Logfire renders native tools and tool definitions per step. Provider and server
        attributes, response metadata, usage, cost when pricing data is available, and per-response metrics
        reuse the classic instrumentation helpers.
        Added vs. the classic span: `gen_ai.output.type` (`speech`/`text`), the one semconv attribute
        specific to voice output. The span keeps the semconv `chat` operation and `chat {model}` name, but
        renders (via `logfire.msg`) as `response {model}`: nothing was "chatted" — no request was sent —
        and this span covers exactly one `ModelResponse`, which is *not* the same as a conversational
        turn (a turn that calls tools produces several). The turn boundary is the `model turn complete` span.
        """
        settings = self.settings
        if settings is None or self.chat_span is not None:
            return
        attributes: dict[str, Any] = {
            'gen_ai.operation.name': 'chat',
            'gen_ai.output.type': self.output_type,
            # Mark the turn as realtime too (see the session span), so a backend can tell a realtime
            # `chat` span apart from a classic model-request `chat` span.
            _REALTIME_SPAN_ATTRIBUTE: True,
            # Render as `response {model}` while keeping the semconv `chat` operation + span name: this
            # span covers one `ModelResponse`, and no request was sent, so "chat" misleads. Verb-first
            # matches the other span messages (`chat {model}`, `execute_tool {name}`, `invoke_agent {name}`).
            'logfire.msg': f'response {self.model_name}' if self.model_name else 'response',
        }
        if self.model_name:
            attributes['gen_ai.request.model'] = self.model_name
        if self.provider_name:
            attributes.update(provider_attributes(self.provider_name, self.provider_url))
        # The session-wide request config, duplicated here so Logfire's per-step rendering fires (see
        # `_request_config_attributes`). `end_chat_span`'s `handle_messages` redeclares
        # `model_request_parameters` in the span's `logfire.json_schema`, so it stays richly rendered.
        attributes.update(self._request_config_attributes(settings))
        name = f'chat {self.model_name}' if self.model_name else 'chat'
        context = self.context
        assert context is not None
        self.chat_span = settings.tracer.start_span(
            name,
            context=context,
            attributes=attributes,
            kind=SpanKind.CLIENT,
        )

    def end_chat_span(self, input_messages: list[ModelMessage], response: ModelResponse | None) -> None:
        """Close the current `chat` span, attaching the response's messages, usage, and state."""
        settings = self.settings
        span = self.chat_span
        if settings is None or span is None:
            return
        self.chat_span = None
        price_calculation = response_price_calculation(response) if response is not None else None
        if response is not None and span.is_recording():
            # Reuse the exact message → gen_ai serialization and response-attribute helpers the
            # instrumented model uses, so realtime `chat` spans can't drift from the classic path.
            if self.model_request_parameters is not None:
                annotate_tool_call_otel_metadata(response, self.model_request_parameters)
            settings.handle_messages(input_messages, response, span)
            span.set_attributes(
                response_attributes(response, response.model_name or self.model_name, price_calculation)
            )
            if response.state != 'complete':
                # How the response ended, when it didn't end normally: `'interrupted'` for a barge-in
                # or an explicit `interrupt()`. The `interrupt` span records the request; this records
                # the outcome on the response it actually cut off.
                span.set_attribute('pydantic_ai.response.state', response.state)
        span.end()
        if response is not None:
            settings.record_metrics(
                response,
                price_calculation,
                model_metric_attributes(
                    self.provider_name,
                    self.model_name,
                    response.model_name or self.model_name,
                ),
            )

    def start_playback_span(self) -> None:
        """Open a `speak {model}` span covering how long the model is actually audible.

        Distinct from the `chat`/`turn complete` spans, which measure *generation*: the provider produces
        audio far faster than it plays, so a response can be complete while the listener still has many
        seconds of speech to hear. That gap is what makes a barge-in feel broken, so it's worth its own
        span. Only opened where the provider reports playback (a WebRTC sideband), so an ordinary
        session's trace is unchanged.
        """
        settings = self.settings
        if settings is None or self.playback_span is not None:
            return
        context = self.context
        assert context is not None
        self.playback_span = settings.tracer.start_span(
            f'speak {self.model_name}' if self.model_name else 'speak',
            context=context,
            attributes={
                _REALTIME_SPAN_ATTRIBUTE: True,
                'logfire.msg': f'speak {self.model_name}' if self.model_name else 'speak',
            },
        )

    def end_playback_span(self) -> None:
        """Close the `speak` span when the model stops being audible."""
        if (span := self.playback_span) is not None:
            self.playback_span = None
            span.end()

    def set_output_type(self, output_type: Literal['speech', 'text']) -> None:
        """Update telemetry from the response content the provider actually emitted."""
        self.output_type = output_type
        if self.session_span is not None:
            self.session_span.set_attribute('gen_ai.output.type', output_type)
        if self._session_span_attributes is not None:
            self._session_span_attributes['gen_ai.output.type'] = output_type
        if self.chat_span is not None:
            self.chat_span.set_attribute('gen_ai.output.type', output_type)

    def _request_config_attributes(self, settings: InstrumentationSettings) -> dict[str, Any]:
        """OTel attribute *values* for the request config the session was opened with.

        A realtime session sends `model_request_parameters` and `model_settings` once at connect (not per
        turn), so they're stable for the whole session. They go on the session span — their honest scope —
        and are duplicated onto each per-turn span, matching where the classic path puts them (the `chat`
        span) so Logfire's per-step rendering of native tools and `gen_ai.tool.definitions` still fires.
        `model_request_parameters` (and the serialized realtime `model_settings`, whose vocabulary —
        provider voice settings, `output_modality`, `thinking`, `turn_detection`, ... — has no OTel-spec `gen_ai.request.*`
        equivalent) are gated on `include_model_request_parameters`; tool definitions and `max_tokens`,
        which have spec homes, are set ungated like the classic path.

        The `logfire.json_schema` declarations that make the serialized blobs render as objects (rather
        than raw strings) are added at span *finalization*: the session span's in `end_session_span`, the
        `chat` span's by `handle_messages` (which redeclares `model_request_parameters`) — both rebuild
        `logfire.json_schema` at the end, so declaring it here would be overwritten. See
        `_request_config_schema_properties`.
        """
        attributes: dict[str, Any] = {}
        if self.model_request_parameters is not None and (
            tool_definitions := build_tool_definitions(self.model_request_parameters)
        ):
            attributes['gen_ai.tool.definitions'] = safe_to_json(tool_definitions).decode()
        if settings.include_model_request_parameters:
            if self.model_request_parameters is not None:
                attributes.update(model_request_parameters_attributes(self.model_request_parameters))
            if self.model_settings:
                attributes['model_settings'] = safe_to_json(serialize_any(self.model_settings)).decode()
        if self.model_settings and (max_tokens := self.model_settings.get('max_tokens')) is not None:
            attributes['gen_ai.request.max_tokens'] = max_tokens
        return attributes

    def _request_config_schema_properties(self, settings: InstrumentationSettings) -> dict[str, dict[str, str]]:
        """`logfire.json_schema` properties declaring the serialized config blobs as objects.

        Merged into the session span's schema in `end_session_span` so Logfire renders
        `model_request_parameters` / `model_settings` richly instead of as raw JSON strings.
        """
        properties: dict[str, dict[str, str]] = {}
        if settings.include_model_request_parameters:
            if self.model_request_parameters is not None:
                properties['model_request_parameters'] = {'type': 'object'}
            if self.model_settings:
                properties['model_settings'] = {'type': 'object'}
        return properties

    def record_user_speech(self, started_at: int | None) -> None:
        """Record the segment the user just spoke, as a `user speech` span with a real duration.

        Emitted on the *end* of speech, backdated to the onset, so the span only exists when the
        provider reported both boundaries. Gemini Live reports onset but never the end, so it records
        no span rather than one whose length was inferred from something else — a duration nobody
        measured is worse than no duration at all.
        """
        if self.settings is None or self.context is None or started_at is None:
            return
        self.settings.tracer.start_span(
            'user speech',
            context=self.context,
            start_time=started_at,
            attributes={_REALTIME_SPAN_ATTRIBUTE: True, 'logfire.msg': 'user speech'},
            kind=SpanKind.INTERNAL,
        ).end()

    def record_lifecycle(self, name: str, *, message: str | None = None, **attributes: Any) -> None:
        """Record a realtime lifecycle moment (barge-in, turn boundary) as a zero-duration child span.

        Turn boundaries and barge-ins have no request/response of their own, so they surface as
        instantaneous spans under the session span, making the stream's progression visible in a trace
        (rather than `logfire.info` calls, which each app would otherwise have to add itself). A span
        rather than a span event because backends surface spans immediately and predictably. Names are
        lowercase to match the surrounding spans; attributes whose value is `None` are dropped so the
        span stays clean. Every span carries `pydantic_ai.realtime` so backends can recognize the
        whole session tree, lifecycle moments included. No-op when instrumentation is disabled.

        `message` sets `logfire.msg` to vary the displayed text without splitting the span name into
        more than one grouping key — e.g. an interrupted turn boundary reads "model turn complete
        (interrupted)" while still counting as a `model turn complete` span.
        """
        if self.settings is None or self.context is None:
            return
        span_attributes: dict[str, Any] = {_REALTIME_SPAN_ATTRIBUTE: True}
        if message is not None:
            span_attributes['logfire.msg'] = message
        span_attributes.update({key: value for key, value in attributes.items() if value is not None})
        self.settings.tracer.start_span(
            name, context=self.context, attributes=span_attributes, kind=SpanKind.INTERNAL
        ).end()

    @staticmethod
    def record_error(span: Span, error: BaseException) -> None:
        if span.is_recording():
            span.record_exception(error, escaped=True)
            span.set_status(StatusCode.ERROR)
