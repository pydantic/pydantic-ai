"""Private OpenTelemetry support for realtime sessions."""

from __future__ import annotations

from typing import Any

from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, StatusCode

from ..models.instrumented import InstrumentationSettings

_REALTIME_SPAN_ATTRIBUTE = 'pydantic_ai.realtime'


class _SessionInstrumentation:  # pyright: ignore[reportUnusedClass]
    """Own the span operations that do not depend on conversation assembly."""

    def __init__(self, settings: InstrumentationSettings | None) -> None:
        self.settings = settings
        self.context: Context | None = None

    def record_user_speech(self, started_at: int | None) -> None:
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
