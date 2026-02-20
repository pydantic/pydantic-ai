"""OpenTelemetry instrumentation for realtime model sessions."""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from opentelemetry.trace import SpanKind
from opentelemetry.util.types import AttributeValue

from ..models.instrumented import InstrumentationSettings
from ..settings import ModelSettings
from ..tools import ToolDefinition
from ._base import RealtimeConnection, RealtimeModel

__all__ = ('InstrumentedRealtimeModel', 'instrument_realtime_model')


def instrument_realtime_model(model: RealtimeModel, instrument: InstrumentationSettings | bool) -> RealtimeModel:
    """Wrap a realtime model with OpenTelemetry instrumentation if not already wrapped."""
    if instrument and not isinstance(model, InstrumentedRealtimeModel):
        if instrument is True:
            instrument = InstrumentationSettings()
        model = InstrumentedRealtimeModel(model, instrument)
    return model


class InstrumentedRealtimeModel(RealtimeModel):
    """Wraps a `RealtimeModel` so that `connect()` creates an OpenTelemetry span covering the session."""

    def __init__(self, wrapped: RealtimeModel, settings: InstrumentationSettings) -> None:
        self._wrapped = wrapped
        self._settings = settings

    @property
    def model_name(self) -> str:
        return self._wrapped.model_name

    @property
    def system(self) -> str:
        return self._wrapped.system

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[RealtimeConnection]:
        operation = 'realtime'
        span_name = f'{operation} {self._wrapped.model_name}'

        attributes: dict[str, AttributeValue] = {
            'gen_ai.operation.name': operation,
            'gen_ai.provider.name': self._wrapped.system,
            'gen_ai.system': self._wrapped.system,
            'gen_ai.request.model': self._wrapped.model_name,
        }

        base_url: str | None = getattr(self._wrapped, 'base_url', None)
        if base_url:
            try:
                parsed = urlparse(base_url)
            except Exception:  # pragma: no cover
                pass
            else:
                if parsed.hostname:  # pragma: no branch
                    attributes['server.address'] = parsed.hostname
                if parsed.port:
                    attributes['server.port'] = parsed.port

        if tools:
            tool_definitions: list[dict[str, Any]] = []
            for tool in tools:
                tool_def: dict[str, Any] = {'type': 'function', 'name': tool.name}
                if tool.description:
                    tool_def['description'] = tool.description
                if tool.parameters_json_schema:
                    tool_def['parameters'] = tool.parameters_json_schema
                tool_definitions.append(tool_def)
            attributes['gen_ai.tool.definitions'] = json.dumps(tool_definitions)

        if instructions and self._settings.include_content:
            attributes['gen_ai.system_instructions'] = instructions

        with self._settings.tracer.start_as_current_span(span_name, attributes=attributes, kind=SpanKind.CLIENT):
            async with self._wrapped.connect(
                instructions=instructions,
                tools=tools,
                model_settings=model_settings,
            ) as connection:
                yield connection
