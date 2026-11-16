from __future__ import annotations

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from pydantic_ai import _instrumentation
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel


async def test_sampled_out_model_request_skips_span_attribute_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a span is not recording (sampled out), expensive JSON attribute serialization is skipped."""
    tracer_provider = TracerProvider(sampler=ALWAYS_OFF)
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    settings = InstrumentationSettings(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        include_content=True,
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError('span attribute serialization should not run for a non-recording span')

    monkeypatch.setattr(_instrumentation, 'build_tool_definitions', fail_if_called)
    monkeypatch.setattr(_instrumentation, 'model_request_parameters_attributes', fail_if_called)
    monkeypatch.setattr(_instrumentation, 'model_settings_attributes', fail_if_called)

    agent = Agent(TestModel(), capabilities=[Instrumentation(settings=settings)])

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    result = await agent.run('compute 1 + 2')
    assert result.output is not None

    # Metrics must still be recorded even when the span itself is sampled out.
    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    assert len(metrics_data.resource_metrics) > 0
