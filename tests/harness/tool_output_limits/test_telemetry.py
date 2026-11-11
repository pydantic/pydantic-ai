"""`ToolOutputLimits` says on the call's `execute_tool` span how it reduced the result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inline_snapshot import snapshot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Instrumentation
from pydantic_ai.messages import ToolReturn, ToolReturnPart
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.tool_output_limits import Band, LocalFileStore, Spill, ToolOutputLimits, Truncate


def instrumented(*, include_content: bool = False) -> tuple[Instrumentation, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(tracer_provider=provider, include_content=include_content)
    return Instrumentation(settings=settings), exporter


def agent_with_tool(
    limits: ToolOutputLimits[object], output: object, instrumentation: Instrumentation | None = None
) -> Agent[object, str]:
    capabilities: list[AbstractCapability[object]] = [limits] if instrumentation is None else [limits, instrumentation]
    agent: Agent[object, str] = Agent(TestModel(call_tools=['big_tool']), capabilities=capabilities)

    @agent.tool_plain
    def big_tool() -> Any:
        return output

    return agent


def limits_attributes(exporter: InMemorySpanExporter) -> dict[str, Any]:
    """The `tool_output_limits.*` attributes on the `execute_tool big_tool` span."""
    [span] = [span for span in exporter.get_finished_spans() if span.name == 'execute_tool big_tool']
    return {key: value for key, value in (span.attributes or {}).items() if key.startswith('tool_output_limits.')}


class FailingStore:
    """An `OverflowStore` that can't take a write."""

    async def write(self, key: str, data: bytes) -> str:
        raise OSError('disk full')

    async def read(self, handle: str) -> bytes:  # pragma: no cover - nothing is ever written
        raise FileNotFoundError(handle)


TRUNCATE = ToolOutputLimits[object](bands=[Band(over=100, action=Truncate(max_chars=50))])


class TestToolOutputLimitsTelemetry:
    async def test_a_truncated_result_is_recorded(self):
        instrumentation, exporter = instrumented()
        await agent_with_tool(TRUNCATE, 'x' * 500, instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.action': 'truncate',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.reduced_size': 50,
                'tool_output_limits.size_unit': 'chars',
            }
        )

    async def test_a_result_under_every_band_records_nothing(self):
        instrumentation, exporter = instrumented()
        await agent_with_tool(TRUNCATE, 'small', instrumentation).run('go')

        assert limits_attributes(exporter) == {}

    async def test_a_truncate_that_cuts_nothing_records_nothing(self):
        """The band matches, but the text already fits `max_chars`, so the model gets it unchanged."""
        limits = ToolOutputLimits[object](bands=[Band(over=100, action=Truncate(max_chars=1_000))])
        instrumentation, exporter = instrumented()
        await agent_with_tool(limits, 'x' * 500, instrumentation).run('go')

        assert limits_attributes(exporter) == {}

    async def test_sizes_are_in_tokens_with_over_tokens(self):
        limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Truncate(max_chars=100))],
            over_tokens=True,
            tokenizer=lambda text: len(text) // 4,
        )
        instrumentation, exporter = instrumented()
        await agent_with_tool(limits, 'x' * 2_000, instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.action': 'truncate',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.reduced_size': 25,
                'tool_output_limits.size_unit': 'tokens',
            }
        )

    async def test_a_failed_spill_records_the_truncate_it_fell_back_to(self):
        limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Spill(then=Truncate(max_chars=50)))], store=FailingStore()
        )
        instrumentation, exporter = instrumented(include_content=True)
        await agent_with_tool(limits, 'x' * 500, instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.action': 'truncate',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.reduced_size': 50,
                'tool_output_limits.size_unit': 'chars',
                'tool_output_limits.fallback_from': 'spill',
            }
        )

    async def test_binary_that_cannot_be_truncated_records_the_spill_it_fell_back_to(self, tmp_path: Path):
        limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Truncate(then=Spill()))], store=LocalFileStore(base_dir=tmp_path)
        )
        instrumentation, exporter = instrumented()
        await agent_with_tool(limits, b'\x00' * 500, instrumentation).run('go')

        attributes = limits_attributes(exporter)
        assert (attributes['tool_output_limits.action'], attributes['tool_output_limits.fallback_from']) == (
            'spill',
            'truncate',
        )

    async def test_content_reduced_on_its_own_gets_its_own_attributes(self):
        instrumentation, exporter = instrumented()
        await agent_with_tool(TRUNCATE, ToolReturn(return_value='small', content='y' * 300), instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.content.action': 'truncate',
                'tool_output_limits.content.original_size': 300,
                'tool_output_limits.content.reduced_size': 50,
                'tool_output_limits.content.size_unit': 'chars',
            }
        )

    async def test_a_spill_records_its_handle_only_with_content(self, tmp_path: Path):
        spill = ToolOutputLimits[object](
            bands=[Band(over=100, action=Spill())], store=LocalFileStore(base_dir=tmp_path)
        )
        output = b'\x00' * 500

        instrumentation, exporter = instrumented()
        await agent_with_tool(spill, output, instrumentation).run('go')
        attributes = limits_attributes(exporter)
        assert attributes.pop('tool_output_limits.reduced_size') > 0
        assert attributes == snapshot(
            {
                'tool_output_limits.action': 'spill',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.size_unit': 'bytes',
            }
        )

        instrumentation, exporter = instrumented(include_content=True)
        result = await agent_with_tool(spill, output, instrumentation).run('go')
        [part] = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert part.metadata is not None
        assert limits_attributes(exporter)['tool_output_limits.handle'] == part.metadata['overflow_handle']

    async def test_an_uninstrumented_run_leaves_the_applications_span_alone(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with provider.get_tracer('app').start_as_current_span('app request'):
            await agent_with_tool(TRUNCATE, 'x' * 500).run('go')

        assert [(span.name, dict(span.attributes or {})) for span in exporter.get_finished_spans()] == [
            ('app request', {})
        ]
