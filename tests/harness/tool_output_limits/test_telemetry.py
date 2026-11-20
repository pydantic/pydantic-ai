"""`ToolOutputLimits` says on the call's `execute_tool` span how it reduced the result."""

from __future__ import annotations

from pathlib import Path

from inline_snapshot import snapshot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Instrumentation
from pydantic_ai.messages import ToolReturn, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.tool_output_limits import (
    Band,
    LocalFileOverflowStore,
    Spill,
    Summarize,
    ToolOutputLimits,
    Truncate,
)
from tests.harness.conftest import Instrumented


def agent_with_tool(
    limits: ToolOutputLimits[object], output: object, instrumentation: Instrumentation | None = None
) -> Agent[object, str]:
    capabilities: list[AbstractCapability[object]] = [limits] if instrumentation is None else [limits, instrumentation]
    agent: Agent[object, str] = Agent(TestModel(call_tools=['big_tool']), capabilities=capabilities)

    @agent.tool_plain
    def big_tool() -> object:
        return output

    return agent


def limits_attributes(exporter: InMemorySpanExporter) -> dict[str, AttributeValue]:
    """The `tool_output_limits.*` attributes on the `execute_tool big_tool` span."""
    [span] = [span for span in exporter.get_finished_spans() if span.name == 'execute_tool big_tool']
    return {key: value for key, value in (span.attributes or {}).items() if key.startswith('tool_output_limits.')}


class FailingStore:
    """An `OverflowStore` that can't take a write."""

    async def write(self, key: str, data: bytes) -> str:
        raise OSError('disk full')

    async def read(self, handle: str) -> bytes:  # pragma: no cover - nothing is ever written
        raise FileNotFoundError(handle)


def failing_summarize(tool_name: str, text: str) -> str:
    raise ValueError('summarizer down')


def short_summarize(tool_name: str, text: str) -> str:
    return 'a short summary'


TRUNCATE = ToolOutputLimits[object](bands=[Band(over=100, action=Truncate(max_chars=50))])


class TestToolOutputLimitsTelemetry:
    async def test_a_truncated_result_is_recorded(self, instrumented: Instrumented):
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(TRUNCATE, 'x' * 500, instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.action': 'truncate',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.reduced_size': 50,
                'tool_output_limits.size_unit': 'chars',
            }
        )

    async def test_a_result_under_every_band_records_nothing(self, instrumented: Instrumented):
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(TRUNCATE, 'small', instrumentation).run('go')

        assert limits_attributes(exporter) == {}

    async def test_a_truncate_that_cuts_nothing_records_nothing(self, instrumented: Instrumented):
        """The band matches, but the text already fits `max_chars`, so the model gets it unchanged."""
        limits = ToolOutputLimits[object](bands=[Band(over=100, action=Truncate(max_chars=1_000))])
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(limits, 'x' * 500, instrumentation).run('go')

        assert limits_attributes(exporter) == {}

    async def test_sizes_are_in_tokens_with_over_tokens(self, instrumented: Instrumented):
        limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Truncate(max_chars=100))],
            over_tokens=True,
            tokenizer=lambda text: len(text) // 4,
        )
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(limits, 'x' * 2_000, instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.action': 'truncate',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.reduced_size': 25,
                'tool_output_limits.size_unit': 'tokens',
            }
        )

    async def test_a_failed_spill_records_the_truncate_it_fell_back_to(self, instrumented: Instrumented):
        limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Spill(then=Truncate(max_chars=50)))], store=FailingStore()
        )
        instrumentation, exporter = instrumented(True)
        await agent_with_tool(limits, 'x' * 500, instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.action': 'truncate',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.reduced_size': 50,
                'tool_output_limits.size_unit': 'chars',
                'tool_output_limits.fallback_from': 'spill',
                'tool_output_limits.fallback_error': 'OSError',
            }
        )

    async def test_a_failed_spill_is_recorded_even_when_its_fallback_cuts_nothing(self, instrumented: Instrumented):
        limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Spill(then=Truncate(max_chars=1_000)))], store=FailingStore()
        )
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(limits, 'x' * 500, instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.action': 'truncate',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.reduced_size': 500,
                'tool_output_limits.size_unit': 'chars',
                'tool_output_limits.fallback_from': 'spill',
                'tool_output_limits.fallback_error': 'OSError',
            }
        )

    async def test_binary_that_cannot_be_truncated_records_the_spill_it_fell_back_to(
        self, tmp_path: Path, instrumented: Instrumented
    ):
        limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Truncate(then=Spill()))], store=LocalFileOverflowStore(base_dir=tmp_path)
        )
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(limits, b'\x00' * 500, instrumentation).run('go')

        attributes = limits_attributes(exporter)
        assert (attributes['tool_output_limits.action'], attributes['tool_output_limits.fallback_from']) == (
            'spill',
            'truncate',
        )
        # Truncate can't take binary, so no exception stopped it.
        assert 'tool_output_limits.fallback_error' not in attributes

    async def test_a_summary_is_recorded(self, instrumented: Instrumented):
        limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Summarize(summarize=short_summarize, then=Truncate(max_chars=50)))]
        )
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(limits, 'x' * 500, instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.action': 'summarize',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.reduced_size': 93,
                'tool_output_limits.size_unit': 'chars',
            }
        )

    async def test_a_failed_summary_records_its_error_type(self, instrumented: Instrumented):
        limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Summarize(summarize=failing_summarize, then=Truncate(max_chars=50)))]
        )
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(limits, 'x' * 500, instrumentation).run('go')

        attributes = limits_attributes(exporter)
        assert (attributes['tool_output_limits.fallback_from'], attributes['tool_output_limits.fallback_error']) == (
            'summarize',
            'ValueError',
        )

    async def test_a_chain_of_failures_records_the_bands_own_action_and_its_error(self, instrumented: Instrumented):
        """The spill fails with `OSError`, then its summary with `ValueError`; the band's own spill is what's kept."""
        summarize = Summarize(summarize=failing_summarize, then=Truncate(max_chars=50))
        limits = ToolOutputLimits[object](bands=[Band(over=100, action=Spill(then=summarize))], store=FailingStore())
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(limits, 'x' * 500, instrumentation).run('go')

        attributes = limits_attributes(exporter)
        assert (
            attributes['tool_output_limits.action'],
            attributes['tool_output_limits.fallback_from'],
            attributes['tool_output_limits.fallback_error'],
        ) == ('truncate', 'spill', 'OSError')

    async def test_content_reduced_on_its_own_gets_its_own_attributes(self, instrumented: Instrumented):
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(TRUNCATE, ToolReturn(return_value='small', content='y' * 300), instrumentation).run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.content.action': 'truncate',
                'tool_output_limits.content.original_size': 300,
                'tool_output_limits.content.reduced_size': 50,
                'tool_output_limits.content.size_unit': 'chars',
            }
        )

    async def test_a_spill_records_its_handle_only_with_content(self, tmp_path: Path, instrumented: Instrumented):
        spill = ToolOutputLimits[object](
            bands=[Band(over=100, action=Spill())], store=LocalFileOverflowStore(base_dir=tmp_path)
        )
        output = b'\x00' * 500

        instrumentation, exporter = instrumented(False)
        await agent_with_tool(spill, output, instrumentation).run('go')
        attributes = limits_attributes(exporter)
        reduced_size = attributes.pop('tool_output_limits.reduced_size')
        assert isinstance(reduced_size, int) and reduced_size > 0
        assert attributes == snapshot(
            {
                'tool_output_limits.action': 'spill',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.size_unit': 'bytes',
            }
        )

        instrumentation, exporter = instrumented(True)
        result = await agent_with_tool(spill, output, instrumentation).run('go')
        [part] = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert part.metadata is not None
        assert limits_attributes(exporter)['tool_output_limits.handle'] == part.metadata['overflow_handle']

    async def test_a_content_spill_records_its_handle_only_with_content(
        self, tmp_path: Path, instrumented: Instrumented
    ):
        spill = ToolOutputLimits[object](
            bands=[Band(over=100, action=Spill())], store=LocalFileOverflowStore(base_dir=tmp_path)
        )
        output = ToolReturn(return_value='small', content='y' * 300)

        instrumentation, exporter = instrumented(False)
        await agent_with_tool(spill, output, instrumentation).run('go')
        attributes = limits_attributes(exporter)
        assert attributes['tool_output_limits.content.action'] == 'spill'
        assert 'tool_output_limits.content.handle' not in attributes

        instrumentation, exporter = instrumented(True)
        result = await agent_with_tool(spill, output, instrumentation).run('go')
        [part] = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert part.metadata is not None
        assert (
            limits_attributes(exporter)['tool_output_limits.content.handle'] == part.metadata['overflow_content_handle']
        )

    async def test_an_uninstrumented_run_leaves_the_applications_span_alone(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with provider.get_tracer('app').start_as_current_span('app request'):
            await agent_with_tool(TRUNCATE, 'x' * 500).run('go')

        assert [(span.name, dict(span.attributes or {})) for span in exporter.get_finished_spans()] == [
            ('app request', {})
        ]
