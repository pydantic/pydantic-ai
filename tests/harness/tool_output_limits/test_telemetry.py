"""`ToolOutputLimits` says on the call's `execute_tool` span how it reduced the result."""

from __future__ import annotations

from pathlib import Path

import pytest
from inline_snapshot import snapshot
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
from tests.harness._recording_durability import RecordingDurability
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

    # Nothing is ever written, so nothing is read.
    async def read(self, handle: str) -> bytes:  # pragma: no cover
        raise FileNotFoundError(handle)


def failing_summarize(tool_name: str, text: str) -> str:
    raise ValueError('summarizer down')


def short_summarize(tool_name: str, text: str) -> str:
    return 'a short summary'


def limits(action: Truncate | Spill | Summarize) -> ToolOutputLimits[object]:
    return ToolOutputLimits[object](bands=[Band(over=100, action=action)], store=FailingStore())


TEXT = 'x' * 500
BINARY = b'\x00' * 500


class TestToolOutputLimitsTelemetry:
    @pytest.mark.parametrize(
        'tool_output_limits, output, expected',
        [
            pytest.param(
                limits(Truncate(max_chars=50)),
                TEXT,
                {
                    'tool_output_limits.action': 'truncate',
                    'tool_output_limits.original_size': 500,
                    'tool_output_limits.reduced_size': 50,
                    'tool_output_limits.size_unit': 'chars',
                },
                id='truncate',
            ),
            pytest.param(limits(Truncate(max_chars=1_000)), TEXT, {}, id='truncate-cuts-nothing'),
            pytest.param(
                ToolOutputLimits[object](
                    bands=[Band(over=100, action=Truncate(max_chars=100))],
                    over_tokens=True,
                    tokenizer=lambda text: len(text) // 4,
                ),
                'x' * 2_000,
                {
                    'tool_output_limits.action': 'truncate',
                    'tool_output_limits.original_size': 500,
                    'tool_output_limits.reduced_size': 25,
                    'tool_output_limits.size_unit': 'tokens',
                },
                id='tokens',
            ),
            pytest.param(
                limits(Summarize(summarize=short_summarize, then=Truncate(max_chars=50))),
                TEXT,
                {
                    'tool_output_limits.action': 'summarize',
                    'tool_output_limits.original_size': 500,
                    'tool_output_limits.reduced_size': 93,
                    'tool_output_limits.size_unit': 'chars',
                },
                id='summarize',
            ),
            pytest.param(
                limits(Summarize(summarize=failing_summarize, then=Truncate(max_chars=50))),
                TEXT,
                snapshot(
                    {
                        'tool_output_limits.action': 'truncate',
                        'tool_output_limits.original_size': 500,
                        'tool_output_limits.reduced_size': 50,
                        'tool_output_limits.size_unit': 'chars',
                        'tool_output_limits.fallback_from': 'summarize',
                        'tool_output_limits.fallback_error': 'ValueError',
                    }
                ),
                id='failed-summary',
            ),
            # The fallback cuts nothing, and is still recorded: the store failing is what an operator needs to see.
            pytest.param(
                limits(Spill(then=Truncate(max_chars=1_000))),
                TEXT,
                {
                    'tool_output_limits.action': 'truncate',
                    'tool_output_limits.original_size': 500,
                    'tool_output_limits.reduced_size': 500,
                    'tool_output_limits.size_unit': 'chars',
                    'tool_output_limits.fallback_from': 'spill',
                    'tool_output_limits.fallback_error': 'OSError',
                },
                id='failed-spill-falls-back',
            ),
            # The spill fails, then its summary: the band's own action and its error are what's kept.
            pytest.param(
                limits(Spill(then=Summarize(summarize=failing_summarize, then=Truncate(max_chars=50)))),
                TEXT,
                snapshot(
                    {
                        'tool_output_limits.action': 'truncate',
                        'tool_output_limits.original_size': 500,
                        'tool_output_limits.reduced_size': 50,
                        'tool_output_limits.size_unit': 'chars',
                        'tool_output_limits.fallback_from': 'spill',
                        'tool_output_limits.fallback_error': 'OSError',
                    }
                ),
                id='chain-of-failures',
            ),
            # The default `Spill(then=Truncate())`: the store fails and `Truncate` can't take binary.
            pytest.param(
                ToolOutputLimits[object](store=FailingStore()),
                b'\x00' * 200_000,
                {'tool_output_limits.fallback_from': 'spill', 'tool_output_limits.fallback_error': 'OSError'},
                id='binary-store-fails',
            ),
            pytest.param(
                limits(Truncate()),
                BINARY,
                {'tool_output_limits.fallback_from': 'truncate'},
                id='binary-under-bare-truncate',
            ),
            pytest.param(
                limits(Truncate(max_chars=50)),
                ToolReturn(return_value='small', content='y' * 300),
                {
                    'tool_output_limits.content.action': 'truncate',
                    'tool_output_limits.content.original_size': 300,
                    'tool_output_limits.content.reduced_size': 50,
                    'tool_output_limits.content.size_unit': 'chars',
                },
                id='separate-content',
            ),
        ],
    )
    async def test_the_span_records_how_the_result_was_reduced(
        self,
        tool_output_limits: ToolOutputLimits[object],
        output: object,
        expected: dict[str, AttributeValue],
        instrumented: Instrumented,
    ):
        """With content off, the exact dict is also the check that nothing content-bearing reaches the span."""
        instrumentation, exporter = instrumented(False)
        await agent_with_tool(tool_output_limits, output, instrumentation).run('go')

        assert limits_attributes(exporter) == expected

    @pytest.mark.parametrize('include_content', [False, True])
    async def test_a_spill_records_its_handle_only_with_content(
        self, include_content: bool, tmp_path: Path, instrumented: Instrumented
    ):
        spill = ToolOutputLimits[object](
            bands=[Band(over=100, action=Spill())], store=LocalFileOverflowStore(base_dir=tmp_path)
        )
        instrumentation, exporter = instrumented(include_content)
        result = await agent_with_tool(spill, BINARY, instrumentation).run('go')

        [part] = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert part.metadata is not None
        handle = {'tool_output_limits.handle': part.metadata['overflow_handle']} if include_content else {}
        assert limits_attributes(exporter) == {
            'tool_output_limits.action': 'spill',
            'tool_output_limits.original_size': 500,
            'tool_output_limits.reduced_size': 322,
            'tool_output_limits.size_unit': 'bytes',
            **handle,
        }

    async def test_a_spill_failure_journaled_by_an_earlier_version_falls_back_without_an_error(
        self, instrumented: Instrumented
    ):
        """An earlier version journaled a failed spill as `None`: a replay still falls back, with no error type."""
        durability = RecordingDurability(
            journal={'replayed__capability__tool_output_limits.spill': {'value': None, 'usage_delta': {}}}
        )
        instrumentation, exporter = instrumented(False)
        agent: Agent[object, str] = Agent(
            TestModel(call_tools=['big_tool']),
            name='replayed',
            capabilities=[limits(Spill(then=Truncate(max_chars=50))), durability, instrumentation],
        )

        @agent.tool_plain
        def big_tool() -> str:
            return TEXT

        await agent.run('go')

        assert limits_attributes(exporter) == snapshot(
            {
                'tool_output_limits.action': 'truncate',
                'tool_output_limits.original_size': 500,
                'tool_output_limits.reduced_size': 50,
                'tool_output_limits.size_unit': 'chars',
                'tool_output_limits.fallback_from': 'spill',
            }
        )
