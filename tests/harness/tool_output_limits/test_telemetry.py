"""`ToolOutputLimits` says on the call's `execute_tool` span how it reduced the result."""

from __future__ import annotations

from pathlib import Path

import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Instrumentation
from pydantic_ai.messages import ToolReturn, ToolReturnPart, UserPromptPart
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
                snapshot(
                    {
                        'tool_output_limits.action': 'truncate',
                        'tool_output_limits.original_size': 500,
                        'tool_output_limits.reduced_size': 50,
                        'tool_output_limits.size_unit': 'chars',
                    }
                ),
                id='truncate',
            ),
            pytest.param(limits(Truncate(max_chars=1_000)), TEXT, snapshot({}), id='truncate-cuts-nothing'),
            pytest.param(
                ToolOutputLimits[object](
                    bands=[Band(over=100, action=Truncate(max_chars=100))],
                    over_tokens=True,
                    tokenizer=lambda text: len(text) // 4,
                ),
                'x' * 2_000,
                snapshot(
                    {
                        'tool_output_limits.action': 'truncate',
                        'tool_output_limits.original_size': 500,
                        'tool_output_limits.reduced_size': 25,
                        'tool_output_limits.size_unit': 'tokens',
                    }
                ),
                id='tokens',
            ),
            pytest.param(
                limits(Summarize(summarize=short_summarize, then=Truncate(max_chars=50))),
                TEXT,
                snapshot(
                    {
                        'tool_output_limits.action': 'summarize',
                        'tool_output_limits.original_size': 500,
                        'tool_output_limits.reduced_size': 93,
                        'tool_output_limits.size_unit': 'chars',
                    }
                ),
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
                    }
                ),
                id='failed-summary',
            ),
            # The fallback cuts nothing, and is still recorded: the store failing is what an operator needs to see.
            pytest.param(
                limits(Spill(then=Truncate(max_chars=1_000))),
                TEXT,
                snapshot(
                    {
                        'tool_output_limits.action': 'truncate',
                        'tool_output_limits.original_size': 500,
                        'tool_output_limits.reduced_size': 500,
                        'tool_output_limits.size_unit': 'chars',
                        'tool_output_limits.fallback_from': 'spill',
                    }
                ),
                id='failed-spill-falls-back',
            ),
            # The spill fails, then its summary: `fallback_from` keeps the band's own action.
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
                    }
                ),
                id='chain-of-failures',
            ),
            # The default `Spill(then=Truncate())`: the store fails and `Truncate` can't take binary.
            pytest.param(
                ToolOutputLimits[object](store=FailingStore()),
                b'\x00' * 200_000,
                snapshot({'tool_output_limits.fallback_from': 'spill'}),
                id='binary-store-fails',
            ),
            pytest.param(
                limits(Truncate()),
                BINARY,
                snapshot({'tool_output_limits.fallback_from': 'truncate'}),
                id='binary-under-bare-truncate',
            ),
            pytest.param(
                limits(Truncate(max_chars=50)),
                ToolReturn(return_value='small', content='y' * 300),
                snapshot(
                    {
                        'tool_output_limits.content.action': 'truncate',
                        'tool_output_limits.content.original_size': 300,
                        'tool_output_limits.content.reduced_size': 50,
                        'tool_output_limits.content.size_unit': 'chars',
                    }
                ),
                id='separate-content',
            ),
            pytest.param(
                ToolOutputLimits[object](store=FailingStore()),
                ToolReturn(return_value='small', content='y' * 200_000),
                snapshot(
                    {
                        'tool_output_limits.content.action': 'truncate',
                        'tool_output_limits.content.original_size': 200000,
                        'tool_output_limits.content.reduced_size': 4000,
                        'tool_output_limits.content.size_unit': 'chars',
                        'tool_output_limits.content.fallback_from': 'spill',
                    }
                ),
                id='separate-content-store-fails',
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

    async def test_a_spilled_value_and_content_are_recorded_side_by_side(
        self, tmp_path: Path, instrumented: Instrumented
    ):
        spill = ToolOutputLimits[object](
            bands=[Band(over=100, action=Spill())], store=LocalFileOverflowStore(base_dir=tmp_path)
        )
        instrumentation, exporter = instrumented(False)
        result = await agent_with_tool(spill, ToolReturn(return_value=BINARY, content='y' * 300), instrumentation).run(
            'go'
        )

        parts = [part for message in result.all_messages() for part in message.parts]
        [value_preview] = [part.content for part in parts if isinstance(part, ToolReturnPart)]
        [content_preview] = [
            part.content for part in parts if isinstance(part, UserPromptPart) and part.content != 'go'
        ]
        assert isinstance(value_preview, str) and isinstance(content_preview, str)
        assert limits_attributes(exporter) == {
            'tool_output_limits.action': 'spill',
            'tool_output_limits.original_size': 500,
            'tool_output_limits.reduced_size': len(value_preview.encode()),
            'tool_output_limits.size_unit': 'bytes',
            'tool_output_limits.content.action': 'spill',
            'tool_output_limits.content.original_size': 300,
            'tool_output_limits.content.reduced_size': len(content_preview),
            'tool_output_limits.content.size_unit': 'chars',
        }

    async def test_a_tokenizer_that_rejects_the_replacement_leaves_the_result_alone(self, instrumented: Instrumented):
        """Measuring the replacement is for the trace only, so its failure drops `reduced_size`, not the run."""

        def tokenizer(text: str) -> int:
            if text != 'x' * 2_000:
                raise ValueError('unknown token')
            return 500

        tool_output_limits = ToolOutputLimits[object](
            bands=[Band(over=100, action=Truncate(max_chars=100))], over_tokens=True, tokenizer=tokenizer
        )
        instrumentation, exporter = instrumented(False)
        result = await agent_with_tool(tool_output_limits, 'x' * 2_000, instrumentation).run('go')

        [part] = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert len(part.model_response_str()) == 100
        assert limits_attributes(exporter) == {
            'tool_output_limits.action': 'truncate',
            'tool_output_limits.original_size': 500,
            'tool_output_limits.size_unit': 'tokens',
        }
