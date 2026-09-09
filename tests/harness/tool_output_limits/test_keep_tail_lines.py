from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturn, ToolReturnPart, UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.tool_output_limits import Band, Passthrough, ToolOutputLimits, Truncate, TruncationStrategy

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def _run(value: object, capability: ToolOutputLimits[object]) -> tuple[ToolReturnPart, list[UserPromptPart]]:
    agent = Agent(TestModel(call_tools=['output']), capabilities=[capability])

    @agent.tool_plain
    def output() -> object:
        return value

    result = await agent.run('go')
    parts = [part for message in result.all_messages() for part in message.parts]
    returns = [part for part in parts if isinstance(part, ToolReturnPart)]
    assert len(returns) == 1
    return returns[0], [part for part in parts if isinstance(part, UserPromptPart)]


async def _truncate(text: str, action: Truncate) -> str:
    part, _ = await _run(text, ToolOutputLimits(bands=[Band(over=0, action=action)]))
    assert isinstance(part.content, str)
    return part.content


class TestKeepTailLines:
    @pytest.mark.parametrize('strategy', TruncationStrategy)
    @pytest.mark.parametrize('budget', [-1, 0, 1, 100, 200, 2000])
    async def test_disabled_is_unchanged(self, strategy: TruncationStrategy, budget: int):
        text = 'header\n' + 'body\r\n' * 200 + 'exit=0\n'
        fallback = Passthrough()
        original = Truncate(strategy, budget, fallback)
        disabled = Truncate(strategy, budget, fallback, keep_tail_lines=0)
        assert original.then is fallback
        assert original.keep_tail_lines == 0
        assert await _truncate(text, original) == await _truncate(text, disabled)

    def test_negative_line_count(self):
        with pytest.raises(ValueError, match='keep_tail_lines'):
            Truncate(keep_tail_lines=-1)

    @pytest.mark.parametrize('strategy', TruncationStrategy)
    @pytest.mark.parametrize('budget', [-1, 0, 6, 7])
    async def test_nonpositive_and_unnecessary_truncation(self, strategy: TruncationStrategy, budget: int):
        text = 'a\nb\nc\n'
        result = await _truncate(text, Truncate(strategy, budget, keep_tail_lines=2))
        assert result == (text if budget > 0 else '')

    @pytest.mark.parametrize('strategy', [TruncationStrategy.head, TruncationStrategy.head_tail])
    @pytest.mark.parametrize(
        'suffix,lines',
        [
            ('status=ok', 1),
            ('status=ok\n', 1),
            ('pid=123\nexit=0', 2),
            ('pid=123\nexit=0\n', 2),
            ('status=ok\r\n', 1),
            ('pid=123\r\nexit=0\r\n', 2),
            ('\n', 1),
            ('status=ok\n\n', 2),
            ('\r\n', 1),
            ('状态=好🙂\n', 1),
            ('state\u2028pid\x85ok\rvalue', 1),
        ],
    )
    async def test_preserves_complete_lines(self, strategy: TruncationStrategy, suffix: str, lines: int):
        text = 'x' * 500 + '\n' + suffix
        expected = (
            f'\n\n[truncated: {len(text) - len(suffix):,} chars omitted from the middle; '
            f'showing first 0 + last {len(suffix):,} of {len(text):,} chars]\n\n{suffix}'
        )
        result = await _truncate(text, Truncate(strategy, len(expected), keep_tail_lines=lines))
        assert result == expected

    @pytest.mark.parametrize('strategy', TruncationStrategy)
    @pytest.mark.parametrize('difference', [-1, 0, 1])
    async def test_tail_budget_boundary(self, strategy: TruncationStrategy, difference: int):
        suffix = 'process_id=' + '1234567890' * 10
        text = 'body\n' * 100 + suffix
        budget = len(suffix) + difference
        result = await _truncate(text, Truncate(strategy, budget, keep_tail_lines=1))
        if difference < 0:
            expected = await _truncate(text, Truncate(TruncationStrategy.tail, budget))
        else:
            expected = text[-budget:]
        assert result == expected
        assert len(result) <= budget

    @pytest.mark.parametrize('strategy', [TruncationStrategy.head, TruncationStrategy.head_tail])
    @pytest.mark.parametrize('difference', [-1, 0, 1])
    async def test_marker_budget_boundary(self, strategy: TruncationStrategy, difference: int):
        text = 'x' * 500 + '\nexit=0'
        marker = '\n\n[truncated: 501 chars omitted from the middle; showing first 0 + last 6 of 507 chars]\n\n'
        budget = len(marker) + 6 + difference
        result = await _truncate(text, Truncate(strategy, budget, keep_tail_lines=1))
        if difference < 0:
            assert result == text[-budget:]
        elif difference == 0:
            assert result == marker + 'exit=0'
        else:
            assert result.endswith('exit=0') and '[truncated:' in result
        assert len(result) <= budget

    @pytest.mark.parametrize('strategy', [TruncationStrategy.head, TruncationStrategy.head_tail])
    async def test_retained_counts_and_body_distribution(self, strategy: TruncationStrategy):
        text = ''.join(chr(0x4E00 + i) for i in range(1000)) + '\npid=123\nexit=0'
        head, tail = (100, 14) if strategy is TruncationStrategy.head else (40, 74)
        expected = (
            f'{text[:head]}\n\n[truncated: 901 chars omitted from the middle; '
            f'showing first {head} + last {tail} of 1,015 chars]\n\n{text[-tail:]}'
        )
        assert await _truncate(text, Truncate(strategy, len(expected), keep_tail_lines=2)) == expected

    @pytest.mark.parametrize('text,lines', [('x' * 1000, 1), ('x\ny\nz' * 200, 10**12)])
    @pytest.mark.parametrize('strategy', TruncationStrategy)
    async def test_oversized_requested_lines_use_tail(self, text: str, lines: int, strategy: TruncationStrategy):
        result = await _truncate(text, Truncate(strategy, 100, keep_tail_lines=lines))
        assert result == await _truncate(text, Truncate(TruncationStrategy.tail, 100))

    async def test_tail_strategy_keeps_existing_marker_when_lines_fit(self):
        text = 'body\n' * 200 + 'pid=123\nexit=0'
        result = await _truncate(text, Truncate(TruncationStrategy.tail, 200, keep_tail_lines=2))
        assert result == await _truncate(text, Truncate(TruncationStrategy.tail, 200))

    async def test_tool_return_fields_and_metadata(self):
        value = 'v' * 1000 + '\nvalue_end'
        content = 'c' * 1000 + '\ncontent_end'
        capability: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=0, action=Truncate(TruncationStrategy.head, 200, keep_tail_lines=1))]
        )
        part, prompts = await _run(ToolReturn(return_value=value, content=content, metadata={'id': 7}), capability)
        assert part.metadata == {'id': 7}
        assert isinstance(part.content, str) and part.content.endswith('value_end') and len(part.content) <= 200
        assert len(prompts) == 2
        assert isinstance(prompts[-1].content, str)
        assert prompts[-1].content.endswith('content_end') and len(prompts[-1].content) <= 200

    async def test_serializer_ansi_and_token_threshold(self):
        text = 'body\n' * 200 + '\x1b[31mpid=123\nexit=0\x1b[0m'

        def serializer(value: object) -> str:
            assert value == {'status': 'ok'}
            return text

        capability: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=1, action=Truncate(TruncationStrategy.head, 200, keep_tail_lines=2))],
            serializer=serializer,
            strip_ansi=True,
            over_tokens=True,
            tokenizer=lambda text: 1,
        )
        part, _ = await _run({'status': 'ok'}, capability)
        assert isinstance(part.content, str)
        assert part.content.endswith('pid=123\nexit=0') and '\x1b' not in part.content
        assert len(part.content) <= 200

    async def test_binary_fallback_is_unchanged(self):
        capability: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=0, action=Truncate(max_chars=1, then=Passthrough(), keep_tail_lines=1))]
        )
        part, _ = await _run(b'raw\nbytes', capability)
        assert part.content == b'raw\nbytes'
