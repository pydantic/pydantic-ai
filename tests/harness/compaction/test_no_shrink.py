"""Automatic `SummarizingCompaction` never commits a history that isn't smaller than the one it replaces (#11436)."""

from __future__ import annotations

from typing import Any

import pytest
from dirty_equals import IsInt
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    estimate_token_count,
)

try:
    from logfire.testing import CaptureLogfire
except ImportError:  # pragma: no cover
    pytest.skip('logfire not installed', allow_module_level=True)

VERBOSE_SUMMARY = '\n\n'.join(
    f'## {heading}\n' + 'The user is planning the search relevance upgrade with Elena and wants a next step. ' * 2
    for heading in ('Intent', 'Key decisions', 'Current state', 'Next steps')
)
"""A summary longer than the short turns it replaces, as a small model can write for the default prompt."""


def _history() -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for turn in range(6):
        history.append(ModelRequest(parts=[UserPromptPart(f'Turn {turn}: short user message.')]))
        history.append(ModelResponse(parts=[TextPart(f'Short reply {turn}.')]))
    return history


def _summarizer(summary: str, calls: list[str]) -> FunctionModel:
    def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(summary)
        return ModelResponse(parts=[TextPart(summary)])

    return FunctionModel(summarize)


def _agent(
    compaction: SummarizingCompaction[None] | FallbackCompaction[None], sent: list[list[ModelMessage]]
) -> Agent[None, str]:
    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    agent = Agent(FunctionModel(answer), deps_type=type(None), capabilities=[compaction])
    agent.instrument = InstrumentationSettings()
    return agent


def _compact_spans(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    return [span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'compact_messages']


def _compaction_attributes(span: dict[str, Any]) -> dict[str, Any]:
    """The attributes compaction sets on its span, without the ones core's instrumentation adds."""
    return {
        key: value
        for key, value in span['attributes'].items()
        if key.startswith('compaction.') or key == 'gen_ai.conversation.compacted'
    }


def _has_summary(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, CompactionPart) for message in messages for part in message.parts)


class TestSummarizingCompaction:
    @pytest.mark.parametrize('keep_tokens', [None, 5], ids=['keep_messages', 'keep_tokens'])
    async def test_a_summary_that_would_grow_the_history_is_not_committed(
        self, keep_tokens: int | None, capfire: CaptureLogfire
    ) -> None:
        calls: list[str] = []
        sent: list[list[ModelMessage]] = []
        compaction = SummarizingCompaction[None](
            model=_summarizer(VERBOSE_SUMMARY, calls),
            max_messages=6,
            keep_messages=2,
            keep_tokens=keep_tokens,
            receipts=True,
        )
        history = _history()

        result = await _agent(compaction, sent).run('Next?', message_history=history)

        assert calls == [VERBOSE_SUMMARY]
        # The request goes out with the history as it was, and the run's history keeps it.
        assert sent[0][:-1] == history
        assert not _has_summary(result.all_messages())
        [span] = _compact_spans(capfire)
        attributes = _compaction_attributes(span)
        assert attributes == snapshot(
            {
                'compaction.strategy': 'SummarizingCompaction',
                'compaction.skipped': 'not_smaller',
                'compaction.messages_before': 13,
                'compaction.messages_after': IsInt(),
                'compaction.tokens_before': 62,
                'compaction.tokens_after': IsInt(),
            }
        )
        assert attributes['compaction.tokens_after'] > attributes['compaction.tokens_before']


class TestFallbackCompaction:
    async def test_a_summary_that_would_grow_the_history_falls_through_to_the_next_strategy(
        self, capfire: CaptureLogfire
    ) -> None:
        calls: list[str] = []
        sent: list[list[ModelMessage]] = []
        compaction = FallbackCompaction[None](
            max_tokens=40,
            fallback_chain=[
                SummarizingCompaction(
                    model=_summarizer(VERBOSE_SUMMARY, calls), max_messages=1, keep_messages=2, receipts=True
                ),
                SlidingWindowCompaction(max_messages=1, keep_messages=2, receipts=True),
            ],
        )
        history = _history()

        await _agent(compaction, sent).run('Next?', message_history=history)

        assert calls == [VERBOSE_SUMMARY]
        assert not _has_summary(sent[0])
        assert estimate_token_count(sent[0]) < estimate_token_count(history)
        skipped, compacted = _compact_spans(capfire)
        assert _compaction_attributes(skipped) == snapshot(
            {
                'compaction.strategy': 'SummarizingCompaction',
                'compaction.skipped': 'not_smaller',
                'compaction.messages_before': 13,
                'compaction.messages_after': 5,
                'compaction.tokens_before': 62,
                'compaction.tokens_after': 239,
            }
        )
        assert _compaction_attributes(compacted) == snapshot(
            {
                'gen_ai.conversation.compacted': True,
                'compaction.strategy': 'FallbackCompaction',
                'compaction.messages_before': 13,
                'compaction.messages_after': 3,
                'compaction.tokens_before': 62,
                'compaction.tokens_after': 51,
            }
        )
        # Only the strategy whose result was kept leaves a receipt.
        assert [event['attributes']['compaction.receipt.strategy'] for event in compacted['events']] == [
            'SlidingWindowCompaction'
        ]

    async def test_a_summary_that_shrinks_the_history_is_kept(self, capfire: CaptureLogfire) -> None:
        calls: list[str] = []
        sent: list[list[ModelMessage]] = []
        compaction = FallbackCompaction[None](
            max_tokens=40,
            fallback_chain=[
                SummarizingCompaction(model=_summarizer('Short.', calls), max_messages=1, keep_messages=2),
                SlidingWindowCompaction(max_messages=1, keep_messages=2),
            ],
        )

        await _agent(compaction, sent).run('Next?', message_history=_history())

        assert calls == ['Short.']
        assert _has_summary(sent[0])
        [span] = _compact_spans(capfire)
        assert _compaction_attributes(span) == snapshot(
            {
                'gen_ai.conversation.compacted': True,
                'compaction.strategy': 'FallbackCompaction',
                'compaction.messages_before': 13,
                'compaction.messages_after': IsInt(),
                'compaction.tokens_before': 62,
                'compaction.tokens_after': IsInt(),
            }
        )

    async def test_the_history_is_unchanged_when_no_strategy_shrinks_it(self, capfire: CaptureLogfire) -> None:
        calls: list[str] = []
        sent: list[list[ModelMessage]] = []
        compaction = FallbackCompaction[None](
            max_tokens=40,
            fallback_chain=[
                SummarizingCompaction(model=_summarizer(VERBOSE_SUMMARY, calls), max_messages=1, keep_messages=2)
            ],
        )
        history = _history()

        result = await _agent(compaction, sent).run('Next?', message_history=history)

        assert calls == [VERBOSE_SUMMARY]
        assert sent[0][:-1] == history
        assert result.all_messages()[:-2] == history
        [span] = _compact_spans(capfire)
        assert _compaction_attributes(span) == snapshot(
            {
                'compaction.strategy': 'SummarizingCompaction',
                'compaction.skipped': 'not_smaller',
                'compaction.messages_before': 13,
                'compaction.messages_after': 4,
                'compaction.tokens_before': 62,
                'compaction.tokens_after': 195,
            }
        )

    @pytest.mark.parametrize('rejected_first', [True, False], ids=['rejected_then_raises', 'raises_then_rejected'])
    async def test_a_rejected_summary_next_to_a_failing_strategy_leaves_the_history_unchanged(
        self, rejected_first: bool
    ) -> None:
        def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ModelAPIError('summarizer', 'unavailable')

        calls: list[str] = []
        sent: list[list[ModelMessage]] = []
        rejected = SummarizingCompaction[None](
            model=_summarizer(VERBOSE_SUMMARY, calls), max_messages=1, keep_messages=2
        )
        failing = SummarizingCompaction[None](model=FunctionModel(fail), max_messages=1, keep_messages=2)
        chain = [rejected, failing] if rejected_first else [failing, rejected]
        history = _history()

        await _agent(FallbackCompaction[None](max_tokens=40, fallback_chain=chain), sent).run(
            'Next?', message_history=history
        )

        assert calls == [VERBOSE_SUMMARY]
        assert sent[0][:-1] == history
