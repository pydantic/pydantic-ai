"""A summary that fails on the automatic path ends the compaction attempt, not the run (#12307)."""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
)

try:
    from logfire.testing import CaptureLogfire
except ImportError:  # pragma: no cover
    pytest.skip('logfire not installed', allow_module_level=True)


def _history() -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for turn in range(6):
        history.append(ModelRequest(parts=[UserPromptPart(f'Turn {turn}: short user message.')]))
        history.append(ModelResponse(parts=[TextPart(f'Short reply {turn}.')]))
    return history


def _imitate_tool_call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A toolless summarizer imitating a tool call it read in the transcript."""
    return ModelResponse(parts=[ToolCallPart('lookup', {'query': 'x'})])


def _empty_summary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(' \n')])


def _unavailable(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError('summarizer', 'unavailable')


FAILING_SUMMARIZERS = {
    'tool_call': (FunctionModel(_imitate_tool_call), 'UnexpectedModelBehavior'),
    'empty_summary': (FunctionModel(_empty_summary), 'UnexpectedModelBehavior'),
    'api_error': (FunctionModel(_unavailable), 'ModelAPIError'),
    'fallback_model': (
        FallbackModel(FunctionModel(_unavailable), FunctionModel(_unavailable)),
        'FallbackExceptionGroup',
    ),
}


def _agent(
    compaction: SummarizingCompaction[None] | FallbackCompaction[None], sent: list[list[ModelMessage]]
) -> Agent[None, str]:
    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    agent = Agent(FunctionModel(answer), deps_type=type(None), capabilities=[compaction])
    agent.instrument = InstrumentationSettings()
    return agent


def _compaction_attributes(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    """The attributes compaction sets on each `compact_messages` span, without the ones core adds."""
    return [
        {
            key: value
            for key, value in span['attributes'].items()
            if key.startswith('compaction.') or key in ('gen_ai.conversation.compacted', 'error.type')
        }
        for span in capfire.exporter.exported_spans_as_dict()
        if span['name'] == 'compact_messages'
    ]


def _has_summary(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, CompactionPart) for message in messages for part in message.parts)


class TestSummarizingCompaction:
    @pytest.mark.parametrize('failure', FAILING_SUMMARIZERS)
    async def test_a_failed_summary_leaves_the_history_unchanged(self, failure: str, capfire: CaptureLogfire) -> None:
        model, error_type = FAILING_SUMMARIZERS[failure]
        sent: list[list[ModelMessage]] = []
        compaction = SummarizingCompaction[None](model=model, max_messages=6, keep_messages=2, receipts=True)
        history = _history()

        result = await _agent(compaction, sent).run('Next?', message_history=history)

        assert result.output == 'ok'
        assert sent[0][:-1] == history
        assert not _has_summary(result.all_messages())
        assert _compaction_attributes(capfire) == [
            {
                'compaction.strategy': 'SummarizingCompaction',
                'compaction.skipped': 'summary_failed',
                'error.type': error_type,
                'compaction.messages_before': 13,
                'compaction.tokens_before': 62,
            }
        ]

    async def test_a_usage_limit_hit_by_the_summary_still_stops_the_run(self) -> None:
        sent: list[list[ModelMessage]] = []
        compaction = SummarizingCompaction[None](model=FunctionModel(_empty_summary), max_messages=6, keep_messages=2)

        # One request is reserved for the parent, so the summary's retry exceeds the parent's own budget.
        with pytest.raises(UsageLimitExceeded):
            await _agent(compaction, sent).run(
                'Next?', message_history=_history(), usage_limits=UsageLimits(request_limit=2)
            )
        assert sent == []

    async def test_a_programming_error_still_stops_the_run(self) -> None:
        def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise UserError('misconfigured summarizer')

        sent: list[list[ModelMessage]] = []
        compaction = SummarizingCompaction[None](model=FunctionModel(broken), max_messages=6, keep_messages=2)

        with pytest.raises(UserError, match='misconfigured summarizer'):
            await _agent(compaction, sent).run('Next?', message_history=_history())
        assert sent == []

    @pytest.mark.parametrize('entrypoint', ['compact', 'compact_now'])
    async def test_an_explicit_compaction_still_raises(self, entrypoint: str) -> None:
        compaction = SummarizingCompaction[None](
            model=FunctionModel(_imitate_tool_call), max_messages=6, keep_messages=2
        )

        with pytest.raises(UnexpectedModelBehavior, match="Tool 'lookup' exceeded max retries count of 1"):
            if entrypoint == 'compact_now':
                await compact_now(compaction, _history(), model=FunctionModel(_imitate_tool_call))
            else:
                await compaction.compact(
                    _history(), RunContext(deps=None, model=FunctionModel(_imitate_tool_call), usage=RunUsage())
                )

    async def test_a_failed_attempt_discards_its_receipts(self, capfire: CaptureLogfire) -> None:
        class FailsAfterWritingReceipt(SummarizingCompaction[None]):
            async def compact(self, messages: list[ModelMessage], ctx: RunContext[None]) -> list[ModelMessage]:
                await super().compact(messages, ctx)
                raise UnexpectedModelBehavior('failed after writing the receipt')

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('Summary.')])

        compaction = FailsAfterWritingReceipt(
            model=FunctionModel(summarize), max_messages=6, keep_messages=2, receipts=True
        )
        sent: list[list[ModelMessage]] = []
        history = _history()

        await _agent(compaction, sent).run('Next?', message_history=history)

        assert sent[0][:-1] == history
        [span] = [s for s in capfire.exporter.exported_spans_as_dict() if s['name'] == 'compact_messages']
        assert span['attributes']['compaction.skipped'] == 'summary_failed'
        assert not span.get('events') or all(event['name'] != 'compaction.receipt' for event in span['events'])


class TestFallbackCompaction:
    async def test_a_failed_summary_falls_through_to_the_next_strategy_by_default(self) -> None:
        sent: list[list[ModelMessage]] = []
        compaction = FallbackCompaction[None](
            max_tokens=40,
            fallback_chain=[
                SummarizingCompaction(model=FunctionModel(_imitate_tool_call), max_messages=1, keep_messages=2),
                SlidingWindowCompaction(max_messages=1, keep_messages=2, receipts=False),
            ],
        )

        await _agent(compaction, sent).run('Next?', message_history=_history())

        assert not _has_summary(sent[0])
        assert len(sent[0]) == snapshot(3)
