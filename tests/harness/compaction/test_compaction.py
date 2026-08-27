"""Tests for pydantic_ai_harness.compaction capabilities."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnContent,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets._tool_search import parse_discovered_tools
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    DeduplicateFileReads,
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    TranscriptHandleProvider,
    WarnNearLimits,
    estimate_context_tokens,
    estimate_token_count,
    is_pinned,
    pin,
)
from pydantic_ai_harness.compaction._clamp_oversized_messages import (
    _CLAMP_ARGS_KEY,
    _CLAMP_MARKER,
)
from pydantic_ai_harness.compaction._shared import (
    _history_changed,
    _is_safe_cutoff,
    compact_with_span,
    find_first_user_message,
    find_safe_cutoff,
    find_token_cutoff,
    iter_tool_pairs,
    prepend_first_user_message,
)
from pydantic_ai_harness.compaction._summarizing_compaction import (
    _SUMMARY_PREFIX,
    _extract_previous_summary,
    _extract_system_prompts,
    _format_messages,
)

try:
    from logfire.testing import CaptureLogfire

    logfire_installed = True
except ImportError:  # pragma: no cover
    logfire_installed = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    requests: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    usage_limits: UsageLimits | None = None,
) -> Any:
    """Build a minimal RunContext-like object for testing hooks."""

    usage = RunUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)

    @dataclasses.dataclass
    class _FakeCtx:
        usage: RunUsage
        usage_limits: UsageLimits | None = None
        model: Model = dataclasses.field(default_factory=TestModel)
        deps: None = None
        tracer: Tracer = dataclasses.field(default_factory=NoOpTracer)
        # A declared field, like the real `RunContext`: a strategy reached from
        # `before_model_request` sees a context rebuilt for the request's model, and an
        # attribute merely assigned onto the fake would not survive that rebuild.
        capabilities: dict[str, AbstractCapability[None]] = dataclasses.field(
            default_factory=dict[str, AbstractCapability[None]]
        )

    return _FakeCtx(usage=usage, usage_limits=usage_limits)


def _make_request_context(messages: list[ModelMessage], model: Model | None = None) -> ModelRequestContext:
    """Build a ModelRequestContext wrapping the given messages.

    *model* is the model the request would be sent to. Pass the context's own model to
    reproduce a run where no capability replaced it, which is the common case.
    """

    @dataclasses.dataclass
    class _FakeModel:
        model_id: str = 'test-model'
        model_name: str = 'test-model'

    return ModelRequestContext(
        model=model if model is not None else _FakeModel(),  # type: ignore[arg-type]
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _tool_call(tool_name: str, call_id: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args='{}', tool_call_id=call_id)])


def _tool_return(tool_name: str, call_id: str, content: str = 'ok') -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name=tool_name, content=content, tool_call_id=call_id)])


# ---------------------------------------------------------------------------
# estimate_token_count
# ---------------------------------------------------------------------------


class TestEstimateTokenCount:
    def test_empty(self):
        assert estimate_token_count([]) == 0

    def test_user_message(self):
        msgs: list[ModelMessage] = [_user('hello world')]  # 11 chars => 2 tokens
        assert estimate_token_count(msgs) == 11 // 4

    def test_system_prompt(self):
        msgs: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart(content='x' * 100)])]
        assert estimate_token_count(msgs) == 25

    def test_assistant_text(self):
        msgs: list[ModelMessage] = [_assistant('y' * 80)]
        assert estimate_token_count(msgs) == 20

    def test_tool_call_and_return(self):
        msgs: list[ModelMessage] = [
            _tool_call('search', 'tc1'),
            _tool_return('search', 'tc1', 'result text here'),
        ]
        assert estimate_token_count(msgs) > 0


# ---------------------------------------------------------------------------
# estimate_context_tokens
# ---------------------------------------------------------------------------


def _assistant_with_usage(text: str, input_tokens: int, output_tokens: int) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=text)],
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class TestEstimateContextTokens:
    def test_no_usage_falls_back_to_heuristic(self):
        msgs: list[ModelMessage] = [_user('x' * 40), _assistant('y' * 40)]
        assert estimate_context_tokens(msgs) == estimate_token_count(msgs)

    def test_anchors_on_reported_usage(self):
        # The heuristic would say ~12 tokens; the provider said the request was 50K.
        msgs: list[ModelMessage] = [_user('x' * 40), _assistant_with_usage('short', 50_000, 500)]
        assert estimate_context_tokens(msgs) == 50_500

    def test_counts_only_messages_after_the_anchor(self):
        msgs: list[ModelMessage] = [
            _user('x' * 40),
            _assistant_with_usage('short', 50_000, 500),
            _tool_return('search', 'tc1', 'z' * 400),
        ]
        assert estimate_context_tokens(msgs) == 50_500 + 100

    def test_anchors_on_the_most_recent_usage(self):
        msgs: list[ModelMessage] = [
            _assistant_with_usage('a', 10_000, 100),
            _user('x' * 40),
            _assistant_with_usage('b', 50_000, 1_000),
        ]
        assert estimate_context_tokens(msgs) == 51_000

    def test_zero_usage_response_is_not_an_anchor(self):
        # TestModel/FunctionModel histories report no usage; the whole history falls back.
        msgs: list[ModelMessage] = [_user('x' * 40), _assistant('y' * 40)]
        assert estimate_context_tokens(msgs) == estimate_token_count(msgs) > 0

    def test_suffix_does_not_recount_unchanged_instructions(self):
        # Instructions travelled inside the anchor's input_tokens; counting the copy carried
        # by a later request would double them.
        request_before = ModelRequest(parts=[UserPromptPart(content='q')], instructions='i' * 4_000)
        request_after = ModelRequest(parts=[UserPromptPart(content='a' * 40)], instructions='i' * 4_000)
        msgs: list[ModelMessage] = [request_before, _assistant_with_usage('short', 50_000, 500), request_after]
        assert estimate_context_tokens(msgs) == 50_500 + 10

    def test_instructions_changed_after_the_anchor_are_counted(self):
        # The anchor paid for the old instructions only; a new set (dynamic instructions, or a
        # resumed history under a new prompt) must be added to the estimate.
        request_before = ModelRequest(parts=[UserPromptPart(content='q')], instructions='old')
        request_after = ModelRequest(parts=[UserPromptPart(content='a' * 40)], instructions='i' * 4_000)
        msgs: list[ModelMessage] = [request_before, _assistant_with_usage('short', 50_000, 500), request_after]
        assert estimate_context_tokens(msgs) == 50_500 + 10 + 1_000

    def test_tokenizer_applies_to_the_suffix(self):
        msgs: list[ModelMessage] = [
            _assistant_with_usage('short', 50_000, 500),
            _user('three short words'),
        ]
        assert estimate_context_tokens(msgs, tokenizer=lambda s: len(s.split())) == 50_503


# ---------------------------------------------------------------------------
# _is_safe_cutoff
# ---------------------------------------------------------------------------


class TestIsSafeCutoff:
    def test_cutoff_beyond_end(self):
        msgs: list[ModelMessage] = [_user('a'), _assistant('b')]
        assert _is_safe_cutoff(msgs, 10) is True

    def test_no_tool_pairs(self):
        msgs: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c')]
        assert _is_safe_cutoff(msgs, 1) is True

    def test_safe_when_both_sides_kept(self):
        msgs: list[ModelMessage] = [
            _user('a'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('b'),
        ]
        # Cutting before the tool pair (index 0) is safe: both call and return are kept.
        assert _is_safe_cutoff(msgs, 0) is True

    def test_unsafe_when_splitting_pair(self):
        msgs: list[ModelMessage] = [
            _user('a'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('b'),
        ]
        # Cutting at index 2: call (idx 1) is before cutoff, return (idx 2) is at cutoff (after).
        assert _is_safe_cutoff(msgs, 2) is False

    def test_safe_when_pair_entirely_discarded(self):
        msgs: list[ModelMessage] = [
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('a'),
            _assistant('b'),
        ]
        # Cutting at 2: both call and return are before cutoff (discarded together).
        assert _is_safe_cutoff(msgs, 2) is True


# ---------------------------------------------------------------------------
# find_safe_cutoff
# ---------------------------------------------------------------------------


class TestFindSafeCutoff:
    def test_keep_zero_returns_length(self):
        msgs: list[ModelMessage] = [_user('a'), _assistant('b')]
        assert find_safe_cutoff(msgs, 0) == 2

    def test_fewer_messages_than_keep(self):
        msgs: list[ModelMessage] = [_user('a')]
        assert find_safe_cutoff(msgs, 5) == 0

    def test_normal_cutoff(self):
        msgs: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        # Keep 2 => target cutoff is 2.
        assert find_safe_cutoff(msgs, 2) == 2

    def test_adjusts_for_tool_pair(self):
        msgs: list[ModelMessage] = [
            _user('a'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('b'),
            _assistant('c'),
        ]
        # Keep 3 => target cutoff is 2, but that splits the tool pair.
        # Should adjust to 1 (keep tool call and return together).
        cutoff = find_safe_cutoff(msgs, 3)
        assert cutoff == 1


# ---------------------------------------------------------------------------
# find_token_cutoff
# ---------------------------------------------------------------------------


class TestFindTokenCutoff:
    def test_already_within_budget(self):
        msgs: list[ModelMessage] = [_user('hi')]
        assert find_token_cutoff(msgs, 999999) == 0

    def test_empty(self):
        assert find_token_cutoff([], 100) == 0

    def test_trims_to_budget(self):
        # Each message contributes ~3 tokens (12 chars / 4).
        msgs: list[ModelMessage] = [_user('x' * 12) for _ in range(20)]
        cutoff = find_token_cutoff(msgs, 30)  # Budget for ~10 messages.
        assert cutoff > 0
        remaining = msgs[cutoff:]
        assert estimate_token_count(remaining) <= 30

    def test_walks_back_over_tool_pair(self):
        # The token-fit cutoff lands between a tool call and its return; the backward
        # walk must skip to a safe index that keeps the pair together.
        msgs: list[ModelMessage] = [
            _user('a' * 8),
            _tool_call('fn', 'tc1'),  # contributes 'fn' + '{}' = 4 tokens
            _tool_return('fn', 'tc1', 'b' * 4),
            _user('c' * 4),
        ]
        # messages[2:] = 8 tokens (fits), messages[1:] = 12 (does not) -> candidate is 2,
        # which splits the pair, so it walks back to 1.
        assert find_token_cutoff(msgs, 8, tokenizer=len) == 1


# ---------------------------------------------------------------------------
# SlidingWindowCompaction
# ---------------------------------------------------------------------------


class TestSlidingWindowCompaction:
    def test_validation_no_trigger(self):
        with pytest.raises(ValueError, match='At least one of max_messages, max_tokens, or max_fraction must be set'):
            SlidingWindowCompaction()

    def test_validation_negative_max_messages(self):
        with pytest.raises(ValueError, match='max_messages must be positive'):
            SlidingWindowCompaction(max_messages=0)

    def test_validation_negative_max_tokens(self):
        with pytest.raises(ValueError, match='max_tokens must be positive'):
            SlidingWindowCompaction(max_tokens=-1)

    def test_validation_negative_keep_messages(self):
        with pytest.raises(ValueError, match='keep_messages must be non-negative'):
            SlidingWindowCompaction(max_messages=10, keep_messages=-1)

    def test_validation_negative_keep_tokens(self):
        with pytest.raises(ValueError, match='keep_tokens must be non-negative'):
            SlidingWindowCompaction(max_messages=10, keep_tokens=-1)

    @pytest.mark.anyio
    async def test_no_trim_below_threshold(self):
        sw = SlidingWindowCompaction(max_messages=10, keep_messages=5)
        messages: list[ModelMessage] = [_user('a'), _assistant('b')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 2

    @pytest.mark.anyio
    async def test_trims_when_above_message_threshold(self):
        sw = SlidingWindowCompaction(max_messages=5, keep_messages=3, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_user(f'msg-{i}') for i in range(8)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) <= 3

    @pytest.mark.anyio
    async def test_trims_by_token_threshold(self):
        sw = SlidingWindowCompaction(max_tokens=10, keep_messages=2)
        messages: list[ModelMessage] = [_user('x' * 40) for _ in range(5)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) < 5

    @pytest.mark.anyio
    async def test_preserves_tool_pairs(self):
        sw = SlidingWindowCompaction(max_messages=4, keep_messages=2)
        messages: list[ModelMessage] = [
            _user('start'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('end'),
            _assistant('done'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        # Should not split the tool pair.
        assert _orphan_free(result.messages)

    @pytest.mark.anyio
    async def test_keep_tokens_mode(self):
        sw = SlidingWindowCompaction(max_messages=3, keep_tokens=10, preserve_first_user_message=False)
        # Each message = 20 chars = 5 tokens.  Total = 50 tokens.
        messages: list[ModelMessage] = [_user('x' * 20) for _ in range(10)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert estimate_token_count(result.messages) <= 10
        assert len(result.messages) < 10


class TestFallbackCompaction:
    @pytest.fixture
    def anyio_backend(self) -> str:
        # The exhausted-model test runs a real Agent; trio hits a core event-loop quirk unrelated to compaction.
        return 'asyncio'

    def test_validation_empty_chain(self):
        with pytest.raises(ValueError, match='fallback_chain must not be empty'):
            FallbackCompaction(fallback_chain=[])

    def test_validation_empty_fallback_on(self):
        with pytest.raises(ValueError, match='fallback_on must not be empty'):
            FallbackCompaction(fallback_chain=[AsyncMock()], fallback_on=())

    def test_validation_cancelled_error(self):
        with pytest.raises(ValueError, match='fallback_on must contain only Exception subclasses'):
            FallbackCompaction(
                fallback_chain=[AsyncMock()],
                fallback_on=(asyncio.CancelledError,),  # pyright: ignore[reportArgumentType]
            )

    @pytest.mark.anyio
    async def test_returns_first_success(self):
        first = AsyncMock()
        first.compact.return_value = [_user('summary')]
        second = AsyncMock()
        fallback = FallbackCompaction(fallback_chain=[first, second])

        result = await fallback.compact([_user('original')], _make_ctx())

        assert result == first.compact.return_value
        second.compact.assert_not_called()

    @pytest.mark.anyio
    async def test_fallback_receives_fresh_original_list(self):
        class MutateThenFail:
            async def compact(self, messages: list[ModelMessage], ctx: RunContext[None]) -> list[ModelMessage]:
                messages.clear()
                raise RuntimeError('summary failed')

        second = AsyncMock()
        messages: list[ModelMessage] = [_user('original')]
        second.compact.return_value = messages
        fallback = FallbackCompaction(fallback_chain=[MutateThenFail(), second], fallback_on=(RuntimeError,))

        result = await fallback.compact(messages, _make_ctx())

        assert result == messages
        assert second.compact.call_args.args[0] == messages

    @pytest.mark.anyio
    async def test_reraises_last_failure(self):
        first = AsyncMock()
        first.compact.side_effect = ValueError('first')
        second = AsyncMock()
        second.compact.side_effect = RuntimeError('last')
        fallback = FallbackCompaction(fallback_chain=[first, second], fallback_on=(Exception,))

        with pytest.raises(RuntimeError, match='last'):
            await fallback.compact([_user('original')], _make_ctx())

    @pytest.mark.anyio
    async def test_non_matching_error_does_not_fallback(self):
        first = AsyncMock()
        first.compact.side_effect = ValueError('programming error')
        second = AsyncMock()
        fallback = FallbackCompaction(fallback_chain=[first, second])

        with pytest.raises(ValueError, match='programming error'):
            await fallback.compact([_user('original')], _make_ctx())
        second.compact.assert_not_called()

    @pytest.mark.anyio
    async def test_cancellation_does_not_fallback(self):
        first = AsyncMock()
        first.compact.side_effect = asyncio.CancelledError
        second = AsyncMock()
        fallback = FallbackCompaction(fallback_chain=[first, second])

        with pytest.raises(asyncio.CancelledError):
            await fallback.compact([_user('original')], _make_ctx())
        second.compact.assert_not_called()

    @pytest.mark.anyio
    async def test_fallback_model_exhaustion_uses_next_strategy(self):
        def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise ModelAPIError('summarizer', 'rejected history')

        summarizer = SummarizingCompaction(
            model=FallbackModel(FunctionModel(fail), FunctionModel(fail)),
            max_messages=1,
            keep_messages=1,
        )
        deterministic = AsyncMock()
        deterministic.compact.return_value = [_user('trimmed')]
        fallback = FallbackCompaction(fallback_chain=[summarizer, deterministic])

        result = await fallback.compact([_user('old'), _assistant('old'), _user('recent')], _make_ctx())

        assert result == deterministic.compact.return_value
        deterministic.compact.assert_awaited_once()

    def test_with_focus_updates_only_supported_strategies(self):
        summarizer = SummarizingCompaction(max_messages=1)
        sliding = SlidingWindowCompaction(max_messages=1)
        fallback = FallbackCompaction(fallback_chain=[summarizer, sliding])

        focused = fallback.with_focus('authentication')

        assert focused is not fallback
        focused_summarizer = focused.fallback_chain[0]
        assert isinstance(focused_summarizer, SummarizingCompaction)
        assert 'Give particular weight to: authentication' in focused_summarizer.summary_prompt
        assert focused.fallback_chain[1] is sliding


# ---------------------------------------------------------------------------
# WarnNearLimits
# ---------------------------------------------------------------------------


class TestWarnNearLimits:
    def test_validation_no_limits(self):
        with pytest.raises(ValueError, match='At least one of'):
            WarnNearLimits()

    def test_validation_negative_max_iterations(self):
        with pytest.raises(ValueError, match='max_iterations must be positive'):
            WarnNearLimits(max_iterations=-1)

    def test_validation_negative_max_context_tokens(self):
        with pytest.raises(ValueError, match='max_context_tokens must be positive'):
            WarnNearLimits(max_context_tokens=0)

    def test_validation_negative_max_total_tokens(self):
        with pytest.raises(ValueError, match='max_total_tokens must be positive'):
            WarnNearLimits(max_total_tokens=-5)

    def test_validation_bad_threshold(self):
        with pytest.raises(ValueError, match='warning_threshold'):
            WarnNearLimits(max_iterations=10, warning_threshold=0)

    def test_validation_negative_critical_remaining(self):
        with pytest.raises(ValueError, match='critical_remaining_iterations'):
            WarnNearLimits(max_iterations=10, critical_remaining_iterations=-1)

    def test_validation_empty_warn_on(self):
        with pytest.raises(ValueError, match='warn_on must not be empty'):
            WarnNearLimits(max_iterations=10, warn_on=[])

    def test_validation_warn_on_without_limit(self):
        with pytest.raises(ValueError, match="'total_tokens' requires"):
            WarnNearLimits(max_iterations=10, warn_on=['total_tokens'])

    @pytest.mark.anyio
    async def test_no_warning_below_threshold(self):
        lw = WarnNearLimits(max_iterations=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=10)
        result = await lw.before_model_request(ctx, rc)
        # No warning appended.
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_iteration_warning_urgent(self):
        lw = WarnNearLimits(max_iterations=20, warning_threshold=0.7, critical_remaining_iterations=3)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        # 15/20 = 75% usage, 5 remaining > critical_remaining_iterations=3 => URGENT.
        ctx = _make_ctx(requests=15)
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 2
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        assert 'URGENT' in text.content
        assert '[WarnNearLimits]' in text.content

    @pytest.mark.anyio
    async def test_iteration_warning_critical(self):
        lw = WarnNearLimits(max_iterations=10, warning_threshold=0.7, critical_remaining_iterations=3)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=9)  # 1 remaining.
        result = await lw.before_model_request(ctx, rc)
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        assert 'CRITICAL' in text.content

    @pytest.mark.anyio
    async def test_context_window_warning(self):
        lw = WarnNearLimits(max_context_tokens=10)
        # Create a message that exceeds 70% of 10 tokens.
        messages: list[ModelMessage] = [_user('x' * 40)]  # ~10 tokens.
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 2

    @pytest.mark.anyio
    async def test_total_tokens_warning(self):
        lw = WarnNearLimits(max_total_tokens=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(input_tokens=50, output_tokens=30)  # 80 total.
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 2

    @pytest.mark.anyio
    async def test_strips_old_warnings(self):
        lw = WarnNearLimits(max_iterations=10, warning_threshold=0.7)
        old_warning = ModelRequest(parts=[UserPromptPart(content='[WarnNearLimits]\nOld warning')])
        messages: list[ModelMessage] = [_user('hi'), old_warning]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)  # Below threshold.
        result = await lw.before_model_request(ctx, rc)
        # Old warning removed, no new warning added (below threshold).
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_strips_legacy_limit_warner_markers(self):
        lw = WarnNearLimits(max_iterations=10, warning_threshold=0.7)
        legacy_warning = ModelRequest(parts=[UserPromptPart(content='[LimitWarner]\nOld warning')])
        messages: list[ModelMessage] = [_user('hi'), legacy_warning]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)  # Below threshold.
        result = await lw.before_model_request(ctx, rc)
        # Warnings injected before the rename are still recognized and stripped.
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_multiple_warnings_ordered(self):
        lw = WarnNearLimits(max_iterations=10, max_total_tokens=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=8, input_tokens=50, output_tokens=30)
        result = await lw.before_model_request(ctx, rc)
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        # Iterations should come before total_tokens.
        assert text.content.index('Iterations') < text.content.index('Total tokens')


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


class TestCompaction:
    def test_validation_no_trigger(self):
        with pytest.raises(ValueError, match='At least one of max_messages, max_tokens, or max_fraction must be set'):
            SummarizingCompaction(model='test', max_messages=None, max_tokens=None)

    def test_validation_negative_max_messages(self):
        with pytest.raises(ValueError, match='max_messages must be positive'):
            SummarizingCompaction(model='test', max_messages=0)

    def test_validation_negative_max_tokens(self):
        with pytest.raises(ValueError, match='max_tokens must be positive'):
            SummarizingCompaction(model='test', max_tokens=-1)

    def test_validation_negative_keep_messages(self):
        with pytest.raises(ValueError, match='keep_messages must be non-negative'):
            SummarizingCompaction(model='test', max_messages=10, keep_messages=-1)

    def test_validation_negative_keep_tokens(self):
        with pytest.raises(ValueError, match='keep_tokens must be non-negative'):
            SummarizingCompaction(model='test', max_messages=10, keep_tokens=-1)

    @pytest.mark.anyio
    async def test_no_compaction_below_threshold(self):
        comp = SummarizingCompaction(model='test', max_messages=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await comp.before_model_request(ctx, rc)
        assert result.messages == messages

    @pytest.mark.anyio
    async def test_summary_model_settings_override_model_defaults_without_mutation(self, anyio_backend: str):
        if anyio_backend != 'asyncio':
            pytest.skip('pydantic-ai Agent execution uses asyncio')
        observed_settings: list[ModelSettings | None] = []

        def summarize(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            observed_settings.append(info.model_settings)
            return ModelResponse(parts=[TextPart(content='A summary.')])

        model_defaults = ModelSettings(temperature=1.0, max_tokens=4_096)
        original_model_defaults = model_defaults.copy()
        model = FunctionModel(summarize, settings=model_defaults)
        summary_settings = ModelSettings(max_tokens=1_024)
        original_summary_settings = summary_settings.copy()
        comp = SummarizingCompaction(
            model=model,
            model_settings=summary_settings,
            max_messages=1,
            keep_messages=1,
        )

        await comp.compact(
            [_user('first'), _assistant('response'), _user('latest')],
            _make_ctx(),
        )

        assert observed_settings == [ModelSettings(temperature=1.0, max_tokens=1_024)]
        assert summary_settings == original_summary_settings
        assert model_defaults == original_model_defaults

    @pytest.mark.anyio
    async def test_compaction_replaces_old_messages(self):
        comp = SummarizingCompaction(model='test:m', max_messages=3, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [
            _user('first'),
            _assistant('response 1'),
            _user('second'),
            _assistant('response 2'),
            _user('third'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Summary of conversation.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        # Should have summary message + 1 kept message.
        assert len(result.messages) == 2
        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        # The summary should be in a SystemPromptPart.
        sys_parts = [p for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert len(sys_parts) >= 1
        assert 'Summary of conversation.' in sys_parts[-1].content

    @pytest.mark.anyio
    async def test_compaction_preserves_system_prompts(self):
        comp = SummarizingCompaction(model='test:m', max_messages=3, keep_messages=1)
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='You are a helpful assistant.')]),
            _user('first'),
            _assistant('response 1'),
            _user('second'),
            _assistant('response 2'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'A summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        # Should have the original system prompt preserved.
        sys_contents = [p.content for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert 'You are a helpful assistant.' in sys_contents

    @pytest.mark.anyio
    async def test_compaction_preserves_tool_pairs(self):
        comp = SummarizingCompaction(model='test:m', max_messages=4, keep_messages=2)
        messages: list[ModelMessage] = [
            _user('start'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('middle'),
            _assistant('response'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        # Tool pairs in remaining messages should be intact.
        assert _orphan_free(result.messages)

    @pytest.mark.anyio
    async def test_compaction_token_trigger(self):
        comp = SummarizingCompaction(model='test:m', max_tokens=5, keep_messages=1)
        messages: list[ModelMessage] = [_user('x' * 40) for _ in range(5)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Token-based summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        assert len(result.messages) >= 1
        # Summary message should exist.
        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)

    @pytest.mark.anyio
    async def test_compaction_keep_tokens_mode(self):
        comp = SummarizingCompaction(model='test:m', max_messages=3, keep_tokens=5)
        messages: list[ModelMessage] = [_user('x' * 40) for _ in range(5)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Token-keep summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        assert len(result.messages) >= 1


# ---------------------------------------------------------------------------
# _format_messages
# ---------------------------------------------------------------------------


class TestFormatMessages:
    def test_user_and_assistant(self):
        msgs: list[ModelMessage] = [_user('hi'), _assistant('hello')]
        text = _format_messages(msgs)
        assert 'User: hi' in text
        assert 'Assistant: hello' in text

    def test_system_prompt(self):
        msgs: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart(content='be helpful')])]
        text = _format_messages(msgs)
        assert 'System: be helpful' in text

    def test_tool_call_and_return(self):
        msgs: list[ModelMessage] = [
            _tool_call('search', 'tc1'),
            _tool_return('search', 'tc1', 'found it'),
        ]
        text = _format_messages(msgs)
        assert 'Tool Call [search]' in text
        assert 'Tool [search]: found it' in text

    def test_long_tool_return_truncated(self):
        msgs: list[ModelMessage] = [_tool_return('fn', 'tc1', 'x' * 600)]
        text = _format_messages(msgs)
        assert '...' in text


# ---------------------------------------------------------------------------
# _extract_system_prompts
# ---------------------------------------------------------------------------


class TestExtractSystemPrompts:
    def test_extracts_leading_system_parts(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='sys1')]),
            _user('hi'),
        ]
        parts = _extract_system_prompts(msgs)
        assert len(parts) == 1
        assert parts[0].content == 'sys1'

    def test_stops_at_non_system(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='sys1'), UserPromptPart(content='hi')]),
        ]
        parts = _extract_system_prompts(msgs)
        assert len(parts) == 1

    def test_empty_when_no_system(self):
        msgs: list[ModelMessage] = [_user('hi')]
        parts = _extract_system_prompts(msgs)
        assert parts == []

    def test_stops_at_non_request(self):
        msgs: list[ModelMessage] = [_assistant('hello'), _user('hi')]
        parts = _extract_system_prompts(msgs)
        assert parts == []


# ---------------------------------------------------------------------------
# Package-level exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_exposed_under_submodule_and_top_level(self):
        import pydantic_ai_harness
        import pydantic_ai_harness.compaction as compaction

        names = [
            'SlidingWindowCompaction',
            'ClearToolResults',
            'DeduplicateFileReads',
            'WarnNearLimits',
            'SummarizingCompaction',
            'TieredCompaction',
            'FallbackCompaction',
        ]
        for name in names:
            assert hasattr(compaction, name)
            assert getattr(pydantic_ai_harness, name) is getattr(compaction, name)


# ---------------------------------------------------------------------------
# Additional coverage — multi-modal content, edge cases
# ---------------------------------------------------------------------------


class TestUserPromptMultiModal:
    """Cover _user_prompt_text_for_counting and _user_prompt_text for non-string UserContent."""

    def test_estimate_with_text_content_parts(self):
        from pydantic_ai.messages import TextContent

        part = UserPromptPart(content=[TextContent(content='hello')])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        # 5 chars / 4 = 1 token.
        assert estimate_token_count(msgs) == 1

    def test_estimate_with_str_content_parts(self):
        """UserContent can also be plain str items in a sequence."""
        part = UserPromptPart(content=['hello', 'world'])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        # 10 chars / 4 = 2 tokens.
        assert estimate_token_count(msgs) == 2

    def test_format_with_text_content(self):
        from pydantic_ai.messages import TextContent

        part = UserPromptPart(content=[TextContent(content='multi-part')])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        text = _format_messages(msgs)
        assert 'User: multi-part' in text

    def test_format_with_str_content(self):
        part = UserPromptPart(content=['one', 'two'])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        text = _format_messages(msgs)
        assert 'User: one two' in text

    def test_format_empty_sequence(self):
        part = UserPromptPart(content=[])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        text = _format_messages(msgs)
        assert 'User: ' in text


class TestWarnNearLimitsEdgeCases:
    """Cover WarnNearLimits edge cases for marker detection and stripping."""

    @pytest.mark.anyio
    async def test_strip_warning_with_only_marker_message(self):
        """A message composed entirely of a marker part should be removed."""
        lw = WarnNearLimits(max_iterations=100)
        marker_msg = ModelRequest(parts=[UserPromptPart(content='[WarnNearLimits]\nold')])
        messages: list[ModelMessage] = [_user('real'), marker_msg]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        # Marker message should be stripped; only the real message remains.
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_strip_warning_system_prompt_marker(self):
        """Marker in a SystemPromptPart should also be detected."""
        lw = WarnNearLimits(max_iterations=100)
        marker_msg = ModelRequest(parts=[SystemPromptPart(content='[WarnNearLimits]\nold')])
        messages: list[ModelMessage] = [_user('real'), marker_msg]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_strip_mixed_parts_keeps_non_marker(self):
        """A message with both marker and non-marker parts should keep the non-marker parts."""
        lw = WarnNearLimits(max_iterations=100)
        mixed = ModelRequest(
            parts=[
                UserPromptPart(content='keep this'),
                UserPromptPart(content='[WarnNearLimits]\nremove this'),
            ]
        )
        messages: list[ModelMessage] = [mixed]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 1
        first = result.messages[0]
        assert isinstance(first, ModelRequest)
        assert len(first.parts) == 1

    @pytest.mark.anyio
    async def test_keeps_already_empty_request(self):
        """An already-empty ModelRequest must survive stripping so history still ends with a
        ModelRequest. pydantic-ai's empty-response retry appends an empty ModelRequest;
        dropping it left history ending on a ModelResponse and crashed the next request with
        `Processed history must end with a ModelRequest`."""
        lw = WarnNearLimits(max_iterations=100)
        messages: list[ModelMessage] = [_user('real'), ModelResponse(parts=[]), ModelRequest(parts=[])]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 3
        assert isinstance(result.messages[-1], ModelRequest)

    @pytest.mark.anyio
    async def test_context_warning_below_threshold(self):
        """Context window should not warn when below threshold."""
        lw = WarnNearLimits(max_context_tokens=1000)
        messages: list[ModelMessage] = [_user('hi')]  # ~0.5 tokens, well below 70%.
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_total_tokens_warning_critical(self):
        """Total tokens at or above limit should produce CRITICAL."""
        lw = WarnNearLimits(max_total_tokens=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(input_tokens=60, output_tokens=50)  # 110 total, above limit.
        result = await lw.before_model_request(ctx, rc)
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        assert 'CRITICAL' in text.content

    @pytest.mark.anyio
    async def test_context_window_critical(self):
        """Context window at or above limit should produce CRITICAL."""
        lw = WarnNearLimits(max_context_tokens=5)
        messages: list[ModelMessage] = [_user('x' * 40)]  # ~10 tokens, well above 5.
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await lw.before_model_request(ctx, rc)
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        assert 'CRITICAL' in text.content

    def test_warn_on_subset(self):
        """Can configure warn_on to only include specific limits."""
        lw = WarnNearLimits(max_iterations=10, max_total_tokens=100, warn_on=['iterations'])
        assert lw._active_kinds == ('iterations',)


class TestCompactionEdgeCases:
    """Cover Compaction edge cases."""

    @pytest.mark.anyio
    async def test_compaction_cutoff_zero_no_change(self):
        """When cutoff is 0, no compaction should occur (messages all kept)."""
        comp = SummarizingCompaction(model='test:m', max_messages=2, keep_messages=10)
        # Only 3 messages, keep_messages=10 means cutoff=0.
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await comp.before_model_request(ctx, rc)
        assert len(result.messages) == 3


class TestSlidingWindowCompactionEdgeCases:
    """Cover SlidingWindowCompaction edge cases."""

    @pytest.mark.anyio
    async def test_cutoff_zero_no_trim(self):
        """When the cutoff resolves to 0, messages should not be trimmed."""
        sw = SlidingWindowCompaction(max_messages=2, keep_messages=10)
        # 3 messages, but keep_messages=10 => cutoff=0.
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 3

    @pytest.mark.anyio
    async def test_token_not_triggered_when_below(self):
        """Token trigger should not fire below threshold."""
        sw = SlidingWindowCompaction(max_tokens=999999, keep_messages=2)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 1


class TestWarnNearLimitsMarkerDetection:
    """Cover _is_marker_part return False for non-text parts."""

    @pytest.mark.anyio
    async def test_non_string_user_prompt_not_detected_as_marker(self):
        """UserPromptPart with non-string content should not match marker."""
        lw = WarnNearLimits(max_iterations=100)
        # Create a ModelRequest with a ToolReturnPart (not a marker).
        messages: list[ModelMessage] = [
            _user('real'),
            _tool_return('fn', 'tc1', 'some result'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 2

    @pytest.mark.anyio
    async def test_strip_preserves_model_responses(self):
        """ModelResponse messages pass through strip unchanged."""
        lw = WarnNearLimits(max_iterations=100)
        messages: list[ModelMessage] = [
            _user('hi'),
            _assistant('response'),
            ModelRequest(parts=[UserPromptPart(content='[WarnNearLimits]\nold')]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        # Marker message removed; user and assistant remain.
        assert len(result.messages) == 2
        assert isinstance(result.messages[1], ModelResponse)


class TestWarnNearLimitsTotalTokensBelowThreshold:
    """Cover _build_total_tokens_warning returning None when below threshold."""

    @pytest.mark.anyio
    async def test_total_tokens_below_threshold(self):
        lw = WarnNearLimits(max_total_tokens=1000)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(input_tokens=10, output_tokens=10)  # 20 total, 2% of 1000.
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 1  # No warning.


# ---------------------------------------------------------------------------
# Tokenizer parameter
# ---------------------------------------------------------------------------


class TestTokenizerParameter:
    """Tests for the optional tokenizer parameter on estimate_token_count,
    SlidingWindowCompaction, and Compaction."""

    def test_estimate_token_count_with_tokenizer(self):
        """Custom tokenizer should override the heuristic."""
        msgs: list[ModelMessage] = [_user('hello world')]
        # Heuristic: 11 chars / 4 = 2 tokens.
        assert estimate_token_count(msgs) == 2
        # Custom tokenizer: count words instead.
        assert estimate_token_count(msgs, tokenizer=lambda s: len(s.split())) == 2

    def test_estimate_token_count_tokenizer_called_per_segment(self):
        """Tokenizer is called once per text segment, results are summed."""
        calls: list[str] = []

        def tracking_tokenizer(s: str) -> int:
            calls.append(s)
            return 10

        msgs: list[ModelMessage] = [_user('a'), _assistant('b')]
        result = estimate_token_count(msgs, tokenizer=tracking_tokenizer)
        assert result == 20
        assert len(calls) == 2

    @pytest.mark.anyio
    async def test_sliding_window_with_tokenizer(self):
        """SlidingWindowCompaction should use the tokenizer for token-based triggers."""
        # Custom tokenizer: 1 token per character.
        sw = SlidingWindowCompaction(
            max_tokens=10,
            keep_tokens=5,
            tokenizer=lambda s: len(s),
            preserve_first_user_message=False,
        )
        # Each message has 4 chars = 4 tokens with this tokenizer. 5 messages = 20 tokens.
        messages: list[ModelMessage] = [_user('abcd') for _ in range(5)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        # With keep_tokens=5 and 4 tokens per message, should keep 1 message.
        remaining_tokens = estimate_token_count(result.messages, tokenizer=lambda s: len(s))
        assert remaining_tokens <= 5

    @pytest.mark.anyio
    async def test_sliding_window_tokenizer_threshold_check(self):
        """SlidingWindowCompaction tokenizer should be used for the trigger check."""
        # Tokenizer that inflates counts: 100 tokens per char.
        sw = SlidingWindowCompaction(
            max_tokens=50,
            keep_messages=1,
            tokenizer=lambda s: len(s) * 100,
            preserve_first_user_message=False,
        )
        # 2 chars * 100 = 200 tokens per message. Only 1 message but still > 50.
        messages: list[ModelMessage] = [_user('ab'), _user('cd')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_compaction_with_tokenizer(self):
        """Compaction should use the tokenizer for token-based triggers."""
        # Tokenizer: 1 token per char.
        comp = SummarizingCompaction(
            model='test:m',
            max_tokens=10,
            keep_messages=1,
            tokenizer=lambda s: len(s),
            preserve_first_user_message=False,
            incremental=False,
        )
        # Each message: 'abcde' = 5 chars = 5 tokens. 4 messages = 20 tokens > 10.
        messages: list[ModelMessage] = [_user('abcde') for _ in range(4)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Token summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        # Should have triggered compaction.
        assert len(result.messages) >= 1
        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        sys_parts = [p for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert any('Token summary.' in p.content for p in sys_parts)

    def testfind_token_cutoff_with_tokenizer(self):
        """find_token_cutoff should use the tokenizer."""
        messages: list[ModelMessage] = [_user('abcde') for _ in range(10)]
        # Tokenizer: 1 token per char. Each message = 5 tokens.
        cutoff = find_token_cutoff(messages, 15, tokenizer=lambda s: len(s))
        remaining = messages[cutoff:]
        assert estimate_token_count(remaining, tokenizer=lambda s: len(s)) <= 15


# ---------------------------------------------------------------------------
# Preserve first user message
# ---------------------------------------------------------------------------


class TestPreserveFirstUserMessage:
    """Tests for the preserve_first_user_message parameter."""

    def testfind_first_user_message_found(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='sys')]),
            _user('first'),
            _user('second'),
        ]
        result = find_first_user_message(msgs)
        assert result is not None
        assert isinstance(result.parts[0], UserPromptPart)
        assert result.parts[0].content == 'first'

    def testfind_first_user_message_none(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='sys')]),
            _assistant('hello'),
        ]
        assert find_first_user_message(msgs) is None

    @pytest.mark.anyio
    async def test_sliding_window_preserves_first_user(self):
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=2, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _user('original task'),
            _assistant('got it'),
            _user('follow-up 1'),
            _assistant('done'),
            _user('follow-up 2'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        # The first user message ('original task') should be preserved even though
        # it was outside the keep window.
        assert 'original task' in _user_texts(result.messages)

    @pytest.mark.anyio
    async def test_sliding_window_no_duplicate_when_in_window(self):
        """First user message should not be duplicated if already in the kept window."""
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=5, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _user('task'),
            _assistant('ok'),
            _user('more'),
            _assistant('done'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 4  # Not triggered since 4 < 5 keep.

    @pytest.mark.anyio
    async def test_sliding_window_disabled_preserve(self):
        """When preserve_first_user_message=False, first user message is not kept."""
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [
            _user('original'),
            _assistant('a'),
            _user('b'),
            _assistant('c'),
            _user('last'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 1
        assert 'original' not in _user_texts(result.messages)

    @pytest.mark.anyio
    async def test_compaction_preserves_first_user(self):
        comp = SummarizingCompaction(model='test:m', max_messages=3, keep_messages=1, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _user('build a web app'),
            _assistant('response 1'),
            _user('second'),
            _assistant('response 2'),
            _user('third'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        # Summary message + first user message + 1 kept = 3.
        assert len(result.messages) == 3
        # First message is the summary (with system prompts).
        assert isinstance(result.messages[0], ModelRequest)
        sys_parts = [p for p in result.messages[0].parts if isinstance(p, SystemPromptPart)]
        assert any('Summary.' in p.content for p in sys_parts)
        # Second message is the preserved first user message.
        assert isinstance(result.messages[1], ModelRequest)
        user_parts = [p for p in result.messages[1].parts if isinstance(p, UserPromptPart)]
        assert len(user_parts) == 1
        assert user_parts[0].content == 'build a web app'

    @pytest.mark.anyio
    async def test_compaction_no_duplicate_first_user_when_in_window(self):
        """First user message already in kept window should not be duplicated."""
        comp = SummarizingCompaction(model='test:m', max_messages=3, keep_messages=5, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _user('task'),
            _assistant('ok'),
            _user('more'),
            _assistant('done'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await comp.before_model_request(ctx, rc)
        # Not triggered since keep_messages > len(messages).
        assert len(result.messages) == 4

    @pytest.mark.anyio
    async def test_sliding_window_no_user_messages(self):
        """When there are no user messages, preservation is a no-op."""
        sw = SlidingWindowCompaction(max_messages=2, keep_messages=1, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _assistant('a'),
            _assistant('b'),
            _assistant('c'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 1


# ---------------------------------------------------------------------------
# Incremental summarization
# ---------------------------------------------------------------------------


class TestIncrementalSummarization:
    """Tests for the incremental parameter on Compaction."""

    def test_extract_previous_summary_found(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}Old summary text.')]),
            _user('hi'),
        ]
        assert _extract_previous_summary(msgs) == 'Old summary text.'

    def test_extract_previous_summary_not_found(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='Regular system prompt.')]),
            _user('hi'),
        ]
        assert _extract_previous_summary(msgs) is None

    def test_extract_previous_summary_empty_messages(self):
        assert _extract_previous_summary([]) is None

    def test_extract_previous_summary_skips_non_requests(self):
        msgs: list[ModelMessage] = [
            _assistant('hi'),
            _user('hello'),
        ]
        assert _extract_previous_summary(msgs) is None

    @pytest.mark.anyio
    async def test_incremental_includes_previous_summary(self):
        """When incremental=True and a prior summary exists, it should be included in the prompt."""
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=4,
            keep_messages=3,
            incremental=True,
            preserve_first_user_message=False,
        )
        # Simulate a conversation that already has a summary from prior compaction.
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}Previous context here.')]),
            _user('new input 1'),
            _assistant('response 1'),
            _user('new input 2'),
            _assistant('response 2'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Extended summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await comp.before_model_request(ctx, rc)

        # Verify the summarization prompt included the previous summary.
        call_args = mock_agent_instance.run.call_args
        prompt_text = call_args[0][0]
        assert '<previous-summary>' in prompt_text
        assert 'Previous context here.' in prompt_text

    @pytest.mark.anyio
    async def test_incremental_no_previous_summary(self):
        """When incremental=True but no prior summary exists, prompt should be plain."""
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            incremental=True,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [
            _user('first'),
            _assistant('response 1'),
            _user('second'),
            _assistant('response 2'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Fresh summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await comp.before_model_request(ctx, rc)

        call_args = mock_agent_instance.run.call_args
        prompt_text = call_args[0][0]
        assert '<previous-summary>' not in prompt_text

    @pytest.mark.anyio
    async def test_incremental_disabled(self):
        """When incremental=False, the previous summary should not be included."""
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            incremental=False,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}Old summary.')]),
            _user('new input'),
            _assistant('response'),
            _user('another'),
            _assistant('another response'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Regenerated summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await comp.before_model_request(ctx, rc)

        call_args = mock_agent_instance.run.call_args
        prompt_text = call_args[0][0]
        assert '<previous-summary>' not in prompt_text

    @pytest.mark.anyio
    async def test_incremental_output_contains_summary(self):
        """The output after incremental compaction should contain the new summary."""
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            incremental=True,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}Old context.')]),
            _user('a'),
            _assistant('b'),
            _user('c'),
            _assistant('d'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Extended context summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        sys_parts = [p for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert any('Extended context summary.' in p.content for p in sys_parts)


# ---------------------------------------------------------------------------
# Helpers for the new strategies
# ---------------------------------------------------------------------------


def _pair(name: str, cid: str, content: str = 'result content here') -> list[ModelMessage]:
    return [_tool_call(name, cid), _tool_return(name, cid, content)]


def _return_contents(messages: list[ModelMessage]) -> list[str]:
    out: list[str] = []
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, ToolReturnPart):
                    out.append(str(p.content))
    return out


def _call_args(messages: list[ModelMessage]) -> list[object]:
    out: list[object] = []
    for m in messages:
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, ToolCallPart):
                    out.append(p.args)
    return out


def _user_texts(messages: list[ModelMessage]) -> list[str]:
    out: list[str] = []
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    out.append(p.content)
    return out


def _orphan_free(messages: list[ModelMessage]) -> bool:
    """True if every kept tool return has its matching tool call among *messages*."""
    call_ids: set[str] = set()
    return_ids: set[str] = set()
    for m in messages:
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, ToolCallPart) and p.tool_call_id:
                    call_ids.add(p.tool_call_id)
        else:
            for p in m.parts:
                if isinstance(p, ToolReturnPart):
                    return_ids.add(p.tool_call_id)
    return return_ids <= call_ids


class TestHelperCoverage:
    """Exercise every branch of the shared test-collection helpers with one diverse input."""

    def test_collection_helpers(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='s'), UserPromptPart(content='u')]),
            ModelResponse(parts=[TextPart(content='t'), ToolCallPart(tool_name='fn', args='{}', tool_call_id='c1')]),
            _tool_return('fn', 'c1', 'r'),
        ]
        assert _user_texts(msgs) == ['u']
        assert _return_contents(msgs) == ['r']
        assert _call_args(msgs) == ['{}']
        assert _orphan_free(msgs)

    def test_file_key_edges(self):
        assert _file_key(ToolCallPart(tool_name='other', args={}, tool_call_id='c')) is None
        assert _file_key(ToolCallPart(tool_name='read_file', args='not-a-dict', tool_call_id='c')) is None
        assert _file_key(ToolCallPart(tool_name='read_file', args={'path': 123}, tool_call_id='c')) is None
        assert _file_key(ToolCallPart(tool_name='read_file', args={'path': 'p.py'}, tool_call_id='c')) == 'p.py'


# ---------------------------------------------------------------------------
# iter_tool_pairs
# ---------------------------------------------------------------------------


class TestIterToolPairs:
    def test_skips_empty_ids_and_orphan_returns(self):
        msgs: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name='fn', args='{}', tool_call_id='')]),
            _tool_return('fn', ''),  # empty id, no matching call
            _tool_return('fn', 'orphan'),  # return with no matching call
            _tool_call('g', 'g1'),
            _tool_return('g', 'g1'),
        ]
        pairs = iter_tool_pairs(msgs)
        assert [p.tool_call_id for p in pairs] == ['g1']
        assert pairs[0].tool_name == 'g'
        assert pairs[0].order == 0


# ---------------------------------------------------------------------------
# ClearToolResults
# ---------------------------------------------------------------------------


class TestClearToolResults:
    def test_validation_no_trigger(self):
        with pytest.raises(ValueError, match='At least one of max_messages, max_tokens, or max_fraction must be set'):
            ClearToolResults()

    def test_validation_negative_max_messages(self):
        with pytest.raises(ValueError, match='max_messages must be positive'):
            ClearToolResults(max_messages=0)

    def test_validation_negative_max_tokens(self):
        with pytest.raises(ValueError, match='max_tokens must be positive'):
            ClearToolResults(max_tokens=-1)

    def test_validation_negative_keep_pairs(self):
        with pytest.raises(ValueError, match='keep_pairs must be non-negative'):
            ClearToolResults(max_messages=1, keep_pairs=-1)

    def test_validation_negative_min_clear_tokens(self):
        with pytest.raises(ValueError, match='min_clear_tokens must be non-negative'):
            ClearToolResults(max_messages=1, min_clear_tokens=-1)

    @pytest.mark.anyio
    async def test_no_clear_below_threshold(self):
        cap = ClearToolResults(max_messages=100, keep_pairs=0)
        messages: list[ModelMessage] = [*_pair('fn', 'tc1'), *_pair('fn', 'tc2')]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert result.messages == messages

    @pytest.mark.anyio
    async def test_clears_old_keeps_recent_pairs(self):
        cap = ClearToolResults(max_messages=1, keep_pairs=1)
        messages: list[ModelMessage] = [
            *_pair('fn', 'tc1'),
            *_pair('fn', 'tc2'),
            *_pair('fn', 'tc3'),
        ]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        contents = _return_contents(result.messages)
        assert contents == ['[tool result cleared]', '[tool result cleared]', 'result content here']

    @pytest.mark.anyio
    async def test_token_trigger(self):
        cap = ClearToolResults(max_tokens=5, keep_pairs=0)
        messages: list[ModelMessage] = [*_pair('fn', 'tc1', 'x' * 80)]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert _return_contents(result.messages) == ['[tool result cleared]']

    @pytest.mark.anyio
    async def test_exclude_tools(self):
        cap = ClearToolResults(max_messages=1, keep_pairs=0, exclude_tools=frozenset({'keep'}))
        messages: list[ModelMessage] = [*_pair('drop', 'tc1'), *_pair('keep', 'tc2')]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert _return_contents(result.messages) == ['[tool result cleared]', 'result content here']

    @pytest.mark.anyio
    async def test_clear_tool_inputs(self):
        cap = ClearToolResults(max_messages=1, keep_pairs=0, clear_tool_inputs=True)
        call = ModelResponse(parts=[ToolCallPart(tool_name='fn', args='{"q": "x"}', tool_call_id='tc1')])
        messages: list[ModelMessage] = [call, _tool_return('fn', 'tc1')]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        # Cleared args stay JSON-valid so they don't reach a provider as malformed function-args.
        assert _call_args(result.messages) == ['{}']

    @pytest.mark.anyio
    async def test_min_clear_tokens_skips_small_gain(self):
        cap = ClearToolResults(max_messages=1, keep_pairs=0, min_clear_tokens=10_000)
        messages: list[ModelMessage] = [*_pair('fn', 'tc1', 'tiny')]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        # Reclaim is far below min_clear_tokens, so nothing is cleared.
        assert _return_contents(result.messages) == ['tiny']

    @pytest.mark.anyio
    async def test_min_clear_tokens_proceeds_on_large_gain(self):
        cap = ClearToolResults(max_messages=1, keep_pairs=0, min_clear_tokens=1)
        messages: list[ModelMessage] = [*_pair('fn', 'tc1', 'x' * 400)]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert _return_contents(result.messages) == ['[tool result cleared]']

    @pytest.mark.anyio
    async def test_no_tool_pairs_is_noop(self):
        cap = ClearToolResults(max_messages=1, keep_pairs=0)
        messages: list[ModelMessage] = [_user('a'), _assistant('b')]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert result.messages == messages

    @pytest.mark.anyio
    async def test_idempotent(self):
        cap = ClearToolResults(max_messages=1, keep_pairs=0, clear_tool_inputs=True)
        call = ModelResponse(parts=[ToolCallPart(tool_name='fn', args='{"q": "x"}', tool_call_id='tc1')])
        messages: list[ModelMessage] = [call, _tool_return('fn', 'tc1')]
        ctx = _make_ctx()
        once = await cap.compact(messages, ctx)
        twice = await cap.compact(once, ctx)
        assert _return_contents(twice) == ['[tool result cleared]']
        assert _call_args(twice) == ['{}']

    @pytest.mark.anyio
    async def test_preserves_typed_tool_search_return(self):
        # `ToolSearchReturnPart` subclasses `ToolReturnPart` but carries structured content that
        # core's `parse_discovered_tools` re-reads on the next request. Blanking it to a string
        # crashed that reader; clearing must skip typed subclasses and touch only plain results.
        cap = ClearToolResults(max_messages=1, keep_pairs=0)
        search_content: ToolSearchReturnContent = {'discovered_tools': [{'name': 'alpha'}, {'name': 'beta'}]}
        messages: list[ModelMessage] = [
            _tool_call('fn', 'u1'),
            _tool_return('fn', 'u1', 'plain result'),
            ModelResponse(parts=[ToolCallPart(tool_name='search_tools', args='{}', tool_call_id='ts1')]),
            ModelRequest(parts=[ToolSearchReturnPart(content=search_content, tool_call_id='ts1')]),
        ]
        result = await cap.before_model_request(_make_ctx(), _make_request_context(messages))

        returns = {
            p.tool_call_id: p
            for m in result.messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        }
        # Plain result blanked, typed search return kept intact (concrete type + structured content).
        assert type(returns['u1']) is ToolReturnPart
        assert returns['u1'].content == cap.placeholder
        assert type(returns['ts1']) is ToolSearchReturnPart
        assert returns['ts1'].content == search_content
        # The real next-request regression: core still recovers the discovered names.
        assert parse_discovered_tools(result.messages) == {'alpha', 'beta'}


# ---------------------------------------------------------------------------
# DeduplicateFileReads
# ---------------------------------------------------------------------------


def _read_call(cid: str, path: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name='read_file', args={'path': path}, tool_call_id=cid)])


def _read_return(cid: str, content: str) -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name='read_file', content=content, tool_call_id=cid)])


def _file_key(call: ToolCallPart) -> str | None:
    if call.tool_name != 'read_file':
        return None
    args = call.args
    if isinstance(args, dict):
        path = args.get('path')
        return path if isinstance(path, str) else None
    return None


class TestDeduplicateFileReads:
    def test_validation_negative_max_messages(self):
        with pytest.raises(ValueError, match='max_messages must be positive'):
            DeduplicateFileReads(file_key=_file_key, max_messages=0)

    def test_validation_negative_max_tokens(self):
        with pytest.raises(ValueError, match='max_tokens must be positive'):
            DeduplicateFileReads(file_key=_file_key, max_tokens=-1)

    @pytest.mark.anyio
    async def test_keeps_latest_read(self):
        cap = DeduplicateFileReads(file_key=_file_key)
        messages: list[ModelMessage] = [
            _read_call('tc1', 'a.py'),
            _read_return('tc1', 'first a'),
            _read_call('tc2', 'b.py'),
            _read_return('tc2', 'b body'),
            _read_call('tc3', 'a.py'),
            _read_return('tc3', 'second a'),
        ]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert _return_contents(result.messages) == ['[superseded file read]', 'b body', 'second a']

    @pytest.mark.anyio
    async def test_non_file_read_ignored(self):
        cap = DeduplicateFileReads(file_key=_file_key)
        messages: list[ModelMessage] = [
            *_pair('search', 'tc1'),
            *_pair('search', 'tc2'),
        ]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        # search is not a file read -> file_key returns None -> nothing cleared.
        assert _return_contents(result.messages) == ['result content here', 'result content here']

    @pytest.mark.anyio
    async def test_no_duplicates_is_noop(self):
        cap = DeduplicateFileReads(file_key=_file_key)
        messages: list[ModelMessage] = [
            _read_call('tc1', 'a.py'),
            _read_return('tc1', 'a body'),
            _read_call('tc2', 'b.py'),
            _read_return('tc2', 'b body'),
        ]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert result.messages == messages

    @pytest.mark.anyio
    async def test_runs_always_without_trigger(self):
        cap = DeduplicateFileReads(file_key=_file_key)
        messages: list[ModelMessage] = [
            _read_call('tc1', 'a.py'),
            _read_return('tc1', 'first'),
            _read_call('tc2', 'a.py'),
            _read_return('tc2', 'second'),
        ]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert _return_contents(result.messages) == ['[superseded file read]', 'second']

    @pytest.mark.anyio
    async def test_trigger_gate_not_exceeded(self):
        cap = DeduplicateFileReads(file_key=_file_key, max_messages=100)
        messages: list[ModelMessage] = [
            _read_call('tc1', 'a.py'),
            _read_return('tc1', 'first'),
            _read_call('tc2', 'a.py'),
            _read_return('tc2', 'second'),
        ]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        # Below the trigger threshold, so no dedup despite the duplicate.
        assert result.messages == messages

    @pytest.mark.anyio
    async def test_trigger_gate_exceeded(self):
        cap = DeduplicateFileReads(file_key=_file_key, max_messages=1)
        messages: list[ModelMessage] = [
            _read_call('tc1', 'a.py'),
            _read_return('tc1', 'first'),
            _read_call('tc2', 'a.py'),
            _read_return('tc2', 'second'),
        ]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert _return_contents(result.messages) == ['[superseded file read]', 'second']


# ---------------------------------------------------------------------------
# TieredCompaction
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _RecordingTier:
    label: str
    calls: list[str]
    drop: int = 0

    async def compact(self, messages: list[ModelMessage], ctx: Any) -> list[ModelMessage]:
        self.calls.append(self.label)
        return messages[self.drop :] if self.drop else messages


class TestTieredCompaction:
    def test_validation_empty_tiers(self):
        with pytest.raises(ValueError, match='tiers must not be empty'):
            TieredCompaction(tiers=[], target_tokens=10)

    def test_validation_target_tokens(self):
        with pytest.raises(ValueError, match='target_tokens must be positive'):
            TieredCompaction(tiers=[ClearToolResults(max_messages=1)], target_tokens=0)

    @pytest.mark.anyio
    async def test_noop_under_target(self):
        calls: list[str] = []
        tier = _RecordingTier('t1', calls)
        cap = TieredCompaction(tiers=[tier], target_tokens=1_000_000)
        messages: list[ModelMessage] = [_user('x' * 40)]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert result.messages == messages
        assert calls == []

    @pytest.mark.anyio
    async def test_short_circuit_first_tier_suffices(self):
        calls: list[str] = []
        # Each message ~10 tokens; 5 messages = 50 tokens. Target 15.
        t1 = _RecordingTier('t1', calls, drop=4)  # leaves 1 message (~10 tokens) <= 15
        t2 = _RecordingTier('t2', calls, drop=0)
        cap = TieredCompaction(tiers=[t1, t2], target_tokens=15)
        messages: list[ModelMessage] = [_user('x' * 40) for _ in range(5)]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert calls == ['t1']  # t2 never reached
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_triggers_on_anchored_usage_the_heuristic_cannot_see(self):
        # ~101 tokens by the heuristic, but the provider reported the request at 90K:
        # dense content the character estimate is blind to. The anchored gate must fire.
        calls: list[str] = []
        t1 = _RecordingTier('t1', calls, drop=1)
        cap = TieredCompaction(tiers=[t1], target_tokens=50_000)
        messages: list[ModelMessage] = [_user('x' * 400), _assistant_with_usage('short', 90_000, 100)]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert calls == ['t1']
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_escalation_subtracts_tier_reclaim_from_anchored_baseline(self):
        # The anchor predates any tier rewrite, so re-anchoring would show no reclaim and
        # always escalate to the last tier. Subtracting the removed text's estimate must
        # register t1's reclaim (14K -> 12K <= 13K target) and never reach t2.
        calls: list[str] = []
        t1 = _RecordingTier('t1', calls, drop=6)
        t2 = _RecordingTier('t2', calls)
        cap = TieredCompaction(tiers=[t1, t2], target_tokens=13_000, tokenizer=len)
        messages: list[ModelMessage] = [_assistant_with_usage('', 10_000, 0)] + [
            _tool_return('search', f'tc{i}', 'z' * 400) for i in range(10)
        ]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert calls == ['t1']
        assert len(result.messages) == 5

    @pytest.mark.anyio
    async def test_escalation_never_counts_fixed_overhead_as_reclaimed(self):
        # 100K of the baseline is anchored overhead (tool definitions, instructions) that no
        # tier can touch. Halving the 4K of message text must leave the estimate at ~102K,
        # not the ~52K a proportional rescale would claim, so escalation continues to t2.
        calls: list[str] = []
        t1 = _RecordingTier('t1', calls, drop=6)
        t2 = _RecordingTier('t2', calls)
        cap = TieredCompaction(tiers=[t1, t2], target_tokens=60_000, tokenizer=len)
        messages: list[ModelMessage] = [_assistant_with_usage('', 100_000, 0)] + [
            _tool_return('search', f'tc{i}', 'z' * 400) for i in range(10)
        ]
        rc = _make_request_context(messages)
        await cap.before_model_request(_make_ctx(), rc)
        assert calls == ['t1', 't2']

    @pytest.mark.anyio
    async def test_full_escalation(self):
        calls: list[str] = []
        t1 = _RecordingTier('t1', calls, drop=1)  # 5 -> 4 messages (~40 tokens) still > 15
        t2 = _RecordingTier('t2', calls, drop=3)  # 4 -> 1 message
        cap = TieredCompaction(tiers=[t1, t2], target_tokens=15)
        messages: list[ModelMessage] = [_user('x' * 40) for _ in range(5)]
        rc = _make_request_context(messages)
        result = await cap.before_model_request(_make_ctx(), rc)
        assert calls == ['t1', 't2']
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_reinjects_pins_before_deciding_to_stop(self):
        calls: list[str] = []
        t1 = _RecordingTier('t1', calls, drop=1)
        t2 = _RecordingTier('t2', calls, drop=1)
        cap = TieredCompaction(tiers=[t1, t2], target_tokens=10)
        messages: list[ModelMessage] = [_pinned_msg('x' * 40), _user('tail')]
        result = await cap.before_model_request(_make_ctx(), _make_request_context(messages))
        assert calls == ['t1', 't2']
        assert _pinned_texts(result.messages) == ['x' * 40]

    @pytest.mark.anyio
    async def test_composes_real_strategies(self):
        # ClearToolResults then SummarizingCompaction, driven by the orchestrator.
        clear = ClearToolResults(max_messages=1, keep_pairs=0)
        summarizer = SummarizingCompaction(
            model='test:m', max_messages=1, keep_messages=1, preserve_first_user_message=False
        )
        cap = TieredCompaction(tiers=[clear, summarizer], target_tokens=1)
        messages: list[ModelMessage] = [*_pair('fn', 'tc1', 'x' * 200), _user('latest')]
        rc = _make_request_context(messages)

        mock_result = AsyncMock()
        mock_result.output = 'Tiered summary.'
        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance
            result = await cap.before_model_request(_make_ctx(), rc)

        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        sys_parts = [p for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert any('Tiered summary.' in p.content for p in sys_parts)


# ---------------------------------------------------------------------------
# SummarizingCompaction — model inheritance + structured prompt
# ---------------------------------------------------------------------------


class TestSummarizingCompactionModel:
    @pytest.mark.anyio
    async def test_model_inherits_from_the_request_when_none(self):
        comp = SummarizingCompaction(
            max_messages=3, keep_messages=1, preserve_first_user_message=False, incremental=False
        )
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Inherited-model summary.'
        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance
            await comp.before_model_request(ctx, rc)

        # The summarizer agent was constructed with the model the request is going to. Core
        # starts that as the run's model, so the two differ only where a capability replaced
        # it -- and then the request's is the one the compaction is being done for.
        assert MockAgent.call_args.args[0] is rc.model
        # Its usage is threaded into the parent run for honest accounting.
        assert mock_agent_instance.run.call_args.kwargs['usage'] is ctx.usage

    @pytest.mark.anyio
    async def test_nested_summary_reserves_parent_usage_limits(self):
        comp = SummarizingCompaction(max_messages=3, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        ctx = _make_ctx(usage_limits=UsageLimits(request_limit=5, tool_calls_limit=2))

        mock_result = AsyncMock()
        mock_result.output = 'Bounded summary.'
        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance
            await comp.before_model_request(ctx, _make_request_context(messages))

        assert mock_agent_instance.run.call_args.kwargs['usage_limits'] == UsageLimits(
            request_limit=4, tool_calls_limit=2
        )

    @pytest.mark.anyio
    async def test_summarizer_agent_gets_the_default_instructions(self):
        comp = SummarizingCompaction(max_messages=3, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]

        mock_result = AsyncMock()
        mock_result.output = 'Default-instructions summary.'
        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance
            await comp.before_model_request(_make_ctx(), _make_request_context(messages))

        assert MockAgent.call_args.kwargs['instructions'] == comp.instructions
        assert 'context summarization assistant' in MockAgent.call_args.kwargs['instructions']

    @pytest.mark.anyio
    async def test_instructions_override_reaches_the_summarizer_agent(self):
        required = 'Required endpoint instruction.'
        comp = SummarizingCompaction(
            max_messages=3,
            keep_messages=1,
            preserve_first_user_message=False,
            instructions=required,
        )
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]

        mock_result = AsyncMock()
        mock_result.output = 'Overridden-instructions summary.'
        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance
            await comp.before_model_request(_make_ctx(), _make_request_context(messages))

        assert MockAgent.call_args.kwargs['instructions'] == required

    def test_default_prompt_has_structured_sections(self):
        from pydantic_ai_harness.compaction._summarizing_compaction import _DEFAULT_SUMMARY_PROMPT

        for heading in (
            '## Intent',
            '## Key decisions',
            '## Artifacts',
            '## Current state',
            '## Next steps',
            '## Open questions',
        ):
            assert heading in _DEFAULT_SUMMARY_PROMPT


class TestClampOversizedMessages:
    def test_validation_no_trigger(self):
        with pytest.raises(ValueError, match='max_part_tokens or max_part_chars'):
            ClampOversizedMessages()

    def test_validation_negative_max_part_tokens(self):
        with pytest.raises(ValueError, match='max_part_tokens must be positive'):
            ClampOversizedMessages(max_part_tokens=0)

    def test_validation_negative_max_part_chars(self):
        with pytest.raises(ValueError, match='max_part_chars must be positive'):
            ClampOversizedMessages(max_part_chars=0)

    def test_validation_negative_keep_head(self):
        with pytest.raises(ValueError, match='keep_head_chars must be non-negative'):
            ClampOversizedMessages(max_part_chars=10, keep_head_chars=-1)

    def test_validation_negative_keep_tail(self):
        with pytest.raises(ValueError, match='keep_tail_chars must be non-negative'):
            ClampOversizedMessages(max_part_chars=10, keep_tail_chars=-1)

    @pytest.mark.anyio
    async def test_clamps_oversized_response_text(self):
        text = 'H' * 50 + ' ' * 5_000 + 'T' * 50
        cap = ClampOversizedMessages(max_part_chars=1_000, keep_head_chars=50, keep_tail_chars=50)
        messages: list[ModelMessage] = [_assistant(text)]
        result = await cap.compact(messages, _make_ctx())

        clamped = result[0]
        assert isinstance(clamped, ModelResponse)
        part = clamped.parts[0]
        assert isinstance(part, TextPart)
        assert part.content.startswith('H' * 50)
        assert part.content.endswith('T' * 50)
        assert '[clamped: removed' in part.content
        assert len(part.content) < len(text)

    @pytest.mark.anyio
    async def test_token_trigger_uses_heuristic(self):
        text = 'x' * 4_000  # ~1000 tokens at the 4-chars heuristic.
        cap = ClampOversizedMessages(max_part_tokens=100, keep_head_chars=20, keep_tail_chars=20)
        result = await cap.compact([_assistant(text)], _make_ctx())
        part = result[0].parts[0]
        assert isinstance(part, TextPart)
        assert '[clamped: removed' in part.content

    @pytest.mark.anyio
    async def test_token_trigger_uses_tokenizer(self):
        text = 'word ' * 1_000
        cap = ClampOversizedMessages(
            max_part_tokens=100,
            keep_head_chars=20,
            keep_tail_chars=20,
            tokenizer=lambda s: len(s.split()),
        )
        result = await cap.compact([_assistant(text)], _make_ctx())
        part = result[0].parts[0]
        assert isinstance(part, TextPart)
        assert '[clamped: removed' in part.content

    @pytest.mark.anyio
    async def test_small_text_untouched(self):
        cap = ClampOversizedMessages(max_part_chars=100_000, max_part_tokens=100_000)
        messages: list[ModelMessage] = [_assistant('short')]
        result = await cap.compact(messages, _make_ctx())
        # Nothing oversized -> the message object is returned unchanged.
        assert result[0] is messages[0]

    @pytest.mark.anyio
    async def test_keep_tail_zero(self):
        text = 'A' * 5_000
        cap = ClampOversizedMessages(max_part_chars=1_000, keep_head_chars=100, keep_tail_chars=0)
        result = await cap.compact([_assistant(text)], _make_ctx())
        part = result[0].parts[0]
        assert isinstance(part, TextPart)
        assert part.content.startswith('A' * 100)
        assert part.content.endswith(']\n')

    @pytest.mark.anyio
    async def test_clamp_skipped_when_not_smaller(self):
        # Oversized by the token trigger, but keep slices exceed the text length, so
        # clamping would not shrink it -- leave it untouched.
        text = 'y' * 100
        cap = ClampOversizedMessages(max_part_tokens=1, keep_head_chars=2_000, keep_tail_chars=2_000)
        messages: list[ModelMessage] = [_assistant(text)]
        result = await cap.compact(messages, _make_ctx())
        assert result[0] is messages[0]

    @pytest.mark.anyio
    async def test_clamps_oversized_tool_call_args(self):
        big = 'p' * 5_000
        call = ModelResponse(parts=[ToolCallPart(tool_name='write_plan', args=big, tool_call_id='c1')])
        cap = ClampOversizedMessages(max_part_chars=1_000, keep_head_chars=50, keep_tail_chars=50)
        result = await cap.compact([call], _make_ctx())

        part = result[0].parts[0]
        assert isinstance(part, ToolCallPart)
        assert isinstance(part.args, dict)
        assert _CLAMP_ARGS_KEY in part.args
        assert '[clamped: removed' in part.args[_CLAMP_ARGS_KEY]
        assert part.tool_call_id == 'c1'

    @pytest.mark.anyio
    async def test_preserves_typed_tool_call_subclasses(self):
        # `ToolSearchCallPart` and `LoadCapabilityCallPart` subclass `ToolCallPart` but narrow
        # `args` to a typed shape that `ModelMessagesTypeAdapter` validates when persisted
        # history is restored (e.g. `StepPersistence` resume). Replacing that shape with the
        # `_clamped` object made the round-trip fail; clamping must skip typed subclasses and
        # touch only plain tool calls.
        search_call = ToolSearchCallPart(args={'queries': ['find tools for ' + 'x' * 5_000]}, tool_call_id='ts1')
        load_call = LoadCapabilityCallPart(args='{"id": "' + 'c' * 5_000 + '"}', tool_call_id='lc1')
        plain_call = ToolCallPart(tool_name='write_plan', args='p' * 5_000, tool_call_id='c1')
        messages: list[ModelMessage] = [ModelResponse(parts=[search_call, load_call, plain_call])]
        cap = ClampOversizedMessages(max_part_chars=1_000, keep_head_chars=50, keep_tail_chars=50)
        result = await cap.compact(messages, _make_ctx())

        search, load, plain = result[0].parts
        # Typed subclasses pass through as the same objects; the plain call is still clamped.
        assert search is search_call
        assert load is load_call
        assert isinstance(plain, ToolCallPart)
        assert plain.args_as_dict().keys() == {_CLAMP_ARGS_KEY}
        # The real regression: the compacted history survives the round-trip `StepPersistence`
        # restores through. Equality covers the concrete part classes too, so this also pins
        # that the typed parts come back typed with their structured args intact.
        assert ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(result)) == result

    @pytest.mark.anyio
    async def test_small_tool_call_args_untouched(self):
        call = ModelResponse(parts=[ToolCallPart(tool_name='t', args='{"a": 1}', tool_call_id='c1')])
        cap = ClampOversizedMessages(max_part_chars=1_000)
        messages: list[ModelMessage] = [call]
        result = await cap.compact(messages, _make_ctx())
        assert result[0] is messages[0]

    @pytest.mark.anyio
    async def test_tool_call_args_not_clamped_when_disabled(self):
        big = 'p' * 5_000
        call = ModelResponse(parts=[ToolCallPart(tool_name='write_plan', args=big, tool_call_id='c1')])
        cap = ClampOversizedMessages(max_part_chars=1_000, clamp_tool_call_args=False)
        messages: list[ModelMessage] = [call]
        result = await cap.compact(messages, _make_ctx())
        assert result[0] is messages[0]

    @pytest.mark.anyio
    async def test_request_messages_and_other_parts_untouched(self):
        from pydantic_ai.messages import ThinkingPart

        big_user = _user('u' * 5_000)
        mixed = ModelResponse(parts=[ThinkingPart(content='t' * 5_000), TextPart(content='z' * 5_000)])
        cap = ClampOversizedMessages(max_part_chars=1_000, keep_head_chars=50, keep_tail_chars=50)
        result = await cap.compact([big_user, mixed], _make_ctx())

        # Request-side message is left as-is (same object).
        assert result[0] is big_user
        # The thinking part is untouched; only the text part is clamped.
        out_mixed = result[1]
        assert isinstance(out_mixed, ModelResponse)
        thinking, text = out_mixed.parts
        assert isinstance(thinking, ThinkingPart)
        assert thinking.content == 't' * 5_000
        assert isinstance(text, TextPart)
        assert '[clamped: removed' in text.content

    @pytest.mark.anyio
    async def test_before_model_request(self):
        text = 'q' * 5_000
        cap = ClampOversizedMessages(max_part_chars=1_000, keep_head_chars=50, keep_tail_chars=50)
        rc = _make_request_context([_assistant(text)])
        result = await cap.before_model_request(_make_ctx(), rc)
        part = result.messages[0].parts[0]
        assert isinstance(part, TextPart)
        assert '[clamped: removed' in part.content

    def test_marker_format(self):
        assert _CLAMP_MARKER.format(removed=10, original=20) == '\n[clamped: removed 10 of 20 characters]\n'


# ---------------------------------------------------------------------------
# Public path — Agent(capabilities=[...])
# ---------------------------------------------------------------------------


class TestPublicPath:
    @pytest.fixture
    def anyio_backend(self) -> str:
        # A full agent.run only needs to be exercised once; the trio backend hits a
        # TestModel event-loop quirk in core unrelated to compaction.
        return 'asyncio'

    @pytest.mark.anyio
    async def test_capabilities_wired_into_agent(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agent = Agent(
            TestModel(),
            capabilities=[ClearToolResults(max_tokens=1, keep_pairs=0)],
        )
        result = await agent.run('hello')
        assert result.output is not None

    @pytest.mark.anyio
    async def test_clamp_oversized_wired_into_agent(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agent = Agent(
            TestModel(),
            capabilities=[ClampOversizedMessages(max_part_chars=1)],
        )
        result = await agent.run('hello')
        assert result.output is not None

    @pytest.mark.anyio
    async def test_clear_does_not_break_tool_search_on_next_request(self):
        # Regression for #380 through the real agent graph. Core re-reads a
        # `ToolSearchReturnPart`'s structured content on every request via
        # `parse_discovered_tools`; blanking it to a string used to crash the *next*
        # request. Drive a multi-request run (search -> clear fires -> another request)
        # and assert it completes with discovery intact.
        from pydantic_ai import Agent, Tool
        from pydantic_ai.capabilities import ToolSearch
        from pydantic_ai.models.function import FunctionModel

        def hidden_gem(x: int) -> int:
            return x + 1

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            step = sum(isinstance(m, ModelResponse) for m in messages)
            if step == 0:  # discover the deferred tool
                args = {'queries': ['hidden']}
                return ModelResponse(parts=[ToolCallPart(tool_name='search_tools', args=args, tool_call_id='ts0')])
            if step == 1:  # call it, forcing the request that re-reads the (cleared) search result
                return ModelResponse(parts=[ToolCallPart(tool_name='hidden_gem', args={'x': 1}, tool_call_id='hg1')])
            return ModelResponse(parts=[TextPart(content='done')])

        agent = Agent(
            FunctionModel(model_fn),
            tools=[Tool(hidden_gem, defer_loading=True)],
            capabilities=[ToolSearch(), ClearToolResults(max_messages=1, keep_pairs=0)],
        )
        result = await agent.run('go')

        assert result.output == 'done'
        messages = result.all_messages()
        search_returns = [
            p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolSearchReturnPart)
        ]
        # The typed search return kept its concrete type and structured (dict) content.
        assert search_returns
        assert all(isinstance(p.content, dict) for p in search_returns)
        # Core still recovers the discovered tool -- the read that crashed in #380.
        assert parse_discovered_tools(messages) == {'hidden_gem'}


# ---------------------------------------------------------------------------
# Remaining branch coverage — defensive paths in shared helpers
# ---------------------------------------------------------------------------


class TestHelperBranchCoverage:
    def test_prepend_returns_trimmed_when_first_user_not_discarded(self):
        first = _user('task')
        messages: list[ModelMessage] = [first, _assistant('a'), _user('b')]
        # cutoff=0 -> first (idx 0) is not before the cut, so it is left as-is.
        assert prepend_first_user_message(messages, 0, messages) == messages

    def test_extract_system_prompts_all_system_loop_completes(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='a')]),
            ModelRequest(parts=[SystemPromptPart(content='b')]),
        ]
        assert [p.content for p in _extract_system_prompts(msgs)] == ['a', 'b']

    def test_a_retry_and_a_thinking_block_are_counted(self):
        """Both are sent to the provider, and a run under load is where they appear."""
        from pydantic_ai.messages import RetryPromptPart, ThinkingPart

        msgs: list[ModelMessage] = [
            ModelRequest(parts=[RetryPromptPart(content='r' * 400)]),
            ModelResponse(parts=[ThinkingPart(content='t' * 400)]),
        ]
        assert estimate_token_count(msgs) == 200
        # `_format_messages` renders history for a summarizer prompt, which is a different
        # question from what the request costs; a retry and a thinking block stay out of it.
        assert _format_messages(msgs) == ''

    def test_a_provider_side_tool_call_and_its_result_are_counted(self):
        """A web search runs on the provider's side and its result still lands in the context."""
        from pydantic_ai.messages import NativeToolCallPart, NativeToolReturnPart

        msgs: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    NativeToolCallPart(tool_name='web_search', args={'q': 'x' * 200}, tool_call_id='c1'),
                    NativeToolReturnPart(tool_name='web_search', content='r' * 200, tool_call_id='c1'),
                ]
            )
        ]

        assert estimate_token_count(msgs) > 100

    def test_instructions_are_counted_once_however_many_turns_carry_them(self):
        """A request sends one set; summing them would multiply a system prompt by the turn count."""
        instructions = 'i' * 400
        one_turn: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='')], instructions=instructions)]
        four_turns: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='')], instructions=instructions) for _ in range(4)
        ]

        assert estimate_token_count(one_turn) == 100
        assert estimate_token_count(four_turns) == 100

    def test_a_history_with_no_instructions_counts_none(self):
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='')])]

        assert estimate_token_count(msgs) == 0

    def test_a_binary_part_is_not_counted_as_characters(self):
        """`FilePart` carries bytes; its length in characters would mean nothing."""
        from pydantic_ai.messages import BinaryContent, FilePart

        msgs: list[ModelMessage] = [
            ModelResponse(parts=[FilePart(content=BinaryContent(data=b'\x00' * 4_000, media_type='image/png'))])
        ]
        assert estimate_token_count(msgs) == 0
        assert _format_messages(msgs) == ''

    def test_user_prompt_text_skips_non_text_content(self):
        from pydantic_ai.messages import ImageUrl

        part = UserPromptPart(content=[ImageUrl(url='https://example.com/y.png'), 'hello'])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        assert estimate_token_count(msgs) == len('hello') // 4
        assert 'hello' in _format_messages(msgs)


class TestSummarizingCompactionPreserveBranches:
    @pytest.mark.anyio
    async def test_preserve_with_no_user_messages(self):
        comp = SummarizingCompaction(
            model='test:m', max_messages=2, keep_messages=1, preserve_first_user_message=True, incremental=False
        )
        messages: list[ModelMessage] = [_assistant('a'), _assistant('b'), _assistant('c')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'No-user summary.'
        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance
            result = await comp.before_model_request(ctx, rc)

        # Summary message + preserved tail, no first-user message prepended.
        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        assert any(isinstance(p, SystemPromptPart) and 'No-user summary.' in p.content for p in first_msg.parts)

    @pytest.mark.anyio
    async def test_preserve_when_first_user_already_in_tail(self):
        comp = SummarizingCompaction(
            model='test:m', max_messages=2, keep_messages=2, preserve_first_user_message=True, incremental=False
        )
        messages: list[ModelMessage] = [_assistant('x'), _assistant('y'), _user('only user'), _assistant('z')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Tail summary.'
        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance
            result = await comp.before_model_request(ctx, rc)

        # The only user message is within the kept tail, so it is not duplicated.
        user_count = sum(
            1 for m in result.messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)
        )
        assert user_count == 1


# ---------------------------------------------------------------------------
# OTel / Logfire instrumentation: the `compact_messages` span
# ---------------------------------------------------------------------------


def _compact_spans(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    """Return only the `compact_messages` spans, which this package controls.

    Core's own instrumentation span and attribute names have changed across pydantic-ai
    versions, so the assertions here stay on the spans and attributes this package emits.
    """
    return [s for s in capfire.exporter.exported_spans_as_dict() if s['name'] == 'compact_messages']


def _make_ctx_with_tracer() -> Any:
    """A fake RunContext whose `tracer` exports to the active `CaptureLogfire` provider.

    The `capfire` fixture configures the global OTel provider, so a tracer fetched from it
    captures the `compact_messages` span without needing a full instrumented `Agent` run.
    """
    from opentelemetry.trace import get_tracer

    ctx = _make_ctx()
    ctx.tracer = get_tracer('test')
    return ctx


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
class TestCompactionSpan:
    @pytest.fixture
    def anyio_backend(self) -> str:
        # A full agent.run only needs the asyncio backend; trio hits a TestModel
        # event-loop quirk in core unrelated to compaction.
        return 'asyncio'

    @pytest.mark.anyio
    async def test_span_emitted_when_threshold_exceeded(self, capfire: CaptureLogfire) -> None:
        from pydantic_ai import Agent
        from pydantic_ai.models.instrumented import InstrumentationSettings
        from pydantic_ai.models.test import TestModel

        agent: Agent[None, str] = Agent(
            TestModel(),
            capabilities=[SlidingWindowCompaction(max_tokens=1, keep_messages=1)],
        )
        agent.instrument = InstrumentationSettings()
        history: list[ModelMessage] = [_user('first'), _assistant('a'), _user('second'), _assistant('b')]
        await agent.run('a reasonably long prompt that exceeds one token', message_history=history)

        spans = _compact_spans(capfire)
        # Exactly one compaction runs (TestModel makes a single request); `== 1` also guards against
        # double-emission, and the `>` checks assert the window actually shrank rather than just
        # reporting integers.
        assert len(spans) == 1
        attrs = spans[0]['attributes']
        assert attrs['gen_ai.conversation.compacted'] is True
        assert attrs['compaction.strategy'] == 'SlidingWindowCompaction'
        assert attrs['compaction.messages_before'] > attrs['compaction.messages_after']
        assert attrs['compaction.tokens_before'] > attrs['compaction.tokens_after']

    @pytest.mark.anyio
    async def test_no_span_when_threshold_not_exceeded(self, capfire: CaptureLogfire) -> None:
        from pydantic_ai import Agent
        from pydantic_ai.models.instrumented import InstrumentationSettings
        from pydantic_ai.models.test import TestModel

        agent: Agent[None, str] = Agent(
            TestModel(),
            capabilities=[SlidingWindowCompaction(max_tokens=1_000_000, keep_messages=1)],
        )
        agent.instrument = InstrumentationSettings()
        await agent.run('short prompt')

        assert _compact_spans(capfire) == []

    @pytest.mark.anyio
    async def test_summarizing_compaction_emits_span(self, capfire: CaptureLogfire) -> None:
        comp = SummarizingCompaction(model='test:m', max_messages=2, keep_messages=1, incremental=False)
        messages: list[ModelMessage] = [_user('first'), _assistant('a'), _user('b'), _assistant('c')]
        rc = _make_request_context(messages)
        ctx = _make_ctx_with_tracer()

        mock_result = AsyncMock()
        mock_result.output = 'Summary.'
        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance
            await comp.before_model_request(ctx, rc)

        spans = _compact_spans(capfire)
        assert len(spans) == 1
        assert spans[0]['attributes']['compaction.strategy'] == 'SummarizingCompaction'

    @pytest.mark.anyio
    async def test_clamp_emits_span_only_when_a_part_is_clamped(self, capfire: CaptureLogfire) -> None:
        comp = ClampOversizedMessages(max_part_chars=4, keep_head_chars=1, keep_tail_chars=1)

        not_oversized: list[ModelMessage] = [_assistant('ab')]
        await comp.before_model_request(_make_ctx_with_tracer(), _make_request_context(not_oversized))
        assert _compact_spans(capfire) == []

        oversized: list[ModelMessage] = [_assistant('a' * 50)]
        await comp.before_model_request(_make_ctx_with_tracer(), _make_request_context(oversized))
        spans = _compact_spans(capfire)
        assert len(spans) == 1
        assert spans[0]['attributes']['compaction.strategy'] == 'ClampOversizedMessages'

    @pytest.mark.anyio
    async def test_clamp_emits_span_for_oversized_tool_call_args(self, capfire: CaptureLogfire) -> None:
        comp = ClampOversizedMessages(max_part_chars=4, keep_head_chars=1, keep_tail_chars=1, clamp_tool_call_args=True)
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name='fn', args={'q': 'x' * 50}, tool_call_id='tc1')])
        ]
        await comp.before_model_request(_make_ctx_with_tracer(), _make_request_context(messages))

        spans = _compact_spans(capfire)
        assert len(spans) == 1
        assert spans[0]['attributes']['compaction.strategy'] == 'ClampOversizedMessages'

    @pytest.mark.anyio
    async def test_clamp_no_span_for_non_oversized_or_skipped_parts(self, capfire: CaptureLogfire) -> None:
        from pydantic_ai.messages import ThinkingPart

        comp = ClampOversizedMessages(max_part_chars=1_000, clamp_tool_call_args=True)
        messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='fn', args={'q': 'x'}, tool_call_id='tc1'),
                    ThinkingPart(content='thinking'),
                ]
            )
        ]
        await comp.before_model_request(_make_ctx_with_tracer(), _make_request_context(messages))

        assert _compact_spans(capfire) == []

    @pytest.mark.anyio
    async def test_tiered_emits_single_span_not_one_per_tier(self, capfire: CaptureLogfire) -> None:
        comp: TieredCompaction[None] = TieredCompaction(
            tiers=[
                ClearToolResults(max_tokens=1, keep_pairs=0),
                SlidingWindowCompaction(max_tokens=1, keep_messages=1),
            ],
            target_tokens=1,
        )
        messages: list[ModelMessage] = [
            _user('first'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1', 'a long tool result that takes up space'),
            _assistant('done'),
        ]
        await comp.before_model_request(_make_ctx_with_tracer(), _make_request_context(messages))

        spans = _compact_spans(capfire)
        # The orchestrator drives each tier's `compact` directly, so only one span is emitted.
        assert len(spans) == 1
        assert spans[0]['attributes']['compaction.strategy'] == 'TieredCompaction'

    @pytest.mark.anyio
    async def test_no_span_when_compaction_is_noop(self, capfire: CaptureLogfire) -> None:
        # DeduplicateFileReads has no threshold, so its trigger always fires, but with no
        # superseded reads `compact` returns the history unchanged and no span should be emitted.
        comp = DeduplicateFileReads(file_key=_file_key)
        messages: list[ModelMessage] = [
            _read_call('tc1', 'a.py'),
            _read_return('tc1', 'a body'),
            _read_call('tc2', 'b.py'),
            _read_return('tc2', 'b body'),
        ]
        await comp.before_model_request(_make_ctx_with_tracer(), _make_request_context(messages))

        assert _compact_spans(capfire) == []

    @pytest.mark.anyio
    async def test_span_emitted_when_dedup_changes_history(self, capfire: CaptureLogfire) -> None:
        comp = DeduplicateFileReads(file_key=_file_key)
        messages: list[ModelMessage] = [
            _read_call('tc1', 'a.py'),
            _read_return('tc1', 'first'),
            _read_call('tc2', 'a.py'),
            _read_return('tc2', 'second'),
        ]
        await comp.before_model_request(_make_ctx_with_tracer(), _make_request_context(messages))

        spans = _compact_spans(capfire)
        assert len(spans) == 1
        assert spans[0]['attributes']['compaction.strategy'] == 'DeduplicateFileReads'

    @pytest.mark.anyio
    async def test_clear_tool_results_emits_span(self, capfire: CaptureLogfire) -> None:
        # ClearToolResults is otherwise only exercised inside TieredCompaction, which reports the
        # orchestrator's name -- so this is the only check on its own `strategy` literal.
        comp = ClearToolResults(max_tokens=1, keep_pairs=0)
        messages: list[ModelMessage] = [
            _user('first'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1', 'a long tool result that takes up space'),
            _assistant('done'),
        ]
        await comp.before_model_request(_make_ctx_with_tracer(), _make_request_context(messages))

        spans = _compact_spans(capfire)
        assert len(spans) == 1
        assert spans[0]['attributes']['compaction.strategy'] == 'ClearToolResults'


# ---------------------------------------------------------------------------
# compact_with_span helper internals
# ---------------------------------------------------------------------------


class TestHistoryChanged:
    def test_same_object_is_unchanged(self):
        msgs: list[ModelMessage] = [_user('a')]
        assert _history_changed(msgs, msgs) is False

    def test_different_length_is_changed(self):
        before: list[ModelMessage] = [_user('a')]
        after: list[ModelMessage] = [_user('a'), _user('b')]
        assert _history_changed(before, after) is True

    def test_same_length_equal_is_unchanged(self):
        # Distinct list objects holding equal elements compares unchanged. A shared message
        # object avoids the per-message timestamp that would otherwise break equality.
        shared = _user('a')
        before: list[ModelMessage] = [shared]
        after: list[ModelMessage] = [shared]
        assert before is not after
        assert _history_changed(before, after) is False

    def test_same_length_unequal_is_changed(self):
        before: list[ModelMessage] = [_user('a')]
        after: list[ModelMessage] = [_user('b')]
        assert _history_changed(before, after) is True


class TestCompactWithSpan:
    @pytest.mark.anyio
    async def test_no_op_returns_without_starting_span(self):
        # A NoOpTracer would never record anyway; this guards that `compact` runs and the
        # unchanged result is returned without attempting to start a span.
        messages: list[ModelMessage] = [_user('a')]

        async def _compact() -> list[ModelMessage]:
            return messages

        result = await compact_with_span(_make_ctx(), strategy='Strat', messages=messages, compact=_compact)
        assert result is messages

    @pytest.mark.anyio
    @pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
    async def test_recording_span_sets_attributes(self, capfire: CaptureLogfire) -> None:
        # Distinct text lengths plus a character-counting tokenizer pin the exact attribute values.
        # A before/after swap, computing both token counts from one list, or ignoring the strategy
        # tokenizer would each change a number this test checks.
        before: list[ModelMessage] = [_user('aaaa'), _user('bb')]
        after: list[ModelMessage] = [_user('aaaa')]
        seen: list[str] = []

        def _tokenizer(text: str) -> int:
            seen.append(text)
            return len(text)

        async def _compact() -> list[ModelMessage]:
            return after

        result = await compact_with_span(
            _make_ctx_with_tracer(), strategy='Strat', messages=before, compact=_compact, tokenizer=_tokenizer
        )
        assert result is after

        spans = _compact_spans(capfire)
        assert len(spans) == 1
        attrs = spans[0]['attributes']
        assert attrs['gen_ai.conversation.compacted'] is True
        assert attrs['compaction.strategy'] == 'Strat'
        assert attrs['compaction.messages_before'] == 2
        assert attrs['compaction.messages_after'] == 1
        # tokenizer counts characters: before = len('aaaa') + len('bb') = 6, after = len('aaaa') = 4.
        assert attrs['compaction.tokens_before'] == 6
        assert attrs['compaction.tokens_after'] == 4
        assert attrs['compaction.tokens_before'] > attrs['compaction.tokens_after']
        assert seen  # the strategy tokenizer reached the span attributes, not the default heuristic

    @pytest.mark.anyio
    async def test_non_recording_tracer_skips_attributes(self):
        # A no-op tracer returns a non-recording span, so attribute computation is skipped.
        before: list[ModelMessage] = [_user('a'), _user('b')]
        after: list[ModelMessage] = [_user('a')]
        called = False

        def _tokenizer(_text: str) -> int:  # pragma: no cover - asserted never called
            nonlocal called
            called = True
            return 1

        async def _compact() -> list[ModelMessage]:
            return after

        result = await compact_with_span(
            _make_ctx(), strategy='Strat', messages=before, compact=_compact, tokenizer=_tokenizer
        )
        assert result is after
        assert called is False


# ---------------------------------------------------------------------------
# Pin contract (survives_compaction)
# ---------------------------------------------------------------------------

_BRIDGE_ANCHOR = 'produced by a different model'
"""Stable fragment of the bridge-prefix wording; the full text is provisional (eval-rig pending)."""

_UPDATE_ANCHOR = 'Update it using the conversation above'
"""Stable fragment of the incremental update instruction; likewise provisional."""


def _pinned_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[pin(text)])


def _part_text(part: UserPromptPart) -> str:
    if isinstance(part.content, str):
        return part.content
    return ''.join(
        item if isinstance(item, str) else item.content if isinstance(item, TextContent) else ''
        for item in part.content
    )


def _receipt_parts(messages: list[ModelMessage]) -> list[str]:
    receipts: list[str] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                text = _part_text(part)
                if 'messages, ~' in text and 'was ' in text:
                    receipts.append(text)
    return receipts


def _pinned_texts(messages: list[ModelMessage]) -> list[str]:
    pins: list[str] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart) and is_pinned(part):
                pins.append(_part_text(part))
    return pins


class TestPinning:
    def test_pin_wraps_and_is_detected(self):
        part = pin('keep me')
        assert isinstance(part, UserPromptPart)
        assert _part_text(part) == 'keep me'
        assert is_pinned(part)

    def test_pin_marker_is_not_content_spoofable(self):
        assert not is_pinned(UserPromptPart(content='<pinned>\nx\n</pinned>'))

    def test_is_pinned_false_cases(self):
        assert not is_pinned(UserPromptPart(content='plain'))
        assert not is_pinned(UserPromptPart(content=['list', 'content']))
        assert not is_pinned(TextPart(content='<pinned>'))

    @pytest.mark.anyio
    async def test_every_dropped_pin_is_reinjected_in_order(self):
        sw = SlidingWindowCompaction(max_messages=4, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [
            _pinned_msg('first pin'),
            _user('chatter'),
            _pinned_msg('second pin'),
            _assistant('a'),
            _user('recent'),
        ]
        result = await sw.compact(messages, _make_ctx())
        assert [t for t in _pinned_texts(result)] == [
            'first pin',
            'second pin',
        ]

    @pytest.mark.anyio
    async def test_reinjected_pin_lands_after_leading_context(self):
        # The summary message is system-only, so pins are placed after it, not above it.
        comp = SummarizingCompaction(
            model='test:m', max_messages=3, keep_messages=1, bridge_prefix=False, preserve_first_user_message=False
        )
        messages: list[ModelMessage] = [_pinned_msg('durable'), _assistant('a'), _user('b'), _assistant('c')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S')):
            result = await comp.compact(messages, _make_ctx())
        first, second = result[0], result[1]
        assert isinstance(first, ModelRequest)
        assert any(isinstance(p, SystemPromptPart) and p.content.startswith(_SUMMARY_PREFIX) for p in first.parts)
        assert isinstance(second, ModelRequest)
        assert is_pinned(second.parts[0])

    @pytest.mark.anyio
    async def test_reinjected_pin_leads_when_no_system_context(self):
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_pinned_msg('durable'), _assistant('a'), _user('b'), _assistant('c')]
        result = await sw.compact(messages, _make_ctx())
        head = result[0]
        assert isinstance(head, ModelRequest)
        assert is_pinned(head.parts[0])

    @pytest.mark.anyio
    async def test_reinjection_appends_when_tail_is_all_system(self):
        # Every surviving message is system-only, so the placement scan runs to the end.
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=2, preserve_first_user_message=False)
        messages: list[ModelMessage] = [
            _pinned_msg('durable'),
            _assistant('a'),
            ModelRequest(parts=[SystemPromptPart(content='s1')]),
            ModelRequest(parts=[SystemPromptPart(content='s2')]),
        ]
        result = await sw.compact(messages, _make_ctx())
        assert _pinned_texts(result) == ['durable']
        last = result[-1]
        assert isinstance(last, ModelRequest)
        assert is_pinned(last.parts[0])

    @pytest.mark.anyio
    async def test_history_without_pins_is_untouched(self):
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        result = await sw.compact(messages, _make_ctx())
        assert _pinned_texts(result) == []

    @pytest.mark.anyio
    async def test_surviving_pin_is_not_duplicated(self):
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=2, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_user('old'), _assistant('a'), _pinned_msg('durable'), _assistant('c')]
        result = await sw.compact(messages, _make_ctx())
        assert _pinned_texts(result) == ['durable']

    @pytest.mark.anyio
    async def test_duplicate_pin_is_reinjected_when_only_one_survives(self):
        sw = SlidingWindowCompaction(max_messages=4, keep_messages=2, preserve_first_user_message=False)
        messages: list[ModelMessage] = [
            _pinned_msg('durable'),
            _assistant('a'),
            _pinned_msg('durable'),
            _assistant('b'),
        ]
        result = await sw.compact(messages, _make_ctx())
        assert _pinned_texts(result) == ['durable', 'durable']


# ---------------------------------------------------------------------------
# Receipts -- formatting, handle discovery, span plumbing
# ---------------------------------------------------------------------------


class _FakeTranscriptStore:
    def __init__(self, handle: str | None) -> None:
        self._handle = handle

    def compaction_transcript_handle(self) -> str | None:
        return self._handle


class _NotAStore:
    pass


class TestReceipts:
    @pytest.mark.anyio
    async def test_only_receipt_shaped_parts_are_de_accumulated(self):
        # A prior receipt is replaced; look-alike and non-user parts in the same history survive.
        sw = SlidingWindowCompaction(max_messages=4, keep_messages=4, receipts=True)
        decoy_system = ModelRequest(parts=[SystemPromptPart(content='[History before this point]')])
        decoy_sequence = ModelRequest(parts=[UserPromptPart(content=['[History before this point]'])])
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), decoy_system, decoy_sequence]
        result = await sw.compact(messages, _make_ctx())
        assert len(_receipt_parts(result)) == 1
        assert decoy_system in result
        assert decoy_sequence in result


class _CtxWith:
    """Build the fake run context variants the handle-discovery paths need."""

    @staticmethod
    def capabilities(**caps: object) -> Any:
        ctx = _make_ctx()
        ctx.capabilities = dict(caps)
        return ctx


async def _receipt_for(ctx: Any) -> str:
    sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True, preserve_first_user_message=False)
    messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
    return _receipt_parts(await sw.compact(messages, ctx))[0]


class TestTranscriptHandleDiscovery:
    @pytest.mark.anyio
    async def test_no_capabilities_attr(self):
        assert 'Persisted run handle' not in await _receipt_for(_make_ctx())

    @pytest.mark.anyio
    async def test_empty_capabilities(self):
        assert 'Persisted run handle' not in await _receipt_for(_CtxWith.capabilities())

    @pytest.mark.anyio
    async def test_capability_without_method_skipped(self):
        assert 'Persisted run handle' not in await _receipt_for(_CtxWith.capabilities(x=_NotAStore()))

    @pytest.mark.anyio
    async def test_capability_returning_none_continues(self):
        ctx = _CtxWith.capabilities(a=_FakeTranscriptStore(None), b=_FakeTranscriptStore('found'))
        assert 'Persisted run handle: found.' in await _receipt_for(ctx)

    def test_isinstance_protocol(self):
        assert isinstance(_FakeTranscriptStore('h'), TranscriptHandleProvider)
        assert not isinstance(_NotAStore(), TranscriptHandleProvider)


# ---------------------------------------------------------------------------
# SummarizingCompaction receipts
# ---------------------------------------------------------------------------


def _patched_summary_agent(output: str) -> Any:
    mock_result = AsyncMock()
    mock_result.output = output
    mock_agent_instance = AsyncMock()
    mock_agent_instance.run.return_value = mock_result
    return mock_agent_instance


class TestSummarizingReceipts:
    @pytest.mark.anyio
    async def test_receipt_present_and_after_summary(self):
        comp = SummarizingCompaction(
            model='openai:gpt-4o-mini',
            max_messages=3,
            keep_messages=1,
            receipts=True,
            bridge_prefix=False,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('SUMMARY')):
            result = await comp.before_model_request(ctx, rc)
        receipts = _receipt_parts(result.messages)
        assert len(receipts) == 1
        assert 'was summarized by gpt' in receipts[0]
        assert 'Persisted run handle' not in receipts[0]

    @pytest.mark.anyio
    async def test_receipt_is_byte_deterministic(self):
        def _run() -> str:
            return 'SUMMARY'

        async def _once() -> str:
            comp = SummarizingCompaction(
                model='openai:gpt-4o-mini',
                max_messages=3,
                keep_messages=1,
                receipts=True,
                bridge_prefix=False,
                preserve_first_user_message=False,
            )
            messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
            rc = _make_request_context(messages)
            with patch('pydantic_ai.Agent', return_value=_patched_summary_agent(_run())):
                result = await comp.before_model_request(_make_ctx(), rc)
            return _receipt_parts(result.messages)[0]

        first = await _once()
        second = await _once()
        assert first == second
        assert first.encode('utf-8') == second.encode('utf-8')

    @pytest.mark.anyio
    async def test_receipt_handle_when_transcript_store_attached(self):
        comp = SummarizingCompaction(
            model='openai:gpt-4o-mini',
            max_messages=3,
            keep_messages=1,
            receipts=True,
            bridge_prefix=False,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        ctx.capabilities = {'sp': _FakeTranscriptStore('librarian-42')}
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('SUMMARY')):
            result = await comp.before_model_request(ctx, rc)
        assert 'Persisted run handle: librarian-42.' in _receipt_parts(result.messages)[0]

    @pytest.mark.anyio
    async def test_receipts_do_not_accumulate(self):
        comp = SummarizingCompaction(
            model='openai:gpt-4o-mini',
            max_messages=3,
            keep_messages=2,
            receipts=True,
            bridge_prefix=False,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S1')):
            first = await comp.compact(messages, _make_ctx())
        # Feed the compacted history back through another compaction.
        extended = [*first, _user('e'), _assistant('f')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S2')):
            second = await comp.compact(extended, _make_ctx())
        assert len(_receipt_parts(second)) == 1

    @pytest.mark.anyio
    async def test_marker_before_system_prompt_does_not_drop_the_system_prompt(self):
        comp = SummarizingCompaction(
            model='test:m', max_messages=3, keep_messages=1, receipts=True, bridge_prefix=False
        )
        receipt_source = SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True)
        seeded = await receipt_source.compact([_user('a'), _assistant('b'), _user('c'), _assistant('d')], _make_ctx())
        messages = [
            seeded[0],
            ModelRequest(parts=[SystemPromptPart('system instruction')]),
            _user('later'),
            _assistant('e'),
        ]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S')):
            result = await comp.compact(messages, _make_ctx())
        first = result[0]
        assert isinstance(first, ModelRequest)
        assert any(isinstance(part, SystemPromptPart) and part.content == 'system instruction' for part in first.parts)


class TestSlidingWindowCompactionReceipts:
    @pytest.mark.anyio
    async def test_receipt_prepended_with_drop_wording(self):
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        rc = _make_request_context(messages)
        result = await sw.before_model_request(_make_ctx(), rc)
        first = result.messages[0]
        assert isinstance(first, ModelRequest)
        receipt = first.parts[0]
        assert isinstance(receipt, UserPromptPart)
        assert _part_text(receipt).startswith('[History before this point')
        assert 'was dropped by the harness' in _part_text(receipt)

    @pytest.mark.anyio
    async def test_receipt_reserves_a_message_slot(self):
        sw = SlidingWindowCompaction(max_messages=4, keep_messages=3, receipts=True, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d'), _user('e')]
        result = await sw.before_model_request(_make_ctx(), _make_request_context(messages))
        assert len(result.messages) == 3
        assert len(_receipt_parts(result.messages)) == 1

    @pytest.mark.anyio
    async def test_receipt_reserves_tokens(self):
        sw = SlidingWindowCompaction(
            max_tokens=10, keep_tokens=5, receipts=True, preserve_first_user_message=False, tokenizer=len
        )
        messages: list[ModelMessage] = [_user('a' * 20), _assistant('b' * 20), _user('c' * 20)]
        result = await sw.before_model_request(_make_ctx(), _make_request_context(messages))
        assert len(_receipt_parts(result.messages)) == 1

    @pytest.mark.anyio
    async def test_receipt_does_not_displace_the_original_first_user_turn(self):
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True)
        messages: list[ModelMessage] = [_user('original task'), _assistant('a'), _user('later'), _assistant('b')]
        first = await sw.compact(messages, _make_ctx())
        second = await sw.compact([*first, _user('next'), _assistant('c')], _make_ctx())
        assert 'original task' in _user_texts(second)

    @pytest.mark.anyio
    async def test_receipt_excludes_restored_messages_from_drop_count(self):
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True)
        messages: list[ModelMessage] = [_user('original task'), _assistant('a'), _user('later'), _assistant('b')]
        result = await sw.compact(messages, _make_ctx())
        assert _receipt_parts(result)[0].startswith('[History before this point (3 messages,')


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
class TestReceiptSpanEvent:
    @pytest.fixture
    def anyio_backend(self) -> str:
        return 'asyncio'

    @pytest.mark.anyio
    async def test_sliding_window_emits_receipt_event(self, capfire: CaptureLogfire) -> None:
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        await sw.before_model_request(_make_ctx_with_tracer(), _make_request_context(messages))
        spans = _compact_spans(capfire)
        assert len(spans) == 1
        events: list[dict[str, Any]] = spans[0].get('events') or []
        receipt_events = [e for e in events if e['name'] == 'compaction.receipt']
        assert len(receipt_events) == 1
        attrs = receipt_events[0]['attributes']
        assert attrs['compaction.receipt.strategy'] == 'SlidingWindowCompaction'
        assert attrs['compaction.receipt.by'] == 'the harness'
        assert 'compaction.receipt.handle' not in attrs

    @pytest.mark.anyio
    async def test_receipt_event_carries_handle(self, capfire: CaptureLogfire) -> None:
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True, preserve_first_user_message=False)
        ctx = _make_ctx_with_tracer()
        ctx.capabilities = {'sp': _FakeTranscriptStore('run-77')}
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        await sw.before_model_request(ctx, _make_request_context(messages))
        events: list[dict[str, Any]] = _compact_spans(capfire)[0].get('events') or []
        receipt_events = [e for e in events if e['name'] == 'compaction.receipt']
        assert receipt_events[0]['attributes']['compaction.receipt.handle'] == 'run-77'


# ---------------------------------------------------------------------------
# keep_user_messages
# ---------------------------------------------------------------------------


class TestKeepUserMessages:
    def test_validation_bad_max_chars(self):
        with pytest.raises(ValueError, match='keep_user_messages_max_chars must be positive'):
            SummarizingCompaction(model='test', max_messages=10, keep_user_messages_max_chars=0)

    @pytest.mark.anyio
    async def test_newest_summarized_user_message_is_preserved_and_truncated(self):
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            keep_user_messages=True,
            keep_user_messages_max_chars=10,
            bridge_prefix=False,
        )
        long_text = 'x' * 40
        messages: list[ModelMessage] = [
            _user('short one'),
            _assistant('a'),
            _user(long_text),
            _assistant('b'),
            _user('recent'),
        ]
        rc = _make_request_context(messages)
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('SUMMARY')):
            result = await comp.before_model_request(_make_ctx(), rc)
        kept = _user_texts(result.messages)
        assert any(t.endswith('[...]') and len(t) == 10 for t in kept)

    @pytest.mark.anyio
    async def test_keeps_non_string_user_content_and_skips_non_user(self):
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            keep_user_messages=True,
            bridge_prefix=False,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [
            _assistant('an assistant turn is skipped'),
            ModelRequest(parts=[SystemPromptPart(content='no user here')]),
            ModelRequest(parts=[UserPromptPart(content=[TextContent(content='multimodal')])]),
            _assistant('recent'),
        ]
        rc = _make_request_context(messages)
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('SUMMARY')):
            result = await comp.before_model_request(_make_ctx(), rc)
        # Exactly one user part survives the summarized prefix, and content that fits is untouched.
        kept = [
            p for m in result.messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)
        ]
        assert len(kept) == 1
        assert kept[0].content == [TextContent(content='multimodal')]

    @pytest.mark.anyio
    async def test_bounds_text_inside_sequence_content(self):
        from pydantic_ai.messages import CachePoint

        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=2,
            keep_user_messages=True,
            keep_user_messages_max_chars=30,
            bridge_prefix=False,
            preserve_first_user_message=False,
        )
        cache_point = CachePoint()
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=['a' * 100, cache_point, TextContent(content='b' * 100)])]),
            _assistant('a'),
            _user('b'),
            _assistant('recent'),
        ]
        rc = _make_request_context(messages)
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('SUMMARY')):
            result = await comp.before_model_request(_make_ctx(), rc)
        bounded = next(
            p
            for m in result.messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart) and not isinstance(p.content, str)
        )
        first, passthrough = bounded.content
        assert isinstance(first, str)
        assert first.endswith('[...]')
        assert len(first) == 30
        # Non-text items ride along untouched.
        assert passthrough is cache_point

    @pytest.mark.anyio
    async def test_retained_messages_do_not_reenter_later_compactions(self):
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=2,
            keep_user_messages=True,
            bridge_prefix=False,
        )
        original = ModelRequest(
            parts=[UserPromptPart('first')],
            run_id='run-1',
            conversation_id='conversation-1',
            metadata={'source': 'caller'},
        )
        messages: list[ModelMessage] = [original, _assistant('a'), _user('second'), _assistant('b')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S1')):
            first = await comp.compact(messages, _make_ctx())
        retained = next(message for message in first if isinstance(message, ModelRequest) and message.run_id == 'run-1')
        assert retained.metadata == {'source': 'caller', 'pydantic-ai-harness.compaction.kept-user-message.v1': True}
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S2')):
            second = await comp.compact([*first, _user('third'), _assistant('c')], _make_ctx())
        assert sum('first' in text for text in _user_texts(second)) == 1

    @pytest.mark.anyio
    async def test_retained_user_messages_converge_within_the_tail_budget(self):
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            keep_user_messages=True,
            bridge_prefix=False,
        )
        messages: list[ModelMessage] = [_user('first'), _assistant('a'), _user('second'), _assistant('b')]
        summary_agent = _patched_summary_agent('S')
        with patch('pydantic_ai.Agent', return_value=summary_agent):
            first = await comp.before_model_request(_make_ctx(), _make_request_context(messages))
            second = await comp.before_model_request(_make_ctx(), _make_request_context(first.messages))
        assert summary_agent.run.await_count == 1
        assert comp.max_messages is not None
        assert len(first.messages) <= comp.max_messages
        assert second.messages == first.messages

    @pytest.mark.anyio
    async def test_retained_users_and_tail_share_the_token_budget(self):
        comp = SummarizingCompaction(
            model='test:m',
            max_tokens=2,
            keep_tokens=1,
            keep_messages=3,
            keep_user_messages=True,
            bridge_prefix=False,
            tokenizer=len,
        )
        messages: list[ModelMessage] = [_user('xx'), _assistant('x')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S')):
            result = await comp.compact(messages, _make_ctx())
        assert _user_texts(result) == []
        assert result[-1] == messages[-1]
        assert comp.keep_tokens is not None
        assert estimate_token_count(result[1:], len) <= comp.keep_tokens

    @pytest.mark.anyio
    async def test_retained_user_can_consume_the_token_budget(self):
        comp = SummarizingCompaction(
            model='test:m',
            max_tokens=2,
            keep_tokens=1,
            keep_messages=3,
            keep_user_messages=True,
            bridge_prefix=False,
            tokenizer=len,
        )
        messages: list[ModelMessage] = [_user('x'), _assistant('x')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S')):
            result = await comp.compact(messages, _make_ctx())
        assert _user_texts(result) == ['x']
        assert comp.keep_tokens is not None
        assert estimate_token_count(result[1:], len) <= comp.keep_tokens

    @pytest.mark.anyio
    async def test_older_user_is_not_retained_when_the_newest_does_not_fit(self):
        comp = SummarizingCompaction(
            model='test:m',
            max_tokens=3,
            keep_tokens=1,
            keep_messages=3,
            keep_user_messages=True,
            bridge_prefix=False,
            tokenizer=len,
        )
        messages: list[ModelMessage] = [_user('x'), _assistant('x'), _user('xx'), _assistant('x')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S')):
            result = await comp.compact(messages, _make_ctx())
        assert _user_texts(result) == []
        assert result[-1] == messages[-1]

    @pytest.mark.anyio
    async def test_pin_is_not_rebuilt_as_a_kept_user_message(self):
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            keep_user_messages=True,
            bridge_prefix=False,
        )
        messages: list[ModelMessage] = [_pinned_msg('durable'), _assistant('a'), _user('recent'), _assistant('b')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S')):
            result = await comp.compact(messages, _make_ctx())
        assert _pinned_texts(result) == ['durable']

    @pytest.mark.anyio
    async def test_tiny_user_message_budget_stays_bounded(self):
        comp = SummarizingCompaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            keep_user_messages=True,
            keep_user_messages_max_chars=3,
            bridge_prefix=False,
        )
        messages: list[ModelMessage] = [_user('long prompt'), _assistant('a'), _user('recent'), _assistant('b')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S')):
            result = await comp.compact(messages, _make_ctx())
        assert any(len(text) <= 3 for text in _user_texts(result))


# ---------------------------------------------------------------------------
# Anchored incremental summarization (opencode mechanism)
# ---------------------------------------------------------------------------


class TestAnchoredIncremental:
    @pytest.fixture
    def anyio_backend(self) -> str:
        # A real FunctionModel run only needs asyncio; trio hits a core event-loop quirk
        # unrelated to compaction (same reason as TestCompactionSpan).
        return 'asyncio'

    @pytest.mark.anyio
    async def test_previous_summary_fed_as_anchor_with_update_instruction(self):
        from pydantic_ai.messages import ModelResponse as _MR
        from pydantic_ai.messages import TextPart as _TP
        from pydantic_ai.models.function import FunctionModel

        captured: list[str] = []

        def summarize_fn(messages: list[ModelMessage], _info: AgentInfo) -> _MR:
            captured.extend(_user_texts(messages))
            return _MR(parts=[_TP(content='UPDATED SUMMARY')])

        comp = SummarizingCompaction(
            model=FunctionModel(summarize_fn),
            max_messages=3,
            keep_messages=1,
            incremental=True,
            bridge_prefix=False,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}Anchored prior summary.')]),
            _user('new work 1'),
            _assistant('r1'),
            _user('new work 2'),
            _assistant('r2'),
        ]
        rc = _make_request_context(messages)
        result = await comp.before_model_request(_make_ctx(), rc)
        prompt = '\n'.join(captured)
        assert _UPDATE_ANCHOR in prompt
        assert '<previous-summary>' in prompt
        assert 'Anchored prior summary.' in prompt
        # The updated anchor lands in the new summary message.
        first = result.messages[0]
        assert isinstance(first, ModelRequest)
        assert any('UPDATED SUMMARY' in p.content for p in first.parts if isinstance(p, SystemPromptPart))


# ---------------------------------------------------------------------------
# Cross-model bridge prefix
# ---------------------------------------------------------------------------


class TestBridgePrefix:
    async def _compact_with(
        self,
        *,
        bridge_prefix: bool = True,
        run_model: str | None = 'anthropic:claude-sonnet',
        summarizer: str | None = 'openai:gpt-4o-mini',
        ctx: Any = None,
    ) -> str:
        comp = SummarizingCompaction(
            model=summarizer,
            max_messages=3,
            keep_messages=1,
            bridge_prefix=bridge_prefix,
            preserve_first_user_message=False,
        )
        tail: list[ModelMessage] = (
            [_assistant('b'), _assistant('d')]
            if run_model is None
            else [
                ModelResponse(parts=[TextPart(content='b')], model_name=run_model),
                ModelResponse(parts=[TextPart(content='d')], model_name=run_model),
            ]
        )
        messages: list[ModelMessage] = [_user('a'), tail[0], _user('c'), tail[1]]
        run_ctx = ctx if ctx is not None else _make_ctx()
        # No capability replaces the model here, so the request goes to the run's own model --
        # which is what the bridge gate reads when the history names no model.
        rc = _make_request_context(messages, run_ctx.model)
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('BASE')):
            result = await comp.before_model_request(run_ctx, rc)
        first = result.messages[0]
        assert isinstance(first, ModelRequest)
        return next(
            p.content for p in first.parts if isinstance(p, SystemPromptPart) and p.content.startswith(_SUMMARY_PREFIX)
        )

    @pytest.mark.anyio
    async def test_prefix_added_on_family_mismatch(self):
        assert _BRIDGE_ANCHOR in await self._compact_with()

    @pytest.mark.anyio
    async def test_no_prefix_same_family(self):
        summary = await self._compact_with(run_model='openai:gpt-4o', summarizer='openai:gpt-4o-mini')
        assert _BRIDGE_ANCHOR not in summary

    @pytest.mark.anyio
    async def test_no_prefix_when_disabled(self):
        assert _BRIDGE_ANCHOR not in await self._compact_with(bridge_prefix=False)

    @pytest.mark.anyio
    async def test_disabled_by_default(self):
        comp = SummarizingCompaction(model='openai:gpt-4o-mini', max_messages=3, keep_messages=1)
        messages: list[ModelMessage] = [
            _user('a'),
            ModelResponse(parts=[TextPart(content='b')], model_name='anthropic:claude-sonnet'),
            _user('c'),
            ModelResponse(parts=[TextPart(content='d')], model_name='anthropic:claude-sonnet'),
        ]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('BASE')):
            result = await comp.compact(messages, _make_ctx())
        summary = next(
            part.content
            for message in result
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX)
        )
        assert _BRIDGE_ANCHOR not in summary

    @pytest.mark.anyio
    async def test_same_fallback_model_does_not_add_a_bridge(self):
        from pydantic_ai.models.fallback import FallbackModel

        fallback = FallbackModel(TestModel(), TestModel())
        ctx = _make_ctx()
        ctx.model = fallback
        assert _BRIDGE_ANCHOR not in await self._compact_with(run_model=None, summarizer=None, ctx=ctx)

    @pytest.mark.anyio
    async def test_run_family_falls_back_to_the_running_model(self):
        # `TestModel.model_name` is `test`, which differs from the summarizer's family.
        assert _BRIDGE_ANCHOR in await self._compact_with(run_model=None)


# ---------------------------------------------------------------------------
# Pins survive every shipped strategy
# ---------------------------------------------------------------------------


class TestPinsSurviveStrategies:
    @pytest.mark.anyio
    async def test_sliding_window(self):
        sw = SlidingWindowCompaction(max_messages=3, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_pinned_msg('PINNED STATE'), _assistant('a'), _user('b'), _assistant('c')]
        result = await sw.compact(messages, _make_ctx())
        assert any(is_pinned(p) for m in result if isinstance(m, ModelRequest) for p in m.parts)

    @pytest.mark.anyio
    async def test_summarizing(self):
        comp = SummarizingCompaction(
            model='test:m', max_messages=3, keep_messages=1, bridge_prefix=False, preserve_first_user_message=False
        )
        messages: list[ModelMessage] = [_pinned_msg('PINNED STATE'), _assistant('a'), _user('b'), _assistant('c')]
        with patch('pydantic_ai.Agent', return_value=_patched_summary_agent('S')):
            result = await comp.compact(messages, _make_ctx())
        assert any(is_pinned(p) for m in result if isinstance(m, ModelRequest) for p in m.parts)

    @pytest.mark.anyio
    async def test_clear_tool_results(self):
        ctr = ClearToolResults(max_tokens=1, keep_pairs=0)
        messages: list[ModelMessage] = [_pinned_msg('PINNED STATE'), *_pair('fn', 'c1', 'big result')]
        result = await ctr.compact(messages, _make_ctx())
        assert any(is_pinned(p) for m in result if isinstance(m, ModelRequest) for p in m.parts)

    @pytest.mark.anyio
    async def test_deduplicate_file_reads(self):
        dfr = DeduplicateFileReads(file_key=_file_key)
        messages: list[ModelMessage] = [
            _pinned_msg('PINNED STATE'),
            _tool_call('read_file', 'c1'),
            ModelRequest(parts=[ToolReturnPart(tool_name='read_file', content='v1', tool_call_id='c1')]),
        ]
        result = await dfr.compact(messages, _make_ctx())
        assert any(is_pinned(p) for m in result if isinstance(m, ModelRequest) for p in m.parts)

    @pytest.mark.anyio
    async def test_clamp_oversized(self):
        clamp = ClampOversizedMessages(max_part_chars=4, keep_head_chars=1, keep_tail_chars=1)
        messages: list[ModelMessage] = [_pinned_msg('PINNED STATE'), _assistant('y' * 40)]
        result = await clamp.compact(messages, _make_ctx())
        assert any(is_pinned(p) for m in result if isinstance(m, ModelRequest) for p in m.parts)


# ---------------------------------------------------------------------------
# StepPersistence transcript handle integration
# ---------------------------------------------------------------------------


class TestStepPersistenceHandle:
    @pytest.mark.anyio
    async def test_handle_is_run_id_and_reaches_the_receipt(self):
        from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence

        sp: StepPersistence[None] = StepPersistence(store=InMemoryStepStore(), run_id='libr-1')
        assert isinstance(sp, TranscriptHandleProvider)
        assert sp.compaction_transcript_handle() == 'libr-1'
        assert 'Persisted run handle: libr-1.' in await _receipt_for(_CtxWith.capabilities(sp=sp))

    def test_handle_none_before_materialization(self):
        from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence

        sp: StepPersistence[None] = StepPersistence(store=InMemoryStepStore())
        assert sp.compaction_transcript_handle() is None


# ---------------------------------------------------------------------------
# Structural features through a real Agent run
# ---------------------------------------------------------------------------


def _recording_model(seen: list[list[ModelMessage]]) -> FunctionModel:
    """A run model that records the history each request actually carried."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(model_fn)


def _recording_summarizer(prompts: list[str], output: str = 'THE SUMMARY') -> FunctionModel:
    """A summarizer model that records the prompt each summarization call carried."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts.append(
            '\n'.join(
                _part_text(part)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, UserPromptPart)
            )
        )
        return ModelResponse(parts=[TextPart(content=output)])

    return FunctionModel(model_fn)


class TestStructuralFeaturesThroughAgent:
    """The four structural features driven through `Agent(..., capabilities=[...])`.

    The rest of this file calls `compact` directly, so it cannot catch a break in the
    `before_model_request` wiring or in how core threads a compacted history into the next
    request. These assert on the history the model was actually sent.
    """

    @pytest.fixture
    def anyio_backend(self) -> str:
        # A full `agent.run` only needs one backend; trio hits a core event-loop quirk here
        # that has nothing to do with compaction.
        return 'asyncio'

    @pytest.mark.anyio
    async def test_receipt_reaches_the_model_and_does_not_accumulate(self):
        seen: list[list[ModelMessage]] = []
        agent = Agent(
            _recording_model(seen),
            capabilities=[SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True)],
        )
        history: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        first = await agent.run('one', message_history=history)

        receipts = _receipt_parts(seen[0])
        assert len(receipts) == 1
        assert 'was dropped by the harness' in receipts[0]

        # A second compaction replaces the receipt rather than stacking a new one beside it.
        await agent.run('two', message_history=first.all_messages())
        assert len(_receipt_parts(seen[1])) == 1

    @pytest.mark.anyio
    async def test_receipt_is_not_mistaken_for_the_first_user_turn(self):
        seen: list[list[ModelMessage]] = []
        agent = Agent(
            _recording_model(seen),
            capabilities=[SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True)],
        )
        history: list[ModelMessage] = [_user('FIRST'), _assistant('b'), _user('c'), _assistant('d')]
        first = await agent.run('one', message_history=history)
        await agent.run('two', message_history=first.all_messages())

        # `preserve_first_user_message` defaults on, so the real opening turn -- not the
        # receipt that now sits ahead of it -- is what gets carried forward.
        assert 'FIRST' in _user_texts(seen[1])

    @pytest.mark.anyio
    async def test_pin_survives_compaction_in_a_run(self):
        seen: list[list[ModelMessage]] = []
        agent = Agent(
            _recording_model(seen),
            capabilities=[SlidingWindowCompaction(max_messages=3, keep_messages=1, preserve_first_user_message=False)],
        )
        history: list[ModelMessage] = [_pinned_msg('DURABLE STATE'), _user('a'), _assistant('b'), _user('c')]
        await agent.run('go', message_history=history)

        sent = seen[0]
        assert len(sent) < len(history) + 1
        assert [
            _part_text(part)
            for message in sent
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and is_pinned(part)
        ] == ['DURABLE STATE']

    @pytest.mark.anyio
    async def test_keep_user_messages_reaches_the_model_truncated(self):
        seen: list[list[ModelMessage]] = []
        prompts: list[str] = []
        agent = Agent(
            _recording_model(seen),
            capabilities=[
                SummarizingCompaction(
                    model=_recording_summarizer(prompts),
                    max_messages=3,
                    keep_messages=2,
                    keep_user_messages=True,
                    keep_user_messages_max_chars=10,
                )
            ],
        )
        await agent.run('go', message_history=[_user('u' * 40), _assistant('b'), _user('v' * 40), _assistant('d')])

        assert len(prompts) == 1
        texts = _user_texts(seen[0])
        assert any(text.startswith('vvvvv') and text.endswith('[...]') for text in texts)

    @pytest.mark.anyio
    async def test_retained_user_turns_arrive_as_a_single_request(self):
        # `keep_user_messages` leaves the summary, the receipt, and the retained turns as
        # adjacent `ModelRequest`s. Core normalizes that into one turn after the hooks run, so
        # providers requiring a single request per turn (Bedrock Converse, Gemini) never see the
        # consecutive shape. The docs promise this; assert core still does it.
        seen: list[list[ModelMessage]] = []
        prompts: list[str] = []
        agent = Agent(
            _recording_model(seen),
            capabilities=[
                SummarizingCompaction(
                    model=_recording_summarizer(prompts),
                    max_messages=3,
                    keep_messages=2,
                    keep_user_messages=True,
                    receipts=True,
                )
            ],
        )
        await agent.run('go', message_history=[_user('a'), _assistant('b'), _user('c'), _assistant('d')])

        sent = seen[0]
        # The compacted history did contain several request-shaped pieces to begin with.
        assert len(_user_texts(sent)) > 1
        assert not any(
            isinstance(earlier, ModelRequest) and isinstance(later, ModelRequest)
            for earlier, later in zip(sent, sent[1:])
        )

    @pytest.mark.anyio
    async def test_incremental_anchors_the_next_summary_on_the_previous_one(self):
        seen: list[list[ModelMessage]] = []
        prompts: list[str] = []
        agent = Agent(
            _recording_model(seen),
            capabilities=[SummarizingCompaction(model=_recording_summarizer(prompts), max_messages=2, keep_messages=1)],
        )
        first = await agent.run('one', message_history=[_user('a'), _assistant('b'), _user('c'), _assistant('d')])
        assert len(prompts) == 1
        assert '<previous-summary>' not in prompts[0]

        await agent.run('two', message_history=first.all_messages())
        assert len(prompts) == 2
        assert '<previous-summary>\nTHE SUMMARY\n</previous-summary>' in prompts[1]
        assert _UPDATE_ANCHOR in prompts[1]
