"""Tests for window-relative triggers, usage reporting, and out-of-run compaction."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SpeechPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import AbstractModel, Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from pydantic_ai_harness.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    ClearToolResults,
    ContextUsage,
    DeduplicateFileReads,
    ReportContextUsage,
    SlidingWindowCompaction,
    SummarizingCompaction,
    SupportsFocus,
    TieredCompaction,
    WarnNearLimits,
    compact_now,
    estimate_context_tokens,
    estimate_token_count,
    resolve_context_window,
)
from pydantic_ai_harness.compaction._shared import resolve_token_trigger

try:
    from logfire.testing import CaptureLogfire

    logfire_installed = True
except ImportError:  # pragma: no cover
    logfire_installed = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeModel:
    """Stands in for a `Model`; only `model_id` is read by window resolution."""

    model_id: str = 'anthropic:claude-sonnet-4-6'


class _FakeRealtimeModel(AbstractModel):
    """A realtime model: an `AbstractModel` that is deliberately not a request-response `Model`.

    A realtime model has no context-window/token semantics, so the token triggers must skip it
    (see `resolve_token_trigger`). Only `system`/`model_name` are needed to satisfy the ABC.
    """

    @property
    def model_name(self) -> str:
        return 'gpt-4o-realtime-preview'

    @property
    def system(self) -> str:
        return 'openai'


def _ctx(model: Any = None) -> Any:
    """Minimal `RunContext`-like object for driving hooks."""

    @dataclasses.dataclass
    class _FakeCtx:
        usage: RunUsage = dataclasses.field(default_factory=RunUsage)
        usage_limits: UsageLimits | None = None
        model: Model = dataclasses.field(default_factory=TestModel)
        deps: None = None
        tracer: Tracer = dataclasses.field(default_factory=NoOpTracer)

    return _FakeCtx(model=model) if model is not None else _FakeCtx()


def _request_context(
    messages: list[ModelMessage],
    model: _FakeModel | None = None,
    model_request_parameters: ModelRequestParameters | None = None,
) -> ModelRequestContext:
    return ModelRequestContext(
        model=model if model is not None else _FakeModel(),  # type: ignore[arg-type]
        messages=messages,
        model_settings=None,
        model_request_parameters=model_request_parameters or ModelRequestParameters(),
    )


def _history(turns: int, filler: str = 'x' * 400) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for index in range(turns):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f'{index} {filler}')]))
        messages.append(ModelResponse(parts=[TextPart(content=f'reply {index} {filler}')]))
    return messages


_CLEARED = '[tool result cleared]'
_SUPERSEDED = '[superseded file read]'


def _tool_history(
    calls: int, tool_name: str = 'search', path: str | None = None, filler: str = 'x' * 400
) -> list[ModelMessage]:
    """A history of completed tool calls, which is what the clearing strategies act on."""
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='go')])]
    for index in range(calls):
        args = {'path': path} if path is not None else {'q': f'{index}'}
        messages.append(
            ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id=f'call-{index}')])
        )
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name=tool_name, content=f'result {index} {filler}', tool_call_id=f'call-{index}'
                    )
                ]
            )
        )
    return messages


def _tool_return_contents(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _message_text(message: ModelMessage) -> str:
    return ' '.join(str(getattr(part, 'content', '')) for part in message.parts)


def _patch_snapshot(monkeypatch: pytest.MonkeyPatch, lookup: Callable[[str], int | None]) -> None:
    """Stand in for the pricing snapshot `resolve_context_window` reads.

    Patching the registry rather than `resolve_context_window` keeps the real resolution path
    under test: id splitting, the `LookupError` catch, and the non-positive guard all still run.
    """

    class _Info:
        def __init__(self, window: int | None) -> None:
            self.context_window = window

    class _Snapshot:
        def find_provider_model(self, *, model_ref: str, **_: Any) -> tuple[object, _Info]:
            return object(), _Info(lookup(model_ref))

    monkeypatch.setattr('genai_prices.data_snapshot.get_snapshot', lambda: _Snapshot())


def _fixed_window(monkeypatch: pytest.MonkeyPatch, window: int | None) -> None:
    """Pin every model's window so a test asserts behaviour, not the pricing registry."""
    _patch_snapshot(monkeypatch, lambda _: window)


def _window_per_model(monkeypatch: pytest.MonkeyPatch, windows: dict[str, int | None]) -> None:
    """Give each model its own window, so a test can tell which model was consulted."""
    _patch_snapshot(monkeypatch, lambda model_ref: windows[model_ref])


# ---------------------------------------------------------------------------
# Window resolution
# ---------------------------------------------------------------------------


class TestResolveContextWindow:
    def test_known_model(self):
        window = resolve_context_window('anthropic:claude-sonnet-4-6')
        assert window is not None and window >= 100_000

    def test_a_bare_model_name_leaves_the_provider_to_the_registry(self):
        assert resolve_context_window('claude-sonnet-4-6') == resolve_context_window('anthropic:claude-sonnet-4-6')

    def test_accepts_a_model_instance(self):
        window = resolve_context_window(_FakeModel())  # type: ignore[arg-type]
        assert window is not None and window >= 100_000

    def test_unknown_model(self):
        assert resolve_context_window('anthropic:definitely-not-a-model-xyz') is None

    def test_model_without_an_id(self):
        assert resolve_context_window(_FakeModel(model_id='')) is None  # type: ignore[arg-type]

    def test_a_fallback_model_does_not_resolve(self):
        """`FallbackModel.model_id` is a composite no registry entry matches.

        Built from a real `FallbackModel` rather than a hand-written id, so the test tracks
        whatever composite core actually emits.
        """
        fallback = FallbackModel(TestModel(), TestModel())
        assert fallback.model_id.startswith('fallback:')
        assert resolve_context_window(fallback) is None

    def test_a_test_model_does_not_resolve(self):
        """`TestModel` is what every downstream suite runs against, and `test:test` is unknown."""
        assert resolve_context_window(TestModel()) is None

    def test_known_model_without_a_recorded_window(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, None)
        assert resolve_context_window('openai:gpt-4.1') is None

    def test_non_positive_window_is_treated_as_absent(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 0)
        assert resolve_context_window('openai:gpt-4.1') is None


# ---------------------------------------------------------------------------
# Trigger validation
# ---------------------------------------------------------------------------


class TestTriggerValidation:
    """The shared validator, exercised through the constructors that call it."""

    def test_rejects_both(self):
        with pytest.raises(ValueError, match='Set at most one of max_tokens or max_fraction'):
            SlidingWindowCompaction(max_tokens=1_000, max_fraction=0.5)

    def test_rejects_non_positive_tokens(self):
        with pytest.raises(ValueError, match='max_tokens must be positive'):
            SlidingWindowCompaction(max_tokens=0)

    def test_rejects_fraction_above_one(self):
        with pytest.raises(ValueError, match='max_fraction must be greater than 0 and at most 1'):
            SlidingWindowCompaction(max_fraction=1.5)

    def test_rejects_zero_fraction(self):
        with pytest.raises(ValueError, match='max_fraction must be greater than 0 and at most 1'):
            SlidingWindowCompaction(max_fraction=0.0)

    def test_accepts_neither(self):
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_messages=10)
        assert (capability.max_tokens, capability.max_fraction) == (None, None)

    def test_reports_the_caller_field_names(self):
        with pytest.raises(ValueError, match='Set at most one of target_tokens or target_fraction'):
            TieredCompaction(tiers=[SlidingWindowCompaction(max_tokens=1)], target_tokens=1, target_fraction=0.5)

    def test_rejects_a_non_positive_fallback(self):
        with pytest.raises(ValueError, match='fallback_context_window must be positive'):
            SlidingWindowCompaction(max_fraction=0.5, fallback_context_window=0)


# ---------------------------------------------------------------------------
# Strategies driven by a fraction
# ---------------------------------------------------------------------------


class TestFractionTriggers:
    async def test_sliding_window_trims_when_over_the_fraction(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_fraction=0.1, keep_messages=2)
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) < 12

    async def test_sliding_window_leaves_a_small_history_alone(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000_000)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_fraction=0.9, keep_messages=2)
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) == 12

    async def test_the_same_fraction_scales_with_the_window(self, monkeypatch: pytest.MonkeyPatch):
        """One configuration, two models: the trigger follows the window."""
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_fraction=0.5, keep_messages=2)

        _fixed_window(monkeypatch, 100)
        small = _request_context(_history(6))
        await capability.before_model_request(_ctx(), small)

        _fixed_window(monkeypatch, 10_000_000)
        large = _request_context(_history(6))
        await capability.before_model_request(_ctx(), large)

        assert len(small.messages) < len(large.messages)

    async def test_clear_tool_results_accepts_a_fraction(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000_000)
        capability: ClearToolResults[None] = ClearToolResults(max_fraction=0.9)
        request_context = _request_context(_history(2))

        assert await capability.before_model_request(_ctx(), request_context) is request_context

    async def test_clear_tool_results_clears_when_over_the_fraction(self, monkeypatch: pytest.MonkeyPatch):
        """Asserting the negative alone cannot tell a working trigger from one that never fires."""
        _fixed_window(monkeypatch, 1_000)
        capability: ClearToolResults[None] = ClearToolResults(max_fraction=0.1, keep_pairs=1)
        request_context = _request_context(_tool_history(4))

        await capability.before_model_request(_ctx(), request_context)

        assert _tool_return_contents(request_context.messages).count(_CLEARED) == 3

    async def test_deduplicate_file_reads_gates_on_a_fraction(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000_000)
        capability: DeduplicateFileReads[None] = DeduplicateFileReads(file_key=lambda call: None, max_fraction=0.9)
        request_context = _request_context(_history(2))

        assert await capability.before_model_request(_ctx(), request_context) is request_context

    async def test_deduplicate_file_reads_drops_stale_reads_over_the_fraction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fixed_window(monkeypatch, 1_000)
        capability: DeduplicateFileReads[None] = DeduplicateFileReads(
            file_key=lambda call: str(call.args.get('path')) if isinstance(call.args, dict) else None,
            max_fraction=0.1,
        )
        request_context = _request_context(_tool_history(3, tool_name='read_file', path='a.py'))

        await capability.before_model_request(_ctx(), request_context)

        contents = _tool_return_contents(request_context.messages)
        assert contents.count(_SUPERSEDED) == 2, 'only the newest read of a.py should survive'

    def test_summarizing_compaction_accepts_a_fraction_alone(self):
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_fraction=0.9)
        assert capability.max_fraction == 0.9

    async def test_summarizing_compaction_summarizes_when_over_the_fraction(self, monkeypatch: pytest.MonkeyPatch):
        """The headline example of the PR, and the one whose trigger nothing exercised."""
        _fixed_window(monkeypatch, 1_000)
        capability: SummarizingCompaction[None] = SummarizingCompaction(
            model='test:m', max_fraction=0.1, keep_messages=2, incremental=False
        )
        request_context = _request_context(_history(6))

        # Stand in for the summarizing sub-agent, as the sibling compaction tests do.
        summary = AsyncMock()
        summary.output = 'a summary'
        with patch('pydantic_ai.Agent') as agent_class:
            agent_class.return_value.run = AsyncMock(return_value=summary)
            await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) < 12
        assert any('a summary' in _message_text(message) for message in request_context.messages)

    def test_a_fraction_alone_satisfies_the_trigger_requirement(self):
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_fraction=0.9)
        assert capability.max_tokens is None

    def test_rejects_both_triggers(self):
        with pytest.raises(ValueError, match='Set at most one of max_tokens or max_fraction'):
            SlidingWindowCompaction(max_tokens=1_000, max_fraction=0.9)


class TestThroughAnAgentRun:
    """The fraction path driven end to end, against real models and the real registry.

    The hook-level tests above pin the window, so none of them would notice a model whose id
    the registry answers differently than expected. These run the id core actually emits.
    """

    @pytest.fixture
    def anyio_backend(self) -> str:
        # These run a real `agent.run`; trio hits a TestModel event-loop quirk in core
        # unrelated to compaction.
        return 'asyncio'

    async def test_a_fraction_compacts_a_real_run(self):
        history = _history(6)
        agent = Agent(
            TestModel(),
            capabilities=[SlidingWindowCompaction(max_fraction=0.1, fallback_context_window=1_000, keep_messages=2)],
        )

        result = await agent.run('go', message_history=history)

        assert len(result.all_messages()) < len(history)

    async def test_a_generous_fraction_leaves_a_real_run_alone(self):
        history = _history(6)
        agent = Agent(
            TestModel(),
            capabilities=[
                SlidingWindowCompaction(max_fraction=0.9, fallback_context_window=10_000_000, keep_messages=2)
            ],
        )

        result = await agent.run('go', message_history=history)

        assert len(result.all_messages()) == len(history) + 2

    async def test_test_model_resolves_to_the_fallback(self):
        """`test:test` is not in the registry, so `max_fraction` is inert without a fallback."""
        seen: list[ContextUsage] = []
        agent = Agent(TestModel(), capabilities=[ReportContextUsage(on_usage=seen.append)])

        await agent.run('go')

        assert (seen[0].window_tokens, seen[0].resolved) == (DEFAULT_CONTEXT_WINDOW, False)

    async def test_a_real_fallback_model_uses_the_configured_fallback(self):
        """A real `FallbackModel` reports a composite id, so resolution fails and the fallback stands in."""
        seen: list[ContextUsage] = []
        agent = Agent(
            FallbackModel(TestModel(), TestModel()),
            capabilities=[ReportContextUsage(on_usage=seen.append, fallback_context_window=32_000)],
        )

        await agent.run('go', message_history=_history(2))

        assert (seen[0].window_tokens, seen[0].resolved) == (32_000, False)


class TestRequestModelIsTheOneResolved:
    """A capability may replace `ModelRequestContext.model`, and that is where the request goes.

    Each test gives the run's model and the request's model different windows, so a strategy
    that consults the run's model reaches the opposite conclusion from the correct one.
    """

    RUN = _FakeModel('run-model')
    REQUEST = _FakeModel('request-model')

    def _windows(self, monkeypatch: pytest.MonkeyPatch, *, run: int, request: int) -> None:
        _window_per_model(monkeypatch, {'run-model': run, 'request-model': request})

    async def test_sliding_window_trims_on_the_request_model(self, monkeypatch: pytest.MonkeyPatch):
        self._windows(monkeypatch, run=10_000_000, request=1_000)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_fraction=0.1, keep_messages=2)
        request_context = _request_context(_history(6), self.REQUEST)

        await capability.before_model_request(_ctx(self.RUN), request_context)

        assert len(request_context.messages) < 12, "the run model's window would not have triggered"

    async def test_sliding_window_stays_quiet_on_the_request_model(self, monkeypatch: pytest.MonkeyPatch):
        self._windows(monkeypatch, run=1_000, request=10_000_000)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_fraction=0.1, keep_messages=2)
        request_context = _request_context(_history(6), self.REQUEST)

        await capability.before_model_request(_ctx(self.RUN), request_context)

        assert len(request_context.messages) == 12, "the run model's window would have triggered"

    async def test_summarizing_compaction(self, monkeypatch: pytest.MonkeyPatch):
        self._windows(monkeypatch, run=1_000, request=10_000_000)
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_fraction=0.1)
        request_context = _request_context(_history(6), self.REQUEST)

        assert await capability.before_model_request(_ctx(self.RUN), request_context) is request_context

    async def test_clear_tool_results(self, monkeypatch: pytest.MonkeyPatch):
        self._windows(monkeypatch, run=1_000, request=10_000_000)
        capability: ClearToolResults[None] = ClearToolResults(max_fraction=0.1)
        request_context = _request_context(_history(6), self.REQUEST)

        assert await capability.before_model_request(_ctx(self.RUN), request_context) is request_context

    async def test_deduplicate_file_reads(self, monkeypatch: pytest.MonkeyPatch):
        self._windows(monkeypatch, run=1_000, request=10_000_000)
        capability: DeduplicateFileReads[None] = DeduplicateFileReads(file_key=lambda call: None, max_fraction=0.1)
        request_context = _request_context(_history(6), self.REQUEST)

        assert await capability.before_model_request(_ctx(self.RUN), request_context) is request_context

    async def test_tiered_compaction(self, monkeypatch: pytest.MonkeyPatch):
        self._windows(monkeypatch, run=10_000_000, request=1_000)
        capability: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
            target_fraction=0.1,
        )
        request_context = _request_context(_history(6), self.REQUEST)

        await capability.before_model_request(_ctx(self.RUN), request_context)

        assert len(request_context.messages) < 12

    async def test_a_nested_tiered_strategy_resolves_the_same_model(self, monkeypatch: pytest.MonkeyPatch):
        """A tier reached through `compact` gets the request's context, not the run's.

        The outer target is absolute, so only the nested strategy's `target_fraction` decides
        whether the inner tier runs. Resolving it against the run's model leaves the history
        alone.
        """
        self._windows(monkeypatch, run=10_000_000, request=1_000)
        capability: TieredCompaction[None] = TieredCompaction(
            tiers=[
                TieredCompaction(
                    tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
                    target_fraction=0.1,
                )
            ],
            target_tokens=1,
        )
        request_context = _request_context(_history(6), self.REQUEST)

        await capability.before_model_request(_ctx(self.RUN), request_context)

        assert len(request_context.messages) < 12

    async def test_a_summarizing_tier_calls_the_request_model(self, monkeypatch: pytest.MonkeyPatch):
        """The same rule for the model a summarizing tier inherits when it has none of its own."""
        self._windows(monkeypatch, run=10_000_000, request=1_000)
        seen: list[Any] = []

        class _Recording:
            async def compact(self, messages: list[ModelMessage], ctx: Any) -> list[ModelMessage]:
                seen.append(ctx.model)
                return messages[:1]

        capability: TieredCompaction[None] = TieredCompaction(tiers=[_Recording()], target_fraction=0.1)

        await capability.before_model_request(_ctx(self.RUN), _request_context(_history(6), self.REQUEST))

        assert seen == [self.REQUEST]

    async def test_limit_warner(self, monkeypatch: pytest.MonkeyPatch):
        self._windows(monkeypatch, run=10_000_000, request=1_000)
        capability: WarnNearLimits[None] = WarnNearLimits(max_context_fraction=0.1, warning_threshold=0.5)
        request_context = _request_context(_history(6), self.REQUEST)

        await capability.before_model_request(_ctx(self.RUN), request_context)

        assert '[WarnNearLimits]' in str(request_context.messages[-1])

    async def test_report_context_usage(self, monkeypatch: pytest.MonkeyPatch):
        self._windows(monkeypatch, run=10_000_000, request=1_000)
        seen: list[ContextUsage] = []
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append)

        await monitor.before_model_request(_ctx(self.RUN), _request_context(_history(2), self.REQUEST))

        assert seen[0].window_tokens == 1_000


_FallbackFactory = Callable[[int], object]

_FALLBACK_FACTORIES: list[tuple[str, _FallbackFactory]] = [
    ('SlidingWindowCompaction', lambda w: SlidingWindowCompaction(max_fraction=0.9, fallback_context_window=w)),
    ('SummarizingCompaction', lambda w: SummarizingCompaction(max_fraction=0.9, fallback_context_window=w)),
    ('ClearToolResults', lambda w: ClearToolResults(max_fraction=0.9, fallback_context_window=w)),
    (
        'DeduplicateFileReads',
        lambda w: DeduplicateFileReads(file_key=lambda call: None, max_fraction=0.9, fallback_context_window=w),
    ),
    (
        'TieredCompaction',
        lambda w: TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1)], target_fraction=0.9, fallback_context_window=w
        ),
    ),
    ('WarnNearLimits', lambda w: WarnNearLimits(max_context_fraction=0.9, fallback_context_window=w)),
]
"""Every capability that resolves a fraction, so the guard below cannot miss a new one."""

_OVERRIDE_FACTORIES: list[tuple[str, _FallbackFactory]] = [
    ('SlidingWindowCompaction', lambda w: SlidingWindowCompaction(max_fraction=0.9, context_window=w)),
    ('SummarizingCompaction', lambda w: SummarizingCompaction(max_fraction=0.9, context_window=w)),
    ('ClearToolResults', lambda w: ClearToolResults(max_fraction=0.9, context_window=w)),
    (
        'DeduplicateFileReads',
        lambda w: DeduplicateFileReads(file_key=lambda call: None, max_fraction=0.9, context_window=w),
    ),
    (
        'TieredCompaction',
        lambda w: TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1)], target_fraction=0.9, context_window=w
        ),
    ),
    ('WarnNearLimits', lambda w: WarnNearLimits(max_context_fraction=0.9, context_window=w)),
]
"""The same registry for the override, so a capability cannot grow a fraction without one."""


class TestStrategyFallbackWindow:
    """A window the registry cannot resolve is the caller's to supply, not just the monitor's."""

    async def test_sliding_window_uses_the_configured_fallback(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, None)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_fraction=0.9, keep_messages=2, fallback_context_window=10_000_000
        )
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) == 12, 'the 200K default would have triggered'

    async def test_tiered_compaction_uses_the_configured_fallback(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, None)
        capability: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
            target_fraction=0.9,
            fallback_context_window=10_000_000,
        )
        request_context = _request_context(_history(6))

        assert await capability.before_model_request(_ctx(), request_context) is request_context

    async def test_limit_warner_uses_the_configured_fallback(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, None)
        capability: WarnNearLimits[None] = WarnNearLimits(max_context_fraction=0.9, fallback_context_window=10_000_000)
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) == 12, 'the 200K default would have warned'

    @pytest.mark.parametrize(('name', 'factory'), _FALLBACK_FACTORIES, ids=[n for n, _ in _FALLBACK_FACTORIES])
    def test_every_strategy_rejects_a_non_positive_fallback(self, name: str, factory: _FallbackFactory):
        factory(1)
        with pytest.raises(ValueError, match='fallback_context_window must be positive'):
            factory(0)


class TestTriggerBoundary:
    """`exceeds` is strict, and the docstrings now say so."""

    async def test_a_history_exactly_at_the_trigger_does_not_compact(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        messages = _history(3)
        exact = estimate_token_count(messages)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_fraction=1.0, keep_messages=2, context_window=exact
        )
        request_context = _request_context(messages)

        assert await capability.before_model_request(_ctx(), request_context) is request_context
        assert len(request_context.messages) == 6

    async def test_one_token_over_the_trigger_compacts(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        messages = _history(3)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_fraction=1.0, keep_messages=2, context_window=estimate_token_count(messages) - 1
        )
        request_context = _request_context(messages)

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) < 6


def test_tool_availability_delta_adds_nothing_to_the_estimate():
    """The part itself carries no message text, so it adds nothing to the estimate.

    This is a statement about the part, not about the reveal being free: the schemas of the
    revealed tools do travel in the request, and this estimator counts no tool definitions at
    all, so a mid-run reveal has a context cost it cannot see. That gap is tracked separately.

    It is also the only `ModelRequestPart` without a `content` attribute, so before this
    was handled the estimator raised `AttributeError` on any history in which a deferred
    capability had loaded (#577).

    Built with no arguments deliberately: the estimator rejects the part on its type and
    never reads the names it carries, and the field holding them was renamed
    (`added` -> `tools_added`) in pydantic-ai 2.26. Naming it here would pin the test to a
    release later than the floor the runtime actually needs.
    """
    messages = _history(1)
    first = messages[0]
    assert isinstance(first, ModelRequest)
    augmented: list[ModelMessage] = [
        ModelRequest(parts=[*first.parts, ToolAvailabilityDeltaPart()]),
        *messages[1:],
    ]

    assert estimate_token_count(augmented) == estimate_token_count(messages)


class TestProgressiveToolSchemas:
    async def test_first_request_after_a_tool_reveal_counts_the_new_schema(self):
        tool = ToolDefinition(
            name='read_workspace',
            description='x' * 1_000,
            parameters_json_schema={'type': 'object', 'properties': {'path': {'type': 'string'}}},
            defer_loading=True,
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='start')]),
            ModelResponse(parts=[TextPart(content='done')], usage=RequestUsage(input_tokens=100, output_tokens=10)),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['read_workspace'])]),
        ]
        parameters = ModelRequestParameters(function_tools=[tool])
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_tokens=200, keep_messages=1, preserve_first_user_message=False
        )
        request_context = _request_context(messages, model_request_parameters=parameters)

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) == 1

    def test_a_tool_reveal_without_usage_counts_the_new_schema(self):
        tool = ToolDefinition(
            name='read_workspace',
            description='x' * 1_000,
            parameters_json_schema={'type': 'object', 'properties': {'path': {'type': 'string'}}},
            defer_loading=True,
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='start')]),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['read_workspace'])]),
        ]
        parameters = ModelRequestParameters(function_tools=[tool])

        assert estimate_context_tokens(messages, model_request_parameters=parameters) > estimate_token_count(messages)


class TestStrategyWindowOverride:
    """A window the registry resolves *wrongly* is the caller's to correct.

    `fallback_context_window` cannot: it is consulted only when resolution fails, and a
    beta-gated maximum or a self-hosted endpoint resolves to a confident wrong number.
    """

    async def test_the_override_beats_a_resolved_window(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000_000)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_fraction=0.1, keep_messages=2, context_window=1_000
        )
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) < 12, 'the resolved 1M window would not have triggered'

    async def test_the_override_beats_the_fallback_too(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, None)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_fraction=0.1, keep_messages=2, context_window=10_000_000, fallback_context_window=1_000
        )
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) == 12, 'the fallback would have triggered'

    async def test_the_override_is_ignored_without_a_fraction(self, monkeypatch: pytest.MonkeyPatch):
        """`context_window` only scales a fraction; an absolute `max_tokens` is the trigger as given."""
        _fixed_window(monkeypatch, 1_000_000)
        messages = _history(3)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_tokens=estimate_token_count(messages) + 1, keep_messages=2, context_window=10
        )
        request_context = _request_context(messages)

        assert await capability.before_model_request(_ctx(), request_context) is request_context
        assert len(request_context.messages) == 6, 'a trigger scaled by context_window=10 would have compacted'

    @pytest.mark.parametrize(('name', 'factory'), _OVERRIDE_FACTORIES, ids=[n for n, _ in _OVERRIDE_FACTORIES])
    def test_every_strategy_takes_the_override_and_rejects_a_non_positive_one(
        self, name: str, factory: _FallbackFactory
    ):
        factory(1)
        with pytest.raises(ValueError, match='context_window must be positive'):
            factory(0)


class TestTieredTargetFraction:
    async def test_escalates_against_a_resolved_target(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        capability: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
            target_fraction=0.1,
        )
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) < 12

    async def test_leaves_a_history_under_target_alone(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000_000)
        capability: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
            target_fraction=0.9,
        )
        request_context = _request_context(_history(2))

        assert await capability.before_model_request(_ctx(), request_context) is request_context

    def test_requires_one_target(self):
        with pytest.raises(ValueError, match='One of target_tokens or target_fraction must be set'):
            TieredCompaction(tiers=[SlidingWindowCompaction(max_tokens=1)])

    def test_rejects_both_targets(self):
        with pytest.raises(ValueError, match='Set at most one of target_tokens or target_fraction'):
            TieredCompaction(tiers=[SlidingWindowCompaction(max_tokens=1)], target_tokens=100, target_fraction=0.5)


class TestLimitWarnerFraction:
    async def test_warns_against_the_resolved_window(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        capability: WarnNearLimits[None] = WarnNearLimits(max_context_fraction=0.1, warning_threshold=0.5)
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        warning = request_context.messages[-1]
        assert isinstance(warning, ModelRequest)
        assert '[WarnNearLimits]' in str(warning.parts[0].content)  # type: ignore[union-attr]

    async def test_stays_quiet_below_the_threshold(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 10_000_000)
        capability: WarnNearLimits[None] = WarnNearLimits(max_context_fraction=0.9)
        request_context = _request_context(_history(2))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) == 4

    def test_a_fraction_alone_satisfies_the_limit_requirement(self):
        capability: WarnNearLimits[None] = WarnNearLimits(max_context_fraction=0.9)
        assert 'context_window' in capability._active_kinds

    def test_rejects_both_context_limits(self):
        with pytest.raises(ValueError, match='Set at most one of max_context_tokens or max_context_fraction'):
            WarnNearLimits(max_context_tokens=1_000, max_context_fraction=0.9)

    def test_warn_on_accepts_a_fraction_configured_kind(self):
        capability: WarnNearLimits[None] = WarnNearLimits(max_context_fraction=0.9, warn_on=['context_window'])
        assert capability._active_kinds == ('context_window',)

    def test_requires_at_least_one_limit(self):
        with pytest.raises(ValueError, match='At least one of max_iterations, max_context_tokens'):
            WarnNearLimits()


# ---------------------------------------------------------------------------
# Usage reporting
# ---------------------------------------------------------------------------


class TestContextUsage:
    def test_fraction_is_used_over_window(self):
        assert ContextUsage(used_tokens=250, window_tokens=1_000, resolved=True).fraction == 0.25


class TestReportContextUsage:
    async def test_reports_before_each_request(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        seen: list[ContextUsage] = []
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append)

        request_context = _request_context(_history(2))
        assert await monitor.before_model_request(_ctx(), request_context) is request_context

        assert len(seen) == 1
        assert seen[0].window_tokens == 1_000
        assert seen[0].used_tokens > 0
        assert seen[0].resolved is True

    async def test_never_edits_the_history(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=lambda _: None)
        messages = _history(3)
        request_context = _request_context(messages)

        await monitor.before_model_request(_ctx(), request_context)

        assert request_context.messages == messages

    async def test_after_a_compactor_reports_the_rewritten_anchored_history(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        seen: list[ContextUsage] = []
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='x' * 4_000)]),
            ModelResponse(parts=[TextPart(content='done')], usage=RequestUsage(input_tokens=10_000, output_tokens=0)),
            ModelRequest(parts=[UserPromptPart(content='continue')]),
        ]
        request_context = _request_context(messages)
        compactor: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_messages=2, keep_messages=2, preserve_first_user_message=False
        )
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append)
        second_monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append)
        before = estimate_token_count(messages)

        await compactor.before_model_request(_ctx(), request_context)
        await monitor.before_model_request(_ctx(), request_context)
        await second_monitor.before_model_request(_ctx(), request_context)

        expected = estimate_context_tokens(messages) - (before - estimate_token_count(request_context.messages))
        assert [usage.used_tokens for usage in seen] == [expected, expected]

    async def test_accumulates_reclaim_from_multiple_compactors(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        seen: list[ContextUsage] = []
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='x' * 4_000)]),
            ModelResponse(parts=[TextPart(content='old reply ' + 'x' * 4_000)]),
            ModelRequest(parts=[UserPromptPart(content='before anchor ' + 'x' * 4_000)]),
            ModelResponse(parts=[TextPart(content='done')], usage=RequestUsage(input_tokens=10_000, output_tokens=0)),
            ModelRequest(parts=[UserPromptPart(content='after anchor ' + 'x' * 4_000)]),
            ModelResponse(parts=[TextPart(content='new reply ' + 'x' * 4_000)]),
            ModelRequest(parts=[UserPromptPart(content='continue')]),
        ]
        request_context = _request_context(messages)
        first_compactor: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_messages=6, keep_messages=6, preserve_first_user_message=False
        )
        second_compactor: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_messages=4, keep_messages=4, preserve_first_user_message=False
        )
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append)
        before = estimate_token_count(messages)

        await first_compactor.before_model_request(_ctx(), request_context)
        await second_compactor.before_model_request(_ctx(), request_context)
        await monitor.before_model_request(_ctx(), request_context)

        expected = estimate_context_tokens(messages) - (before - estimate_token_count(request_context.messages))
        assert [usage.used_tokens for usage in seen] == [expected]

    async def test_ignores_reclaim_from_a_different_request_context(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        compacted_context = _request_context(
            [
                ModelRequest(parts=[UserPromptPart(content='x' * 4_000)]),
                ModelResponse(
                    parts=[TextPart(content='done')], usage=RequestUsage(input_tokens=10_000, output_tokens=0)
                ),
                ModelRequest(parts=[UserPromptPart(content='continue')]),
            ]
        )
        compactor: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_messages=2, keep_messages=2, preserve_first_user_message=False
        )
        seen: list[ContextUsage] = []
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append)
        unrelated_context = _request_context(
            [
                ModelRequest(parts=[UserPromptPart(content='different request')]),
                ModelResponse(parts=[TextPart(content='done')], usage=RequestUsage(input_tokens=500, output_tokens=0)),
                ModelRequest(parts=[UserPromptPart(content='continue')]),
            ]
        )

        await compactor.before_model_request(_ctx(), compacted_context)
        await monitor.before_model_request(_ctx(), unrelated_context)

        assert [usage.used_tokens for usage in seen] == [estimate_context_tokens(unrelated_context.messages)]

    async def test_after_a_compactor_reports_the_rewritten_unanchored_history(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        seen: list[ContextUsage] = []
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='x' * 4_000)]),
            ModelResponse(parts=[TextPart(content='done')]),
            ModelRequest(parts=[UserPromptPart(content='continue')]),
        ]
        request_context = _request_context(messages)
        compactor: SlidingWindowCompaction[None] = SlidingWindowCompaction(
            max_messages=2, keep_messages=2, preserve_first_user_message=False
        )
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append)

        await compactor.before_model_request(_ctx(), request_context)
        await monitor.before_model_request(_ctx(), request_context)

        assert seen[0].used_tokens == estimate_context_tokens(request_context.messages)

    async def test_an_async_callback_is_awaited(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, 1_000)
        seen: list[ContextUsage] = []

        async def record(usage: ContextUsage) -> None:
            seen.append(usage)

        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=record)

        await monitor.before_model_request(_ctx(), _request_context(_history(2)))

        assert len(seen) == 1

    async def test_override_skips_resolution(self, monkeypatch: pytest.MonkeyPatch):
        def _explode(_model_ref: str) -> int | None:  # pragma: no cover -- must not be called
            raise AssertionError('an explicit window must not consult the registry')

        _patch_snapshot(monkeypatch, _explode)
        seen: list[ContextUsage] = []
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append, context_window=4_242)

        await monitor.before_model_request(_ctx(), _request_context(_history(2)))

        assert (seen[0].window_tokens, seen[0].resolved) == (4_242, True)

    async def test_falls_back_for_an_unknown_model(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, None)
        seen: list[ContextUsage] = []
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append)

        await monitor.before_model_request(_ctx(), _request_context(_history(2)))

        assert (seen[0].window_tokens, seen[0].resolved) == (DEFAULT_CONTEXT_WINDOW, False)

    async def test_custom_fallback(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, None)
        seen: list[ContextUsage] = []
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append, fallback_context_window=32_000)

        await monitor.before_model_request(_ctx(), _request_context(_history(2)))

        assert seen[0].window_tokens == 32_000

    async def test_each_request_resolves_independently(self, monkeypatch: pytest.MonkeyPatch):
        seen: list[ContextUsage] = []
        monitor: ReportContextUsage[None] = ReportContextUsage(on_usage=seen.append)

        _fixed_window(monkeypatch, 1_000)
        await monitor.before_model_request(_ctx(), _request_context(_history(2)))
        _fixed_window(monkeypatch, 2_000)
        await monitor.before_model_request(_ctx(), _request_context(_history(2)))

        assert [reading.window_tokens for reading in seen] == [1_000, 2_000]

    def test_rejects_a_non_positive_window(self):
        with pytest.raises(ValueError, match='context_window must be positive'):
            ReportContextUsage(on_usage=lambda _: None, context_window=0)

    def test_rejects_a_non_positive_fallback(self):
        with pytest.raises(ValueError, match='fallback_context_window must be positive'):
            ReportContextUsage(on_usage=lambda _: None, fallback_context_window=0)


# ---------------------------------------------------------------------------
# Out-of-run compaction
# ---------------------------------------------------------------------------


class _RecordingStrategy:
    """Captures what it was handed, so the throwaway context can be inspected."""

    def __init__(self) -> None:
        self.seen_model: Any = None
        self.seen_deps: Any = None
        self.seen_usage: RunUsage | None = None

    async def compact(self, messages: list[ModelMessage], ctx: Any) -> list[ModelMessage]:
        self.seen_model = ctx.model
        self.seen_deps = ctx.deps
        self.seen_usage = ctx.usage
        return messages[:1]


class TestCompactNow:
    async def test_runs_without_an_agent_run(self):
        strategy = _RecordingStrategy()
        messages = _history(3)

        result = await compact_now(strategy, messages, model=TestModel())  # type: ignore[arg-type]

        assert len(result) == 1
        assert isinstance(strategy.seen_model, TestModel)

    async def test_resolves_a_model_name(self):
        strategy = _RecordingStrategy()

        await compact_now(strategy, _history(1), model='test')  # type: ignore[arg-type]

        assert strategy.seen_model is not None

    async def test_passes_deps_and_usage_through(self):
        strategy = _RecordingStrategy()
        usage = RunUsage(requests=3)

        await compact_now(strategy, _history(1), model=TestModel(), deps='my-deps', usage=usage)  # type: ignore[arg-type]

        assert strategy.seen_deps == 'my-deps'
        assert strategy.seen_usage is usage

    async def test_focus_is_ignored_by_a_strategy_that_cannot_honour_it(self):
        strategy = _RecordingStrategy()

        result = await compact_now(strategy, _history(2), model=TestModel(), focus='auth')  # type: ignore[arg-type]

        assert len(result) == 1

    async def test_focus_steers_a_summarizing_strategy(self):
        seen: list[str] = []

        class _Focusable(_RecordingStrategy):
            def with_focus(self, focus: str) -> _Focusable:
                seen.append(focus)
                return self

        await compact_now(_Focusable(), _history(1), model=TestModel(), focus='the auth refactor')  # type: ignore[arg-type]

        assert seen == ['the auth refactor']


class TestCompactNowSummarizes:
    """`compact_now` driving a strategy that really calls a model."""

    @pytest.fixture
    def anyio_backend(self) -> str:
        # These run a real `agent.run`; trio hits a TestModel event-loop quirk in core
        # unrelated to compaction.
        return 'asyncio'

    async def test_a_summary_call_lands_on_the_supplied_usage(self):
        """The documented reason `usage` exists: an out-of-run summary stays on someone's counter."""
        usage = RunUsage()
        strategy: SummarizingCompaction[None] = SummarizingCompaction(max_messages=1, keep_messages=2)

        await compact_now(strategy, _history(4), model=TestModel(), usage=usage)

        assert usage.requests == 1
        assert usage.total_tokens > 0

    async def test_the_history_is_replaced_by_a_summary(self):
        strategy: SummarizingCompaction[None] = SummarizingCompaction(max_messages=1, keep_messages=2)

        result = await compact_now(strategy, _history(4), model=TestModel())

        assert len(result) < 8


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
class TestCompactNowSpan:
    """An out-of-run compaction emits the same span the in-run path does."""

    @pytest.fixture
    def anyio_backend(self) -> str:
        return 'asyncio'

    def _spans(self, capfire: CaptureLogfire) -> list[dict[str, Any]]:
        return [s for s in capfire.exporter.exported_spans_as_dict() if s['name'] == 'compact_messages']

    async def test_emits_the_compaction_span(self, capfire: CaptureLogfire):
        from opentelemetry.trace import get_tracer

        strategy: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_tokens=1, keep_messages=2)

        await compact_now(strategy, _history(4), model=TestModel(), tracer=get_tracer('test'))

        spans = self._spans(capfire)
        assert len(spans) == 1
        attrs = spans[0]['attributes']
        assert attrs['gen_ai.conversation.compacted'] is True
        assert attrs['compaction.strategy'] == 'SlidingWindowCompaction'
        assert attrs['compaction.messages_before'] > attrs['compaction.messages_after']

    async def test_the_span_is_measured_with_the_tokenizer_it_was_given(self, capfire: CaptureLogfire):
        """Without it the manual path reports the heuristic where the in-run path reported a real count."""
        from opentelemetry.trace import get_tracer

        strategy: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_tokens=1, keep_messages=2)
        messages = _history(4)
        characters = sum(len(_message_text(message)) for message in messages)

        await compact_now(strategy, messages, model=TestModel(), tracer=get_tracer('test'), tokenizer=len)

        attributes = self._spans(capfire)[0]['attributes']
        assert attributes['compaction.tokens_before'] == characters, 'the 4-characters heuristic was used instead'

    async def test_no_span_when_the_history_is_unchanged(self, capfire: CaptureLogfire):
        from opentelemetry.trace import get_tracer

        tiered: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
            target_tokens=1_000_000,
        )

        await compact_now(tiered, _history(3, filler='x' * 40), model=TestModel(), tracer=get_tracer('test'))

        assert self._spans(capfire) == []

    async def test_the_span_names_the_focused_strategy(self, capfire: CaptureLogfire):
        from opentelemetry.trace import get_tracer

        tiered: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
            target_tokens=1,
        )

        await compact_now(tiered, _history(6), model=TestModel(), focus='auth', tracer=get_tracer('test'))

        assert self._spans(capfire)[0]['attributes']['compaction.strategy'] == 'TieredCompaction'

    async def test_a_default_tracer_records_nothing(self, capfire: CaptureLogfire):
        strategy: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_tokens=1, keep_messages=2)

        await compact_now(strategy, _history(4), model=TestModel())

        assert self._spans(capfire) == []


class TestWithFocus:
    def test_appends_the_focus_to_the_prompt(self):
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_fraction=0.9)
        focused = capability.with_focus('the auth refactor')
        assert 'the auth refactor' in focused.summary_prompt

    def test_leaves_the_original_untouched(self):
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_fraction=0.9)
        original = capability.summary_prompt
        capability.with_focus('anything')
        assert capability.summary_prompt == original

    def test_braces_survive_prompt_formatting(self):
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_fraction=0.9)
        focused = capability.with_focus('the {config} dict')
        assert 'the {config} dict' in focused.summary_prompt.format(messages='...')


class TestNewExports:
    def test_capability_exposed_at_top_level(self):
        import pydantic_ai_harness
        import pydantic_ai_harness.compaction as compaction

        assert pydantic_ai_harness.ReportContextUsage is compaction.ReportContextUsage


class TestPositionalCompatibility:
    """The new fields are keyword-only, so they cannot shift an existing positional argument.

    Every strategy here is a plain dataclass whose own fields are positional, so inserting a
    field next to a related one would rebind every argument after it in existing call sites.
    """

    def test_sliding_window(self):
        assert SlidingWindowCompaction(None, 1_000, 40).keep_messages == 40

    def test_summarizing_compaction(self):
        assert SummarizingCompaction('openai:gpt-4o', None, 1_000, 15).keep_messages == 15

    def test_clear_tool_results(self):
        assert ClearToolResults(None, 1_000, 5).keep_pairs == 5

    def test_deduplicate_file_reads(self):
        strategy = DeduplicateFileReads(lambda call: None, '[superseded]', None, 1_000)
        assert strategy.max_tokens == 1_000

    def test_tiered_compaction(self):
        strategy = TieredCompaction([SlidingWindowCompaction(max_tokens=1)], 100, len)
        assert strategy.tokenizer is len

    def test_limit_warner(self):
        assert WarnNearLimits(10, 1_000, 2_000).max_total_tokens == 2_000

    def test_the_new_fields_stay_keyword_only(self):
        import dataclasses

        cases = [
            (SlidingWindowCompaction, 'max_fraction'),
            (SummarizingCompaction, 'max_fraction'),
            (ClearToolResults, 'max_fraction'),
            (DeduplicateFileReads, 'max_fraction'),
            (TieredCompaction, 'target_fraction'),
            (WarnNearLimits, 'max_context_fraction'),
            (SlidingWindowCompaction, 'fallback_context_window'),
            (SummarizingCompaction, 'fallback_context_window'),
            (ClearToolResults, 'fallback_context_window'),
            (DeduplicateFileReads, 'fallback_context_window'),
            (TieredCompaction, 'fallback_context_window'),
            (WarnNearLimits, 'fallback_context_window'),
        ]
        for cls, name in cases:
            field = next(f for f in dataclasses.fields(cls) if f.name == name)
            assert field.kw_only, f'{cls.__name__}.{name} must stay keyword-only'


class TestFocusPropagation:
    """A composing strategy has to pass the focus down to the tier that writes the summary."""

    def _tiered(self) -> TieredCompaction[None]:
        return TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2), SummarizingCompaction(max_messages=1)],
            target_tokens=100,
        )

    def test_tiered_is_focusable(self):
        assert isinstance(self._tiered(), SupportsFocus)

    def test_focus_reaches_the_summarizing_tier(self):
        focused = self._tiered().with_focus('the auth flow')
        summarizer = focused.tiers[1]
        assert isinstance(summarizer, SummarizingCompaction)
        assert 'the auth flow' in summarizer.summary_prompt

    def test_tiers_that_cannot_be_focused_pass_through(self):
        tiered = self._tiered()
        assert tiered.with_focus('anything').tiers[0] is tiered.tiers[0]

    def test_the_original_is_left_alone(self):
        tiered = self._tiered()
        tiered.with_focus('the auth flow')
        summarizer = tiered.tiers[1]
        assert isinstance(summarizer, SummarizingCompaction)
        assert 'the auth flow' not in summarizer.summary_prompt

    def test_nesting_propagates_all_the_way_down(self):
        outer = TieredCompaction(tiers=[self._tiered()], target_tokens=100)
        focused = outer.with_focus('nested topic')
        inner = focused.tiers[0]
        assert isinstance(inner, TieredCompaction)
        summarizer = inner.tiers[1]
        assert isinstance(summarizer, SummarizingCompaction)
        assert 'nested topic' in summarizer.summary_prompt

    async def test_compact_now_focuses_through_a_tiered_strategy(self):
        seen: list[str] = []

        class _Tier:
            def with_focus(self, focus: str) -> _Tier:
                seen.append(focus)
                return self

            async def compact(self, messages: list[ModelMessage], ctx: Any) -> list[ModelMessage]:
                return messages[:1]

        tiered = TieredCompaction(tiers=[_Tier()], target_tokens=1)
        await compact_now(tiered, _history(4), model=TestModel(), focus='the auth flow')

        assert seen == ['the auth flow']


class TestManualCompactionSemantics:
    """`compact_now` adds no trigger; a strategy's own stop condition still applies."""

    def _short_history(self) -> list[ModelMessage]:
        return _history(3, filler='x' * 40)

    async def test_an_unconditional_strategy_runs_whatever_the_size(self):
        history = self._short_history()
        strategy: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_tokens=1_000_000, keep_messages=2)

        result = await compact_now(strategy, history, model=TestModel())

        assert len(result) < len(history), 'no trigger of its own means it runs regardless'

    async def test_a_tiered_strategy_honours_its_own_target(self):
        """Already under target is not a missed compaction: there is nothing left to reclaim."""
        history = self._short_history()
        tiered: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
            target_tokens=1_000_000,
        )

        result = await compact_now(tiered, history, model=TestModel())

        assert result == history

    async def test_a_tiered_strategy_over_target_still_escalates(self):
        tiered: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
            target_tokens=1,
        )

        result = await compact_now(tiered, _history(6), model=TestModel())

        assert len(result) < 12


# ---------------------------------------------------------------------------
# Realtime models (#585)
# ---------------------------------------------------------------------------


class TestSpeechPartCounting:
    """A `SpeechPart` transcript is text the provider bills, so the estimator counts it."""

    def test_a_speech_transcript_counts_like_equivalent_text(self):
        transcript = 'tell me about the weather today ' * 8
        spoken: list[ModelMessage] = [
            ModelRequest(parts=[SpeechPart(speaker='user', transcript=transcript)]),
            ModelResponse(parts=[SpeechPart(speaker='assistant', transcript=transcript)]),
        ]
        as_text: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=transcript)]),
            ModelResponse(parts=[TextPart(content=transcript)]),
        ]
        assert estimate_token_count(spoken) == estimate_token_count(as_text) > 0

    def test_audio_only_speech_contributes_no_text(self):
        """`transcript` is optional; an audio-only turn has no characters to count."""
        silent: list[ModelMessage] = [
            ModelRequest(parts=[SpeechPart(speaker='user')]),
            ModelResponse(parts=[SpeechPart(speaker='assistant')]),
        ]
        assert estimate_token_count(silent) == 0


class TestRealtimeModelSkipsTokenTriggers:
    """A realtime session never compacts (its history can't change mid-run), so this is a
    type-level guard: `resolve_token_trigger` returns `None` for a realtime model however it is
    configured."""

    def test_a_realtime_model_resolves_to_no_trigger(self):
        model = _FakeRealtimeModel()
        # A genuine `AbstractModel` identity, not a mock: it is skipped for being a non-`Model`.
        assert model.model_id == 'openai:gpt-4o-realtime-preview'
        assert resolve_token_trigger(None, 0.5, model) is None  # fraction
        assert resolve_token_trigger(1_000, None, model) is None  # absolute budget

    def test_a_request_response_model_still_resolves(self):
        assert resolve_token_trigger(1_000, None, TestModel()) == 1_000

    async def test_sliding_window_leaves_a_realtime_request_untouched(self, monkeypatch: pytest.MonkeyPatch):
        """A fraction that trims a normal run does not fire when the request model is realtime."""
        _fixed_window(monkeypatch, 1_000)
        capability: SlidingWindowCompaction[None] = SlidingWindowCompaction(max_fraction=0.1, keep_messages=2)
        request_context = ModelRequestContext(
            model=_FakeRealtimeModel(),  # type: ignore[arg-type]
            messages=_history(6),
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) == 12, 'no token trigger means nothing is trimmed'

    async def test_tiered_compaction_does_not_compact_a_realtime_run(self):
        tiered: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindowCompaction(max_tokens=1, keep_messages=2)],
            target_fraction=0.1,
        )
        history = _history(6)

        result = await compact_now(tiered, history, model=_FakeRealtimeModel())  # type: ignore[arg-type]

        assert result == history, 'no token target means the tiers never escalate'

    async def test_summarizing_compaction_requires_a_model_for_a_realtime_run(self):
        """With no summarizer `model=` configured, a realtime run cannot summarize -- say so."""
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_messages=1)
        ctx = _ctx(model=_FakeRealtimeModel())

        with pytest.raises(UserError, match='needs a request-response model'):
            await capability._summarize(_history(2), ctx)  # pyright: ignore[reportPrivateUsage]
