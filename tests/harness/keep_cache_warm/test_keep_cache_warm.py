"""Tests for `KeepCacheWarm`.

The capability is driven through `Agent(..., capabilities=[...])` with a `FunctionModel` whose profile publishes a
two-second cache retention on Anthropic's tiers. With a 1.8-second margin, a refresh is due 0.2 seconds after a
request and dropped after 1.1 seconds, so the tests wait for the refresh task (or an event it sets) under a generous
hang guard instead of asserting on timing. A refresh is the request the model receives with `max_tokens=1`.
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import CacheRetention
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness import KeepCacheWarm as RootKeepCacheWarm
from pydantic_ai_harness.keep_cache_warm import CacheWarmingStoppedWarning, KeepCacheWarm
from tests.harness._recording_durability import RecordingDurability

HANG_GUARD = 10
"""Seconds to wait for a refresh before failing: a hang guard, not a timing assertion."""

ANTHROPIC_TIERS: tuple[CacheRetention, ...] = ('5m', '1h')


def _profile(retention: timedelta | None, *, tiers: tuple[CacheRetention, ...] = ANTHROPIC_TIERS) -> ModelProfile:
    return ModelProfile(supports_cache=True, supported_cache_retentions=tiers, default_cache_retention=retention)


TWO_SECONDS = _profile(timedelta(seconds=2))
FIVE_MINUTES = _profile(timedelta(minutes=5))


class CachingModel:
    """A `FunctionModel` function that reports a cache write per request, and records the refreshes it receives."""

    def __init__(
        self, *, refresh_read: int = 100, refresh_output: int = 1, refresh_error: Exception | None = None
    ) -> None:
        self.refresh_read = refresh_read
        self.refresh_output = refresh_output
        self.refresh_error = refresh_error
        self.refreshes: list[list[ModelMessage]] = []
        self.refreshed = asyncio.Event()

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if (info.model_settings or {}).get('max_tokens') == 1:
            self.refreshes.append(messages)
            self.refreshed.set()
            if self.refresh_error is not None:
                raise self.refresh_error
            usage = RequestUsage(
                input_tokens=10, output_tokens=self.refresh_output, cache_read_tokens=self.refresh_read
            )
            return ModelResponse(parts=[TextPart('.')], usage=usage)
        return ModelResponse(parts=[TextPart('done')], usage=RequestUsage(input_tokens=10, cache_write_tokens=100))


def _warmers() -> set[asyncio.Task[object]]:
    return {task for task in asyncio.all_tasks() if task.get_name().startswith('keep cache warm')}


def _agent(
    model: CachingModel, warm: KeepCacheWarm[None], *, profile: ModelProfile | None = TWO_SECONDS
) -> Agent[None, str]:
    return Agent(FunctionModel(model, profile=profile), deps_type=type(None), capabilities=[warm])


def _fast(*, max_idle: timedelta = timedelta(minutes=10), between_runs: bool = True) -> KeepCacheWarm[None]:
    return KeepCacheWarm(margin=timedelta(seconds=1.8), max_idle=max_idle, between_runs=between_runs)


class TestKeepCacheWarm:
    async def test_replays_the_last_request_without_changing_history(self) -> None:
        model = CachingModel()
        warm = _fast(max_idle=timedelta(seconds=2.6))
        result = await _agent(model, warm).run('hello')
        (warmer,) = _warmers()

        await asyncio.wait_for(warmer, HANG_GUARD)

        assert model.refreshes[0] == result.all_messages()[:-1]
        assert len(result.all_messages()) == 2
        assert warm.usage.requests == len(model.refreshes)
        assert warm.usage.cache_read_tokens == 100 * len(model.refreshes)
        assert result.usage.requests == 1
        assert not _warmers()

    async def test_next_request_of_the_conversation_cancels_the_pending_refresh(self) -> None:
        model = CachingModel()
        warm = KeepCacheWarm[None](between_runs=True)
        agent = _agent(model, warm, profile=FIVE_MINUTES)

        first = await agent.run('hello')
        (pending,) = _warmers()
        await agent.run('again', message_history=first.all_messages())
        await asyncio.sleep(0)

        assert pending.cancelled()
        (replacement,) = _warmers()
        assert replacement is not pending
        await warm.stop()
        assert not _warmers()
        assert model.refreshes == []

    async def test_each_conversation_has_its_own_refresh(self) -> None:
        warm = KeepCacheWarm[None](between_runs=True)
        agent = _agent(CachingModel(), warm, profile=FIVE_MINUTES)

        await agent.run('one')
        await agent.run('two')

        assert len(_warmers()) == 2
        await warm.stop()
        assert not _warmers()

    @pytest.mark.parametrize(
        'profile',
        [
            pytest.param(
                ModelProfile(supported_cache_retentions=ANTHROPIC_TIERS, default_cache_retention=timedelta(seconds=2)),
                id='caching-not-configurable',
            ),
            # OpenAI's GPT-5.6 keeps a cached prefix for *at least* 30 minutes, so a refresh buys nothing certain.
            pytest.param(_profile(timedelta(seconds=2), tiers=('30m',)), id='minimum-retention'),
            pytest.param(_profile(None), id='retention-unknown'),
            pytest.param(_profile(timedelta(hours=1)), id='outlives-max-idle'),
            pytest.param(_profile(timedelta(seconds=1)), id='within-margin'),
        ],
    )
    async def test_not_warmed_when_the_model_cache_cant_be_kept_warm(self, profile: ModelProfile) -> None:
        await _agent(CachingModel(), _fast(max_idle=timedelta(minutes=30)), profile=profile).run('hello')

        assert not _warmers()

    async def test_not_warmed_when_nothing_was_cached(self) -> None:
        def uncached(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('done')], usage=RequestUsage(input_tokens=10))

        await Agent(FunctionModel(uncached, profile=TWO_SECONDS), deps_type=type(None), capabilities=[_fast()]).run(
            'hello'
        )

        assert not _warmers()

    async def test_not_warmed_behind_a_fallback_model(self) -> None:
        model = FallbackModel(FunctionModel(CachingModel(), profile=TWO_SECONDS))
        await Agent(model, deps_type=type(None), capabilities=[_fast()]).run('hello')

        assert not _warmers()

    async def test_not_warmed_while_a_native_tool_call_awaits_its_result(self) -> None:
        # The provider runs an unpaired native call when it gets the history, so each replay would run it again.
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Look it up, then add 2 and 2.')]),
            ModelResponse(
                parts=[
                    NativeToolCallPart(tool_name='mcp_server:wiki', args={}, tool_call_id='native-1'),
                    ToolCallPart(tool_name='add', args={'a': 2, 'b': 2}, tool_call_id='local-1'),
                ]
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content='4', tool_call_id='local-1')]),
        ]
        await _agent(CachingModel(), _fast()).run(message_history=history)

        assert not _warmers()

    async def test_warmed_when_every_native_tool_call_has_its_result(self) -> None:
        warm = _fast()
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Look it up.')]),
            ModelResponse(
                parts=[
                    NativeToolCallPart(tool_name='mcp_server:wiki', args={}, tool_call_id='native-1'),
                    NativeToolReturnPart(tool_name='mcp_server:wiki', content='found', tool_call_id='native-1'),
                    TextPart('found it'),
                ]
            ),
        ]
        await _agent(CachingModel(), warm).run('thanks', message_history=history)

        assert len(_warmers()) == 1
        await warm.stop()

    async def test_rejects_durable_execution(self) -> None:
        agent = Agent(
            FunctionModel(CachingModel(), profile=TWO_SECONDS),
            deps_type=type(None),
            name='durable',
            capabilities=[_fast(), RecordingDurability()],
        )

        with pytest.raises(UserError, match='cannot be used inside a durable workflow'):
            await agent.run('hello')

        assert not _warmers()

    async def test_failed_request_starts_no_refresh(self) -> None:
        def failing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ModelAPIError('function', 'overloaded')

        agent = Agent(FunctionModel(failing, profile=TWO_SECONDS), deps_type=type(None), capabilities=[_fast()])
        with pytest.raises(ModelAPIError):
            await agent.run('hello')

        assert not _warmers()

    async def test_failed_refresh_warns_and_stops(self) -> None:
        model = CachingModel(refresh_error=ModelAPIError('function', 'max_tokens too small'))
        await _agent(model, _fast()).run('hello')
        (warmer,) = _warmers()

        with pytest.warns(CacheWarmingStoppedWarning, match='the refresh request failed with'):
            await asyncio.wait_for(warmer, HANG_GUARD)

        assert len(model.refreshes) == 1

    async def test_refresh_that_rewrites_the_cache_warns_and_stops(self) -> None:
        model = CachingModel(refresh_read=50)
        warm = _fast()
        await _agent(model, warm).run('hello')
        (warmer,) = _warmers()

        with pytest.warns(CacheWarmingStoppedWarning, match='read back 50 of the 100 cached tokens'):
            await asyncio.wait_for(warmer, HANG_GUARD)

        assert len(model.refreshes) == 1
        assert warm.usage.requests == 1

    async def test_refresh_whose_max_tokens_was_ignored_warns_and_stops(self) -> None:
        model = CachingModel(refresh_output=500)
        await _agent(model, _fast()).run('hello')
        (warmer,) = _warmers()

        with pytest.warns(CacheWarmingStoppedWarning, match=r'produced 500 output tokens'):
            await asyncio.wait_for(warmer, HANG_GUARD)

        assert len(model.refreshes) == 1

    async def test_warms_during_a_long_tool_call(self) -> None:
        model = CachingModel()

        def call_tool_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            response = model(messages, info)
            if len(messages) == 1 and not model.refreshes:
                response.parts = [ToolCallPart('wait_for_refresh')]
            return response

        agent = Agent(
            FunctionModel(call_tool_then_answer, profile=TWO_SECONDS),
            deps_type=type(None),
            capabilities=[KeepCacheWarm[None](margin=timedelta(seconds=1.8))],
        )

        @agent.tool_plain
        async def wait_for_refresh() -> str:
            await asyncio.wait_for(model.refreshed.wait(), HANG_GUARD)
            return 'done'

        result = await agent.run('hello')

        assert model.refreshes[0] == result.all_messages()[:1]
        assert not _warmers()

    async def test_stops_when_the_run_ends_by_default(self) -> None:
        await _agent(CachingModel(), KeepCacheWarm[None](), profile=FIVE_MINUTES).run('hello')

        assert not _warmers()

    async def test_late_refresh_is_dropped(self) -> None:
        model = CachingModel()
        await _agent(model, _fast()).run('hello')
        (warmer,) = _warmers()

        # Block the event loop past the refresh's due time plus half the margin.
        time.sleep(1.2)
        await asyncio.wait_for(warmer, HANG_GUARD)

        assert model.refreshes == []

    async def test_retention_is_timed_from_the_start_of_the_request(self) -> None:
        model = CachingModel()

        async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Outlast the refresh's due time plus half the margin, measured from when the request started.
            await asyncio.sleep(1.2)
            return model(messages, info)

        await Agent(FunctionModel(slow, profile=TWO_SECONDS), deps_type=type(None), capabilities=[_fast()]).run('hello')
        # The refresh is already overdue when the response arrives, so its task may have finished by now.
        await asyncio.wait_for(asyncio.gather(*_warmers()), HANG_GUARD)

        assert model.refreshes == []

    async def test_refreshes_until_the_cache_outlives_max_idle(self) -> None:
        model = CachingModel()
        await _agent(model, _fast(max_idle=timedelta(seconds=2.6))).run('hello')
        (warmer,) = _warmers()

        await asyncio.wait_for(warmer, HANG_GUARD)

        assert model.refreshes
        assert all(refresh == model.refreshes[0] for refresh in model.refreshes)

    async def test_refresh_span_is_a_root_span(self) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        model = CachingModel()
        warm = _fast(max_idle=timedelta(seconds=2.6))
        instrumentation = Instrumentation(settings=InstrumentationSettings(tracer_provider=provider))
        agent = Agent(
            FunctionModel(model, profile=TWO_SECONDS), deps_type=type(None), capabilities=[warm, instrumentation]
        )

        await agent.run('hello')
        (warmer,) = _warmers()
        await asyncio.wait_for(warmer, HANG_GUARD)

        refresh_spans = [span for span in exporter.get_finished_spans() if span.name == 'keep_cache_warm']
        assert refresh_spans
        span = refresh_spans[0]
        assert span.parent is None
        assert span.attributes == {
            'keep_cache_warm.cache_read_tokens': 100,
            'keep_cache_warm.cache_write_tokens': 0,
            'keep_cache_warm.retention_seconds': 2.0,
        }

    def test_rejects_non_positive_durations(self) -> None:
        with pytest.raises(ValueError, match='max_idle must be positive'):
            KeepCacheWarm[None](max_idle=timedelta(0))
        with pytest.raises(ValueError, match='margin must be positive'):
            KeepCacheWarm[None](margin=timedelta(0))

    def test_exported_from_the_package_root(self) -> None:
        assert RootKeepCacheWarm is KeepCacheWarm
