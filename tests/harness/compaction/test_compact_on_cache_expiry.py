"""Tests for `CompactOnCacheExpiry`.

The capability's provider-specific behavior is the retention window it compares the idle gap
against: the provider's documented default, or the retention the request's settings ask for. These
tests use the real model classes, so that resolution runs exactly as in production, and replace
only the network call (`_Local`). A recorded test can't show the decision either way: whether the
cache is cold depends on the wall-clock gap between two requests, which a replayed cassette doesn't
have, so the tests set the gap by moving the stored history's timestamps back, as it looks to an
application that loads a conversation the user left idle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Caching, Hooks
from pydantic_ai.messages import (
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    CompactOnCacheExpiry,
    SlidingWindowCompaction,
    TieredCompaction,
    compact_now,
    estimate_token_count,
    is_pinned,
    pin,
)
from tests.conftest import try_import

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as bedrock_imports:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as openrouter_imports:
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

try:
    from logfire.testing import CaptureLogfire

    logfire_installed = True
except ImportError:  # pragma: no cover
    logfire_installed = False


TOOL_STEPS = 8
"""Tool calls in the stored conversation: more than the default strategy's 5 kept results."""


class _Local(WrapperModel):
    """A real provider model whose requests are answered locally.

    Profile and `resolve_cache_retention` forward to the wrapped model, so the capability resolves
    the retention window exactly as it would against the provider.
    """

    def __init__(self, wrapped: Model, tool_steps: int = 0, report_usage: bool = False) -> None:
        super().__init__(wrapped)
        self.tool_steps = tool_steps
        self.report_usage = report_usage
        self.sent: list[list[ModelMessage]] = []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.sent.append(list(messages))
        # Reported like a provider would: about what the history weighs, which size-based strategies anchor on.
        usage = RequestUsage(input_tokens=estimate_token_count(messages)) if self.report_usage else RequestUsage()
        step = sum(isinstance(message, ModelResponse) for message in messages[_last_user_prompt(messages) :])
        if step < self.tool_steps:
            part = ToolCallPart('lookup', {'n': step}, tool_call_id=f'call_{len(self.sent)}')
            return ModelResponse(parts=[part], model_name=self.model_name, usage=usage)
        return ModelResponse(parts=[TextPart('done')], model_name=self.model_name, usage=usage)


def _last_user_prompt(messages: Sequence[ModelMessage]) -> int:
    return max(
        i
        for i, message in enumerate(messages)
        if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)
    )


def _agent(model: Model, *capabilities: AbstractCapability[None], **kwargs: Any) -> Agent[None, str]:
    agent: Agent[None, str] = Agent(model, capabilities=list(capabilities), **kwargs)

    @agent.tool_plain
    def lookup(n: int) -> str:
        return f'result {n}: ' + 'x' * 400

    return agent


async def _conversation(model: Model, report_usage: bool = False, **kwargs: Any) -> list[ModelMessage]:
    """A stored conversation with `TOOL_STEPS` tool results, made by running the agent without the capability."""
    result = await _agent(_Local(model, tool_steps=TOOL_STEPS, report_usage=report_usage), **kwargs).run('investigate')
    return result.all_messages()


def _idle(messages: list[ModelMessage], gap: timedelta) -> list[ModelMessage]:
    """The stored conversation as it looks after the user left it idle for `gap`."""
    return [
        replace(message, timestamp=message.timestamp - gap) if message.timestamp is not None else message
        for message in messages
    ]


def _cleared(messages: Sequence[ModelMessage]) -> list[str]:
    """Tool call ids whose result has been cleared."""
    return [
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.content == '[tool result cleared]'
    ]


def _anthropic() -> Model:
    return AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key='mock-api-key'))


def _openai_responses() -> Model:
    return OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(api_key='mock-api-key'))


def _openai_chat_before_breakpoints() -> Model:
    return OpenAIChatModel('gpt-4.1', provider=OpenAIProvider(api_key='mock-api-key'))


def _bedrock(model_name: str) -> Model:
    provider = BedrockProvider(region_name='us-east-1', aws_access_key_id='x', aws_secret_access_key='x')
    return BedrockConverseModel(model_name, provider=provider)


def _openrouter_anthropic() -> Model:
    return OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='mock-api-key'))


def _openrouter_gemini() -> Model:
    return OpenRouterModel('google/gemini-2.5-flash', provider=OpenRouterProvider(api_key='mock-api-key'))


def _google() -> Model:
    return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='mock-api-key'))


@dataclass(frozen=True)
class Case:
    id: str
    model: str
    retention: timedelta
    settings: Callable[[], ModelSettings] | None = None
    caching: Caching | None = None
    marks: tuple[pytest.MarkDecorator, ...] = ()


_needs_anthropic = (pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),)
_needs_openai = (pytest.mark.skipif(not openai_imports(), reason='openai not installed'),)
_needs_bedrock = (pytest.mark.skipif(not bedrock_imports(), reason='boto3 not installed'),)
_needs_openrouter = (pytest.mark.skipif(not openrouter_imports(), reason='openai not installed'),)
_needs_google = (pytest.mark.skipif(not google_imports(), reason='google-genai not installed'),)

_MODELS = {
    'anthropic': _anthropic,
    'openai-responses': _openai_responses,
    'openai-chat-before-breakpoints': _openai_chat_before_breakpoints,
    'bedrock-claude': lambda: _bedrock('us.anthropic.claude-sonnet-4-6'),
    'openrouter-anthropic': _openrouter_anthropic,
    'google': _google,
    'bedrock-gpt-5.6': lambda: _bedrock('openai.gpt-5.6-sol'),
    'openrouter-gemini': _openrouter_gemini,
}

KNOWN_RETENTION = [
    Case('anthropic-default', 'anthropic', timedelta(minutes=5), marks=_needs_anthropic),
    Case(
        'anthropic-caching-1h',
        'anthropic',
        timedelta(hours=1),
        caching=Caching(retention='1h'),
        marks=_needs_anthropic,
    ),
    Case(
        'anthropic-cache-setting-1h',
        'anthropic',
        timedelta(hours=1),
        settings=lambda: AnthropicModelSettings(anthropic_cache='1h'),
        marks=_needs_anthropic,
    ),
    Case('openai-gpt-5.6', 'openai-responses', timedelta(minutes=30), marks=_needs_openai),
    Case(
        'openai-24h-retention-setting',
        'openai-chat-before-breakpoints',
        timedelta(hours=24),
        settings=lambda: OpenAIChatModelSettings(openai_prompt_cache_retention='24h'),
        marks=_needs_openai,
    ),
    Case('bedrock-claude-default', 'bedrock-claude', timedelta(minutes=5), marks=_needs_bedrock),
    Case(
        'bedrock-claude-caching-1h',
        'bedrock-claude',
        timedelta(hours=1),
        caching=Caching(retention='1h'),
        marks=_needs_bedrock,
    ),
    Case('openrouter-anthropic', 'openrouter-anthropic', timedelta(minutes=5), marks=_needs_openrouter),
    Case('openrouter-gemini', 'openrouter-gemini', timedelta(minutes=5), marks=_needs_openrouter),
    Case('bedrock-gpt-5.6', 'bedrock-gpt-5.6', timedelta(minutes=30), marks=_needs_bedrock),
]


class TestCompactOnCacheExpiry:
    @pytest.mark.moves_cache_prefix(reason='Clearing tool results after the cache expired is the behavior under test.')
    @pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in KNOWN_RETENTION])
    async def test_compacts_only_once_the_retention_window_has_passed(self, case: Case) -> None:
        """Within the window the history goes out unchanged; past it, the old tool results are cleared."""
        model = _MODELS[case.model]()
        caching = [case.caching] if case.caching is not None else []
        kwargs: dict[str, Any] = {'model_settings': case.settings()} if case.settings is not None else {}
        history = await _conversation(model, **kwargs)
        margin = timedelta(minutes=1)

        warm = _Local(model)
        result = await _agent(warm, *caching, CompactOnCacheExpiry(max_fraction=None), **kwargs).run(
            'next', message_history=_idle(history, case.retention - margin)
        )
        assert _cleared(warm.sent[0]) == []
        assert _cleared(result.all_messages()) == []

        cold = _Local(model)
        result = await _agent(cold, *caching, CompactOnCacheExpiry(max_fraction=None), **kwargs).run(
            'next', message_history=_idle(history, case.retention + margin)
        )
        # The 5 newest of the 8 tool results stay, as in Claude Code's idle clearing.
        assert len(_cleared(cold.sent[0])) == TOOL_STEPS - 5
        # The edit persists into the run's history, so the next turn doesn't clear (or move the prefix) again.
        assert _cleared(result.all_messages()) == _cleared(cold.sent[0])

    @pytest.mark.parametrize(
        'make_model',
        [
            pytest.param(_google, id='google', marks=_needs_google),
            pytest.param(_openai_chat_before_breakpoints, id='openai-before-gpt-5.6', marks=_needs_openai),
            pytest.param(lambda: _bedrock('us.amazon.nova-pro-v1:0'), id='bedrock-nova', marks=_needs_bedrock),
            pytest.param(
                lambda: FallbackModel(_anthropic(), _openai_responses()),
                id='fallback',
                marks=(*_needs_anthropic, *_needs_openai),
            ),
            pytest.param(TestModel, id='test-model'),
        ],
    )
    async def test_unknown_retention_never_compacts(self, make_model: Any) -> None:
        """Without a documented retention there is no expiry to detect, so even a day-old history is left alone.

        `FallbackModel` is unknown because the model that will serve the request isn't chosen yet.
        """
        model: Model = make_model()
        history = _idle(
            await _conversation(_anthropic() if isinstance(model, FallbackModel) else model), timedelta(days=1)
        )
        local = _Local(model)
        await _agent(local, CompactOnCacheExpiry(max_fraction=None)).run('next', message_history=history)
        assert _cleared(local.sent[0]) == []

    @pytest.mark.moves_cache_prefix(
        reason='Clearing tool results after the assumed retention is the behavior under test.'
    )
    @pytest.mark.skipif(not google_imports(), reason='google-genai not installed')
    async def test_fallback_retention_covers_an_undocumented_provider(self) -> None:
        model = _google()
        history = await _conversation(model)
        capability = CompactOnCacheExpiry(max_fraction=None, fallback_retention=timedelta(hours=1))

        warm = _Local(model)
        await _agent(warm, capability).run('next', message_history=_idle(history, timedelta(minutes=59)))
        assert _cleared(warm.sent[0]) == []

        cold = _Local(model)
        await _agent(cold, capability).run('next', message_history=_idle(history, timedelta(minutes=61)))
        assert len(_cleared(cold.sent[0])) == TOOL_STEPS - 5

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_fallback_retention_does_not_override_a_documented_one(self) -> None:
        """Anthropic documents 5 minutes, so a longer fallback doesn't delay compaction there."""
        model = _anthropic()
        history = await _conversation(model)
        local = _Local(model)
        capability = CompactOnCacheExpiry(max_fraction=None, fallback_retention=timedelta(hours=1))
        await _agent(local, capability).run('next', message_history=_idle(history, timedelta(minutes=6)))
        assert len(_cleared(local.sent[0])) == TOOL_STEPS - 5

    @pytest.mark.skipif(not openai_imports(), reason='openai not installed')
    async def test_fallback_retention_does_not_override_a_requested_one(self) -> None:
        """A retention the settings ask for wins over the fallback, even where the provider documents none."""
        model = _openai_chat_before_breakpoints()
        settings = OpenAIChatModelSettings(openai_prompt_cache_retention='24h')
        history = _idle(await _conversation(model, model_settings=settings), timedelta(hours=23))
        local = _Local(model)
        capability = CompactOnCacheExpiry(max_fraction=None, fallback_retention=timedelta(minutes=1))
        await _agent(local, capability, model_settings=settings).run('next', message_history=history)
        assert _cleared(local.sent[0]) == []

    @pytest.mark.moves_cache_prefix(reason='Clearing tool results after the cache expired is the behavior under test.')
    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_decides_on_the_model_the_request_goes_to(self) -> None:
        """A capability that replaces the request's model earlier in the chain changes the retention used."""
        anthropic = _Local(_anthropic())
        history = _idle(await _conversation(_anthropic()), timedelta(minutes=6))
        hooks = Hooks[None]()

        @hooks.on.before_model_request
        async def route(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
            return replace(request_context, model=anthropic)

        # The agent's own `TestModel` has no documented retention, so without the swap nothing would be cleared.
        await _agent(TestModel(), hooks, CompactOnCacheExpiry(max_fraction=None)).run('next', message_history=history)
        assert len(_cleared(anthropic.sent[0])) == TOOL_STEPS - 5

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_never_rewrites_a_resumed_response(self) -> None:
        """Resuming a response the provider paused sends the history back exactly, however long the pause.

        Called on the hook directly: a paused provider response can't be produced without a recording, and
        the hook sees the resume as a history ending in that response rather than in a new request.
        """
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        capability = CompactOnCacheExpiry[None](max_fraction=None)

        async def sent(messages: list[ModelMessage]) -> list[ModelMessage]:
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
            request_context = ModelRequestContext(
                model=model, messages=messages, model_settings=None, model_request_parameters=ModelRequestParameters()
            )
            return (await capability.before_model_request(ctx, request_context)).messages

        assert isinstance(history[-1], ModelResponse)
        assert await sent(list(history)) == history
        # The same expired history followed by a new request is compacted.
        assert len(_cleared(await sent([*history, ModelRequest(parts=[UserPromptPart('next')])]))) == TOOL_STEPS - 5

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_measures_the_gap_to_the_request_being_sent(self) -> None:
        """The gap ends at the request's own timestamp, not the wall clock, so the stored history shows why it fired.

        Naive timestamps, as an older or hand-built history may carry, are read as UTC.
        """
        model = _anthropic()
        history = [
            replace(message, timestamp=message.timestamp.replace(tzinfo=None))
            if isinstance(message, ModelResponse)
            else message
            for message in await _conversation(model)
        ]
        last_response = history[-1]
        assert isinstance(last_response, ModelResponse)
        request = ModelRequest(parts=[UserPromptPart('next')], timestamp=last_response.timestamp + timedelta(minutes=6))
        messages = [*history, request]
        ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
        request_context = ModelRequestContext(
            model=model, messages=messages, model_settings=None, model_request_parameters=ModelRequestParameters()
        )
        sent = (await CompactOnCacheExpiry[None](max_fraction=None).before_model_request(ctx, request_context)).messages
        # By the wall clock the last response is seconds old, so only the request's timestamp makes this expired.
        assert len(_cleared(sent)) == TOOL_STEPS - 5

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_one_hour_cache_point_extends_the_window(self) -> None:
        """A `CachePoint(ttl='1h')` the provider already received keeps a 30-minute-old cache warm."""
        model = _anthropic()
        history = await _conversation(model)
        first = history[0]
        assert isinstance(first, ModelRequest)
        history[0] = replace(first, parts=[UserPromptPart(['investigate', CachePoint(ttl='1h')])])
        local = _Local(model)
        await _agent(local, CompactOnCacheExpiry(max_fraction=None)).run(
            'next', message_history=_idle(history, timedelta(minutes=30))
        )
        assert _cleared(local.sent[0]) == []

    @pytest.mark.moves_cache_prefix(reason='The first request of the turn clears tool results after the cache expired.')
    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_compacts_once_per_expiry(self) -> None:
        """Requests later in the same run follow a fresh response, so the cache is warm again and nothing more is cleared."""
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        local = _Local(model, tool_steps=3)
        result = await _agent(local, CompactOnCacheExpiry(max_fraction=None)).run('next', message_history=history)

        cleared = [_cleared(messages) for messages in local.sent]
        assert [len(ids) for ids in cleared] == [TOOL_STEPS - 5] * 4
        # Each later request extends the first one's prefix rather than rewriting it.
        for earlier, later in zip(local.sent, local.sent[1:]):
            assert later[: len(earlier)] == earlier
        assert result.output == 'done'

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_default_size_floor_leaves_a_small_expired_history_alone(self) -> None:
        """By default only a history above a fifth of the window is cleared: a short chat keeps its tool results."""
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))

        default = _Local(model)
        await _agent(default, CompactOnCacheExpiry()).run('next', message_history=history)
        assert _cleared(default.sent[0]) == []

        # The same history is above a fifth of a 1,000-token window.
        small_window = _Local(model)
        await _agent(small_window, CompactOnCacheExpiry(context_window=1_000)).run('next', message_history=history)
        assert len(_cleared(small_window.sent[0])) == TOOL_STEPS - 5

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_max_tokens_replaces_the_default_floor(self) -> None:
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        assert CompactOnCacheExpiry(max_tokens=100).max_fraction is None

        small = _Local(model)
        await _agent(small, CompactOnCacheExpiry(max_tokens=100_000)).run('next', message_history=history)
        assert _cleared(small.sent[0]) == []

        large = _Local(model)
        await _agent(large, CompactOnCacheExpiry(max_tokens=100)).run('next', message_history=history)
        assert len(_cleared(large.sent[0])) == TOOL_STEPS - 5

    @pytest.mark.moves_cache_prefix(reason='The size-based strategy clears tool results; that is the setup.')
    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_size_floor_sees_what_an_earlier_strategy_cleared(self) -> None:
        """Listed after a size-based strategy, as the docs advise, the floor measures the history that strategy left.

        The usage the last response reported predates that clear, so the floor subtracts what was reclaimed.
        """

        @dataclass
        class Recording:
            calls: int = 0

            async def compact(self, messages: list[ModelMessage], ctx: RunContext[None]) -> list[ModelMessage]:
                self.calls += 1
                return messages

        model = _anthropic()
        history = _idle(await _conversation(model, report_usage=True), timedelta(minutes=6))
        size = estimate_token_count(history)

        alone = Recording()
        await _agent(_Local(model), CompactOnCacheExpiry(alone, max_tokens=size // 2)).run(
            'next', message_history=history
        )
        assert alone.calls == 1

        after_clear = Recording()
        tiered = TieredCompaction(tiers=[ClearToolResults(max_tokens=1, keep_pairs=1)], target_tokens=size // 4)
        await _agent(_Local(model), tiered, CompactOnCacheExpiry(after_clear, max_tokens=size // 2)).run(
            'next', message_history=history
        )
        assert after_clear.calls == 0

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_strategy_is_configurable_and_pins_survive(self) -> None:
        """Any compaction strategy can run on expiry; pinned content survives one that drops messages."""
        model = _anthropic()
        history = await _conversation(model)
        history.insert(1, ModelRequest(parts=[pin('keep this task state')]))
        history = _idle(history, timedelta(minutes=6))
        local = _Local(model)
        capability = CompactOnCacheExpiry(
            SlidingWindowCompaction(max_messages=1, keep_messages=2, preserve_first_user_message=False),
            max_fraction=None,
        )
        await _agent(local, capability).run('next', message_history=history)

        sent = local.sent[0]
        assert len(sent) < len(history)
        assert any(is_pinned(part) for message in sent if isinstance(message, ModelRequest) for part in message.parts)

    async def test_compact_runs_the_strategy_unconditionally(self) -> None:
        """`compact` ignores the cache, so `compact_now` can drive it from a `/compact` command."""
        history = await _conversation(TestModel())
        compacted = await compact_now(
            CompactOnCacheExpiry(ClearToolResults(max_tokens=1, keep_pairs=1)), history, model=TestModel()
        )
        assert len(_cleared(compacted)) == TOOL_STEPS - 1

    def test_with_focus_reaches_a_focusable_strategy(self) -> None:
        @dataclass
        class Focusable:
            focus: str | None = None

            async def compact(
                self, messages: list[ModelMessage], ctx: RunContext[None]
            ) -> list[ModelMessage]:  # pragma: no cover
                return messages

            def with_focus(self, focus: str) -> Focusable:
                return replace(self, focus=focus)

        focused = CompactOnCacheExpiry(Focusable()).with_focus('the auth refactor')
        assert focused.strategy == Focusable(focus='the auth refactor')

        plain = CompactOnCacheExpiry()
        assert plain.with_focus('anything') is plain

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'fallback_retention': timedelta(0)}, 'fallback_retention must be positive.'),
            ({'max_tokens': 10, 'max_fraction': 0.5}, 'Set at most one of max_tokens or max_fraction.'),
        ],
    )
    def test_invalid_configuration(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            CompactOnCacheExpiry(**kwargs)

    @pytest.mark.moves_cache_prefix(reason='Clearing tool results after the cache expired is the behavior under test.')
    @pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_emits_compact_messages_span(self, capfire: CaptureLogfire) -> None:
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        agent = _agent(_Local(model), CompactOnCacheExpiry(max_fraction=None))
        agent.instrument = InstrumentationSettings()
        await agent.run('next', message_history=history)

        spans = [s for s in capfire.exporter.exported_spans_as_dict() if s['name'] == 'compact_messages']
        (span,) = spans
        # The history was moved back 6 minutes, so the gap is 360 seconds plus the test's own run time.
        assert 360 <= span['attributes']['compaction.idle_seconds'] < 420
        assert [
            {k: v for k, v in s['attributes'].items() if k.startswith(('compaction.strategy', 'compaction.messages'))}
            for s in spans
        ] == snapshot(
            [
                {
                    'compaction.strategy': 'CompactOnCacheExpiry',
                    'compaction.messages_before': 19,
                    'compaction.messages_after': 19,
                }
            ]
        )
