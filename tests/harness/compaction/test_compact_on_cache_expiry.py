"""Tests for `CompactOnCacheExpiry`.

It requests compaction with a `CompactionRequestedEvent` that the agent's compaction capabilities honour,
and runs its own fallback when none does. The capability's provider-specific behavior is the retention window it compares the idle gap
against: the provider's documented default, or the retention the request's settings ask for. These
tests use the real model classes, so that resolution runs exactly as in production, and replace
only the network call (`_Local`). A recorded test can't show the decision either way: whether the
cache is cold depends on the wall-clock gap between two requests, which a replayed cassette doesn't
have, so the tests set the gap by moving the stored history's timestamps back, as it looks to an
application that loads a conversation the user left idle.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import (
    AbstractCapability,
    Caching,
    CompactionRequestedEvent,
    SelectModel,
    on_event,
)
from pydantic_ai.messages import (
    CachePoint,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    ModelSelectionContext,
    StreamedResponse,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    CompactOnCacheExpiry,
    FallbackCompaction,
    RelevanceCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    estimate_token_count,
    is_pinned,
    pin,
)
from tests.conftest import RequestCapture, try_import

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_imports:
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAICompaction,
        OpenAIResponsesModel,
    )
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

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        response = await self.request(messages, model_settings, model_request_parameters)

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            for index, part in enumerate(response.parts):
                if isinstance(part, ToolCallPart):
                    yield {
                        index: DeltaToolCall(part.tool_name, part.args_as_json_str(), tool_call_id=part.tool_call_id)
                    }
                else:
                    assert isinstance(part, TextPart)
                    yield part.content

        async with FunctionModel(stream_function=stream).request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as streamed:
            yield streamed


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


HINT = '[Tool result cleared to save context. Call the tool again if you need it.]'
"""The default fallback's placeholder."""


def _cleared(messages: Sequence[ModelMessage], placeholder: str = HINT) -> list[str]:
    """Tool call ids whose result has been cleared with `placeholder`."""
    return [
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.content == placeholder
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

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_decides_on_the_selected_model(self) -> None:
        """A model chosen by `SelectModel` sets the retention used, not the agent's default model."""
        anthropic = _Local(_anthropic())
        history = _idle(await _conversation(_anthropic()), timedelta(minutes=6))

        def select(ctx: ModelSelectionContext[None]) -> Model:
            return anthropic

        # The agent's own `TestModel` has no documented retention, so without the selection nothing would be cleared.
        await _agent(TestModel(), SelectModel(select), CompactOnCacheExpiry(max_fraction=None)).run(
            'next', message_history=history
        )
        assert len(_cleared(anthropic.sent[0])) == TOOL_STEPS - 5

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_never_rewrites_a_resumed_response(self) -> None:
        """Resuming a response the provider paused sends the history back exactly, however long the pause."""
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        last = history[-1]
        assert isinstance(last, ModelResponse)
        history[-1] = replace(last, state='suspended')
        local = _Local(model)
        await _agent(local, CompactOnCacheExpiry(max_fraction=None)).run(message_history=history)
        assert local.sent[0] == history

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
        # A run stamps the request it sends with the wall clock, so the request is stamped by hand and the hook
        # driven from inside a run.
        request = ModelRequest(parts=[UserPromptPart('next')], timestamp=last_response.timestamp + timedelta(minutes=6))
        capability = CompactOnCacheExpiry[None](max_fraction=None)
        sent: list[list[ModelMessage]] = []

        @dataclass
        class Driver(AbstractCapability[None]):
            async def before_model_request(
                self, ctx: RunContext[None], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                ctx.messages[:] = [*history, request]
                compacted = await capability.before_model_request(ctx, replace(request_context, model=model))
                sent.append(compacted.messages)
                return request_context

        await _agent(TestModel(), Driver()).run('go')
        # By the wall clock the last response is seconds old, so only the request's timestamp makes this expired.
        assert len(_cleared(sent[0])) == TOOL_STEPS - 5

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

        small = _Local(model)
        await _agent(small, CompactOnCacheExpiry(max_tokens=100_000)).run('next', message_history=history)
        assert _cleared(small.sent[0]) == []

        large = _Local(model)
        await _agent(large, CompactOnCacheExpiry(max_tokens=100)).run('next', message_history=history)
        assert len(_cleared(large.sent[0])) == TOOL_STEPS - 5

    @pytest.mark.parametrize(
        ('compaction', 'name'),
        [
            pytest.param(
                lambda: SlidingWindowCompaction(max_messages=1_000, keep_messages=4, preserve_first_user_message=False),
                'SlidingWindowCompaction',
                id='sliding-window',
            ),
            pytest.param(
                lambda: ClearToolResults(max_messages=1_000, keep_pairs=2), 'ClearToolResults', id='clear-tool-results'
            ),
            pytest.param(
                lambda: SummarizingCompaction(
                    FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('8 lookups ran.')])),
                    max_messages=1_000,
                    keep_messages=2,
                ),
                'SummarizingCompaction',
                id='summarizing',
            ),
            pytest.param(
                lambda: RelevanceCompaction(
                    FunctionModel(
                        lambda messages, info: ModelResponse(
                            parts=[
                                ToolCallPart(
                                    info.output_tools[0].name, {'needed_verbatim': 0.05, 'still_matters': 0.05}
                                )
                            ]
                        )
                    ),
                    max_tokens=1_000_000,
                    target_tokens=100,
                    keep_tokens=10,
                ),
                'RelevanceCompaction',
                id='relevance',
            ),
            pytest.param(
                lambda: TieredCompaction(tiers=[ClearToolResults(max_tokens=1, keep_pairs=2)], target_tokens=1_000_000),
                'TieredCompaction',
                id='tiered',
            ),
            pytest.param(
                lambda: FallbackCompaction(
                    fallback_chain=[
                        SlidingWindowCompaction(max_messages=1, keep_messages=4, preserve_first_user_message=False)
                    ]
                ),
                'FallbackCompaction',
                id='fallback-without-a-threshold',
            ),
        ],
    )
    @pytest.mark.parametrize('stream', [False, True], ids=['run', 'stream'])
    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_a_registered_compaction_handles_the_request(
        self, compaction: Callable[[], AbstractCapability[None]], name: str, stream: bool
    ) -> None:
        """The agent's own compaction honours the request, whatever its own trigger says, and the fallback stays out."""
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        local = _Local(model)
        events: list[CompactionRequestedEvent] = []

        @dataclass
        class Observer(AbstractCapability[None]):
            @on_event(CompactionRequestedEvent)
            async def record(self, ctx: RunContext[None], event: CompactionRequestedEvent) -> None:
                events.append(event)

        agent = _agent(local, compaction(), CompactOnCacheExpiry(max_fraction=None), Observer())
        if stream:
            async with agent.run_stream('next', message_history=history) as result:
                await result.get_output()
            messages = result.all_messages()
        else:
            messages = (await agent.run('next', message_history=history)).all_messages()

        assert [(event.reason, event.compacted_by) for event in events] == [('cache_expired', name)]
        sent = local.sent[0]
        assert sent[:-1] != history
        assert _cleared(sent) == []
        # The compacted history is the run's history from then on.
        assert messages[: len(history)] != history

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_the_first_compaction_in_the_list_wins(self) -> None:
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))

        clearing = ClearToolResults(max_messages=1_000, keep_pairs=2)
        sliding = SlidingWindowCompaction(max_messages=1_000, keep_messages=4, preserve_first_user_message=False)
        clearing_first = _Local(model)
        await _agent(clearing_first, clearing, sliding, CompactOnCacheExpiry(max_fraction=None)).run(
            'next', message_history=history
        )
        # Only the clearing ran: the sliding window would have dropped all but 4 messages.
        assert len(clearing_first.sent[0]) == len(history) + 1
        assert len(_cleared(clearing_first.sent[0], '[tool result cleared]')) == TOOL_STEPS - 2

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_a_compaction_that_changes_nothing_leaves_it_to_the_fallback(self) -> None:
        """Keeping more tool results than the history has changes nothing, so the request falls through."""
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        local = _Local(model)
        keeps_everything = ClearToolResults(max_messages=1_000, keep_pairs=100)
        await _agent(local, keeps_everything, CompactOnCacheExpiry(max_fraction=None)).run(
            'next', message_history=history
        )
        assert len(_cleared(local.sent[0])) == TOOL_STEPS - 5

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_without_a_fallback_an_unhandled_request_changes_nothing(self) -> None:
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        local = _Local(model)
        await _agent(local, CompactOnCacheExpiry(None, max_fraction=None)).run('next', message_history=history)
        assert local.sent[0][:-1] == history

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_a_fallback_that_changes_nothing_leaves_the_history_alone(self) -> None:
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        local = _Local(model)
        keeps_everything = ClearToolResults(max_tokens=1, keep_pairs=100)
        result = await _agent(local, CompactOnCacheExpiry(keeps_everything, max_fraction=None)).run(
            'next', message_history=history
        )
        assert local.sent[0][:-1] == history
        assert result.all_messages()[: len(history)] == history

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_tiered_runs_its_first_tier_even_under_target(self) -> None:
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        local = _Local(model)
        tiered = TieredCompaction(
            tiers=[
                ClearToolResults(max_tokens=1, keep_pairs=3),
                SlidingWindowCompaction(max_messages=1, keep_messages=2, preserve_first_user_message=False),
            ],
            target_tokens=1_000_000,
        )
        await _agent(local, tiered, CompactOnCacheExpiry(max_fraction=None)).run('next', message_history=history)
        # Only the first tier ran: every message is still there, with all but 3 tool results cleared.
        assert len(local.sent[0]) == len(history) + 1
        assert len(_cleared(local.sent[0], '[tool result cleared]')) == TOOL_STEPS - 3

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_a_summary_written_for_the_request_is_not_written_again(self) -> None:
        """A summarizer over its own threshold handles the request, then its own hook sees the compacted size.

        The last response's reported usage still describes the history before the summary, so without
        subtracting what was reclaimed the summarizer would summarize the summary on the same request.
        """
        model = _anthropic()
        history = _idle(await _conversation(model, report_usage=True), timedelta(minutes=6))
        summaries = 0

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal summaries
            summaries += 1
            return ModelResponse(parts=[TextPart('The user is investigating; 8 lookups ran.')])

        local = _Local(model)
        summarizer = SummarizingCompaction(
            FunctionModel(summarize), max_tokens=estimate_token_count(history) // 2, keep_messages=2
        )
        await _agent(local, summarizer, CompactOnCacheExpiry(max_fraction=None)).run('next', message_history=history)
        assert summaries == 1

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_tiered_measures_the_history_its_first_tier_left(self) -> None:
        """After its first tier handles the request, `TieredCompaction` doesn't escalate on the stale reported usage."""
        model = _anthropic()
        history = _idle(await _conversation(model, report_usage=True), timedelta(minutes=6))
        local = _Local(model)
        tiered = TieredCompaction(
            tiers=[
                ClearToolResults(max_tokens=1, keep_pairs=3),
                SlidingWindowCompaction(max_messages=1, keep_messages=2, preserve_first_user_message=False),
            ],
            # Below the history's size before clearing, above it after.
            target_tokens=estimate_token_count(history) - 250,
        )
        # Listed first, the expiry request comes before Tiered's own size check on the same request.
        await _agent(local, CompactOnCacheExpiry(max_fraction=None), tiered).run('next', message_history=history)
        assert len(local.sent[0]) == len(history) + 1
        assert len(_cleared(local.sent[0], '[tool result cleared]')) == TOOL_STEPS - 3

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_a_request_already_compacted_is_not_compacted_again(self) -> None:
        """A compaction whose own trigger fired first for the request leaves nothing for the expiry to ask for."""
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        local = _Local(model)
        events: list[CompactionRequestedEvent] = []

        @dataclass
        class Observer(AbstractCapability[None]):
            @on_event(CompactionRequestedEvent)
            async def record(self, ctx: RunContext[None], event: CompactionRequestedEvent) -> None:
                events.append(event)

        clearing = ClearToolResults(max_tokens=1, keep_pairs=2)
        await _agent(local, clearing, CompactOnCacheExpiry(max_fraction=None), Observer()).run(
            'next', message_history=history
        )
        assert events == []
        assert len(_cleared(local.sent[0], '[tool result cleared]')) == TOOL_STEPS - 2
        assert _cleared(local.sent[0]) == []

    @pytest.mark.vcr
    @pytest.mark.moves_cache_prefix(reason='OpenAI compacts the history after the cache expired.')
    @pytest.mark.skipif(not openai_imports(), reason='openai not installed')
    async def test_openai_compaction_handles_the_request(
        self, allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
    ) -> None:
        """With `OpenAICompaction` registered, the expired history is compacted by OpenAI's `/responses/compact`."""
        provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
        model = OpenAIResponsesModel('gpt-5.6', provider=provider)
        compacted_by: list[str | None] = []

        @dataclass
        class Observer(AbstractCapability[None]):
            @on_event(CompactionRequestedEvent)
            async def record(self, ctx: RunContext[None], event: CompactionRequestedEvent) -> None:
                compacted_by.append(event.compacted_by)

        instructions = 'You are a helpful math assistant. Give short answers.'
        first = await _agent(model, instructions=instructions).run('What is 2+2?')
        # GPT-5.6 keeps the cache 30 minutes.
        history = _idle(first.all_messages(), timedelta(minutes=31))
        agent = _agent(
            model,
            OpenAICompaction(message_count_threshold=1_000),
            CompactOnCacheExpiry(max_fraction=None),
            Observer(),
            instructions=instructions,
        )
        second = await agent.run('And 3+3?', message_history=history)

        assert '6' in second.output
        assert compacted_by == ['OpenAICompaction']
        assert len(request_capture.bodies('/responses/compact')) == 1
        first_message = second.all_messages()[0]
        assert isinstance(first_message, ModelResponse)
        assert isinstance(first_message.parts[0], CompactionPart)

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_several_compact_an_expired_request_once(self) -> None:
        """A second `CompactOnCacheExpiry` leaves a request the first already acted on alone."""
        model = _anthropic()
        history = _idle(await _conversation(model), timedelta(minutes=6))
        local = _Local(model)
        second = CompactOnCacheExpiry(ClearToolResults(max_tokens=1, keep_pairs=1), max_fraction=None, id='second')
        await _agent(local, CompactOnCacheExpiry(max_fraction=None), second).run('next', message_history=history)
        assert len(_cleared(local.sent[0])) == TOOL_STEPS - 5
        assert _cleared(local.sent[0], '[tool result cleared]') == []

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    async def test_fallback_is_configurable_and_pins_survive(self) -> None:
        """Any compaction strategy can be the fallback; pinned content survives one that drops messages."""
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
        # The history was moved back 6 minutes.
        assert span['attributes']['compaction.idle_seconds'] >= 360
        assert [
            {
                k: v
                for k, v in s['attributes'].items()
                if k.startswith(('compaction.strategy', 'compaction.messages', 'compaction.reason'))
            }
            for s in spans
        ] == snapshot(
            [
                {
                    'compaction.strategy': 'CompactOnCacheExpiry',
                    'compaction.messages_before': 19,
                    'compaction.messages_after': 19,
                    'compaction.reason': 'cache_expired',
                }
            ]
        )
