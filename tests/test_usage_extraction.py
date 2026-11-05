from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from pydantic_ai import Agent, UsageExtractionFailedWarning, UsageLimitUnavailableWarning, usage as usage_module
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ._inline_snapshot import snapshot
from .conftest import IsNow, try_import

with try_import() as openai_imports:
    from openai.types import chat
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAI, completion_message


class FakeProvider:
    def __init__(self, outcome: SimpleNamespace | Exception):
        self.outcome = outcome

    def extract_usage(self, data: Any, *, api_flavor: str) -> tuple[str, SimpleNamespace]:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return 'model', self.outcome


class FakeSnapshot:
    def __init__(self, outcomes: dict[tuple[str | None, str | None], FakeProvider | Exception]):
        self.outcomes = outcomes

    def find_provider(self, model_ref: None, provider_id: str | None, provider_api_url: str | None) -> FakeProvider:
        outcome = self.outcomes.get((provider_id, provider_api_url), LookupError())
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_usage_extraction_failure_warns_once(monkeypatch: pytest.MonkeyPatch):
    """Usage extraction is best effort, but unexpected failures must be observable."""
    provider = FakeProvider(RuntimeError('provider extractor raised'))
    snapshot = FakeSnapshot(
        {
            (None, 'https://example.com'): provider,
            ('provider', None): provider,
            ('fallback', None): provider,
        }
    )
    monkeypatch.setattr(usage_module, 'get_snapshot', lambda: snapshot)

    with pytest.warns(
        UsageExtractionFailedWarning, match='Failed to extract usage with `genai-prices`: RuntimeError'
    ) as caught:
        extracted = RequestUsage.extract(
            {}, provider='provider', provider_url='https://example.com', provider_fallback='fallback'
        )

    assert extracted == RequestUsage(unmeasured_requests=1)
    assert len(caught) == 1


def test_usage_extraction_falls_back_without_warning(monkeypatch: pytest.MonkeyPatch):
    """A fake snapshot isolates fallback behavior that no fixed provider response guarantees."""
    snapshot = FakeSnapshot(
        {
            (None, 'https://example.com'): FakeProvider(RuntimeError('gateway extractor failed')),
            ('provider', None): FakeProvider(SimpleNamespace(input_tokens=42)),
        }
    )
    monkeypatch.setattr(usage_module, 'get_snapshot', lambda: snapshot)

    extracted = RequestUsage.extract(
        {}, provider='provider', provider_url='https://example.com', provider_fallback='fallback'
    )

    assert extracted == RequestUsage(input_tokens=42)


def test_usage_construction_failure_warns(monkeypatch: pytest.MonkeyPatch):
    """A fake usage shape reproduces incompatibility with a future `genai-prices` field."""
    provider = FakeProvider(SimpleNamespace(requests=1))
    snapshot = FakeSnapshot(
        {
            (None, 'https://example.com'): provider,
            ('provider', None): provider,
            ('fallback', None): provider,
        }
    )
    monkeypatch.setattr(usage_module, 'get_snapshot', lambda: snapshot)

    with pytest.warns(UsageExtractionFailedWarning, match='AttributeError'):
        extracted = RequestUsage.extract(
            {}, provider='provider', provider_url='https://example.com', provider_fallback='fallback'
        )

    assert extracted == RequestUsage(unmeasured_requests=1)


def test_usage_provider_lookup_failure_warns(monkeypatch: pytest.MonkeyPatch):
    """A fake snapshot exercises an unexpected lookup failure without corrupting pricing data."""
    snapshot = FakeSnapshot({(None, 'https://example.com'): RuntimeError('snapshot failed')})
    monkeypatch.setattr(usage_module, 'get_snapshot', lambda: snapshot)

    with pytest.warns(UsageExtractionFailedWarning, match='RuntimeError: snapshot failed'):
        extracted = RequestUsage.extract(
            {}, provider='unknown', provider_url='https://example.com', provider_fallback='fallback'
        )

    assert extracted == RequestUsage(unmeasured_requests=1)


class FailingSnapshot:
    def find_provider(self, *args: Any) -> FakeProvider:
        return FakeProvider(RuntimeError('provider extractor raised'))


@pytest.fixture
def failing_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every usage extraction fail unexpectedly, while pricing keeps using its real snapshot."""
    monkeypatch.setattr(usage_module, 'get_snapshot', FailingSnapshot)


def openai_response_model() -> OpenAIChatModel:
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    completion = completion_message(ChatCompletionMessage(content='hello', role='assistant'), usage=usage)
    completion = completion.model_copy(update={'model': 'gpt-4o'})
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(completion)))


def openai_chunk(content: str | None, usage: CompletionUsage | None = None) -> chat.ChatCompletionChunk:
    choices = [ChunkChoice(index=0, delta=ChoiceDelta(content=content, role='assistant'))] if content else []
    return chat.ChatCompletionChunk(
        id='123', choices=choices, created=1704067200, model='gpt-4o', object='chat.completion.chunk', usage=usage
    )


def openai_stream_model() -> OpenAIChatModel:
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    client = MockOpenAI.create_mock_stream([openai_chunk('hello'), openai_chunk(None, usage)])
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client))


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize(
    'usage_limits',
    [UsageLimits(input_tokens_limit=100), UsageLimits(cost_limit=Decimal('0.01'))],
    ids=['token-limit', 'cost-limit'],
)
@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
async def test_failed_extraction_warns_that_limit_is_unavailable(
    allow_model_requests: None, failing_extraction: None, usage_limits: UsageLimits, stream: bool
):
    """A limit that can't see the response's usage warns once, and the cost stays unknown rather than priced as zero."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if stream:
            async with Agent(openai_stream_model()).run_stream('hello', usage_limits=usage_limits) as result:
                await result.get_output()
            usage = result.usage
        else:
            usage = (await Agent(openai_response_model()).run('hello', usage_limits=usage_limits)).usage

    assert usage == snapshot(RunUsage(requests=1, unmeasured_requests=1))
    assert usage.cost is None
    assert [w.category for w in caught if w.category is UsageLimitUnavailableWarning] == [UsageLimitUnavailableWarning]
    assert any(w.category is UsageExtractionFailedWarning for w in caught)


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_failed_extraction_without_token_or_cost_limit_only_reports_the_failure(
    allow_model_requests: None, failing_extraction: None
):
    """The default `UsageLimits` only caps requests, which don't depend on the provider's usage."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', UsageLimitUnavailableWarning)
        with pytest.warns(UsageExtractionFailedWarning):
            result = await Agent(openai_response_model()).run('hello')

    assert result.usage == snapshot(RunUsage(requests=1, unmeasured_requests=1))


@dataclass
class RecordedUsageStream(StreamedResponse):
    """Exposes `_add_usage`, since no recording reliably produces an extraction failure on chosen stream events."""

    _usage_received: bool | None = False

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        raise NotImplementedError
        yield  # pragma: no cover

    @property
    def model_name(self) -> str:
        return 'model'

    @property
    def provider_name(self) -> str:
        return 'provider'

    @property
    def provider_url(self) -> str:
        return 'https://example.com'

    @property
    def timestamp(self) -> datetime:
        return IsNow()


FAILED = RequestUsage(unmeasured_requests=1)
MEASURED = RequestUsage(input_tokens=10)


@pytest.mark.parametrize(
    ('events', 'cumulative', 'expected'),
    [
        pytest.param([FAILED, FAILED], False, RequestUsage(unmeasured_requests=1), id='repeat-counts-one-response'),
        pytest.param(
            [MEASURED, FAILED], False, RequestUsage(input_tokens=10, unmeasured_requests=1), id='later-failure-sticks'
        ),
        pytest.param(
            [FAILED, MEASURED], False, RequestUsage(input_tokens=10, unmeasured_requests=1), id='earlier-failure-sticks'
        ),
        pytest.param([FAILED, MEASURED], True, MEASURED, id='cumulative-success-replaces-failure'),
        pytest.param(
            [MEASURED, FAILED], True, RequestUsage(unmeasured_requests=1), id='cumulative-failure-replaces-success'
        ),
    ],
)
def test_stream_extraction_failure_marks_the_response_once(
    events: list[RequestUsage], cumulative: bool, expected: RequestUsage
):
    response = RecordedUsageStream(ModelRequestParameters())
    for event in events:
        response._add_usage(event, reported=True, cumulative=cumulative)  # pyright: ignore[reportPrivateUsage]

    assert response.usage == expected


def test_stream_event_without_usage_keeps_earlier_extraction_failure():
    """Gemini's mapping hands back the usage so far for a chunk without metadata, which must not erase the failure."""
    response = RecordedUsageStream(ModelRequestParameters())
    response._add_usage(FAILED, reported=True, cumulative=True)  # pyright: ignore[reportPrivateUsage]
    response._add_usage(FAILED, reported=False, cumulative=True)  # pyright: ignore[reportPrivateUsage]

    assert response.usage == RequestUsage(unmeasured_requests=1)
