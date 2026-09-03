"""Tests for Anthropic thinking-block binding and stale-block recovery."""

from __future__ import annotations

import base64
import json
import warnings
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import httpx2
import pytest
from opentelemetry import trace

from pydantic_ai import (
    Agent,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    ThinkingPart,
    UsageLimits,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.instrumented import InstrumentedModel

from ..._inline_snapshot import snapshot
from ...conftest import IsInt, RequestCapture, message, try_import
from ..conftest import AnthropicModelFactory
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs, mock_anthropic_client

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire

with try_import() as anthropic_imports_successful:
    from anthropic import (
        APIStatusError,
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
        Omit,
        omit as OMIT,
    )
    from anthropic.types.beta import (
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaMessageTokensCount,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaTextBlock,
        BetaThinkingDroppedInputTransformation,
        BetaUsage,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        AnthropicModelSettings,
        AnthropicStaleThinkingBlockWarning,
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

if not anthropic_imports_successful():  # pragma: lax no cover
    AsyncAnthropicBedrock = AsyncAnthropicBedrockMantle = AsyncAnthropicVertex = AsyncAnthropicFoundry = None

pytestmark = [
    pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

_THINKING_BINDING_BETA = 'thinking-binding-controls-2026-08-01'


def sent_betas(mock_client: AsyncAnthropic) -> list[str]:
    """The `betas` the model sent, as a list — an empty set reaches the SDK as `OMIT`, not `[]`."""
    betas: list[str] | Omit = get_mock_chat_completion_kwargs(mock_client)[0].get('betas', OMIT)
    return [] if isinstance(betas, Omit) else betas


def stale_thinking_block_error() -> APIStatusError:
    """Anthropic's rejection of a replayed thinking block, verbatim from a live 400."""
    return APIStatusError(
        'stale thinking block',
        response=httpx2.Response(status_code=400, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'message': 'messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a '
                'different conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` '
                'to "drop_block". The `system` prompt differs from the one this block was created with.',
            },
        },
    )


def recovered_thinking_history() -> list[ModelMessage]:
    """History after Anthropic has dropped a stale thinking block once."""
    return [
        ModelRequest.user_text_prompt('First turn'),
        ModelResponse(
            parts=[ThinkingPart(content='reasoning', signature='signature', provider_name='anthropic')],
            provider_name='anthropic',
            provider_details={
                'input_transformations': [
                    {'path': 'messages.1.content.0', 'reason': 'prefix_binding_mismatch', 'type': 'thinking_dropped'}
                ]
            },
        ),
        ModelRequest.user_text_prompt('Third turn'),
    ]


@pytest.mark.parametrize('model_name', ['claude-fable-5-1', 'claude-fable-5'])
@pytest.mark.parametrize('settings', [None, ModelSettings(thinking='high')])
async def test_anthropic_sends_no_block_binding_by_default(
    allow_model_requests: None, model_name: str, settings: ModelSettings | None
):
    """No request asks for a binding behavior, so Anthropic's account default stands.

    An account created before 2026-08-31 has the mismatch recorded but not acted on, and keeps
    replaying its reasoning. Asking for `drop_block` up front would take that away from it.
    """
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='4', type='text')], usage=BetaUsage(input_tokens=10, output_tokens=1))
    )
    m = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=mock_client))

    await Agent(m, model_settings=settings).run('What is 2+2?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    thinking = kwargs['thinking']
    assert thinking is OMIT or 'block_binding' not in thinking
    assert kwargs.get('extra_body') is None
    assert _THINKING_BINDING_BETA not in sent_betas(mock_client)


async def test_anthropic_retries_a_stale_thinking_block_with_drop_block(allow_model_requests: None):
    """A rejected replay is retried once asking Anthropic to drop the block, and the run continues.

    The retried `thinking` object rides in `extra_body`: a request that configured no thinking has
    no typed home for `block_binding`, and the SDK's `thinking` union always requires a `type` the
    caller never asked for.
    """
    mock_client = MockAnthropic.create_mock(
        [
            stale_thinking_block_error(),
            completion_message(
                [BetaTextBlock(text='4', type='text')], usage=BetaUsage(input_tokens=10, output_tokens=1)
            ),
        ]
    )
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.warns(AnthropicStaleThinkingBlockWarning, match='rejected a replayed thinking block'):
        result = await Agent(m).run('What is 2+2?')

    assert result.output == '4'
    first, retried = get_mock_chat_completion_kwargs(mock_client)
    assert first.get('extra_body') is None
    assert retried['thinking'] is OMIT
    assert retried['extra_body'] == snapshot(
        {'thinking': {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}
    )
    assert _THINKING_BINDING_BETA in retried['betas']


async def test_anthropic_retries_a_stale_thinking_block_streamed(allow_model_requests: None):
    """The retry covers streamed requests too: the SDK raises the 400 out of `create()`, before any event."""
    mock_client = MockAnthropic.create_stream_mock(
        [
            stale_thinking_block_error(),
            dropped_thinking_stream(start_transformation=dropped_thinking_transformation()),
        ]
    )
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.warns(AnthropicStaleThinkingBlockWarning, match='rejected a replayed thinking block'):
        async with Agent(m).run_stream('What is 2+2?') as result:
            assert await result.get_output() == '4'

    first, retried = get_mock_chat_completion_kwargs(mock_client)
    assert first.get('extra_body') is None
    assert retried['thinking'] is OMIT
    assert retried['extra_body'] == snapshot(
        {'thinking': {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}
    )
    assert _THINKING_BINDING_BETA in retried['betas']


async def test_anthropic_keeps_dropping_stale_thinking_blocks_for_the_rest_of_the_session(
    allow_model_requests: None,
):
    """A later turn carrying the recovery response must not pay for another rejected request."""
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='6', type='text')], usage=BetaUsage(input_tokens=12, output_tokens=1))
    )
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    await model.request(recovered_thinking_history(), None, ModelRequestParameters())

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['extra_body'] == snapshot(
        {'thinking': {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}
    )
    assert _THINKING_BINDING_BETA in request['betas']


async def test_anthropic_keeps_dropping_stale_thinking_blocks_for_the_rest_of_a_streamed_session(
    allow_model_requests: None,
):
    """History-scoped recovery has the same first-attempt behavior for streaming."""
    mock_client = MockAnthropic.create_stream_mock(dropped_thinking_stream())
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    async with model.request_stream(recovered_thinking_history(), None, ModelRequestParameters()) as response:
        async for _ in response:
            pass

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['extra_body'] == snapshot(
        {'thinking': {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}
    )
    assert _THINKING_BINDING_BETA in request['betas']


async def test_anthropic_count_tokens_retries_a_stale_thinking_block(allow_model_requests: None):
    """Token counting runs the same prefix check, so it needs the same one-retry recovery."""
    mock_client = MockAnthropic.create_mock(
        completion_message(
            [BetaTextBlock(text='unused', type='text')], usage=BetaUsage(input_tokens=1, output_tokens=1)
        )
    )
    requests: list[dict[str, Any]] = []

    async def count_tokens(**kwargs: Any) -> BetaMessageTokensCount:
        requests.append(kwargs)
        if len(requests) == 1:
            raise stale_thinking_block_error()
        return BetaMessageTokensCount(input_tokens=10)

    mock_client.beta.messages.count_tokens = count_tokens
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Count this')]
    messages[0].metadata = {'__pydantic_ai__': {'other': True}}

    with pytest.warns(AnthropicStaleThinkingBlockWarning, match='rejected a replayed thinking block'):
        result = await model.count_tokens(messages, None, ModelRequestParameters())

    assert result.input_tokens == 10
    assert messages[0].metadata == snapshot(
        {
            '__pydantic_ai__': {
                'other': True,
                'anthropic_count_tokens_drop_stale_thinking_blocks': True,
            }
        }
    )
    assert requests[0].get('extra_body') is None
    assert requests[1]['extra_body'] == snapshot(
        {'thinking': {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}
    )
    assert _THINKING_BINDING_BETA in requests[1]['betas']


async def test_anthropic_count_tokens_recovery_only_carries_into_later_counts(allow_model_requests: None):
    """A count-only prefix mismatch must not make an otherwise-valid inference drop its thinking."""
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='4', type='text')], usage=BetaUsage(input_tokens=10, output_tokens=1))
    )
    count_requests: list[dict[str, Any]] = []

    async def count_tokens(**kwargs: Any) -> BetaMessageTokensCount:
        count_requests.append(kwargs)
        if len(count_requests) == 1:
            raise stale_thinking_block_error()
        return BetaMessageTokensCount(input_tokens=10)

    mock_client.beta.messages.count_tokens = count_tokens
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.filterwarnings('always', category=AnthropicStaleThinkingBlockWarning)
        first = await agent.run(
            'What is 2+2?', usage_limits=UsageLimits(count_tokens_before_request=True, input_tokens_limit=100)
        )
        await agent.run(
            'And again?',
            message_history=first.all_messages(),
            usage_limits=UsageLimits(count_tokens_before_request=True, input_tokens_limit=100),
        )

    assert len(caught_warnings) == 1
    assert len(count_requests) == 3
    assert count_requests[2]['extra_body'] == snapshot(
        {'thinking': {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}
    )
    first_inference, next_turn = get_mock_chat_completion_kwargs(mock_client)
    assert first_inference.get('extra_body') is None
    assert next_turn.get('extra_body') is None


async def test_anthropic_count_tokens_does_not_retry_an_unrelated_bad_request(allow_model_requests: None):
    """Token counting only recovers the exact stale-binding rejection."""
    mock_client = MockAnthropic.create_mock(
        completion_message(
            [BetaTextBlock(text='unused', type='text')], usage=BetaUsage(input_tokens=1, output_tokens=1)
        )
    )
    requests: list[dict[str, Any]] = []

    async def count_tokens(**kwargs: Any) -> BetaMessageTokensCount:
        requests.append(kwargs)
        raise APIStatusError(
            'bad request',
            response=httpx2.Response(status_code=400, request=httpx2.Request('POST', 'https://example.com/v1')),
            body={'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'max_tokens: too large'}},
        )

    mock_client.beta.messages.count_tokens = count_tokens
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.raises(ModelHTTPError, match='max_tokens: too large'):
        await model.count_tokens([ModelRequest.user_text_prompt('Count this')], None, ModelRequestParameters())

    assert len(requests) == 1


async def test_anthropic_count_tokens_keeps_dropping_after_recovery(allow_model_requests: None):
    """Counting a later turn follows the same history-scoped recovery as inference."""
    mock_client = MockAnthropic.create_mock(
        completion_message(
            [BetaTextBlock(text='unused', type='text')], usage=BetaUsage(input_tokens=1, output_tokens=1)
        )
    )
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    await model.count_tokens(recovered_thinking_history(), None, ModelRequestParameters())

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['extra_body'] == snapshot(
        {'thinking': {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}
    )
    assert _THINKING_BINDING_BETA in request['betas']


async def test_anthropic_count_tokens_preserves_an_explicit_binding_after_recovery(allow_model_requests: None):
    """A history marker never overrides the caller's typed choice for token counting."""
    mock_client = MockAnthropic.create_mock(
        completion_message(
            [BetaTextBlock(text='unused', type='text')], usage=BetaUsage(input_tokens=1, output_tokens=1)
        )
    )
    settings = AnthropicModelSettings(
        anthropic_thinking={'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'error'}}
    )
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    await model.count_tokens(recovered_thinking_history(), settings, ModelRequestParameters())

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['thinking'] == snapshot({'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'error'}})
    assert request.get('extra_body') is None


async def test_anthropic_bedrock_count_tokens_sends_persisted_binding_on_the_wire(allow_model_requests: None):
    """Legacy Bedrock carries the documented beta in its InvokeModel-style JSON body."""
    client = mock_anthropic_client(AsyncAnthropicBedrock, 'https://example.com')
    client.post.return_value = {'inputTokens': 10}
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=client))

    await model.count_tokens(recovered_thinking_history(), None, ModelRequestParameters())

    content = json.loads(client.post.await_args.kwargs['content'])
    invoke_model_body = json.loads(base64.b64decode(content['input']['invokeModel']['body']))
    assert invoke_model_body['anthropic_beta'] == [_THINKING_BINDING_BETA]
    assert invoke_model_body['thinking'] == snapshot({'block_binding': {'prefix_mismatch_behavior': 'drop_block'}})


async def test_anthropic_retry_carries_the_configured_thinking_forward(allow_model_requests: None):
    """The retry keeps the caller's own `thinking` config; only `block_binding` is added to it."""
    mock_client = MockAnthropic.create_mock(
        [
            stale_thinking_block_error(),
            completion_message(
                [BetaTextBlock(text='4', type='text')], usage=BetaUsage(input_tokens=10, output_tokens=1)
            ),
        ]
    )
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'adaptive', 'display': 'summarized'})
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.warns(AnthropicStaleThinkingBlockWarning):
        await Agent(m, model_settings=settings).run('What is 2+2?')

    _, retried = get_mock_chat_completion_kwargs(mock_client)
    assert retried['extra_body'] == snapshot(
        {
            'thinking': {
                'type': 'adaptive',
                'display': 'summarized',
                'block_binding': {'prefix_mismatch_behavior': 'drop_block'},
            }
        }
    )


async def test_anthropic_retry_survives_a_caller_extra_body_thinking(allow_model_requests: None):
    """A hand-rolled `extra_body['thinking']` must not swallow the binding the retry adds.

    `extra_body` normally wins over anything Pydantic AI builds, which would make the retried
    request byte-identical to the rejected one — the caller's keys still win here, only the
    `block_binding` they didn't set comes from the retry.
    """
    mock_client = MockAnthropic.create_mock(
        [
            stale_thinking_block_error(),
            completion_message(
                [BetaTextBlock(text='4', type='text')], usage=BetaUsage(input_tokens=10, output_tokens=1)
            ),
        ]
    )
    settings = AnthropicModelSettings(extra_body={'thinking': {'display': 'updates'}})
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.warns(AnthropicStaleThinkingBlockWarning):
        await Agent(m, model_settings=settings).run('What is 2+2?')

    first, retried = get_mock_chat_completion_kwargs(mock_client)
    assert first['extra_body'] == snapshot({'thinking': {'display': 'updates'}})
    assert retried['extra_body'] == snapshot(
        {'thinking': {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}, 'display': 'updates'}}
    )


async def test_anthropic_does_not_retry_a_block_binding_set_through_extra_body(allow_model_requests: None):
    """`extra_body` is also how a caller sets `block_binding`, and that choice is theirs to keep."""
    mock_client = MockAnthropic.create_mock(stale_thinking_block_error())
    settings = AnthropicModelSettings(extra_body={'thinking': {'block_binding': None}})
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.raises(ModelHTTPError, match='bound to a different conversation'):
        await Agent(m, model_settings=settings).run('What is 2+2?')

    assert len(get_mock_chat_completion_kwargs(mock_client)) == 1
    assert _THINKING_BINDING_BETA in sent_betas(mock_client)


async def test_anthropic_respects_block_binding_in_mapping_extra_body(allow_model_requests: None):
    """The SDK accepts any mapping for `extra_body`, not only a concrete `dict`."""
    mock_client = MockAnthropic.create_mock(stale_thinking_block_error())
    extra_body = MappingProxyType({'thinking': {'block_binding': None}, 'custom': 1})
    settings = AnthropicModelSettings(extra_body=extra_body)
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.raises(ModelHTTPError, match='bound to a different conversation'):
        await Agent(m, model_settings=settings).run('What is 2+2?')

    assert len(get_mock_chat_completion_kwargs(mock_client)) == 1
    assert get_mock_chat_completion_kwargs(mock_client)[0]['extra_body'] == extra_body
    assert _THINKING_BINDING_BETA in sent_betas(mock_client)


@pytest.mark.parametrize(
    'client_cls,binds_thinking_blocks',
    [
        pytest.param(MockAnthropic, True, id='direct'),
        pytest.param(AsyncAnthropicBedrock, True, id='bedrock'),
        pytest.param(AsyncAnthropicBedrockMantle, False, id='bedrock-mantle'),
        pytest.param(AsyncAnthropicFoundry, False, id='foundry'),
        pytest.param(AsyncAnthropicVertex, True, id='vertex'),
    ],
)
def test_anthropic_thinking_block_binding_profile_matches_verified_transports(
    client_cls: type, binds_thinking_blocks: bool
) -> None:
    """Enable only transports whose documented and SDK wire paths carry the binding beta."""
    client = mock_anthropic_client(client_cls, 'https://example.com')
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=client))

    assert model.profile.get('anthropic_binds_thinking_blocks', False) is binds_thinking_blocks


async def test_anthropic_extra_body_thinking_overrides_typed_block_binding(allow_model_requests: None):
    """Retry classification and beta selection follow the final wire object after `extra_body` wins."""
    mock_client = MockAnthropic.create_mock(
        [
            stale_thinking_block_error(),
            completion_message(
                [BetaTextBlock(text='4', type='text')], usage=BetaUsage(input_tokens=10, output_tokens=1)
            ),
        ]
    )
    settings = AnthropicModelSettings(
        anthropic_thinking={'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'error'}},
        extra_body={'thinking': {'display': 'updates'}},
    )
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.warns(AnthropicStaleThinkingBlockWarning):
        await Agent(m, model_settings=settings).run('What is 2+2?')

    first, retried = get_mock_chat_completion_kwargs(mock_client)
    assert _THINKING_BINDING_BETA not in sent_betas(mock_client)
    assert first['extra_body'] == snapshot({'thinking': {'display': 'updates'}})
    assert retried['extra_body'] == snapshot(
        {'thinking': {'display': 'updates', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}
    )
    assert _THINKING_BINDING_BETA in retried['betas']


async def test_anthropic_failed_stale_thinking_retry_does_not_warn_that_run_continued(allow_model_requests: None):
    """The recovery warning is emitted only after the retry has succeeded."""
    retry_error = APIStatusError(
        'service unavailable',
        response=httpx2.Response(status_code=503, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'type': 'error', 'error': {'type': 'api_error', 'message': 'Service unavailable'}},
    )
    mock_client = MockAnthropic.create_mock([stale_thinking_block_error(), retry_error])
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.filterwarnings('always', category=AnthropicStaleThinkingBlockWarning)
        with pytest.raises(ModelHTTPError) as exc_info:
            await Agent(m).run('What is 2+2?')

    assert exc_info.value.status_code == 503
    assert not caught_warnings


@pytest.mark.parametrize(
    'model_name,asks_to_fail',
    [
        pytest.param('claude-fable-5', False, id='model_does_not_bind'),
        pytest.param('claude-fable-5-1', True, id='caller_asked_to_fail'),
    ],
)
async def test_anthropic_does_not_retry_a_stale_thinking_block(
    allow_model_requests: None, model_name: str, asks_to_fail: bool
):
    """The retry is scoped: a model that doesn't bind can't produce this, and an explicit
    `'error'` is a caller asking to fail rather than to lose reasoning."""
    settings = (
        AnthropicModelSettings(
            anthropic_thinking={'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'error'}}
        )
        if asks_to_fail
        else None
    )
    mock_client = MockAnthropic.create_mock(stale_thinking_block_error())
    m = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.raises(ModelHTTPError, match='bound to a different conversation'):
        await Agent(m, model_settings=settings).run('What is 2+2?')

    assert len(get_mock_chat_completion_kwargs(mock_client)) == 1


async def test_anthropic_does_not_retry_an_unrelated_bad_request(allow_model_requests: None):
    """Only the binding rejection is retried; every other 400 propagates unchanged."""
    mock_client = MockAnthropic.create_mock(
        APIStatusError(
            'bad request',
            response=httpx2.Response(status_code=400, request=httpx2.Request('POST', 'https://example.com/v1')),
            body={'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'max_tokens: too large'}},
        )
    )
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with pytest.raises(ModelHTTPError, match='max_tokens: too large'):
        await Agent(m).run('What is 2+2?')

    assert len(get_mock_chat_completion_kwargs(mock_client)) == 1


@pytest.mark.parametrize('model_name', ['claude-fable-5-1', 'claude-fable-5'])
async def test_anthropic_explicit_block_binding_is_preserved(allow_model_requests: None, model_name: str):
    """An explicit `block_binding` is sent as given on every model, with the beta the field needs.

    Without the beta the field is a 400 (`Extra inputs are not permitted`), so the beta follows the
    field rather than the profile flag — otherwise the documented opt-in would not work.
    """
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='4', type='text')], usage=BetaUsage(input_tokens=10, output_tokens=1))
    )
    settings = AnthropicModelSettings(
        anthropic_thinking={'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}
    )
    m = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=mock_client))

    await Agent(m, model_settings=settings).run('What is 2+2?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['thinking']['block_binding'] == snapshot({'prefix_mismatch_behavior': 'drop_block'})
    assert _THINKING_BINDING_BETA in sent_betas(mock_client)


async def test_anthropic_null_block_binding_is_preserved(allow_model_requests: None):
    """`block_binding: None` is how a caller asks for Anthropic's account default explicitly.

    Live-verified against a pre-2026-08-31 account: the field goes out as `null` and the stale block
    is replayed intact, with no `input_transformations` reported.
    """
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='4', type='text')], usage=BetaUsage(input_tokens=10, output_tokens=1))
    )
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'adaptive', 'block_binding': None})
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    await Agent(m, model_settings=settings).run('What is 2+2?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['thinking'] == snapshot({'type': 'adaptive', 'block_binding': None})
    assert _THINKING_BINDING_BETA in sent_betas(mock_client)


async def test_anthropic_empty_block_binding_still_gets_the_beta(allow_model_requests: None):
    """`block_binding: {}` means "every binding default" and needs the beta just as much.

    Live-verified: without `thinking-binding-controls-2026-08-01` the empty mapping is
    `400 thinking.adaptive.block_binding: Extra inputs are not permitted`, and 200 with it. So the
    beta is attached on membership, never on truthiness.
    """
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='4', type='text')], usage=BetaUsage(input_tokens=10, output_tokens=1))
    )
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'adaptive', 'block_binding': {}})
    m = AnthropicModel('claude-sonnet-5', provider=AnthropicProvider(anthropic_client=mock_client))

    await Agent(m, model_settings=settings).run('What is 2+2?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['thinking'] == snapshot({'type': 'adaptive', 'block_binding': {}})
    assert _THINKING_BINDING_BETA in sent_betas(mock_client)


def dropped_thinking_transformation(path: str = 'messages.1.content.0') -> BetaThinkingDroppedInputTransformation:
    return BetaThinkingDroppedInputTransformation(path=path, reason='prefix_binding_mismatch', type='thinking_dropped')


def dropped_thinking_stream(
    start_transformation: BetaThinkingDroppedInputTransformation | None = None,
    delta_transformations: list[BetaThinkingDroppedInputTransformation] | None = None,
) -> list[BetaRawMessageStreamEvent]:
    return [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-fable-5-1',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=5, output_tokens=0),
                input_transformations=[start_transformation] if start_transformation else None,
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start', index=0, content_block=BetaTextBlock(type='text', text='4')
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=5, output_tokens=1),
            input_transformations=delta_transformations or None,
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]


async def test_anthropic_records_dropped_thinking_blocks(allow_model_requests: None):
    """A dropped block is otherwise invisible: the model just answered without the reasoning we sent."""
    response = BetaMessage(
        id='123',
        content=[BetaTextBlock(text='4', type='text')],
        model='claude-fable-5-1',
        role='assistant',
        stop_reason='end_turn',
        type='message',
        usage=BetaUsage(input_tokens=5, output_tokens=10),
        input_transformations=[dropped_thinking_transformation()],
    )
    mock_client = MockAnthropic.create_mock(response)
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('What is 2+2?')

    response = message(result.all_messages(), ModelResponse, index=-1)
    assert response.provider_details == snapshot(
        {
            'finish_reason': 'end_turn',
            'input_transformations': [
                {'path': 'messages.1.content.0', 'reason': 'prefix_binding_mismatch', 'type': 'thinking_dropped'}
            ],
        }
    )


async def test_anthropic_records_dropped_thinking_blocks_streamed(allow_model_requests: None):
    """A live stream reports the drop on `message_start`, before any content arrives."""
    mock_client = MockAnthropic.create_stream_mock(
        dropped_thinking_stream(start_transformation=dropped_thinking_transformation())
    )
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('What is 2+2?') as result:
        await result.get_output()

    response = message(result.all_messages(), ModelResponse, index=-1)
    assert response.provider_details == snapshot(
        {
            'finish_reason': 'end_turn',
            'input_transformations': [
                {'path': 'messages.1.content.0', 'reason': 'prefix_binding_mismatch', 'type': 'thinking_dropped'}
            ],
        }
    )


async def test_anthropic_dropped_thinking_blocks_from_message_delta_replace_message_start(
    allow_model_requests: None,
):
    """A `message_delta` report means a mid-stream model fallback, so its array replaces `message_start`'s."""
    mock_client = MockAnthropic.create_stream_mock(
        dropped_thinking_stream(
            start_transformation=dropped_thinking_transformation(),
            delta_transformations=[dropped_thinking_transformation('messages.3.content.0')],
        )
    )
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('What is 2+2?') as result:
        await result.get_output()

    response = message(result.all_messages(), ModelResponse, index=-1)
    assert response.provider_details == snapshot(
        {
            'finish_reason': 'end_turn',
            'input_transformations': [
                {'path': 'messages.3.content.0', 'reason': 'prefix_binding_mismatch', 'type': 'thinking_dropped'},
            ],
        }
    )


async def test_anthropic_dropped_thinking_blocks_from_message_delta_are_not_duplicated(
    allow_model_requests: None,
):
    """The serving model's array repeats the request-side entries `message_start` already reported."""
    mock_client = MockAnthropic.create_stream_mock(
        dropped_thinking_stream(
            start_transformation=dropped_thinking_transformation(),
            delta_transformations=[
                dropped_thinking_transformation(),
                dropped_thinking_transformation('messages.3.content.0'),
            ],
        )
    )
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('What is 2+2?') as result:
        await result.get_output()

    response = message(result.all_messages(), ModelResponse, index=-1)
    assert response.provider_details == snapshot(
        {
            'finish_reason': 'end_turn',
            'input_transformations': [
                {'path': 'messages.1.content.0', 'reason': 'prefix_binding_mismatch', 'type': 'thinking_dropped'},
                {'path': 'messages.3.content.0', 'reason': 'prefix_binding_mismatch', 'type': 'thinking_dropped'},
            ],
        }
    )


def dropped_thinking_span_events(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    return [event for span in capfire.exporter.exported_spans_as_dict() for event in span.get('events', [])]


async def test_anthropic_dropped_thinking_blocks_reach_the_trace(allow_model_requests: None, capfire: CaptureLogfire):
    """`provider_details` is only readable after the run; the span event puts the drop in the trace."""
    mock_client = MockAnthropic.create_mock(
        BetaMessage(
            id='123',
            content=[BetaTextBlock(text='4', type='text')],
            model='claude-fable-5-1',
            role='assistant',
            stop_reason='end_turn',
            type='message',
            usage=BetaUsage(input_tokens=5, output_tokens=10),
            input_transformations=[dropped_thinking_transformation()],
        )
    )
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    await Agent(InstrumentedModel(m)).run('What is 2+2?')

    assert dropped_thinking_span_events(capfire) == snapshot(
        [
            {
                'name': 'anthropic.input_transformations',
                'timestamp': IsInt(),
                'attributes': {
                    'anthropic.input_transformations': '[{"path":"messages.1.content.0","reason":"prefix_binding_mismatch","type":"thinking_dropped"}]'
                },
            }
        ]
    )


async def test_anthropic_dropped_thinking_blocks_reach_the_trace_streamed(
    allow_model_requests: None, capfire: CaptureLogfire
):
    """The streamed report arrives mid-iteration, so it has to land while the request span is open."""
    mock_client = MockAnthropic.create_stream_mock(
        dropped_thinking_stream(start_transformation=dropped_thinking_transformation())
    )
    m = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    async with Agent(InstrumentedModel(m)).run_stream('What is 2+2?') as result:
        await result.get_output()

    assert dropped_thinking_span_events(capfire) == snapshot(
        [
            {
                'name': 'anthropic.input_transformations',
                'timestamp': IsInt(),
                'attributes': {
                    'anthropic.input_transformations': '[{"path":"messages.1.content.0","reason":"prefix_binding_mismatch","type":"thinking_dropped"}]'
                },
            }
        ]
    )


async def test_anthropic_dropped_thinking_blocks_do_not_reach_an_unrelated_ambient_span(
    allow_model_requests: None, capfire: CaptureLogfire
):
    """Provider response parsing must not mutate a caller's arbitrary active span."""
    mock_client = MockAnthropic.create_mock(
        BetaMessage(
            id='123',
            content=[BetaTextBlock(text='4', type='text')],
            model='claude-fable-5-1',
            role='assistant',
            stop_reason='end_turn',
            type='message',
            usage=BetaUsage(input_tokens=5, output_tokens=10),
            input_transformations=[dropped_thinking_transformation()],
        )
    )
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))

    with trace.get_tracer(__name__).start_as_current_span('ambient'):
        await Agent(model).run('What is 2+2?')

    assert dropped_thinking_span_events(capfire) == []


_STALE_THINKING_BLOCK_PREFIX_CHANGE = pytest.mark.moves_cache_prefix(
    reason='the changed instructions string is what invalidates the thinking block'
)


async def stale_thinking_block_history(model: AnthropicModel) -> list[ModelMessage]:
    """A conversation whose thinking block is bound to a prefix the next request will not match."""
    agent = Agent(model, instructions='You are a helpful assistant. Answer briefly.')
    result = await agent.run('Think about it, then say what 17*23 is.')
    thought = message(result.all_messages(), ModelResponse, index=-1)
    assert any(isinstance(part, ThinkingPart) for part in thought.parts), 'no thinking block to invalidate'
    return result.all_messages()


@_STALE_THINKING_BLOCK_PREFIX_CHANGE
@pytest.mark.vcr()
async def test_anthropic_fable_5_1_replays_a_stale_thinking_block_on_a_legacy_account(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """Pins the account-age carve-out this PR's default depends on.

    Anthropic enforces the prefix check for accounts created on or after 2026-08-31; for older ones
    it "records the mismatch but acts on it only when the request sets
    `thinking.block_binding.prefix_mismatch_behavior`". Pydantic AI sets nothing by default, so on
    the legacy account this cassette was recorded against the replay succeeds untouched: no 400, and
    no `input_transformations`, meaning the model still saw the reasoning.

    The response half alone would not pin that: the default matchers ignore the request body, so the
    recorded absence of `input_transformations` replays even if the default started asking for the
    drop. Setting nothing is the claim, so the outbound `thinking` and the beta header are what the
    assertion has to reach.

    https://platform.claude.com/docs/en/models/fable-5-1/whats-new-fable-5-1#editing-earlier-turns-invalidates-thinking-blocks
    """
    m = anthropic_model('claude-fable-5-1', capture=True)
    history = await stale_thinking_block_history(m)

    second = Agent(m, instructions='You are a helpful assistant. Answer briefly. Today is 2026-09-01.')
    replayed = await second.run('And times two?', message_history=history)

    response = message(replayed.all_messages(), ModelResponse, index=-1)
    assert response.provider_details == snapshot({'finish_reason': 'end_turn'})
    assert 'thinking' not in request_capture.bodies('/v1/messages')[-1]
    assert 'thinking-binding-controls-2026-08-01' not in request_capture.headers[-1].get('anthropic-beta', '')


@_STALE_THINKING_BLOCK_PREFIX_CHANGE
@pytest.mark.vcr()
async def test_anthropic_fable_5_1_drops_a_stale_thinking_block(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """Asking for `drop_block` drops the stale block and lets the run continue.

    This is the shape Pydantic AI retries with after Anthropic rejects a replay, and the shape a
    caller sets to skip that rejected request altogether.

    The response half alone would not pin the claim: the default matchers ignore the request body,
    so the recorded `thinking_dropped` replays even if the setting stopped reaching the wire. The
    outbound body and the beta header are what tie the transformation to what we actually sent.
    """
    m = anthropic_model('claude-fable-5-1', capture=True)
    history = await stale_thinking_block_history(m)

    settings = AnthropicModelSettings(
        anthropic_thinking={'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}
    )
    second = Agent(
        m, instructions='You are a helpful assistant. Answer briefly. Today is 2026-09-01.', model_settings=settings
    )
    replayed = await second.run('And times two?', message_history=history)

    response = message(replayed.all_messages(), ModelResponse, index=-1)
    assert response.provider_details == snapshot(
        {
            'finish_reason': 'end_turn',
            'input_transformations': [
                {'path': 'messages.1.content.0', 'reason': 'prefix_binding_mismatch', 'type': 'thinking_dropped'}
            ],
        }
    )
    dropping_request = request_capture.bodies('/v1/messages')[-1]
    assert dropping_request['thinking'] == snapshot(
        {'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}
    )
    assert 'thinking-binding-controls-2026-08-01' in request_capture.headers[-1]['anthropic-beta']


@_STALE_THINKING_BLOCK_PREFIX_CHANGE
@pytest.mark.vcr()
async def test_anthropic_fable_5_1_drops_a_stale_thinking_block_streamed(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """The same drop over a stream, where Anthropic reports it on `message_start`.

    Recorded rather than mocked because the event carrying `input_transformations` is the whole
    point: a `message_delta` never carries one on a live stream. The outbound body and beta header
    are asserted for the same reason as in the non-streamed sibling: without them the recorded
    `thinking_dropped` replays even if the setting stopped reaching the wire.
    """
    m = anthropic_model('claude-fable-5-1', capture=True)
    history = await stale_thinking_block_history(m)

    settings = AnthropicModelSettings(
        anthropic_thinking={'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}
    )
    second = Agent(
        m, instructions='You are a helpful assistant. Answer briefly. Today is 2026-09-01.', model_settings=settings
    )
    async with second.run_stream('And times two?', message_history=history) as streamed:
        await streamed.get_output()

    response = message(streamed.all_messages(), ModelResponse, index=-1)
    assert response.provider_details == snapshot(
        {
            'finish_reason': 'end_turn',
            'input_transformations': [
                {'path': 'messages.1.content.0', 'reason': 'prefix_binding_mismatch', 'type': 'thinking_dropped'}
            ],
        }
    )
    dropping_request = request_capture.bodies('/v1/messages')[-1]
    assert dropping_request['thinking'] == snapshot(
        {'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}
    )
    assert 'thinking-binding-controls-2026-08-01' in request_capture.headers[-1]['anthropic-beta']
