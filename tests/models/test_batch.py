"""`BatchModel` and the batch provider contract, outside any durable engine.

The engine paths (queueing, collection, cancellation) are tested through real Temporal workflows in
`tests/durable_exec/temporal/test_batch.py`. These tests pin what holds without one: validation, the
errors raised before anything is queued, and the parts of
the Anthropic codec that involve no network call.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from pydantic_ai import Agent, ModelRequest, UserPromptPart
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.batch import BatchModel, BatchResult, BatchSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import

with try_import() as anthropic_available:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider


class BatchCapableTestModel(TestModel):
    @property
    def batch_max_requests(self) -> int:
        return 100_000


@pytest.mark.parametrize(
    'settings',
    [
        {'max_requests': 0},
        {'max_wait': timedelta(0)},
        {'poll_interval': timedelta(seconds=-1)},
    ],
)
def test_batch_settings_reject_non_positive_values(settings: dict[str, object]) -> None:
    with pytest.raises(UserError, match='must be positive'):
        BatchModel(BatchCapableTestModel(), batch_settings=BatchSettings(**settings))  # pyright: ignore[reportArgumentType]


def test_batch_model_needs_a_model_with_a_batch_api() -> None:
    with pytest.raises(UserError, match=r"TestModel 'test' has no batch API."):
        BatchModel(TestModel())


async def test_batch_model_fails_outside_a_durable_context_before_queueing() -> None:
    agent = Agent(BatchModel(BatchCapableTestModel()))

    with pytest.raises(UserError, match='need a durable execution context with batch support'):
        await agent.run('Hello')


async def test_batch_model_rejects_streaming() -> None:
    agent = Agent(BatchModel(BatchCapableTestModel()))

    with pytest.raises(UserError, match='`BatchModel` does not support streaming'):
        async with agent.run_stream('Hello'):
            pass  # pragma: no cover


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
def test_anthropic_batch_api_is_only_on_the_claude_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('AWS_BEARER_TOKEN_BEDROCK', raising=False)
    direct = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))
    bedrock = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(
            anthropic_client=AsyncAnthropicBedrock(aws_region='us-east-1', aws_access_key='a', aws_secret_key='b')
        ),
    )

    assert direct.batch_max_requests == 100_000
    assert bedrock.batch_max_requests is None
    with pytest.raises(UserError, match='has no batch API'):
        BatchModel(bedrock)


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_batch_request_carries_the_messages_params_without_secrets() -> None:
    """The codec runs the same request builder as `AnthropicModel.request`, and its group never holds the key.

    Not a VCR test: encoding makes no request. Submission and results are recorded in the Temporal suite.
    """
    model = AnthropicModel(
        'claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key='secret-key'))
    )
    assert model is not None

    request = await model.batch_encode(
        [ModelRequest(parts=[UserPromptPart('Hello')])],
        {'max_tokens': 100, 'stop_sequences': ['END']},
        ModelRequestParameters(),
        custom_id='item1',
    )

    body = json.loads(request.body)
    assert body['model'] == 'claude-sonnet-4-5'
    assert body['max_tokens'] == 100
    assert body['stop_sequences'] == ['END']
    assert body['messages'][0]['role'] == 'user'
    assert 'secret-key' not in request.group
    group = json.loads(request.group)
    assert group['betas'] == []
    assert 'api.anthropic.com' not in request.group


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_batch_rejects_per_request_headers() -> None:
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))
    assert model is not None

    with pytest.raises(UserError, match='`extra_headers` are not supported for `BatchModel` requests'):
        await model.batch_encode(
            [ModelRequest(parts=[UserPromptPart('Hello')])],
            {'extra_headers': {'X-Tenant': 'a'}},
            ModelRequestParameters(),
            custom_id='item1',
        )


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.parametrize('status', ['expired', 'canceled'])
def test_anthropic_batch_request_that_never_ran_is_a_model_error(status: str) -> None:
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))
    assert model is not None

    with pytest.raises(ModelAPIError, match=f'The batch request was {status} before the model ran it.'):
        model.batch_decode(
            BatchResult(custom_id='item1', status=status),  # pyright: ignore[reportArgumentType]
            None,
            ModelRequestParameters(),
        )


def test_an_engine_without_batch_support_rejects_the_model() -> None:
    from tests.durable_exec.test_durable_exec_compat import JournalDurability

    with pytest.raises(UserError, match='Journal compatibility stub does not provide'):
        Agent(BatchModel(BatchCapableTestModel()), name='unsupported-batch-agent', capabilities=[JournalDurability()])


async def test_an_engine_without_batch_support_rejects_a_runtime_model() -> None:
    from pydantic_ai.capabilities import ResolveModelId
    from tests.durable_exec.test_durable_exec_compat import JournalDurability

    agent = Agent(
        TestModel(),
        name='unsupported-runtime-batch',
        capabilities=[ResolveModelId(lambda ctx, model_id: BatchModel(BatchCapableTestModel())), JournalDurability()],
    )
    with pytest.raises(UserError, match='Journal compatibility stub does not provide'):
        await agent.run('hello', model='test:runtime-batch')


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_batch_refuses_a_different_registered_client(allow_model_requests: None) -> None:
    first = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='first-key'))
    second = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='second-key'))
    assert first is not None and second is not None
    request = await first.batch_encode(
        [ModelRequest(parts=[UserPromptPart('hello')])], None, ModelRequestParameters(), custom_id='item'
    )
    with pytest.raises(UserError, match='does not match the batch request group'):
        await second.batch_create([request])
    with pytest.raises(UserError, match='does not match the batch request group'):
        await second.batch_status('batch', group=request.group)
    with pytest.raises(UserError, match='does not match the batch request group'):
        await second.batch_results('batch', group=request.group)
    with pytest.raises(UserError, match='does not match the batch request group'):
        await second.batch_cancel('batch', group=request.group)
    other = await second.batch_encode(
        [ModelRequest(parts=[UserPromptPart('hello')])], None, ModelRequestParameters(), custom_id='other'
    )
    with pytest.raises(UserError, match='does not match the batch request group'):
        await first.batch_create([request, other])
    with pytest.raises(UserError, match='at least one request'):
        await first.batch_create([])


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.parametrize(
    'settings',
    [
        {'anthropic_speed': 'fast'},
        {'max_tokens': 0},
        {'timeout': 10},
        {'extra_body': {'speed': 'standard'}},
        {'extra_body': {'stream': True}},
    ],
)
async def test_anthropic_batch_rejects_synchronous_only_parameters(settings: ModelSettings) -> None:
    provider = AnthropicModel('claude-opus-4-6', provider=AnthropicProvider(api_key='test'))
    assert provider is not None
    with pytest.raises(UserError, match='not support'):
        await provider.batch_encode(
            [ModelRequest(parts=[UserPromptPart('hello')])], settings, ModelRequestParameters(), custom_id='item'
        )


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_batch_preserves_supported_settings() -> None:
    from pydantic_ai.models.anthropic import AnthropicModelSettings
    from pydantic_ai.tools import ToolDefinition

    provider = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test'))
    assert provider is not None
    request = await provider.batch_encode(
        [ModelRequest(parts=[UserPromptPart('hello')], instructions='Answer briefly')],
        AnthropicModelSettings(
            max_tokens=100,
            anthropic_metadata={'user_id': 'user'},
            anthropic_service_tier='standard_only',
            anthropic_cache=True,
            anthropic_cache_diagnostics=True,
            extra_body={'temperature': 0.5},
        ),
        ModelRequestParameters(
            function_tools=[ToolDefinition(name='weather', parameters_json_schema={'type': 'object', 'properties': {}})]
        ),
        custom_id='item',
    )
    body = json.loads(request.body)
    assert body['metadata'] == {'user_id': 'user'}
    assert body['service_tier'] == 'standard_only'
    assert body['diagnostics'] == {'previous_message_id': None}
    assert body['temperature'] == 0.5
    assert body['system'][0]['text'] == 'Answer briefly'
    assert body['tools'][0]['name'] == 'weather'


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
def test_anthropic_batch_requires_static_credentials() -> None:
    client = AsyncAnthropic(api_key='')
    with pytest.raises(UserError, match='requires static Anthropic API credentials'):
        BatchModel(AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client)))


async def test_wrapped_non_batch_model_keeps_durable_execution() -> None:
    from pydantic_ai.models.wrapper import WrapperModel
    from tests.durable_exec.test_durable_exec_compat import JournalDurability

    result = await Agent(WrapperModel(TestModel()), name='wrapped-plain', capabilities=[JournalDurability()]).run(
        'hello'
    )
    assert isinstance(result.output, str)


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_direct_batch_methods_reject_a_bedrock_client(
    monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
) -> None:
    monkeypatch.delenv('AWS_BEARER_TOKEN_BEDROCK', raising=False)
    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(
            anthropic_client=AsyncAnthropicBedrock(aws_region='us-east-1', aws_access_key='a', aws_secret_key='b')
        ),
    )
    with pytest.raises(UserError, match='does not support Message Batches'):
        await model.batch_status('batch')
