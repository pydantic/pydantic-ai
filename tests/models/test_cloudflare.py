"""Recorded Workers AI chat contracts through the public Agent interface."""

import os

import pytest

from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.models import infer_model
from pydantic_ai.providers.cloudflare import CloudflareProvider

from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.cloudflare import CloudflareChatModel

pytestmark = [
    pytest.mark.usefixtures('allow_model_requests'),
    pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK not installed'),
    pytest.mark.vcr,
]


@pytest.fixture
def cloudflare_provider(request_capture: RequestCapture) -> CloudflareProvider:
    return CloudflareProvider(
        api_key=os.getenv('CLOUDFLARE_API_TOKEN', 'test-token'),
        account_id=os.getenv('CLOUDFLARE_ACCOUNT_ID', 'test-account'),
        http_client=request_capture.http_client(timeout=120),
    )


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize(
    'model_name',
    [
        '@cf/meta/llama-4-scout-17b-16e-instruct',
        '@cf/qwen/qwen3-30b-a3b-fp8',
        '@cf/openai/gpt-oss-20b',
    ],
)
async def test_chat(
    model_name: str, stream: bool, cloudflare_provider: CloudflareProvider, request_capture: RequestCapture
):
    model = infer_model(f'cloudflare:{model_name}', provider_factory=lambda _: cloudflare_provider)
    assert isinstance(model, CloudflareChatModel)
    agent = Agent(model, instructions='Reply with one short greeting.')
    if stream:
        async with agent.run_stream('Hello') as run:
            output = await run.get_output()
            usage = run.usage
    else:
        result = await agent.run('Hello')
        output = result.output
        usage = result.usage
    assert output
    assert usage.input_tokens > 0
    assert request_capture.body('/chat/completions')['model'] == model_name
    assert request_capture.body('/chat/completions')['stream'] is stream


async def test_tool_roundtrip(cloudflare_provider: CloudflareProvider, request_capture: RequestCapture):
    agent = Agent(CloudflareChatModel('@cf/meta/llama-4-scout-17b-16e-instruct', provider=cloudflare_provider))

    @agent.tool_plain
    def lookup_order(order_id: str) -> str:
        """Look up an order's shipping status."""
        assert order_id == 'A123'
        return 'dispatched'

    result = await agent.run('Use lookup_order to find the shipping status of order A123. Report the result.')
    assert 'dispatched' in result.output.lower()
    bodies = request_capture.bodies('/chat/completions')
    assert len(bodies) >= 2
    assert 'tools' in bodies[0]


async def test_structured_output(cloudflare_provider: CloudflareProvider):
    agent = Agent(
        CloudflareChatModel('@cf/meta/llama-4-scout-17b-16e-instruct', provider=cloudflare_provider),
        output_type=bool,
        instructions='Return true if the message reports an outage, otherwise false.',
    )
    result = await agent.run('Production is down; all API calls return 503.')
    assert result.output is True


async def test_vision(cloudflare_provider: CloudflareProvider, image_content: BinaryContent):
    agent = Agent(CloudflareChatModel('@cf/meta/llama-4-scout-17b-16e-instruct', provider=cloudflare_provider))
    result = await agent.run(['Describe this image in one sentence.', image_content])
    assert result.output


async def test_json_mode(cloudflare_provider: CloudflareProvider):
    model = CloudflareChatModel(
        '@cf/meta/llama-4-scout-17b-16e-instruct',
        provider=cloudflare_provider,
    )
    agent = Agent(model, output_type=NativeOutput(bool))
    result = await agent.run('Return true.')
    assert result.output is True
