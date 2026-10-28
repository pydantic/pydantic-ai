"""Following a Pydantic AI Gateway fallback to a model in another API format.

These tests talk to a fake gateway on an `httpx2.MockTransport` rather than replaying cassettes: the gateway doesn't
hand requests back yet, so there is no real exchange to record. The fake answers the way the gateway's protocol
specifies (a `424` naming the next `gateway/` model and a state token), and serves real provider response shapes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import httpx2
import pytest

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart
from pydantic_ai.capabilities import AbstractCapability, BatchModelRequests, ModelFallback
from pydantic_ai.capabilities._gateway_fallback import GatewayFallback
from pydantic_ai.durable_exec._utils import reject_serving_wrapper
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel

from ._inline_snapshot import snapshot
from .batch_utils import FAST_BATCH_SETTINGS
from .conftest import try_import
from .durable_exec.test_durable_exec_compat import JournalDurability

with try_import() as imports_successful:
    import google.genai  # noqa: F401  # advertised as `gemini`, so `ADVERTISED` needs it
    from botocore.awsrequest import AWSResponse, HTTPHeaders

    # Imported here rather than when the gateway hands a request to an Anthropic model, which is in the event loop.
    from pydantic_ai.models.anthropic import AnthropicModel as AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.profiles import merge_profile
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.gateway import (
        _fallback_provider,  # pyright: ignore[reportPrivateUsage]
        _shareable_http_client,  # pyright: ignore[reportPrivateUsage]
        gateway_provider,
        normalize_gateway_provider,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

if not imports_successful():
    pytest.skip(
        'openai, anthropic, google-genai or boto3 not installed', allow_module_level=True
    )  # pragma: lax no cover


API_KEY = 'pylf_v1_us_test'
BASE_URL = 'https://gateway.example/proxy'

Handler = Callable[[httpx2.Request], httpx2.Response]


def hand_back(model: str, state: str) -> httpx2.Response:
    """The gateway's hand-back: the request should be sent to `model` instead."""
    return httpx2.Response(
        424,
        headers={
            'pydantic-ai-gateway-fallback-model': model,
            'pydantic-ai-gateway-fallback-state': state,
            'x-should-retry': 'false',
        },
        json={'error': {'code': 'pydantic_ai_fallback', 'message': f'Upstream failed (503); retry with {model}'}},
    )


def unavailable() -> httpx2.Response:
    return httpx2.Response(
        503,
        headers={'x-should-retry': 'false'},
        json={'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'All providers failed'}},
    )


def anthropic_message(request: httpx2.Request, text: str) -> httpx2.Response:
    """An Anthropic Messages response, streamed when the request asks for it, as `AnthropicModel` always does."""
    message: dict[str, Any] = {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-5',
        'content': [],
        'stop_reason': None,
        'stop_sequence': None,
        'usage': {'input_tokens': 10, 'output_tokens': 1},
    }
    events: list[tuple[str, dict[str, Any]]] = [
        ('message_start', {'type': 'message_start', 'message': message}),
        (
            'content_block_start',
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        ),
        (
            'content_block_delta',
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}},
        ),
        ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
        (
            'message_delta',
            {
                'type': 'message_delta',
                'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
                'usage': {'output_tokens': 5},
            },
        ),
        ('message_stop', {'type': 'message_stop'}),
    ]
    assert json.loads(request.content)['stream']
    body = ''.join(f'event: {event}\ndata: {json.dumps(data)}\n\n' for event, data in events)
    return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=body.encode())


def openai_response(text: str) -> dict[str, Any]:
    """A Responses API `response` object."""
    return {
        'id': 'resp_1',
        'object': 'response',
        'created_at': 1761600000,
        'model': 'gpt-5',
        'status': 'completed',
        'output': [
            {
                'type': 'message',
                'id': 'msg_1',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': text, 'annotations': []}],
            }
        ],
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'usage': {
            'input_tokens': 10,
            'input_tokens_details': {'cached_tokens': 0},
            'output_tokens': 5,
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': 15,
        },
    }


@dataclass
class SeenRequest:
    path: str
    fallback_headers: dict[str, str]


@dataclass
class FakeGateway:
    """Serves requests with its handlers in order, each for a path ending in its suffix, and records what reached it."""

    handlers: list[tuple[str, Handler]]
    seen: list[SeenRequest] = field(default_factory=list[SeenRequest])

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert request.headers['authorization'] == f'Bearer {API_KEY}'
        self.seen.append(
            SeenRequest(
                request.url.path,
                {name: value for name, value in request.headers.items() if name.startswith('x-pydantic-ai-fallback')},
            )
        )
        suffix, handler = self.handlers.pop(0)
        assert request.url.path.endswith(suffix), f'{request.method} {request.url.path}, expected {suffix}'
        return handler(request)

    def openai_model(
        self, *, route: str | None = None, profile: OpenAIModelProfile | None = None
    ) -> OpenAIResponsesModel:
        """A `gateway/openai-responses:gpt-5` model whose provider talks to this fake gateway."""
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(self))
        provider = gateway_provider(
            'openai-responses', route=route, api_key=API_KEY, base_url=BASE_URL, http_client=client
        )
        merged = None if profile is None else merge_profile(provider.model_profile('gpt-5'), profile)
        return OpenAIResponsesModel('gpt-5', provider=provider, profile=merged)


ADVERTISED = {'x-pydantic-ai-fallback': '1; formats=openai,anthropic,gemini,bedrock-converse'}


@pytest.mark.parametrize('stream', [False, True])
async def test_the_gateway_sends_the_request_to_a_model_in_another_format(allow_model_requests: None, stream: bool):
    """The request goes to the model the gateway names, on the same route with the same key, carrying its state."""
    gateway = FakeGateway(
        [
            ('/responses', lambda _: hand_back('gateway/anthropic:claude-sonnet-4-5', 'state-1')),
            ('/v1/messages', lambda request: anthropic_message(request, 'Hello from Claude')),
        ]
    )
    agent = Agent(gateway.openai_model(route='my-endpoint'))

    if stream:
        async with agent.run_stream('Hello') as result:
            output = await result.get_output()
        messages = result.all_messages()
    else:
        run = await agent.run('Hello')
        output, messages = run.output, run.all_messages()

    assert output == 'Hello from Claude'
    assert gateway.seen == snapshot(
        [
            SeenRequest(path='/proxy/my-endpoint/responses', fallback_headers=ADVERTISED),
            SeenRequest(
                path='/proxy/my-endpoint/v1/messages',
                fallback_headers={**ADVERTISED, 'x-pydantic-ai-fallback-state': 'state-1'},
            ),
        ]
    )
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert (response.model_name, response.provider_name) == ('claude-sonnet-4-5', 'anthropic')
    assert [(attempt.model_name, attempt.outcome) for attempt in response.failed_attempts or []] == [('gpt-5', 'error')]


class PassThroughWrapper(AbstractCapability[Any]):
    """Wraps the model serving each attempt, as a capability that observes requests would."""

    def get_wrapper_model(self, model: Model) -> Model | None:
        return WrapperModel(model)


async def test_a_hand_back_is_followed_under_another_capabilitys_model_wrapper(allow_model_requests: None):
    """A capability's wrapper around the serving model doesn't hide the hand-back, or how many came before it."""
    gateway = FakeGateway(
        [
            ('/responses', lambda _: hand_back('gateway/anthropic:claude-sonnet-4-5', 'state-1')),
            ('/v1/messages', lambda _: hand_back('gateway/openai-responses:gpt-5', 'state-2')),
            ('/responses', lambda _: httpx2.Response(200, json=openai_response('Hello from GPT'))),
        ]
    )
    agent = Agent(gateway.openai_model(), capabilities=[PassThroughWrapper()])

    result = await agent.run('Hello')

    assert result.output == 'Hello from GPT'
    assert [request.fallback_headers.get('x-pydantic-ai-fallback-state') for request in gateway.seen] == [
        None,
        'state-1',
        'state-2',
    ]


def client_side_fallback(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('Hello from the client-side fallback')])


async def test_the_agents_own_fallback_takes_over_once_the_gateways_chain_fails(allow_model_requests: None):
    """`ModelFallback` only moves on once the model the gateway handed the request to fails without a hand-back."""
    gateway = FakeGateway(
        [
            ('/responses', lambda _: hand_back('gateway/anthropic:claude-sonnet-4-5', 'state-1')),
            ('/v1/messages', lambda _: unavailable()),
        ]
    )
    agent = Agent(
        gateway.openai_model(),
        capabilities=[ModelFallback(FunctionModel(client_side_fallback))],
    )

    result = await agent.run('Hello')

    assert result.output == 'Hello from the client-side fallback'
    assert [request.path for request in gateway.seen] == ['/proxy/openai/responses', '/proxy/openai/v1/messages']
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [(attempt.model_name, attempt.outcome) for attempt in response.failed_attempts or []] == [
        ('gpt-5', 'error'),
        ('claude-sonnet-4-5', 'error'),
    ]


async def test_a_step_follows_the_gateway_at_most_five_times(allow_model_requests: None):
    """A gateway that keeps handing the request back gets its error raised after five hand-backs."""
    models = ['gateway/anthropic:claude-sonnet-4-5', 'gateway/openai-responses:gpt-5']
    gateway = FakeGateway(
        [
            (path, lambda _, model=models[index % 2]: hand_back(model, 'state'))
            for index, path in enumerate(['/responses', '/v1/messages'] * 3)
        ]
    )
    agent = Agent(gateway.openai_model())

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Hello')

    assert exc_info.value.status_code == 424
    assert len(gateway.seen) == 6


@pytest.mark.parametrize(
    'response',
    [
        pytest.param(lambda: hand_back('anthropic:claude-sonnet-4-5', 'state-1'), id='outside-the-gateway'),
        pytest.param(lambda: hand_back('gateway/anthropic', 'state-1'), id='no-model-name'),
        pytest.param(lambda: httpx2.Response(424, headers={'x-should-retry': 'false'}), id='no-model-header'),
    ],
)
async def test_a_424_that_names_no_gateway_model_is_not_followed(
    allow_model_requests: None, response: Callable[[], httpx2.Response]
):
    """Only a `gateway/<provider>:<model>` is followed: it goes back to the same gateway, with the same credentials."""
    gateway = FakeGateway([('/responses', lambda _: response())])
    agent = Agent(gateway.openai_model())

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Hello')

    assert exc_info.value.status_code == 424
    assert len(gateway.seen) == 1


async def test_a_hand_back_to_a_model_this_install_cant_build_is_the_error(allow_model_requests: None):
    """A model string of a format Pydantic AI doesn't know leaves the hand-back as the request's error."""
    gateway = FakeGateway([('/responses', lambda _: hand_back('gateway/unknown-format:some-model', 'state-1'))])
    agent = Agent(gateway.openai_model())

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Hello')

    assert exc_info.value.status_code == 424
    assert isinstance(exc_info.value.__cause__, UserError)


async def test_requests_to_other_providers_neither_advertise_nor_follow(allow_model_requests: None):
    """A model that doesn't go through the gateway doesn't say it can follow one, nor follow a lookalike response."""
    seen: list[httpx2.Headers] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers)
        return hand_back('gateway/anthropic:claude-sonnet-4-5', 'state-1')

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test', http_client=client))

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run('Hello')

    assert exc_info.value.status_code == 424
    assert [name for headers in seen for name in headers if name.startswith('x-pydantic-ai-fallback')] == []


async def test_batched_requests_dont_advertise(allow_model_requests: None):
    """A request sent through the batch API doesn't say it can follow a hand-back, which a batch result can't be.

    The gateway doesn't serve OpenAI batches yet, so the model is given a profile that says it does, as it will.
    """
    custom_ids: list[str] = []

    def upload(request: httpx2.Request) -> httpx2.Response:
        line = next(line for line in request.content.decode().splitlines() if line.startswith('{'))
        custom_ids.append(json.loads(line)['custom_id'])
        return httpx2.Response(
            200,
            json={
                'id': 'file-in',
                'object': 'file',
                'bytes': 1,
                'created_at': 0,
                'filename': 'batch.jsonl',
                'purpose': 'batch',
            },
        )

    def batch(status: str) -> Handler:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                200,
                json={
                    'id': 'batch-1',
                    'object': 'batch',
                    'endpoint': '/v1/responses',
                    'completion_window': '24h',
                    'created_at': 0,
                    'input_file_id': 'file-in',
                    'status': status,
                    **({'output_file_id': 'file-out'} if status == 'completed' else {}),
                },
            )

        return handler

    def results(request: httpx2.Request) -> httpx2.Response:
        line = {
            'id': 'line-1',
            'custom_id': custom_ids[0],
            'response': {'status_code': 200, 'request_id': 'req-1', 'body': openai_response('Hello from a batch')},
            'error': None,
        }
        return httpx2.Response(200, content=json.dumps(line).encode())

    gateway = FakeGateway(
        [
            ('/files', upload),
            ('/batches', batch('validating')),
            ('/batches/batch-1', batch('completed')),
            ('/files/file-out/content', results),
            (
                '/files/file-in',
                lambda _: httpx2.Response(200, json={'id': 'file-in', 'object': 'file', 'deleted': True}),
            ),
        ]
    )
    batched = gateway.openai_model(profile=OpenAIModelProfile(openai_supports_batch_api=True))
    agent = Agent(batched, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    result = await agent.run('Hello')

    assert result.output == 'Hello from a batch'
    assert [request.fallback_headers for request in gateway.seen] == [{}, {}, {}, {}, {}]


async def test_batching_a_gateway_model_without_a_batch_api_names_the_model_class():
    """The error for a model without a batch API names the model, not the wrapper that sends the fallback headers."""
    gateway = FakeGateway([])
    agent = Agent(gateway.openai_model(), capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    with pytest.raises(UserError, match=r"^`OpenAIResponsesModel` 'gpt-5' has no batch API"):
        await agent.run('Hello')


async def test_durable_runs_neither_advertise_nor_follow(allow_model_requests: None):
    """Under durable execution the request is sent from a durable unit, where the next model couldn't be built.

    The gateway then falls back within the request's format, as it does for any other client.
    """
    gateway = FakeGateway([('/responses', lambda _: hand_back('gateway/anthropic:claude-sonnet-4-5', 'state-1'))])
    agent = Agent(gateway.openai_model(), name='durable', capabilities=[JournalDurability()])

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Hello')

    assert exc_info.value.status_code == 424
    assert gateway.seen == [SeenRequest(path='/proxy/openai/responses', fallback_headers={})]


def test_deprecated_durable_agents_dont_refuse_the_headers_wrapper():
    """`TemporalAgent`, `DBOSAgent` and `PrefectAgent` refuse capabilities that wrap the model, which this one sits out.

    Checked on the helper their models call, as the engines' own agents aren't needed to show it.
    """
    model = FakeGateway([]).openai_model()
    agent = Agent(model)
    durable = WrapperModel(model)

    def wrap(model: Model) -> Model:
        return agent.root_capability.get_wrapper_model(model) or model

    assert wrap(model) is not model
    assert reject_serving_wrapper(durable, wrap, 'TemporalAgent', 'TemporalDurability') is durable


@pytest.mark.parametrize(
    'client_type, upstream, shared',
    [
        pytest.param(httpx2.AsyncClient, 'openai-chat', True, id='openai-httpx2'),
        pytest.param(httpx.AsyncClient, 'google', True, id='google-httpx'),
        pytest.param(httpx2.AsyncClient, 'anthropic', True, id='anthropic-httpx2'),
        pytest.param(httpx.AsyncClient, 'anthropic', False, id='anthropic-httpx'),
        pytest.param(httpx.AsyncClient, 'groq', True, id='groq-httpx'),
        pytest.param(httpx2.AsyncClient, 'groq', False, id='groq-httpx2'),
        pytest.param(httpx2.AsyncClient, 'bedrock', False, id='bedrock'),
    ],
)
async def test_the_callers_http_client_is_shared_where_the_sdk_accepts_it(
    client_type: type[httpx.AsyncClient] | type[httpx2.AsyncClient], upstream: str, shared: bool
):
    """Which SDK takes which client is checked directly: the fake gateway only serves clients it gave the provider."""
    async with client_type() as client:
        assert (_shareable_http_client(normalize_gateway_provider(upstream), client) is client) is shared


def test_the_next_provider_is_created_once_per_format():
    """Checked directly, as reusing a provider isn't visible on the wire: the requests go to the same URL either way.

    That the next provider keeps the route and key is tested through the agent, above.
    """
    original = gateway_provider('openai', route='my-endpoint', api_key=API_KEY, base_url=BASE_URL)

    fallback = _fallback_provider(original, 'anthropic')
    assert _fallback_provider(original, 'anthropic') is fallback
    assert _fallback_provider(original, 'bedrock') is not fallback


@contextmanager
def converse_responses(model: BedrockConverseModel, seen: list[dict[str, str]]) -> Generator[None]:
    """Answer Converse requests without sending them, recording the headers of the request that would go out."""

    def send(request: Any, **_: Any) -> AWSResponse:
        seen.append(
            {name.lower(): _text(value) for name, value in request.headers.items() if 'fallback' in name.lower()}
        )
        body = {
            'output': {'message': {'role': 'assistant', 'content': [{'text': 'Hello from Converse'}]}},
            'stopReason': 'end_turn',
            'usage': {'inputTokens': 10, 'outputTokens': 5, 'totalTokens': 15},
            'metrics': {'latencyMs': 1},
        }
        headers = HTTPHeaders()
        headers['content-type'] = 'application/json'
        return AWSResponse(request.url, 200, headers, _RawBody(json.dumps(body).encode()))

    event = 'before-send.bedrock-runtime.Converse'
    # botocore sends the response a `before-send` handler returns instead of the request, though the stubs type every
    # handler as returning `None`.
    model.client.meta.events.register_last(event, send)  # pyright: ignore[reportArgumentType]
    try:
        yield
    finally:
        model.client.meta.events.unregister(event, send)  # pyright: ignore[reportArgumentType]


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


@dataclass
class _RawBody:
    content: bytes

    def stream(self, **_: Any) -> Iterator[bytes]:
        yield self.content


async def test_converse_requests_advertise_too(allow_model_requests: None):
    """Bedrock Converse requests, which go through botocore rather than httpx, say the client can follow a hand-back."""
    provider = gateway_provider('bedrock', api_key=API_KEY, base_url=BASE_URL)
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=provider)
    seen: list[dict[str, str]] = []

    with converse_responses(model, seen):
        result = await Agent(model).run('Hello')
        # Outside an agent run, nothing follows a hand-back, so nothing says so.
        await model.request([ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters())

    assert result.output == 'Hello from Converse'
    assert seen == [ADVERTISED, {}]


def test_the_capability_is_not_in_agent_specs():
    """`GatewayFallback` is auto-injected only, so an `AgentSpec` can't name it."""
    with pytest.raises(
        ValueError, match="Capability 'GatewayFallback' is not in the provided `custom_capability_types`"
    ):
        Agent.from_spec({'model': 'test', 'capabilities': [{'GatewayFallback': {}}]})
    # Nor could it be passed as a custom capability type: it has no name to load it by.
    assert GatewayFallback.get_serialization_name() is None
