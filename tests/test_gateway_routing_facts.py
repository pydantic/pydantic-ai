"""`GatewayRoutingFacts`: the headers an agent sends with each model request that goes through the Pydantic AI Gateway.

The run tests aren't cassette tests. Their claims are about the headers each attempt carries across steps, retries
and fallback attempts, which a recording doesn't hold and its matchers ignore. So each test sends its requests to a
`MockTransport` that plays scripted Chat Completions responses, and reads the headers off the request as it goes out.
`test_gateway_routing_facts_reach_the_wire` in `tests/providers/test_gateway.py` replays a real recording on every
Gateway route.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import httpx2
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import GatewayRoutingFacts, ModelFallback
from pydantic_ai.capabilities.gateway_routing_facts import _request_kind  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryFeedbackPart,
    RetryInfo,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from ._inline_snapshot import snapshot
from .conftest import try_import
from .history_strategies import histories

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def _routing_headers(headers: httpx2.Headers) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.startswith(('x-pydantic-agent-', 'x-routing-'))}


def _completion(text: str | None = None, *, tool_call: tuple[str, str] | None = None) -> dict[str, Any]:
    """A Chat Completions response with `text`, or with one call to the tool named in `tool_call`, with its arguments."""
    message: dict[str, Any] = {'role': 'assistant', 'content': text}
    if tool_call is not None:
        name, arguments = tool_call
        message['tool_calls'] = [
            {'id': 'call_1', 'type': 'function', 'function': {'name': name, 'arguments': arguments}}
        ]
    return {
        'id': 'chatcmpl-1',
        'object': 'chat.completion',
        'created': 0,
        'model': 'gpt-5',
        'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if tool_call else 'stop'}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


_REJECTED = 400
"""A response status that fails the attempt without the SDK retrying it, so a fallback moves on."""


@dataclass
class _Route:
    """A Chat Completions endpoint that plays scripted responses and records the headers of each request."""

    responses: list[dict[str, Any] | int]
    headers: list[httpx2.Headers] = field(default_factory=list[httpx2.Headers])

    @property
    def sent(self) -> list[dict[str, str]]:
        """The routing headers of each request, in order."""
        return [_routing_headers(headers) for headers in self.headers]

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.headers.append(request.headers)
        response = self.responses.pop(0)
        if isinstance(response, int):
            return httpx2.Response(response, json={'error': {'message': 'Rejected.', 'type': 'invalid_request_error'}})
        return httpx2.Response(200, json=response)


RouteFactory = Callable[..., tuple[Model, _Route]]


@pytest.fixture
async def route(allow_model_requests: None) -> AsyncIterator[RouteFactory]:
    """Builds an `OpenAIChatModel` on a scripted route: through the Gateway, or directly to the provider."""
    clients: list[httpx2.AsyncClient] = []

    def build(
        *responses: dict[str, Any] | int,
        gateway: bool = True,
        settings: ModelSettings | None = None,
    ) -> tuple[Model, _Route]:
        scripted = _Route(list(responses))
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(scripted.handle))
        clients.append(client)
        provider = (
            gateway_provider(
                'openai-chat', api_key='gw-key', base_url='https://gateway.example.com', http_client=client
            )
            if gateway
            else OpenAIProvider(api_key='key', base_url='https://api.example.com/v1', http_client=client)
        )
        return OpenAIChatModel('gpt-5', provider=provider, settings=settings), scripted

    yield build
    for client in clients:
        await client.aclose()


async def test_the_headers_follow_the_run(route: RouteFactory):
    """Each step says what it answers: the prompt, a tool retry, tool results, then an output retry."""
    model, gateway = route(
        _completion(tool_call=('lookup', '{}')),
        _completion(tool_call=('lookup', '{}')),
        _completion('Paris'),
        _completion('Paris, France'),
    )
    support_agent = Agent(model)
    lookups: list[None] = []

    @support_agent.tool_plain
    def lookup() -> str:
        lookups.append(None)
        if len(lookups) == 1:
            raise ModelRetry('The lookup service is busy, call it again.')
        return 'Paris is the capital of France.'

    @support_agent.output_validator
    def name_the_country(output: str) -> str:
        if 'France' not in output:
            raise ModelRetry('Name the country too.')
        return output

    result = await support_agent.run('What is the capital of France?', conversation_id='conversation-1')

    assert result.output == 'Paris, France'
    assert gateway.sent == snapshot(
        [
            {
                'x-pydantic-agent-name': 'support_agent',
                'x-pydantic-agent-step': '1',
                'x-pydantic-agent-request-kind': 'user_prompt',
                'x-pydantic-agent-retry': '0',
                'x-routing-conversation-id': 'conversation-1',
            },
            {
                'x-pydantic-agent-name': 'support_agent',
                'x-pydantic-agent-step': '2',
                'x-pydantic-agent-request-kind': 'tool_retry',
                'x-pydantic-agent-retry': '1',
                'x-routing-conversation-id': 'conversation-1',
            },
            {
                'x-pydantic-agent-name': 'support_agent',
                'x-pydantic-agent-step': '3',
                'x-pydantic-agent-request-kind': 'tool_results',
                'x-pydantic-agent-retry': '0',
                'x-routing-conversation-id': 'conversation-1',
            },
            {
                'x-pydantic-agent-name': 'support_agent',
                'x-pydantic-agent-step': '4',
                'x-pydantic-agent-request-kind': 'output_retry',
                'x-pydantic-agent-retry': '1',
                'x-routing-conversation-id': 'conversation-1',
            },
        ]
    )


class City(BaseModel):
    name: str
    country: str


async def test_a_retried_output_tool_is_an_output_retry(route: RouteFactory):
    """An output tool call whose arguments fail validation is answered with an output retry, not a tool retry."""
    model, gateway = route(
        _completion(tool_call=('final_result', '{"name": "Paris"}')),
        _completion(tool_call=('final_result', '{"name": "Paris", "country": "France"}')),
    )
    agent = Agent(model, output_type=City, name='geographer')

    result = await agent.run('What is the capital of France?', conversation_id='conversation-1')

    assert result.output == City(name='Paris', country='France')
    assert [(sent['x-pydantic-agent-request-kind'], sent['x-pydantic-agent-retry']) for sent in gateway.sent] == [
        ('user_prompt', '0'),
        ('output_retry', '1'),
    ]


async def test_headers_the_caller_set_win(route: RouteFactory):
    """A routing header set in `extra_headers`, in any case, is sent as set, and the caller's other headers stay."""
    model, gateway = route(_completion('Paris'))
    agent = Agent(model, name='geographer', model_settings={'extra_headers': {'x-tenant': 'acme'}})

    await agent.run(
        'What is the capital of France?',
        conversation_id='conversation-1',
        model_settings={'extra_headers': {'x-tenant': 'acme', 'X-Pydantic-Agent-Name': 'billing'}},
    )

    [headers] = gateway.headers
    assert headers['x-tenant'] == 'acme'
    assert gateway.sent == snapshot(
        [
            {
                'x-pydantic-agent-name': 'billing',
                'x-pydantic-agent-step': '1',
                'x-pydantic-agent-request-kind': 'user_prompt',
                'x-pydantic-agent-retry': '0',
                'x-routing-conversation-id': 'conversation-1',
            }
        ]
    )


@pytest.mark.parametrize('gateway', [True, False], ids=['gateway', 'direct'])
async def test_only_gateway_requests_carry_the_headers(route: RouteFactory, gateway: bool):
    """A model behind the Gateway sends the headers; one that isn't sends none."""
    model, scripted = route(_completion('Paris'), gateway=gateway)
    agent = Agent(model, name='geographer')

    await agent.run('What is the capital of France?')

    [sent] = scripted.sent
    assert bool(sent) is gateway


async def test_model_fallback_sends_the_headers_only_to_gateway_attempts(route: RouteFactory):
    """`ModelFallback` prepares each attempt for its own model, so only an attempt through the Gateway carries them.

    The Gateway candidate's own `extra_headers` are still sent: the routing headers are merged with them rather
    than replacing them.
    """
    direct, direct_route = route(_REJECTED, gateway=False)
    gateway, gateway_route = route(_completion('Paris'), _REJECTED, settings={'extra_headers': {'x-tenant': 'acme'}})
    fallback_direct, fallback_direct_route = route(_completion('Paris'), gateway=False)

    agent = Agent(direct, name='geographer', capabilities=[ModelFallback(gateway)])
    await agent.run('What is the capital of France?')
    agent = Agent(gateway, name='geographer', capabilities=[ModelFallback(fallback_direct)])
    await agent.run('What is the capital of France?')

    assert direct_route.sent == [{}]
    assert fallback_direct_route.sent == [{}]
    assert [headers['x-tenant'] for headers in gateway_route.headers] == ['acme', 'acme']
    assert [sent.get('x-pydantic-agent-request-kind') for sent in gateway_route.sent] == ['user_prompt', 'user_prompt']


@dataclass(frozen=True)
class FallbackCase:
    id: str
    first_gateway: bool
    first_settings: ModelSettings | None = None
    sends_headers: bool = True


FALLBACK_CASES = [
    FallbackCase('all-behind-the-gateway', first_gateway=True),
    FallbackCase('one-direct', first_gateway=False, sends_headers=False),
    FallbackCase(
        'own-headers-differ',
        first_gateway=True,
        first_settings={'extra_headers': {'x-tenant': 'acme'}},
        sends_headers=False,
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in FALLBACK_CASES])
async def test_fallback_model_sends_the_headers_only_when_every_model_can(route: RouteFactory, case: FallbackCase):
    """A `FallbackModel` sends one set of settings to every model it tries.

    So it carries the headers only when all of its models are behind the Gateway, and when they would send the same
    `extra_headers`: headers set for the request replace the ones a model sets for itself.
    """
    first, first_route = route(_REJECTED, gateway=case.first_gateway, settings=case.first_settings)
    second, second_route = route(_completion('Paris'))
    # Wrapped, as a durable execution engine wraps the agent's model.
    agent = Agent(WrapperModel(FallbackModel(first, second)), name='geographer')

    await agent.run('What is the capital of France?')

    assert [bool(sent) for sent in [*first_route.sent, *second_route.sent]] == [case.sends_headers] * 2


async def test_opting_out(route: RouteFactory):
    """`GatewayRoutingFacts(enabled=False)` stops the headers: on the agent, in its spec, or for one run."""
    model, gateway = route(*[_completion('Paris')] * 4)
    opted_out = Agent(model, capabilities=[GatewayRoutingFacts(enabled=False)])
    from_spec = Agent.from_spec({'capabilities': [{'GatewayRoutingFacts': {'enabled': False}}]}, model=model)
    agent = Agent(model)

    await opted_out.run('What is the capital of France?')
    await from_spec.run('What is the capital of France?')
    await agent.run('What is the capital of France?', capabilities=[GatewayRoutingFacts(enabled=False)])
    await agent.run('What is the capital of France?')

    assert [bool(sent) for sent in gateway.sent] == [False, False, False, True]


@pytest.mark.parametrize(
    'name, sent',
    [
        pytest.param('geographer', 'geographer', id='plain'),
        pytest.param('  geographer\n', 'geographer', id='surrounding-whitespace-trimmed'),
        pytest.param('city guide', 'city guide', id='inner-space-kept'),
        pytest.param('g' * 128, 'g' * 128, id='128-characters'),
        pytest.param('g' * 129, None, id='129-characters'),
        pytest.param('géographe', None, id='non-ascii'),
        pytest.param('geo\tgrapher', None, id='control-character'),
        pytest.param('   ', None, id='blank'),
        pytest.param(None, None, id='no-name'),
    ],
)
async def test_agent_name(route: RouteFactory, name: str | None, sent: str | None):
    """The agent's name is sent trimmed, and left out when it can't be a short printable-ASCII header value."""
    model, gateway = route(_completion('Paris'))
    # In a list, so no name is inferred from a variable for the `None` case.
    agents = [Agent(model, name=name)]

    await agents[0].run('What is the capital of France?')

    assert gateway.sent[0].get('x-pydantic-agent-name') == sent


@pytest.mark.parametrize(
    'conversation_id, sent',
    [
        pytest.param('c' * 128, 'c' * 128, id='128-characters'),
        pytest.param('c' * 129, None, id='129-characters'),
        pytest.param('conversación', None, id='non-ascii'),
    ],
)
async def test_conversation_id(route: RouteFactory, conversation_id: str, sent: str | None):
    """The conversation ID is left out when it can't be a short printable-ASCII header value."""
    model, gateway = route(_completion('Paris'))
    agent = Agent(model)

    await agent.run('What is the capital of France?', conversation_id=conversation_id)

    assert gateway.sent[0].get('x-routing-conversation-id') == sent


# --- The request kind, over drawn message histories ---
#
# Unit tests rather than runs: the kind is a pure function of the history, and a run reaches only the histories the
# agent loop writes, while stored and hand-built histories reach it too.

PROPERTY_SETTINGS = settings(
    max_examples=50, derandomize=True, database=None, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)

_RANK: dict[str, int] = {'user_prompt': 0, 'tool_results': 1, 'tool_retry': 2, 'output_retry': 3}
_RETRY_KINDS = {'tool_retry', 'output_retry'}


def _trailing_messages(history: Sequence[ModelMessage]) -> list[ModelMessage]:
    """The requests after the last response: the request about to be sent, once the agent merges them."""
    trailing: list[ModelMessage] = []
    for message in reversed(history):
        if isinstance(message, ModelResponse):
            break
        trailing.insert(0, message)
    return trailing


def _trailing_parts(history: Sequence[ModelMessage]) -> list[ModelRequestPart]:
    return [
        part for message in _trailing_messages(history) if isinstance(message, ModelRequest) for part in message.parts
    ]


def _tool_names(history: Sequence[ModelMessage]) -> set[str]:
    return {part.tool_name for part in _trailing_parts(history) if isinstance(part, ToolReturnPart)}


@st.composite
def _histories_with_attempts(draw: st.DrawFn) -> list[ModelMessage]:
    """Damaged histories, as stored or hand-built ones are, with a drawn `attempt` on each retry, up to past 100."""
    history = draw(histories(well_formed=False))
    attempts = st.none() | st.integers(1, 150)
    redrawn: list[ModelMessage] = []
    for message in history:
        if isinstance(message, ModelRequest):
            parts: list[ModelRequestPart] = []
            for part in message.parts:
                if isinstance(part, RetryFeedbackPart | ToolReturnPart) and part.retry is not None:
                    part = replace(part, retry=replace(part.retry, attempt=draw(attempts)))
                parts.append(part)
            message = replace(message, parts=parts)
        redrawn.append(message)
    return redrawn


_OUTPUT_TOOLS = st.sampled_from(['none', 'all'])


@PROPERTY_SETTINGS
@given(_histories_with_attempts(), _histories_with_attempts(), _OUTPUT_TOOLS)
def test_only_the_request_being_sent_decides_the_kind(
    history: list[ModelMessage], earlier: list[ModelMessage], output_tools: Literal['none', 'all']
):
    """The requests after the last response decide the kind, whatever history came before them."""
    names = _tool_names(history) if output_tools == 'all' else set[str]()
    moved = [*earlier, ModelResponse(parts=[TextPart('Earlier answer.')]), *_trailing_messages(history)]
    assert _request_kind(moved, names) == _request_kind(history, names)


@PROPERTY_SETTINGS
@given(_histories_with_attempts(), _OUTPUT_TOOLS)
def test_the_kind_follows_the_parts(history: list[ModelMessage], output_tools: Literal['none', 'all']):
    """Retry feedback makes an output retry, and no tool return or feedback leaves the prompt as the kind.

    The retry is between 0 and 100, and 0 for a kind that isn't a retry.
    """
    names = _tool_names(history) if output_tools == 'all' else set[str]()
    parts = _trailing_parts(history)
    kind, retry = _request_kind(history, names)

    if any(isinstance(part, RetryFeedbackPart) for part in parts):
        assert kind == 'output_retry'
    if not any(isinstance(part, RetryFeedbackPart | ToolReturnPart) for part in parts):
        assert kind == 'user_prompt'
    assert 0 <= retry <= 100
    if kind not in _RETRY_KINDS:
        assert retry == 0


@PROPERTY_SETTINGS
@given(_histories_with_attempts())
def test_output_tools_only_turn_tool_retries_into_output_retries(history: list[ModelMessage]):
    """Naming the retried tools as output tools turns a tool retry into an output retry, and changes no other kind."""
    without, _ = _request_kind(history, set[str]())
    with_output_tools, _ = _request_kind(history, _tool_names(history))

    if without == 'tool_retry':
        assert with_output_tools == 'output_retry'
    else:
        assert with_output_tools == without


_EXTRA_PARTS: st.SearchStrategy[ModelRequestPart] = st.one_of(
    st.builds(UserPromptPart, st.just('And the population?')),
    st.builds(ToolReturnPart, st.just('lookup'), st.just('Paris.'), st.just('call_extra')),
    st.builds(
        ToolReturnPart,
        st.just('lookup'),
        st.just('Busy, call again.'),
        st.just('call_extra'),
        outcome=st.just('retried'),
        retry=st.just(RetryInfo('model_retry', attempt=1)),
    ),
    st.builds(RetryFeedbackPart, st.just('Name the country too.'), retry=st.just(RetryInfo('model_retry', attempt=1))),
)


@PROPERTY_SETTINGS
@given(_histories_with_attempts(), _EXTRA_PARTS)
def test_more_parts_never_lower_the_kind(history: list[ModelMessage], extra: ModelRequestPart):
    """The kinds are in order of precedence: a part added to the request can only keep the kind or raise it."""
    with_extra = [*history, ModelRequest(parts=[extra])]
    before, _ = _request_kind(history, set[str]())
    after, _ = _request_kind(with_extra, set[str]())
    assert _RANK[after] >= _RANK[before]
