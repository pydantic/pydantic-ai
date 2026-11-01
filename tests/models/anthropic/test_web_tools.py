"""Anthropic `web_search` / `web_fetch` native-tool versioning and per-client support.

The wire tool version (`web_search_20260209` / `web_fetch_20260209` vs the earlier ones) and the
beta headers are narrowed by the concrete Anthropic client the provider wraps — first-party,
both Bedrock APIs, Vertex, and Foundry each support a different subset. These tests pin that matrix,
the dynamic-filtering `caller` round-trip on web-fetch history, and the live request shape via VCR.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import httpx2
import pytest

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    DeferredToolResults,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    WebCitationSource,
)
from pydantic_ai.capabilities import AbstractCapability, Citations, NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.batch import BatchRequest, BatchResult
from pydantic_ai.native_tools import (
    SUPPORTED_NATIVE_TOOLS,
    AbstractNativeTool,
    CodeExecutionTool,
    WebFetchTool,
    WebSearchTool,
)

from ..._inline_snapshot import snapshot
from ...cassette_utils import single_request_body
from ...conftest import IsStr, RequestCapture, TestEnv, try_import
from ..conftest import AnthropicModelFactory
from ..test_anthropic import (
    MockAnthropic,
    completion_message,
    get_mock_chat_completion_kwargs,
    mock_anthropic_client,
)

if TYPE_CHECKING:
    from cassetter import Cassette

with try_import() as imports_successful:
    from anthropic import (
        APIStatusError,
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )
    from anthropic.types.beta import (
        BetaCodeExecutionResultBlock,
        BetaCodeExecutionToolResultBlock,
        BetaContentBlock,
        BetaDirectCaller,
        BetaDocumentBlock,
        BetaPlainTextSource,
        BetaServerToolCaller20260120,
        BetaServerToolUseBlock,
        BetaTextBlock,
        BetaUsage,
        BetaWebFetchBlock,
        BetaWebFetchToolResultBlock,
    )
    from anthropic.types.beta.beta_container_params import BetaContainerParams

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings, AnthropicWebToolCaller
    from pydantic_ai.providers.anthropic import AnthropicProvider

if not imports_successful():  # pragma: lax no cover
    AsyncAnthropicBedrock = AsyncAnthropicBedrockMantle = AsyncAnthropicVertex = AsyncAnthropicFoundry = None
    if not TYPE_CHECKING:
        # `AsyncAnthropic` is referenced in a module-level `pytest.param`, so it must resolve at collection
        # time even without `anthropic` installed; guarded from the type checker to keep it typed as the class
        # at its annotation sites (which only execute inside tests skipped when `anthropic` is absent).
        AsyncAnthropic = None

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `NativeToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `NativeToolReturnPart` instead.:DeprecationWarning'
    ),
]


@dataclass(frozen=True)
class ClientSupportCase:
    """One (client, requested-tools) row of the web-tool support matrix.

    `rejected_tool` set → the request must raise `UserError`; otherwise the request is accepted and
    `expected_tool_types` / `expected_betas` pin the narrowed wire payload.
    """

    id: str
    client_cls: Any
    base_url: str
    native_tools: list[AbstractNativeTool]
    expected_tool_types: list[str] = field(default_factory=list[str])
    expected_betas: list[str] = field(default_factory=list[str])
    rejected_tool: type[AbstractNativeTool] | None = None


CLIENT_SUPPORT_CASES = [
    ClientSupportCase(
        id='anthropic',
        client_cls=AsyncAnthropic,
        base_url='https://api.anthropic.com',
        native_tools=[WebSearchTool(), WebFetchTool()],
        expected_tool_types=['web_search_20260209', 'web_fetch_20260209'],
    ),
    ClientSupportCase(
        id='foundry',
        client_cls=AsyncAnthropicFoundry,
        base_url='https://example.services.ai.azure.com/anthropic',
        native_tools=[WebSearchTool(), WebFetchTool()],
        expected_tool_types=['web_search_20260209', 'web_fetch_20260209'],
    ),
    ClientSupportCase(
        id='vertex-web-search',
        client_cls=AsyncAnthropicVertex,
        base_url='https://us-central1-aiplatform.googleapis.com',
        native_tools=[WebSearchTool()],
        expected_tool_types=['web_search_20250305'],
    ),
    ClientSupportCase(
        id='bedrock-web-search-rejected',
        client_cls=AsyncAnthropicBedrock,
        base_url='https://bedrock-runtime.us-east-1.amazonaws.com',
        native_tools=[WebSearchTool()],
        rejected_tool=WebSearchTool,
    ),
    ClientSupportCase(
        id='bedrock-web-fetch-rejected',
        client_cls=AsyncAnthropicBedrock,
        base_url='https://bedrock-runtime.us-east-1.amazonaws.com',
        native_tools=[WebFetchTool()],
        rejected_tool=WebFetchTool,
    ),
    ClientSupportCase(
        id='bedrock-mantle-web-search-rejected',
        client_cls=AsyncAnthropicBedrockMantle,
        base_url='https://bedrock-mantle.us-east-1.api.aws',
        native_tools=[WebSearchTool()],
        rejected_tool=WebSearchTool,
    ),
    ClientSupportCase(
        id='bedrock-mantle-web-fetch-rejected',
        client_cls=AsyncAnthropicBedrockMantle,
        base_url='https://bedrock-mantle.us-east-1.api.aws',
        native_tools=[WebFetchTool()],
        rejected_tool=WebFetchTool,
    ),
    ClientSupportCase(
        id='vertex-web-fetch-rejected',
        client_cls=AsyncAnthropicVertex,
        base_url='https://us-central1-aiplatform.googleapis.com',
        native_tools=[WebFetchTool()],
        rejected_tool=WebFetchTool,
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CLIENT_SUPPORT_CASES])
def test_anthropic_web_tools_client_support(case: ClientSupportCase):
    """Web-tool wire versions and beta headers are narrowed by the client the provider wraps.

    `_add_native_tools` is the internal entry point: the public `prepare_request` returns
    `ModelRequestParameters`, not the wire tool dicts, so it can only assert the rejection path. This
    matches the sibling tool-search tests in the Anthropic suite, which reach the same private helper.
    """
    m = AnthropicModel(
        'claude-sonnet-4-6',
        provider=AnthropicProvider(anthropic_client=mock_anthropic_client(case.client_cls, case.base_url)),
    )
    params = ModelRequestParameters(native_tools=case.native_tools)

    if case.rejected_tool is not None:
        assert case.rejected_tool not in m.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)
        with pytest.raises(
            UserError, match=rf"Native tool\(s\) \['{case.rejected_tool.__name__}'\] not supported by this model"
        ):
            m.prepare_request(None, params)
        return

    tools, _, beta_features = m._add_native_tools([], params, AnthropicModelSettings())  # pyright: ignore[reportPrivateUsage]
    assert [tool.get('type') for tool in tools] == case.expected_tool_types
    assert sorted(beta_features) == case.expected_betas


def test_anthropic_explicit_profile_instance_narrows_web_tools():
    """A non-callable `profile` instance is still narrowed by the client when resolved."""
    provider = AnthropicProvider(
        anthropic_client=mock_anthropic_client(AsyncAnthropicBedrock, 'https://bedrock-runtime.us-east-1.amazonaws.com')
    )
    profile = provider.model_profile('claude-sonnet-4-6')
    m = AnthropicModel('claude-sonnet-4-6', provider=provider, profile=profile)
    assert WebSearchTool not in m.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)


_FETCH_PROMPT = 'Use web fetch to read https://ai.pydantic.dev and say in one sentence what Pydantic AI is.'


@dataclass(frozen=True)
class AllowedCallersCase:
    id: str
    model_name: str
    native_tools: list[AbstractNativeTool]
    expected_tools: list[tuple[str, list[str] | None]]
    allowed_callers: list[AnthropicWebToolCaller] | None = None
    citations: bool = False
    prompt: str = 'Reply with exactly: ok'
    expected_web_tool_calls: list[tuple[str, dict[str, Any] | None]] = field(
        default_factory=list[tuple[str, dict[str, Any] | None]]
    )
    expected_cited_domains: list[str] = field(default_factory=list[str])


ALLOWED_CALLERS_CASES = [
    AllowedCallersCase(
        id='setting-direct',
        model_name='claude-sonnet-5-5',
        native_tools=[WebSearchTool(max_uses=1), WebFetchTool(max_uses=1)],
        allowed_callers=['direct'],
        prompt=_FETCH_PROMPT,
        expected_tools=snapshot([('web_search_20260209', ['direct']), ('web_fetch_20260209', ['direct'])]),
        expected_web_tool_calls=snapshot([('web_fetch', None)]),
    ),
    AllowedCallersCase(
        id='setting-direct-previous-versions',
        model_name='claude-sonnet-4-5',
        native_tools=[WebSearchTool(max_uses=1), WebFetchTool(max_uses=1)],
        allowed_callers=['direct'],
        expected_tools=snapshot([('web_search_20250305', ['direct']), ('web_fetch_20250910', ['direct'])]),
    ),
    AllowedCallersCase(
        id='citations',
        model_name='claude-sonnet-5-5',
        native_tools=[WebSearchTool(max_uses=1), WebFetchTool(max_uses=1)],
        citations=True,
        prompt="Use web search, not web fetch, to find Pydantic AI's documentation and say in one sentence where it is.",
        expected_tools=snapshot([('web_search_20260209', ['direct']), ('web_fetch_20260209', ['direct'])]),
        expected_web_tool_calls=snapshot([('web_search', None)]),
        expected_cited_domains=snapshot(['github.com']),
    ),
    AllowedCallersCase(
        id='web-fetch-enable-citations',
        model_name='claude-sonnet-5-5',
        native_tools=[WebSearchTool(max_uses=1), WebFetchTool(max_uses=1, enable_citations=True)],
        prompt=_FETCH_PROMPT,
        expected_tools=snapshot([('web_search_20260209', None), ('web_fetch_20260209', ['direct'])]),
        expected_web_tool_calls=snapshot([('web_fetch', None)]),
        expected_cited_domains=snapshot(['ai.pydantic.dev']),
    ),
    AllowedCallersCase(
        id='setting-takes-precedence-over-citations',
        model_name='claude-sonnet-5-5',
        native_tools=[WebSearchTool(max_uses=1), WebFetchTool(max_uses=1)],
        allowed_callers=['code_execution_20260120'],
        citations=True,
        expected_tools=snapshot(
            [('web_search_20260209', ['code_execution_20260120']), ('web_fetch_20260209', ['code_execution_20260120'])]
        ),
    ),
    AllowedCallersCase(
        id='citations-previous-versions',
        model_name='claude-sonnet-4-5',
        native_tools=[WebSearchTool(max_uses=1), WebFetchTool(max_uses=1)],
        citations=True,
        expected_tools=snapshot([('web_search_20250305', None), ('web_fetch_20250910', None)]),
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in ALLOWED_CALLERS_CASES])
async def test_anthropic_web_tools_allowed_callers(
    case: AllowedCallersCase,
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
):
    """The web tools' `allowed_callers`: the setting on either tool version, else direct calls for cited tools.

    Results the model reads from code execution, the default caller of the `20260209` versions, come back uncited, so
    a cited tool on those versions is sent with direct calls. The older versions already default to direct calls and
    are sent unchanged.
    """
    settings = AnthropicModelSettings()
    if case.allowed_callers is not None:
        settings['anthropic_web_tools_allowed_callers'] = case.allowed_callers
    capabilities: list[AbstractCapability[None]] = [NativeTool(tool) for tool in case.native_tools]
    if case.citations:
        capabilities.append(Citations())
    agent = Agent(
        anthropic_model(case.model_name, capture=True),
        deps_type=type(None),
        capabilities=capabilities,
        model_settings=settings,
    )

    result = await agent.run(case.prompt)

    tools = request_capture.body('/v1/messages')['tools']
    assert isinstance(tools, list)
    assert [
        (tool['type'], tool.get('allowed_callers')) for tool in tools if isinstance(tool, dict)
    ] == case.expected_tools
    response_parts = [part for message in result.all_messages() for part in message.parts]
    assert [
        (part.tool_name, part.provider_details)
        for part in response_parts
        if isinstance(part, NativeToolCallPart) and part.tool_name in ('web_search', 'web_fetch')
    ] == case.expected_web_tool_calls
    assert (
        sorted(
            {
                urlparse(source.url).netloc
                for part in response_parts
                if isinstance(part, TextPart)
                for citation in part.citations or []
                for source in citation.sources
                if isinstance(source, WebCitationSource)
            }
        )
        == case.expected_cited_domains
    )


@pytest.mark.vcr()
@pytest.mark.moves_cache_prefix(reason='the second run adds `Citations()`, which sends web search with direct calls')
async def test_anthropic_citations_after_code_execution_web_search_history(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """History from a run that searched from code execution replays into a run whose cited web search is direct.

    The new user prompt opens a turn without code execution, so its cited web search is called directly, and the
    history's container id isn't reused: Anthropic rejects a container on a request without code execution.
    """
    model = anthropic_model('claude-sonnet-5-5', capture=True)
    web_search = NativeTool(WebSearchTool(max_uses=1))
    first = await Agent(model, capabilities=[web_search]).run(
        "Use web search to find Pydantic AI's documentation and say in one sentence where it is."
    )
    second = await Agent(model, deps_type=type(None), capabilities=[web_search, Citations()]).run(
        'Now use web search again to find its GitHub repository and say in one sentence where it is.',
        message_history=first.all_messages(),
    )

    assert [
        (part.tool_name, part.provider_details)
        for message in first.new_messages()
        for part in message.parts
        if isinstance(part, NativeToolCallPart) and part.tool_name == 'web_search'
    ] == snapshot(
        [
            (
                'web_search',
                {
                    'anthropic_caller': {
                        'tool_id': 'srvtoolu_01PWm8WM4K3rytTP31T92bF6',
                        'type': 'code_execution_20260120',
                    }
                },
            )
        ]
    )
    last_request = request_capture.bodies('/v1/messages')[-1]
    tools = last_request['tools']
    assert isinstance(tools, list)
    assert (
        'container' in last_request,
        [(tool['type'], tool.get('allowed_callers')) for tool in tools if isinstance(tool, dict)],
    ) == snapshot((False, [('web_search_20260209', ['direct'])]))
    assert sorted(
        {
            urlparse(source.url).netloc
            for message in second.new_messages()
            for part in message.parts
            if isinstance(part, TextPart)
            for citation in part.citations or []
            for source in citation.sources
            if isinstance(source, WebCitationSource)
        }
    ) == snapshot(['github.com'])


def _anthropic_response(
    message_id: str, blocks: list[dict[str, Any]], stop_reason: str, container: str | None, stream: bool
) -> httpx2.Response:
    """A Messages API response with these content blocks, as JSON or as server-sent events."""
    message: dict[str, Any] = {
        'id': message_id,
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-6',
        'usage': {'input_tokens': 5, 'output_tokens': 1},
    }
    if container is not None:
        message['container'] = {'id': container, 'expires_at': '2026-12-01T00:00:00Z'}
    if not stream:
        return httpx2.Response(200, json={**message, 'content': blocks, 'stop_reason': stop_reason})
    events: list[dict[str, Any]] = [
        {'type': 'message_start', 'message': {**message, 'content': [], 'stop_reason': None}}
    ]
    for index, block in enumerate(blocks):
        delta: dict[str, Any] | None
        if block['type'] == 'text':
            start: dict[str, Any] = {'type': 'text', 'text': ''}
            delta = {'type': 'text_delta', 'text': block['text']}
        elif block['type'] == 'server_tool_use':
            start = {**block, 'input': {}}
            delta = {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}
        else:
            start, delta = block, None
        events.append({'type': 'content_block_start', 'index': index, 'content_block': start})
        if delta is not None:
            events.append({'type': 'content_block_delta', 'index': index, 'delta': delta})
        events.append({'type': 'content_block_stop', 'index': index})
    events += [
        {
            'type': 'message_delta',
            'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
            'usage': {'output_tokens': 1},
        },
        {'type': 'message_stop'},
    ]
    body = ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events)
    return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=body.encode())


async def _drain(_: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for _event in events:
        pass


def _code_execution_call(
    tool_call_id: str, tool_name: str = 'code_execution', provider_name: str = 'anthropic'
) -> NativeToolCallPart:
    return NativeToolCallPart(
        tool_name=tool_name, args={'code': 'print(1)'}, tool_call_id=tool_call_id, provider_name=provider_name
    )


def _code_execution_return(tool_call_id: str, provider_name: str = 'anthropic') -> NativeToolReturnPart:
    return NativeToolReturnPart(
        tool_name='code_execution',
        content={'type': 'code_execution_result', 'stdout': '1', 'stderr': '', 'return_code': 0, 'content': []},
        tool_call_id=tool_call_id,
        provider_name=provider_name,
    )


def _mixed_turn_history(with_tool_result: bool) -> list[ModelMessage]:
    """A turn that left a sandbox call pending beside a client tool call, so the API runs it on the next request."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Search for X and run my tool.')]),
        ModelResponse(
            parts=[
                _code_execution_call('srvtoolu_pending'),
                ToolCallPart(tool_name='my_tool', args={}, tool_call_id='toolu_1'),
            ],
            provider_name='anthropic',
            provider_details={'container_id': 'container_H'},
            finish_reason='tool_call',
        ),
    ]
    if with_tool_result:
        history.append(ModelRequest(parts=[ToolReturnPart(tool_name='my_tool', content='ok', tool_call_id='toolu_1')]))
    return history


def _suspended_history(*parts: ModelResponsePart) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content='Search for X.')]),
        ModelResponse(
            parts=list(parts),
            provider_name='anthropic',
            provider_details={'container_id': 'container_S'},
            state='suspended',
        ),
    ]


_SKILLS_CONTAINER: BetaContainerParams = {'skills': [{'type': 'anthropic', 'skill_id': 'xlsx', 'version': 'latest'}]}


@dataclass(frozen=True)
class OpenTurnCase:
    id: str
    expected_callers: list[str] | None
    expected_container: BetaContainerParams | str | None
    native_tools: list[AbstractNativeTool] = field(default_factory=lambda: [WebSearchTool()])
    container: BetaContainerParams | str | None = None
    history: list[ModelMessage] | None = None
    prompt: str | None = 'Look it up.'
    deferred_tool_results: DeferredToolResults | None = None


OPEN_TURN_CASES = [
    OpenTurnCase('explicit-container-web-search', None, 'container_A', container='container_A'),
    OpenTurnCase(
        'explicit-container-web-fetch', None, 'container_A', native_tools=[WebFetchTool()], container='container_A'
    ),
    OpenTurnCase('skills-container', None, _SKILLS_CONTAINER, container=_SKILLS_CONTAINER),
    OpenTurnCase(
        'resumed-pending-code-execution',
        None,
        'container_S',
        history=_suspended_history(TextPart(content='Searching.'), _code_execution_call('srvtoolu_pending')),
        prompt=None,
    ),
    OpenTurnCase(
        'resumed-none-pending',
        None,
        'container_S',
        history=_suspended_history(
            _code_execution_call('srvtoolu_done'), _code_execution_return('srvtoolu_done'), TextPart(content='More.')
        ),
        prompt=None,
    ),
    OpenTurnCase(
        'resumed-legacy-tool-name',
        None,
        'container_S',
        history=_suspended_history(_code_execution_call('srvtoolu_pending', tool_name='bash_code_execution')),
        prompt=None,
    ),
    OpenTurnCase('mixed-turn-after-tool-result', None, 'container_H', history=_mixed_turn_history(True), prompt=None),
    OpenTurnCase(
        'mixed-turn-deferred-approval',
        None,
        'container_H',
        history=_mixed_turn_history(False),
        prompt=None,
        deferred_tool_results=DeferredToolResults(approvals={'toolu_1': True}),
    ),
    OpenTurnCase(
        'new-prompt-after-sandboxed-turn',
        ['direct'],
        None,
        history=[
            ModelRequest(parts=[UserPromptPart(content='Search for X.')]),
            ModelResponse(
                parts=[
                    _code_execution_call('srvtoolu_done'),
                    _code_execution_return('srvtoolu_done'),
                    TextPart(content='Found X.'),
                ],
                provider_name='anthropic',
                provider_details={'container_id': 'container_S'},
            ),
        ],
    ),
    OpenTurnCase(
        'other-provider-code-execution',
        ['direct'],
        None,
        history=[
            ModelRequest(parts=[UserPromptPart(content='Search for X and run my tool.')]),
            ModelResponse(
                parts=[
                    _code_execution_call('ci_1', provider_name='openai'),
                    _code_execution_return('ci_1', provider_name='openai'),
                    ToolCallPart(tool_name='my_tool', args={}, tool_call_id='call_1'),
                ],
                provider_name='openai',
                finish_reason='tool_call',
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='my_tool', content='ok', tool_call_id='call_1')]),
        ],
        prompt=None,
    ),
]


@pytest.mark.parametrize('stream', [pytest.param(False, id='request'), pytest.param(True, id='stream')])
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in OPEN_TURN_CASES])
async def test_anthropic_citations_keep_default_web_tool_callers_when_the_request_needs_code_execution(
    allow_model_requests: None, case: OpenTurnCase, stream: bool
):
    """`Citations()` leaves the web tools on Anthropic's default caller when the request depends on code execution.

    An explicit or Skills container needs the sandbox, and so does a user turn that already used it: its container
    and pending calls stay in the turn until the next user prompt, including across client tool results and a resumed
    pause, under any of the code execution tool names. Their results come back uncited, as without `Citations()`. A
    new user prompt, or a turn whose sandbox ran on another provider, calls the cited tools directly and sends no
    container. The response filter follows the same callers, so a sandbox call is kept only where it can happen.

    Unit test: the claim is the request shape and the response filter, and recording these needs a live container.
    """
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        blocks = [
            {'type': 'server_tool_use', 'id': 'srvtoolu_new', 'name': 'code_execution', 'input': {'code': 'print(1)'}},
            {'type': 'text', 'text': 'Done.'},
        ]
        return _anthropic_response('msg_1', blocks, 'end_turn', container=None, stream=stream)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        model = AnthropicModel(
            'claude-sonnet-4-6',
            provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key='x', http_client=http_client)),
        )
        settings: AnthropicModelSettings = {'max_tokens': 1024}
        if case.container is not None:
            settings['anthropic_container'] = case.container
        capabilities: list[AbstractCapability[None]] = [NativeTool(tool) for tool in case.native_tools]
        agent = Agent(model, deps_type=type(None), capabilities=[*capabilities, Citations()], model_settings=settings)

        @agent.tool_plain(requires_approval=True)
        def my_tool() -> str:
            return 'ok'

        result = await agent.run(
            case.prompt,
            message_history=case.history,
            deferred_tool_results=case.deferred_tool_results,
            event_stream_handler=_drain if stream else None,
        )

    [body] = bodies
    assert [tool.get('allowed_callers') for tool in body['tools'] if tool['name'] != 'my_tool'] == [
        case.expected_callers
    ]
    assert body.get('container') == case.expected_container
    calls = {
        part.tool_call_id
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, NativeToolCallPart)
    }
    assert ('srvtoolu_new' in calls) is (case.expected_callers is None)


def _code_execution_blocks(tool_call_id: str, result: bool) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {'type': 'server_tool_use', 'id': tool_call_id, 'name': 'code_execution', 'input': {'code': 'print(1)'}}
    ]
    if result:
        blocks.append(
            {
                'type': 'code_execution_tool_result',
                'tool_use_id': tool_call_id,
                'content': {
                    'type': 'code_execution_result',
                    'stdout': '1',
                    'stderr': '',
                    'return_code': 0,
                    'content': [],
                },
            }
        )
    return blocks


@dataclass(frozen=True)
class PauseChainCase:
    id: str
    native_tools: list[AbstractNativeTool]
    expected_callers: list[tuple[str, list[str] | None]]
    expected_containers: list[str | None]
    citations: bool = True
    history: list[ModelMessage] | None = None


PAUSE_CHAIN_CASES = [
    PauseChainCase(
        'cited-web-fetch-beside-uncited-web-search',
        [WebFetchTool(enable_citations=True), WebSearchTool()],
        [('web_fetch_20260209', ['direct']), ('web_search_20260209', None)],
        [None, 'container_A', 'container_A'],
        citations=False,
    ),
    PauseChainCase(
        'cited-web-search-beside-code-execution',
        [WebSearchTool(), CodeExecutionTool()],
        [('web_search_20260209', ['direct']), ('code_execution_20260120', None)],
        [None, 'container_A', 'container_A'],
    ),
    PauseChainCase(
        'resumed-turn-that-used-code-execution',
        [WebSearchTool()],
        [('web_search_20260209', None)],
        ['container_S', 'container_A', 'container_A'],
        history=_suspended_history(TextPart(content='Searching.'), _code_execution_call('srvtoolu_0')),
    ),
]


@pytest.mark.parametrize('stream', [pytest.param(False, id='request'), pytest.param(True, id='stream')])
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in PAUSE_CHAIN_CASES])
async def test_anthropic_citations_keep_web_tool_callers_across_pause_turn_segments(
    allow_model_requests: None, case: PauseChainCase, stream: bool
):
    """Every request of a turn that pauses sends the web tools with the callers its first request had.

    A turn routed to direct calls keeps them while code execution stays on the request through another tool, and a
    resumed turn that used code execution keeps the default callers in every segment. Each segment's sandbox calls
    are kept.

    Unit test: a live `pause_turn` chain can't be requested reliably, and the claim is the request shape.
    """
    bodies: list[dict[str, Any]] = []
    segments = iter(
        [
            ([{'type': 'text', 'text': 'Step 1.'}, *_code_execution_blocks('srvtoolu_1', result=False)], 'pause_turn'),
            (
                [
                    *_code_execution_blocks('srvtoolu_1', result=True)[1:],
                    {'type': 'text', 'text': 'Step 2.'},
                    *_code_execution_blocks('srvtoolu_2', result=True),
                ],
                'pause_turn',
            ),
            ([{'type': 'text', 'text': 'Done.'}], 'end_turn'),
        ]
    )

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        blocks, stop_reason = next(segments)
        return _anthropic_response(f'msg_{len(bodies)}', blocks, stop_reason, container='container_A', stream=stream)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        model = AnthropicModel(
            'claude-sonnet-4-6',
            provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key='x', http_client=http_client)),
        )
        capabilities: list[AbstractCapability[None]] = [NativeTool(tool) for tool in case.native_tools]
        if case.citations:
            capabilities.append(Citations())
        agent = Agent(model, deps_type=type(None), capabilities=capabilities, model_settings={'max_tokens': 1024})
        result = await agent.run(
            None if case.history else 'Look it up.',
            message_history=case.history,
            event_stream_handler=_drain if stream else None,
        )

    assert [[(tool['type'], tool.get('allowed_callers')) for tool in body['tools']] for body in bodies] == [
        case.expected_callers
    ] * len(bodies)
    assert [body.get('container') for body in bodies] == case.expected_containers
    assert {
        part.tool_call_id
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, NativeToolCallPart)
    } >= {'srvtoolu_1', 'srvtoolu_2'}


@pytest.mark.parametrize(
    ('history', 'expected_calls'),
    [
        pytest.param(
            _suspended_history(_code_execution_call('srvtoolu_0')),
            ['srvtoolu_new'],
            id='turn-that-used-code-execution',
        ),
        pytest.param([ModelRequest(parts=[UserPromptPart(content='Look it up.')])], [], id='new-turn'),
        pytest.param([], [], id='no-messages'),
    ],
)
def test_anthropic_citations_batch_decode_follows_the_open_turn(history: list[ModelMessage], expected_calls: list[str]):
    """A batch result is filtered with the web tool callers its request was sent with.

    Unit test: a batch decodes a stored result without a request, so no recording exercises the decode on its own.
    """
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='x'))
    message = {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-6',
        'content': [*_code_execution_blocks('srvtoolu_new', result=False), {'type': 'text', 'text': 'Done.'}],
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 5, 'output_tokens': 1},
    }
    response = model.batch_decode(
        BatchResult(custom_id='item', status='succeeded', body=json.dumps({'type': 'succeeded', 'message': message})),
        BatchRequest(
            custom_id='item',
            messages=history,
            model_settings={'citations': True},
            model_request_parameters=ModelRequestParameters(native_tools=[WebSearchTool()]),
        ),
    )

    assert [part.tool_call_id for part in response.parts if isinstance(part, NativeToolCallPart)] == expected_calls


@pytest.mark.vcr()
async def test_anthropic_explicit_code_execution_caller_on_previous_web_tool_versions(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """A code execution caller set on a web tool version that defaults to direct calls gets a sandbox too.

    Its `code_execution` calls are kept, so the follow-up replays every call its results point at, and the follow-up
    reuses the container.
    """
    model = anthropic_model('claude-sonnet-4-5', capture=True)
    agent = Agent(
        model,
        capabilities=[NativeTool(WebSearchTool(max_uses=1))],
        model_settings=AnthropicModelSettings(anthropic_web_tools_allowed_callers=['code_execution_20260120']),
    )
    async with agent.run_stream(
        'First, in the code sandbox, write the text "sentinel-42" to /tmp/state.txt. '
        'Then use web search once to find the latest pydantic-ai version on PyPI. Reply with just the version.'
    ) as result:
        await result.get_output()
    second = await agent.run(
        'In the code sandbox, print the contents of /tmp/state.txt. Reply with exactly what it printed.',
        message_history=result.all_messages(),
    )

    bodies = request_capture.bodies('/v1/messages')
    tools = bodies[0]['tools']
    assert isinstance(tools, list)
    assert [(tool['type'], tool.get('allowed_callers')) for tool in tools if isinstance(tool, dict)] == snapshot(
        [('web_search_20250305', ['code_execution_20260120'])]
    )
    assert sorted(
        {
            part.tool_name
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, NativeToolCallPart)
        }
    ) == snapshot(['code_execution', 'web_search'])
    assert ['container' in body for body in bodies] == snapshot([False, True])
    assert second.output == snapshot('sentinel-42')


async def test_anthropic_web_fetch_20260209_caller_pass_history_back(env: TestEnv, allow_model_requests: None):
    """Pass Anthropic dynamic-filtering caller metadata back with web fetch history.

    Unit (not VCR) test: it asserts the `caller` re-emitted onto the *outgoing* second request's
    `server_tool_use` and `web_fetch_tool_result` blocks via `get_mock_chat_completion_kwargs`. The VCR
    cassette matcher isn't sensitive to the request body, so a recording wouldn't catch a regression in
    that replayed payload; capturing the mock client's call kwargs is what pins it.
    """
    code_tool_id = 'srvtoolu_code'
    fetch_tool_id = 'srvtoolu_fetch'
    fetch_caller = BetaServerToolCaller20260120(tool_id=code_tool_id, type='code_execution_20260120')
    content: list[BetaContentBlock] = [
        BetaServerToolUseBlock(
            id=code_tool_id,
            name='code_execution',
            input={'code': 'result = await web_fetch({"url": "https://example.com"})'},
            type='server_tool_use',
            caller=BetaDirectCaller(type='direct'),
        ),
        BetaServerToolUseBlock(
            id=fetch_tool_id,
            name='web_fetch',
            input={'url': 'https://example.com'},
            type='server_tool_use',
            caller=fetch_caller,
        ),
        BetaWebFetchToolResultBlock(
            tool_use_id=fetch_tool_id,
            type='web_fetch_tool_result',
            content=BetaWebFetchBlock(
                content=BetaDocumentBlock(
                    type='document',
                    source=BetaPlainTextSource(type='text', media_type='text/plain', data='Example Domain'),
                ),
                type='web_fetch_result',
                url='https://example.com',
            ),
            caller=fetch_caller,
        ),
        BetaCodeExecutionToolResultBlock(
            tool_use_id=code_tool_id,
            type='code_execution_tool_result',
            content=BetaCodeExecutionResultBlock(
                content=[],
                return_code=0,
                stderr='',
                stdout='Example Domain\n',
                type='code_execution_result',
            ),
        ),
        BetaTextBlock(text='Fetched Example Domain.', type='text'),
    ]
    first_response = completion_message(content, BetaUsage(input_tokens=20, output_tokens=30))
    second_response = completion_message(
        [BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=50, output_tokens=5)
    )

    mock_client = MockAnthropic.create_mock([first_response, second_response])
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, capabilities=[NativeTool(WebFetchTool())])

    result = await agent.run('Fetch https://example.com')
    web_fetch_call = next(
        p
        for message in result.all_messages()
        for p in message.parts
        if isinstance(p, NativeToolCallPart) and p.tool_name == 'web_fetch'
    )
    assert web_fetch_call.provider_details == snapshot(
        {'anthropic_caller': {'tool_id': 'srvtoolu_code', 'type': 'code_execution_20260120'}}
    )

    await agent.run('Continue.', message_history=result.all_messages())

    assistant_content = cast(
        list[dict[str, Any]], get_mock_chat_completion_kwargs(mock_client)[1]['messages'][1]['content']
    )
    web_fetch_use = next(
        item
        for item in assistant_content
        if isinstance(item, dict) and item.get('type') == 'server_tool_use' and item.get('name') == 'web_fetch'
    )
    web_fetch_result = next(
        item for item in assistant_content if isinstance(item, dict) and item.get('type') == 'web_fetch_tool_result'
    )
    assert {
        'server_tool_use': web_fetch_use['caller'],
        'web_fetch_tool_result': web_fetch_result['caller'],
    } == snapshot(
        {
            'server_tool_use': {'tool_id': 'srvtoolu_code', 'type': 'code_execution_20260120'},
            'web_fetch_tool_result': {'tool_id': 'srvtoolu_code', 'type': 'code_execution_20260120'},
        }
    )


@pytest.mark.vcr()
async def test_anthropic_supported_model_uses_20260209_web_tools(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette
):
    m = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, capabilities=[NativeTool(WebSearchTool()), NativeTool(WebFetchTool())])

    result = await agent.run('Use web fetch to read https://ai.pydantic.dev and reply with exactly the page title.')

    assert result.output
    assert [tool['type'] for tool in single_request_body(vcr)['tools']] == snapshot(
        ['web_search_20260209', 'web_fetch_20260209']
    )
    response_parts = [part for message in result.all_messages() for part in message.parts]
    web_fetch_parts = [
        part
        for part in response_parts
        if isinstance(part, NativeToolCallPart | NativeToolReturnPart) and part.tool_name == 'web_fetch'
    ]
    assert len(web_fetch_parts) == 2
    caller_details = [part.provider_details for part in web_fetch_parts]
    assert caller_details == snapshot(
        [
            {'anthropic_caller': {'tool_id': IsStr(), 'type': 'code_execution_20260120'}},
            {'anthropic_caller': {'tool_id': IsStr(), 'type': 'code_execution_20260120'}},
        ]
    )
    assert caller_details[0] == caller_details[1]


@pytest.mark.vcr()
async def test_anthropic_unsupported_model_uses_previous_web_tools(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette
):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, capabilities=[NativeTool(WebSearchTool()), NativeTool(WebFetchTool())])

    result = await agent.run('Reply with exactly: ok')

    assert result.output
    tool_types = [tool['type'] for tool in single_request_body(vcr)['tools']]
    assert tool_types == ['web_search_20250305', 'web_fetch_20250910']


@pytest.mark.vcr()
@pytest.mark.parametrize(
    'tool',
    [
        pytest.param(WebSearchTool(), id='web-search'),
        pytest.param(WebFetchTool(), id='web-fetch'),
    ],
)
@pytest.mark.parametrize('dynamic_filtering', [pytest.param(False, id='basic'), pytest.param(True, id='20260209')])
async def test_anthropic_web_tools_rejected_by_bedrock_messages_api(
    allow_model_requests: None,
    request_capture: RequestCapture,
    request: pytest.FixtureRequest,
    tool: AbstractNativeTool,
    dynamic_filtering: bool,
):
    """The Bedrock Messages API rejects both versions of both web tools, which is why its client doesn't get them.

    `AnthropicModel` no longer sends them there, so this sends the definitions and betas it builds through the SDK.
    That pins the provider behavior the `bedrock-mantle` rows of the support matrix rest on.
    https://github.com/pydantic/pydantic-ai/issues/11481
    """
    if isinstance(tool, WebSearchTool):
        tool_param = AnthropicModel._map_web_search_tool(tool, dynamic_filtering)  # pyright: ignore[reportPrivateUsage]
        beta = None
    else:
        assert isinstance(tool, WebFetchTool)
        tool_param, beta = AnthropicModel._map_web_fetch_tool(tool, dynamic_filtering, False)  # pyright: ignore[reportPrivateUsage]
    client = request.getfixturevalue('anthropic_bedrock_mantle_client')

    with pytest.raises(APIStatusError, match=rf"tool type '{tool_param['type']}' is not supported for this model"):
        await client.beta.messages.create(
            model='anthropic.claude-opus-5-5',
            max_tokens=256,
            messages=[{'role': 'user', 'content': 'Look up https://example.com and give me its title.'}],
            tools=[tool_param],
            betas=[beta] if beta else [],
        )
