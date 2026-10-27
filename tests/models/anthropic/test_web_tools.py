"""Anthropic `web_search` / `web_fetch` native-tool versioning and per-client support.

The wire tool version (`web_search_20260209` / `web_fetch_20260209` vs the earlier ones) and the
beta headers are narrowed by the concrete Anthropic client the provider wraps — first-party,
both Bedrock APIs, Vertex, and Foundry each support a different subset. These tests pin that matrix,
the dynamic-filtering `caller` round-trip on web-fetch history, and the live request shape via VCR.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import pytest

from pydantic_ai import Agent, NativeToolCallPart, NativeToolReturnPart, TextPart, WebCitationSource
from pydantic_ai.capabilities import AbstractCapability, Citations, NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool, WebFetchTool, WebSearchTool

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

    The history's container id isn't reused: Anthropic rejects a container on a request without code execution.
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
        tool_param = AnthropicModel._map_web_search_tool(tool, dynamic_filtering, False, None)  # pyright: ignore[reportPrivateUsage]
        beta = None
    else:
        assert isinstance(tool, WebFetchTool)
        tool_param, beta = AnthropicModel._map_web_fetch_tool(tool, dynamic_filtering, False, None)  # pyright: ignore[reportPrivateUsage]
    client = request.getfixturevalue('anthropic_bedrock_mantle_client')

    with pytest.raises(APIStatusError, match=rf"tool type '{tool_param['type']}' is not supported for this model"):
        await client.beta.messages.create(
            model='anthropic.claude-opus-5-5',
            max_tokens=256,
            messages=[{'role': 'user', 'content': 'Look up https://example.com and give me its title.'}],
            tools=[tool_param],
            betas=[beta] if beta else [],
        )
