"""Anthropic `web_search` / `web_fetch` native-tool versioning and per-client support.

The wire tool version (`web_search_20260209` / `web_fetch_20260209` vs the earlier ones) and the
beta headers are narrowed by the concrete Anthropic client the provider wraps — first-party,
Bedrock(-Mantle), Vertex, and Foundry each support a different subset. These tests pin that matrix,
the dynamic-filtering `caller` round-trip on web-fetch history, and the live request shape via VCR.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest

from pydantic_ai import Agent, NativeToolCallPart, NativeToolReturnPart
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool, WebFetchTool, WebSearchTool

from ..._inline_snapshot import snapshot
from ...cassette_utils import single_request_body
from ...conftest import IsStr, TestEnv, try_import
from ..conftest import AnthropicModelFactory, RequestCapture
from ..test_anthropic import (
    MockAnthropic,
    _mock_anthropic_client,  # pyright: ignore[reportPrivateUsage]
    completion_message,
    get_mock_chat_completion_kwargs,
)

if TYPE_CHECKING:
    from vcr.cassette import Cassette

with try_import() as imports_successful:
    from anthropic import (
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

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        AnthropicModelSettings,
        AnthropicWebFetchToolSettings,
        AnthropicWebSearchToolSettings,
    )
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
    pytest.mark.anyio,
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
        id='bedrock-mantle',
        client_cls=AsyncAnthropicBedrockMantle,
        base_url='https://bedrock-mantle.us-east-1.api.aws',
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
        provider=AnthropicProvider(anthropic_client=_mock_anthropic_client(case.client_cls, case.base_url)),
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


@pytest.mark.parametrize(
    'tool_kind',
    [
        'search',
        'fetch',
    ],
)
def test_anthropic_20260318_web_tools_enable_implicit_code_execution(tool_kind: str):
    """Anthropic retains implicit code-execution blocks produced by `20260318` web tools.

    This whitelist controls response filtering and is not visible in the outgoing request, so the
    test exercises the helper that supplies it to streamed and non-streamed response processing.
    """
    tool: AbstractNativeTool
    if tool_kind == 'search':
        tool = WebSearchTool(settings=AnthropicWebSearchToolSettings(anthropic_response_inclusion='full'))
    else:
        tool = WebFetchTool(settings=AnthropicWebFetchToolSettings(anthropic_use_cache=False))

    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test'))
    params = ModelRequestParameters(native_tools=[tool])

    enabled_names = m._get_enabled_server_tool_names(  # pyright: ignore[reportPrivateUsage]
        params, AnthropicModelSettings()
    )

    assert {'code_execution', 'bash_code_execution', 'text_editor_code_execution'} <= enabled_names


@pytest.mark.vcr()
async def test_anthropic_20260318_web_search_response_inclusion(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
):
    """Anthropic accepts response exclusion and omits search results consumed by code execution."""
    model = anthropic_model('claude-sonnet-4-6', capture=True)
    agent = Agent(
        model,
        capabilities=[
            NativeTool(WebSearchTool(settings=AnthropicWebSearchToolSettings(anthropic_response_inclusion='excluded')))
        ],
    )

    result = await agent.run(
        "In code execution, use web_search with the query 'site:ai.pydantic.dev Pydantic AI' "
        'and reply with the first result title.'
    )

    assert result.output
    assert request_capture.body('/v1/messages')['tools'] == snapshot(
        [
            {
                'allowed_domains': None,
                'blocked_domains': None,
                'max_uses': None,
                'name': 'web_search',
                'response_inclusion': 'excluded',
                'type': 'web_search_20260318',
                'user_location': None,
            }
        ]
    )
    assert not any(
        isinstance(part, NativeToolCallPart | NativeToolReturnPart) and part.tool_name == 'web_search'
        for message in result.all_messages()
        for part in message.parts
    )


@pytest.mark.parametrize(
    ('setting_name', 'expected_setting'),
    [
        ('anthropic_use_cache', {'use_cache': False}),
        ('anthropic_response_inclusion', {'response_inclusion': 'excluded'}),
    ],
)
def test_anthropic_web_fetch_settings_map_to_20260318(setting_name: str, expected_setting: dict[str, object]):
    """Each Anthropic web fetch setting selects and maps to the `20260318` wire tool.

    The settings are built inside the test: the module collects without `anthropic` installed, so
    parametrize arguments cannot reference the guarded imports.
    """
    provider = AnthropicProvider(
        anthropic_client=_mock_anthropic_client(AsyncAnthropicBedrockMantle, 'https://bedrock-mantle.us-east-1.api.aws')
    )
    m = AnthropicModel('claude-sonnet-4-5', provider=provider)
    if setting_name == 'anthropic_use_cache':
        settings = AnthropicWebFetchToolSettings(anthropic_use_cache=False)
    else:
        settings = AnthropicWebFetchToolSettings(anthropic_response_inclusion='excluded')
    params = ModelRequestParameters(native_tools=[WebFetchTool(settings=settings)])

    tools, _, _ = m._add_native_tools(  # pyright: ignore[reportPrivateUsage]
        [], params, AnthropicModelSettings()
    )

    assert tools == [
        {
            'name': 'web_fetch',
            'type': 'web_fetch_20260318',
            'max_uses': None,
            'allowed_domains': None,
            'blocked_domains': None,
            'citations': None,
            'max_content_tokens': None,
            **expected_setting,
        },
    ]


@pytest.mark.vcr()
async def test_anthropic_20260318_web_fetch_settings(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
):
    """Anthropic accepts the tool settings and excludes the consumed fetch result."""
    model = anthropic_model('claude-sonnet-4-6', capture=True)
    agent = Agent(
        model,
        capabilities=[
            NativeTool(
                WebFetchTool(
                    settings=AnthropicWebFetchToolSettings(
                        anthropic_use_cache=False, anthropic_response_inclusion='excluded'
                    )
                )
            )
        ],
    )

    result = await agent.run('Use web fetch to read https://ai.pydantic.dev and reply with exactly the page title.')

    assert result.output
    assert request_capture.body('/v1/messages')['tools'] == snapshot(
        [
            {
                'allowed_domains': None,
                'blocked_domains': None,
                'citations': None,
                'max_content_tokens': None,
                'max_uses': None,
                'name': 'web_fetch',
                'response_inclusion': 'excluded',
                'type': 'web_fetch_20260318',
                'use_cache': False,
            }
        ]
    )
    assert not any(
        isinstance(part, NativeToolCallPart | NativeToolReturnPart) and part.tool_name == 'web_fetch'
        for message in result.all_messages()
        for part in message.parts
    )


def test_anthropic_explicit_profile_instance_narrows_web_tools():
    """A non-callable `profile` instance is still narrowed by the client when resolved."""
    provider = AnthropicProvider(
        anthropic_client=_mock_anthropic_client(
            AsyncAnthropicBedrock, 'https://bedrock-runtime.us-east-1.amazonaws.com'
        )
    )
    profile = provider.model_profile('claude-sonnet-4-6')
    m = AnthropicModel('claude-sonnet-4-6', provider=provider, profile=profile)
    assert WebSearchTool not in m.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)


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
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
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
