"""Errors that name the other Google model class for a feature only that class serves.

`GoogleModel` uses GenerateContent and `GoogleInteractionsModel` uses the Interactions API. Each points at the other
on a request that fails either way; a request that works keeps working.
"""

from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.capabilities import BatchModelRequests, NativeTool
from pydantic_ai.exceptions import ModelHTTPError, ModelRequestRejectedError, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import ComputerTool, MCPServerTool
from pydantic_ai.usage import UsageLimits

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.google_interactions import GoogleInteractionsModel
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'), pytest.mark.vcr]

DEEPWIKI = MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp')
DISPLAY_SIZE = (1024, 768)


async def test_mcp_server_tool_on_google_model_names_interactions(allow_model_requests: None):
    """Not a VCR test: the tool is rejected in `prepare_request`, before any request is built."""
    model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='mock-api-key'))
    agent = Agent(model, capabilities=[NativeTool(DEEPWIKI)])
    with pytest.raises(UserError) as exc_info:
        await agent.run('What is the pydantic/pydantic-ai repository about?')
    assert str(exc_info.value) == snapshot(
        '`MCPServerTool` is not supported by `GoogleModel`. Use `GoogleInteractionsModel`, with the `google-interactions:` prefix, or `MCP(local=True)` for a local fallback.'
    )


@pytest.mark.parametrize(
    'params',
    [
        pytest.param(
            ModelRequestParameters(native_tools=[MCPServerTool(id='deepwiki', url=DEEPWIKI.url, optional=True)]),
            id='optional',
        ),
        pytest.param(
            ModelRequestParameters(
                native_tools=[DEEPWIKI],
                function_tools=[ToolDefinition(name='ask_deepwiki', unless_native=DEEPWIKI.unique_id)],
            ),
            id='local-fallback',
        ),
    ],
)
def test_mcp_server_tool_on_google_model_without_error(params: ModelRequestParameters):
    """An optional `MCPServerTool`, or one with a local fallback, is left out as before rather than raising.

    Not a VCR test: the tool is resolved in `prepare_request`, before any request is built.
    """
    model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='mock-api-key'))
    _, prepared = model.prepare_request(None, params)
    assert prepared.native_tools == []


async def test_interactions_only_model_on_google_model_names_interactions(
    allow_model_requests: None, gemini_api_key: str
):
    """The Gemini API rejects a model it serves only through Interactions; the error names `GoogleInteractionsModel`.

    The recorded 400 `INVALID_ARGUMENT` says "This model only supports Interactions API.", which the hint matches.
    """
    model = GoogleModel('gemini-omni-1.1-flash', provider=GoogleProvider(api_key=gemini_api_key))
    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run('A red ball rolling across a plain white floor, fixed camera.')
    assert isinstance(exc_info.value, ModelRequestRejectedError)
    assert str(exc_info.value) == snapshot(
        "status_code: 400, model_name: gemini-omni-1.1-flash, body: {'error': {'code': 400, 'message': 'This model only supports Interactions API.', 'status': 'INVALID_ARGUMENT'}}. Use `GoogleInteractionsModel`, with the `google-interactions:` prefix, for this model."
    )


async def test_interactions_count_tokens_names_google_model(allow_model_requests: None):
    """Not a VCR test: the Interactions API has no token-counting endpoint, so nothing is sent."""
    model = GoogleInteractionsModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='mock-api-key'))
    with pytest.raises(NotImplementedError) as exc_info:
        await Agent(model).run('Hello', usage_limits=UsageLimits(count_tokens_before_request=True))
    assert str(exc_info.value) == snapshot(
        'Token counting ahead of the request is not supported by `GoogleInteractionsModel`, as the Interactions API has no token-counting endpoint. Use `GoogleModel` to count tokens.'
    )


@pytest.mark.parametrize('google_cloud', [False, True])
async def test_interactions_batch_names_google_model(allow_model_requests: None, google_cloud: bool):
    """Not a VCR test: the Interactions API has no batch API, so the batch is refused before anything is sent."""
    provider = GoogleCloudProvider(api_key='mock-api-key') if google_cloud else GoogleProvider(api_key='mock-api-key')
    model = GoogleInteractionsModel('gemini-3-flash-preview', provider=provider)
    with pytest.raises(UserError) as exc_info:
        await Agent(model, capabilities=[BatchModelRequests()]).run('Hello')
    assert str(exc_info.value) == snapshot(
        "`GoogleInteractionsModel` 'gemini-3-flash-preview' has no batch API: batch it with `GoogleModel` through the Gemini API, with the `google:` provider."
    )


async def test_interactions_computer_tool_names_google_model():
    """A computer use model on the Gemini API points at `GoogleModel`, which sends the tool there.

    Not a VCR test: the tool is rejected in `prepare_request`, before any request is built.
    """
    model = GoogleInteractionsModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='mock-api-key'))
    assert model.profile.get('google_supports_computer_use') is True
    with pytest.raises(UserError) as exc_info:
        model.prepare_request(None, ModelRequestParameters(native_tools=[ComputerTool(display_size=DISPLAY_SIZE)]))
    assert str(exc_info.value) == snapshot(
        "`ComputerTool` is not supported by `GoogleInteractionsModel`. Use `GoogleModel`, with the `google:` prefix, for computer use with 'gemini-3-flash-preview'."
    )


@pytest.mark.parametrize(
    'model_name,google_cloud',
    [
        pytest.param('gemini-2.5-flash', False, id='no-computer-use'),
        pytest.param('gemini-3-flash-preview', True, id='google-cloud'),
    ],
)
def test_interactions_computer_tool_without_google_model_support(model_name: str, google_cloud: bool):
    """Where `GoogleModel` doesn't send `ComputerTool` either, the error stays the generic one.

    `gemini-2.5-flash` isn't a computer use model, and `GoogleModel` turns computer use off on Google Cloud.
    Not a VCR test: the tool is rejected in `prepare_request`, before any request is built.
    """
    provider = GoogleCloudProvider(api_key='mock-api-key') if google_cloud else GoogleProvider(api_key='mock-api-key')
    model = GoogleInteractionsModel(model_name, provider=provider)
    with pytest.raises(UserError, match=r"^Native tool\(s\) \['ComputerTool'\] not supported by this model\."):
        model.prepare_request(None, ModelRequestParameters(native_tools=[ComputerTool(display_size=DISPLAY_SIZE)]))


def test_interactions_optional_computer_tool_is_left_out():
    """An optional `ComputerTool`, as the harness's `ComputerUse` capability sends, is still left out.

    Not a VCR test: the tool is resolved in `prepare_request`, before any request is built.
    """
    model = GoogleInteractionsModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='mock-api-key'))
    _, params = model.prepare_request(
        None, ModelRequestParameters(native_tools=[ComputerTool(display_size=DISPLAY_SIZE, optional=True)])
    )
    assert params.native_tools == []
