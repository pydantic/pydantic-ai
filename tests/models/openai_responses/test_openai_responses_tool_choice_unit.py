"""OpenAI Responses API tool_choice translation unit tests.

These tests verify the translation from PydanticAI's internal tool_choice format
to the OpenAI Responses API format. They complement the VCR-based integration
tests by explicitly asserting on the exact API values sent.

Key OpenAI Responses API format details (differs from Chat API):
- 'auto', 'none', 'required' are passed as strings
- Specific tools: {type: 'allowed_tools', mode: ..., tools: [{type: 'function', name: X}]}
- Note: mode is at top level, not nested. Tool format uses 'name' not 'function.name'
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.settings import ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition, ToolKind

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def make_tool(name: str, *, kind: ToolKind = 'function') -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f'Test tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}},
        kind=kind,
    )


@pytest.fixture
def responses_model() -> OpenAIResponsesModel:
    """Create an OpenAI Responses model for testing (no API calls made)."""
    return OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))


@pytest.fixture
def responses_model_no_required_support() -> OpenAIResponsesModel:
    """Create a Responses model that doesn't support tool_choice='required'."""
    profile = OpenAIModelProfile(openai_supports_tool_choice_required=False)
    return OpenAIResponsesModel('custom-model', provider=OpenAIProvider(api_key='test-key'), profile=profile)


class TestToolChoiceTranslation:
    """Unit tests for OpenAI Responses tool_choice API format translation."""

    def test_auto_format(self, responses_model: OpenAIResponsesModel):
        """tool_choice='auto' passes 'auto' string to API."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = responses_model._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'auto'
        assert len(tools) == 1
        assert tools[0]['name'] == 'get_weather'

    def test_none_format(self, responses_model: OpenAIResponsesModel):
        """tool_choice='none' passes 'none' string to API."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = responses_model._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'none'
        assert len(tools) == 1

    def test_required_format(self, responses_model: OpenAIResponsesModel):
        """tool_choice='required' passes 'required' string to API when supported."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = responses_model._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'required'
        assert len(tools) == 1

    def test_required_fallback_when_not_supported(self, responses_model_no_required_support: OpenAIResponsesModel):
        """tool_choice='required' falls back to 'auto' with warning when not supported."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        with pytest.warns(UserWarning, match="tool_choice='required' is not supported"):
            _, tool_choice = responses_model_no_required_support._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'auto'

    def test_single_tool_in_list_format(self, responses_model: OpenAIResponsesModel):
        """Single tool in list uses allowed_tools format (Responses API style)."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather'), make_tool('get_time')])

        tools, tool_choice = responses_model._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Responses API format: mode at top level, tool uses 'name' not 'function.name'
        assert tool_choice == snapshot(
            {
                'type': 'allowed_tools',
                'mode': 'required',
                'tools': [{'type': 'function', 'name': 'get_weather'}],
            }
        )
        # All tools are sent
        assert len(tools) == 2

    def test_multiple_tools_in_list_format(self, responses_model: OpenAIResponsesModel):
        """Multiple tools in list uses allowed_tools with multiple tool entries."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time'), make_tool('get_population')]
        )

        tools, tool_choice = responses_model._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == snapshot(
            {
                'type': 'allowed_tools',
                'mode': 'required',
                'tools': [
                    {'type': 'function', 'name': 'get_weather'},
                    {'type': 'function', 'name': 'get_time'},
                ],
            }
        )
        assert len(tools) == 3

    def test_tools_plus_output_format(self, responses_model: OpenAIResponsesModel):
        """ToolsPlusOutput uses allowed_tools format with both function and output tools."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time')],
            output_tools=[make_tool('final_result', kind='output')],
        )

        tools, tool_choice = responses_model._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == snapshot(
            {
                'type': 'allowed_tools',
                'mode': 'required',
                'tools': [
                    {'type': 'function', 'name': 'get_weather'},
                    {'type': 'function', 'name': 'final_result'},
                ],
            }
        )
        assert len(tools) == 3

    def test_none_with_output_tools_uses_auto_mode(self, responses_model: OpenAIResponsesModel):
        """tool_choice='none' with output tools returns allowed_tools with 'auto' mode."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather')],
            output_tools=[make_tool('final_result', kind='output')],
            allow_text_output=True,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            _, tool_choice = responses_model._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Returns allowed_tools with 'auto' mode
        assert tool_choice == snapshot(
            {
                'type': 'allowed_tools',
                'mode': 'auto',
                'tools': [{'type': 'function', 'name': 'final_result'}],
            }
        )

    def test_no_tools_returns_none(self, responses_model: OpenAIResponsesModel):
        """When no tools are available, returns empty list and None."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[])

        tools, tool_choice = responses_model._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tools == []
        assert tool_choice is None

    def test_tool_definition_format(self, responses_model: OpenAIResponsesModel):
        """Verify the complete tool definition format sent to Responses API."""
        settings: OpenAIResponsesModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, _tool_choice = responses_model._get_responses_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Responses API uses flatter tool structure
        assert tools == snapshot(
            [
                {
                    'type': 'function',
                    'name': 'get_weather',
                    'description': 'Test tool get_weather',
                    'parameters': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
                    'strict': False,
                }
            ]
        )
