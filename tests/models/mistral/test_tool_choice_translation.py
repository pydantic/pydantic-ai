"""Mistral API tool_choice translation unit tests.

These tests verify the translation from PydanticAI's internal tool_choice format
to the Mistral API format. They complement the VCR-based integration tests by
explicitly asserting on the exact API values sent.

Key Mistral API format details:
- Uses strings: 'auto', 'none', 'required'
- Specific tools: filters tool list + uses 'required' mode
- No named tool choice format - always filters + mode string
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition, ToolKind

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='mistralai not installed')


def make_tool(name: str, *, kind: ToolKind = 'function') -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f'Test tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}},
        kind=kind,
    )


@pytest.fixture
def mistral_model() -> MistralModel:
    """Create a Mistral model for testing (no API calls made)."""
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test-key'))


class TestToolChoiceTranslation:
    """Unit tests for Mistral tool_choice API format translation."""

    def test_auto_format(self, mistral_model: MistralModel):
        """tool_choice='auto' passes 'auto' string to API."""
        settings: ModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = mistral_model._get_tool_choice(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'auto'
        assert tools is not None
        assert len(tools) == 1

    def test_none_format(self, mistral_model: MistralModel):
        """tool_choice='none' passes 'none' string to API."""
        settings: ModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = mistral_model._get_tool_choice(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'none'
        assert tools is not None

    def test_required_format(self, mistral_model: MistralModel):
        """tool_choice='required' passes 'required' string to API."""
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = mistral_model._get_tool_choice(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'required'
        assert tools is not None

    def test_single_tool_in_list_format(self, mistral_model: MistralModel):
        """Single tool in list uses 'required' with filtered tools."""
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather'), make_tool('get_time')])

        tools, tool_choice = mistral_model._get_tool_choice(params, settings)  # pyright: ignore[reportPrivateUsage]

        # Mistral doesn't have named tool format, uses 'required' with filtered list
        assert tool_choice == 'required'
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].function.name == 'get_weather'

    def test_multiple_tools_in_list_format(self, mistral_model: MistralModel):
        """Multiple tools in list uses 'required' with filtered tools."""
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time'), make_tool('get_population')]
        )

        tools, tool_choice = mistral_model._get_tool_choice(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'required'
        assert tools is not None
        assert len(tools) == 2
        tool_names = {t.function.name for t in tools}
        assert tool_names == {'get_weather', 'get_time'}

    def test_tools_plus_output_format(self, mistral_model: MistralModel):
        """ToolsPlusOutput uses 'required' with combined function and output tools."""
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time')],
            output_tools=[make_tool('final_result', kind='output')],
        )

        tools, tool_choice = mistral_model._get_tool_choice(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'required'
        assert tools is not None
        assert len(tools) == 2
        tool_names = {t.function.name for t in tools}
        assert tool_names == {'get_weather', 'final_result'}

    def test_none_with_output_tools_uses_auto(self, mistral_model: MistralModel):
        """tool_choice='none' with output tools returns 'auto' with filtered tools."""
        settings: ModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather')],
            output_tools=[make_tool('final_result', kind='output')],
            allow_text_output=True,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            tools, tool_choice = mistral_model._get_tool_choice(params, settings)  # pyright: ignore[reportPrivateUsage]

        # Returns 'auto' with only output tools
        assert tool_choice == 'auto'
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].function.name == 'final_result'

    def test_no_tools_returns_none(self, mistral_model: MistralModel):
        """When no tools are available, returns None for both."""
        settings: ModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[])

        tools, tool_choice = mistral_model._get_tool_choice(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tools is None
        assert tool_choice is None

    def test_tool_definition_format(self, mistral_model: MistralModel):
        """Verify the tool definition uses MistralTool wrapper."""
        settings: ModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, _tool_choice = mistral_model._get_tool_choice(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tools is not None
        assert len(tools) == 1
        # Check MistralTool structure
        tool = tools[0]
        assert tool.function.name == 'get_weather'
        assert tool.function.description == 'Test tool get_weather'
        assert tool.function.parameters == snapshot({'type': 'object', 'properties': {'x': {'type': 'string'}}})
