"""HuggingFace API tool_choice translation unit tests.

These tests verify the translation from PydanticAI's internal tool_choice format
to the HuggingFace API format. They complement the VCR-based integration tests by
explicitly asserting on the exact API values sent.

Key HuggingFace API format details:
- Uses strings: 'auto', 'none', 'required'
- Single specific tool: ChatCompletionInputToolChoiceClass with function name
- Multiple specific tools: 'required' + filtered tool list
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition, ToolKind

from ...conftest import try_import

with try_import() as imports_successful:
    from huggingface_hub import ChatCompletionInputToolChoiceClass

    from pydantic_ai.models.huggingface import HuggingFaceModel, HuggingFaceModelSettings
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='huggingface_hub not installed')


def make_tool(name: str, *, kind: ToolKind = 'function') -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f'Test tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}},
        kind=kind,
    )


@pytest.fixture
def huggingface_model() -> HuggingFaceModel:
    """Create a HuggingFace model for testing (no API calls made)."""
    return HuggingFaceModel('meta-llama/Llama-3.3-70B-Instruct', provider=HuggingFaceProvider(api_key='test-key'))


class TestToolChoiceTranslation:
    """Unit tests for HuggingFace tool_choice API format translation."""

    def test_auto_format(self, huggingface_model: HuggingFaceModel):
        """tool_choice='auto' passes 'auto' string to API."""
        settings: HuggingFaceModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = huggingface_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'auto'
        assert len(tools) == 1

    def test_none_format(self, huggingface_model: HuggingFaceModel):
        """tool_choice='none' passes 'none' string to API."""
        settings: HuggingFaceModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = huggingface_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'none'
        assert len(tools) == 1

    def test_required_format(self, huggingface_model: HuggingFaceModel):
        """tool_choice='required' passes 'required' string to API."""
        settings: HuggingFaceModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = huggingface_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'required'
        assert len(tools) == 1

    def test_single_tool_in_list_format(self, huggingface_model: HuggingFaceModel):
        """Single tool in list uses ChatCompletionInputToolChoiceClass format."""
        settings: HuggingFaceModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather'), make_tool('get_time')])

        tools, tool_choice = huggingface_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Single tool uses named format
        assert isinstance(tool_choice, ChatCompletionInputToolChoiceClass)
        assert tool_choice.function.name == 'get_weather'
        # Tools are filtered
        assert len(tools) == 1
        assert tools[0]['function']['name'] == 'get_weather'

    def test_multiple_tools_in_list_format(self, huggingface_model: HuggingFaceModel):
        """Multiple tools in list uses 'required' with filtered tools."""
        settings: HuggingFaceModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time'), make_tool('get_population')]
        )

        tools, tool_choice = huggingface_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Falls back to 'required' since multiple tools
        assert tool_choice == 'required'
        assert len(tools) == 2
        tool_names: set[str] = {t['function']['name'] for t in tools}
        assert tool_names == {'get_weather', 'get_time'}

    def test_tools_plus_output_format(self, huggingface_model: HuggingFaceModel):
        """ToolsPlusOutput uses 'required' with filtered tools (multiple tools)."""
        settings: HuggingFaceModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time')],
            output_tools=[make_tool('final_result', kind='output')],
        )

        tools, tool_choice = huggingface_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # 2 tools (get_weather + final_result), falls back to 'required'
        assert tool_choice == 'required'
        assert len(tools) == 2
        tool_names: set[str] = {t['function']['name'] for t in tools}
        assert tool_names == {'get_weather', 'final_result'}

    def test_none_with_output_tools_uses_auto(self, huggingface_model: HuggingFaceModel):
        """tool_choice='none' with output tools returns 'auto' with filtered tools."""
        settings: HuggingFaceModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather')],
            output_tools=[make_tool('final_result', kind='output')],
            allow_text_output=True,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            tools, tool_choice = huggingface_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Returns 'auto' with only output tools
        assert tool_choice == 'auto'
        assert len(tools) == 1
        assert tools[0]['function']['name'] == 'final_result'

    def test_no_tools_returns_none(self, huggingface_model: HuggingFaceModel):
        """When no tools are available, returns empty list and None."""
        settings: HuggingFaceModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[])

        tools, tool_choice = huggingface_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tools == []
        assert tool_choice is None

    def test_tool_definition_format(self, huggingface_model: HuggingFaceModel):
        """Verify the complete tool definition format sent to API."""
        settings: HuggingFaceModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, _ = huggingface_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tools == snapshot(
            [
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'description': 'Test tool get_weather',
                        'parameters': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
                    },
                }
            ]
        )
