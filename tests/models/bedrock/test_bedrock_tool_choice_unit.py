"""AWS Bedrock API tool_choice translation unit tests.

These tests verify the translation from PydanticAI's internal tool_choice format
to the AWS Bedrock API format. They complement the VCR-based integration tests by
explicitly asserting on the exact API values sent.

Key Bedrock API format details:
- Uses dicts: {auto: {}}, {any: {}}, {tool: {name: X}}
- No native 'none' mode - returns None to omit tools entirely
- Single specific tool: {tool: {name: X}}
- Multiple specific tools: {any: {}} + filtered tool list
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition, ToolKind

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='boto3 not installed')


def make_tool(name: str, *, kind: ToolKind = 'function') -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f'Test tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}},
        kind=kind,
    )


@pytest.fixture
def bedrock_model() -> BedrockConverseModel:
    """Create a Bedrock model for testing (no API calls made)."""
    return BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-20250514-v1:0',
        provider=BedrockProvider(region_name='us-east-1'),
    )


class TestToolChoiceTranslation:
    """Unit tests for Bedrock tool_choice API format translation."""

    def test_auto_format(self, bedrock_model: BedrockConverseModel):
        """tool_choice='auto' uses {auto: {}} format."""
        settings: BedrockModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tool_config = bedrock_model._map_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config is not None
        assert tool_config.get('toolChoice') == snapshot({'auto': {}})
        assert len(tool_config['tools']) == 1

    def test_none_returns_none(self, bedrock_model: BedrockConverseModel):
        """tool_choice='none' returns None (Bedrock has no native 'none' mode)."""
        settings: BedrockModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tool_config = bedrock_model._map_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        # Bedrock doesn't support 'none', so we don't send tools at all
        assert tool_config is None

    def test_required_format(self, bedrock_model: BedrockConverseModel):
        """tool_choice='required' uses {any: {}} format."""
        settings: BedrockModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tool_config = bedrock_model._map_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        # 'required' maps to 'any' in Bedrock terminology
        assert tool_config is not None
        assert tool_config.get('toolChoice') == snapshot({'any': {}})

    def test_single_tool_in_list_format(self, bedrock_model: BedrockConverseModel):
        """Single tool in list uses {tool: {name: X}} format."""
        settings: BedrockModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather'), make_tool('get_time')])

        tool_config = bedrock_model._map_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config is not None
        # Single tool uses named format
        assert tool_config.get('toolChoice') == snapshot({'tool': {'name': 'get_weather'}})
        # Tools are filtered
        assert len(tool_config['tools']) == 1

    def test_multiple_tools_in_list_format(self, bedrock_model: BedrockConverseModel):
        """Multiple tools in list uses {any: {}} with filtered tools."""
        settings: BedrockModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time'), make_tool('get_population')]
        )

        tool_config = bedrock_model._map_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config is not None
        # Falls back to 'any' since multiple tools specified
        assert tool_config.get('toolChoice') == snapshot({'any': {}})
        # Tools are filtered
        assert len(tool_config['tools']) == 2

    def test_tools_plus_output_format(self, bedrock_model: BedrockConverseModel):
        """ToolsPlusOutput uses appropriate format based on tool count."""
        settings: BedrockModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time')],
            output_tools=[make_tool('final_result', kind='output')],
        )

        tool_config = bedrock_model._map_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config is not None
        # 2 tools (get_weather + final_result), so uses 'any'
        assert tool_config.get('toolChoice') == snapshot({'any': {}})
        assert len(tool_config['tools']) == 2

    def test_none_with_output_tools_uses_auto(self, bedrock_model: BedrockConverseModel):
        """tool_choice='none' with output tools returns {auto: {}} with filtered tools."""
        settings: BedrockModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather')],
            output_tools=[make_tool('final_result', kind='output')],
            allow_text_output=True,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            tool_config = bedrock_model._map_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]]

        assert tool_config is not None
        # Uses auto mode with only output tools
        assert tool_config.get('toolChoice') == snapshot({'auto': {}})
        assert len(tool_config['tools']) == 1

    def test_no_tools_returns_none(self, bedrock_model: BedrockConverseModel):
        """When no tools are available, returns None."""
        settings: BedrockModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[])

        tool_config = bedrock_model._map_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config is None

    def test_tool_definition_format(self, bedrock_model: BedrockConverseModel):
        """Verify the complete tool definition format sent to API."""
        settings: BedrockModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tool_config = bedrock_model._map_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config is not None
        assert tool_config['tools'] == snapshot(
            [
                {
                    'toolSpec': {
                        'name': 'get_weather',
                        'description': 'Test tool get_weather',
                        'inputSchema': {'json': {'type': 'object', 'properties': {'x': {'type': 'string'}}}},
                    }
                }
            ]
        )
