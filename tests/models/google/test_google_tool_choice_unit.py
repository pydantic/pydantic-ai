"""Google AI API tool_choice translation unit tests.

These tests verify the translation from PydanticAI's internal tool_choice format
to the Google AI API format. They complement the VCR-based integration tests by
explicitly asserting on the exact API values sent.

Key Google API format details:
- Uses ToolConfigDict with function_calling_config containing mode and allowed_function_names
- Modes: FunctionCallingConfigMode.AUTO, NONE, ANY
- Specific tools: {mode: ANY, allowed_function_names: [...]}
- For tuple with 'auto' mode: filters tools, doesn't use allowed_function_names
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition, ToolKind

from ...conftest import try_import

with try_import() as imports_successful:
    from google.genai.types import FunctionCallingConfigMode

    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


def make_tool(name: str, *, kind: ToolKind = 'function') -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f'Test tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}},
        kind=kind,
    )


@pytest.fixture
def google_model() -> GoogleModel:
    """Create a Google model for testing (no API calls made)."""
    return GoogleModel('gemini-2.0-flash', provider=GoogleProvider(api_key='test-key'))


class TestToolChoiceTranslation:
    """Unit tests for Google AI tool_choice API format translation."""

    def test_auto_format(self, google_model: GoogleModel):
        """tool_choice='auto' uses mode=AUTO."""
        settings: GoogleModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_config, _image_config = google_model._get_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config == snapshot({'function_calling_config': {'mode': FunctionCallingConfigMode.AUTO}})
        assert tools is not None
        assert len(tools) == 1

    def test_none_format(self, google_model: GoogleModel):
        """tool_choice='none' uses mode=NONE."""
        settings: GoogleModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_config, _image_config = google_model._get_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config == snapshot({'function_calling_config': {'mode': FunctionCallingConfigMode.NONE}})
        assert tools is not None

    def test_required_format(self, google_model: GoogleModel):
        """tool_choice='required' uses mode=ANY."""
        settings: GoogleModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_config, _image_config = google_model._get_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        # 'required' maps to ANY in Google terminology
        assert tool_config == snapshot({'function_calling_config': {'mode': FunctionCallingConfigMode.ANY}})
        assert tools is not None

    def test_single_tool_in_list_format(self, google_model: GoogleModel):
        """Single tool in list uses mode=ANY with allowed_function_names."""
        settings: GoogleModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather'), make_tool('get_time')])

        tools, tool_config, _image_config = google_model._get_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        # Uses allowed_function_names to restrict which tools can be called
        assert tool_config == snapshot(
            {
                'function_calling_config': {
                    'mode': FunctionCallingConfigMode.ANY,
                    'allowed_function_names': ['get_weather'],
                }
            }
        )
        # All tools are sent - allowed_function_names enforces the restriction
        assert tools is not None
        assert len(tools) == 2

    def test_multiple_tools_in_list_format(self, google_model: GoogleModel):
        """Multiple tools in list uses mode=ANY with allowed_function_names."""
        settings: GoogleModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time'), make_tool('get_population')]
        )

        tools, tool_config, _image_config = google_model._get_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config == snapshot(
            {
                'function_calling_config': {
                    'mode': FunctionCallingConfigMode.ANY,
                    'allowed_function_names': ['get_weather', 'get_time'],
                }
            }
        )
        assert tools is not None
        assert len(tools) == 3

    def test_tools_plus_output_format(self, google_model: GoogleModel):
        """ToolsPlusOutput uses mode=ANY with allowed_function_names."""
        settings: GoogleModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time')],
            output_tools=[make_tool('final_result', kind='output')],
        )

        tools, tool_config, _image_config = google_model._get_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tool_config == snapshot(
            {
                'function_calling_config': {
                    'mode': FunctionCallingConfigMode.ANY,
                    'allowed_function_names': ['get_weather', 'final_result'],
                }
            }
        )
        assert tools is not None
        assert len(tools) == 3

    def test_none_with_output_tools_uses_auto_mode(self, google_model: GoogleModel):
        """tool_choice='none' with output tools returns mode=AUTO with filtered tools."""
        settings: GoogleModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather')],
            output_tools=[make_tool('final_result', kind='output')],
            allow_text_output=True,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            tools, tool_config, _image_config = google_model._get_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        # Returns AUTO mode with only output tools
        assert tool_config == snapshot({'function_calling_config': {'mode': FunctionCallingConfigMode.AUTO}})
        # Tools are filtered to only output tools
        assert tools is not None
        assert len(tools) == 1

    def test_no_tools_returns_none(self, google_model: GoogleModel):
        """When no tools are available, returns None for tools and tool_config."""
        settings: GoogleModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[])

        tools, tool_config, _image_config = google_model._get_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        assert tools is None
        assert tool_config is None

    def test_tool_definition_format(self, google_model: GoogleModel):
        """Verify the tool definition format includes function_declarations."""
        settings: GoogleModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, _, _image_config = google_model._get_tool_config(params, settings)  # pyright: ignore[reportPrivateUsage]

        # Google wraps tools in function_declarations
        assert tools == snapshot(
            [
                {
                    'function_declarations': [
                        {
                            'name': 'get_weather',
                            'description': 'Test tool get_weather',
                            'parameters_json_schema': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
                        }
                    ]
                }
            ]
        )
