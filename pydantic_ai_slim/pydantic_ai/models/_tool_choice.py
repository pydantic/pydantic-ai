from typing import Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

ToolChoiceValue = Literal['none', 'auto', 'required'] | list[str]
"""The validated tool_choice value: a mode string or a list of specific tool names."""


def filter_tools_for_choice(
    tool_choice: ToolChoiceValue | None,
    function_tools: list[ToolDefinition],
    output_tools: list[ToolDefinition],
) -> list[ToolDefinition]:
    """Filter tools based on the tool_choice value.

    This is a helper function for model implementations that need to filter
    tools before sending them to the API. Some providers support native
    tool filtering (like Google's `allowed_function_names` or OpenAI's
    `allowed_tools`).

    This is only called when necessary:
        when there are multiple specific tools to be called, and the provider doesn't support a "require one of multiple tools" feature.

    Args:
        tool_choice: The validated tool_choice value from `validate_tool_choice`.
        function_tools: The available function tools.
        output_tools: The available output tools (for structured output).

    Returns:
        The filtered list of tools to send to the API:
        - None or 'auto': all tools (function + output)
        - 'required': only function_tools (no output tools)
        - 'none': only output_tools (no function tools)
        - list[str]: only the specified tools from both lists (no auto-inclusion)
    """
    if tool_choice is None or tool_choice == 'auto':
        return [*function_tools, *output_tools]
    elif tool_choice == 'required':
        return list(function_tools)
    elif tool_choice == 'none':
        return list(output_tools)
    else:
        # list[str] - only include explicitly named tools from both lists
        allowed = set(tool_choice)
        return [t for t in [*function_tools, *output_tools] if t.name in allowed]


def validate_tool_choice(
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ToolChoiceValue | None:
    """Validate and normalize tool_choice from model settings.

    This is a public helper for model implementations to validate and normalize
    the user's `tool_choice` setting. Custom model implementations may need
    to call this function.

    Args:
        model_settings: The model settings containing tool_choice.
        model_request_parameters: The request parameters containing tool definitions.

    Returns:
        The normalized tool_choice value:
        - None if tool_choice was not set (provider uses default behavior)
        - 'none', 'auto', or 'required' for mode strings
        - list[str] for specific tool names (validated against available function and output tools)

    Raises:
        UserError: If tool names in list[str] are not valid tool names.
    """
    user_tool_choice = (model_settings or {}).get('tool_choice')

    if user_tool_choice is None:
        return None

    if user_tool_choice == 'none':
        return 'none'

    if user_tool_choice in ('auto', 'required'):
        if user_tool_choice == 'required' and not model_request_parameters.function_tools:
            raise UserError(
                '`tool_choice` was set to "required", but no function tools are defined. '
                'Please define function tools or change `tool_choice` to "auto" or "none".'
            )
        return user_tool_choice

    if isinstance(user_tool_choice, list):
        if not user_tool_choice:
            return 'none'
        function_tool_names = {t.name for t in model_request_parameters.function_tools}
        output_tool_names = {t.name for t in model_request_parameters.output_tools}
        all_tool_names = function_tool_names | output_tool_names
        invalid_names = set(user_tool_choice) - all_tool_names
        if invalid_names:
            raise UserError(
                f'Invalid tool names in `tool_choice`: {invalid_names}. Available tools: {all_tool_names or "none"}'
            )
        return list(user_tool_choice)

    return None  # pragma: no cover
