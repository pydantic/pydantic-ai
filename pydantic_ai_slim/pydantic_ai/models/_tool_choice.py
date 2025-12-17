from typing import Literal

from typing_extensions import assert_never

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
    # TODO: Remove
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
) -> (
    Literal['none', 'auto', 'required'] | tuple[list[str], Literal['auto', 'required']]
):  # second bool is text_allowed?
    """TODO: Update docstring.

    Validate and normalize tool_choice from model settings.

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

        # TODO: Second argument is to force a tool list filter

    Raises:
        UserError: If tool names in list[str] are not valid tool names.
    """
    function_tool_choice = (model_settings or {}).get('tool_choice')

    # TODO: think about builtin tools

    # TODO: is there a use case for "call a function tool or return text", i.e. "no output"?

    if function_tool_choice in (None, 'auto'):
        if model_request_parameters.allow_text_output:
            return 'auto'  # text or function tool or output tool
        else:
            return 'required'  # function tool or output tool
    elif function_tool_choice in ('none', []):
        # call output tools OR (if allowed) return text or image etc
        # TODO: Do we need to consider allow_image_output as well?
        output_tool_names = [t.name for t in model_request_parameters.output_tools]

        if model_request_parameters.output_tools:
            if model_request_parameters.allow_text_output:
                return (output_tool_names, 'auto')  # text or output tool
            elif model_request_parameters.function_tools:
                return (output_tool_names, 'required')
            else:
                return 'required'  # only output tools
        elif model_request_parameters.allow_text_output:  # text allowed, no output tools
            return 'none'
        else:  # pragma: no cover
            assert False, 'Either output_tools or allow_text_output must be set'
    elif function_tool_choice == 'required':
        function_tool_names = [t.name for t in model_request_parameters.function_tools]

        if model_request_parameters.function_tools:
            if model_request_parameters.output_tools:
                return (function_tool_names, 'required')
            else:
                return 'required'
        else:
            raise UserError(
                '`tool_choice` was set to "required", but no function tools are defined. '
                'Please define function tools or change `tool_choice` to "auto" or "none".'
            )
    elif isinstance(function_tool_choice, list):
        all_tool_names = model_request_parameters.tool_defs.keys()
        selected_tool_names = set(function_tool_choice)
        if selected_tool_names == all_tool_names:
            return 'required'

        invalid_names = selected_tool_names - all_tool_names
        if invalid_names:
            raise UserError(
                f'Invalid tool names in `tool_choice`: {invalid_names}. Available tools: {all_tool_names or "none"}'
            )
        return (list(selected_tool_names), 'required')
    else:
        assert_never(function_tool_choice)
