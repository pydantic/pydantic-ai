import warnings
from typing import Literal

from typing_extensions import assert_never

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ToolOrOutput

_AutoOrRequired = Literal['auto', 'required']
ResolvedToolChoice = Literal['none', 'auto', 'required'] | tuple[_AutoOrRequired, list[str]]


def resolve_tool_choice(  # noqa: C901
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ResolvedToolChoice:
    """Validate and normalize tool_choice settings into a canonical form for providers.

    Args:
        model_settings: Optional settings containing the tool_choice value.
        model_request_parameters: Parameters describing available tools and output configuration.

    Returns:
        A canonical tool_choice value for providers:

        - `'none'`: No tools should be called. Only valid when direct output (text/image) is allowed.
        - `'auto'`: Model chooses whether to use tools. Direct output is allowed.
        - `'required'`: Model must use a tool. Direct output is not allowed.
        - `(tool_names, 'auto')`: Only these tools are available, direct output is allowed.
        - `(tool_names, 'required')`: Only these tools are available, must use one.

    Raises:
        UserError: If tool_choice is incompatible with the available tools or output configuration.

    Input behavior:

        - `None` / `'auto'`: Returns `'auto'` if direct output allowed, else `'required'`.
        - `'none'` / `[]`: Disables function tools. If output tools exist, returns them with
          appropriate mode and warns. Otherwise returns `'none'`.
        - `'required'`: Requires function tool use. Raises if no function tools are defined.
        - `list[str]`: Restricts to specified tools with `'required'` mode. Validates tool names.
        - `ToolsPlusOutput`: Combines specified function tools with all output tools.
          Returns `'auto'` mode if direct output is allowed, otherwise `'required'`.
    """
    function_tool_choice = (model_settings or {}).get('tool_choice')

    allow_direct_output = model_request_parameters.allow_text_output or model_request_parameters.allow_image_output

    tool_names = set(model_request_parameters.tool_defs.keys())

    def _warn(msg: str) -> None:
        warnings.warn(msg, UserWarning)

    def _invalid_tool_names(names: set[str], available: set[str], *, available_label: str) -> None:
        invalid = names - available
        if invalid:
            raise UserError(f'Invalid tool names in `tool_choice`: {invalid}. {available_label}: {available or "none"}')

    # Default / auto
    if function_tool_choice in (None, 'auto'):
        return 'auto' if allow_direct_output else 'required'

    # none / []: disable function tools, but output tools may still exist
    elif function_tool_choice in ('none', []):
        output_tool_names = [t.name for t in model_request_parameters.output_tools]

        if output_tool_names:
            # If direct output is allowed, output-tools-only behaves like "auto";
            # otherwise if function tools exist, enforce "required" to avoid silent dead-ends.
            if allow_direct_output:
                mode: _AutoOrRequired = 'auto'
            elif model_request_parameters.function_tools:
                mode = 'required'
            else:
                return 'required'  # only output tools exist and direct output isn't allowed

            _warn(
                f"tool_choice='none' but output tools {output_tool_names} are defined - "
                f'defaulting to tool_choice={mode!r} for output tools only'
            )
            return (mode, output_tool_names)

        if allow_direct_output:
            return 'none'

        # pragma: no cover
        assert False, 'Either output_tools or allow_text_output/allow_image_output must be set'

    # required (only function tools allowed)
    elif function_tool_choice == 'required':
        if not model_request_parameters.function_tools:
            raise UserError(
                '`tool_choice` was set to "required", but no function tools are defined. '
                'Please define function tools or change `tool_choice` to "auto" or "none".'
            )
        return 'required'

    # list[str]: required, restricted to these tools
    elif isinstance(function_tool_choice, list):
        # dict.fromkeys keeps the order while removing duplicates
        # set() doesn't preserve order
        # and we need both the list (for the return value) and the set (for validation)
        chosen = list(dict.fromkeys(function_tool_choice))
        chosen_set = set(chosen)
        _invalid_tool_names(chosen_set, tool_names, available_label='Available tools')

        if chosen_set == tool_names:
            return 'required'

        return ('required', chosen)

    # ToolsPlusOutput: specific function tools + all output tools or direct text/image output
    elif isinstance(function_tool_choice, ToolOrOutput):
        output_tool_names = [t.name for t in model_request_parameters.output_tools]

        # stable order, unique
        if not function_tool_choice.function_tools:
            if output_tool_names:
                _warn('ToolsPlusOutput with empty function_tools - using output tools only')
                return 'auto' if allow_direct_output else 'required'
            _warn("ToolsPlusOutput with empty function_tools - defaulting to 'none'")
            return 'none'

        chosen_function = list(dict.fromkeys(function_tool_choice.function_tools))
        chosen_function_set = set(chosen_function)
        all_function_tool_names = {t.name for t in model_request_parameters.function_tools}

        _invalid_tool_names(
            chosen_function_set,
            all_function_tool_names,
            available_label='Available function tools',
        )

        allowed_tools = chosen_function + output_tool_names
        if set(allowed_tools) == tool_names:
            return 'required'

        # If direct output is allowed, use 'auto' mode to permit text/image responses
        mode: _AutoOrRequired = 'auto' if allow_direct_output else 'required'
        return (mode, allowed_tools)
    else:
        assert_never(function_tool_choice)
