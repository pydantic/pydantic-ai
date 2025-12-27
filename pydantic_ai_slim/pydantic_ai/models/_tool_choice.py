import warnings
from typing import Literal

from typing_extensions import assert_never

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ToolsPlusOutput


def validate_tool_choice(  # noqa: C901
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> Literal['none', 'auto', 'required'] | tuple[list[str], Literal['auto', 'required']]:
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

        - `None` / `'auto'`: Returns `'auto'` if text output allowed, else `'required'`.
        - `'none'` / `[]`: Disables function tools. If output tools exist, returns them with
          appropriate mode and warns. Otherwise returns `'none'`.
        - `'required'`: Requires function tool use. Raises if output tools exist or no
          function tools are defined.
        - `list[str]`: Restricts to specified tools with `'required'` mode. Validates tool names.
        - `ToolsPlusOutput`: Combines specified function tools with all output tools.
    """
    tool_choice = (model_settings or {}).get('tool_choice')

    allow_direct_output = model_request_parameters.allow_text_output or model_request_parameters.allow_image_output

    tool_defs_keys = model_request_parameters.tool_defs.keys()  # includes builtin tools when present

    def _warn(msg: str) -> None:
        warnings.warn(msg, UserWarning, stacklevel=6)

    def _invalid(names: set[str], available: set[str], *, available_label: str) -> None:
        invalid = names - available
        if invalid:
            raise UserError(f'Invalid tool names in `tool_choice`: {invalid}. {available_label}: {available or "none"}')

    # Default / auto
    if tool_choice in (None, 'auto'):
        return 'auto' if model_request_parameters.allow_text_output else 'required'

    # none / []: disable function tools, but output tools may still exist
    if tool_choice in ('none', []):
        output_tool_names = [t.name for t in model_request_parameters.output_tools]

        if output_tool_names:
            # If direct output is allowed, output-tools-only behaves like "auto";
            # otherwise if function tools exist, enforce "required" to avoid silent dead-ends.
            if allow_direct_output:
                mode: Literal['auto', 'required'] = 'auto'
            elif model_request_parameters.function_tools:
                mode = 'required'
            else:
                return 'required'  # only output tools exist and direct output isn't allowed

            _warn(
                f"tool_choice='none' but output tools {output_tool_names} are defined - "
                f'defaulting to tool_choice={mode!r} for output tools only'
            )
            return (output_tool_names, mode)

        if allow_direct_output:
            return 'none'

        # pragma: no cover
        assert False, 'Either output_tools or allow_text_output/allow_image_output must be set'

    # required (only function tools allowed)
    if tool_choice == 'required':
        if model_request_parameters.output_tools:
            raise UserError(
                "`tool_choice='required'` is incompatible with output types. "
                'Use `ToolsPlusOutput` to specify function tools while allowing structured output.'
            )
        if not model_request_parameters.function_tools:
            raise UserError(
                '`tool_choice` was set to "required", but no function tools are defined. '
                'Please define function tools or change `tool_choice` to "auto" or "none".'
            )
        return 'required'

    # list[str]: required, restricted to these tools
    if isinstance(tool_choice, list):
        # stable order, unique
        chosen = list(dict.fromkeys(tool_choice))
        chosen_set = set(chosen)

        if chosen_set == set(tool_defs_keys):
            return 'required'

        _invalid(chosen_set, set(tool_defs_keys), available_label='Available tools')
        return (chosen, 'required')

    # ToolsPlusOutput: specific function tools + all output tools
    if isinstance(tool_choice, ToolsPlusOutput):
        output_tool_names = [t.name for t in model_request_parameters.output_tools]

        # stable order, unique
        if not tool_choice.function_tools:
            _warn("ToolsPlusOutput with empty function_tools - defaulting to 'none'")
            if output_tool_names:
                return (output_tool_names, 'auto' if allow_direct_output else 'required')
            return 'none'

        if not output_tool_names:
            _warn('ToolsPlusOutput used but no output tools exist - defaulting to list[str] behavior')

        chosen_function = list(dict.fromkeys(tool_choice.function_tools))
        chosen_function_set = set(chosen_function)
        all_function_tool_names = {t.name for t in model_request_parameters.function_tools}

        _invalid(
            chosen_function_set,
            all_function_tool_names,
            available_label='Available function tools',
        )

        allowed_tools = chosen_function + output_tool_names
        if set(allowed_tools) == set(tool_defs_keys):
            return 'required'
        return (allowed_tools, 'required')

    assert_never(tool_choice)
