from typing import Literal

from typing_extensions import assert_never

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ToolOrOutput

ResolvedToolChoice = Literal['none', 'auto', 'required'] | tuple[Literal['auto', 'required'], set[str]]


def resolve_tool_choice(  # noqa: C901
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ResolvedToolChoice:
    """Resolve user-facing tool_choice into a canonical form for providers.

    Pydantic AI distinguishes between function tools (e.g. user-registered via @agent.tool)
    and output tools (framework-internal for structured output). The user-facing
    `tool_choice` setting controls function tools only - this function resolves that
    into a canonical form that providers can use, incorporating output tools as needed.

    Args:
        model_settings: Optional settings containing the tool_choice value.
        model_request_parameters: Parameters describing available tools and output configuration.

    Input behavior:

        - `None` / `'auto'`: Returns `'auto'` if direct output allowed, else `'required'`.
        - `'none'` / `[]`: Disables function tools. If output tools exist, returns them with
            appropriate mode. Otherwise returns `'none'`.
        - `'required'`: Requires function tool use. Raises if no function tools are defined.
        - `list[str]`: Restricts to specified tools with `'required'` mode. Validates tool names.
        - `ToolOrOutput`: Combines specified function tools with all output tools.
            Returns `'auto'` mode if direct output is allowed, otherwise `'required'`.

    Raises:
        UserError: If tool_choice is incompatible with the available tools or output configuration.

    Returns:
        A canonical tool_choice value for providers:

        - `'none'`: No tools should be called. Only valid when direct output (text/image) is allowed.
        - `'auto'`: Model chooses whether to use tools. Direct output is allowed.
        - `'required'`: Model must use a tool. Direct output is not allowed.
        - `('auto', tool_names)`: Only these tools are available, direct output is allowed.
        - `('required', tool_names)`: Only these tools are available, must use one.
    """
    function_tool_choice = (model_settings or {}).get('tool_choice')

    allow_direct_output = model_request_parameters.allow_text_output or model_request_parameters.allow_image_output

    available_tools = set(model_request_parameters.tool_defs.keys())

    def _invalid_tools(chosen_tool_names: set[str], available_tools: set[str], *, available_label: str) -> None:
        invalid = chosen_tool_names - available_tools
        if invalid:
            raise UserError(
                f'Invalid tool names in `tool_choice`: {invalid}. {available_label}: {available_tools or "none"}'
            )

    # Default / auto
    if function_tool_choice in (None, 'auto'):
        return 'auto' if allow_direct_output else 'required'

    # none / []: disable function tools, but output tools may still exist
    elif function_tool_choice in ('none', []):
        output_tool_names = {t.name for t in model_request_parameters.output_tools}

        if output_tool_names:
            if allow_direct_output:
                mode: Literal['auto', 'required'] = 'auto'
            elif model_request_parameters.function_tools:
                mode = 'required'
            else:
                return 'required'  # only output tools exist and direct output isn't allowed

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
        chosen_set = set(function_tool_choice)
        # we'll only raise here if none of the chosen tools are valid https://github.com/pydantic/pydantic-ai/pull/3611#discussion_r2677602549
        if chosen_set - available_tools == chosen_set:
            _invalid_tools(chosen_set, available_tools, available_label='Available tools')

        if chosen_set == available_tools:
            return 'required'

        return ('required', chosen_set)

    # ToolOrOutput: specific function tools + all output tools or direct text/image output
    elif isinstance(function_tool_choice, ToolOrOutput):
        output_tool_names = {t.name for t in model_request_parameters.output_tools}

        if not function_tool_choice.function_tools:
            if output_tool_names:
                mode: Literal['auto', 'required'] = 'auto' if allow_direct_output else 'required'
                return (mode, output_tool_names)
            return 'none'

        chosen_function_set = set(function_tool_choice.function_tools)
        all_function_tool_names = {t.name for t in model_request_parameters.function_tools}
        # same as above - only raise if none are valid
        if chosen_function_set - all_function_tool_names == chosen_function_set:
            _invalid_tools(
                chosen_function_set,
                all_function_tool_names,
                available_label='Available function tools',
            )

        allowed_tools = chosen_function_set | output_tool_names
        mode: Literal['auto', 'required'] = 'auto' if allow_direct_output else 'required'
        if allowed_tools == available_tools:
            return mode

        return (mode, allowed_tools)
    else:
        assert_never(function_tool_choice)
