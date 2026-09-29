import warnings
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

    # *Known*, deliberately not *available*: this is every tool the request defines, including the
    # ones `visibility_of` withholds from the wire. `tool_choice` validation is about whether a name
    # exists at all — `RunContext.is_tool_available`'s narrower sense would reject a real but
    # not-yet-revealed tool as a typo.
    known_tool_names = set(model_request_parameters.tool_defs.keys())

    def _filter_withheld_tools(chosen_tool_names: set[str], *, has_output_fallback: bool = False) -> set[str]:
        # Named for the `ToolVisibility` literal it filters on rather than the looser "hidden":
        # only `'withheld'` names are absent from the wire. A `'deferred'` declaration sits in the
        # `tools` collection and a `'via_history'` definition travels on the tool-addition channel,
        # and OpenAI honors `tool_choice` forcing for both, so neither is withheld here — though
        # prose elsewhere would call all three "hidden from the model".
        withheld_tool_names = {
            tool.name
            for tool in model_request_parameters.function_tools
            if model_request_parameters.visibility_of(tool.name) == 'withheld'
        }
        filtered = chosen_tool_names - withheld_tool_names
        # At least one name that will actually reach the wire must survive: withheld names are
        # filtered here, and unknown names pass through by design (dynamic tool availability, see
        # `_check_invalid_tools`) — but a choice left with only unknown names would force the
        # provider toward tools it was never sent. A `ToolOrOutput` choice whose output tools
        # remain usable degrades to those instead of failing.
        if chosen_tool_names and not (filtered & known_tool_names) and not has_output_fallback:
            raise UserError(
                f'No tool in `tool_choice` is currently available: {sorted(chosen_tool_names)}. '
                'Hidden tools must be revealed with tool search, `load_capability`, or '
                '`ToolReturn.tools` before they can be forced.'
            )
        return filtered

    def _check_invalid_tools(chosen_tool_names: set[str], known_tools: set[str], *, known_label: str) -> None:
        invalid = chosen_tool_names - known_tools
        if not invalid:
            return
        if invalid == chosen_tool_names:
            raise UserError(f'Invalid tool names in `tool_choice`: {invalid}. {known_label}: {known_tools or "none"}')
        # Partial match: some chosen tools are valid, some aren't. This is allowed to support
        # dynamic tool availability (e.g. toolsets that expose different tools per request),
        # but we warn so typos don't pass silently.
        # https://github.com/pydantic/pydantic-ai/pull/3611#discussion_r2677602549
        warnings.warn(
            f'Some tools in `tool_choice` are not currently available and will be ignored: '
            f'{sorted(invalid)}. {known_label}: {sorted(known_tools)}',
            UserWarning,
            stacklevel=3,
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
        if all(
            model_request_parameters.visibility_of(tool.name) == 'withheld'
            for tool in model_request_parameters.function_tools
        ):
            # Nothing would reach the wire: the provider would see `required` alongside an empty
            # `tools` list and either reject the request or silently degrade.
            raise UserError(
                '`tool_choice` was set to "required", but every function tool is hidden until revealed. '
                'Reveal tools with tool search, `load_capability`, or `ToolReturn.tools`, '
                'or change `tool_choice`.'
            )
        return 'required'

    # list[str]: required, restricted to these tools
    elif isinstance(function_tool_choice, list):
        chosen_set = set(function_tool_choice)
        _check_invalid_tools(chosen_set, known_tool_names, known_label='Known tools')
        # A deferred declaration or a tool-addition definition is already on the wire and remains
        # callable; only tools absent from the wire cannot be forced by name.
        chosen_set = _filter_withheld_tools(chosen_set)

        if chosen_set == known_tool_names:
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
        known_function_tool_names = {t.name for t in model_request_parameters.function_tools}
        _check_invalid_tools(
            chosen_function_set,
            known_function_tool_names,
            known_label='Known function tools',
        )
        chosen_function_set = _filter_withheld_tools(chosen_function_set, has_output_fallback=bool(output_tool_names))

        allowed_tools = chosen_function_set | output_tool_names
        mode: Literal['auto', 'required'] = 'auto' if allow_direct_output else 'required'
        if allowed_tools == known_tool_names:
            return mode

        return (mode, allowed_tools)
    else:
        assert_never(function_tool_choice)
