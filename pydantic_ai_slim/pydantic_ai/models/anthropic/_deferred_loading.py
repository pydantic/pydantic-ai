"""Tools a request sends with `defer_loading`: their order in `tools`, and the guard against all of them deferred."""

from __future__ import annotations as _annotations

from anthropic.types.beta import BetaToolUnionParam

from ...exceptions import UserError
from ...messages import ModelMessage
from ...native_tools._tool_search import ToolSearchTool
from ...toolsets._tool_search import discovered_tool_names_in_order
from .. import ModelRequestParameters


def revealed_deferred_tool_order(
    messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
) -> list[str]:
    """Order revealed non-corpus deferred entries at the tail by their first reveal in history.

    Deferred entries are outside Anthropic's cache key, so this is a determinism rule, not a cache rule.
    A mid-run newcomer (`introduced_tool_names`) is declared deferred for the same reason and joins them.

    Without a tool-search surface, `resolve_request_tools` declares a deferred tool from the first request,
    so its reveal leaves it where earlier requests sent it: moving it would reorder entries already sent.
    """
    tool_defs = model_request_parameters.tool_defs
    pre_declared = not any(isinstance(tool, ToolSearchTool) for tool in model_request_parameters.native_tools)
    return [
        name
        for name in discovered_tool_names_in_order(messages)
        if name in tool_defs
        and (
            name in model_request_parameters.introduced_tool_names
            or (tool_defs[name].defer_loading and not pre_declared)
        )
        and tool_defs[name].with_native is None
        and model_request_parameters.visibility_of(name) == 'deferred'
    ]


def validate_wire_tools_not_all_deferred(tools: list[BetaToolUnionParam]) -> None:
    """Raise before sending a `tools` list in which every tool has `defer_loading=True`, which Anthropic rejects."""
    if tools and all(tool.get('defer_loading') is True for tool in tools):
        raise UserError(
            'Anthropic rejects a request whose wire tools all have `defer_loading=True`. '
            'Make at least one tool visible, or add a tool-search surface or capability catalog.'
        )


def append_revealed_tool_params(tools: list[BetaToolUnionParam], revealed_tool_order: list[str]) -> None:
    """Move revealed deferred definitions behind the request's stable tool segments."""
    by_name = {tool.get('name'): tool for tool in tools}
    revealed_names = [name for name in revealed_tool_order if name in by_name]
    if revealed_names:
        revealed_name_set = set(revealed_names)
        tools[:] = [tool for tool in tools if tool.get('name') not in revealed_name_set] + [
            by_name[name] for name in revealed_names
        ]
