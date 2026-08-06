"""Request preparation shared by deferred capabilities."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .._run_context import AgentDepsT, RunContext
from ..messages import ModelRequest, ToolAvailabilityDeltaPart
from ..toolsets._capability_owned import tool_defs_from_pre_definition_load_returns

if TYPE_CHECKING:
    from ..models import ModelRequestContext


def record_loaded_capability_tools(
    ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Record tools reconstructed from capability loads in pre-definition histories."""
    loaded = tool_defs_from_pre_definition_load_returns(ctx, request_context.model_request_parameters.function_tools)
    newly_loaded = [tool_def for name, tool_def in loaded.items() if name not in ctx.discovered_tool_names]
    if not newly_loaded:
        return request_context

    newly_loaded = sorted(newly_loaded, key=lambda tool_def: tool_def.name)
    tools_added = [tool_def.name for tool_def in newly_loaded]
    request_context.messages.append(ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=tools_added)]))
    ctx.discovered_tool_names.update(tools_added)
    request_context.model_request_parameters = replace(
        request_context.model_request_parameters,
        revealed_tool_names=request_context.model_request_parameters.revealed_tool_names | set(tools_added),
    )
    return request_context
