from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.durable_exec._toolset import EnqueueGuard
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._types import TaskConfig


def enqueue_guard() -> EnqueueGuard:
    return EnqueueGuard(
        '`ctx.enqueue()` is not supported inside Prefect task-wrapped tools because task-cache replay '
        'would drop the enqueued messages. Enqueue messages from flow-level code instead.'
    )


def with_non_retryable_errors(config: TaskConfig) -> TaskConfig:
    """Ensure framework configuration errors are not retried by Prefect."""
    config = config.copy()
    configured_condition = config.get('retry_condition_fn')

    async def retry_condition(task: Any, task_run: Any, state: Any) -> bool:
        result = state.result(raise_on_failure=False)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, (UserError, UnexpectedModelBehavior)):
            return False
        if configured_condition is None:
            return True
        decision = configured_condition(task, task_run, state)
        return await decision if inspect.isawaitable(decision) else decision

    config['retry_condition_fn'] = retry_condition
    return config


def resolve_tool_task_config(
    tool: ToolsetTool[Any] | None,
    tool_name: str,
    tool_task_config: Mapping[str, TaskConfig | None],
) -> TaskConfig | Literal[False]:
    """Resolve per-tool Prefect task config.

    Reads `tool.tool_def.metadata['prefect']` first, then falls back to the explicit
    `tool_task_config` dict keyed by tool name. Returns a `TaskConfig` dict (possibly
    empty), or `False` to skip task wrapping.
    """
    # Metadata set on the tool (via @toolset.tool(metadata={'prefect': ...}), with_metadata, or
    # the `SetToolMetadata` capability) is the primary path.
    if tool is not None and tool.tool_def.metadata is not None:
        metadata_config = tool.tool_def.metadata.get('prefect')
        if metadata_config is False:
            return False
        if metadata_config is not None:
            if not isinstance(metadata_config, dict):
                raise UserError(
                    f"Tool {tool_name!r} has invalid 'prefect' metadata: expected a dict "
                    f'(`TaskConfig`) or `False`, got {type(metadata_config).__name__}.'
                )
            return cast('TaskConfig', metadata_config)
    # Fallback: per-tool dict passed to the deprecated `PrefectAgent`. An explicit `None`
    # disables wrapping; a missing key means "use the base config".
    if tool_name in tool_task_config:
        fallback = tool_task_config[tool_name]
        return False if fallback is None else fallback
    return {}


def prefectify_toolset(
    toolset: AbstractToolset[AgentDepsT],
    mcp_task_config: TaskConfig,
    tool_task_config: TaskConfig,
    tool_task_config_by_name: dict[str, TaskConfig | None],
) -> AbstractToolset[AgentDepsT]:
    """Wrap a toolset to integrate it with Prefect.

    Args:
        toolset: The toolset to wrap.
        mcp_task_config: The Prefect task config to use for MCP server tasks.
        tool_task_config: The default Prefect task config to use for tool calls.
        tool_task_config_by_name: Per-tool task configuration. Keys are tool names, values are TaskConfig or None.
    """
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import prefectify_function_toolset

        return prefectify_function_toolset(
            wrapped=toolset,
            task_config=tool_task_config,
            tool_task_config=tool_task_config_by_name,
        )

    if isinstance(toolset, DynamicToolset):
        # The deprecated `PrefectAgent` still accepts anonymous dynamic toolsets and
        # must retain its existing inline behavior. The capability path validates IDs
        # before dispatching here.
        if toolset.id is None:
            return toolset
        from ._dynamic_toolset import prefectify_dynamic_toolset

        return prefectify_dynamic_toolset(
            wrapped=toolset,
            task_config=tool_task_config,
            tool_task_config={},
        )

    try:
        from pydantic_ai.mcp import MCPToolset

        from ._mcp_toolset import prefectify_mcp_toolset
    except ImportError:
        pass
    else:
        if isinstance(toolset, MCPToolset):
            return prefectify_mcp_toolset(
                wrapped=toolset,
                task_config=mcp_task_config,
            )

    return toolset
