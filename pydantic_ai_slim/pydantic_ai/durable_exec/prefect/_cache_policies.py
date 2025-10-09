from dataclasses import fields, is_dataclass
from typing import Any, TypeGuard

from prefect.cache_policies import INPUTS, RUN_ID, TASK_SOURCE, CachePolicy
from prefect.context import TaskRunContext

from pydantic_ai import ToolsetTool
from pydantic_ai.tools import RunContext


def _is_dict(obj: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(obj, dict)


def _is_list(obj: Any) -> TypeGuard[list[Any]]:
    return isinstance(obj, list)


def _is_tuple(obj: Any) -> TypeGuard[tuple[Any, ...]]:
    return isinstance(obj, tuple)


def _is_toolset_tool(obj: Any) -> TypeGuard[ToolsetTool]:
    return isinstance(obj, ToolsetTool)


def _replace_run_context(
    inputs: dict[str, Any],
) -> Any:
    """Replace RunContext objects with a dict containing only hashable fields."""
    for key, value in inputs.items():
        if isinstance(value, RunContext):
            inputs[key] = {
                'retries': value.retries,
                'tool_call_id': value.tool_call_id,
                'tool_name': value.tool_name,
                'tool_call_approved': value.tool_call_approved,
                'retry': value.retry,
                'max_retries': value.max_retries,
                'run_step': value.run_step,
            }

    return inputs


def _strip_timestamps(
    obj: Any | dict[str, Any] | list[Any] | tuple[Any, ...],
) -> Any:
    """Recursively convert dataclasses to dicts, excluding timestamp fields."""
    if is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for f in fields(obj):
            if f.name != 'timestamp':
                value = getattr(obj, f.name)
                result[f.name] = _strip_timestamps(value)
        return result
    elif _is_dict(obj):
        return {k: _strip_timestamps(v) for k, v in obj.items() if k != 'timestamp'}
    elif _is_list(obj):
        return [_strip_timestamps(item) for item in obj]
    elif _is_tuple(obj):
        return tuple(_strip_timestamps(item) for item in obj)
    return obj


def _replace_toolsets(
    inputs: dict[str, Any],
) -> Any:
    """Replace Toolset objects with a dict containing only hashable fields."""
    inputs = inputs.copy()
    for key, value in inputs.items():
        if _is_toolset_tool(value):
            inputs[key] = {field.name: getattr(value, field.name) for field in fields(value) if field.name != 'toolset'}
    return inputs


class PrefectAgentInputs(CachePolicy):
    """Cache policy designed to handle input hashing for PrefectAgent cache keys.

    Computes a cache key based on inputs, ignoring nested 'timestamp' fields
    and serializing RunContext objects to only include hashable fields.
    """

    def compute_key(
        self,
        task_ctx: TaskRunContext,
        inputs: dict[str, Any],
        flow_parameters: dict[str, Any],
        **kwargs: Any,
    ) -> str | None:
        """Compute cache key from inputs with timestamps removed and RunContext serialized."""
        if not inputs:
            return None

        inputs_without_toolsets = _replace_toolsets(inputs)
        inputs_with_hashable_context = _replace_run_context(inputs_without_toolsets)
        filtered_inputs = _strip_timestamps(inputs_with_hashable_context)

        return INPUTS.compute_key(task_ctx, filtered_inputs, flow_parameters, **kwargs)


DEFAULT_PYDANTIC_AI_CACHE_POLICY = PrefectAgentInputs() + TASK_SOURCE + RUN_ID
