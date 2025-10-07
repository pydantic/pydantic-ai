from dataclasses import fields, is_dataclass
from typing import Any, TypeGuard

from prefect.cache_policies import RUN_ID, TASK_SOURCE, CachePolicy, Inputs
from prefect.context import TaskRunContext


def _is_dict(obj: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(obj, dict)


def _is_list(obj: Any) -> TypeGuard[list[Any]]:
    return isinstance(obj, list)


def _is_tuple(obj: Any) -> TypeGuard[tuple[Any, ...]]:
    return isinstance(obj, tuple)


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


class InputsWithoutTimestamps(CachePolicy):
    """Cache policy that computes a cache key based on inputs, ignoring nested 'timestamp' fields.

    This is similar to the INPUTS cache policy, but recursively removes all 'timestamp'
    fields from the messages inputs before computing the hash. This is useful when you want to
    cache based on the content of inputs but not their timestamps.
    """

    def compute_key(
        self,
        task_ctx: TaskRunContext,
        inputs: dict[str, Any],
        flow_parameters: dict[str, Any],
        **kwargs: Any,
    ) -> str | None:
        """Compute cache key from inputs with timestamps removed."""
        if not inputs:
            return None

        filtered_inputs = _strip_timestamps(inputs)

        # Exclude run_ctx from inputs as it contains non-hashable objects
        return Inputs(exclude=['run_ctx']).compute_key(task_ctx, filtered_inputs, flow_parameters, **kwargs)


DEFAULT_PYDANTIC_AI_CACHE_POLICY = InputsWithoutTimestamps() + TASK_SOURCE + RUN_ID
