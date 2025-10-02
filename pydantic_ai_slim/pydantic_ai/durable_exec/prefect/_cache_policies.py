from dataclasses import fields, is_dataclass
from typing import Any

from prefect.cache_policies import INPUTS, RUN_ID, TASK_SOURCE, CachePolicy
from prefect.context import TaskRunContext


def _strip_timestamps(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts, excluding timestamp fields."""
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in fields(obj):
            if f.name != 'timestamp':
                value = getattr(obj, f.name)
                result[f.name] = _strip_timestamps(value)
        return result
    elif isinstance(obj, dict):
        return {k: _strip_timestamps(v) for k, v in obj.items() if k != 'timestamp'}
    elif isinstance(obj, list):
        return [_strip_timestamps(item) for item in obj]
    elif isinstance(obj, tuple):
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

        return INPUTS.compute_key(task_ctx, filtered_inputs, flow_parameters, **kwargs)


DEFAULT_PYDANTIC_AI_CACHE_POLICY = InputsWithoutTimestamps() + TASK_SOURCE + RUN_ID
