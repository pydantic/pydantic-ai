from dataclasses import fields, is_dataclass
from typing import Any, TypeGuard

from prefect.cache_policies import INPUTS, RUN_ID, TASK_SOURCE, CachePolicy
from prefect.context import TaskRunContext
from prefect.utilities.hashing import hash_objects
from pydantic import BaseModel

from pydantic_ai import ToolsetTool
from pydantic_ai._utils import TOOL_CALL_ID_PREFIX
from pydantic_ai.tools import RunContext

_NON_SERIALIZABLE = '<non-serializable>'


def _is_dict(obj: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(obj, dict)


def _is_list(obj: Any) -> TypeGuard[list[Any]]:
    return isinstance(obj, list)


def _is_tuple(obj: Any) -> TypeGuard[tuple[Any, ...]]:
    return isinstance(obj, tuple)


def _is_toolset_tool(obj: Any) -> TypeGuard[ToolsetTool]:
    return isinstance(obj, ToolsetTool)


def _is_run_context(obj: Any) -> TypeGuard[RunContext[object]]:
    return isinstance(obj, RunContext)


def _cacheable_deps(deps: Any) -> Any:
    """Project `deps` for cache-key hashing, excluding non-serializable values.

    Dependencies routinely hold live resources (HTTP clients, DB connections, locks) that
    Prefect can't hash; those values are replaced with a stable sentinel rather than failing
    the task, while serializable siblings still fork the key. Plain non-dataclass and
    non-`BaseModel` objects are treated as indivisible values.
    """
    projected = _strip_cache_excluded_fields(deps)

    def exclude_non_serializable(value: Any) -> Any:
        if hash_objects(value, raise_on_failure=False) is not None:
            return value
        if _is_dict(value):
            return {key: exclude_non_serializable(item) for key, item in value.items()}
        if _is_list(value):
            return [exclude_non_serializable(item) for item in value]
        if _is_tuple(value):
            return tuple(exclude_non_serializable(item) for item in value)
        if isinstance(value, BaseModel):
            return {name: exclude_non_serializable(getattr(value, name)) for name in type(value).model_fields}
        return _NON_SERIALIZABLE

    return exclude_non_serializable(projected)


def _replace_run_context(
    inputs: dict[str, Any],
) -> Any:
    """Replace RunContext objects with a dict containing only hashable fields."""
    for key, value in inputs.items():
        if _is_run_context(value):
            inputs[key] = {
                'deps': _cacheable_deps(value.deps),
                'agent': value.agent.name if value.agent is not None else None,
                'model': value.model.model_id,
                'retries': value.retries,
                'tool_call_id': value.tool_call_id,
                'tool_name': value.tool_name,
                'tool_call_approved': value.tool_call_approved,
                'tool_call_metadata': value.tool_call_metadata,
                'retry': value.retry,
                'max_retries': value.max_retries,
                'run_step': value.run_step,
                # Deferred-load state must be part of the key: two runs identical except for which
                # capabilities/tools have been loaded see different tools and must not share a cache
                # entry. Sorted for a deterministic key (sets have no stable iteration order).
                # `capability_loaded` is deliberately omitted (unlike Temporal's serializer, which
                # round-trips every field a hook might read): it's derived from `loaded_capability_ids`
                # plus the static capability set, so it adds no entropy the two fields above don't.
                'loaded_capability_ids': sorted(value.loaded_capability_ids),
                'discovered_tool_names': sorted(value.discovered_tool_names),
                # A tool or capability may read `usage_limits` to fork its behavior (e.g. budget
                # disclosure), so two runs identical except for their limits must not share a cache
                # entry. `_strip_cache_excluded_fields` recurses into the `UsageLimits` dataclass to
                # hash it by value; `None` (bare/synthetic context) hashes distinctly.
                'usage_limits': value.usage_limits,
            }

    return inputs


_CACHE_EXCLUDED_FIELDS = frozenset({'timestamp', 'run_id', 'conversation_id'})
"""Framework dataclass fields excluded from cache key computation as they vary per-run."""


def _strip_cache_excluded_fields(
    obj: Any | dict[str, Any] | list[Any] | tuple[Any, ...],
) -> Any:
    """Recursively convert dataclasses to dicts, excluding cache-irrelevant fields.

    Only framework (`pydantic_ai.*`) dataclass fields are excluded. Fields on user-provided
    dataclasses and plain dict keys are meaningful input data and must fork the key even when
    they share a name with a per-run framework field.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        module = type(obj).__module__
        is_framework = module == 'pydantic_ai' or module.startswith('pydantic_ai.')
        excluded_fields = _CACHE_EXCLUDED_FIELDS if is_framework else ()
        for f in fields(obj):
            if f.name not in excluded_fields:
                value = getattr(obj, f.name)
                if (
                    is_framework
                    and f.name == 'tool_call_id'
                    and isinstance(value, str)
                    and value.startswith(TOOL_CALL_ID_PREFIX)
                ):
                    value = '<framework-generated>'
                result[f.name] = _strip_cache_excluded_fields(value)
        return result
    elif _is_dict(obj):
        return {k: _strip_cache_excluded_fields(v) for k, v in obj.items()}
    elif _is_list(obj):
        return [_strip_cache_excluded_fields(item) for item in obj]
    elif _is_tuple(obj):
        return tuple(_strip_cache_excluded_fields(item) for item in obj)
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

    Computes a cache key based on inputs, ignoring per-run fields like 'timestamp' and 'run_id',
    and serializing RunContext objects to only include hashable fields.
    """

    def compute_key(
        self,
        task_ctx: TaskRunContext,
        inputs: dict[str, Any],
        flow_parameters: dict[str, Any],
        **kwargs: Any,
    ) -> str | None:
        """Compute cache key from inputs with per-run fields removed and RunContext serialized."""
        if not inputs:
            return None

        inputs_without_toolsets = _replace_toolsets(inputs)
        inputs_with_hashable_context = _replace_run_context(inputs_without_toolsets)
        filtered_inputs = _strip_cache_excluded_fields(inputs_with_hashable_context)

        return INPUTS.compute_key(task_ctx, filtered_inputs, flow_parameters, **kwargs)


DEFAULT_PYDANTIC_AI_CACHE_POLICY = PrefectAgentInputs() + TASK_SOURCE + RUN_ID
