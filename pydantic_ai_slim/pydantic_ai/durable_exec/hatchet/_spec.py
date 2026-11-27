from __future__ import annotations

from pydantic_core import PydanticSerializationError

from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._spec import DurabilityEngineSpec
from pydantic_ai.exceptions import UserError


def _serialization_failure(error: Exception) -> BaseException:
    if isinstance(error, PydanticSerializationError | TypeError):
        return UserError(
            f'A value passed to a Hatchet task failed to be serialized ({error}). Hatchet task inputs and '
            "outputs are JSON, so every value crossing into a task must be serializable with Pydantic's "
            '`TypeAdapter`. Besides `deps`, this includes `model_settings`, the `RunContext` `metadata` and '
            '`tool_call_metadata`, and tool `metadata`.'
        )
    return error


# Module-level so the run-context module can word its errors without importing the capability.
HATCHET_ENGINE_SPEC = DurabilityEngineSpec(
    engine_name='Hatchet',
    durable_unit_noun='task',
    durable_container_noun='durable task',
    # Task inputs and outputs are JSON, so values are reduced to JSON-compatible payloads inside the
    # task (where a non-serializable value fails the task) and validated back outside it.
    codec=JSON_CODEC,
    serialization_failure=_serialization_failure,
    wrapped_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
    # A child task may run on another worker, so the durable task can't hand it an entered toolset.
    toolset_lifecycles={
        'function': 'enter-outside-durable',
        'mcp': 'enter-outside-durable',
        'dynamic': 'enter-never',
    },
    unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
    journal_discovery=True,
    sequential_tools_in_durable_context=False,
    tool_config_key='hatchet',
)
