from __future__ import annotations

from pydantic_ai.durable_exec._codec import IDENTITY_CODEC
from pydantic_ai.durable_exec._spec import DurabilityEngineSpec

# Module-level so the run-context module can word its errors without importing the capability.
TEMPORAL_ENGINE_SPEC = DurabilityEngineSpec(
    engine_name='Temporal',
    durable_unit_noun='activity',
    durable_unit_plural='activities',
    durable_container_noun='workflow',
    codec=IDENTITY_CODEC,
    unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
    wrapped_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
    toolset_lifecycles={
        'function': 'enter-outside-durable',
        'mcp': 'enter-outside-durable',
        'dynamic': 'enter-never',
    },
    tool_call_result_upgrade_lenient=False,
    journal_discovery=True,
    sequential_tools_in_durable_context=False,
    tool_config_key='temporal',
    # Workflow code and `@workflow.signal` handlers share Temporal's deterministic event loop, so a
    # token created in the workflow and fired from a signal handler cancels the run at the same
    # point on replay.
    accepts_cancellation_token=True,
)
