"""The Agent Control contract: what a published `AgentConfig` holds, and what it does to a request.

This is Pydantic AI's copy of the framework-neutral core in logfire's `logfire.agent_control`. Each
module here is logfire's module of the same name, so a resync is a plain diff; the modules this
package has no use for (`_control`, `_hint`, `_merge`) are not carried, and `report_unmatched`, which
nothing here calls, is left out of `_reporting`. Each copy of the contract -- this one, the TypeScript
SDK's, and the Logfire UI's -- is checked against the same `SCHEMA_SHA256` and the same cross-language
vectors under `tests/harness/logfire_variables/contract/spec/`, which is what keeps them from
drifting. Change a rule there first, and in every copy after.
"""

from ._apply import (
    AppliedInstructions,
    AppliedSettings,
    AppliedTools,
    Block,
    CollisionScope,
    ToolDef,
    apply_instructions,
    apply_settings,
    apply_tool_definitions,
    build_baseline,
    canonical_settings,
)
from ._config import (
    AgentConfig,
    AgentConfigSettings,
    InstructionBlock,
    ParameterOverride,
    ToolDefinitionOverride,
    ToolKey,
)
from ._names import AGENT_VARIABLE_PREFIX, agent_variable_name, normalize_agent_name
from ._reporting import ApplyIssue, ApplyIssueReason, OnUnmatched, UnmatchedConfigError, report_issues
from ._schema import AGENT_CONFIG_JSON_SCHEMA, MAX_MODEL_FACING_TEXT_LENGTH, SCHEMA_SHA256, canonical_json
from ._support import AgentSupport, Destination, Section
from ._units import MAX_TIMEOUT_MILLISECONDS, MAX_TIMEOUT_SECONDS, is_representable_timeout, to_milliseconds

__all__ = (
    'AGENT_CONFIG_JSON_SCHEMA',
    'AGENT_VARIABLE_PREFIX',
    'MAX_MODEL_FACING_TEXT_LENGTH',
    'MAX_TIMEOUT_MILLISECONDS',
    'MAX_TIMEOUT_SECONDS',
    'SCHEMA_SHA256',
    'AgentConfig',
    'AgentConfigSettings',
    'AgentSupport',
    'AppliedInstructions',
    'AppliedSettings',
    'AppliedTools',
    'ApplyIssue',
    'ApplyIssueReason',
    'Block',
    'CollisionScope',
    'Destination',
    'InstructionBlock',
    'OnUnmatched',
    'ParameterOverride',
    'Section',
    'ToolDef',
    'ToolDefinitionOverride',
    'ToolKey',
    'UnmatchedConfigError',
    'agent_variable_name',
    'apply_instructions',
    'apply_settings',
    'apply_tool_definitions',
    'build_baseline',
    'canonical_json',
    'canonical_settings',
    'is_representable_timeout',
    'normalize_agent_name',
    'report_issues',
    'to_milliseconds',
)
