"""Repair malformed JSON before tool schema validation."""

from __future__ import annotations

import json

import json_repair
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, RawToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, ToolDefinition


class RepairToolArguments(AbstractCapability[AgentDepsT]):
    """Repair malformed JSON tool arguments with `json-repair`.

    Valid JSON and parsed arguments pass through unchanged. Repairs are heuristic;
    normal schema validation and retries still apply after a repair attempt.
    """

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
    ) -> RawToolArgs:
        """Repair malformed JSON arguments before normal tool schema validation."""
        if not isinstance(args, str):
            return args
        try:
            json.loads(args)
        except (json.JSONDecodeError, RecursionError):
            with ctx.tracer.start_as_current_span('repair_tool_arguments'):
                try:
                    return json_repair.repair_json(args, skip_json_loads=True, ensure_ascii=False)
                except (ValueError, RecursionError):
                    return args
        return args
