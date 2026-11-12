"""Repair malformed JSON before tool schema validation."""

from __future__ import annotations

import json

try:
    import json_repair
except ImportError as _import_error:
    raise ImportError(
        '`json-repair` is required for `RepairToolArguments`. Install `pydantic-ai-harness[coder]` to use `Coder`, '
        'or `pydantic-ai-harness[repair-tool-arguments]` to use `RepairToolArguments` on its own.'
    ) from _import_error

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
        """Repair malformed JSON arguments before normal tool schema validation.

        A repair is also written back to `call`, the part in the run's message history, so later
        requests replay the arguments that ran rather than the malformed ones. The model's original
        arguments and provider item ID are kept in `call.provider_details` when the call has a
        `provider_name`.
        """
        if not isinstance(args, str):
            return args
        try:
            json.loads(args)
        except (json.JSONDecodeError, RecursionError):
            with ctx.tracer.start_as_current_span('repair_tool_arguments'):
                try:
                    repaired = json_repair.repair_json(args, skip_json_loads=True, ensure_ascii=False)
                except (ValueError, RecursionError):
                    return args
            _write_back(call, original=args, repaired=repaired)
            return repaired
        return args


def _write_back(call: ToolCallPart, *, original: str, repaired: str) -> None:
    """Record *repaired* as the arguments of *call* in history, keeping the original where possible.

    The provider item ID is dropped too: OpenAI Responses replays its own stored copy of a call
    that keeps its item ID, so the repaired arguments would never reach the model.
    """
    if call.provider_name is not None:
        details = {'original_args': original}
        if call.id is not None:
            details['original_id'] = call.id
        call.provider_details = {**(call.provider_details or {}), **details}
    call.args = repaired
    call.id = None
