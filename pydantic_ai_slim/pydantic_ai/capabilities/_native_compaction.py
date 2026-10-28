"""Compacting with the compaction API of the model serving a request, shared by `NativeCompaction` and `OpenAICompaction`."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.profiles import CompactionInstructionsMode

from .compaction import CompactionRequest, CompactionResult


async def compact_with_model(
    request: CompactionRequest,
    instructions_mode: CompactionInstructionsMode | None,
    *,
    focus_as_system_prompt: bool,
) -> CompactionResult | None:
    """Compact `request` with its model's `compact_messages`, keeping the turn being answered.

    `instructions_mode` is how the model's compaction takes instructions, which decides whether a focus is sent or the
    request left to the next capability. Shared with `OpenAICompaction`, whose stateless mode compacts on any
    `OpenAIResponsesModel` whatever its profile says.
    """
    instructions: str | None = None
    if request.focus is not None and (focus := request.focus.strip()):
        if instructions_mode == 'replaces_prompt' or (instructions_mode == 'system_prompt' and focus_as_system_prompt):
            instructions = focus
        else:
            return None

    kept_from = _turn_start(request.messages)
    if kept_from == 0:
        # Nothing before the turn being answered, so nothing to compact.
        return None
    model = request.model
    compact_context = ModelRequestContext(
        model=model,
        messages=request.messages[:kept_from],
        model_settings=request.model_settings,
        model_request_parameters=request.model_request_parameters,
    )
    response = await model.compact_messages(compact_context, instructions=instructions)
    part = next((part for part in response.parts if isinstance(part, CompactionPart)), None)
    if part is None:
        raise UnexpectedModelBehavior(f'{model.model_name!r} returned no compaction part from `compact_messages`.')
    return CompactionResult(
        messages=[response, *request.messages[kept_from:]],
        part=part,
        handler='native',
        usage=response.usage,
        kept_from=1,
    )


def _turn_start(messages: Sequence[ModelMessage]) -> int:
    """Where the turn being answered starts: its request, or the response whose tool calls it answers.

    Kept as it was, since a tool result can't be sent without its call.
    """
    start = len(messages) - 1
    if start > 0 and isinstance(previous := messages[start - 1], ModelResponse) and previous.tool_calls:
        start -= 1
    return start
