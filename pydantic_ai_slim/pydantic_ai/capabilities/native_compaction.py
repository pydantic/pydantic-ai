"""The [`NativeCompaction`][pydantic_ai.capabilities.NativeCompaction] capability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

from .abstract import AbstractCapability
from .compaction import CompactionRequest, CompactionResult

__all__ = ('NativeCompaction',)


@dataclass
class NativeCompaction(AbstractCapability[Any]):
    """Compacts the message history with the compaction API of the model that serves the request, when it has one.

    When compaction is requested, for example with
    [`RunContext.request_compaction`][pydantic_ai.tools.RunContext.request_compaction] or
    [`Agent.compact`][pydantic_ai.agent.Agent.compact], it calls
    [`compact_messages`][pydantic_ai.models.Model.compact_messages] on the
    [model the request is for][pydantic_ai.capabilities.CompactionRequest.model], if its profile has
    [`supports_compaction_on_demand`][pydantic_ai.profiles.ModelProfile.supports_compaction_on_demand]: the OpenAI
    Responses API on OpenAI and Azure OpenAI, and Anthropic's Claude models that compact on demand. Otherwise it returns
    `None`, so the next capability with a
    [`compact_history`][pydantic_ai.capabilities.AbstractCapability.compact_history] hook compacts instead.

    The compaction request is sent with the conversation's own settings, tools and instructions, so it reads the
    prompt cache the conversation's last request wrote. It summarizes everything before the turn being answered:
    the request, and the response whose tool calls it answers. The compacted history is the provider's
    [`CompactionPart`][pydantic_ai.messages.CompactionPart], in the `ModelResponse` it arrived in, followed by
    that turn.

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import NativeCompaction

    agent = Agent('anthropic:claude-sonnet-5', capabilities=[NativeCompaction()])
    ```
    """

    _: KW_ONLY

    focus_as_system_prompt: bool = False
    """Whether to compact a request with a [`focus`][pydantic_ai.capabilities.CompactionRequest.focus] on a model whose compaction takes instructions as its system prompt.

    On a model whose [`compaction_instructions`][pydantic_ai.profiles.ModelProfile.compaction_instructions] are
    `'replaces_prompt'`, like Anthropic's, the focus is sent as the compaction's instructions, which the model
    wraps in a full summarization prompt. On one whose are `'system_prompt'`, like OpenAI's, the instructions
    replace the agent's for the compaction request, so it no longer reads the prompt cache the conversation
    wrote. By default, a focused request is left to the next capability that compacts, which can keep the cache.
    With `True`, the focus is sent as the compaction request's system prompt instead.

    A model whose compaction takes no instructions always leaves a focused request to the next capability.
    """

    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        model = request.model
        try:
            profile = model.profile
        except NotImplementedError:
            # A `FallbackModel` has no profile of its own: which of its models would compact isn't known.
            return None
        if not profile.get('supports_compaction_on_demand', False):
            return None

        instructions: str | None = None
        if request.focus is not None and (focus := request.focus.strip()):
            mode = profile.get('compaction_instructions')
            if mode == 'replaces_prompt' or (mode == 'system_prompt' and self.focus_as_system_prompt):
                instructions = focus
            else:
                return None

        kept_from = _turn_start(request.messages)
        if kept_from == 0:
            # Nothing before the turn being answered, so nothing to compact.
            return None
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
