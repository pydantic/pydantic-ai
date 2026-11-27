from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, replace
from fnmatch import fnmatchcase
from typing import TypeAlias

from pydantic_ai._instructions import AgentInstructions, SourcedInstruction
from pydantic_ai._system_prompt import SystemPromptRunner
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import AbstractModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

__all__ = ('ModelInstructions', 'ModelInstructionsFunc')

ModelInstructionsFunc: TypeAlias = (
    Callable[[RunContext[AgentDepsT]], str | None] | Callable[[RunContext[AgentDepsT]], Awaitable[str | None]]
)
"""A sync or async function returning the instructions for the model handling a request, or `None` for none."""

_BLOCK_NAME = 'text'
"""The name of the block, so its id is `capability:<id>:text`."""

_FALLBACK_MODEL_WARNING = (
    "`ModelInstructions` can't tell which model a `FallbackModel` will use, so no model-specific instructions are "
    "sent; use the `ModelFallback` capability so each attempt gets its model's instructions."
)


@dataclass
class ModelInstructions(AbstractCapability[AgentDepsT]):
    """Add instructions that depend on the model handling each request.

    Pass a mapping from glob patterns to text, or a function that returns the text for a request. The text is sent
    in one block of its own, `capability:model_instructions:text` by default, after the instructions that come
    before this capability, and is resolved again for each model
    [`ModelFallback`][pydantic_ai.capabilities.ModelFallback] tries.

    In a mapping, the first pattern that matches wins, in the mapping's order, and a model that no pattern matches
    gets no text. A pattern without a `:` is matched against the model name, and, for a name like OpenRouter's
    `openai/gpt-6-luna`, also against the part after the last `/`, so `'gpt-6*'` matches a GPT-6 model on every
    route. A pattern with a `:` is matched against the model ID, `<provider>:<model name>`, like
    `'azure:gpt-6*'`. `'*'` matches every model.

    A function receives the [`RunContext`][pydantic_ai.tools.RunContext], whose `model` is the model handling the
    request, and may be sync or async. Its block is dynamic, as it can read anything in the context; a mapping's
    depends only on the model, so it's cached with the literal instructions.

    When a conversation moves to another model, a change to the block is appended
    ([`on_change='append'`][pydantic_ai.messages.InstructionPart.on_change]), so the history records that the
    instructions changed and when. If the new model gets no text, the block is rewritten away instead, so its
    instructions are what a fresh run on it would send.

    On a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel], which picks its model inside the model call,
    no text is sent and a warning is emitted.
    """

    instructions: Mapping[str, str] | ModelInstructionsFunc[AgentDepsT]
    """Text per model pattern, or a function returning the text for a request."""

    _: KW_ONLY

    id: str | None = 'model_instructions'
    """The id the block is addressed by, as `capability:<id>:text`.

    Give each instance its own id when an agent has several: two that share one and differ raise a `UserError`.
    """

    def get_instructions(self) -> AgentInstructions[AgentDepsT]:
        return self._instruction

    def _collect_own_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        sourced = super()._collect_own_instructions()
        if callable(self.instructions):
            return sourced
        # A mapping's text depends only on the model, so the block sits with the literal instructions in the
        # cached prefix: it changes only when the conversation moves to another model, which is cache-cold anyway.
        return [replace(instruction, dynamic=False) for instruction in sourced]

    async def _instruction(self, ctx: RunContext[AgentDepsT]) -> InstructionPart:
        text = await self._text(ctx)
        return InstructionPart(
            text or '',
            name=_BLOCK_NAME,
            # Rewriting a changed block left models unsure whether anything had changed, and readier to follow a
            # forged `<system>` block, so a change is appended. A model with no text gets what a fresh run on it
            # would: no block, rather than a withdrawal it may repeat as a rule. Without an id the block isn't
            # addressable, so its changes can't be appended.
            on_change='append' if text and self.id is not None else 'rewrite',
        )

    async def _text(self, ctx: RunContext[AgentDepsT]) -> str | None:
        if _is_fallback_model(ctx.model):
            # Python's default warning filter shows this once per process: it's raised from one line.
            warnings.warn(_FALLBACK_MODEL_WARNING, UserWarning, stacklevel=1)
            return None
        if callable(self.instructions):
            return await SystemPromptRunner[AgentDepsT, str](self.instructions).run(ctx)
        return _match(self.instructions, ctx.model)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one block stated twice; two with different instructions raise rather than merge.

        Merging two mappings would make one's patterns shadow the other's, depending on their order. Agreeing ones
        merge as any capability's fields do, which keeps the lowest authority ceiling either was given.
        """
        first = capabilities[0]
        assert isinstance(first, ModelInstructions)
        for other in capabilities[1:]:
            assert isinstance(other, ModelInstructions)
            if other.instructions != first.instructions:
                raise UserError(
                    f'Capability id {first.id!r} is used by multiple `ModelInstructions` capabilities that disagree '
                    "on 'instructions'. Give each its own `id`, or make them agree."
                )
        return super().combine(capabilities)


def _match(instructions: Mapping[str, str], model: AbstractModel) -> str | None:
    name = model.model_name
    names = {name, name.rpartition('/')[2]}
    for pattern, text in instructions.items():
        if ':' in pattern:
            if fnmatchcase(model.model_id, pattern):
                return text
        elif any(fnmatchcase(candidate, pattern) for candidate in names):
            return text
    return None


def _is_fallback_model(model: AbstractModel) -> bool:
    while isinstance(model, WrapperModel):
        model = model.wrapped
    return isinstance(model, FallbackModel)
