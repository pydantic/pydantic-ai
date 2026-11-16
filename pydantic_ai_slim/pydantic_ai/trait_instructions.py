"""Instructions gated on how the model handling a request tends to behave."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import dataclasses_no_defaults_repr
from pydantic_ai.messages import InstructionPart

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.profiles import ModelTrait

__all__ = ('TraitInstruction', 'model_has_trait')


@dataclass(frozen=True, repr=False)
class TraitInstruction:
    """An instruction only the models with a [trait][pydantic_ai.profiles.ModelTraits] receive.

    Pass it wherever an instruction is accepted, such as a capability's `get_instructions`. It's resolved against the
    model handling each request, through that model's [profile][pydantic_ai.profiles.ModelProfile.traits], so after a
    switch to a model without the trait (a fallback, or a later run on another model) the text no longer applies.

    The block always appends its changes (`on_change='append'`) and is contributed from the first request, empty for a
    model without the trait, so a switch never rewrites the instructions the conversation started with: a model that
    gains the trait is sent the text as a change, and a model that loses it is sent the block's withdrawal.

    ```python
    from pydantic_ai import Agent, TraitInstruction
    from pydantic_ai.capabilities import Capability

    no_unrequested_delegation = TraitInstruction(
        'delegates_unprompted',
        "Don't spawn sub-agents unless the user asks.",
        name='delegation',
    )
    agent = Agent('openai:gpt-5.2', capabilities=[Capability(id='house_rules', instructions=no_unrequested_delegation)])
    ```

    A model whose profile can't be read, such as a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel],
    which only knows the model per request, has no traits. Use the
    [`ModelFallback`][pydantic_ai.capabilities.ModelFallback] capability to have each attempt resolved against the
    model that serves it.
    """

    trait: ModelTrait
    """The trait the receiving model's profile has to set for the text to apply."""

    text: str
    """The instruction text, sent to the models with the trait."""

    _: KW_ONLY

    name: str
    """The block's name, qualified into its id by the source contributing it; see [`InstructionPart.name`][pydantic_ai.messages.InstructionPart.name].

    Name it after what the text is about, not after the trait, so the id stays the same if the gate changes.
    """

    def __call__(self, ctx: RunContext[object]) -> InstructionPart:
        return InstructionPart(
            self.text if model_has_trait(ctx.model, self.trait) else '', name=self.name, on_change='append'
        )

    __repr__ = dataclasses_no_defaults_repr


def model_has_trait(model: Model, trait: ModelTrait) -> bool:
    """Whether `model`'s profile sets `trait`, which a model without a profile of its own doesn't."""
    try:
        profile = model.profile
    except NotImplementedError:
        # `FallbackModel` picks its model per request, so no profile speaks for it here.
        return False
    return profile.get('traits', {}).get(trait, False)
