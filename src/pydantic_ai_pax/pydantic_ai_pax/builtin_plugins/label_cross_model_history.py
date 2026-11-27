"""Tell the model when earlier assistant turns came from another model, as after `/model` or an import.

The built-in `label_cross_model_history` plugin: harness `LabelCrossModelHistory`, whose notice is appended
to the history where it starts to apply, so a conversation on one model sends the same requests as without it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import InstructionPart, ModelMessage, ModelResponse
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.label_cross_model_history import LabelCrossModelHistory
from pydantic_ai_pax.models.profiles import base_model
from pydantic_ai_pax.plugins import DepsT, NoSettings, Plugin


@dataclass
class _AnyAccountLabels(LabelCrossModelHistory[DepsT]):
    """`LabelCrossModelHistory` that counts the same model on another account, `PROVIDER@PROFILE:NAME`, as its own.

    With `accounts.pool` on, a request falls back to another signed-in account of the same model, and its
    response records that account's model ID; the next request, back on the first account, is still the
    same model answering.
    """

    def get_instructions(self) -> Callable[[RunContext[DepsT]], InstructionPart]:
        label = super().get_instructions()

        def _instructions(ctx: RunContext[DepsT]) -> InstructionPart:
            current = ctx.model_id
            if current is None:
                # A model passed as an instance, as a supplied agent's own, names no account.
                return label(ctx)
            model = base_model(current)

            def counted(message: ModelMessage) -> ModelMessage:
                if isinstance(message, ModelResponse) and message.model_id and base_model(message.model_id) == model:
                    return replace(message, model_id=current)
                return message

            return label(replace(ctx, messages=[counted(message) for message in ctx.messages]))

        return _instructions


class LabelCrossModelHistoryPlugin(Plugin[NoSettings, DepsT]):
    """Harness `LabelCrossModelHistory` with its default notice, blind to which account served a turn."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (_AnyAccountLabels[DepsT](),)
