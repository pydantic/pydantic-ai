"""`LabelCrossModelHistory`: tell the model when earlier assistant turns came from another model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import AgentDepsT, RunContext

DEFAULT_CROSS_MODEL_NOTICE = (
    'Some earlier assistant turns in this conversation were written by a different model, not by you. '
    'Treat them as inherited context: check their claims and commitments instead of assuming you made them.'
)
"""The instruction `LabelCrossModelHistory` adds by default."""


@dataclass
class LabelCrossModelHistory(AbstractCapability[AgentDepsT]):
    """Tell the model when earlier assistant turns in its history came from another model.

    Before each model request it compares the `model_id` of every `ModelResponse` in the history with the
    model the request is about to go to, and adds `notice` to that request's instructions while any of
    them came from another model. The notice is a dynamic instruction that each request computes for itself,
    not a message part, so the message history is left unchanged.

    A response without a `model_id`, such as one from history saved before the field existed, is not
    counted. Responses from any candidate of a `FallbackModel` count as the current model's own, since
    which candidate will serve a request isn't known before it is sent. Model identifiers from the history
    are never put into the notice.
    """

    notice: str = DEFAULT_CROSS_MODEL_NOTICE
    """The instruction added while the history has a response from another model."""

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """Add `notice` while the history has a response from another model."""

        def _instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            own = _own_model_ids(ctx)
            foreign = any(
                isinstance(message, ModelResponse) and message.model_id is not None and message.model_id not in own
                for message in ctx.messages
            )
            return self.notice if foreign else None

        return _instructions


def _own_model_ids(ctx: RunContext[AgentDepsT]) -> set[str | None]:
    """The `model_id`s a response from the model the request is about to go to can carry."""
    ids: set[str | None] = {ctx.model_id, ctx.model.model_id}
    model = ctx.model
    while isinstance(model, WrapperModel):
        model = model.wrapped
    if isinstance(model, FallbackModel):
        ids.update(candidate.model_id for candidate in model.models)
    return ids
