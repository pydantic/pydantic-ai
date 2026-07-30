from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai._run_context import AgentDepsT

from .abstract import AbstractCapability, ModelSelector


@dataclass
class SelectModel(AbstractCapability[AgentDepsT]):
    """Select a model before each logical model request step.

    The selector receives a [`ModelSelectionContext`][pydantic_ai.models.ModelSelectionContext]
    containing the run dependencies, message history, accumulated usage, and lower-precedence
    model. It may be synchronous or asynchronous and return either a model instance or model ID.
    """

    selector: ModelSelector[AgentDepsT]

    def get_model(self) -> ModelSelector[AgentDepsT]:
        return self.selector

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
