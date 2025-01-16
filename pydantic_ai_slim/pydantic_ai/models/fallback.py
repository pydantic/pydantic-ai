from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..tools import ToolDefinition
from . import AgentModel, KnownModelName, Model, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage, ModelResponse
    from ..result import Usage
    from ..settings import ModelSettings
    from ..tools import ToolDefinition


@dataclass(init=False)
class FallbackModel(Model):
    """A model that uses one or more fallback models upon failure.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    models: list[Model]

    def __init__(
        self,
        default_model: Model | KnownModelName,
        *fallback_models: Model | KnownModelName,
        # TODO: do we need any of the other classic init args (base url, etc?)
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
        """
        # TODO: should we do this lazily?
        default_model_ = default_model if isinstance(default_model, Model) else infer_model(default_model)
        fallback_models_ = [model if isinstance(model, Model) else infer_model(model) for model in fallback_models]

        self.models = [default_model_, *fallback_models_]

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        return FallbackAgentModel(
            models=self.models,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )

    def name(self) -> str:
        return f'FallbackModel([{", ".join(m.name() for m in self.models)}])'


@dataclass
class FallbackAgentModel(AgentModel):
    """Implementation of `AgentModel` for [FallbackModel][pydantic_ai.models.fallback.FallbackModel]."""

    models: list[Model]
    function_tools: list[ToolDefinition]
    allow_text_result: bool
    result_tools: list[ToolDefinition]

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Try each model in sequence until one succeeds.

        TODO: improve exception handling here, the current logic is oversimplified.
        TODO: should we add logfire logging at this level (around the for loop?)
        """
        errors: list[Exception] = []

        for model in self.models:
            agent_model = await model.agent_model(
                function_tools=self.function_tools,
                allow_text_result=self.allow_text_result,
                result_tools=self.result_tools,
            )

            try:
                return await agent_model.request(messages, model_settings)
            except Exception as exc_info:
                errors.append(exc_info)
                continue

        raise RuntimeError(f'All fallback models failed: {errors}')
