from dataclasses import dataclass

from ..tools import ToolDefinition
from . import AgentModel, Model


@dataclass
class FallbackModel(Model):
    """A model that falls back to subsequent models if earlier ones fail with server errors."""

    models: list[Model]

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        for model in self.models:
            try:
                return await model.agent_model(
                    function_tools=function_tools, allow_text_result=allow_text_result, result_tools=result_tools
                )
            except Exception as e:
                # Check if this is a server error (status code >= 500)
                if hasattr(e, 'status_code') and getattr(e, 'status_code') >= 500:
                    continue
                # If it's not a server error, raise immediately
                raise

        raise Exception('All models failed with server errors.')

    def name(self) -> str:
        model_names = [model.name() for model in self.models]
        return f"Fallback[{', '.join(model_names)}]"
