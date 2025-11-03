import logging
import time

try:
    from fastapi import HTTPException
    from openai.types import ErrorObject
    from openai.types.model import Model
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to enable the fastapi openai compatible endpoint, '
        'you can use the `openai` and `fastapi` optional group — `pip install "pydantic-ai-slim[openai,fastapi]"`'
    ) from _import_error

from pydantic_ai.fastapi.data_models import (
    ErrorResponse,
    ModelsResponse,
)
from pydantic_ai.fastapi.registry import AgentRegistry

logger = logging.getLogger(__name__)


class AgentModelsAPI:
    """Models API for pydantic-ai agents."""

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    async def list_models(self) -> ModelsResponse:
        """List available models (OpenAI-compatible endpoint)."""
        agents = self.registry.all_agents

        models = [
            Model(
                id=name,
                object='model',
                created=int(time.time()),
                owned_by='model_owner',
            )
            for name in agents
        ]
        return ModelsResponse(data=models)

    async def get_model(self, name: str) -> Model:
        """Get information about a specific model (OpenAI-compatible endpoint)."""
        if name in self.registry.all_agents:
            return Model(id=name, object='model', created=int(time.time()), owned_by='NDIA')
        else:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error=ErrorObject(
                        type='not_found_error',
                        message=f"Model '{name}' not found",
                    ),
                ).model_dump(),
            )
