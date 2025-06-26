from temporalio import activity

from . import models
from .messages import ModelMessage, ModelResponse


class TemporalActivities:
    """Temporal activities for the agent."""

    model: models.Model

    def __init__(self, model: models.Model):
        self.model = model

    @activity.defn
    async def make_temporal_request(
        self,
        messages: list[ModelMessage],
        model_settings: models.ModelSettings | None,
        model_request_parameters: models.ModelRequestParameters,
    ) -> ModelResponse:
        return await self.model.request(messages, model_settings, model_request_parameters)
