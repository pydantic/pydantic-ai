"""A model whose response the provider finishes in the background, shared by the durable engines' sleep tests."""

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


class BackgroundJobModel(FunctionModel):
    """Leaves its first response to the provider, which has it ready when it's next checked on, `delay` seconds later.

    Records how often it was requested and which jobs were cancelled, so a test can tell a request replayed from a
    checkpoint from a live one, and a job left running from a cancelled one.
    """

    def __init__(self, delay: float) -> None:
        super().__init__(self._respond, model_name='background')
        self.delay = delay
        self.requests = 0
        self.cancelled: list[str | None] = []

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests += 1
        if isinstance(messages[-1], ModelResponse):
            return ModelResponse(parts=[TextPart('ready')], provider_response_id='job')
        return ModelResponse(parts=[], provider_response_id='job', state='suspended')

    def continuation_delay(self, response: ModelResponse) -> float | None:
        return self.delay

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        self.cancelled.append(response.provider_response_id)
