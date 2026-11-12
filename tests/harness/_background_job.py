"""A model whose response the provider finishes in the background, shared by the durable engines' sleep tests."""

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


class BackgroundJobModel(FunctionModel):
    """Leaves its response to the provider, which has it ready after `checks` checks, each `delay` seconds after the last.

    Records how often it was requested live and which jobs were cancelled, so a test can tell a request replayed from a
    checkpoint from a live one, and a job left running from a cancelled one.
    """

    def __init__(self, delay: float, *, checks: int) -> None:
        super().__init__(self._respond, model_name='background')
        self.delay = delay
        self.checks = checks
        self.requests = 0
        self.cancelled: list[str | None] = []

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests += 1
        if self.requests > self.checks:
            return ModelResponse(parts=[TextPart('ready')], provider_response_id='job')
        return ModelResponse(parts=[], provider_response_id='job', state='suspended')

    def continuation_delay(self, response: ModelResponse) -> float | None:
        return self.delay

    # Only reached if a regression cancels the job, which these tests check is left running for the woken run.
    async def cancel_suspended_response(self, response: ModelResponse) -> None:  # pragma: no cover
        self.cancelled.append(response.provider_response_id)
