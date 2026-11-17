"""A model whose first request is rate-limited with a `retry_after`, shared by the durable engine suites."""

from __future__ import annotations

from pydantic_ai.exceptions import ModelHTTPError, ModelRateLimitError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def rate_limited_once(counter: dict[str, int], retry_after: float) -> FunctionModel:
    """A model that fails with a rate limit asking to wait `retry_after` seconds, then answers `'ok'`.

    `counter['calls']` counts the requests it receives.
    """

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        counter['calls'] += 1
        if counter['calls'] == 1:
            raise ModelHTTPError.for_category(
                ModelRateLimitError, status_code=429, model_name='fn', retry_after=retry_after
            )
        return ModelResponse(parts=[TextPart(content='ok')])

    return FunctionModel(fn, model_name='fn')
