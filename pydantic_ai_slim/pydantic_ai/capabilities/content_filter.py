from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class RaiseContentFilterError(AbstractCapability[AgentDepsT]):
    """Raises `ContentFilterError` when a model response has `finish_reason='content_filter'`.

    Add this capability to opt into treating content-filtered responses as run-ending errors,
    even when the provider returns partial text or refusal text. The full
    [`ModelResponse`][pydantic_ai.messages.ModelResponse] is serialized into
    [`ContentFilterError.body`][pydantic_ai.exceptions.UnexpectedModelBehavior.body] so callers
    can inspect any partial content.
    """

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if response.finish_reason == 'content_filter':
            details = response.provider_details or {}
            body = ModelMessagesTypeAdapter.dump_json([response]).decode()

            if reason := details.get('finish_reason'):
                message = f"Content filter triggered. Finish reason: '{reason}'"
            elif reason := details.get('block_reason'):
                message = f"Content filter triggered. Block reason: '{reason}'"
            elif refusal := details.get('refusal'):
                message = f'Content filter triggered. Refusal: {refusal!r}'
            else:  # pragma: no cover
                message = 'Content filter triggered.'

            raise ContentFilterError(message, body=body)

        return response
