"""Persist the main conversation's observed text time without touching provider timestamps."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_pax.ui.rendering.status import TEXT_TIME, Status


@dataclass
class FooterActivity(AbstractCapability[AgentDepsT]):
    """Keep a streamed delta's time in response metadata before the conversation is saved."""

    status: Status
    run_id: Callable[[], str | None]
    _text_updates: int = field(default=0, init=False)

    async def prepare_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if ctx.run_id == self.run_id():
            self._text_updates = self.status.text_updates
        return request_context

    async def after_model_request(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if (
            'assistant_age' not in self.status.fields()
            or ctx.run_id != self.run_id()
            or self.status.last_text_at is None
            or self.status.text_updates == self._text_updates
            or not any(isinstance(part, TextPart) and part.content for part in response.parts)
        ):
            return response
        return replace(response, metadata={**(response.metadata or {}), TEXT_TIME: self.status.last_text_at})
