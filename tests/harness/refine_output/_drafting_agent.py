"""Models that answer again when `RefineOutput` asks, shared by the `RefineOutput` tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

ORIGIN = 'pydantic-ai-harness.refine_output.v1'

model_calls: list[None] = []
"""One entry per `drafting_model` call, so a test can tell a replayed response from a new one."""


def drafting_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answers `'draft'`, then `'revised'` when asked again."""
    model_calls.append(None)
    answered = any(isinstance(message, ModelResponse) for message in messages)
    return ModelResponse(parts=[TextPart('revised' if answered else 'draft')])


def numbered_model() -> FunctionModel:
    """A model that answers `'answer 1'`, `'answer 2'`, and so on, streamed or not."""

    def answer(messages: list[ModelMessage]) -> str:
        return f'answer {sum(isinstance(message, ModelResponse) for message in messages) + 1}'

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(answer(messages))])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield answer(messages)

    return FunctionModel(model, stream_function=stream)


def reviews(messages: Sequence[ModelMessage]) -> list[tuple[str, object]]:
    """The content and round of each review instruction in `messages`."""
    return [
        (part.content, part.metadata['round'])
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart) and part.metadata and part.metadata['origin'] == ORIGIN
    ]
