"""Deprecation coverage for `Agent(history_processors=...)`.

The kwarg is deprecated in favor of `Agent(capabilities=[ProcessHistory(...)])`; the
kwarg path is preserved for 1.x and remapped onto `ProcessHistory` capabilities at
construction time. Removed at the v2 cut.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime

pytestmark = [pytest.mark.anyio]


@pytest.fixture
def received_messages() -> list[ModelMessage]:
    return []


@pytest.fixture
def function_model(received_messages: list[ModelMessage]) -> FunctionModel:
    async def llm(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        received_messages.extend(messages)
        return ModelResponse(parts=[TextPart(content='Done')])

    return FunctionModel(llm)


async def test_history_processors_kwarg_warns_and_remaps(
    function_model: FunctionModel, received_messages: list[ModelMessage]
) -> None:
    """`Agent(history_processors=[fn])` emits `PydanticAIDeprecationWarning` and remaps onto `capabilities=[ProcessHistory(fn)]`."""

    def drop_first(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[1:]

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`Agent\(history_processors=\[fn, \.\.\.\]\)` is deprecated and will be removed in v2\.0\. '
        r'Replace with `Agent\(capabilities=\[ProcessHistory\(fn\), \.\.\.\]\)`\.',
    ):
        agent = Agent(function_model, history_processors=[drop_first])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='First')]),
        ModelResponse(parts=[TextPart(content='Answer')]),
    ]
    await agent.run('Second', message_history=message_history)

    user_prompts = [part for msg in received_messages for part in msg.parts if isinstance(part, UserPromptPart)]
    assert user_prompts == snapshot([UserPromptPart(content='Second', timestamp=IsDatetime())])
