"""Deprecation coverage for `Agent(history_processors=...)`.

The kwarg is removed from the `Agent.__init__`/`from_spec`/`from_file` signatures and
caught via `**_deprecated_kwargs`, then remapped onto
`capabilities=[ProcessHistory(...)]`. Removed at the v2 cut.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import ProcessHistory
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
        r'Replace with `Agent\(capabilities=\[ProcessHistory\(fn\), \.\.\.\]\)`, or hook the '
        r'`before_model_request` lifecycle event directly via `Hooks\(before_model_request=fn\)`\.',
    ):
        agent = Agent(function_model, history_processors=[drop_first])  # pyright: ignore[reportCallIssue]

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='First')]),
        ModelResponse(parts=[TextPart(content='Answer')]),
    ]
    await agent.run('Second', message_history=message_history)

    user_prompts = [part for msg in received_messages for part in msg.parts if isinstance(part, UserPromptPart)]
    assert user_prompts == snapshot([UserPromptPart(content='Second', timestamp=IsDatetime())])


async def test_history_processors_kwarg_equivalent_to_capabilities_path(function_model: FunctionModel) -> None:
    """Legacy kwarg and `capabilities=[ProcessHistory(...)]` produce equivalent message history."""

    def drop_first(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[1:]

    legacy_received: list[ModelMessage] = []
    new_received: list[ModelMessage] = []

    async def legacy_llm(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        legacy_received.extend(messages)
        return ModelResponse(parts=[TextPart(content='Done')])

    async def new_llm(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        new_received.extend(messages)
        return ModelResponse(parts=[TextPart(content='Done')])

    with pytest.warns(PydanticAIDeprecationWarning):
        legacy_agent = Agent(FunctionModel(legacy_llm), history_processors=[drop_first])  # pyright: ignore[reportCallIssue]
    new_agent = Agent(FunctionModel(new_llm), capabilities=[ProcessHistory(drop_first)])

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='First')]),
        ModelResponse(parts=[TextPart(content='Answer')]),
    ]

    await legacy_agent.run('Second', message_history=list(message_history))
    await new_agent.run('Second', message_history=list(message_history))

    legacy_prompts = [part.content for msg in legacy_received for part in msg.parts if isinstance(part, UserPromptPart)]
    new_prompts = [part.content for msg in new_received for part in msg.parts if isinstance(part, UserPromptPart)]
    assert legacy_prompts == new_prompts == ['Second']


async def test_history_processors_kwarg_exposed_on_agent_attribute(function_model: FunctionModel) -> None:
    """`Agent.history_processors` attribute still reflects the legacy kwarg value for 1.x readers."""

    def drop_first(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[1:]  # pragma: no cover

    with pytest.warns(PydanticAIDeprecationWarning):
        agent = Agent(function_model, history_processors=[drop_first])  # pyright: ignore[reportCallIssue]

    assert agent.history_processors == [drop_first]
