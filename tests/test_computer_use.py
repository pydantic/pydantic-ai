"""Tests for the `ComputerUse` capability and its `computer` tool."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserError,
)
from pydantic_ai.capabilities import AbstractCapability, ComputerUse
from pydantic_ai.messages import BinaryImage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.workspaces import (
    Click,
    KeyPress,
    TypeText,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)

from .workspace_fakes import FAKE_SCREENSHOT, InMemoryProvider, ProviderBackend

pytestmark = pytest.mark.anyio


def _scripted(*actions: list[dict[str, Any]]) -> FunctionModel:
    """A model that calls `computer` once per action list, then answers with the tool returns it saw."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls = sum(isinstance(part, ToolCallPart) for message in messages for part in message.parts)
        if calls < len(actions):
            return ModelResponse(parts=[ToolCallPart('computer', {'actions': actions[calls]})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def _returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


async def test_actions_run_in_order_and_each_call_returns_a_screenshot() -> None:
    provider = InMemoryProvider()
    agent = Agent(
        _scripted(
            [{'kind': 'screenshot'}, {'kind': 'click', 'x': 10, 'y': 20}, {'kind': 'type_text', 'text': 'hi'}],
            [{'kind': 'key_press', 'keys': ['CTRL', 's']}, {'kind': 'wait', 'seconds': 0}],
        ),
        capabilities=[provider.capability(), ComputerUse()],
    )
    result = await agent.run('Type hi and save.')

    assert result.output == 'done'
    assert provider.actions == [Click(x=10, y=20), TypeText(text='hi'), KeyPress(keys=('ctrl', 's'))]
    first, second = _returns(result.all_messages())
    # `screenshot` and `wait` are the tool's own: they count as performed but never reach the workspace.
    assert first.content == [
        {'executed': 3, 'screen_size': [1280, 800]},
        BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png'),
    ]
    assert second.content == [
        {'executed': 2, 'screen_size': [1280, 800]},
        BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png'),
    ]
    assert first.tool_kind == 'computer'


async def test_a_failed_action_stops_the_rest_and_still_returns_a_screenshot() -> None:
    provider = InMemoryProvider()
    agent = Agent(
        _scripted(
            [
                {'kind': 'click', 'x': 1, 'y': 1},
                {'kind': 'hold_key', 'keys': ['shift'], 'seconds': 1},
                {'kind': 'type_text', 'text': 'never typed'},
            ]
        ),
        capabilities=[provider.capability(), ComputerUse()],
    )
    result = await agent.run('Go.')

    assert provider.actions == [Click(x=1, y=1)]
    (tool_return,) = _returns(result.all_messages())
    assert tool_return.content == [
        {
            'executed': 1,
            'error': "hold_key failed: the fake display cannot perform 'hold_key'",
            'screen_size': [1280, 800],
        },
        BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png'),
    ]


async def test_a_read_only_workspace_reports_refused_actions_to_the_model() -> None:
    provider = InMemoryProvider()
    agent = Agent(_scripted([{'kind': 'click'}]), capabilities=[provider.capability(read_only=True), ComputerUse()])
    result = await agent.run('Go.')

    (tool_return,) = _returns(result.all_messages())
    assert tool_return.content == [
        {
            'executed': 0,
            'error': 'click failed: This workspace is read-only: running commands, modifying files, and acting on the display are disabled. Reading files, listing directories, checking that paths exist, and taking screenshots are allowed.',
            'screen_size': [1280, 800],
        },
        BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png'),
    ]
    assert provider.actions == []


async def test_the_tool_is_sequential_and_marked_as_the_computer_kind() -> None:
    seen: list[Any] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(info.function_tools)
        return ModelResponse(parts=[TextPart('done')])

    await Agent(FunctionModel(respond), capabilities=[InMemoryProvider().capability(), ComputerUse()]).run('Hi.')
    (tool,) = seen
    assert (tool.name, tool.sequential, tool.tool_kind) == ('computer', True, 'computer')


async def test_a_run_without_a_workspace_fails_at_its_start() -> None:
    agent = Agent(_scripted([]), capabilities=[ComputerUse()])
    with pytest.raises(UserError, match='`ComputerUse` needs a workspace with a display'):
        await agent.run('Go.')


async def test_a_model_without_image_input_fails_at_its_start() -> None:
    model = FunctionModel(
        lambda messages, info: ModelResponse(parts=[TextPart('done')]),
        model_name='text-only',
        profile=ModelProfile(supports_image_input=False),
    )
    agent = Agent(model, capabilities=[InMemoryProvider().capability(), ComputerUse()])
    with pytest.raises(UserError, match="needs a model that accepts images, and 'text-only' does not"):
        await agent.run('Go.')


@pytest.mark.parametrize(
    'actions', [[{'kind': 'click', 'x': 1, 'y': 1}], [{'kind': 'screenshot'}]], ids=['act', 'screenshot']
)
async def test_an_environment_that_is_gone_ends_the_run(actions: list[dict[str, Any]]) -> None:
    provider = InMemoryProvider()
    agent = Agent(_scripted(actions), capabilities=[provider.capability(), ComputerUse()])
    with pytest.raises(WorkspaceUnavailableError, match="environment 'expired' does not exist"):
        await agent.run('Go.', workspace=WorkspaceRef(provider='fake', id='expired'))


async def test_a_failed_screenshot_is_reported_as_a_failed_call() -> None:
    provider = InMemoryProvider()

    class NoScreenshots(ProviderBackend):
        async def screenshot(self) -> bytes:
            raise WorkspaceError('scrot: not found')

    class Supplier(AbstractCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return NoScreenshots(provider, ref)

    agent = Agent(_scripted([{'kind': 'click', 'x': 1, 'y': 1}]), capabilities=[Supplier(), ComputerUse()])
    result = await agent.run('Go.')

    (tool_return,) = _returns(result.all_messages())
    assert tool_return.outcome == 'failed'
    assert tool_return.content == 'Taking the screenshot failed: scrot: not found'
    assert provider.actions == [Click(x=1, y=1)]
