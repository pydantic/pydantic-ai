"""Tests for the `ComputerUse` capability and its `computer` tool."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelMessage,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserError,
)
from pydantic_ai.capabilities import AbstractCapability, ComputerUse, computer_use as computer_use_module
from pydantic_ai.capabilities.computer_use import _to_pixels  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.messages import BinaryContent, BinaryImage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.workspaces import (
    Click,
    Drag,
    MoveMouse,
    Point,
    PressKeys,
    Scroll,
    TypeText,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)

from .workspace_fakes import FAKE_SCREENSHOT, InMemoryProvider, ProviderBackend, WorkspaceCapability

pytestmark = pytest.mark.anyio

SCREENSHOT = BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png')


def _scripted(*calls: list[dict[str, Any]], profile: ModelProfile | None = None) -> FunctionModel:
    """A model that calls `computer` once per action list, then answers `done`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        made = sum(isinstance(part, ToolCallPart) for message in messages for part in message.parts)
        if made < len(calls):
            return ModelResponse(parts=[ToolCallPart('computer', {'actions': calls[made]})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond, profile=profile)


def _returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


async def test_actions_run_in_order_and_the_model_gets_a_summary_and_a_screenshot() -> None:
    provider = InMemoryProvider()
    agent = Agent(
        _scripted(
            [{'type': 'screenshot'}, {'type': 'click', 'x': 10, 'y': 20}, {'type': 'type', 'text': 'hi'}],
            [{'type': 'keypress', 'keys': ['ctrl+s']}, {'type': 'wait', 'seconds': 0.01}],
        ),
        capabilities=[provider.capability(), ComputerUse(settle_seconds=0)],
    )
    result = await agent.run('Type hi and save.')

    assert result.output == 'done'
    # `screenshot` and `wait` are the tool's own: they never reach the workspace.
    assert provider.actions == [Click(x=10, y=20), TypeText(text='hi'), PressKeys(keys=('ctrl', 's'))]
    first, second = _returns(result.all_messages())
    assert first.content == ['Ran 3 actions. Screenshot attached (1280x800).', SCREENSHOT]
    assert first.metadata == {'performed': 3, 'screenshot_size': (1280, 800)}
    assert second.content == ['Ran 2 actions. Screenshot attached (1280x800).', SCREENSHOT]
    assert first.tool_kind == 'computer'


async def test_a_failed_action_stops_the_rest_and_still_returns_a_screenshot() -> None:
    provider = InMemoryProvider()
    agent = Agent(
        _scripted(
            [
                {'type': 'click', 'x': 1, 'y': 1},
                {'type': 'hold_key', 'keys': ['shift'], 'seconds': 1},
                {'type': 'type', 'text': 'never typed'},
            ]
        ),
        capabilities=[provider.capability(), ComputerUse(settle_seconds=0)],
    )
    result = await agent.run('Go.')

    assert provider.actions == [Click(x=1, y=1)]
    (tool_return,) = _returns(result.all_messages())
    assert tool_return.content == [
        "Action 2 of 3 (hold_key) failed: the fake display cannot perform 'hold_key'. The 1 action after it did not "
        'run. Screenshot attached (1280x800).',
        SCREENSHOT,
    ]
    assert tool_return.metadata == {
        'performed': 1,
        'error': "the fake display cannot perform 'hold_key'",
        'screenshot_size': (1280, 800),
    }


@pytest.mark.parametrize(
    'actions,pauses',
    [
        pytest.param([{'type': 'click', 'x': 1, 'y': 1}], [0.5], id='settles-after-acting'),
        pytest.param(
            [{'type': 'click', 'x': 1, 'y': 1}, {'type': 'wait', 'seconds': 2}], [2], id='a-final-wait-is-enough'
        ),
        pytest.param([{'type': 'screenshot'}], [], id='looking-needs-no-settling'),
    ],
)
async def test_the_screen_settles_after_acting_before_the_screenshot(
    actions: list[dict[str, Any]], pauses: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        if seconds:  # zero is a checkpoint other code yields at, not a pause
            slept.append(seconds)

    monkeypatch.setattr(computer_use_module.anyio, 'sleep', sleep)
    agent = Agent(_scripted(actions), capabilities=[InMemoryProvider().capability(), ComputerUse()])
    await agent.run('Go.')
    assert slept == pauses


async def test_a_read_only_workspace_reports_refused_actions_to_the_model() -> None:
    provider = InMemoryProvider()
    agent = Agent(
        _scripted([{'type': 'click'}]),
        capabilities=[provider.capability(read_only=True), ComputerUse(settle_seconds=0)],
    )
    (tool_return,) = _returns((await agent.run('Go.')).all_messages())
    assert tool_return.metadata is not None
    assert tool_return.metadata['performed'] == 0
    assert 'acting on the display are disabled' in tool_return.metadata['error']
    assert provider.actions == []


async def test_a_grid_model_points_on_its_grid_and_the_workspace_gets_pixels() -> None:
    provider = InMemoryProvider()
    seen_instructions: list[str | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_instructions.append(info.instructions)
        if not any(isinstance(part, ToolCallPart) for message in messages for part in message.parts):
            actions = [
                {'type': 'click', 'x': 500, 'y': 500},
                {'type': 'drag', 'path': [{'x': 0, 'y': 0}, {'x': 999, 'y': 999}]},
                {'type': 'scroll', 'direction': 'down'},
                {'type': 'scroll', 'direction': 'up', 'x': 500, 'y': 250},
                {'type': 'move', 'x': 250, 'y': 750},
            ]
            return ModelResponse(parts=[ToolCallPart('computer', {'actions': actions})])
        return ModelResponse(parts=[TextPart('done')])

    model = FunctionModel(respond, profile=ModelProfile(computer_use_coordinate_grid=1000))
    agent = Agent(model, capabilities=[provider.capability(), ComputerUse(settle_seconds=0)])
    (tool_return,) = _returns((await agent.run('Go.')).all_messages())

    assert provider.actions == [
        Click(x=640, y=400),
        Drag(path=(Point(x=0, y=0), Point(x=1279, y=799))),
        Scroll(direction='down'),
        Scroll(direction='up', x=640, y=200),
        MoveMouse(x=320, y=600),
    ]
    assert seen_instructions[0] is not None
    assert 'on a grid from 0 to 999 over the most recent screenshot' in seen_instructions[0]
    assert tool_return.content == ['Ran 5 actions. Screenshot attached.', SCREENSHOT]


def test_grid_coordinates_scale_to_the_last_pixel_at_most() -> None:
    # On a display narrower than the grid, the grid's last point rounds past the last pixel.
    assert _to_pixels(Click(x=999, y=0), 400, 300, 1000) == Click(x=399, y=0)


async def test_an_approved_call_from_a_grid_model_still_gets_pixels() -> None:
    provider = InMemoryProvider()
    model = _scripted([{'type': 'click', 'x': 500, 'y': 500}], profile=ModelProfile(computer_use_coordinate_grid=1000))
    agent = Agent(
        model,
        capabilities=[provider.capability(), ComputerUse(require_approval=True, settle_seconds=0)],
        output_type=[str, DeferredToolRequests],
    )
    first = await agent.run('Go.')
    assert isinstance(first.output, DeferredToolRequests)
    (call,) = first.output.approvals

    await agent.run(
        message_history=first.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: True}),
    )
    assert provider.actions == [Click(x=640, y=400)]


async def test_a_pixel_model_is_told_to_use_pixels() -> None:
    seen: list[str | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions)
        return ModelResponse(parts=[TextPart('done')])

    await Agent(FunctionModel(respond), capabilities=[InMemoryProvider().capability(), ComputerUse()]).run('Hi.')
    assert seen[0] is not None and 'Coordinates are pixels in the most recent screenshot' in seen[0]


async def test_require_approval_lets_looking_through_and_defers_acting() -> None:
    provider = InMemoryProvider()
    agent = Agent(
        _scripted([{'type': 'screenshot'}], [{'type': 'click', 'x': 1, 'y': 1}]),
        capabilities=[provider.capability(), ComputerUse(require_approval=True, settle_seconds=0)],
        output_type=[str, DeferredToolRequests],
    )
    result = await agent.run('Go.')

    assert isinstance(result.output, DeferredToolRequests)
    assert [call.args_as_dict() for call in result.output.approvals] == [
        {'actions': [{'type': 'click', 'x': 1, 'y': 1}]}
    ]
    (looked,) = _returns(result.all_messages())
    assert looked.metadata == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert provider.actions == []


async def test_only_the_newest_screenshots_are_sent_and_history_keeps_them_all() -> None:
    sent_images: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = [part for message in messages for part in message.parts]
        returns = [part for part in parts if isinstance(part, ToolReturnPart)]
        sent_images.append(sum(isinstance(item, BinaryContent) for part in returns for item in part.content_items()))
        if sum(isinstance(part, ToolCallPart) for part in parts) < 4:
            return ModelResponse(parts=[ToolCallPart('computer', {'actions': [{'type': 'screenshot'}]})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(respond), capabilities=[InMemoryProvider().capability(), ComputerUse(keep_screenshots=2)]
    )
    result = await agent.run('Look four times.')
    assert sent_images == [0, 1, 2, 2, 2]

    sent_images.clear()
    every = Agent(
        FunctionModel(respond), capabilities=[InMemoryProvider().capability(), ComputerUse(keep_screenshots=None)]
    )
    await every.run('Look four times.')
    assert sent_images == [0, 1, 2, 3, 4]
    assert [part.content_items()[1] for part in _returns(result.all_messages())] == [SCREENSHOT] * 4


async def test_unknown_fields_are_ignored_and_a_batch_is_capped() -> None:
    provider = InMemoryProvider()
    agent = Agent(
        _scripted(
            [{'type': 'screenshot', 'scale': 0.5}, {'type': 'click', 'x': 1, 'y': 1, 'coordinate_space': 'pixels'}],
            [{'type': 'screenshot'}] * 21,
        ),
        capabilities=[provider.capability(), ComputerUse(settle_seconds=0)],
    )
    result = await agent.run('Go.')

    assert provider.actions == [Click(x=1, y=1)]
    capped = [part for part in _returns(result.all_messages()) if part.outcome == 'retried']
    assert len(capped) == 1
    assert 'List should have at most 20 items' in capped[0].model_response_str()


async def test_other_tools_run_unchanged_beside_the_computer() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart('add', {'a': 2, 'b': 3})])
        return ModelResponse(parts=[TextPart('done')])

    model = FunctionModel(respond, profile=ModelProfile(computer_use_coordinate_grid=1000))
    agent = Agent(
        model, capabilities=[InMemoryProvider().capability(), ComputerUse(require_approval=True, settle_seconds=0)]
    )

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    (tool_return,) = _returns((await agent.run('Add.')).all_messages())
    assert tool_return.content == 5


async def test_the_tool_is_sequential_bounded_and_marked_as_the_computer_kind() -> None:
    seen: list[Any] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(info.function_tools)
        return ModelResponse(parts=[TextPart('done')])

    await Agent(FunctionModel(respond), capabilities=[InMemoryProvider().capability(), ComputerUse()]).run('Hi.')
    (tool,) = seen
    assert (tool.name, tool.sequential, tool.tool_kind) == ('computer', True, 'computer')
    # Under durable execution, each workspace call is its own activity or task rather than the whole call one.
    assert tool.metadata == {'temporal': False, 'prefect': False}
    assert tool.parameters_json_schema['properties']['actions']['maxItems'] == 20


@pytest.mark.parametrize(
    'workspace',
    [pytest.param([], id='no-workspace'), pytest.param([WorkspaceCapability()], id='no-display')],
)
async def test_a_run_without_a_display_fails_at_its_start(workspace: list[AbstractCapability[Any]]) -> None:
    agent = Agent(_scripted([{'type': 'click', 'x': 1, 'y': 1}]), capabilities=[*workspace, ComputerUse()])
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
    'actions', [[{'type': 'click', 'x': 1, 'y': 1}], [{'type': 'screenshot'}]], ids=['act', 'look']
)
async def test_an_environment_that_is_gone_ends_the_run(actions: list[dict[str, Any]]) -> None:
    provider = InMemoryProvider()
    agent = Agent(_scripted(actions), capabilities=[provider.capability(), ComputerUse(settle_seconds=0)])
    with pytest.raises(WorkspaceUnavailableError, match="environment 'expired' does not exist"):
        await agent.run('Go.', workspace=WorkspaceRef(provider='fake', id='expired'))


@pytest.mark.parametrize(
    'screenshot, message',
    [
        pytest.param(WorkspaceError('scrot: not found'), 'Taking the screenshot failed: scrot: not found', id='fails'),
        pytest.param(b'\xff\xd8\xff not a png', 'The screenshot was not a PNG.', id='not-png'),
    ],
)
async def test_a_bad_screenshot_is_a_failed_call_that_says_what_ran(screenshot: Any, message: str) -> None:
    provider = InMemoryProvider()

    class BadScreen(ProviderBackend):
        async def screenshot(self) -> bytes:
            if isinstance(screenshot, Exception):
                raise screenshot
            return screenshot

    class Supplier(AbstractCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return BadScreen(provider, ref)

    agent = Agent(
        _scripted([{'type': 'click', 'x': 1, 'y': 1}]), capabilities=[Supplier(), ComputerUse(settle_seconds=0)]
    )
    (tool_return,) = _returns((await agent.run('Go.')).all_messages())

    assert tool_return.outcome == 'failed'
    assert tool_return.content == f'Ran 1 action. {message}'
    assert provider.actions == [Click(x=1, y=1)]


def test_settings_are_checked() -> None:
    with pytest.raises(UserError, match='`settle_seconds` must not be negative'):
        ComputerUse(settle_seconds=-1)
    with pytest.raises(UserError, match='`keep_screenshots` must be at least 1'):
        ComputerUse(keep_screenshots=0)
