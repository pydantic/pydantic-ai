"""`ComputerUse` driven through `Agent` with a scripted model and a recording computer."""

from __future__ import annotations

import json
import struct
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeGuard

import pytest

from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, on_event
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai_harness import ComputerUse, computer_use
from pydantic_ai_harness.computer_use import ComputerActionsEvent, ComputerError, MouseButton, ScrollDirection

pytestmark = pytest.mark.anyio


def png(width: int, height: int) -> bytes:
    """A PNG signature and IHDR chunk: enough for the size the tool reports."""
    return b'\x89PNG\r\n\x1a\n' + struct.pack('>I4sII', 13, b'IHDR', width, height) + b'\x08\x02\x00\x00\x00'


@dataclass
class RecordingComputer:
    """Records every call; `fail_on` makes that method raise `ComputerError`."""

    calls: list[tuple[Any, ...]] = field(default_factory=list[tuple[Any, ...]])
    image: bytes = field(default_factory=lambda: png(1280, 800))
    fail_on: str | None = None

    def _record(self, *call: Any) -> None:
        if call[0] == self.fail_on:
            raise ComputerError(f'{call[0]} broke')
        self.calls.append(call)

    async def screenshot(self) -> bytes:
        self._record('screenshot')
        return self.image

    async def click(
        self, x: int, y: int, *, button: MouseButton = 'left', count: int = 1, modifiers: Sequence[str] = ()
    ) -> None:
        self._record('click', x, y, button, count, tuple(modifiers))

    async def move(self, x: int, y: int) -> None:
        self._record('move', x, y)

    async def drag(self, path: Sequence[tuple[int, int]]) -> None:
        self._record('drag', tuple(path))

    async def scroll(self, x: int, y: int, *, direction: ScrollDirection, amount: int) -> None:
        self._record('scroll', x, y, direction, amount)

    async def type_text(self, text: str) -> None:
        self._record('type', text)

    async def press_keys(self, keys: Sequence[str]) -> None:
        self._record('keypress', tuple(keys))


def scripted(*batches: list[dict[str, Any]]) -> tuple[FunctionModel, list[list[ModelMessage]]]:
    """A model that sends each batch as one `computer` call, then answers; it records what it was sent.

    It streams too, because an `on_event` listener makes the run stream its model responses.
    """
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        step = len(seen) - 1
        if step < len(batches):
            return ModelResponse(parts=[ToolCallPart('computer', {'actions': batches[step]}, tool_call_id=f'c{step}')])
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        [part] = respond(messages, info).parts
        if isinstance(part, ToolCallPart):
            yield {
                0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id)
            }
        else:
            assert isinstance(part, TextPart)
            yield part.content

    return FunctionModel(respond, stream_function=stream), seen


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def items(part: ToolReturnPart) -> list[object]:
    """The items of a `computer` result, which is always a list."""
    content = part.content
    assert _is_list(content)
    return content


def tool_returns(messages: Sequence[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


EVERY_ACTION: list[dict[str, Any]] = [
    {'type': 'screenshot'},
    {'type': 'click', 'x': 10, 'y': 20},
    {'type': 'click', 'x': 1, 'y': 2, 'button': 'right', 'count': 2, 'modifiers': ['shift']},
    {'type': 'move', 'x': 3, 'y': 4},
    {'type': 'drag', 'path': [{'x': 0, 'y': 0}, {'x': 5, 'y': 6}]},
    {'type': 'scroll', 'x': 7, 'y': 8, 'direction': 'down'},
    {'type': 'type', 'text': 'hello'},
    {'type': 'keypress', 'keys': ['ctrl', 'a']},
    {'type': 'wait', 'seconds': 0.01},
]


class TestComputerUse:
    async def test_runs_every_action_in_order_and_returns_the_screenshot(self) -> None:
        computer = RecordingComputer()
        model, _ = scripted(EVERY_ACTION)
        agent = Agent(model, capabilities=[ComputerUse(computer=computer, settle_seconds=0)])

        result = await agent.run('use the computer')

        assert computer.calls == [
            ('click', 10, 20, 'left', 1, ()),
            ('click', 1, 2, 'right', 2, ('shift',)),
            ('move', 3, 4),
            ('drag', ((0, 0), (5, 6))),
            ('scroll', 7, 8, 'down', 3),
            ('type', 'hello'),
            ('keypress', ('ctrl', 'a')),
            ('screenshot',),
        ]
        [returned] = tool_returns(result.all_messages())
        assert returned.content == [
            'Ran 9 actions. Screenshot attached (1280x800); use its pixel coordinates.',
            BinaryContent(data=computer.image, media_type='image/png'),
        ]

    async def test_a_failed_action_skips_the_rest_and_still_shows_the_screen(self) -> None:
        computer = RecordingComputer(fail_on='keypress')
        events: list[ComputerActionsEvent] = []

        class Recorder(AbstractCapability[None]):
            @on_event(ComputerActionsEvent)
            async def record(self, ctx: RunContext[None], event: ComputerActionsEvent) -> None:
                events.append(event)

        model, _ = scripted(
            [
                {'type': 'click', 'x': 1, 'y': 1},
                {'type': 'keypress', 'keys': ['hyper']},
                {'type': 'type', 'text': 'never'},
                {'type': 'type', 'text': 'typed'},
            ]
        )
        agent = Agent(
            model,
            deps_type=type(None),
            capabilities=[ComputerUse[None](computer=computer, settle_seconds=0.001), Recorder()],
        )

        result = await agent.run('go')

        assert computer.calls == [('click', 1, 1, 'left', 1, ()), ('screenshot',)]
        [returned] = tool_returns(result.all_messages())
        assert items(returned)[0] == (
            'Action 2 of 4 (keypress) failed: keypress broke. The 2 actions after it did not run. '
            'Screenshot attached (1280x800); use its pixel coordinates.'
        )
        [event] = events
        assert (event.performed, event.error, event.screenshot_size) == (1, 'keypress broke', (1280, 800))
        assert event.tool_call_id == 'c0'
        assert [action.type for action in event.actions] == ['click', 'keypress', 'type', 'type']

    async def test_a_failure_on_the_last_action_reports_no_skipped_actions(self) -> None:
        computer = RecordingComputer(fail_on='move')
        model, _ = scripted([{'type': 'move', 'x': 1, 'y': 1}])
        agent = Agent(model, capabilities=[ComputerUse(computer=computer)])

        result = await agent.run('go')

        [returned] = tool_returns(result.all_messages())
        assert str(items(returned)[0]).startswith('Action 1 of 1 (move) failed: move broke. Screenshot attached')

    async def test_a_failed_screenshot_returns_text_alone(self) -> None:
        computer = RecordingComputer(fail_on='screenshot')
        model, _ = scripted([{'type': 'wait', 'seconds': 0.01}])
        agent = Agent(model, capabilities=[ComputerUse(computer=computer)])

        result = await agent.run('go')

        [returned] = tool_returns(result.all_messages())
        assert returned.content == ['Ran 1 action. The screenshot failed: screenshot broke']

    async def test_an_image_without_a_png_header_has_no_reported_size(self) -> None:
        computer = RecordingComputer(image=b'not a png')
        model, _ = scripted([{'type': 'screenshot'}])
        agent = Agent(model, capabilities=[ComputerUse(computer=computer)])

        result = await agent.run('go')

        [returned] = tool_returns(result.all_messages())
        assert items(returned)[0] == 'Ran 1 action. Screenshot attached.'

    async def test_an_invalid_action_is_sent_back_for_a_retry(self) -> None:
        computer = RecordingComputer()
        model, seen = scripted([{'type': 'teleport'}], [{'type': 'screenshot'}])
        agent = Agent(model, capabilities=[ComputerUse(computer=computer)])

        await agent.run('go')

        assert computer.calls == [('screenshot',)]
        assert len(seen) == 3

    async def test_instructions_describe_the_computer(self) -> None:
        model, seen = scripted()
        agent = Agent(model, capabilities=[ComputerUse(computer=RecordingComputer(), environment='an Ubuntu VM')])

        await agent.run('go')

        [request] = seen[0]
        assert isinstance(request, ModelRequest)
        assert request.instructions is not None
        assert 'You can see and control an Ubuntu VM with the `computer` tool.' in request.instructions

    async def test_instructions_default_to_a_generic_computer(self) -> None:
        model, seen = scripted()
        agent = Agent(model, capabilities=[ComputerUse(computer=RecordingComputer())])

        await agent.run('go')

        [request] = seen[0]
        assert isinstance(request, ModelRequest)
        assert request.instructions is not None
        assert 'control a computer with' in request.instructions

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'settle_seconds': -1}, 'settle_seconds'),
            ({'keep_screenshots': 0}, 'keep_screenshots'),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(UserError, match=message):
            ComputerUse(computer=RecordingComputer(), **kwargs)


class TestComputerUseApproval:
    async def test_acting_waits_for_approval(self) -> None:
        computer = RecordingComputer()
        requested: list[DeferredToolRequests] = []

        async def approve(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            requested.append(requests)
            assert computer.calls == []
            return requests.build_results(approve_all=True)

        model, _ = scripted([{'type': 'type', 'text': 'hi'}])
        agent = Agent(
            model,
            deps_type=type(None),
            capabilities=[
                ComputerUse[None](computer=computer, require_approval=True, settle_seconds=0),
                HandleDeferredToolCalls(handler=approve),
            ],
        )

        await agent.run('go')

        assert [call.tool_name for request in requested for call in request.approvals] == ['computer']
        assert computer.calls == [('type', 'hi'), ('screenshot',)]

    async def test_a_denied_call_does_not_touch_the_computer(self) -> None:
        computer = RecordingComputer()

        async def deny(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            return requests.build_results(approvals={'c0': ToolDenied('not now')})

        model, _ = scripted([{'type': 'click', 'x': 1, 'y': 1}])
        agent = Agent(
            model,
            deps_type=type(None),
            capabilities=[
                ComputerUse[None](computer=computer, require_approval=True),
                HandleDeferredToolCalls(handler=deny),
            ],
        )

        result = await agent.run('go')

        assert computer.calls == []
        [returned] = tool_returns(result.all_messages())
        assert returned.content == 'not now'

    async def test_looking_and_waiting_need_no_approval(self) -> None:
        computer = RecordingComputer()

        async def never(ctx: RunContext[None], requests: DeferredToolRequests) -> None:
            raise AssertionError('approval was requested')  # pragma: no cover

        model, _ = scripted([{'type': 'screenshot'}, {'type': 'wait', 'seconds': 0.01}])
        agent = Agent(
            model,
            deps_type=type(None),
            capabilities=[
                ComputerUse[None](computer=computer, require_approval=True),
                HandleDeferredToolCalls(handler=never),
            ],
        )

        await agent.run('go')

        assert computer.calls == [('screenshot',)]


class TestComputerUseScreenshotHistory:
    async def test_only_the_latest_screenshots_reach_the_model(self) -> None:
        batches = [[{'type': 'screenshot'}] for _ in range(3)]
        model, seen = scripted(*batches)
        agent = Agent(model, capabilities=[ComputerUse(computer=RecordingComputer(), keep_screenshots=2)])

        await agent.run('go')

        images = [any(isinstance(item, BinaryContent) for item in items(part)) for part in tool_returns(seen[-1])]
        assert images == [False, True, True]
        [first, *_] = tool_returns(seen[-1])
        assert str(items(first)[1]).startswith('[Older screenshot removed')

    async def test_keep_screenshots_none_keeps_every_screenshot(self) -> None:
        batches = [[{'type': 'screenshot'}] for _ in range(3)]
        model, seen = scripted(*batches)
        agent = Agent(model, capabilities=[ComputerUse(computer=RecordingComputer(), keep_screenshots=None)])

        await agent.run('go')

        assert all(isinstance(items(part)[1], BinaryContent) for part in tool_returns(seen[-1]))

    async def test_other_tools_and_text_only_results_are_left_alone(self) -> None:
        computer = RecordingComputer()
        seen: list[list[ModelMessage]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            calls = [
                [ToolCallPart('photo', {}, tool_call_id='p0')],
                [ToolCallPart('computer', {'actions': [{'type': 'screenshot'}]}, tool_call_id='c0')],
                [ToolCallPart('computer', {'actions': [{'type': 'screenshot'}]}, tool_call_id='c1')],
            ]
            if len(seen) <= len(calls):
                if len(seen) == 2:
                    computer.fail_on = 'screenshot'
                elif len(seen) == 3:
                    computer.fail_on = None
                return ModelResponse(parts=calls[len(seen) - 1])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(respond), capabilities=[ComputerUse(computer=computer, keep_screenshots=1)])

        @agent.tool_plain
        def photo() -> list[str | BinaryContent]:
            return ['a photo', BinaryContent(data=png(1, 1), media_type='image/png')]

        await agent.run('go')

        photo_return, failed, latest = tool_returns(seen[-1])
        assert isinstance(items(photo_return)[1], BinaryContent)
        assert failed.content == ['Ran 1 action. The screenshot failed: screenshot broke']
        assert isinstance(items(latest)[1], BinaryContent)


def test_unknown_attributes_are_not_invented() -> None:
    with pytest.raises(AttributeError, match='no attribute'):
        getattr(computer_use, 'RemoteComputer')


def has_screenshot(part: ToolReturnPart) -> bool:
    return any(isinstance(item, BinaryContent) for item in items(part))


class TestComputerUseParallelCallsAndHistory:
    async def test_screenshots_the_model_has_not_seen_are_never_removed(self) -> None:
        computer = RecordingComputer()
        seen: list[list[ModelMessage]] = []
        shot = {'actions': [{'type': 'screenshot'}]}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            if len(seen) == 1:
                return ModelResponse(parts=[ToolCallPart('computer', shot, tool_call_id=f'p{i}') for i in range(3)])
            if len(seen) == 2:
                return ModelResponse(parts=[ToolCallPart('computer', shot, tool_call_id='last')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(respond), capabilities=[ComputerUse(computer=computer, keep_screenshots=2)])

        result = await agent.run('go')

        assert computer.calls == [('screenshot',)] * 4
        assert [has_screenshot(part) for part in tool_returns(seen[1])] == [True, True, True]
        assert [(part.tool_call_id, has_screenshot(part)) for part in tool_returns(seen[2])] == [
            ('p0', False),
            ('p1', False),
            ('p2', True),
            ('last', True),
        ]
        assert [has_screenshot(part) for part in tool_returns(result.all_messages())] == [True, True, True, True]

    async def test_a_continued_conversation_still_sends_only_the_latest_screenshots(self) -> None:
        model, _ = scripted(*[[{'type': 'screenshot'}] for _ in range(2)])
        agent = Agent(model, capabilities=[ComputerUse(computer=RecordingComputer(), keep_screenshots=1)])
        first = await agent.run('go')

        model_again, seen = scripted([{'type': 'screenshot'}])
        second = await agent.run('again', model=model_again, message_history=first.all_messages())

        assert [has_screenshot(part) for part in tool_returns(seen[-1])] == [False, False, True]
        assert str(items(tool_returns(seen[-1])[0])[1]).startswith('[Older screenshot removed')
        assert [has_screenshot(part) for part in tool_returns(second.all_messages())] == [True, True, True]


def test_the_spec_schema_lists_only_serializable_options() -> None:
    schema = json.dumps(AgentSpec.model_json_schema_with_capabilities([ComputerUse]))

    assert 'ComputerUse' in schema
    assert 'keep_screenshots' in schema
    assert '"computer"' not in schema
