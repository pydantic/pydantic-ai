"""`ComputerUse` against real provider APIs, sent as each provider's native computer tool.

The model adapter tests in `tests/models/` use a minimal `computer` tool, so the core test suite doesn't need the
Harness. These recordings run the real capability end to end, with its instructions, screenshot scaling and results.
"""

from __future__ import annotations

import pytest
from dirty_equals import IsInt, IsPartialDict
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.computer_use import Click, ComputerCallPart, PressKeys, TypeText
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.computer_use import ComputerUse, DisplayAction
from tests.models.conftest import json_objects

from ...conftest import RequestCapture, try_import
from .display_fakes import FAKE_DISPLAY_SIZE, ScreenBackend, ScreenProvider, png
from .shared_live import vcr_config as vcr_config

with try_import() as google_installed:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

needs_google = pytest.mark.skipif(not google_installed(), reason='google-genai not installed')


RED_SQUARE = (974, 176, 1022, 224)
"""Left, top, right and bottom of the red square on `_RedSquareScreen`, in pixels."""


class _RedSquareScreen(ScreenProvider):
    def backend(self, ref: WorkspaceRef | None) -> ScreenBackend:
        return _RedSquareBackend(self, ref)


class _RedSquareBackend(ScreenBackend):
    async def screenshot(self) -> bytes:
        return png(*FAKE_DISPLAY_SIZE, background=b'\xff\xff\xff', red_box=RED_SQUARE)


def _gemini(name: str, gemini_api_key: str, capture: RequestCapture) -> GoogleModel:
    return GoogleModel(
        name, provider=GoogleProvider(api_key=gemini_api_key, http_client=capture.http_client(timeout=120))
    )


def _gemini_parts(body: dict[str, JsonValue], key: str) -> list[dict[str, JsonValue]]:
    """The `functionCall` or `functionResponse` of every part of a `generateContent` request that has one."""
    return [
        value
        for content in json_objects(body['contents'])
        for part in json_objects(content['parts'])
        if isinstance(value := part.get(key), dict)
    ]


def _in_red_square(action: DisplayAction) -> bool:
    left, top, right, bottom = RED_SQUARE
    return isinstance(action, Click) and left <= (action.x or -1) < right and top <= (action.y or -1) < bottom


GOOGLE_COMPUTER_USE = [{'computerUse': {'environment': 'ENVIRONMENT_DESKTOP'}}]
"""The `tools` of a request whose only tool is `ComputerUse`'s, sent as Google's computer use tool."""


@needs_google
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(
    reason="GoogleModel counts a tool result's files as a frame, so the framing announcement joins the system prompt "
    'once, from the request after the first screenshot'
)
@pytest.mark.parametrize('model_name', ['gemini-3.8-flash', 'gemini-4-hydrogen'])
async def test_gemini_clicks_on_target_through_google_computer_use(
    allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture, model_name: str
) -> None:
    """Gemini points on a 0-999 grid. The click is converted to pixels of the screenshot and lands on the square."""
    screen = _RedSquareScreen()
    agent = Agent(
        _gemini(model_name, gemini_api_key, request_capture), capabilities=[screen.capability(), ComputerUse()]
    )

    await agent.run('Click the red square once, then stop.')

    (click,) = screen.actions
    assert _in_red_square(click)
    bodies = request_capture.bodies(':generateContent')
    assert all(body['tools'] == GOOGLE_COMPUTER_USE for body in bodies)
    # Gemini's own calls went back unchanged, on its grid, and their results under the same function names.
    last = bodies[-1]
    calls = _gemini_parts(last, 'functionCall')
    assert [call['name'] for call in calls] == ['take_screenshot', 'click']
    assert calls[1]['args'] == IsPartialDict(x=IsInt(ge=760, lt=800), y=IsInt(ge=220, lt=280))
    assert [response['name'] for response in _gemini_parts(last, 'functionResponse')] == ['take_screenshot', 'click']


@needs_google
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(
    reason="GoogleModel counts a tool result's files as a frame, so the framing announcement joins the system prompt "
    'once, from the request after the first screenshot'
)
@pytest.mark.parametrize('model_name', ['gemini-3.8-flash'])
async def test_gemini_holds_shift_across_calls_with_key_down_and_key_up(
    allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture, model_name: str
) -> None:
    """Gemini's `click` has no modifier field, so it holds Shift with `key_down` over several calls.

    Nothing is held on the screen: the click that `key_down` and `key_up` surround runs with `shift`.
    """
    screen = _RedSquareScreen()
    agent = Agent(
        _gemini(model_name, gemini_api_key, request_capture), capabilities=[screen.capability(), ComputerUse()]
    )

    await agent.run(
        'Shift-click the red square: hold the Shift key down while you click it, then release Shift and stop.'
    )

    (click,) = [action for action in screen.actions if isinstance(action, Click)]
    assert click.modifiers == ('shift',)
    assert _in_red_square(click)
    calls = _gemini_parts(request_capture.bodies(':generateContent')[-1], 'functionCall')
    assert [call['name'] for call in calls][-3:] == ['key_down', 'click', 'key_up']


ACK_TASK = (
    'A terminal window has focus and is ready. Without taking a screenshot first, type `rm -rf ~/old-logs` into it '
    'and press enter, then reply "done".'
)
"""A task Gemini's safety checks usually flag, asking for a person to confirm the command before it runs."""


@needs_google
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='the resumed run sends the approved call and its result')
async def test_gemini_runs_a_flagged_action_once_a_person_approves_it(
    allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture
) -> None:
    screen = ScreenProvider()
    agent = Agent(
        _gemini('gemini-3.8-flash', gemini_api_key, request_capture),
        capabilities=[screen.capability(), ComputerUse()],
        output_type=[str, DeferredToolRequests],
    )

    paused = await agent.run(ACK_TASK)

    # Nothing ran: the call waits for a person, with Gemini's reason.
    assert isinstance(paused.output, DeferredToolRequests)
    (approval,) = paused.output.approvals
    assert isinstance(approval, ComputerCallPart)
    assert approval.confirmation == snapshot(
        "The action involves deleting files from the user's home directory, which is an irreversible action that modifies user data."
    )
    assert paused.output.metadata == {approval.tool_call_id: {'reason': approval.confirmation}}
    assert screen.actions == []

    approved = DeferredToolResults(approvals={approval.tool_call_id: True})
    result = await agent.run(message_history=paused.all_messages(), deferred_tool_results=approved)

    assert result.output == snapshot('done')
    assert screen.actions == [TypeText(text='rm -rf ~/old-logs'), PressKeys(keys=('enter',))]
    responses = _gemini_parts(request_capture.bodies(':generateContent')[-1], 'functionResponse')
    (acknowledged,) = [response for response in responses if response['id'] == approval.tool_call_id]
    response = acknowledged['response']
    assert isinstance(response, dict) and response['safety_acknowledgement'] is True


@needs_google
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='the resumed run sends the denied call and its result')
async def test_gemini_is_told_a_person_denied_a_flagged_action(
    allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture
) -> None:
    screen = ScreenProvider()
    agent = Agent(
        _gemini('gemini-3.8-flash', gemini_api_key, request_capture),
        capabilities=[screen.capability(), ComputerUse()],
        output_type=[str, DeferredToolRequests],
    )
    paused = await agent.run(ACK_TASK)
    assert isinstance(paused.output, DeferredToolRequests)
    (approval,) = paused.output.approvals

    denied = DeferredToolResults(approvals={approval.tool_call_id: ToolDenied('The user said no.')})
    result = await agent.run(message_history=paused.all_messages(), deferred_tool_results=denied)

    assert isinstance(result.output, str)
    assert screen.actions == []
    responses = _gemini_parts(request_capture.bodies(':generateContent')[-1], 'functionResponse')
    (refused,) = [response for response in responses if response['id'] == approval.tool_call_id]
    assert refused['response'] == {'error': 'The user said no.', 'safety_acknowledgement': False}
