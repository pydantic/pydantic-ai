"""Google's computer use tool, sent in place of a tool of the `computer` kind.

These aren't recorded (VCR) tests. They check the exact request and response shapes, and cover cases no live model
produces on demand, such as stored histories and changed calls. The recorded runs are in `tests/harness/computer_use/`.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from httpx2 import AsyncClient, MockTransport, Request, Response
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelMessagesTypeAdapter,
    RunContext,
    ToolDenied,
)
from pydantic_ai.capabilities import AbstractCapability, Hooks, ProcessHistory, RawToolArgs, ValidatedToolArgs
from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    Click,
    ComputerCallPart,
    ComputerReturnPart,
    MouseDown,
    MouseMove,
    MouseUp,
    PressKeys,
    TypeText,
)
from pydantic_ai.exceptions import ModelRetry, SkipToolValidation, ToolFailed, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import AbstractNativeTool, ComputerTool
from pydantic_ai.profiles.google import GoogleModelProfile
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef

from ...computer_fakes import FAKE_DISPLAY_SIZE, FakeComputer
from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider

with try_import() as imports_successful:
    from pydantic_ai.models._google_computer import map_computer_call
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider
    from pydantic_ai_harness.computer_use import ComputerUse

    from ...harness.computer_use.display_fakes import ScreenBackend, ScreenProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai or pydantic-ai-harness not installed')

COMPUTER = ToolDefinition(name='computer', parameters_json_schema={'type': 'object'}, tool_kind=COMPUTER_KIND.id)
COMPUTER_TOOL = ComputerTool(display_size=FAKE_DISPLAY_SIZE)
GOOGLE_COMPUTER_USE = {'computerUse': {'environment': 'ENVIRONMENT_DESKTOP'}}

Turn = list[dict[str, JsonValue]]
"""The parts of one scripted Gemini response."""


def _call(name: str, id: str = 'call_1', **args: JsonValue) -> dict[str, JsonValue]:
    return {'functionCall': {'name': name, 'args': args, 'id': id}}


def _response(parts: Turn) -> dict[str, JsonValue]:
    content: dict[str, JsonValue] = {'role': 'model', 'parts': list[JsonValue](parts)}
    return {
        'candidates': [{'content': content, 'finishReason': 'STOP'}],
        'usageMetadata': {'promptTokenCount': 1, 'candidatesTokenCount': 1, 'totalTokenCount': 2},
        'modelVersion': 'gemini-3.8-flash',
    }


class _Gemini:
    """A Gemini API that answers each request with the next scripted turn, and keeps the request bodies."""

    def __init__(self, *turns: Turn) -> None:
        self.turns = list(turns)
        self.bodies: list[dict[str, JsonValue]] = []

    async def handle(self, request: Request) -> Response:
        self.bodies.append(json.loads(request.content))
        response = _response(self.turns.pop(0))
        if 'streamGenerateContent' in request.url.path:
            return Response(
                200, content=f'data: {json.dumps(response)}\n\n'.encode(), headers={'content-type': 'text/event-stream'}
            )
        return Response(200, json=response)

    @asynccontextmanager
    async def model(self, name: str = 'gemini-3.8-flash') -> AsyncGenerator[GoogleModel]:
        async with AsyncClient(transport=MockTransport(self.handle)) as client:
            provider = GoogleProvider(api_key='test-key', http_client=client, base_url='http://localhost')
            yield GoogleModel(name, provider=provider)


def _params(*extra: ToolDefinition) -> ModelRequestParameters:
    return ModelRequestParameters(
        function_tools=[COMPUTER, *extra], native_tools=[COMPUTER_TOOL], allow_text_output=True
    )


def _function_parts(body: dict[str, JsonValue], key: str) -> list[JsonValue]:
    contents = body['contents']
    assert isinstance(contents, list)
    return [
        part[key]
        for content in contents
        if isinstance(content, dict) and isinstance(parts := content['parts'], list)
        for part in parts
        if isinstance(part, dict) and key in part
    ]


@pytest.mark.parametrize(
    'name, args, actions',
    [
        ('click', {'x': 500, 'y': 500}, [{'type': 'click', 'x': 640, 'y': 400, 'button': 'left', 'count': 1}]),
        ('double_click', {'x': 0, 'y': 0}, [{'type': 'click', 'x': 0, 'y': 0, 'button': 'left', 'count': 2}]),
        ('triple_click', {'x': 999, 'y': 999}, [{'type': 'click', 'x': 1278, 'y': 799, 'button': 'left', 'count': 3}]),
        ('middle_click', {'x': 500, 'y': 500}, [{'type': 'click', 'x': 640, 'y': 400, 'button': 'middle', 'count': 1}]),
        ('right_click', {'x': 500, 'y': 500}, [{'type': 'click', 'x': 640, 'y': 400, 'button': 'right', 'count': 1}]),
        ('move', {'x': 500, 'y': 500}, [{'type': 'mouse_move', 'x': 640, 'y': 400}]),
        ('mouse_down', {'x': 500, 'y': 500}, [{'type': 'mouse_move', 'x': 640, 'y': 400}, {'type': 'mouse_down'}]),
        ('mouse_up', {'x': 500, 'y': 500}, [{'type': 'mouse_move', 'x': 640, 'y': 400}, {'type': 'mouse_up'}]),
        (
            'drag_and_drop',
            {'start_x': 0, 'start_y': 0, 'end_x': 500, 'end_y': 500},
            [{'type': 'drag', 'path': [{'x': 0, 'y': 0}, {'x': 640, 'y': 400}]}],
        ),
        (
            'scroll',
            {'x': 500, 'y': 500, 'direction': 'down', 'magnitude_in_wheel_clicks': 3},
            [{'type': 'scroll', 'x': 640, 'y': 400, 'direction': 'down', 'amount': 3}],
        ),
        (
            'scroll',
            {'x': 500, 'y': 500, 'direction': 'up'},
            [{'type': 'scroll', 'x': 640, 'y': 400, 'direction': 'up', 'amount': 3}],
        ),
        ('type', {'text': 'hi'}, [{'type': 'type_text', 'text': 'hi'}]),
        (
            'type',
            {'text': 'hi', 'press_enter': True},
            [{'type': 'type_text', 'text': 'hi'}, {'type': 'press_keys', 'keys': ['enter']}],
        ),
        ('wait', {}, [{'type': 'wait'}]),
        ('wait', {'seconds': 2.5}, [{'type': 'wait', 'seconds': 2.5}]),
        # Gemini sends X11 key names, which become the tool's.
        ('press_key', {'key': 'Return'}, [{'type': 'press_keys', 'keys': ['enter']}]),
        ('press_key', {'key': 'KP_Add'}, [{'type': 'press_keys', 'keys': ['+']}]),
        (
            'hotkey',
            {'keys': ['Ctrl', 'Shift', 'Page_Down']},
            [{'type': 'press_keys', 'keys': ['ctrl', 'shift', 'pagedown']}],
        ),
        # Nothing is held on the screen: the modifier goes on Gemini's next calls.
        ('key_down', {'key': 'Shift_L'}, [{'type': 'screenshot'}]),
        ('key_up', {'key': 'Shift_L'}, [{'type': 'screenshot'}]),
        (
            'key_down',
            {'key': 'Hyper_L'},
            [{'type': 'screenshot', 'unsupported_fields': ["key 'Hyper_L', which is not a modifier"]}],
        ),
        ('take_screenshot', {'intent': 'Look.'}, [{'type': 'screenshot'}]),
        # What doesn't match becomes an action the `computer` tool refuses, naming it.
        ('click', {'x': '5', 'y': 5, 'z': 1}, [{'type': 'click', 'unsupported_fields': ['x', 'z']}]),
        ('open_web_browser', {'url': 'x'}, [{'type': 'open_web_browser', 'unsupported_fields': ['url']}]),
        (
            'click',
            {'x': 1, 'y': 1, 'safety_decision': {'decision': 'block'}},
            [{'type': 'click', 'unsupported_fields': ['safety_decision']}],
        ),
    ],
)
def test_each_function_becomes_computer_actions_in_pixels(
    name: str, args: dict[str, JsonValue], actions: list[dict[str, object]]
) -> None:
    assert map_computer_call(name, args, FAKE_DISPLAY_SIZE)[:2] == (actions, None)


def test_a_flagged_call_carries_geminis_explanation() -> None:
    flagged: dict[str, JsonValue] = {'decision': 'require_confirmation', 'explanation': 'It sends a message.'}
    assert (
        map_computer_call('click', {'x': 0, 'y': 0, 'safety_decision': flagged}, (10, 10))[1] == 'It sends a message.'
    )
    # Google documents an explanation on every flagged call, so a decision without one isn't run.
    unexplained: dict[str, JsonValue] = {'decision': 'require_confirmation'}
    assert map_computer_call('click', {'x': 0, 'y': 0, 'safety_decision': unexplained}, (10, 10))[:2] == (
        [{'type': 'click', 'unsupported_fields': ['safety_decision']}],
        None,
    )


async def test_the_computer_is_sent_as_googles_tool_beside_other_function_tools(allow_model_requests: None) -> None:
    gemini = _Gemini([_call('click', x=500, y=500)])
    lookup = ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})
    async with gemini.model() as model:
        response = await model.request([ModelRequest.user_text_prompt('Go.')], None, _params(lookup))

    (body,) = gemini.bodies
    tools = body['tools']
    assert isinstance(tools, list) and len(tools) == 2 and GOOGLE_COMPUTER_USE in tools
    # The `computer` function tool isn't declared: Google's tool takes its place.
    assert "'name': 'computer'" not in str(tools) and "'name': 'lookup'" in str(tools)
    (call,) = response.parts
    assert isinstance(call, ComputerCallPart)
    assert call.actions is not None and call.args_as_dict() == {
        'actions': [{'type': 'click', 'x': 640, 'y': 400, 'button': 'left', 'count': 1}]
    }
    assert call.provider_name == 'google'
    assert call.tool_call_id == 'call_1'
    assert call.confirmation is None


@pytest.mark.parametrize(
    'tool_choice, function_calling_config, tools',
    [
        ('auto', None, [GOOGLE_COMPUTER_USE]),
        ('required', {'mode': 'ANY'}, [GOOGLE_COMPUTER_USE]),
        (['computer'], {'mode': 'ANY'}, [GOOGLE_COMPUTER_USE]),
    ],
)
async def test_tool_choice_reaches_googles_tool(
    allow_model_requests: None,
    tool_choice: ToolOrOutput,
    function_calling_config: JsonValue,
    tools: list[JsonValue],
) -> None:
    gemini = _Gemini([{'text': 'ok'}])
    async with gemini.model() as model:
        params = _params()
        await model.request([ModelRequest.user_text_prompt('Go.')], ModelSettings(tool_choice=tool_choice), params)

    (body,) = gemini.bodies
    assert body['tools'] == tools
    tool_config = body.get('toolConfig')
    assert (
        tool_config.get('functionCallingConfig') if isinstance(tool_config, dict) else None
    ) == function_calling_config


async def test_forcing_the_computer_leaves_out_the_other_function_tools(allow_model_requests: None) -> None:
    gemini = _Gemini([{'text': 'ok'}])
    lookup = ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})
    async with gemini.model() as model:
        settings = ModelSettings(tool_choice=['computer'])
        await model.request([ModelRequest.user_text_prompt('Go.')], settings, _params(lookup))

    (body,) = gemini.bodies
    assert body['tools'] == [GOOGLE_COMPUTER_USE]
    assert body['toolConfig'] == {'functionCallingConfig': {'mode': 'ANY'}}


async def test_choosing_the_computer_and_a_function_tool_keeps_both_callable(allow_model_requests: None) -> None:
    """`allowedFunctionNames` lists only the function tool: Gemini refuses a computer function's name there, and keeps
    its computer functions callable beside the names listed (checked live on gemini-3.8-flash)."""
    gemini = _Gemini([{'text': 'ok'}])
    lookup = ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})
    other = ToolDefinition(name='other', parameters_json_schema={'type': 'object'})
    async with gemini.model() as model:
        settings = ModelSettings(tool_choice=['computer', 'lookup'])
        await model.request([ModelRequest.user_text_prompt('Go.')], settings, _params(lookup, other))

    (body,) = gemini.bodies
    assert body['toolConfig'] == {'functionCallingConfig': {'mode': 'ANY', 'allowedFunctionNames': ['lookup']}}
    tools = body['tools']
    assert isinstance(tools, list) and GOOGLE_COMPUTER_USE in tools


async def test_forcing_a_function_tool_sends_no_computer(allow_model_requests: None) -> None:
    """Gemini could still call its computer functions beside `allowedFunctionNames`, so Google's tool is left out."""
    gemini = _Gemini([{'text': 'ok'}])
    lookup = ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})
    async with gemini.model() as model:
        settings = ModelSettings(tool_choice=['lookup'])
        await model.request([ModelRequest.user_text_prompt('Go.')], settings, _params(lookup))

    (body,) = gemini.bodies
    tools = body['tools']
    assert isinstance(tools, list) and GOOGLE_COMPUTER_USE not in tools
    assert body['toolConfig'] == {'functionCallingConfig': {'mode': 'ANY', 'allowedFunctionNames': ['lookup']}}


async def test_tool_choice_none_sends_no_computer(allow_model_requests: None) -> None:
    gemini = _Gemini([{'text': 'ok'}])
    async with gemini.model() as model:
        await model.request([ModelRequest.user_text_prompt('Go.')], ModelSettings(tool_choice='none'), _params())

    (body,) = gemini.bodies
    tools = body.get('tools')
    assert not isinstance(tools, list) or GOOGLE_COMPUTER_USE not in tools


async def test_a_function_tool_named_like_a_predefined_function_is_refused(allow_model_requests: None) -> None:
    click = ToolDefinition(name='click', parameters_json_schema={'type': 'object'})
    async with _Gemini().model() as model:
        with pytest.raises(UserError, match="predefined functions named 'click'"):
            await model.request([ModelRequest.user_text_prompt('Go.')], None, _params(click))


async def test_an_optional_computer_tool_falls_back_beside_a_clashing_function_tool(allow_model_requests: None) -> None:
    """`ComputerUse` sends an optional `ComputerTool`, so a clashing name sends the `computer` function tool instead."""
    gemini = _Gemini([_call('click', x=1, y=2)])
    click = ToolDefinition(name='click', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(
        function_tools=[COMPUTER, click],
        native_tools=[ComputerTool(display_size=FAKE_DISPLAY_SIZE, optional=True)],
        allow_text_output=True,
    )
    async with gemini.model() as model:
        response = await model.request([ModelRequest.user_text_prompt('Go.')], None, params)

    (body,) = gemini.bodies
    tools = body['tools']
    assert isinstance(tools, list) and GOOGLE_COMPUTER_USE not in tools
    assert "'name': 'computer'" in str(tools) and "'name': 'click'" in str(tools)
    # Gemini's `click` call goes to the function tool of that name, not to the computer.
    (call,) = response.parts
    assert isinstance(call, ToolCallPart) and not isinstance(call, ComputerCallPart)
    assert call.tool_name == 'click'


async def test_calls_go_back_as_geminis_and_changed_calls_as_pixels(allow_model_requests: None) -> None:
    """An unchanged call goes back as Gemini sent it. A call a hook changed goes back as a plain `computer` call."""
    gemini = _Gemini(
        [_call('click', 'call_1', x=500, y=500, intent='Click.'), _call('type', 'call_2', text='hi')],
        [{'text': 'done'}],
    )
    async with gemini.model() as model:
        params = _params()
        response = await model.request([ModelRequest.user_text_prompt('Go.')], None, params)
        first, second = response.parts
        assert isinstance(first, ComputerCallPart) and isinstance(second, ComputerCallPart)
        second.args = {'actions': [{'type': 'type_text', 'text': 'actions_changed'}]}
        results = [
            ComputerReturnPart(tool_name='computer', content='ok', tool_call_id=part.tool_call_id)
            for part in (first, second)
        ]
        history = [ModelRequest.user_text_prompt('Go.'), response, ModelRequest(parts=results)]
        await model.request(history, None, params)

    calls = _function_parts(gemini.bodies[1], 'functionCall')
    assert calls == snapshot(
        [
            {'name': 'click', 'args': {'x': 500, 'y': 500, 'intent': 'Click.'}, 'id': 'call_1'},
            {
                'name': 'computer',
                'args': {'actions': [{'type': 'type_text', 'text': 'actions_changed'}]},
                'id': 'call_2',
            },
        ]
    )
    names = [part['name'] for part in _function_parts(gemini.bodies[1], 'functionResponse') if isinstance(part, dict)]
    assert names == ['click', 'computer']


FLAGGED: dict[str, JsonValue] = {'decision': 'require_confirmation', 'explanation': 'It deletes files.'}


async def _answer_to_flagged_call(
    approval: bool | ToolDenied | None = True,
    *,
    hooks: Hooks[object] | None = None,
    provider: ScreenProvider | None = None,
    call: dict[str, JsonValue] | None = None,
) -> tuple[JsonValue, ScreenProvider]:
    """What Gemini receives for its flagged call, run by the real `ComputerUse`, and the screen it ran on.

    The run pauses for approval and resumes with `approval`. With `approval=None`, the run must not pause.
    """
    provider = provider or ScreenProvider()
    gemini = _Gemini([call or _call('type', text='send', safety_decision=FLAGGED)], [{'text': 'done'}])
    async with gemini.model() as model:
        agent = Agent(
            model,
            capabilities=[provider.capability(), ComputerUse(), *([hooks] if hooks else [])],
            output_type=[str, DeferredToolRequests],
        )
        result = await agent.run('Go.')
        if approval is None:
            assert result.output == 'done'
        else:
            assert isinstance(result.output, DeferredToolRequests)
            assert provider.actions == []
            (flagged,) = result.output.approvals
            answer = DeferredToolResults(approvals={flagged.tool_call_id: approval})
            await agent.run(message_history=result.all_messages(), deferred_tool_results=answer)
    (function_response,) = _function_parts(gemini.bodies[-1], 'functionResponse')
    assert isinstance(function_response, dict)
    return function_response['response'], provider


RAN = 'Ran 1 action. Screenshot attached (1280x800).'


async def test_an_approved_flagged_call_that_ran_is_acknowledged(allow_model_requests: None) -> None:
    response, provider = await _answer_to_flagged_call()
    assert provider.actions == [TypeText(text='send')]
    assert response == {'return_value': RAN, 'safety_acknowledgement': True}


async def test_a_hook_that_skips_validation_still_waits_for_approval(allow_model_requests: None) -> None:
    hooks = Hooks[object]()

    @hooks.on.before_tool_validate
    async def skip(
        ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs
    ) -> RawToolArgs:
        raise SkipToolValidation({'actions': [TypeText(text='send')]})

    response, provider = await _answer_to_flagged_call(hooks=hooks)
    assert provider.actions == [TypeText(text='send')]
    assert response == {'return_value': RAN, 'safety_acknowledgement': True}


@pytest.mark.parametrize('refusal', [ModelRetry, ToolFailed])
async def test_a_flagged_call_a_validation_hook_refuses_is_not_acknowledged(
    allow_model_requests: None, refusal: type[ModelRetry | ToolFailed]
) -> None:
    """The hook refuses the call before anyone is asked, so nobody approved it."""
    hooks = Hooks[object]()

    @hooks.on.after_tool_validate
    async def refuse(
        ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        raise refusal('Typing is limited to 3 characters.')

    response, provider = await _answer_to_flagged_call(None, hooks=hooks)
    assert provider.actions == []
    assert response == {'error': 'Typing is limited to 3 characters.', 'safety_acknowledgement': False}


async def test_a_flagged_call_a_runner_ran_without_asking_is_not_acknowledged(allow_model_requests: None) -> None:
    """A `computer` tool that ignores the call's confirmation never sets `confirmed`, so Gemini hears nobody did."""
    computer = FakeComputer()
    gemini = _Gemini([_call('type', text='send', safety_decision=FLAGGED)], [{'text': 'done'}])
    async with gemini.model() as model:
        agent = Agent(model, capabilities=[computer, InMemoryProvider().capability()])
        assert (await agent.run('Go.')).output == 'done'

    assert computer.actions == [TypeText(text='send')]
    (function_response,) = _function_parts(gemini.bodies[-1], 'functionResponse')
    assert isinstance(function_response, dict)
    assert function_response['response'] == {'error': 'Ran 1 of 1 actions.', 'safety_acknowledgement': False}


async def test_an_approved_flagged_call_whose_screenshot_fails_is_not_acknowledged(
    allow_model_requests: None,
) -> None:
    """The call failed after the approval, so Gemini hears `false` beside why, and the run goes on."""

    class BadScreen(ScreenBackend):
        async def screenshot(self) -> bytes:
            raise WorkspaceError('scrot: not found')

    class Provider(ScreenProvider):
        def backend(self, ref: WorkspaceRef | None) -> ScreenBackend:
            return BadScreen(self, ref)

    response, provider = await _answer_to_flagged_call(provider=Provider())
    assert provider.actions == [TypeText(text='send')]
    assert response == {
        'error': 'Ran 1 action. Taking the screenshot failed: scrot: not found',
        'safety_acknowledgement': False,
    }


async def test_a_denied_flagged_call_says_so(allow_model_requests: None) -> None:
    response, provider = await _answer_to_flagged_call(ToolDenied('No.'))
    assert provider.actions == []
    assert response == {'error': 'No.', 'safety_acknowledgement': False}


async def test_a_flagged_call_with_arguments_that_fail_validation_is_not_acknowledged(
    allow_model_requests: None,
) -> None:
    """Arguments that don't validate are refused before anyone is asked to approve the call, so nobody approved it."""
    response, provider = await _answer_to_flagged_call(None, call=_call('click', x='5', y=5, safety_decision=FLAGGED))
    assert provider.actions == []
    assert isinstance(response, dict) and response['safety_acknowledgement'] is False
    assert 'unsupported_fields' in str(response['error'])


async def test_a_saved_history_keeps_the_acknowledgement(allow_model_requests: None) -> None:
    provider = ScreenProvider()
    gemini = _Gemini([_call('type', text='send', safety_decision=FLAGGED)], [{'text': 'done'}], [{'text': 'again'}])
    async with gemini.model() as model:
        agent = Agent(
            model, capabilities=[provider.capability(), ComputerUse()], output_type=[str, DeferredToolRequests]
        )
        paused = await agent.run('Go.')
        assert isinstance(paused.output, DeferredToolRequests)
        (flagged,) = paused.output.approvals
        answer = DeferredToolResults(approvals={flagged.tool_call_id: True})
        done = await agent.run(message_history=paused.all_messages(), deferred_tool_results=answer)
        saved = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(done.all_messages()))
        await agent.run('Again.', message_history=saved)

    (function_response,) = _function_parts(gemini.bodies[-1], 'functionResponse')
    assert isinstance(function_response, dict)
    assert function_response['response'] == {'return_value': RAN, 'safety_acknowledgement': True}


@pytest.mark.parametrize('stream', [False, True])
async def test_an_agent_drives_the_computer_through_googles_tool(allow_model_requests: None, stream: bool) -> None:
    gemini = _Gemini([_call('key_down', key='shift')], [_call('click', 'call_2', x=500, y=500)], [{'text': 'done'}])
    computer = FakeComputer()
    async with gemini.model() as model:
        agent = Agent(model, capabilities=[computer, InMemoryProvider().capability()])
        if stream:
            async with agent.run_stream('Go.') as result:
                await result.get_output()
        else:
            await agent.run('Go.')

    # `key_down` becomes a screenshot, which this fake doesn't record as the last action.
    assert computer.actions == [Click(x=640, y=400, modifiers=('shift',))]
    assert [part['name'] for part in _function_parts(gemini.bodies[-1], 'functionCall') if isinstance(part, dict)] == [
        'key_down',
        'click',
    ]


async def _display_actions(
    *turns: Turn, capabilities: tuple[AbstractCapability[object], ...] = ()
) -> tuple[ScreenProvider, _Gemini]:
    """Run an agent with the real `ComputerUse` on Gemini's scripted turns, then answer `done`.

    Returns the screen, which records the actions that reached it, and the Gemini API with the request bodies.
    """
    provider = ScreenProvider()
    gemini = _Gemini(*turns, [{'text': 'done'}])
    async with gemini.model() as model:
        agent = Agent(model, capabilities=[provider.capability(), ComputerUse(), *capabilities])
        assert (await agent.run('Go.')).output == 'done'
    return provider, gemini


async def test_a_shift_click_across_three_turns_clicks_with_shift_and_key_up_lets_go(
    allow_model_requests: None,
) -> None:
    provider, _ = await _display_actions(
        [_call('key_down', key='shift')],
        [_call('click', 'call_2', x=500, y=500)],
        [_call('key_up', 'call_3', key='shift')],
        [_call('click', 'call_4', x=500, y=500)],
    )
    assert provider.actions == [Click(x=640, y=400, modifiers=('shift',)), Click(x=640, y=400)]


async def test_a_shift_click_in_one_turn_clicks_with_shift(allow_model_requests: None) -> None:
    """Gemini often sends `key_down`, `click` and `key_up` together, and they apply in order."""
    provider, _ = await _display_actions(
        [
            _call('key_down', key='Shift'),
            _call('click', 'call_2', x=500, y=500),
            _call('key_up', 'call_3', key='Shift'),
            _call('click', 'call_4', x=500, y=500),
        ]
    )
    assert provider.actions == [Click(x=640, y=400, modifiers=('shift',)), Click(x=640, y=400)]


async def test_a_hotkey_while_ctrl_is_held_presses_ctrl_with_it(allow_model_requests: None) -> None:
    provider, _ = await _display_actions(
        [_call('key_down', key='Control_L')], [_call('hotkey', 'call_2', keys=['shift', 'tab'])]
    )
    assert provider.actions == [PressKeys(keys=('ctrl', 'shift', 'tab'))]


async def test_holding_a_key_that_is_not_a_modifier_is_refused(allow_model_requests: None) -> None:
    provider, gemini = await _display_actions([_call('key_down', key='a')], [_call('click', 'call_2', x=500, y=500)])
    (refused,) = _function_parts(gemini.bodies[1], 'functionResponse')
    assert isinstance(refused, dict) and isinstance(response := refused['response'], dict)
    assert "key 'a', which is not a modifier" in str(response['error'])
    # The refused `key_down` holds nothing, so the next click has no modifiers.
    assert provider.actions == [Click(x=640, y=400)]


def _names(body: dict[str, JsonValue]) -> list[JsonValue]:
    return [part['name'] for part in _function_parts(body, 'functionCall') if isinstance(part, dict)]


def _error(gemini: _Gemini, call_id: str) -> str:
    """The error Gemini got back for its call `call_id`."""
    (content,) = [
        part['response']
        for part in _function_parts(gemini.bodies[-1], 'functionResponse')
        if isinstance(part, dict) and part['id'] == call_id
    ]
    assert isinstance(content, dict)
    return str(content['error'])


async def test_a_changed_key_up_still_lets_go(allow_model_requests: None) -> None:
    """The held keys come from Gemini's own calls, so a changed `key_up` still lets go, and other calls go back as-is."""

    def change_key_up(messages: list[ModelMessage]) -> list[ModelMessage]:
        for part in (part for m in messages for part in m.parts):
            if isinstance(part, ComputerCallPart) and part.tool_call_id == 'call_2':
                part.args = {'actions': [{'type': 'wait'}]}
        return messages

    provider, gemini = await _display_actions(
        [_call('key_down', key='shift')],
        [_call('key_up', 'call_2', key='shift')],
        [_call('click', 'call_3', x=500, y=500)],
        capabilities=(ProcessHistory(change_key_up),),
    )
    assert provider.actions == [Click(x=640, y=400)]
    assert _names(gemini.bodies[-1]) == ['key_down', 'computer', 'click']


async def test_a_refused_key_down_holds_nothing(allow_model_requests: None) -> None:
    provider, gemini = await _display_actions(
        [_call('key_down', key='shift', bogus=1)], [_call('click', 'call_2', x=500, y=500)]
    )
    assert 'bogus' in _error(gemini, 'call_1')
    assert provider.actions == [Click(x=640, y=400)]


async def test_a_key_down_a_hook_refuses_holds_nothing(allow_model_requests: None) -> None:
    hooks = Hooks[object]()

    @hooks.on.after_tool_validate
    async def refuse_key_down(
        ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        if call.tool_call_id == 'call_1':
            raise ToolFailed('Holding keys is not allowed.')
        return args

    provider, _ = await _display_actions(
        [_call('key_down', key='shift')], [_call('click', 'call_2', x=500, y=500)], capabilities=(hooks,)
    )
    assert provider.actions == [Click(x=640, y=400)]


async def test_mouse_up_while_a_key_is_held_still_lets_go(allow_model_requests: None) -> None:
    """A button pressed before the key was held is released, so it never stays down."""
    provider, _ = await _display_actions(
        [_call('mouse_down', x=500, y=500)],
        [_call('key_down', 'call_2', key='shift')],
        [_call('mouse_up', 'call_3', x=500, y=500)],
    )
    assert provider.actions == [MouseMove(x=640, y=400), MouseDown(), MouseMove(x=640, y=400), MouseUp()]


async def test_a_hotkey_naming_a_held_key_presses_it_once(allow_model_requests: None) -> None:
    provider, _ = await _display_actions(
        [_call('key_down', key='Shift_L')], [_call('hotkey', 'call_2', keys=['shift', 'tab'])]
    )
    assert provider.actions == [PressKeys(keys=('shift', 'tab'))]


async def test_a_key_stays_held_when_a_history_processor_drops_its_key_down(allow_model_requests: None) -> None:
    """Each call stores the keys held when it ran, so the last one left in the history says what is held."""

    def drop_key_down_after_a_click(messages: list[ModelMessage]) -> list[ModelMessage]:
        ids = [getattr(part, 'tool_call_id', None) for m in messages for part in m.parts]
        if 'call_2' not in ids:
            return messages
        return [m for m in messages if all(getattr(part, 'tool_call_id', None) != 'call_1' for part in m.parts)]

    provider, gemini = await _display_actions(
        [_call('key_down', key='shift')],
        [_call('click', 'call_2', x=500, y=500)],
        [_call('click', 'call_3', x=500, y=500)],
        capabilities=(ProcessHistory(drop_key_down_after_a_click),),
    )
    shift_click = Click(x=640, y=400, modifiers=('shift',))
    assert provider.actions == [shift_click, shift_click]
    assert _names(gemini.bodies[-1]) == ['click', 'click']


@pytest.mark.parametrize('model_name', ['gemini-2.5-flash', 'gemini-3.1-flash-image', 'gemini-3.1-pro-preview'])
def test_models_google_does_not_list_for_computer_use_leave_it_out(model_name: str) -> None:
    model = GoogleModel(model_name, provider=GoogleProvider(api_key='test-key'))
    assert ComputerTool not in model.profile.get('supported_native_tools', frozenset[type[AbstractNativeTool]]())


@pytest.mark.parametrize(
    'model_name', ['gemini-3.8-flash', 'gemini-3.5-flash-lite', 'gemini-4-hydrogen', 'gemini-3-flash-preview']
)
def test_computer_use_models_support_it_on_the_gemini_api_only(model_name: str) -> None:
    gemini_api = GoogleModel(model_name, provider=GoogleProvider(api_key='test-key'))
    assert ComputerTool in gemini_api.profile.get('supported_native_tools', frozenset[type[AbstractNativeTool]]())
    vertex = GoogleModel(model_name, provider=GoogleCloudProvider(project='p', location='us-central1'))
    assert ComputerTool not in vertex.profile.get('supported_native_tools', frozenset[type[AbstractNativeTool]]())
    # A user who has checked the tool on Vertex AI turns it on with `profile=`.
    checked = GoogleModel(
        model_name,
        provider=GoogleCloudProvider(project='p', location='us-central1'),
        profile=GoogleModelProfile(google_supports_computer_use=True),
    )
    assert ComputerTool in checked.profile.get('supported_native_tools', frozenset[type[AbstractNativeTool]]())


async def test_calls_gemini_did_not_make_go_back_as_function_calls(allow_model_requests: None) -> None:
    """A `computer` call from another model and another tool's call have no Gemini call to send back."""
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Go.'),
        ModelResponse(
            parts=[
                ToolCallPart('lookup', {}, tool_call_id='call_a'),
                ComputerCallPart('computer', {'actions': [{'type': 'screenshot'}]}, tool_call_id='call_b'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('lookup', 'found', tool_call_id='call_a'),
                ComputerReturnPart('computer', 'Ran 1 action.', tool_call_id='call_b'),
            ]
        ),
    ]
    gemini = _Gemini([{'functionCall': {'name': 'click', 'args': {'x': 500, 'y': 500}}}])
    lookup = ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})
    async with gemini.model() as model:
        response = await model.request(history, None, _params(lookup))

    assert _names(gemini.bodies[0]) == ['lookup', 'computer']
    # Gemini sent no ID, so the call keeps the one the part generated, not one from the history.
    (call,) = response.parts
    assert isinstance(call, ComputerCallPart)
    assert call.tool_call_id.startswith('pyd_ai_')
