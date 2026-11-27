"""The republished part start against the real JavaScript clients. #11734

What a retried activity's republished `*_START` does on the wire depends on the protocol, and
only the real clients can settle it — a Python port of the client logic can't. These scenarios
feed the republished-start streams straight to the AI SDK's `AbstractChat` and
`@ag-ui/client`'s `AbstractAgent`:

- Vercel AI has no chunk that closes a tool input early: a tool input ends with
  `tool-input-available`/`tool-input-error`, never with an "end" of its own, so the encoder's
  fixed stream still repeats `tool-input-start` for the same call. The real client tolerates
  it — the second start resets the input buffer, and the replayed deltas rebuild the whole
  input, so no Vercel-side follow-up is needed.
- AG-UI has `TOOL_CALL_END`, and the real client rejects a second `TOOL_CALL_START` for a call
  that is still open — the hard error behind this PR's end-before-restart rule. With the end
  in place the republished call is accepted, and its arguments are the full replay.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.ui_roundtrip

TOOL_START = {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'get_time', 'parentMessageId': 'msg-0'}


def _agui_turn(js_driver: Any, session: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    js_driver.request(
        'agui.submit', session=session, message={'id': f'{session}-user', 'role': 'user', 'content': 'go'}
    )
    return js_driver.request('agui.receive', session=session, events=events)


def _agui_events(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A finished AG-UI run whose assistant message carries the tool-call events given."""
    return [
        {'type': 'RUN_STARTED', 'threadId': 't-1', 'runId': 'r-1', 'protocolVersion': '1.0'},
        {'type': 'TEXT_MESSAGE_START', 'messageId': 'msg-0', 'role': 'assistant'},
        {'type': 'TEXT_MESSAGE_END', 'messageId': 'msg-0'},
        *calls,
        {'type': 'TOOL_CALL_RESULT', 'messageId': 'msg-1', 'toolCallId': 'call_0', 'content': '12:00', 'role': 'tool'},
        {'type': 'RUN_FINISHED', 'threadId': 't-1', 'runId': 'r-1', 'outcome': {'type': 'success'}},
    ]


def test_ag_ui_client_rejects_second_start_for_open_tool_call(js_driver: Any) -> None:
    """Without this PR's `TOOL_CALL_END`, the real client hard-errors on the republished start."""
    events = _agui_events(
        [
            TOOL_START,
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"tz": '},
            TOOL_START,
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"tz": "UTC"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
        ]
    )
    result = _agui_turn(js_driver, 'republished-reject', events)
    assert result['error'] is not None
    assert 'already in progress' in result['error']


def test_ag_ui_client_accepts_ended_then_republished_tool_call(js_driver: Any) -> None:
    """With the end before the republished start, the client takes the call and its full replay."""
    events = _agui_events(
        [
            TOOL_START,
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"tz": '},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            TOOL_START,
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"tz": '},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '"UTC"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
        ]
    )
    result = _agui_turn(js_driver, 'republished-accept', events)
    assert result.get('error') is None
    [assistant] = [m for m in result['messages'] if m.get('role') == 'assistant']
    [call] = assistant['toolCalls']
    assert call['id'] == 'call_0'
    assert call['function']['name'] == 'get_time'
    assert call['function']['arguments'] == '{"tz": "UTC"}'


def test_vercel_client_tolerates_republished_tool_input_start(js_driver: Any) -> None:
    """Vercel AI has no way to close a tool input early, so the fixed stream repeats the start.

    The real `AbstractChat` handles it: the second `tool-input-start` resets the input buffer
    and the replayed deltas rebuild the whole input, so no Vercel-adapter follow-up is needed.
    """
    js_driver.request('vercel.submit', session='republished-vercel', message={'text': 'go'})
    chunks: list[dict[str, Any]] = [
        {'type': 'start'},
        {'type': 'start-step'},
        {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'get_time'},
        {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"tz": '},
        # The activity retry republishes its part start: no tool-input-end exists to send.
        {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'get_time'},
        {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"tz": '},
        {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '"UTC"}'},
        {'type': 'tool-input-available', 'toolCallId': 'call_0', 'toolName': 'get_time', 'input': {'tz': 'UTC'}},
        {'type': 'tool-output-available', 'toolCallId': 'call_0', 'output': '12:00'},
        {'type': 'finish-step'},
        {'type': 'finish'},
    ]
    result = js_driver.request('vercel.receive', session='republished-vercel', chunks=chunks)
    assert result.get('error') is None
    [assistant] = [m for m in result['messages'] if m.get('role') == 'assistant']
    [tool] = [p for p in assistant['parts'] if p['type'] == 'tool-get_time']
    assert tool['toolCallId'] == 'call_0'
    assert tool['input'] == {'tz': 'UTC'}
    assert tool['output'] == '12:00'
