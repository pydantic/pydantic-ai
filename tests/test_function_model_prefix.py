"""Tests for the prompt-cache prefix check of `FunctionModel` and `TestModel` conversations."""

from __future__ import annotations

from typing import cast

import pytest

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from .conftest import try_import
from .function_model_prefix import (
    RequestRecorder,
    check_function_model_prefix_stability,
    fail_on_prefix_violations,
    function_model_prefix_violations,
)

with try_import() as imports_successful:
    # The check renders each request with `AnthropicModel`.
    from pydantic_ai.models.anthropic import AnthropicModel as AnthropicModel

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


def _respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    # The model is only recorded, never run.
    raise NotImplementedError  # pragma: no cover


def _request(text: str, run_id: str = 'run') -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(text)], run_id=run_id)


def _response(text: str, run_id: str = 'run') -> ModelResponse:
    return ModelResponse(parts=[TextPart(text)], run_id=run_id)


def _tools(*names: str, deferred: tuple[str, ...] = ()) -> ModelRequestParameters:
    return ModelRequestParameters(
        function_tools=[
            *(ToolDefinition(name=name) for name in names),
            *(ToolDefinition(name=name, defer_loading=True) for name in deferred),
        ]
    )


def _levels(recorder: RequestRecorder) -> list[tuple[int, str]]:
    return [
        (violation.pair_index, violation.level) for violation in function_model_prefix_violations(recorder.requests)
    ]


def test_a_conversation_that_extends_its_prefix_passes() -> None:
    model = FunctionModel(_respond)
    recorder = RequestRecorder()
    first = _request('hi')
    recorder.record(model, [first], None, _tools('a'))
    recorder.record(model, [first, _response('hello'), _request('more')], None, _tools('a'))

    assert _levels(recorder) == []
    fail_on_prefix_violations([])


def test_a_changed_tool_moves_the_prefix() -> None:
    model = FunctionModel(_respond)
    recorder = RequestRecorder()
    first = _request('hi')
    recorder.record(model, [first], None, _tools('a', 'b'))
    recorder.record(model, [first, _response('hello'), _request('more')], None, _tools('b'))

    assert _levels(recorder) == [(1, 'tools')]
    with pytest.raises(pytest.fail.Exception, match=r'(?s)request 1, tools block 0:.*moves_cache_prefix'):
        fail_on_prefix_violations(function_model_prefix_violations(recorder.requests))


def test_a_shrunk_history_moves_the_prefix() -> None:
    model = FunctionModel(_respond)
    recorder = RequestRecorder()
    first = _request('hi')
    recorder.record(model, [first, _response('hello'), _request('more')], None, _tools())
    recorder.record(model, [first], None, _tools())

    assert _levels(recorder) == [(1, 'shrunk')]


def test_a_withdrawn_deferred_tool_is_a_violation() -> None:
    model = FunctionModel(_respond)
    recorder = RequestRecorder()
    first = _request('hi')
    recorder.record(model, [first], None, _tools('a', deferred=('d',)))
    recorder.record(model, [first, _response('hello'), _request('more')], None, _tools('a'))

    assert _levels(recorder) == [(1, 'deferred-tools')]


def test_interleaved_runs_are_each_compared_with_themselves() -> None:
    """Two runs of one prompt on one model, interleaved, don't compare one run's request with the other's."""
    model = FunctionModel(_respond)
    recorder = RequestRecorder()
    first_a, first_b = _request('hi', run_id='a'), _request('hi', run_id='b')
    recorder.record(model, [first_a], None, _tools('a'))
    recorder.record(model, [first_a, _response('hello', 'a'), _request('more', 'a')], None, _tools('a'))
    # Run `b`'s first request has fewer messages than run `a`'s latest one, so it continues no earlier request.
    recorder.record(model, [first_b], None, _tools('b'))
    recorder.record(model, [first_b, _response('hello', 'b'), _request('more', 'b')], None, _tools('b'))

    assert _levels(recorder) == []


def test_a_run_from_an_earlier_runs_history_continues_it() -> None:
    model = FunctionModel(_respond)
    recorder = RequestRecorder()
    first = _request('hi', run_id='a')
    recorder.record(model, [first], None, _tools('a'))
    recorder.record(model, [first, _response('hello', 'a'), _request('more', 'b')], None, _tools('b'))

    assert _levels(recorder) == [(1, 'tools')]


def test_requests_to_other_models_are_other_conversations() -> None:
    recorder = RequestRecorder()
    first = _request('hi')
    recorder.record(FunctionModel(_respond), [first], None, _tools('a'))
    recorder.record(FunctionModel(_respond), [first, _response('hello'), _request('more')], None, _tools('b'))

    assert _levels(recorder) == []


def test_a_cache_control_key_in_a_tool_schema_still_counts() -> None:
    """Breakpoints are stripped before comparing, but a tool parameter named `cache_control` is the tool's own."""
    model = FunctionModel(_respond)
    recorder = RequestRecorder()
    first = _request('hi')

    def tool(*parameters: str) -> ModelRequestParameters:
        schema = {'type': 'object', 'properties': {name: {'type': 'string'} for name in parameters}}
        return ModelRequestParameters(function_tools=[ToolDefinition(name='a', parameters_json_schema=schema)])

    recorder.record(model, [first], None, tool('cache_control', 'x'))
    recorder.record(model, [first, _response('hello'), _request('more')], None, tool('x'))

    assert _levels(recorder) == [(1, 'tools')]


@pytest.mark.moves_cache_prefix
def test_a_marker_without_a_reason_fails(request: pytest.FixtureRequest) -> None:
    model = FunctionModel(_respond)
    recorder = RequestRecorder()
    first = _request('hi')
    recorder.record(model, [first], None, _tools())
    recorder.record(model, [first, _response('hello'), _request('more')], None, _tools())

    node = cast(pytest.Item, request.node)  # pyright: ignore[reportUnknownMemberType]
    with pytest.raises(pytest.fail.Exception, match='requires reason='):
        check_function_model_prefix_stability(node, recorder.requests)
