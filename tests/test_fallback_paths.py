"""Every way of falling back between models judges an attempt by the same `fallback_on` rules.

`FallbackModel` judges an attempt on four paths of its own: `request` and `request_stream`, each either walking its
chain or continuing a suspended response on the model it's pinned to. The `ModelFallback` capability judges it in the
agent's attempt loop, streamed or not. The predicates are shared, but each path decides on its own what to do with
their verdict, and the paths have drifted apart before: a pinned continuation once accepted a response its response
handler rejected (#11085), a tuple holding handlers was read as a tuple of exception types (#10031), and a stream
that failed before its first event didn't fall back (#10386). So this runs every path against every form
`fallback_on` takes, and against an error, an error a stream raises before its first event, and a rejected response,
and checks which model answers and whether the streamed-response warning is emitted.

These are offline tests with `FunctionModel` rather than recorded requests to providers: how a path judges an
attempt doesn't depend on the provider, and scripted models fail or answer on cue.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.fallback import FallbackModel, FallbackOn
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel

Failure = Literal['error', 'error_before_first_event', 'rejected']


@dataclass(frozen=True)
class FallbackPath:
    id: str
    via: Literal['FallbackModel', 'ModelFallback']
    streamed: bool
    pinned: bool = False
    """Whether the request continues a suspended response pinned to the primary model."""


PATHS = [
    FallbackPath('request', 'FallbackModel', streamed=False),
    FallbackPath('request_pinned', 'FallbackModel', streamed=False, pinned=True),
    FallbackPath('stream', 'FallbackModel', streamed=True),
    FallbackPath('stream_pinned', 'FallbackModel', streamed=True, pinned=True),
    FallbackPath('capability', 'ModelFallback', streamed=False),
    FallbackPath('capability_stream', 'ModelFallback', streamed=True),
]


def is_model_api_error(error: Exception) -> bool:
    return isinstance(error, ModelAPIError)


async def is_model_api_error_async(error: Exception) -> bool:
    return isinstance(error, ModelAPIError)


def rejects(response: ModelResponse) -> bool:
    return response.text == 'rejected'


async def rejects_async(response: ModelResponse) -> bool:
    return response.text == 'rejected'


@dataclass(frozen=True)
class Shape:
    id: str
    fallback_on: FallbackOn
    falls_back_on_error: bool
    """Whether `fallback_on` moves on from the primary model's `ModelAPIError`."""
    judges_responses: bool
    """Whether `fallback_on` has a response handler, which rejects the primary model's response."""


SHAPES = [
    Shape('exception_type', ModelAPIError, falls_back_on_error=True, judges_responses=False),
    Shape('tuple_of_types', (ModelAPIError,), falls_back_on_error=True, judges_responses=False),
    Shape('list_of_types', [ModelAPIError], falls_back_on_error=True, judges_responses=False),
    Shape('exception_handler', is_model_api_error, falls_back_on_error=True, judges_responses=False),
    Shape('async_exception_handler', is_model_api_error_async, falls_back_on_error=True, judges_responses=False),
    Shape('response_handler', rejects, falls_back_on_error=False, judges_responses=True),
    Shape('async_response_handler', rejects_async, falls_back_on_error=False, judges_responses=True),
    Shape('tuple_with_handlers', (is_model_api_error, rejects), falls_back_on_error=True, judges_responses=True),
    Shape('mixed_list', [ModelAPIError, rejects_async], falls_back_on_error=True, judges_responses=True),
]

FAILURES: list[Failure] = ['error', 'error_before_first_event', 'rejected']

PINNED_STREAM_WARNING = pytest.mark.xfail(
    strict=True,
    reason=(
        '`FallbackModel.request_stream` warns that response handlers are not applied only after its pinned '
        'continuation block, so a streamed continuation of a suspended response never warns, though the docstring '
        'says every streamed request does and `ModelFallback` warns in the same case.'
    ),
)

CELLS = [
    pytest.param(
        path,
        shape,
        failure,
        id=f'{path.id}-{shape.id}-{failure}',
        marks=[PINNED_STREAM_WARNING]
        if path.id == 'stream_pinned'
        and shape.judges_responses
        and (failure == 'rejected' or not shape.falls_back_on_error)
        else [],
    )
    for path in PATHS
    for shape in SHAPES
    for failure in FAILURES
    if path.streamed or failure != 'error_before_first_event'
]


@pytest.mark.parametrize(('path', 'shape', 'failure'), CELLS)
async def test_fallback_on_is_judged_alike_on_every_path(path: FallbackPath, shape: Shape, failure: Failure) -> None:
    """The primary model fails or answers as `failure` says, and the backup always answers.

    An error is fallen back from when `fallback_on` covers it, and propagates otherwise. A response handler rejects
    a response only on a non-streamed request: a streamed one has already reached the consumer, so it's accepted, with
    a warning. `FallbackModel` warns before its first attempt, `ModelFallback` once it accepts a streamed response.
    """

    def primary_request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if failure == 'rejected':
            return ModelResponse(parts=[TextPart('rejected')])
        raise ModelAPIError('primary', 'down')

    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if failure == 'rejected':
            yield 'rejected'
            return
        if failure == 'error_before_first_event':
            # A chunk that produces no event, like a provider's `response.created`, so the stream opens fine.
            yield {}
        raise ModelAPIError('primary', 'down')

    def backup_request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('backup')])

    async def backup_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'backup'

    primary = FunctionModel(primary_request, stream_function=primary_stream, model_name='primary')
    backup = FunctionModel(backup_request, stream_function=backup_stream, model_name='backup')
    agent: Agent[None, str] = (
        Agent(FallbackModel(primary, backup, fallback_on=shape.fallback_on))
        if path.via == 'FallbackModel'
        else Agent(primary, capabilities=[ModelFallback(backup, fallback_on=shape.fallback_on)])
    )
    prompt = None if path.pinned else 'hi'
    history: list[ModelMessage] | None = (
        [
            ModelRequest(parts=[UserPromptPart('hi')]),
            ModelResponse(
                parts=[TextPart('paused')],
                model_name='primary',
                state='suspended',
                metadata={'__pydantic_ai__': {'fallback_model_id': primary.model_id}},
            ),
        ]
        if path.pinned
        else None
    )

    if failure == 'rejected':
        expected = 'backup' if shape.judges_responses and not path.streamed else 'primary'
    else:
        expected = 'backup' if shape.falls_back_on_error else 'ModelAPIError'
    warns = path.streamed and shape.judges_responses and (path.via == 'FallbackModel' or expected != 'ModelAPIError')
    expected_warnings = (
        [f'`{path.via}` response handlers in `fallback_on` are not applied to streamed requests'] if warns else []
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            if path.streamed:
                async with agent.run_stream(prompt, message_history=history) as stream:
                    await stream.get_output()
                    answered_by = stream.response.model_name
            else:
                answered_by = (await agent.run(prompt, message_history=history)).response.model_name
        except ModelAPIError as error:
            answered_by = type(error).__name__

    assert (answered_by, [str(warning.message).split(',')[0] for warning in caught]) == (expected, expected_warnings)
