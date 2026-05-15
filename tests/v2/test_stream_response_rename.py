"""`stream_responses()` -> `stream_response()` rename.

In 1.x, calling `stream_responses()` (plural) on `AgentStream`,
`StreamedRunResult`, or `StreamedRunResultSync` emits a
`PydanticAIDeprecationWarning`.

`AgentStream.stream_response()` is a pure rename: both names yield
`ModelResponse` items.

`StreamedRunResult.stream_response()` and `StreamedRunResultSync.stream_response()`
ship the new yield shape directly in 1.x: the singular yields `ModelResponse`,
while the deprecated plural keeps yielding `(ModelResponse, is_last: bool)` for
backwards compatibility. Callers migrating to the singular read `is_last` as
`response.state != 'incomplete'`.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


def _assert_no_deprecation(getter: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        return getter()


# AgentStream ────────────────────────────────────────────────────────────────


async def test_agent_stream_stream_response_singular_silent_and_plural_warns():
    """`AgentStream.stream_response` is silent; `stream_responses` warns; both yield the same items."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    singular: list[ModelResponse] = []
    plural: list[ModelResponse] = []
    async with agent.iter('hi') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:

                    async def collect_singular():
                        async for r in stream.stream_response(debounce_by=None):
                            singular.append(r)

                    await _assert_no_deprecation(lambda: collect_singular())

    async with agent.iter('hi') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    with pytest.warns(
                        PydanticAIDeprecationWarning,
                        match=r'`AgentStream\.stream_responses\(\)` is deprecated',
                    ):
                        async for r in stream.stream_responses(debounce_by=None):  # pyright: ignore[reportDeprecated]
                            plural.append(r)

    assert [r.parts for r in singular] == [r.parts for r in plural]
    assert len(singular) > 0


# StreamedRunResult ──────────────────────────────────────────────────────────


async def test_streamed_run_result_stream_response_singular_yields_modelresponse():
    """`StreamedRunResult.stream_response` is silent and yields `ModelResponse` (no tuple).

    Asserts the new singular yield shape (the v2 contract, landed in 1.x because the
    singular is a brand-new method — additive, non-breaking). `response.state` reads
    `'incomplete'` mid-stream and `'complete'` on the trailing yield, matching the
    semantics of the old `is_last` boolean.
    """
    agent = Agent(TestModel(custom_output_text='hello world'))

    items: list[ModelResponse] = []

    async def collect():
        async with agent.run_stream('hi') as result:
            async for item in result.stream_response(debounce_by=None):
                items.append(item)

    await _assert_no_deprecation(lambda: collect())

    assert len(items) > 0
    assert all(isinstance(item, ModelResponse) for item in items)
    assert items[-1].state == 'complete'
    assert all(item.state == 'incomplete' for item in items[:-1])


async def test_streamed_run_result_stream_responses_plural_yields_tuple_and_warns():
    """`StreamedRunResult.stream_responses` keeps the legacy `(ModelResponse, is_last)` tuple and warns."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    async with agent.run_stream('hi') as result:
        plural: list[tuple[ModelResponse, bool]] = []
        with pytest.warns(
            PydanticAIDeprecationWarning,
            match=r'`StreamedRunResult\.stream_responses\(\)` is deprecated',
        ):
            async for item in result.stream_responses(debounce_by=None):  # pyright: ignore[reportDeprecated]
                plural.append(item)

    assert len(plural) > 0
    assert plural[-1][1] is True
    assert all(is_last is False for _msg, is_last in plural[:-1])


# StreamedRunResultSync ──────────────────────────────────────────────────────


def test_streamed_run_result_sync_stream_response_singular_yields_modelresponse():
    """`StreamedRunResultSync.stream_response` is silent and yields `ModelResponse` (no tuple)."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    result = agent.run_stream_sync('hi')
    items = list(_assert_no_deprecation(lambda: list(result.stream_response(debounce_by=None))))

    assert len(items) > 0
    assert all(isinstance(item, ModelResponse) for item in items)
    assert items[-1].state == 'complete'
    assert all(item.state == 'incomplete' for item in items[:-1])


def test_streamed_run_result_sync_stream_responses_plural_yields_tuple_and_warns():
    """`StreamedRunResultSync.stream_responses` keeps the legacy `(ModelResponse, is_last)` tuple and warns."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    result = agent.run_stream_sync('hi')
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`StreamedRunResultSync\.stream_responses\(\)` is deprecated',
    ):
        plural = list(result.stream_responses(debounce_by=None))  # pyright: ignore[reportDeprecated]

    assert len(plural) > 0
    assert plural[-1][1] is True
    assert all(is_last is False for _msg, is_last in plural[:-1])
