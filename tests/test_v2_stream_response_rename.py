"""Card 32: `stream_responses()` -> `stream_response()`.

In 1.x, calling `stream_responses()` (plural) on `AgentStream`,
`StreamedRunResult`, or `StreamedRunResultSync` emits a `DeprecationWarning`.
The new singular `stream_response()` is the contract going forward; both names
yield identical items in 1.x. The semantic shift the v2 card describes (yielded
type changing from `tuple[ModelResponse, bool]` to `ModelResponse` on the result
classes) is deferred to the v2 cut.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


def _assert_no_deprecation(getter: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        return getter()


# AgentStream ────────────────────────────────────────────────────────────────


async def test_agent_stream_stream_response_singular_silent_and_plural_warns():
    """`AgentStream.stream_response` is silent; `stream_responses` warns; both yield the same items."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    singular: list[Any] = []
    plural: list[Any] = []
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
                        DeprecationWarning,
                        match=r'`AgentStream\.stream_responses\(\)` is deprecated',
                    ):
                        async for r in stream.stream_responses(debounce_by=None):  # pyright: ignore[reportDeprecated]
                            plural.append(r)

    assert [r.parts for r in singular] == [r.parts for r in plural]
    assert len(singular) > 0


# StreamedRunResult ──────────────────────────────────────────────────────────


async def test_streamed_run_result_stream_response_singular_silent_and_plural_warns():
    """`StreamedRunResult.stream_response` is silent; `stream_responses` warns; equivalent items."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    async with agent.run_stream('hi') as result:
        singular: list[tuple[Any, bool]] = []

        async def collect():
            async for item in result.stream_response(debounce_by=None):
                singular.append(item)

        await _assert_no_deprecation(lambda: collect())

    async with agent.run_stream('hi') as result:
        plural: list[tuple[Any, bool]] = []
        with pytest.warns(
            DeprecationWarning,
            match=r'`StreamedRunResult\.stream_responses\(\)` is deprecated',
        ):
            async for item in result.stream_responses(debounce_by=None):  # pyright: ignore[reportDeprecated]
                plural.append(item)

    assert [(msg.parts, last) for msg, last in singular] == [(msg.parts, last) for msg, last in plural]
    assert len(singular) > 0
    assert singular[-1][1] is True


# StreamedRunResultSync ──────────────────────────────────────────────────────


def test_streamed_run_result_sync_stream_response_singular_silent_and_plural_warns():
    """`StreamedRunResultSync.stream_response` is silent; `stream_responses` warns; equivalent items."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    result = agent.run_stream_sync('hi')
    singular = list(_assert_no_deprecation(lambda: list(result.stream_response(debounce_by=None))))

    result = agent.run_stream_sync('hi')
    with pytest.warns(
        DeprecationWarning,
        match=r'`StreamedRunResultSync\.stream_responses\(\)` is deprecated',
    ):
        plural = list(result.stream_responses(debounce_by=None))  # pyright: ignore[reportDeprecated]

    assert [(msg.parts, last) for msg, last in singular] == [(msg.parts, last) for msg, last in plural]
    assert len(singular) > 0
    assert singular[-1][1] is True
