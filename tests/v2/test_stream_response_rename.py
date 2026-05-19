"""`stream_response()` singular yield contract.

In v2, `stream_responses()` (plural) is dropped from `AgentStream`,
`StreamedRunResult`, and `StreamedRunResultSync`. The singular
`stream_response()` is the only form: it yields `ModelResponse`
snapshots whose `state` is `'incomplete'` while streaming and
`'complete'` (or `'interrupted'`) on the final yield. Callers
migrating from the old `(ModelResponse, is_last)` tuple read
`is_last` as `response.state != 'incomplete'`.
"""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


# AgentStream ────────────────────────────────────────────────────────────────


async def test_agent_stream_stream_response_yields_model_response_snapshots():
    """`AgentStream.stream_response` yields `ModelResponse` snapshots."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    items: list[ModelResponse] = []
    async with agent.iter('hi') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for r in stream.stream_response(debounce_by=None):
                        items.append(r)

    assert len(items) > 0
    assert all(isinstance(item, ModelResponse) for item in items)
    assert items[-1].state == 'complete'
    assert all(item.state == 'incomplete' for item in items[:-1])


# StreamedRunResult ──────────────────────────────────────────────────────────


async def test_streamed_run_result_stream_response_yields_modelresponse():
    """`StreamedRunResult.stream_response` yields `ModelResponse` (no tuple).

    `response.state` reads `'incomplete'` mid-stream and `'complete'` on the
    trailing yield, matching the semantics of the old `is_last` boolean.
    """
    agent = Agent(TestModel(custom_output_text='hello world'))

    items: list[ModelResponse] = []
    async with agent.run_stream('hi') as result:
        async for item in result.stream_response(debounce_by=None):
            items.append(item)

    assert len(items) > 0
    assert all(isinstance(item, ModelResponse) for item in items)
    assert items[-1].state == 'complete'
    assert all(item.state == 'incomplete' for item in items[:-1])


# StreamedRunResultSync ──────────────────────────────────────────────────────


def test_streamed_run_result_sync_stream_response_yields_modelresponse():
    """`StreamedRunResultSync.stream_response` yields `ModelResponse` (no tuple)."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    result = agent.run_stream_sync('hi')
    items = list(result.stream_response(debounce_by=None))

    assert len(items) > 0
    assert all(isinstance(item, ModelResponse) for item in items)
    assert items[-1].state == 'complete'
    assert all(item.state == 'incomplete' for item in items[:-1])
