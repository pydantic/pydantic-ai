"""Card 01: result-class method-to-property migration.

In 1.x, accessing `result.usage`/`result.timestamp` and `stream.get` as a method
(with parentheses) emits a `DeprecationWarning`. Attribute-style access (no
parentheses) is the new contract and does not warn. The 12 affected sites span
`run.py`, `result.py`, and `direct.py`.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

import pytest

from pydantic_ai import Agent, ModelRequest, UserPromptPart
from pydantic_ai.direct import model_request_stream_sync
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage

pytestmark = pytest.mark.anyio


def _assert_no_deprecation(getter: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        return getter()


# AgentRun ────────────────────────────────────────────────────────────────


async def test_agent_run_usage_property_then_call():
    """`AgentRun.usage` — property access silent, method call warns; both yield a `RunUsage`."""
    agent = Agent(TestModel())
    async with agent.iter('hello') as run:
        async for _ in run:
            pass
        usage_attr = _assert_no_deprecation(lambda: run.usage)
        assert isinstance(usage_attr, RunUsage)

        with pytest.warns(DeprecationWarning, match=r'`AgentRun\.usage` is no longer a method'):
            usage_call = run.usage()
        assert isinstance(usage_call, RunUsage)
        assert usage_attr == usage_call


# AgentRunResult ──────────────────────────────────────────────────────────


async def test_agent_run_result_usage_property_then_call():
    """`AgentRunResult.usage` — property access silent, method call warns; both yield a `RunUsage`."""
    agent = Agent(TestModel())
    result = await agent.run('hello')
    usage_attr = _assert_no_deprecation(lambda: result.usage)
    assert isinstance(usage_attr, RunUsage)

    with pytest.warns(DeprecationWarning, match=r'`AgentRunResult\.usage` is no longer a method'):
        usage_call = result.usage()
    assert isinstance(usage_call, RunUsage)
    assert usage_attr == usage_call
    # Wrapper repr/eq behave like the underlying type.
    assert repr(usage_attr).startswith('RunUsage(')
    assert (usage_attr == 'not-a-run-usage') is False


async def test_agent_run_result_timestamp_property_then_call():
    """`AgentRunResult.timestamp` — property access silent, method call warns; both yield a `datetime`."""
    agent = Agent(TestModel())
    result = await agent.run('hello')
    ts_attr = _assert_no_deprecation(lambda: result.timestamp)
    assert isinstance(ts_attr, datetime)

    with pytest.warns(DeprecationWarning, match=r'`AgentRunResult\.timestamp` is no longer a method'):
        ts_call = result.timestamp()
    assert isinstance(ts_call, datetime)
    assert ts_attr == ts_call

    # Wrappers preserve repr-as-parent so doc snapshots and user logs read naturally.
    assert repr(ts_attr).startswith('datetime.datetime(')


# AgentStream (returned via `agent.iter` while a streaming step is active) ──


async def test_agent_stream_get_response_warns_and_response_property_silent():
    """`AgentStream.get` is deprecated in favor of the `response` property."""
    agent = Agent(TestModel())
    async with agent.iter('hello') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    response_attr = _assert_no_deprecation(lambda: stream.response)
                    assert isinstance(response_attr, ModelResponse)

                    with pytest.warns(DeprecationWarning, match=r'`AgentStream\.get` is deprecated'):
                        response_call = stream.get()
                    assert isinstance(response_call, ModelResponse)


async def test_agent_stream_usage_and_timestamp_property_then_call():
    """`AgentStream.usage` and `AgentStream.timestamp` — same migration shape."""
    agent = Agent(TestModel())
    async with agent.iter('hello') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    usage_attr = _assert_no_deprecation(lambda: stream.usage)
                    assert isinstance(usage_attr, RunUsage)
                    with pytest.warns(DeprecationWarning, match=r'`AgentStream\.usage` is no longer a method'):
                        stream.usage()

                    ts_attr = _assert_no_deprecation(lambda: stream.timestamp)
                    assert isinstance(ts_attr, datetime)
                    with pytest.warns(DeprecationWarning, match=r'`AgentStream\.timestamp` is no longer a method'):
                        stream.timestamp()


# StreamedRunResult ───────────────────────────────────────────────────────


async def test_streamed_run_result_usage_and_timestamp_property_then_call():
    """`StreamedRunResult.usage` and `.timestamp` — property access silent, method call warns."""
    agent = Agent(TestModel())
    async with agent.run_stream('hello') as result:
        async for _ in result.stream_output():
            pass
        usage_attr = _assert_no_deprecation(lambda: result.usage)
        assert isinstance(usage_attr, RunUsage)
        with pytest.warns(DeprecationWarning, match=r'`StreamedRunResult\.usage` is no longer a method'):
            result.usage()

        ts_attr = _assert_no_deprecation(lambda: result.timestamp)
        assert isinstance(ts_attr, datetime)
        with pytest.warns(DeprecationWarning, match=r'`StreamedRunResult\.timestamp` is no longer a method'):
            result.timestamp()


# StreamedRunResultSync ───────────────────────────────────────────────────


def test_streamed_run_result_sync_usage_and_timestamp_property_then_call():
    """`StreamedRunResultSync.usage` and `.timestamp` — property access silent, method call warns."""
    agent = Agent(TestModel())
    result = agent.run_stream_sync('hello')
    for _ in result.stream_output():
        pass
    usage_attr = _assert_no_deprecation(lambda: result.usage)
    assert isinstance(usage_attr, RunUsage)
    with pytest.warns(DeprecationWarning, match=r'`StreamedRunResultSync\.usage` is no longer a method'):
        result.usage()

    ts_attr = _assert_no_deprecation(lambda: result.timestamp)
    assert isinstance(ts_attr, datetime)
    with pytest.warns(DeprecationWarning, match=r'`StreamedRunResultSync\.timestamp` is no longer a method'):
        result.timestamp()


# StreamedResponseSync (direct.py) ────────────────────────────────────────


def test_streamed_response_sync_get_and_usage_property_then_call():
    """`StreamedResponseSync.get` (deprecated in favor of `response`) and `.usage` (now a property)."""
    messages = [ModelRequest(parts=[UserPromptPart(content='hello')])]
    with model_request_stream_sync(TestModel(), messages) as stream:
        for _ in stream:
            pass
        response_attr = _assert_no_deprecation(lambda: stream.response)
        assert isinstance(response_attr, ModelResponse)
        with pytest.warns(DeprecationWarning, match=r'`StreamedResponseSync\.get` is deprecated'):
            response_call = stream.get()
        # `__eq__` and `__repr__` make the wrapper indistinguishable from the underlying type.
        assert response_call == response_attr
        assert repr(response_call).startswith('ModelResponse(')

        usage_attr = _assert_no_deprecation(lambda: stream.usage)
        assert isinstance(usage_attr, RequestUsage)
        with pytest.warns(DeprecationWarning, match=r'`StreamedResponseSync\.usage` is no longer a method'):
            usage_call = stream.usage()
        assert usage_call == usage_attr
        assert repr(usage_call).startswith('RequestUsage(')
        assert (usage_call == 'not-a-request-usage') is False
        assert (response_call == 'not-a-model-response') is False
