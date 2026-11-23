"""Tests for the [`Browser`][pydantic_ai.capabilities.Browser] capability."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import Browser
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = pytest.mark.anyio


@dataclass
class FakeSession:
    """Stands in for `CDPSession`: records commands, answers `Runtime.evaluate`, fires queued events."""

    sent: list[tuple[str, dict[str, Any]]] = field(default_factory=list[tuple[str, dict[str, Any]]])
    events: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    _waiters: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict[str, asyncio.Future[dict[str, Any]]]
    )

    async def send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.sent.append((method, params or {}))
        if (future := self._waiters.pop(method.replace('navigate', 'loadEventFired'), None)) and method in self.events:
            future.set_result(self.events[method])
        return {'result': {'type': 'string', 'value': 'Example Domain'}} if method == 'Runtime.evaluate' else {}

    def wait_for_event(self, method: str) -> asyncio.Future[dict[str, Any]]:
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiters[method] = future
        return future


@pytest.fixture
def fake_chrome(monkeypatch: pytest.MonkeyPatch) -> list[FakeSession]:
    """Replace Chrome with `FakeSession`s, one per launch."""
    launched: list[FakeSession] = []

    @asynccontextmanager
    async def launch_chrome(*, headless: bool = True, chrome_path: str | None = None) -> AsyncGenerator[FakeSession]:
        launched.append(session := FakeSession(events={'Page.navigate': {'timestamp': 1.0}}))
        yield session

    monkeypatch.setattr('pydantic_ai._cdp.launch_chrome', launch_chrome)
    return launched


def calls_then_reply(*calls: dict[str, Any]) -> FunctionModel:
    """A model that makes each `browser_cdp` call in turn, then replies with the last tool result."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = len(tool_returns(messages))
        if step < len(calls):
            return ModelResponse(parts=[ToolCallPart('browser_cdp', calls[step])])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def tool_returns(messages: list[ModelMessage]) -> list[Any]:
    return [p.content for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]


async def test_tool_sends_commands_to_a_browser_per_run(fake_chrome: list[FakeSession]):
    agent = Agent(
        calls_then_reply(
            {
                'method': 'Page.navigate',
                'params': {'url': 'https://example.com'},
                'wait_for_event': 'Page.loadEventFired',
            },
            {'method': 'Runtime.evaluate', 'params': {'expression': 'document.title', 'returnByValue': True}},
        ),
        capabilities=[Browser()],
    )
    result = await agent.run('title?')
    assert tool_returns(result.all_messages()) == snapshot(
        [
            {'result': {}, 'event': {'timestamp': 1.0}},
            {'result': {'type': 'string', 'value': 'Example Domain'}},
        ]
    )
    await agent.run('again')
    assert len(fake_chrome) == 2  # one browser per run
    assert [m for m, _ in fake_chrome[0].sent] == snapshot(['Page.navigate', 'Runtime.evaluate'])


async def test_disallowed_commands_and_urls_are_refused(fake_chrome: list[FakeSession]):
    agent = Agent(
        calls_then_reply(
            {'method': 'Browser.close'},
            {'method': 'Page.navigate', 'params': {'url': 'file:///etc/passwd'}},
        ),
        capabilities=[Browser()],
    )
    result = await agent.run('go')
    refusals = tool_returns(result.all_messages())
    assert refusals[0].startswith('Browser.close is not allowed, allowed commands: DOM.describeNode, ')
    assert refusals[1] == "Only http and https URLs can be opened, got 'file:///etc/passwd'"
    assert fake_chrome[0].sent == []


async def test_allowed_methods_can_be_narrowed(fake_chrome: list[FakeSession]):
    agent = Agent(
        calls_then_reply({'method': 'Input.insertText', 'params': {'text': 'hi'}}),
        capabilities=[Browser(allowed_methods=['Runtime.evaluate'])],
    )
    result = await agent.run('go')
    assert tool_returns(result.all_messages()) == [
        'Input.insertText is not allowed, allowed commands: Runtime.evaluate'
    ]


async def test_instructions_describe_the_tool(fake_chrome: list[FakeSession]):
    seen: list[str | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions)
        assert [t.name for t in info.function_tools] == ['browser_cdp']
        return ModelResponse(parts=[TextPart('ok')])

    await Agent(FunctionModel(respond), capabilities=[Browser()]).run('hi')
    assert seen[0] is not None and 'Chrome DevTools Protocol' in seen[0]


@pytest.mark.skipif(not (shutil.which('google-chrome') or shutil.which('chromium')), reason='needs Chrome')
async def test_real_chrome():  # pragma: no cover
    from pydantic_ai._cdp import launch_chrome

    async with launch_chrome() as session:
        result = await session.send('Runtime.evaluate', {'expression': '1 + 1', 'returnByValue': True})
    assert result['result']['value'] == 2
