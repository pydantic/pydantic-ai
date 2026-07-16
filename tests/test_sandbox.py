"""Tests for the sandbox concept: the `Sandbox` protocol and the read-only `RunContext.sandbox`
field, populated from the `sandbox=` run argument or a capability's `get_sandbox` contribution."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.sandbox import Sandbox
from pydantic_ai.toolsets import FunctionToolset, WrapperToolset
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class _Result:
    exit_code: int
    stdout: str
    stderr: str
    stdout_dropped: int = 0
    stderr_dropped: int = 0


@dataclass(frozen=True)
class _Entry:
    name: str
    path: str
    is_dir: bool
    size: int | None = None


class _Fs:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def read_bytes(self, path: str) -> bytes:
        return self.files[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        return self.files[path].decode(encoding)

    async def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        self.files[path] = content.encode(encoding)

    async def stat(self, path: str) -> _Entry:
        return _Entry(name=path.rsplit('/', 1)[-1], path=path, is_dir=False, size=len(self.files[path]))

    async def list_dir(self, path: str) -> Sequence[_Entry]:
        return [await self.stat(p) for p in self.files]

    async def make_dir(self, path: str) -> None:
        pass

    async def remove(self, path: str) -> None:
        self.files.pop(path, None)

    async def exists(self, path: str) -> bool:
        return path in self.files


class FakeSandbox:
    """A minimal in-memory implementation of the `Sandbox` protocol."""

    provider = 'fake'

    def __init__(self, name: str) -> None:
        self.name = name
        self._fs = _Fs()

    @property
    def sandbox_id(self) -> str:
        return f'fake-{self.name}'

    @property
    def fs(self) -> _Fs:
        return self._fs

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> _Result:
        return _Result(exit_code=0, stdout=f'ran:{command}', stderr='')

    async def start(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> Any:
        raise NotImplementedError('FakeSandbox cannot start background processes; use `run` instead.')

    async def working_dir(self) -> str:
        return '/workspace'

    async def resolve(self, path: str, *, base: str | None = None) -> str:
        if path.startswith('/'):
            return path
        return f'{base or "/workspace"}/{path}'


def _describe(sandbox: Sandbox | None) -> str:
    if sandbox is None:
        return 'none'
    return getattr(sandbox, 'name', sandbox.sandbox_id)


def _tool_call_then_text(tool_name: str = 'probe') -> FunctionModel:
    """A model that calls `tool_name` on the first step and returns text on the second."""

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, {})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model_func)


def make_probe_agent(seen: list[str], **kwargs: Any) -> Agent:
    agent: Agent = Agent(_tool_call_then_text(), **kwargs)

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(_describe(ctx.sandbox))
        return 'ok'

    return agent


def test_fake_sandbox_conforms_to_protocol():
    sandbox = FakeSandbox('x')
    assert isinstance(sandbox, Sandbox)
    # Static conformance too: pyright checks this assignment because tests are type-checked.
    typed: Sandbox = sandbox
    assert typed.provider == 'fake'


async def test_fake_sandbox_protocol_surface():
    """Exercise the in-memory protocol implementation used by the run tests."""
    sandbox = FakeSandbox('surface')
    await sandbox.fs.write_bytes('/workspace/data.bin', b'123')
    assert await sandbox.fs.read_bytes('/workspace/data.bin') == b'123'
    await sandbox.fs.write_text('/workspace/notes.txt', 'hello')
    assert await sandbox.fs.read_text('/workspace/notes.txt') == 'hello'
    assert await sandbox.fs.stat('/workspace/notes.txt') == _Entry(
        name='notes.txt', path='/workspace/notes.txt', is_dir=False, size=5
    )
    assert {entry.name for entry in await sandbox.fs.list_dir('/workspace')} == {'data.bin', 'notes.txt'}
    await sandbox.fs.make_dir('/workspace/subdir')
    assert await sandbox.fs.exists('/workspace/notes.txt')
    await sandbox.fs.remove('/workspace/notes.txt')
    assert not await sandbox.fs.exists('/workspace/notes.txt')

    assert await sandbox.working_dir() == '/workspace'
    assert await sandbox.resolve('data.bin') == '/workspace/data.bin'
    assert await sandbox.resolve('data.bin', base='/tmp') == '/tmp/data.bin'
    assert await sandbox.resolve('/absolute') == '/absolute'
    assert (await sandbox.run(['echo', 'ok'])).stdout == "ran:['echo', 'ok']"
    with pytest.raises(NotImplementedError, match='cannot start background processes'):
        await sandbox.start(['echo', 'ok'])


def test_bare_run_context_sandbox_defaults_to_none():
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    assert ctx.sandbox is None


async def test_run_argument_sandbox_reaches_tools():
    seen: list[str] = []
    agent = make_probe_agent(seen)
    sandbox = FakeSandbox('direct')
    result = await agent.run('go', sandbox=sandbox)
    assert result.output == 'done'
    assert seen == ['direct']


async def test_run_without_sandbox_sees_none():
    seen: list[str] = []
    agent = make_probe_agent(seen)
    await agent.run('go')
    assert seen == ['none']


async def test_wrapper_agent_forwards_sandbox():
    seen: list[str] = []
    agent = make_probe_agent(seen)
    await WrapperAgent(agent).run('go', sandbox=FakeSandbox('wrapped'))
    assert seen == ['wrapped']


async def test_run_argument_sandbox_available_in_all_hooks():
    """A run-argument sandbox is available from `for_run` through `after_run`."""
    log: list[str] = []

    @dataclass
    class Watcher(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            log.append(f'for_run:{_describe(ctx.sandbox)}')
            return self

        async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
            log.append(f'wrap_enter:{_describe(ctx.sandbox)}')
            return await handler()

        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            log.append(f'after_run:{_describe(ctx.sandbox)}')
            return result

    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[Watcher()])
    await agent.run('go', sandbox=FakeSandbox('direct'))
    assert log == ['for_run:direct', 'wrap_enter:direct', 'after_run:direct']


async def test_sandbox_identity_stable_across_steps():
    """Two tool calls in different run steps observe the same sandbox object."""
    observed: list[Any] = []

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) in (1, 3):
            return ModelResponse(parts=[ToolCallPart('probe', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent = Agent(FunctionModel(model_func))

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox)
        return 'ok'

    sandbox = FakeSandbox('stable')
    await agent.run('go', sandbox=sandbox)
    assert len(observed) == 2
    assert observed[0] is sandbox
    assert observed[1] is sandbox


async def test_sandbox_available_during_streamed_run():
    seen: list[str] = []
    agent: Agent = Agent(TestModel())  # TestModel calls every registered tool, then streams output

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(_describe(ctx.sandbox))
        return 'ok'

    async with agent.run_stream('go', sandbox=FakeSandbox('streamed')) as stream:
        async for _chunk in stream.stream_text():
            pass
    assert seen == ['streamed']


@dataclass
class SandboxCapability(AbstractCapability[Any]):
    """Canonical contributor: a fresh sandbox per run, entered and exited by the run itself."""

    name: str = 'cap'
    events: list[str] = field(default_factory=lambda: [])

    def get_sandbox(self, ctx: RunContext[Any]) -> AbstractAsyncContextManager[Sandbox] | None:
        self.events.append(f'{self.name}:offered')

        @asynccontextmanager
        async def per_run_sandbox() -> AsyncGenerator[Sandbox]:
            self.events.append(f'{self.name}:enter')
            try:
                yield FakeSandbox(self.name)
            finally:
                self.events.append(f'{self.name}:exit')

        return per_run_sandbox()


async def test_capability_sandbox_reaches_tools_and_is_bracketed_by_the_run():
    cap = SandboxCapability()
    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[cap])
    result = await agent.run('go')
    assert result.output == 'done'
    assert seen == ['cap']
    assert cap.events == ['cap:offered', 'cap:enter', 'cap:exit']


async def test_capability_sandbox_live_through_after_run():
    """The run exits the sandbox after `after_run`, so hooks never see a dead handle."""

    @dataclass
    class ContributingWatcher(SandboxCapability):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            self.events.append(f'after_run:{_describe(ctx.sandbox)}')
            return result

    cap = ContributingWatcher()
    await make_probe_agent([], capabilities=[cap]).run('go')
    assert cap.events == ['cap:offered', 'cap:enter', 'after_run:cap', 'cap:exit']


async def test_run_argument_wins_over_capability():
    cap = SandboxCapability(name='loser')
    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[cap])
    await agent.run('go', sandbox=FakeSandbox('direct'))
    assert seen == ['direct']
    assert cap.events == []


async def test_last_capability_in_chain_wins_and_losers_are_never_consulted():
    first = SandboxCapability(name='first')
    last = SandboxCapability(name='last')
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[first, last]).run('go')
    assert seen == ['last']
    assert first.events == []


async def test_capability_returning_none_falls_back_to_earlier_capability():
    @dataclass
    class Abstaining(SandboxCapability):
        def get_sandbox(self, ctx: RunContext[Any]) -> AbstractAsyncContextManager[Sandbox] | None:
            self.events.append(f'{self.name}:offered-none')
            return None

    contributor = SandboxCapability(name='contributor')
    abstainer = Abstaining(name='abstainer')
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[contributor, abstainer]).run('go')
    assert seen == ['contributor']
    assert abstainer.events == ['abstainer:offered-none']


async def test_warm_sandbox_shared_across_runs():
    warm = FakeSandbox('warm')

    @dataclass
    class WarmSandboxCapability(AbstractCapability[Any]):
        def get_sandbox(self, ctx: RunContext[Any]) -> AbstractAsyncContextManager[Sandbox] | None:
            return nullcontext(warm)

    observed: list[Any] = []
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[WarmSandboxCapability()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox)
        return 'ok'

    await agent.run('one')
    await agent.run('two')
    assert observed == [warm, warm]


async def test_deferred_capability_never_contributes():
    cap = SandboxCapability(name='deferred')
    cap.id = 'deferred-sandbox'
    cap.defer_loading = True
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[cap]).run('go')
    assert seen == ['none']
    assert cap.events == []


async def test_wrapper_capability_forwards_get_sandbox():
    inner = SandboxCapability(name='inner')
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[WrapperCapability(wrapped=inner)]).run('go')
    assert seen == ['inner']
    assert inner.events == ['inner:offered', 'inner:enter', 'inner:exit']


async def test_capability_sandbox_exits_when_a_tool_raises():
    cap = SandboxCapability()

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('explode', {})])

    agent: Agent = Agent(FunctionModel(model_func), capabilities=[cap])

    @agent.tool
    async def explode(ctx: RunContext[Any]) -> str:
        assert ctx.sandbox is not None
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('go')
    assert cap.events == ['cap:offered', 'cap:enter', 'cap:exit']


async def test_capability_sandbox_exits_when_toolset_entry_fails():
    """The exit stack owns the bracket, so teardown runs even when the run never starts."""

    class ExplodingToolset(WrapperToolset[Any]):
        async def __aenter__(self) -> Any:
            raise RuntimeError('toolset entry failed')

    cap = SandboxCapability()
    agent: Agent = Agent(TestModel(), toolsets=[ExplodingToolset(wrapped=FunctionToolset())], capabilities=[cap])
    with pytest.raises(RuntimeError, match='toolset entry failed'):
        await agent.run('go')
    assert cap.events == ['cap:offered', 'cap:enter', 'cap:exit']
