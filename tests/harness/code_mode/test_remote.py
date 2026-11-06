"""Tests for CodeMode's remote transport and sandbox error routing.

Refusal tests exercise native errors from a loopback gRPC server. Local-backed factory pool
doubles cover factory plumbing and deterministic recovery branches, not remote execution.
Real subprocess bridge tests remain skipped until #8824 is resolved.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Never
from unittest.mock import AsyncMock, MagicMock

import pytest
from logfire.exceptions import LogfireConfigError
from pydantic_monty import (
    AsyncFutureSnapshot,
    AsyncMonty,
    AsyncMontyRemote,
    AsyncMontySession,
    AsyncSnapshot,
    ExternalSettledResult,
    MontyRefusedError,
    MountDir,
)

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai_harness import CodeMode, HarnessDeprecationWarning
from pydantic_ai_harness.code_mode import CodeModeToolset
from tests.harness.code_mode.grpc_relay import MontyRelay, serve_monty_relay
from tests.harness.code_mode.repl_store import MemoryReplStore


def _parts(messages: list[ModelMessage], part_type: type[Any]) -> list[Any]:
    return [part for message in messages for part in message.parts if isinstance(part, part_type)]


def _retries(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for part in _parts(messages, ToolReturnPart) if part.outcome == 'retried']


def _snippets_model(*snippets: str) -> FunctionModel:
    """A model that calls `run_code` with each snippet in turn, then says 'done'."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        done = len(_parts(messages, ToolReturnPart))
        if done < len(snippets):
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': snippets[done]})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model)


@pytest.mark.skip(reason='Hangs intermittently in CI: https://github.com/pydantic/pydantic-ai/issues/8824')
async def test_code_mode_runs_remotely(monty_relay: MontyRelay, tmp_path: Path) -> None:
    """Remote feeds keep REPL state while tools, prints, mounts, `gather`, and barriers stay host-side."""
    (tmp_path / 'input.txt').write_text('mounted data')
    agent = Agent(
        _snippets_model(
            'value = await add(a=2, b=3)\nprint("remote tool result", value)',
            'from pathlib import Path\nprint(Path("/workspace/input.txt").read_text())\nvalue * 10',
            'import asyncio\nfirst, second = await asyncio.gather(add(a=1, b=1), add(a=2, b=2))\n[first, second, barrier()]',
        ),
        capabilities=[
            CodeMode(
                monty_sandbox=monty_relay.url,
                mount=MountDir(virtual_path='/workspace', host_path=tmp_path),
            )
        ],
    )

    @agent.tool_plain
    async def add(a: int, b: int) -> int:
        return a + b

    @agent.tool_plain(sequential=True)
    def barrier() -> str:
        return 'barrier'

    result = await agent.run('exercise the remote sandbox')

    assert result.output == 'done'
    assert [part.content for part in _parts(result.all_messages(), ToolReturnPart)] == [
        {'output': 'remote tool result 5\n'},
        {'output': 'mounted data\n', 'result': 50},
        [2, 4, 'barrier'],
    ]


@pytest.mark.parametrize(
    'url',
    [
        'https://sandbox.example.com',
        'http://monty-server:8000/',
        # Monty reads its earlier WebSocket schemes as `http://` and `https://`.
        'ws://monty-server:8000',
        'wss://sandbox.example.com/',
    ],
)
async def test_remote_sandbox_url_accepted(url: str) -> None:
    """A server root URL passes validation; workers dial lazily, not at enter."""
    agent = Agent(_snippets_model(), capabilities=[CodeMode(monty_sandbox=url)])
    result = await agent.run('no run_code call, so nothing dials')
    assert result.output == 'done'


async def test_unsupported_scheme_rejected() -> None:
    """A URL Monty's remote client cannot dial fails when the run starts, not at every `run_code`."""
    agent = Agent(_snippets_model(), capabilities=[CodeMode(monty_sandbox='sk-live-abc123:x@sandbox.example.com')])
    with pytest.raises(UserError, match='must be an `http://`') as exc_info:
        await agent.run('never dials')
    assert 'sk-live-abc123' not in str(exc_info.value)


@pytest.mark.parametrize(
    'url',
    [
        'https://sandbox.example.com/monty',
        'wss://sandbox.example.com/monty-ws/',
        'https://sandbox.example.com/?token=hunter2',
        'https://sandbox.example.com?',
        'https://sandbox.example.com#hunter2',
    ],
)
async def test_url_beyond_server_root_rejected(url: str) -> None:
    """The client calls one fixed method at the server root, so a path, query, or fragment fails up front."""
    agent = Agent(_snippets_model(), capabilities=[CodeMode(monty_sandbox=url)])
    with pytest.raises(UserError, match='must be the server root') as exc_info:
        await agent.run('never dials')
    assert url not in str(exc_info.value)
    assert 'hunter2' not in str(exc_info.value)


@pytest.mark.parametrize(
    ('resource_limits', 'request_timeout'),
    [(None, 40.0), ({'max_duration_secs': 90}, 100.0), ('unlimited', None)],
)
async def test_remote_turn_deadline_follows_max_duration(
    monkeypatch: pytest.MonkeyPatch, resource_limits: Any, request_timeout: float | None
) -> None:
    """The transport's per-turn deadline leaves room for `max_duration_secs` to fire first."""
    seen: list[float | None] = []

    def recording_pool(url: str, *, request_timeout: float | None) -> Never:
        seen.append(request_timeout)
        raise RuntimeError('not dialing')

    monkeypatch.setattr('pydantic_ai_harness._monty_exec.AsyncMontyRemote', recording_pool)
    agent = Agent(
        _snippets_model('1'),
        capabilities=[CodeMode(monty_sandbox='https://sandbox.example.com', resource_limits=resource_limits)],
    )
    with pytest.raises(RuntimeError, match='not dialing'):
        await agent.run('record the deadline')
    assert seen == [request_timeout]


async def test_dial_failure_reaches_caller() -> None:
    """Dial errors reach the caller unchanged, without adding a credential-bearing model retry."""
    agent = Agent(_snippets_model('1 + 1'), capabilities=[CodeMode(monty_sandbox='http://alice:hunter2@127.0.0.1:1')])

    with capture_run_messages() as messages, pytest.raises(RuntimeError):
        await agent.run('dial a dead worker')
    assert _retries(messages) == []
    assert 'hunter2' not in str(messages)


def test_repr_hides_sandbox_url() -> None:
    """The URL may carry credentials, so neither the capability nor its toolset shows it in `repr`."""
    capability = CodeMode[None](monty_sandbox='https://alice:s3cr3t@sandbox.example.com')
    toolset = capability.get_wrapper_toolset(FunctionToolset[None]())

    assert 'sandbox.example.com' not in repr(capability)
    assert 'sandbox.example.com' not in repr(toolset)


async def test_monty_sandbox_builds_each_run_pool() -> None:
    """`monty_sandbox` is called once per run for a fresh pool, which dials with its own metadata."""
    pools: list[AsyncMontyRemote] = []
    async with serve_monty_relay(refuse=True) as relay:

        def build_pool() -> AsyncMontyRemote:
            pools.append(AsyncMontyRemote(relay.url, metadata=lambda: {'authorization': 'Bearer k-pool'}))
            return pools[-1]

        agent = Agent(_snippets_model('1 + 1'), capabilities=[CodeMode(monty_sandbox=build_pool)])
        for _ in range(2):
            with capture_run_messages() as messages, pytest.raises(MontyRefusedError, match='UNAUTHENTICATED'):
                await agent.run('dial')
            assert _retries(messages) == []

    assert len(pools) == 2 and pools[0] is not pools[1]
    assert [metadata['authorization'] for metadata in relay.metadata] == ['Bearer k-pool', 'Bearer k-pool']


async def test_async_monty_sandbox_is_awaited() -> None:
    """An `async def` `monty_sandbox` is awaited for its pool."""
    async with serve_monty_relay(refuse=True) as relay:

        async def build_pool() -> AsyncMontyRemote:
            return AsyncMontyRemote(relay.url, metadata=lambda: {'authorization': 'Bearer k-async'})

        agent = Agent(_snippets_model('1 + 1'), capabilities=[CodeMode(monty_sandbox=build_pool)])
        with pytest.raises(MontyRefusedError, match='UNAUTHENTICATED'):
            await agent.run('dial')

    assert [metadata['authorization'] for metadata in relay.metadata] == ['Bearer k-async']


@pytest.mark.parametrize('error_type', [RuntimeError, UserError, LogfireConfigError])
@pytest.mark.parametrize('async_factory', [False, True])
async def test_failing_monty_sandbox_reaches_caller(error_type: type[Exception], async_factory: bool) -> None:
    """Factory errors reach the caller on the first attempt, not the model, even with credentials in the message."""
    calls: list[None] = []
    error = error_type('cannot connect to https://alice:factory-secret@sandbox.example.com')

    def build_pool() -> AsyncMontyRemote:
        calls.append(None)
        raise error

    async def build_pool_async() -> AsyncMontyRemote:
        return build_pool()

    agent = Agent(
        _snippets_model('1 + 1', '2 + 2'),
        capabilities=[CodeMode(monty_sandbox=build_pool_async if async_factory else build_pool)],
    )
    with capture_run_messages() as messages, pytest.raises(error_type) as exc_info:
        await agent.run('dial')

    assert exc_info.value is error
    assert _retries(messages) == []
    assert 'factory-secret' not in str(messages)
    assert len(calls) == 1


async def test_pool_instead_of_function_rejected() -> None:
    """The pool itself, instead of a function that returns it, fails as the run starts."""
    agent = Agent(
        _snippets_model('1 + 1'),
        capabilities=[CodeMode(monty_sandbox=AsyncMontyRemote('https://sandbox.example.com'))],  # pyright: ignore[reportArgumentType]
    )
    with pytest.raises(UserError, match='not the pool itself'):
        await agent.run('dial')


def test_deprecated_sandbox_url_becomes_monty_sandbox() -> None:
    """`monty_sandbox_url` still works, warns once at the caller's line, and moves into `monty_sandbox`."""
    with pytest.warns(HarnessDeprecationWarning, match='monty_sandbox_url') as warned:
        capability = CodeMode(monty_sandbox_url='https://sandbox.example.com')
    assert warned[0].filename == __file__
    assert capability.monty_sandbox == 'https://sandbox.example.com'
    assert capability.monty_sandbox_url is None
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        replace(capability)

    with pytest.warns(HarnessDeprecationWarning, match='monty_sandbox_url') as warned:
        toolset = CodeModeToolset(FunctionToolset(), monty_sandbox_url='https://sandbox.example.com')
    assert warned[0].filename == __file__
    assert toolset.monty_sandbox == 'https://sandbox.example.com'

    with pytest.raises(UserError, match='`monty_sandbox_url` is its deprecated form'):
        CodeMode(monty_sandbox='https://a.example.com', monty_sandbox_url='https://b.example.com')


@pytest.mark.skip(reason='Hangs intermittently in CI: https://github.com/pydantic/pydantic-ai/issues/8824')
async def test_disconnect_mid_snippet_reports_started_calls(monty_relay: MontyRelay) -> None:
    """A dropped worker connection resets the session and lists the calls that already started."""
    agent = Agent(
        _snippets_model('await drop_connection()'),
        capabilities=[CodeMode(monty_sandbox=monty_relay.url)],
    )

    @agent.tool_plain
    async def drop_connection() -> None:
        # One statement: the harness cancels this while the reset is in flight, so a separate
        # wait line would only sometimes run. The wait never ends on its own.
        await asyncio.gather(monty_relay.drop_streams(), asyncio.Event().wait())

    result = await agent.run('lose the worker mid-snippet')

    (retry,) = _retries(result.all_messages())
    assert 'sandbox connection was lost' in str(retry.content)
    assert 'drop_connection({}) did not finish' in str(retry.content)


_SECRET = 'https://alice:secret@sandbox.example.com/?token=private Bearer factory-key'


@dataclass
class _DisconnectError(Exception):
    cause: str | None
    message: str


@dataclass
class _CrashedError(Exception):
    timed_out: bool
    message: str


@pytest.fixture
async def local_backed_factory_pool() -> AsyncIterator[MagicMock]:
    """A factory pool double backed by local native sessions, not remote transport."""
    async with AsyncMonty() as local_pool:
        pool = MagicMock(spec=AsyncMontyRemote)
        pool.__aenter__.return_value = pool
        pool.__aexit__.return_value = None
        pool.checkout.side_effect = local_pool.checkout
        yield pool


class TestCodeMode:
    async def test_local_backed_factory_keeps_execution_time_limit(self, local_backed_factory_pool: MagicMock) -> None:
        """A factory pool must receive the sandbox execution limit in addition to its own transport deadline."""
        agent = Agent(
            _snippets_model('while True:\n    pass'),
            capabilities=[
                CodeMode(
                    monty_sandbox=MagicMock(return_value=local_backed_factory_pool),
                    resource_limits={'max_duration_secs': 0.01},
                )
            ],
        )
        result = await agent.run('exhaust the configured execution time limit')
        (retry,) = _retries(result.all_messages())
        assert 'ran longer than its time limit' in str(retry.content)
        assert local_backed_factory_pool.checkout.call_args.kwargs['limits']['max_feed_duration_secs'] == 0.01

    @pytest.mark.parametrize(
        ('causes', 'expected_cause', 'expected_status', 'expected_checkouts'),
        [
            *[
                pytest.param((cause,), cause, status, 2, id=cause)
                for cause, status in (
                    ('caller_cap', 'RESOURCE_EXHAUSTED'),
                    ('capacity', 'RESOURCE_EXHAUSTED'),
                    ('worker_unavailable', 'UNAVAILABLE'),
                )
            ],
            *[
                pytest.param((cause, 'unauthenticated'), 'unauthenticated', 'UNAUTHENTICATED', 2, id=f'{cause}-auth')
                for cause in ('caller_cap', 'capacity', 'worker_unavailable')
            ],
            (('unauthenticated',), 'unauthenticated', 'UNAUTHENTICATED', 1),
            (('permission_denied',), 'permission_denied', 'PERMISSION_DENIED', 1),
            (('unknown',), None, 'RESOURCE_EXHAUSTED', 1),
            (None, None, 'UNAUTHENTICATED', 1),
        ],
    )
    async def test_native_remote_checkout_refusals(
        self,
        monkeypatch: pytest.MonkeyPatch,
        causes: tuple[str, ...] | None,
        expected_cause: str | None,
        expected_status: str,
        expected_checkouts: int,
    ) -> None:
        """Real transport refusals retry capacity once, propagate the final native error, and never reach the model."""
        sleep = AsyncMock()
        monkeypatch.setattr('anyio.sleep', sleep)
        async with serve_monty_relay(refuse=True, refusal_causes=causes) as relay:
            agent = Agent(_snippets_model('1 + 1'), capabilities=[CodeMode(monty_sandbox=relay.url)])
            with capture_run_messages() as messages, pytest.raises(MontyRefusedError) as exc_info:
                await agent.run('checkout')

        assert exc_info.value.cause == expected_cause
        assert exc_info.value.status_code == expected_status
        assert exc_info.value.status_message == ('unauthorized' if causes is None else 'refused')
        assert _retries(messages) == []
        assert str(exc_info.value) not in str(messages)
        assert relay.url not in str(messages)
        assert len(relay.metadata) == expected_checkouts
        assert sleep.await_args_list == ([((0.1,), {})] if expected_checkouts == 2 else [])

    @pytest.mark.parametrize('cause', ['caller_cap', 'capacity', 'worker_unavailable'])
    async def test_local_backed_factory_recovers_after_native_capacity_refusal(
        self, monkeypatch: pytest.MonkeyPatch, local_backed_factory_pool: MagicMock, cause: str
    ) -> None:
        """A local-backed checkout proves successful retry without relying on the skipped remote bridge.

        The first checkout raises an actual transport refusal; the second uses a real local session.
        """
        async with serve_monty_relay(refusal_causes=(cause,)) as relay:
            async with AsyncMontyRemote(relay.url) as pool:
                with pytest.raises(MontyRefusedError) as exc_info:
                    async with pool.checkout():
                        pytest.fail('refusal server admitted a session')  # pragma: no cover

        sleep = AsyncMock()
        monkeypatch.setattr('anyio.sleep', sleep)
        checkout = local_backed_factory_pool.checkout.side_effect
        attempts = 0

        @asynccontextmanager
        async def checkout_after_refusal(**kwargs: Any) -> AsyncGenerator[AsyncMontySession]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise exc_info.value
            async with checkout(**kwargs) as session:
                yield session

        local_backed_factory_pool.checkout.side_effect = checkout_after_refusal
        factory = MagicMock(return_value=local_backed_factory_pool)
        agent = Agent(_snippets_model('1 + 1'), capabilities=[CodeMode(monty_sandbox=factory)])
        result = await agent.run('recover after capacity refusal')

        assert _parts(result.all_messages(), ToolReturnPart)[-1].content == 2
        assert _retries(result.all_messages()) == []
        assert attempts == 2
        factory.assert_called_once_with()
        sleep.assert_awaited_once_with(0.1)

    async def test_refusal_sequence_repeats_final_cause(self) -> None:
        """The relay repeats the last refusal, including across new checkouts on the same native pool."""
        async with serve_monty_relay(refusal_causes=('capacity', 'unauthenticated')) as relay:
            async with AsyncMontyRemote(relay.url) as pool:
                for cause in ('capacity', 'unauthenticated', 'unauthenticated'):
                    with pytest.raises(MontyRefusedError) as exc_info:
                        async with pool.checkout():
                            pytest.fail('refusal server admitted a session')  # pragma: no cover
                    assert exc_info.value.cause == cause
        assert len(relay.metadata) == 3

    async def test_empty_refusal_sequence_rejected(self) -> None:
        """An empty explicit sequence must fail before the relay can accept connections."""
        with pytest.raises(ValueError, match='at least one cause'):
            async with serve_monty_relay(refusal_causes=()):
                pytest.fail('empty refusal sequence accepted')  # pragma: no cover

    @pytest.mark.parametrize('error_type', [RuntimeError, UserError, LogfireConfigError])
    async def test_local_backed_factory_checkout_error(
        self, local_backed_factory_pool: MagicMock, error_type: type[Exception]
    ) -> None:
        """A deterministic checkout double pins non-native setup errors that gRPC cannot produce."""
        error = error_type(_SECRET)
        local_backed_factory_pool.checkout.side_effect = error
        factory = MagicMock(return_value=local_backed_factory_pool)
        agent = Agent(_snippets_model('1 + 1'), capabilities=[CodeMode(monty_sandbox=factory)])
        with capture_run_messages() as messages, pytest.raises(error_type) as exc_info:
            await agent.run('checkout')

        assert exc_info.value is error
        assert _retries(messages) == []
        assert _SECRET not in str(messages)
        local_backed_factory_pool.checkout.assert_called_once()
        factory.assert_called_once_with()

    @pytest.mark.parametrize(
        ('error', 'guidance'),
        [
            (_CrashedError(True, _SECRET), 'ran longer than its time limit'),
            (_CrashedError(False, _SECRET), 'crashed the sandbox worker'),
            (_DisconnectError('turn_timeout', _SECRET), 'ran longer than its time limit'),
            (_DisconnectError('session_timeout', _SECRET), 'session ended; state is lost'),
            (_DisconnectError('idle_timeout', _SECRET), 'session ended; state is lost'),
            (_DisconnectError(None, _SECRET), 'sandbox connection was lost'),
            (_DisconnectError('unknown', _SECRET), 'sandbox connection was lost'),
            (RuntimeError(_SECRET), 'Code execution failed and the session was reset'),
        ],
    )
    async def test_local_backed_factory_execution_failure_resets_without_leaking_pool_details(
        self, monkeypatch: pytest.MonkeyPatch, local_backed_factory_pool: MagicMock, error: Exception, guidance: str
    ) -> None:
        """Local-backed doubles pin crash/disconnect guidance while real bridge execution is unavailable (#8824).

        Native crash/disconnect errors cannot be constructed in Python, so inject typed properties
        at the public native snapshot boundary after a real tool call, not into the harness executor.
        """
        assert _SECRET in str(error)
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.MontyCrashedError', _CrashedError)
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.MontyDisconnectError', _DisconnectError)
        original_resume = AsyncFutureSnapshot.resume

        async def fail_after_tool(
            snapshot: AsyncFutureSnapshot, results: dict[int, ExternalSettledResult]
        ) -> AsyncSnapshot:
            await original_resume(snapshot, results)
            raise error

        monkeypatch.setattr(AsyncFutureSnapshot, 'resume', fail_after_tool)
        agent = Agent(
            _snippets_model('saved = 42\nawait ping()', 'saved', '1 + 1'),
            capabilities=[CodeMode(monty_sandbox=MagicMock(return_value=local_backed_factory_pool))],
        )
        calls: list[None] = []

        @agent.tool_plain
        def ping() -> int:
            calls.append(None)
            return 7

        result = await agent.run('fail after a tool call')
        first, second = _retries(result.all_messages())
        assert guidance in str(first.content)
        assert 'ping({}) returned 7' in str(first.content)
        assert 'Name `saved` used when not defined' in str(second.content)
        assert _SECRET not in str(result.all_messages())
        assert 'secret' not in str(first.content)
        assert 'factory-key' not in str(first.content)
        assert calls == [None]
        assert local_backed_factory_pool.checkout.call_count == 3

    @pytest.mark.parametrize(
        ('error_name', 'code'),
        [
            ('MemoryError', 'raise MemoryError("allocation failed")'),
            ('MemoryError', 'await fail_memory()'),
            ('ValueError', 'raise ValueError("runtime failure")'),
        ],
    )
    async def test_local_backed_factory_runtime_error(
        self, local_backed_factory_pool: MagicMock, error_name: str, code: str
    ) -> None:
        """Local-backed factory plumbing preserves state for snippet/tool exceptions with clear source frames."""
        agent = Agent(
            _snippets_model(f'saved = 42\nawait ping()\n{code}', 'saved'),
            capabilities=[
                CodeMode(
                    monty_sandbox=MagicMock(return_value=local_backed_factory_pool),
                    resource_limits={'max_memory': 8 * 1024 * 1024},
                )
            ],
        )

        @agent.tool_plain
        def ping() -> int:
            return 7

        @agent.tool_plain
        def fail_memory() -> None:
            raise MemoryError('host allocation failed')

        result = await agent.run('fail during execution')
        retries = _retries(result.all_messages())
        assert len(retries) == 1, [retry.content for retry in retries]
        (retry,) = retries
        assert error_name in str(retry.content)
        assert 'Traceback' in str(retry.content)
        if 'fail_memory' in code:
            assert 'host allocation failed' in str(retry.content)
        elif error_name == 'MemoryError':
            assert 'allocation failed' in str(retry.content)
        assert 'session was reset' not in str(retry.content)
        assert _parts(result.all_messages(), ToolReturnPart)[-1].content == 42

    @pytest.mark.parametrize('code', ['data = [0] * 1_000_000', 'raise MemoryError("memory limit exceeded")'])
    async def test_local_backed_factory_memory_limit_resets(
        self, local_backed_factory_pool: MagicMock, code: str
    ) -> None:
        """Heap limits invalidate state even on live workers; an indistinguishable spoof resets conservatively."""
        agent = Agent(
            _snippets_model(f'saved = 42\nawait ping()\n{code}', 'saved', '1 + 1'),
            capabilities=[
                CodeMode(
                    monty_sandbox=MagicMock(return_value=local_backed_factory_pool),
                    resource_limits={'max_memory': 8 * 1024 * 1024},
                )
            ],
        )

        @agent.tool_plain
        def ping() -> int:
            return 7

        result = await agent.run('exhaust the sandbox heap limit')
        first, second = _retries(result.all_messages())
        assert 'memory limit exceeded' in str(first.content)
        assert 'The sandbox ran out of memory and the session was reset.' in str(first.content)
        assert 'ping({}) returned 7' in str(first.content)
        assert 'Name `saved` used when not defined' in str(second.content)
        assert _parts(result.all_messages(), ToolReturnPart)[-1].content == 2
        assert local_backed_factory_pool.checkout.call_count == 3

    async def test_local_backed_factory_memory_limit_before_execution_resets(
        self, local_backed_factory_pool: MagicMock
    ) -> None:
        """A frameless pre-execution heap limit also invalidates the session rather than reusing its heap."""
        agent = Agent(
            _snippets_model('value = 42', 'value'),
            capabilities=[
                CodeMode(
                    monty_sandbox=MagicMock(return_value=local_backed_factory_pool), resource_limits={'max_memory': 1}
                )
            ],
        )
        result = await agent.run('exhaust the sandbox budget before execution')
        retries = _retries(result.all_messages())
        assert len(retries) == 2
        for retry in retries:
            assert 'memory limit exceeded' in str(retry.content)
            assert 'The sandbox ran out of memory and the session was reset.' in str(retry.content)
        assert local_backed_factory_pool.checkout.call_count == 2

    @pytest.mark.parametrize(
        'remote',
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skip(
                    reason='Hangs intermittently in CI: https://github.com/pydantic/pydantic-ai/issues/8824'
                ),
            ),
        ],
        ids=['local', 'remote-bridge'],
    )
    async def test_native_worker_out_of_memory_resets(self, monty_relay: MontyRelay, remote: bool) -> None:
        """A refused 1 EiB allocation kills a native worker; real remote coverage awaits the bridge fix (#8824).

        The allocation exceeds the address space, so the allocator refuses it without consuming host memory.
        Monty marks the checkout finished after this error; soft heap limits must also reset state.
        """
        agent = Agent(
            _snippets_model("saved = 42\nawait ping()\ndata = ' ' * (1 << 60)", 'saved', '1 + 1'),
            capabilities=[
                CodeMode(
                    monty_sandbox=monty_relay.url if remote else None,
                    resource_limits='unlimited',
                )
            ],
        )

        @agent.tool_plain
        def ping() -> int:
            return 7

        result = await agent.run('exhaust the worker address space')
        first, second = _retries(result.all_messages())

        assert 'the worker exceeded its memory limit and was terminated' in str(first.content)
        assert 'session was reset' in str(first.content)
        assert 'use less memory' in str(first.content)
        assert 'process smaller batches' in str(first.content)
        assert 'ping({}) returned 7' in str(first.content)
        assert 'Name `saved` used when not defined' in str(second.content)
        assert _parts(result.all_messages(), ToolReturnPart)[-1].content == 2

    async def test_local_backed_factory_restore_checkout_failure_reaches_caller(
        self, local_backed_factory_pool: MagicMock
    ) -> None:
        """Local-backed restore plumbing must not turn setup errors into model feedback or overwrite saved state."""
        store = MemoryReplStore()
        agent = Agent(
            _snippets_model('saved = 42'),
            deps_type=type(None),
            capabilities=[
                CodeMode[None](monty_sandbox=MagicMock(return_value=local_backed_factory_pool), repl_store=store)
            ],
        )
        await agent.run('save', conversation_id='conversation')
        saved = store.states.copy()
        error = RuntimeError(_SECRET)
        local_backed_factory_pool.checkout.side_effect = error

        with capture_run_messages() as messages, pytest.raises(RuntimeError) as exc_info:
            await agent.run('restore', conversation_id='conversation')

        assert exc_info.value is error
        assert _retries(messages) == []
        assert _SECRET not in str(messages)
        assert store.states == saved
