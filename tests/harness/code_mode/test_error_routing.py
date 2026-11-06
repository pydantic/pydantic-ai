"""CodeMode routes setup errors to callers and recoverable execution errors to models.

Native transport errors cannot be constructed from Python, so these tests inject their
typed properties at checkout or after a real feed. The pool double runs real local sessions.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import logfire
import pytest
from logfire.exceptions import LogfireConfigError
from pydantic_monty import AsyncMonty, AsyncMontyRemote, AsyncMontySession, AsyncSnapshot, MontyComplete

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness._monty_exec import MontyExecutor
from tests.harness.code_mode.test_remote import retry_parts, snippets_model
from tests.harness.code_mode.test_repl_store import MemoryReplStore

_SECRET = 'https://alice:secret@sandbox.example.com/?token=private Bearer factory-key'


@dataclass
class _RefusedError(Exception):
    cause: str | None


@dataclass
class _DisconnectError(Exception):
    cause: str | None
    message: str


@dataclass
class _CrashedError(Exception):
    timed_out: bool
    message: str


@pytest.fixture
async def remote_pool() -> AsyncIterator[MagicMock]:
    """Run factory-pool feeds locally without depending on a hosted service or the gRPC relay."""
    async with AsyncMonty() as local_pool:
        pool = MagicMock(spec=AsyncMontyRemote)
        pool.__aenter__.return_value = pool
        pool.__aexit__.return_value = None
        pool.checkout.side_effect = local_pool.checkout
        yield pool


class TestCodeMode:
    @pytest.mark.parametrize(
        ('error', 'failures', 'expected_checkouts'),
        [
            *[
                pytest.param(_RefusedError(cause), failures, 2, id=f'{cause}-{failures}')
                for cause in ('caller_cap', 'capacity', 'worker_unavailable')
                for failures in (1, 2)
            ],
            *[
                pytest.param(_RefusedError(cause), 1, 1, id=f'refused-{cause}')
                for cause in ('unauthenticated', 'permission_denied', 'unknown', None)
            ],
            (RuntimeError(_SECRET), 1, 1),
            (UserError(_SECRET), 1, 1),
            (LogfireConfigError(_SECRET), 1, 1),
        ],
    )
    async def test_checkout_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        remote_pool: MagicMock,
        error: Exception,
        failures: int,
        expected_checkouts: int,
    ) -> None:
        """Only capacity refusals get one delayed checkout retry; exhausted or non-capacity errors reach callers."""
        monkeypatch.setattr('pydantic_ai_harness._monty_exec.MontyRefusedError', _RefusedError)
        sleep = AsyncMock()
        monkeypatch.setattr('pydantic_ai_harness._monty_exec.anyio.sleep', sleep)
        checkout = remote_pool.checkout.side_effect
        attempts = 0

        @asynccontextmanager
        async def failing_checkout(**kwargs: Any) -> AsyncGenerator[AsyncMontySession]:
            nonlocal attempts
            attempts += 1
            if attempts <= failures:
                raise error
            async with checkout(**kwargs) as session:
                yield session

        remote_pool.checkout.side_effect = failing_checkout
        factory = MagicMock(return_value=remote_pool)
        agent = Agent(snippets_model('1 + 1'), capabilities=[CodeMode(monty_sandbox=factory)])

        with capture_run_messages() as messages:
            if failures < expected_checkouts:
                result = await agent.run('checkout')
                assert result.output == 'done'
            else:
                with pytest.raises(type(error)) as exc_info:
                    await agent.run('checkout')
                assert exc_info.value is error

        assert retry_parts(messages) == []
        assert _SECRET not in str(messages)
        assert attempts == expected_checkouts
        factory.assert_called_once_with()
        assert sleep.await_args_list == ([((0.1,), {})] if expected_checkouts == 2 else [])

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
    async def test_execution_failure_resets_without_leaking_pool_details(
        self, monkeypatch: pytest.MonkeyPatch, remote_pool: MagicMock, error: Exception, guidance: str
    ) -> None:
        """Execution guidance names the failure and lost state, preserves completed calls, and excludes transport secrets."""
        assert _SECRET in str(error)
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.MontyCrashedError', _CrashedError)
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.MontyDisconnectError', _DisconnectError)
        original_run = MontyExecutor.run
        failed = False

        async def fail_after_feed(
            executor: MontyExecutor, feed_start: Callable[[], Awaitable[AsyncSnapshot]]
        ) -> MontyComplete:
            nonlocal failed
            result = await original_run(executor, feed_start)
            if not failed:
                failed = True
                raise error
            return result

        monkeypatch.setattr(MontyExecutor, 'run', fail_after_feed)
        agent = Agent(
            snippets_model('saved = 42\nawait ping()', 'saved', '1 + 1'),
            capabilities=[CodeMode(monty_sandbox=MagicMock(return_value=remote_pool))],
        )
        calls: list[None] = []

        @agent.tool_plain
        def ping() -> int:
            calls.append(None)
            return 7

        result = await agent.run('fail after a tool call')
        first, second = retry_parts(result.all_messages())
        assert guidance in str(first.content)
        assert 'ping({}) returned 7' in str(first.content)
        assert 'Name `saved` used when not defined' in str(second.content)
        assert _SECRET not in str(result.all_messages())
        assert 'secret' not in str(first.content)
        assert 'factory-key' not in str(first.content)
        assert calls == [None]
        assert remote_pool.checkout.call_count == 3

    @pytest.mark.parametrize('error_name', ['MemoryError', 'ValueError'])
    async def test_remote_runtime_error(self, remote_pool: MagicMock, error_name: str) -> None:
        """Remote memory errors reset with allocation advice; ordinary runtime errors keep their REPL state."""
        agent = Agent(
            snippets_model(f'saved = 42\nawait ping()\nraise {error_name}("allocation failed")', 'saved'),
            capabilities=[CodeMode(monty_sandbox=MagicMock(return_value=remote_pool))],
        )

        @agent.tool_plain
        def ping() -> int:
            return 7

        result = await agent.run('fail during execution')
        retries = retry_parts(result.all_messages())
        assert error_name in str(retries[0].content)
        if error_name == 'MemoryError':
            assert 'use less memory' in str(retries[0].content)
            assert 'process smaller batches' in str(retries[0].content)
            assert 'ping({}) returned 7' in str(retries[0].content)
            assert len(retries) == 2
            assert 'Name `saved` used when not defined' in str(retries[1].content)
        else:
            assert len(retries) == 1
            assert 'session was reset' not in str(retries[0].content)

    async def test_restore_checkout_failure_reaches_caller(self, remote_pool: MagicMock) -> None:
        """Restoring a saved REPL must not convert a setup error into model feedback or overwrite saved state."""
        store = MemoryReplStore()
        agent = Agent(
            snippets_model('saved = 42'),
            deps_type=type(None),
            capabilities=[CodeMode[None](monty_sandbox=MagicMock(return_value=remote_pool), repl_store=store)],
        )
        await agent.run('save', conversation_id='conversation')
        saved = store.states.copy()
        error = RuntimeError(_SECRET)
        remote_pool.checkout.side_effect = error

        with capture_run_messages() as messages, pytest.raises(RuntimeError) as exc_info:
            await agent.run('restore', conversation_id='conversation')

        assert exc_info.value is error
        assert retry_parts(messages) == []
        assert _SECRET not in str(messages)
        assert store.states == saved

    async def test_logfire_factory_example(self, monkeypatch: pytest.MonkeyPatch, remote_pool: MagicMock) -> None:
        """The documented Logfire factory is lazy and runs real sessions when called by CodeMode."""
        factory = MagicMock(return_value=remote_pool)
        monkeypatch.setattr(logfire, 'agent_sandboxes', factory, raising=False)
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
        readme = (
            Path(__file__).parents[3] / 'src/pydantic_ai_harness/pydantic_ai_harness/code_mode/README.md'
        ).read_text()
        example = next(
            block.split('```', 1)[0]
            for block in readme.split('```python {test="skip"}\n')[1:]
            if 'monty_sandbox=logfire.agent_sandboxes' in block.split('```', 1)[0]
        )
        namespace: dict[str, Any] = {}
        exec(example, namespace)
        agent: Agent[None, str] = namespace['agent']
        factory.assert_not_called()

        result = await agent.run('run a snippet', model=snippets_model('1 + 1'))

        assert result.output == 'done'
        factory.assert_called_once_with()
