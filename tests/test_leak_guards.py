"""The leak guards in `tests/leak_guards.py`, run against small in-process pytest sessions."""

from __future__ import annotations as _annotations

import asyncio
import gc
import importlib.util
import textwrap
import warnings
from collections.abc import AsyncIterator

import pytest

pytest_plugins = ['pytester']

# Each test runs a session in this process that leaks on purpose, so what it leaks is the test's own garbage.
# `run_session` collects it, but pytester's hook recorder lives until the test's teardown and can keep some of it
# reachable past that collection: the teardown's collection must not fail the test on its `ResourceWarning` either.
pytestmark = pytest.mark.filterwarnings('ignore::ResourceWarning')

WORKER_LOOP = """
import gc
import threading

from pydantic_ai._utils import get_event_loop


def on_worker_thread(close: bool) -> None:
    def run() -> None:
        # No automatic collection while the loop is built. One that ran after the loop was allocated and before its
        # self-pipe was would leave the loop in an older generation than the pipe's sockets: a collection that finds
        # them all garbage then finalizes the sockets first, and the loop's finalizer raises closing them again.
        gc.disable()
        try:
            loop = get_event_loop()
        finally:
            gc.enable()
        if close:
            loop.close()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
"""


def run_session(pytester: pytest.Pytester, source: str) -> pytest.RunResult:
    pytester.makepyfile(test_session=WORKER_LOOP + textwrap.dedent(source))
    # The session inherits this suite's warning filters, under which a collected loop's `ResourceWarning` fails the
    # test too: ignored here, so each outcome is the guard's own.
    result = pytester.runpytest(
        '-p', 'tests.leak_guards', '-p', 'no:cacheprovider', '-p', 'no:pretty', '-W', 'ignore::ResourceWarning'
    )
    # The session runs in this process, so what its tests leak on purpose is this test's garbage: collected here, where
    # the suite's filters would otherwise fail this test for it. So is the loop its `run_sync()` calls leave set on this
    # thread, which would otherwise be dropped when this test's teardown sets another.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ResourceWarning)
        asyncio.set_event_loop(None)
        gc.collect()
    return result


def test_a_loop_left_open_on_a_worker_thread_fails_the_test_that_created_it(pytester: pytest.Pytester):
    result = run_session(
        pytester,
        """
        def test_leaks():
            on_worker_thread(close=False)

        def test_closes():
            on_worker_thread(close=True)

        def test_leaks_until_collected():
            import gc

            on_worker_thread(close=False)
            gc.collect()
        """,
    )
    result.assert_outcomes(passed=3, errors=2)
    created_at = r'\s+pydantic_ai_slim/pydantic_ai/_utils\.py:\d+ in get_event_loop'
    result.stdout.re_match_lines(
        [
            r'.*::test_leaks left 1 event loop\(s\) open, created at:',
            created_at,
            r'.*::test_leaks_until_collected left 1 event loop\(s\) open, created at:',
            created_at,
        ]
    )


def test_a_loop_on_the_main_thread_or_in_a_shared_fixture_is_left_to_its_owner(pytester: pytest.Pytester):
    result = run_session(
        pytester,
        """
        import pytest

        @pytest.fixture(scope='module')
        def module_loop():
            on_worker_thread(close=False)

        def test_main_thread():
            get_event_loop()

        def test_shared_fixture(module_loop):
            pass
        """,
    )
    result.assert_outcomes(passed=2)


CLIENTS = """
import pytest

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from tests.conftest import track_httpx_clients  # the autouse fixture that closes what a test leaves open


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'mock-api-key')


async def keep_open(self, *args):
    pass


def run_with_model_named(agent):
    with pytest.raises(RuntimeError, match='Model requests are not allowed'):
        agent.run_sync('hello', model='anthropic:claude-sonnet-4-5')
"""


@pytest.mark.skipif(importlib.util.find_spec('anthropic') is None, reason='anthropic not installed')
def test_a_client_an_agent_run_leaves_open_fails_the_test_and_a_client_its_builder_owns_does_not(
    pytester: pytest.Pytester,
):
    result = run_session(
        pytester,
        CLIENTS
        + textwrap.dedent(
            """
        def test_run_leaves_the_model_it_built_open(monkeypatch):
            monkeypatch.setattr(Model, '__aexit__', keep_open)
            run_with_model_named(Agent(TestModel()))

        def test_run_closes_the_model_it_built():
            run_with_model_named(Agent(TestModel()))

        def test_provider_the_test_built():
            AnthropicProvider()

        def test_agent_model_built_from_a_name_outside_a_run():
            Agent('anthropic:claude-sonnet-4-5')

        def test_provider_a_tool_builds_during_a_run():
            agent = Agent(TestModel())

            @agent.tool_plain
            def build_provider() -> str:
                AnthropicProvider()
                return 'built'

            agent.run_sync('hello')
        """
        ),
    )
    result.assert_outcomes(passed=5, errors=1)
    result.stdout.re_match_lines(
        [
            r'.*::test_run_leaves_the_model_it_built_open left 1 HTTP client\(s\) that an agent run built open, built by:',
            r'\s+pydantic_ai_slim/pydantic_ai/(agent/__init__|_agent_graph)\.py:\d+ in \w+',
        ]
    )


async def numbers() -> AsyncIterator[int]:
    """An async generator of this repository's code, for the sessions below to leave open or close."""
    while True:
        yield 1


GENERATORS = """
import asyncio

from tests import leak_guards
from tests.test_leak_guards import numbers


def iterate_once(close: bool) -> None:
    async def main() -> None:
        generator = numbers()
        await anext(generator)
        if close:
            await generator.aclose()

    asyncio.run(main())
"""


def test_an_async_generator_left_for_the_collector_fails_the_test_unless_the_baseline_lists_it(
    pytester: pytest.Pytester,
):
    result = run_session(
        pytester,
        GENERATORS
        + textwrap.dedent(
            """
        def test_leaves_it():
            iterate_once(close=False)

        def test_closes_it():
            iterate_once(close=True)

        def test_leaves_one_the_baseline_lists(monkeypatch):
            monkeypatch.setattr(leak_guards, 'GC_FINALIZED_BASELINE', frozenset({'tests/test_leak_guards.py:numbers'}))
            iterate_once(close=False)
        """
        ),
    )
    result.assert_outcomes(passed=3, errors=1)
    result.stdout.re_match_lines(
        [
            r'.*::test_leaves_it left 1 async generator\(s\) for the collector to finalize:',
            r'\s+tests/test_leak_guards\.py:numbers',
        ]
    )
