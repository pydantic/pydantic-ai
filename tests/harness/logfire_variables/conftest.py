"""Shared fixtures for the `logfire_variables` test package.

The directory is named `logfire_variables`, not `logfire`, on purpose: pyright scopes test-only
report overrides with `executionEnvironments = [{ root = 'tests' }]`, which makes `tests/` an
import root -- so a `tests/logfire/` directory would shadow the third-party `logfire` package for
every test file's `import logfire`.

All resolution runs against the code default (no Logfire provider is configured) unless a test
installs one via `variables_provider`, which is exactly the safety-net behavior the managed-variable
capabilities rely on. Each test uses a unique variable name because the default Logfire instance
keeps its variable registry across `configure()` calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from typing import Any

import logfire
import pytest
from logfire.agent_control._reporting import reset_warned_messages
from logfire.testing import CaptureLogfire
from logfire.variables.local import LocalVariableProvider
from pydantic import BaseModel

from pydantic_ai_harness.logfire import _agent_control

from ._helpers import Publish, published_value, variables_provider

_LOGFIRE_CREDENTIAL_VARS = ('LOGFIRE_TOKEN', 'LOGFIRE_API_KEY')


@pytest.fixture(autouse=True)
def _forget_process_state() -> Iterator[None]:
    """Start each test with the once-per-process guards empty, on both sides of the contract.

    A drop warns once per process and an unconfigured agent is reported once per process, by design,
    so without this a test's outcome would depend on which tests ran before it. The parser's guard
    lives in `logfire.agent_control` and the capability's in this package; the two never emit the same
    message, and a test that counts warnings has to reach both.
    """
    reset_warned_messages()
    _agent_control._warned_drops.clear()
    _agent_control._reset_config_hint_guard()
    yield
    reset_warned_messages()
    _agent_control._warned_drops.clear()
    _agent_control._reset_config_hint_guard()


@pytest.fixture(autouse=True)
def _refuse_variable_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test in which a managed capability writes to a variable.

    These capabilities read a variable's value and nothing else. An agent Logfire holds no config for
    reports itself on a span and Logfire creates the config from that, so a deployment needs no write
    scope and a code baseline can never overwrite an edit saved in the UI. Stated as a fixture over
    the whole package rather than as one test, because the claim is about every code path here rather
    than about the one a single test happens to take.

    Only the writes are refused. `get_variable_config` reads, and resolution itself calls it whenever
    a `label` is pinned -- `get_serialized_value_for_label` is implemented on top of it.
    """

    def refuse(method: str) -> Callable[..., Any]:
        def refused(self: LocalVariableProvider, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                f'A managed capability called {method!r} on the variable provider. '
                'Reading a value is the only thing these capabilities may do to a variable.'
            )

        return refused

    for method in ('create_variable', 'update_variable', 'delete_variable'):
        monkeypatch.setattr(LocalVariableProvider, method, refuse(method))


@pytest.fixture(autouse=True, scope='package')
def _scrub_logfire_credentials() -> Iterator[None]:
    """Remove ambient Logfire credentials while this package's tests run.

    Logfire lazily creates a `LogfireRemoteVariableProvider` that issues requests to the Logfire API
    whenever a credential such as `LOGFIRE_API_KEY` is present in `os.environ`, even with
    `send_to_logfire=False`. A credential inherited from the developer's shell, or from a `.env`
    file loaded into the process, would therefore make these tests write to the real Logfire API
    from background publish threads. Package scope orders this ahead of every `logfire.configure()`
    below and in the test modules, which capture `api_key` at configure time.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in _LOGFIRE_CREDENTIAL_VARS:
            monkeypatch.delenv(name, raising=False)
        yield


@pytest.fixture(autouse=True, scope='module')
def _configure_logfire() -> None:
    """Configure Logfire once so variable resolution does not warn (warnings are errors)."""
    logfire.configure(send_to_logfire=False, console=False)


@pytest.fixture
def anyio_backend() -> str:
    # Pin to asyncio: some tests use `asyncio` primitives directly, and the resolution behavior is
    # backend-agnostic, so running the trio leg too would only duplicate work.
    return 'asyncio'


@pytest.fixture
def publish(capfire: CaptureLogfire) -> Iterator[Publish]:
    """Publish a managed config for `agent__<agent_name>` for the rest of the test.

    `AgentControl` has no code-side default to set -- the agent itself is the code side -- so a test
    that wants a managed value in play has to put one in the project, which is also the only way a
    real deployment gets one. Written as a fixture rather than a `with` block so the provider is torn
    down at the end of the test even when a test publishes and then asserts across several runs.
    """
    stack = ExitStack()

    def _publish(agent_name: str, config: Any) -> None:
        value = config.model_dump(exclude_none=True) if isinstance(config, BaseModel) else config
        stack.enter_context(variables_provider(capfire, published_value(f'agent__{agent_name}', value)))

    try:
        yield _publish
    finally:
        stack.close()
