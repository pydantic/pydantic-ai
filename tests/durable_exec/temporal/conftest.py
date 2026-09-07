"""Fixtures for the Temporal suite. Two things must never be done at module level in this file.

**No requirement gate.** Unlike the test modules and `_shared` beside it, this file carries no
`pytest.skip(..., allow_module_level=True)`, and has to stay importable without `temporalio`,
`logfire`, `mcp` or `openai` and on Python 3.14. `pytest.skip` raises `Skipped`, a `BaseException`;
pytest loads the conftest of every command-line argument's directory up front, from
`PytestPluginManager._set_initial_conftests`, and `_importconftest` catches only `Exception` — so a
gate here escapes as a traceback with exit 1 on `pytest tests/durable_exec/temporal` rather than
reporting a skip. Naming a parent directory hides it, because the conftest is then imported during
collection, which does handle `Skipped`. The test modules' own gates report the skip, and the
fixtures below run only once those gates have passed, so they import what they need when called.

**Nothing sandbox-sensitive at module level.** `_shared`, `pandas` and the root `tests/conftest.py`
(which loads `vcr`) are imported by the test modules inside their own
`workflow.unsafe.imports_passed_through()` blocks. Importing any of them at module level here would
re-enter sandbox territory with no passthrough of its own, and the gate rule above forbids the
`temporalio` import that a passthrough block would need. A fixture body is the sanctioned place —
see `close_cached_httpx_client` — because it runs in the main process at test time, never during the
sandbox's re-import of this module.

This is why the suite does not use `tests/conftest.py`'s `try_import()`, which is the repo's default
for optional dependencies elsewhere: importing it would drag `vcr` in at module level.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pydantic_ai import (
    Agent,
)
from pydantic_ai._warnings import PydanticAIDeprecationWarning

if TYPE_CHECKING:
    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment

# `TemporalAgent` is deprecated in favor of `capabilities=[TemporalDurability(...)]`.
# These tests exercise the wrapper-agent path on purpose; suppress the warning here
# rather than globally in `pyproject.toml`. The `pytestmark` entry below covers warnings
# emitted *inside* test functions; the `filterwarnings` call below covers warnings emitted
# at module import time (e.g. module-level construction of `TemporalAgent`).
warnings.filterwarnings('ignore', message='`TemporalAgent` is deprecated', category=PydanticAIDeprecationWarning)


@pytest.fixture
def blockbuster_enabled() -> bool:
    """Disable detection for Temporal's synchronous worker and integration setup.

    It performs module/config introspection above Pydantic AI plugin frames; BlockBuster changes
    its error handling and makes these tests unusably slow. Rebenchmark after
    https://github.com/cbornet/blockbuster/pull/61 is released, but retain this opt-out until the
    synchronous-introspection false positives are isolated too.
    """
    return False


# Scoped to `session` rather than `module`: the `http_client` and the module-level agents that
# capture it are constructed at import time, so they must outlive a single module entry. This is a
# sync fixture so it doesn't force AnyIO to reuse a session-level event loop for all Temporal async
# fixtures; the `temporal_env` teardown can make that loop unusable for later tests.
@pytest.fixture(autouse=True, scope='session')
def close_cached_httpx_client() -> Iterator[None]:
    from ._shared import http_client

    try:
        yield
    finally:
        asyncio.run(http_client.aclose())


# `LogfirePlugin` calls `logfire.instrument_pydantic_ai()`, so we need to make sure this doesn't bleed into other tests.
@pytest.fixture(autouse=True, scope='module')
def uninstrument_pydantic_ai() -> Iterator[None]:
    try:
        yield
    finally:
        Agent.instrument_all(False)


# One dev server (and thus one xdist group) per test module so the four temporal files can
# run on four xdist workers concurrently instead of forming one serial chain. `start_local`
# picks the port: anyio's `free_tcp_port_factory` only probes and releases, so the port sits
# unowned for the rest of fixture setup, and its dedup set is per factory instance — one per
# xdist worker — so two workers can be handed the same port. The Temporal SDK instead allocates
# immediately before spawning the server, and on Linux holds the port in `TIME_WAIT` so the kernel
# won't hand it to another process's `bind(0)`:
# https://github.com/temporalio/sdk-core/blob/5962c094869d691b78b9732f09851a9183173db9/crates/sdk-core/src/ephemeral_server/mod.rs#L548
@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    from temporalio.testing import WorkflowEnvironment

    # `start_local` downloads the dev-server binary to the system temp dir by default, which is empty on
    # every CI run, so a CDN hiccup used to fail the entire suite at setup (#5399). Download to a stable
    # per-user cache dir instead so CI can restore it via `actions/cache` and local runs reuse it across
    # reboots. Resolved here rather than at module level: the workflow sandbox re-imports this module and
    # restricts `Path.home()` access.
    download_dest_dir = Path.home() / '.cache' / 'temporal-dev-server'
    download_dest_dir.mkdir(parents=True, exist_ok=True)
    # Leave `ui` off (the `start_local` default). With `ui=True` and no explicit `ui_port`, the dev
    # server binds `port + 1000` without probing it first, and a bind failure there aborts the whole
    # process — surfacing as `ConnectionRefused` on the healthy gRPC port. No test reads the UI.
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
        download_dest_dir=str(download_dest_dir),
    ) as env:
        yield env


# The `host:port` the dev server actually bound — read back rather than assumed — for tests that
# need a client of their own rather than `temporal_env.client`.
@pytest.fixture(scope='module')
def temporal_target(temporal_env: WorkflowEnvironment) -> str:
    return temporal_env.client.service_client.config.target_host


@pytest.fixture
async def client(temporal_target: str) -> Client:
    from temporalio.client import Client

    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

    return await Client.connect(
        temporal_target,
        plugins=[PydanticAIPlugin()],
    )


@pytest.fixture
async def client_with_logfire(temporal_target: str) -> Client:
    from temporalio.client import Client

    from pydantic_ai.durable_exec.temporal import LogfirePlugin, PydanticAIPlugin

    return await Client.connect(
        temporal_target,
        plugins=[PydanticAIPlugin(), LogfirePlugin()],
    )
