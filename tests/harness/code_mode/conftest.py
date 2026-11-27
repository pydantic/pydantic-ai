"""Fixtures shared by the CodeMode test modules."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from tests.harness.code_mode.grpc_relay import MontyRelay, serve_monty_relay


@pytest.fixture
async def monty_relay() -> AsyncIterator[MontyRelay]:
    """Serve remote Monty workers on an ephemeral loopback port."""
    async with serve_monty_relay() as relay:
        yield relay
