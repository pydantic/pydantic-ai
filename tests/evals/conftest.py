from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from opentelemetry import context as otel_ctx


@pytest.fixture(autouse=True)
async def _cleanup_background_evaluations() -> AsyncIterator[None]:
    """Drain background evaluation tasks after each test.

    Prevents leaked tasks from a failed test from affecting subsequent tests.
    """
    yield
    try:
        from pydantic_evals.online import wait_for_evaluations
    except ImportError:  # pragma: no cover
        return
    await wait_for_evaluations()


# The sync `fresh_logfire` fixture resets the main-thread OTel context, but async
# test bodies run in anyio's runner task whose context is copied before that fixture
# runs. A leaked non-sampled span in that task causes parent-based sampling to drop
# descendant spans, which surfaces as an empty `context_subtree()` tree. We keep a
# per-worker setup token so each async eval test can detach the clean context from
# the previous test and attach its own.
_setup_token: Any = None


@pytest.fixture(autouse=True)
async def _reset_async_otel_context() -> AsyncIterator[None]:
    """Reset the anyio runner task's OTel context around each async eval test."""
    global _setup_token
    if _setup_token is not None:
        otel_ctx.detach(_setup_token)
    _setup_token = otel_ctx.attach(otel_ctx.Context())
    yield
