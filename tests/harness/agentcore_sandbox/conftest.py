"""Shared fixtures for AgentCoreSandbox tests."""

from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

_HAS_BOTO3 = importlib.util.find_spec('boto3') is not None
collect_ignore = (
    []
    if _HAS_BOTO3
    else ['test_agentcore_live.py', 'test_agentcore_sandbox.py', 'test_backend.py', 'test_conformance.py']
)

# AWS region of the live tier; the session it starts lives there.
LIVE_REGION = os.getenv('AWS_REGION', 'us-east-1')

if TYPE_CHECKING or _HAS_BOTO3:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    from .fake_agentcore import FakeAgentCore


def _has_aws_credentials() -> bool:  # pragma: no cover - live job
    import boto3

    return boto3.Session().get_credentials() is not None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can start a session.

    It runs only with `PYDANTIC_AI_HARNESS_AGENTCORE_LIVE=1` and AWS credentials boto3 can find.
    Otherwise it skips, unless `AGENTCORE_REQUIRE_LIVE=1`, where missing credentials fail instead:
    a live job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('agentcore_live') is None:
        return
    if os.getenv('PYDANTIC_AI_HARNESS_AGENTCORE_LIVE') == '1' and _has_aws_credentials():  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_AGENTCORE_LIVE=1 and AWS credentials'
    if os.getenv('AGENTCORE_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'AGENTCORE_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


class _PoisonedBoto3(types.ModuleType):
    """A `boto3` stand-in that fails loudly on any attribute access.

    boto3 is installed in the dev venv, so a unit test that forgets the `fake_agentcore` fixture
    would otherwise reach AWS and, with credentials configured, start real billed sessions.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'An agentcore_sandbox unit test touched the real `boto3` package. '
            'Use the `fake_agentcore` fixture, or mark the test `agentcore_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_boto3(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `boto3` for every test here except the opt-in live tier."""
    if 'agentcore_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend.boto3', _PoisonedBoto3('boto3'))
    yield


if _HAS_BOTO3:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    def fake_agentcore(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeAgentCore]:
        """Swap `boto3` for the fake, whose sessions run commands and code in a temporary host directory."""
        fake = FakeAgentCore(home=tmp_path.resolve())
        monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend.boto3', fake.module)
        # Nothing here waits on a real service.
        monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend._POLL_INTERVAL', 0.01)
        try:
            yield fake
        finally:
            fake.close()
