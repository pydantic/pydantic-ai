"""Shared fixtures for `KubernetesSandbox` tests.

Every test here runs against `fake_sandboxd`: a fake agent-sandbox SDK whose claims are in-process
sandboxd servers running host commands, except `test_kubernetes_live.py`, the live tier on a kind
cluster. The same conformance suite also runs against real sandboxd, without a cluster, in
`src/pydantic_ai_harness/integration_tests/kubernetes` (`make integration-kubernetes`).
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

_HAS_SDK = all(importlib.util.find_spec(name) is not None for name in ('k8s_agent_sandbox', 'grpc'))
collect_ignore = [] if _HAS_SDK else ['test_conformance.py', 'test_kubernetes_live.py', 'test_kubernetes_sandbox.py']

# The SDK-installed and slim jobs take opposite branches.
if TYPE_CHECKING or _HAS_SDK:  # pragma: no branch
    from .fake_sandboxd import FakeCluster


# The fixture cannot be defined without its SDK-backed fake.
if _HAS_SDK:  # pragma: no branch

    @pytest.fixture
    async def cluster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeCluster]:
        """Replace the SDK's client with one on a fake cluster; every claim it makes is stopped afterwards."""
        fake = FakeCluster(tmp_path)
        monkeypatch.setattr('pydantic_ai_harness.kubernetes_sandbox._backend.AsyncSandboxClient', fake.client)
        try:
            yield fake
        finally:
            await fake.close()
