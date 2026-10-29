"""The harness suite runs under the autouse blocking-call detector, like the rest of `tests/`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pydantic_ai_harness.capability_creation import CapabilityStore


async def test_a_blocking_harness_call_in_the_event_loop_is_detected(tmp_path: Path) -> None:
    if os.getenv('BLOCKBUSTER_ENABLED') == 'false':
        pytest.skip('BlockBuster is disabled in this CI lane')
    from blockbuster import BlockingError

    # The store reads its manifest from disk synchronously; its async callers offload it to a thread.
    with pytest.raises(BlockingError):
        CapabilityStore(tmp_path).list_all()
