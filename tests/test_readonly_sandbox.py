"""Tests for the read-only sandbox policy wrapper."""

from __future__ import annotations

import pytest

from pydantic_ai import UserError
from pydantic_ai.sandboxes import ReadOnlySandbox, Sandbox, SandboxRef

from .sandbox_fakes import FakeSandbox

pytestmark = pytest.mark.anyio


async def test_read_only_sandbox_forwards_reads_and_refuses_run_and_writes() -> None:
    ref = SandboxRef(sandbox_id='existing')
    backend = FakeSandbox('read-only', {'/workspace/data.txt': b'original'}, ref=ref)
    sandbox = Sandbox(ReadOnlySandbox(backend))

    assert await sandbox.read_text('data.txt') == 'original'
    assert sandbox.ref == ref

    with pytest.raises(UserError, match='read-only'):
        await sandbox.run(['rm', 'data.txt'])
    with pytest.raises(UserError, match='read-only'):
        await sandbox.write_text('data.txt', 'changed')

    assert backend.fs.files['/workspace/data.txt'] == b'original'
