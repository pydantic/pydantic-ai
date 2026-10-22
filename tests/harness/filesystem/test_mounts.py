"""File tools in a workspace with mounted storage that shell commands can't see."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, Mount, Workspace
from pydantic_ai_harness.bucket import Bucket
from pydantic_ai_harness.filesystem import FileSystem

from .._tool_calls import call_tools

pytestmark = pytest.mark.anyio


async def _workspace(tmp_path: Path) -> tuple[Workspace, Bucket]:
    bucket = Bucket('memory:///')
    await bucket.write_bytes('/2026/refunds.csv', b'region,refund\neu,12\n')
    (tmp_path / 'notes.txt').write_text('refund policy: 30 days\n')
    return Workspace(LocalWorkspaceBackend(tmp_path), mounts=[Mount('data', bucket)]), bucket


async def test_searches_from_above_a_mount_include_it(tmp_path: Path) -> None:
    workspace, _ = await _workspace(tmp_path)

    grep, search, listing = await call_tools(
        [FileSystem[None](tools=['grep', 'search_files', 'list_files'])],
        [('grep', {'pattern': 'refund'}), ('search_files', {'pattern': 'refund'}), ('list_files', {})],
        workspace=workspace,
    )

    for found in (grep, search):
        assert 'data/2026/refunds.csv' in found
        assert 'notes.txt' in found
    assert listing.splitlines() == ['data/2026/refunds.csv', 'notes.txt']


async def test_writes_through_a_mount_reach_its_storage(tmp_path: Path) -> None:
    workspace, bucket = await _workspace(tmp_path)

    await call_tools(
        [FileSystem[None]()],
        [('write_file', {'path': 'data/summary.md', 'content': '# EU refunds: 12'})],
        workspace=workspace,
    )

    assert await bucket.read_bytes('/summary.md') == b'# EU refunds: 12'
    assert not (tmp_path / 'data').exists()


async def test_the_model_is_told_about_mounts(tmp_path: Path) -> None:
    exports = Mount('exports', Bucket('memory:///'), read_only=True, description='Daily exports')
    agent = Agent(TestModel(call_tools=[]), capabilities=[FileSystem()])

    result = await agent.run('Go.', workspace=Workspace(LocalWorkspaceBackend(tmp_path), mounts=[exports]))

    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions is not None
    assert (
        'Mounted storage in this workspace:\n'
        '- `exports` (read-only; file tools only, shell commands do not see it): Daily exports'
    ) in request.instructions


async def test_mounts_of_a_read_only_workspace_are_read_only_to_the_model(tmp_path: Path) -> None:
    exports = Mount('exports', Bucket('memory:///'))
    agent = Agent(
        TestModel(call_tools=[]),
        capabilities=[LocalWorkspace(tmp_path, mounts=[exports], read_only=True), FileSystem()],
    )

    result = await agent.run('Go.')

    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert '- `exports` (read-only; file tools only, shell commands do not see it)' in (request.instructions or '')
