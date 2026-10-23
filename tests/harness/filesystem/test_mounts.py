"""File tools in a workspace with mounted filesystems that commands can't see."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip('obstore')

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, Mount, Workspace
from pydantic_ai_harness.filesystem import FileSystem

from ...test_workspace_mounts import MountingWorkspace
from ...workspace_fakes import MemoryFilesystem
from .._tool_calls import call_tools
from ..cloud_storage.memory_bucket import MemoryBucket

pytestmark = pytest.mark.anyio


async def _workspace(tmp_path: Path) -> tuple[Workspace, MemoryBucket]:
    bucket = MemoryBucket()
    await bucket.write_bytes('/2026/refunds.csv', b'region,refund\neu,12\n')
    (tmp_path / 'notes.txt').write_text('refund policy: 30 days\n')
    return Workspace(LocalWorkspaceBackend(tmp_path), mounts=[Mount('data', bucket)]), bucket


async def test_searches_from_above_and_inside_a_mount_reach_it(tmp_path: Path) -> None:
    workspace, _ = await _workspace(tmp_path)

    grep, search, listing, inside = await call_tools(
        [FileSystem[None](tools=['grep', 'search_files', 'list_files'])],
        [
            ('grep', {'pattern': 'refund'}),
            ('search_files', {'pattern': 'refund'}),
            ('list_files', {}),
            ('grep', {'pattern': 'eu', 'path': 'data/2026'}),
        ],
        workspace=workspace,
    )

    for found in (grep, search):
        assert 'data/2026/refunds.csv' in found
        assert 'notes.txt' in found
    assert listing.splitlines() == ['data/2026/refunds.csv', 'notes.txt']
    assert 'refunds.csv' in inside


async def test_a_symlink_into_a_mount_is_refused(tmp_path: Path) -> None:
    workspace, bucket = await _workspace(tmp_path)
    # A folder of the host hidden by the mount, and a link to it, which the host resolves without the mount.
    (tmp_path / 'data').mkdir()
    (tmp_path / 'alias').symlink_to('data')

    written, refused = await call_tools(
        [FileSystem[None]()],
        [
            ('write_file', {'path': 'data/summary.md', 'content': '# EU refunds: 12'}),
            ('write_file', {'path': 'alias/summary.md', 'content': 'elsewhere'}),
        ],
        workspace=workspace,
    )

    assert 'summary.md' in written
    assert 'leads into or out of a mounted folder through a symlink' in refused
    assert await bucket.read_bytes('/summary.md') == b'# EU refunds: 12'
    assert list((tmp_path / 'data').iterdir()) == []


async def test_the_model_is_told_about_mounts(tmp_path: Path) -> None:
    exports = Mount('exports', MemoryBucket(), read_only=True, description='Daily exports')
    agent = Agent(TestModel(call_tools=[]), capabilities=[FileSystem()])

    result = await agent.run('Go.', workspace=Workspace(LocalWorkspaceBackend(tmp_path), mounts=[exports]))

    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions is not None
    assert (
        'Mounted folders in this workspace:\n'
        '- `exports` (read-only; file tools only, commands do not see it): Daily exports'
    ) in request.instructions


async def test_the_model_is_told_which_mounts_commands_see() -> None:
    agent = Agent(TestModel(call_tools=[]), capabilities=[FileSystem()])
    seen = Mount('seen', MemoryFilesystem(), commands=True)

    result = await agent.run('Go.', workspace=Workspace(MountingWorkspace(), mounts=[seen]))

    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert '- `seen` (read-write; commands see it too)' in (request.instructions or '')


async def test_mounts_of_a_read_only_workspace_are_read_only_to_the_model(tmp_path: Path) -> None:
    exports = Mount('exports', MemoryBucket())
    agent = Agent(
        TestModel(call_tools=[]),
        capabilities=[LocalWorkspace(tmp_path, mounts=[exports], read_only=True), FileSystem()],
    )

    result = await agent.run('Go.')

    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    # A read-only workspace runs no commands, so there is no shell to mention.
    assert 'Mounted folders in this workspace:\n- `exports` (read-only)' in (request.instructions or '')


async def test_without_a_shell_the_model_is_not_told_about_commands() -> None:
    agent = Agent(TestModel(call_tools=[]), capabilities=[FileSystem()])

    result = await agent.run('Go.', workspace=Workspace(MemoryFilesystem(), mounts=[Mount('data', MemoryBucket())]))

    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert 'Mounted folders in this workspace:\n- `data` (read-write)' in (request.instructions or '')
