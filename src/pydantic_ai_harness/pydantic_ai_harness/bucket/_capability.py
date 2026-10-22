"""Capability that makes a bucket the run's whole workspace."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import (
    FileEntry,
    ReadOnlyWorkspace,
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceRef,
)
from pydantic_ai_harness.bucket._bucket import Bucket


@dataclass
class BucketWorkspace(AbstractCapability[AgentDepsT]):
    """Make a bucket the run's workspace: file tools work in it, and there are no shell commands.

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai_harness import FileSystem
    from pydantic_ai_harness.bucket import Bucket, BucketWorkspace

    agent = Agent(
        'anthropic:claude-opus-5-5',
        capabilities=[BucketWorkspace(Bucket('s3://acme-docs/handbook/'), read_only=True), FileSystem()],
    )
    ```
    """

    bucket: Bucket
    """The bucket, or folder in one, that the workspace is."""

    read_only: bool = False
    """Whether to refuse changes."""

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        backend = BucketWorkspaceBackend(self.bucket)
        if ref is not None and ref != backend.ref:
            return None
        return ReadOnlyWorkspace(Workspace(backend)) if self.read_only else backend


class BucketWorkspaceBackend(WorkspaceBackend, SupportsFilesystem):
    """A workspace backend whose environment is a bucket: its working directory is the bucket's `/`."""

    def __init__(self, bucket: Bucket) -> None:
        self._bucket = bucket

    @property
    def ref(self) -> WorkspaceRef:
        return WorkspaceRef(provider='bucket', id=self._bucket.url)

    async def working_dir(self) -> str:
        return '/'

    async def read_bytes(self, path: str) -> bytes:
        return await self._bucket.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._bucket.write_bytes(path, data)

    async def stat(self, path: str) -> FileEntry:
        return await self._bucket.stat(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return await self._bucket.list_dir(path)

    async def make_dir(self, path: str) -> None:
        await self._bucket.make_dir(path)

    async def remove(self, path: str) -> None:
        await self._bucket.remove(path)

    async def exists(self, path: str) -> bool:
        return await self._bucket.exists(path)
