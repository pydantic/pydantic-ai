"""The `workspaces` plugin's settings: named kinds of workspace, each a backend with its options."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Teardown = Literal['remove', 'keep']
"""What destroying a workspace does to its checkout: remove it, or keep it on disk."""

KindName = Annotated[str, StringConstraints(pattern=r'^[a-z][a-z0-9-]{0,31}$')]
"""A kind's name, as `/workspaces create KIND` takes it."""

LOCAL_WORKTREE = 'local-worktree'


class WorktreeKind(BaseModel):
    """A Git worktree on this machine, whose sessions run in this CLAI as any other does."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    backend: Literal['local-worktree'] = 'local-worktree'
    repo: str = Field(
        default='',
        description=(
            "The repository to use when none is given: a local path (relative to the session's folder), "
            'or a URL to clone. Empty: always give one.'
        ),
    )
    ref: str = Field(
        default='',
        description="The branch, tag, or commit to start from when none is given. Empty: the repository's HEAD.",
    )
    root: str = Field(
        default='',
        description=(
            'The folder checkouts go in, as an absolute path. Empty: `.worktrees/` in the repository, '
            'where `clai2 --worktree` puts them.'
        ),
    )
    branch_prefix: str = Field(
        default='clai-',
        pattern=r'^[A-Za-z0-9][A-Za-z0-9._/-]{0,63}$',
        description="Each workspace's new branch is this prefix and the workspace's ID.",
    )
    teardown: Teardown = Field(
        default='remove',
        description='Destroying a workspace removes its checkout, uncommitted changes included, or keeps it on disk.',
    )
    model: str = Field(
        default='',
        description=(
            "Retired: the model of a workspace's own CLAI process. Its sessions run in this CLAI now and start with "
            'the saved default, so a value an older CLAI saved is kept for it but not used.'
        ),
    )

    @field_validator('branch_prefix')
    @classmethod
    def _branch(cls, value: str) -> str:
        # What `git check-ref-format` refuses inside a name; the pattern above covers the characters.
        if any(part in value for part in ('..', '//', '/.', '.lock/')):
            raise ValueError('Git refuses branch names with `..`, `//`, `/.`, or `.lock/`')
        return value

    @field_validator('root')
    @classmethod
    def _absolute(cls, value: str) -> str:
        if value and not Path(value).expanduser().is_absolute():
            raise ValueError('give an absolute path, or leave it empty')
        return value


def _default_kinds() -> dict[str, WorktreeKind]:
    return {LOCAL_WORKTREE: WorktreeKind()}


class WorkspacesSettings(BaseModel):
    """The kinds of workspace `/workspaces create` and the orchestrator can make, by name."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    kinds: dict[KindName, WorktreeKind] = Field(
        default_factory=_default_kinds, description='Kinds of workspace by name, each with its options.'
    )
