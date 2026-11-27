"""Pin the instruction files a user accepted, so a later change to one gets a lower `authority_of`."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from pydantic_ai.messages import Authority
from pydantic_ai.workspaces import AuthorityWorkspace, Workspace
from pydantic_ai.workspaces._authority_paths import lowest_authority


async def file_digest(workspace: Workspace, path: str) -> str:
    """The SHA-256 hex digest of the file at `path`, read through `workspace`.

    Record it when the user accepts the file, and pass it to `pin_accepted_files` on later runs.
    """
    return hashlib.sha256(await workspace.read_bytes(path)).hexdigest()


async def pin_accepted_files(
    workspace: Workspace, accepted: Mapping[str, str], *, changed: Authority = 'none'
) -> Workspace:
    """`workspace`, with each file in `accepted` whose content no longer matches its digest capped at `changed`.

    `accepted` maps workspace paths, such as `'AGENTS.md'`, `'.agents/skills/review/SKILL.md'` or
    `'.claude/agents/reviewer.md'`, to the [`file_digest`][pydantic_ai_harness.repo_context.file_digest] recorded when
    the user accepted them. A file that still matches keeps the level the workspace gives it. A changed file, or one
    that is no longer a file (so a deleted file recreated later is capped too), gets `changed`. A cap never raises a
    file above the level the workspace gives it, so a file outside the working directory stays at `'none'`. Files not
    in `accepted` are untouched. The capped files are wrapped in an
    [`AuthorityWorkspace`][pydantic_ai.workspaces.AuthorityWorkspace] `path_authority`, so
    [`authority_of`][pydantic_ai.workspaces.Workspace.authority_of] reports the lower level until the user accepts the
    new content and the app records its digest.

    Files are compared when this is called, so call it before each run and pass the result as the run's workspace.
    """
    path_authority: dict[str, Authority] = {}
    for path, digest in accepted.items():
        try:
            current: str | None = await file_digest(workspace, path)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            current = None
        if current != digest:
            path_authority[path] = lowest_authority((changed, await workspace.authority_of(path)))
    if not path_authority:
        return workspace
    return AuthorityWorkspace(workspace, authority='operator', path_authority=path_authority)
