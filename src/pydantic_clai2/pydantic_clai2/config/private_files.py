"""Atomically replace a private state file."""

import os
from pathlib import Path
from uuid import uuid4


def write_private(path: Path, value: str, *, replace: bool = True) -> None:
    """Publish a private file atomically, without writing through an existing path.

    A unique staging name with `O_EXCL` refuses a planted symlink instead of following it and keeps
    concurrent writers from sharing a staging file. Mode `0600` applies to the file this call creates.
    `replace=False` raises `FileExistsError` instead of overwriting a file another writer published first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            file.write(value)
        if replace:
            os.replace(staging, path)
        else:
            os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)
