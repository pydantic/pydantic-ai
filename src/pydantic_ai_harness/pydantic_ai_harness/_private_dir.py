"""Keep a store's directory on this machine reachable by its owner only."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def check_private_dir(path: Path, label: str, purpose: str) -> None:
    """Refuse `path` when it is a symlink or another user owns it, then make it owner-only (`0700`).

    `label` names the directory in the error, and `purpose` says what the ownership check refused.
    """
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        raise PermissionError(f'{label} {str(path)!r} is a symbolic link; refusing to use it.')
    geteuid = getattr(os, 'geteuid', None)
    if geteuid is None:
        return
    if st.st_uid != geteuid():
        raise PermissionError(f'{label} {str(path)!r} is not owned by the current user; refusing to {purpose}.')
    if stat.S_IMODE(st.st_mode) & 0o077:
        path.chmod(0o700)
