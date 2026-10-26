"""Atomically replace a private state file."""

import os
import secrets
from pathlib import Path


def write_private(path: Path, text: str) -> None:
    """Replace `path` with `text` in a file only you can read, through a fresh temporary file renamed over it."""
    temporary = path.with_name(f'.{path.name}.{secrets.token_hex(8)}.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            file.write(text)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
