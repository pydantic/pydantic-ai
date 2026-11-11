"""Prompt recall stored separately from preferences and model messages."""

import os
from pathlib import Path

from prompt_toolkit.history import FileHistory


def input_history(path: Path) -> FileHistory:
    """Create an owner-readable history file before prompt-toolkit appends to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.close(descriptor)
    path.chmod(0o600)
    return FileHistory(str(path))
