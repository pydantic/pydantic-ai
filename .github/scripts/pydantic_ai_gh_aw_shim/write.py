"""Claude's `Write` tool -- create or overwrite a workspace text file.

Backed by pydantic-ai-harness's `FileSystemToolset.write_file` (path
containment, symlink resolution). Claude's `Write` creates missing parent
directories, and the harness `write_file` requires the parent to exist, so the
adapter calls `create_directory` first to keep that behavior.
"""

import os

from pydantic_ai.exceptions import ModelRetry

from ._backends import filesystem
from .shared import attach_context


async def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file under the workspace."""
    fs = filesystem()
    parent = os.path.dirname(file_path)
    try:
        if parent:
            await fs.create_directory(parent)
        result = await fs.write_file(file_path, content)
    except (ModelRetry, OSError) as exc:
        # The harness converts its own recoverable errors to `ModelRetry`, but
        # `create_directory` -> `Path.mkdir(exist_ok=True)` still raises a bare
        # `FileExistsError` when a path segment is an existing file. The old
        # hand-rolled `Write` caught `OSError`, so keep doing that: a bad path is
        # a returned error, not a run-aborting exception.
        return f'error: {exc}'
    return attach_context(file_path) + result
