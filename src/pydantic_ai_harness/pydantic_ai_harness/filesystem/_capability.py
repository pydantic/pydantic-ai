"""Filesystem capability that provides sandboxed file system access."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.filesystem._toolset import FileSystemToolset

_DEFAULT_PROTECTED: list[str] = [
    '.git/*',
    '.env',
    '.env.*',
    '*.pem',
    '*.key',
    '**/secrets*',
]


@dataclass
class FileSystem(AbstractCapability[Any]):
    """Capability that provides file system access scoped to a root directory.

    All paths supplied by the model are resolved relative to `root_dir`.
    Traversal above the root is rejected. Symlinks are resolved before
    authorization to prevent escape via symlink.

    Security features:
    - Path traversal prevention (canonical path resolution)
    - Symlink-aware containment checks
    - Glob-based allow/deny filtering
    - Protected path patterns (secrets, keys, .git by default)
    - Binary file detection
    - Optimistic concurrency via content hashing

    Example::

        from pydantic_ai import Agent
        from pydantic_ai_harness.filesystem import FileSystem

        agent = Agent('openai:gpt-4o', capabilities=[FileSystem(root_dir='.')])
    """

    root_dir: str | Path = '.'
    """Root directory for all file operations. Defaults to the current directory."""

    allowed_patterns: Sequence[str] = field(default_factory=lambda: list[str]())
    """If non-empty, only paths matching at least one glob pattern are accessible."""

    denied_patterns: Sequence[str] = field(default_factory=lambda: list[str]())
    """Paths matching any of these glob patterns are rejected."""

    protected_patterns: Sequence[str] = field(default_factory=lambda: list(_DEFAULT_PROTECTED))
    """Paths matching these patterns are read-only (writes are rejected).

    Defaults to protecting `.git/`, `.env`, key files, and secrets.
    Set to an empty list to disable protection.
    """

    max_read_lines: int = 2000
    """Maximum number of lines returned by a single `read_file` call."""

    max_search_results: int = 1000
    """Maximum number of matches returned by `search_files`."""

    max_find_results: int = 1000
    """Maximum number of matches returned by `find_files`."""

    def get_toolset(self) -> AgentToolset[Any] | None:
        """Build and return the filesystem toolset."""
        return FileSystemToolset(
            root_dir=Path(self.root_dir),
            allowed_patterns=self.allowed_patterns,
            denied_patterns=self.denied_patterns,
            protected_patterns=self.protected_patterns,
            max_read_lines=self.max_read_lines,
            max_search_results=self.max_search_results,
            max_find_results=self.max_find_results,
        )
