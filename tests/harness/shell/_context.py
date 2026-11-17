"""A run context for calling a shell toolset's tools directly."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace


def run_context(working_dir: Path) -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        workspace=Workspace(LocalWorkspaceBackend(working_dir)),
    )
