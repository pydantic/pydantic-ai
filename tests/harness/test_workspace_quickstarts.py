"""The first Shell and FileSystem examples work in a fresh project."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai.workspaces import WorkspaceRef

from ._docs_examples import python_blocks, run_block


@pytest.mark.parametrize(
    'page',
    [
        'docs/harness/shell.md',
        'docs/harness/filesystem.md',
        'src/pydantic_ai_harness/pydantic_ai_harness/shell/README.md',
        'src/pydantic_ai_harness/pydantic_ai_harness/filesystem/README.md',
    ],
)
def test_first_example_creates_its_workspace(page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    example = python_blocks(page)[0]
    monkeypatch.chdir(tmp_path)

    async def cleanup(ref: WorkspaceRef) -> None:
        pass

    _, runs = run_block(example, cleanup=cleanup)
    assert len(runs) == 1
    assert runs[0].ref is not None
    assert runs[0].outputs
    assert (tmp_path / 'workspace').is_dir()
