"""Regression checks for workspace recipes that cannot be executed without a harness install."""

from pathlib import Path


def test_no_unavailable_no_file_access_recipe() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert "no_files = UnavailableWorkspace(reason='This run has no file access.')" not in page
    assert 'no-file-tools agent' in page


def test_post_run_unused_workspace_access_warning() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert 'accessing `result.workspace` after a run with `ref=None`' in page


def test_sandbox_paths_are_portable() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert 'Prefer relative paths' in page


def test_timeout_security_and_platform_guidance() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert '## Timeouts and clocks' in page
    assert '## Security choices' in page
    assert '## Platforms' in page
    assert 'stdin at EOF' in page
    assert '`timeout=None`' in page
    assert 'durable history' in page


def test_deleted_sandbox_history_restarts_with_new_workspace() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert "delete the sandbox after each run, pass `workspace='new'`" in page
