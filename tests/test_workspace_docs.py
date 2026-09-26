"""Regression checks for workspace recipes that cannot be executed without a harness install."""

from pathlib import Path


def test_no_unavailable_no_file_access_recipe() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert "no_files = UnavailableWorkspace(reason='This run has no file access.')" not in page
    assert 'no-file-tools agent' in page


def test_deleted_sandbox_history_restarts_with_new_workspace() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert "delete the sandbox after each run, pass `workspace='new'`" in page
