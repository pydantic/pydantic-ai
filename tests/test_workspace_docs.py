"""Regression checks for workspace recipes that cannot be executed without a harness install."""

from pathlib import Path


def test_wrapper_policy_documents_command_and_symlink_escape() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert '`run()` bypasses file-method policies' in page
    assert 'realpath' in page
    assert 'symlink' in page


def test_local_background_jobs_document_pipe_drain_grace() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert 'background jobs outlive' in page
    assert 'redirect their output' in page
    assert 'two-second drain' in page


def test_no_unavailable_no_file_access_recipe() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert "no_files = UnavailableWorkspace(reason='This run has no file access.')" not in page
    assert 'no-file-tools agent' in page


def test_post_run_unused_workspace_access_warning() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert 'accessing `result.workspace` after a run with `ref=None`' in page


def test_durable_workspace_call_arguments_are_journaled() -> None:
    docs = Path(__file__).resolve().parents[1] / 'docs' / 'durable_execution'
    for engine in ('temporal', 'dbos'):
        page = (docs / f'{engine}.md').read_text()
        assert 'workflow-side workspace call arguments' in page
        assert 'env=' in page


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
