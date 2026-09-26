"""Regression checks for workspace recipes that cannot be executed without a harness install."""

import subprocess
import sys
from pathlib import Path


def test_first_quickstart_runs_main(tmp_path: Path) -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    example = page.split('```python {title="workspace_agent.py"}', 1)[1].split('```', 1)[0]
    # Stub the model call so this checks the script entry point without credentials.
    setup = """
import sys
from types import ModuleType
from pathlib import Path
ai = ModuleType('pydantic_ai')
class Agent:
    def __init__(self, *args, **kwargs): pass
    def tool(self, fn): return fn
    async def run(self, prompt): Path('fizzbuzz.py').write_text('created')
ai.Agent = Agent
ai.ModelRetry = type('ModelRetry', (Exception,), {})
ai.RunContext = type('RunContext', (), {})
cap = ModuleType('pydantic_ai.capabilities')
cap.LocalWorkspace = lambda path: path
ws = ModuleType('pydantic_ai.workspaces')
ws.WorkspaceError = type('WorkspaceError', (Exception,), {})
sys.modules.update({'pydantic_ai': ai, 'pydantic_ai.capabilities': cap, 'pydantic_ai.workspaces': ws})
"""
    subprocess.run([sys.executable, '-c', setup + example], cwd=tmp_path, check=True)
    assert (tmp_path / 'fizzbuzz.py').read_text() == 'created'


def test_host_backend_serializes_first_use() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    example = page.split('```python {title="host_workspace.py"}', 1)[1].split('```', 1)[0]
    assert 'self._lock = anyio.Lock()' in example
    assert 'async with self._lock:' in example


def test_host_backend_rejects_untrusted_ref_id() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    example = page.split('```python {title="host_workspace.py"}', 1)[1].split('```', 1)[0]
    assert "re.fullmatch(r'[0-9a-f]{32}', self._ref.id)" in example
    assert 'ref ids can come from' in page.lower()


def test_ui_adapter_workspace_ref_round_trip_limit_and_approval_recipe() -> None:
    root = Path(__file__).resolve().parents[1]
    overview = (root / 'docs/ui/overview.md').read_text()
    approval = (root / 'docs/ui/vercel-ai.md').read_text()
    adapter = (root / 'pydantic_ai_slim/pydantic_ai/ui/_adapter.py').read_text()
    for text in (overview, adapter):
        assert 'Vercel AI and AG-UI protocols do not carry workspace references' in text
    assert 'result.workspace.ref' in approval
    assert 'workspace=' in approval.split('## Tool Approval', 1)[1].split('## Tool input validation', 1)[0]


def test_response_ref_timing_and_local_absolute_root_guidance() -> None:
    root = Path(__file__).resolve().parents[1]
    response = (root / 'pydantic_ai_slim/pydantic_ai/messages.py').read_text()
    docs = (root / 'docs/workspace.md').read_text()
    assert 'Each response records the ref when it is produced' in response
    assert 'Create the configured local root before use, including absolute file writes beneath it' in docs


def test_limits_explain_durable_run_start_creation_and_caller_cleanup() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    limits = page.split('## Limits', 1)[1]
    assert 'Non-durable runs do not create or delete sandboxes solely at run boundaries' in limits
    assert 'durable runs eagerly create or attach an environment at their start' in limits
    assert "deletion remains the caller's job" in limits


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


def test_temporal_command_retry_requires_explicit_policy() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert 'workspace_activity_config' in page
    assert 'command retries' in page


def test_deleted_sandbox_history_restarts_with_new_workspace() -> None:
    page = (Path(__file__).resolve().parents[1] / 'docs' / 'workspace.md').read_text()
    assert "delete the sandbox after each run, pass `workspace='new'`" in page
