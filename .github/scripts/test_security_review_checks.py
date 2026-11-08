"""Select Security Review checks and the files each one reviews."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from security_review_checks import CHECKS, ChangedFile, SecurityCheck, changed_files, select

PROMPTS = Path(__file__).parents[1] / 'security-review' / 'prompts'


def _file(path: str, *lines: str, previous: str = '') -> ChangedFile:
    return ChangedFile(path, previous, lines)


def test_every_check_has_a_prompt_fragment() -> None:
    """A config entry without its prompt would start a review with no instructions."""
    assert sorted(path.stem for path in PROMPTS.glob('*.md') if path.stem not in {'shared', 'security'}) == sorted(
        check.name for check in CHECKS
    )


@pytest.mark.parametrize('check', CHECKS, ids=lambda check: check.name)
def test_every_check_has_a_budget(check: SecurityCheck) -> None:
    """A zero step or minute budget fails the check on every pull request."""
    assert check.max_steps > 0
    assert check.max_minutes > 0


def test_content_only_change_selects_nothing() -> None:
    """Docs and agent instructions need no security review, so the status passes at once."""
    files = [_file('docs/agent.md'), _file('THREAT_MODEL.md'), _file('src/pydantic_ai_harness/AGENTS.md')]
    assert select(files) == {}


@pytest.mark.parametrize(
    'path',
    [
        'tests/test_agent.py',
        'tests/models/cassettes/test_x/y.yaml',
        'uv.lock',
        'pydantic_ai_slim/pyproject.toml',
        '.github/scripts/pydantic-ai-runner.lock',
        'tests/ui_roundtrip/package-lock.json',
    ],
)
def test_tests_and_dependencies_check_owns_tests_and_locks(path: str) -> None:
    """A tests-only or dependency-only pull request is reviewed against the testing and dependency rules."""
    assert select([_file(path)]) == {'tests-and-dependencies': [path]}


def test_compiled_workflow_lock_selects_nothing() -> None:
    """The general check reviews the workflow source the lock is compiled from."""
    assert select([_file('.github/workflows/ci.lock.yml')]) == {}


def test_general_reviews_source_and_workflows() -> None:
    """Library code and repository CI both get the general review."""
    files = [_file('pydantic_ai_slim/pydantic_ai/agent/__init__.py'), _file('.github/workflows/bots.yml')]
    assert select(files) == {'general': [file.path for file in files]}


def test_specialist_owns_its_files_and_general_reviews_the_rest() -> None:
    """Each check sees only the files that selected it; the general check takes no owned file."""
    files = [
        _file('pydantic_ai_slim/pydantic_ai/_ssrf.py'),
        _file('pydantic_ai_slim/pydantic_ai/agent/__init__.py'),
        _file('src/pydantic_ai_harness/pydantic_ai_harness/code_mode/_toolset.py'),
    ]
    assert select(files) == {
        'general': ['pydantic_ai_slim/pydantic_ai/agent/__init__.py'],
        'url-fetch': ['pydantic_ai_slim/pydantic_ai/_ssrf.py'],
        'sandboxes': ['src/pydantic_ai_harness/pydantic_ai_harness/code_mode/_toolset.py'],
    }


def test_rename_selects_by_previous_path() -> None:
    """Moving a file out of a check's paths still gets the check's review."""
    moved = _file('pydantic_ai_slim/pydantic_ai/_net.py', previous='pydantic_ai_slim/pydantic_ai/_ssrf.py')
    assert select([moved])['url-fetch'] == ['pydantic_ai_slim/pydantic_ai/_net.py']


@pytest.mark.parametrize(
    ('path', 'line', 'check'),
    [
        ('pydantic_ai_slim/pydantic_ai/models/google.py', '    data = await download_item(item)', 'url-fetch'),
        ('src/pydantic_ai_harness/pydantic_ai_harness/media/_x.py', "force_download='allow-local'", 'url-fetch'),
        ('pydantic_ai_slim/pydantic_ai/messages.py', '    def otel_events(self):', 'telemetry'),
        ('pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_x.py', 'span.set_attribute(k, v)', 'telemetry'),
    ],
)
def test_diff_token_selects_a_file_outside_the_paths(path: str, line: str, check: str) -> None:
    """A changed line, not the file's location, decides."""
    assert select([_file(path, line)])[check] == [path]


def test_token_selected_file_stays_with_the_general_check() -> None:
    """A diff token adds a reviewer; only path ownership takes a file from the general check."""
    path = 'pydantic_ai_slim/pydantic_ai/models/google.py'
    assert select([_file(path, 'await download_item(item)')]) == {'general': [path], 'url-fetch': [path]}


@pytest.mark.parametrize(
    ('path', 'line'),
    [
        ('pydantic_ai_slim/pydantic_ai/models/google.py', 'unrelated = 1'),
        ('pydantic_ai_slim/pydantic_ai/agent/__init__.py', 'span.set_attribute(k, v)'),
        ('docs/input.md', 'download_item('),
    ],
)
def test_diff_token_outside_its_token_paths_does_not_select(path: str, line: str) -> None:
    """A token in an unrelated file or a doc must not start a specialist review."""
    selection = select([_file(path, line)])
    assert 'url-fetch' not in selection
    assert 'telemetry' not in selection


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repository), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_changed_files_reads_renames_deletions_and_lines(tmp_path: Path) -> None:
    """Reads changes from git, including removed lines of a deleted file and lines that look like headers."""
    repository = tmp_path / 'repo'
    repository.mkdir()
    _git(repository, 'init', '-q', '-b', 'main')
    _git(repository, 'config', 'user.email', 'test@example.com')
    _git(repository, 'config', 'user.name', 'Test')
    (repository / 'old.py').write_text('a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n')
    (repository / 'gone.py').write_text('await safe_download(url)\n')
    (repository / 'kept.py').write_text('x = 1\n')
    _git(repository, 'add', '.')
    _git(repository, 'commit', '-q', '-m', 'base')
    base = _git(repository, 'rev-parse', 'HEAD')
    _git(repository, 'mv', 'old.py', 'new.py')
    (repository / 'gone.py').unlink()
    (repository / 'kept.py').write_text('x = 1\n++ b/fake\n')
    _git(repository, 'add', '-A')
    _git(repository, 'commit', '-q', '-m', 'head')

    files = {file.path: file for file in changed_files(repository, base, 'HEAD')}

    assert files['new.py'] == ChangedFile('new.py', 'old.py', ())
    assert files['gone.py'] == ChangedFile('gone.py', '', ('await safe_download(url)',))
    assert files['kept.py'] == ChangedFile('kept.py', '', ('++ b/fake',))
