from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_status_error_catches

_REPO = Path(__file__).parents[1]
_ADAPTER = 'pydantic_ai_slim/pydantic_ai/models/provider.py'

_DEBT = """\
from openai import APIStatusError


async def send(client):
    try:
        return await client.send()
    except APIStatusError as e:
        raise ValueError(e.status_code) from e
"""

_UNPAIRED = """
async def other(client):
    try:
        return await client.other()
    except APIStatusError as e:
        raise ValueError(e.status_code) from e
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, _ADAPTER, _DEBT)
    _git(tmp_path, 'init', '--quiet')
    _commit(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(repo: Path, name: str, source: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def _commit(repo: Path) -> None:
    _git(repo, 'add', '--all')
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '--quiet', '-m', 'commit')


def _check(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = check_status_error_catches.main(list(args))
    return code, capsys.readouterr().out


def _staged(repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str) -> tuple[int, str]:
    _write(repo, name, source)
    _git(repo, 'add', '--all')
    return _check(capsys, '--staged')


def _flagged(out: str) -> list[str]:
    """The `path:line: handler` rows the report lists."""
    return [line.strip() for line in out.splitlines() if line.startswith('    ')]


def test_an_unchanged_tree_passes_silently(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_a_new_unpaired_handler_fails_and_says_what_to_do(repo: Path, capsys: pytest.CaptureFixture[str]):
    code, out = _staged(repo, capsys, _ADAPTER, _DEBT + _UNPAIRED)

    assert code == 1
    assert f'{_ADAPTER}: 1 -> 2 SDK status-error handlers without `HTTPStatusError`' in out
    assert _flagged(out) == [f'{_ADAPTER}:11: except APIStatusError']
    assert '`_model_errors.transport_status_error`' in out


@pytest.mark.parametrize(
    'handler',
    [
        'except APIStatusError',
        'except (APIStatusError, APIConnectionError)',
        'except openai.APIError',
        'except ApiError',
        'except SDKError',
        'except HfHubHTTPError',
        'except TypeSafeAPIError',
        'except errors.APIError',
        'except errors.ClientError',
        'except google.genai.errors.ServerError',
    ],
)
def test_each_sdk_status_error_counts(repo: Path, capsys: pytest.CaptureFixture[str], handler: str):
    source = f'def f():\n    try:\n        pass\n    {handler}:\n        pass\n'

    code, out = _staged(repo, capsys, 'pydantic_ai_slim/pydantic_ai/embeddings/new.py', source)

    assert code == 1, handler
    assert '0 -> 1' in out


@pytest.mark.parametrize(
    'source',
    [
        'try:\n    pass\nexcept APIStatusError:\n    pass\nexcept httpx2.HTTPStatusError:\n    pass\n',
        'try:\n    pass\nexcept (APIError, httpx.HTTPStatusError):\n    pass\n',
        'try:\n    pass\nexcept ClientError:\n    pass\n',
        'try:\n    pass\nexcept grpc.RpcError:\n    pass\n',
        'try:\n    pass\nexcept:\n    pass\n',
        'with suppress(APIStatusError, HTTPStatusError):\n    pass\n',
        'with suppress(KeyError):\n    pass\n',
    ],
)
def test_a_paired_or_unrelated_handler_does_not_count(repo: Path, capsys: pytest.CaptureFixture[str], source: str):
    assert _staged(repo, capsys, 'pydantic_ai_slim/pydantic_ai/images/new.py', source) == (0, '')


def test_a_suppress_naming_a_status_error_counts(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = 'from contextlib import suppress\n\nwith suppress(APIStatusError):\n    pass\n'

    code, out = _staged(repo, capsys, 'pydantic_ai_slim/pydantic_ai/models/new.py', source)

    assert code == 1
    assert _flagged(out) == ['pydantic_ai_slim/pydantic_ai/models/new.py:3: suppress(APIStatusError)']


def test_debt_that_holds_or_shrinks_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, _ADAPTER, _DEBT.replace('APIStatusError', 'KeyError')) == (0, '')


def test_moving_a_file_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _git(repo, 'mv', _ADAPTER, 'pydantic_ai_slim/pydantic_ai/models/renamed.py')

    assert _check(capsys, '--staged') == (0, '')


def test_errors_modules_and_files_outside_the_adapters_are_skipped(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pydantic_ai_slim/pydantic_ai/models/openai/_errors.py', _DEBT + _UNPAIRED)
    _write(repo, 'pydantic_ai_slim/pydantic_ai/realtime/openai.py', _DEBT + _UNPAIRED)
    _write(repo, 'pydantic_ai_slim/pydantic_ai/models/broken.py', 'def (:\n')

    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_staged_mode_reads_the_index_not_the_working_tree(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, _ADAPTER, _DEBT + _UNPAIRED)

    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD')[0] == 1


def _merge_in(repo: Path, theirs: str) -> None:
    """Commit `theirs` as the adapter on another branch, then start merging it into this one."""
    _git(repo, 'checkout', '--quiet', '-b', 'other')
    _write(repo, _ADAPTER, theirs)
    _commit(repo)
    _git(repo, 'checkout', '--quiet', '-')
    _write(repo, 'pydantic_ai_slim/pydantic_ai/models/mine.py', '')
    _commit(repo)
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'merge', '--no-commit', '--no-ff', 'other')


def test_a_merge_is_not_blamed_for_what_the_merged_branch_added(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _DEBT + _UNPAIRED)

    assert _check(capsys, '--staged') == (0, '')


def test_a_merge_that_adds_more_than_both_parents_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _DEBT + _UNPAIRED)

    code, out = _staged(repo, capsys, _ADAPTER, _DEBT + _UNPAIRED + _UNPAIRED.replace('other', 'third'))

    assert code == 1
    assert f'{_ADAPTER}: 2 -> 3' in out


def test_github_format_annotates_each_added_handler(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, _ADAPTER, _DEBT + _UNPAIRED)

    code, out = _check(capsys, '--base', 'HEAD', '--format', 'github')

    assert code == 1
    annotation = next(line for line in out.splitlines() if line.startswith('::error'))
    assert annotation.startswith(f'::error file={_ADAPTER},line=11,title=Status-error handlers 1 -> 2::')


def test_the_roots_exist():
    for path in check_status_error_catches.ROOTS:
        assert (_REPO / path).is_dir(), path
