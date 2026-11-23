from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_shared_parsers

_MODELS = 'pydantic_ai_slim/pydantic_ai/models'

_DEBT = """\
def map_usage(usage_data, RequestUsage):
    return RequestUsage(details={k: v for k, v in usage_data.items() if isinstance(v, int)})
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, f'{_MODELS}/debt.py', _DEBT)
    _write(tmp_path, 'outside/loose.py', 'from email.utils import parsedate_to_datetime\nparsedate_to_datetime("x")\n')
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


def _staged(repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str) -> tuple[int, str]:
    _write(repo, name, source)
    _git(repo, 'add', '--all')
    code = check_shared_parsers.main(['--staged'])
    return code, capsys.readouterr().out


def _flagged(out: str) -> list[str]:
    """The `path:line: source` rows the report lists."""
    return [line.strip() for line in out.splitlines() if line.startswith('    ')]


def test_an_unchanged_tree_passes_silently(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert check_shared_parsers.main(['--staged']) == 0
    assert check_shared_parsers.main(['--base', 'HEAD']) == 0
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize(
    ('name', 'source', 'rule'),
    [
        pytest.param(
            'pydantic_ai_slim/pydantic_ai/retries.py',
            'from email.utils import parsedate_to_datetime\n\n\ndef wait(value):\n    return parsedate_to_datetime(value)\n',
            'retry-after',
            id='retry-after-date-outside-exceptions',
        ),
        pytest.param(
            'src/pkg/fetch.py',
            'import email.utils\n\n\ndef wait(value):\n    return email.utils.parsedate_to_datetime(value)\n',
            'retry-after',
            id='retry-after-date-by-module',
        ),
        pytest.param(
            'pydantic_ai_slim/pydantic_ai/download.py',
            "def kind(response):\n    return response.headers.get('content-type', '').split(';')[0].strip()\n",
            'media-type',
            id='content-type-header-split',
        ),
        pytest.param(
            'pydantic_evals/pkg/media.py',
            "def kind(media_type):\n    return media_type.partition(';')[0]\n",
            'media-type',
            id='media-type-partition',
        ),
        pytest.param(
            'src/pkg/mime.py',
            "def kind(mime):\n    return mime.rsplit(';', 1)[0]\n",
            'media-type',
            id='mime-rsplit',
        ),
        pytest.param(
            f'{_MODELS}/provider.py',
            'def map_usage(data):\n    return RequestUsage(details={k: v for k, v in data.items() if isinstance(v, int)})\n',
            'usage-count',
            id='usage-int-filter',
        ),
        pytest.param(
            f'{_MODELS}/provider.py',
            'def map_usage(data):\n    if isinstance(value := data.get("n"), int | float):\n        return RequestUsage(n=int(value))\n',
            'usage-count',
            id='usage-int-or-float-filter',
        ),
        pytest.param(
            'pydantic_ai_slim/pydantic_ai/embeddings/provider.py',
            'def map_usage(data):\n    return RequestUsage(details={k: v for k, v in data.items() if isinstance(v, int)})\n',
            'usage-count',
            id='embeddings-usage-int-filter',
        ),
    ],
)
def test_a_hand_written_parse_fails_and_says_what_to_do(
    repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str, rule: str
):
    code, out = _staged(repo, capsys, name, source)

    assert code == 1
    assert f'{name}: {rule} 0 -> 1' in out
    assert len(_flagged(out)) == 1
    assert f'{rule}: ' in out


@pytest.mark.parametrize(
    ('name', 'source'),
    [
        pytest.param(
            'pydantic_ai_slim/pydantic_ai/exceptions.py',
            'from email.utils import parsedate_to_datetime\n\n\ndef parse(value):\n    return parsedate_to_datetime(value)\n',
            id='retry-after-date-in-its-home',
        ),
        pytest.param(
            'pydantic_ai_slim/pydantic_ai/_utils.py',
            "def bare_media_type(media_type):\n    return media_type.split(';', 1)[0].strip()\n",
            id='media-type-split-in-its-home',
        ),
        pytest.param(
            'pydantic_ai_slim/pydantic_ai/_utils.py',
            "def kind(media_type):\n    return media_type.split('/', 1)[0]\n",
            id='media-type-split-on-another-separator',
        ),
        pytest.param(
            'pydantic_ai_slim/pydantic_ai/workspaces/windows.py',
            "def dirs(env):\n    return env.get('PATH', '').split(';')\n",
            id='split-of-a-value-that-is-no-media-type',
        ),
        pytest.param(
            f'{_MODELS}/provider.py',
            'def map_usage(data):\n'
            '    return RequestUsage(details={k: v for k, v in data.items() if isinstance(v, int) and not isinstance(v, bool)})\n',
            id='usage-int-filter-excluding-bool',
        ),
        pytest.param(
            f'{_MODELS}/provider.py',
            'def status(error):\n    return error.status if isinstance(error.status, int) else None\n',
            id='int-check-outside-a-usage-builder',
        ),
        pytest.param(
            'pydantic_ai_slim/pydantic_ai/usage.py',
            'def merge(value):\n    return RequestUsage(n=value) if isinstance(value, int) else None\n',
            id='int-check-outside-the-usage-mappers',
        ),
    ],
)
def test_the_shared_form_passes(repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str):
    assert _staged(repo, capsys, name, source) == (0, '')


def test_existing_debt_passes_until_a_change_adds_to_it(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, f'{_MODELS}/debt.py', _DEBT + '\n\nUNRELATED = 1\n') == (0, '')

    code, out = _staged(
        repo,
        capsys,
        f'{_MODELS}/debt.py',
        _DEBT
        + '\n\ndef map_more(data, RequestUsage):\n    return RequestUsage(n=1) if isinstance(data, int) else None\n',
    )
    assert code == 1
    assert f'{_MODELS}/debt.py: usage-count 1 -> 2' in out
    assert _flagged(out) == [
        f'{_MODELS}/debt.py:6: return RequestUsage(n=1) if isinstance(data, int) else None',
    ]


def test_files_outside_the_roots_are_not_checked(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = 'from email.utils import parsedate_to_datetime\nparsedate_to_datetime("y")\nparsedate_to_datetime("z")\n'
    assert _staged(repo, capsys, 'outside/loose.py', source) == (0, '')


def test_base_mode_annotates_for_github(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'src/pkg/fetch.py', "def kind(content_type):\n    return content_type.split(';')[0]\n")

    code = check_shared_parsers.main(['--base', 'HEAD', '--format', 'github'])
    out = capsys.readouterr().out

    assert code == 1
    assert '::error file=src/pkg/fetch.py,line=2,title=Shared parsers%3A media-type 0 -> 1::' in out
