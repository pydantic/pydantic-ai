from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import _commands
import pytest
from _command_fixtures import Recording, Replay
from _commands import scrub

MISSING_OBJECT = '0' * 40


def test_a_recorded_call_replays_as_the_real_command_ran(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = tmp_path / 'git.json'
    monkeypatch.setenv(_commands.RECORD_ENV, str(fixture))
    for name in ('GITHUB_SERVER_URL', 'GITHUB_REPOSITORY', 'GITHUB_RUN_ID'):
        monkeypatch.delenv(name, raising=False)

    version = _commands.run(['git', '--version'], check=True)
    missing = _commands.run(['git', 'cat-file', '-e', MISSING_OBJECT])
    with pytest.raises(subprocess.CalledProcessError):
        _commands.run(['git', 'cat-file', '-e', MISSING_OBJECT], check=True)

    replay = Replay(fixture)
    assert replay.recording.recorded_from.run == 'local'
    assert replay.run(['git', '--version']).stdout == version.stdout
    replayed = replay.run(['git', 'cat-file', '-e', MISSING_OBJECT])
    assert (replayed.returncode, replayed.stderr) == (missing.returncode, missing.stderr) != (0, '')
    with pytest.raises(subprocess.CalledProcessError):
        replay.run(['git', 'cat-file', '-e', MISSING_OBJECT], check=True)
    assert replay.unconsumed == []
    with pytest.raises(AssertionError, match='no unconsumed call `git --version`'):
        replay.run(['git', '--version'])


def test_input_must_match_the_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = tmp_path / 'hash.json'
    monkeypatch.setenv(_commands.RECORD_ENV, str(fixture))
    recorded = _commands.run(['git', 'hash-object', '--stdin'], input='a\n', check=True)

    replay = Replay(fixture)
    with pytest.raises(AssertionError):
        replay.run(['git', 'hash-object', '--stdin'], input='b\n')
    assert replay.run(['git', 'hash-object', '--stdin'], input='a\n').stdout == recorded.stdout


def test_bytes_keep_every_byte_through_the_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = tmp_path / 'bytes.json'
    monkeypatch.setenv(_commands.RECORD_ENV, str(fixture))
    text = _commands.run_bytes(['printf', r'a\r\nb'], check=True)
    binary = _commands.run_bytes(['printf', r'b\377'], check=True)
    assert (text.stdout, binary.stdout) == (b'a\r\nb', b'b\xff')

    replay = Replay(fixture)
    assert [call.encoding for call in replay.recording.calls] == ['utf-8', 'base64']
    assert replay.run_bytes(['printf', r'a\r\nb']).stdout == text.stdout
    assert replay.run_bytes(['printf', r'b\377']).stdout == binary.stdout


def test_the_header_names_the_actions_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = tmp_path / 'run.json'
    monkeypatch.setenv(_commands.RECORD_ENV, str(fixture))
    monkeypatch.setenv('GITHUB_SERVER_URL', 'https://github.com')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'pydantic/pydantic-ai')
    monkeypatch.setenv('GITHUB_RUN_ID', '123')
    _commands.run(['git', '--version'])

    recording = Recording.model_validate_json(fixture.read_text(encoding='utf-8'))
    assert recording.recorded_from.run == 'https://github.com/pydantic/pydantic-ai/actions/runs/123'
    assert len(recording.calls) == 1


def test_nothing_is_recorded_without_a_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_commands.RECORD_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    _commands.run(['git', '--version'])
    assert list(tmp_path.iterdir()) == []


def test_scrub_removes_credentials():
    token = 'ghp_' + 'a' * 36
    text = '\n'.join(
        [
            f'token {token} and github_pat_{"b" * 30}',
            'Authorization: Bearer abc.def',
            'by dev@example.org and 1+octo@users.noreply.github.com',
            'key zai-secret-value',
            'File "/home/dev/.cache/uv/griffe/loader.py"',
        ]
    )
    environ = {'ZAI_API_KEY': 'zai-secret-value', 'SHORT_TOKEN': 'abc', 'HOME': '/home/dev'}
    assert scrub(text, environ) == '\n'.join(
        [
            'token <github-token> and <github-token>',
            'Authorization: <redacted>',
            'by user@example.com and 1+octo@users.noreply.github.com',
            'key <ZAI_API_KEY>',
            'File "~/.cache/uv/griffe/loader.py"',
        ]
    )


def test_a_recorded_secret_never_reaches_the_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = tmp_path / 'secret.json'
    monkeypatch.setenv(_commands.RECORD_ENV, str(fixture))
    monkeypatch.setenv('GH_TOKEN', 'not-a-real-token-value')
    _commands.run(['git', 'hash-object', '--stdin'], input='not-a-real-token-value')

    assert 'not-a-real-token-value' not in fixture.read_text(encoding='utf-8')
    assert Replay(fixture).recording.calls[0].input == '<GH_TOKEN>'


def test_every_command_fixture_says_where_it_was_recorded_and_is_scrubbed():
    fixtures = sorted((Path(__file__).parent / 'fixtures' / 'commands').glob('*.json'))
    assert fixtures
    for fixture in fixtures:
        recording = Replay(fixture).recording
        assert recording.recorded_from.command, fixture.name
        assert recording.calls, fixture.name
        for call in recording.calls:
            texts = [*call.args, call.input or '', call.stderr, call.stdout if call.encoding == 'utf-8' else '']
            assert [scrub(text, {}) for text in texts] == texts, fixture.name


def test_the_seam_imports_only_the_standard_library():
    # Workflows run gate scripts on the runner's bare `python3`, which has no third-party packages.
    tree = ast.parse(Path(_commands.__file__).read_text(encoding='utf-8'))
    imported = {
        alias.name.split('.')[0]
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or '')])
    }
    assert imported <= sys.stdlib_module_names | {'__future__'}
