from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _ratchet import Change
from check_command_stubs import Growth, growth, hand_written_outputs, main

_STUBS = """\
import subprocess
from subprocess import CompletedProcess


def fake_run(args):
    subprocess.CompletedProcess(args, 0, '[{"number": 1}]', '')
    CompletedProcess(args, 1, '', 'gh: Not Found (HTTP 404)')
    CompletedProcess(args, 0, stdout=f'open at {args[0]}')
    CompletedProcess(args, 0, b'bytes', b'')
    CompletedProcess(args, 0, '', '')
    CompletedProcess(args, 1)
    CompletedProcess(args, 0, recorded.stdout, recorded.stderr)
    output = json.dumps([{'number': 1}])
    CompletedProcess(args, 0, output, '')
    CompletedProcess(args, 0, stdout=json.dumps({'state': 'open'}))
    CompletedProcess(args, 0, None, None)
"""


def test_hand_written_outputs_are_counted_and_empty_or_recorded_ones_are_not():
    assert hand_written_outputs(_STUBS) == [6, 7, 8, 9, 14, 15]
    assert hand_written_outputs(None) == []


def test_only_a_gate_test_whose_count_grew_is_reported():
    one = "from subprocess import CompletedProcess\nCompletedProcess([], 0, 'x', '')\n"
    two = one + "CompletedProcess([], 0, 'y', '')\n"
    changes = [
        Change('.github/scripts/test_a.py', '.github/scripts/test_a.py', one, two),
        Change('.github/scripts/test_b.py', '.github/scripts/test_b.py', two, one),
        Change(None, '.github/scripts/test_new.py', None, one),
        Change(None, '.github/scripts/helper.py', None, two),
        Change('.github/scripts/test_gone.py', None, two, None),
    ]

    assert growth(changes) == [
        Growth('.github/scripts/test_a.py', 1, 2, [2, 3]),
        Growth('.github/scripts/test_new.py', 0, 1, [2]),
    ]


def test_main_reports_growth_in_a_scratch_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    test = tmp_path / '.github' / 'scripts' / 'test_gate.py'
    test.parent.mkdir(parents=True)
    test.write_text("CompletedProcess([], 0, 'x', '')\n", encoding='utf-8')
    git = ['git', '-c', 'user.name=test', '-c', 'user.email=test@example.com']
    for args in (['init', '--quiet'], ['add', '--all'], ['commit', '--quiet', '-m', 'base']):
        subprocess.run([*git, *args], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    assert main(['--base', 'HEAD']) == 0

    test.write_text("CompletedProcess([], 0, 'x', '')\nCompletedProcess([], 1, '', 'y')\n", encoding='utf-8')
    assert main(['--base', 'HEAD', '--format', 'github']) == 1
    assert capsys.readouterr().out.startswith('::error file=.github/scripts/test_gate.py,line=1::')
    subprocess.run([*git, 'add', '--all'], cwd=tmp_path, check=True)
    assert main(['--staged']) == 1
    assert 'grew from 1 to 2' in capsys.readouterr().out
