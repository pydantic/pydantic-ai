"""Verify the OpenCode threat detector records only verdicts `threat-detect conclude` accepts."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

import gh_aw_opencode_detect

CLEAN: dict[str, object] = {'prompt_injection': False, 'secret_leak': False, 'malicious_patch': False, 'reasons': []}


@pytest.fixture
def detection_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stage gh-aw's detection inputs, a rendered prompt, and a stale verdict from an earlier run."""
    directory = tmp_path / 'threat-detection'
    (directory / 'aw-prompts').mkdir(parents=True)
    (directory / 'agent_output.json').write_text('{"items":[]}')
    (directory / 'detection.log').touch()
    (directory / 'detection_result.json').write_text(json.dumps(CLEAN))
    (directory / 'detection_result_full.json').write_text(json.dumps(CLEAN))
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('# Threat Detection Analysis\nOutput format: THREAT_DETECTION_RESULT:{...}')
    runner_temp = tmp_path / 'runner-temp'
    runner_temp.mkdir()
    monkeypatch.setattr(gh_aw_opencode_detect, 'DETECTION_DIR', directory)
    monkeypatch.setenv('GH_AW_PROMPT', str(prompt))
    monkeypatch.setenv('RUNNER_TEMP', str(runner_temp))
    monkeypatch.setenv('OPENCODE_API_KEY', 'fake-key')
    monkeypatch.setenv('GH_TOKEN', 'fake-github-token')
    return directory


def answer(monkeypatch: pytest.MonkeyPatch, text: str) -> list[dict[str, object]]:
    """Make OpenCode stop normally with `text` as its final message, recording each launch."""
    invoked: list[dict[str, object]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        invoked.append({'args': args, **kwargs})
        stdout = kwargs['stdout']
        assert isinstance(stdout, io.TextIOBase)
        events = [
            {'type': 'text', 'part': {'messageID': 'msg_final', 'text': text}},
            {'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'stop'}},
        ]
        stdout.write('\n'.join(json.dumps(event) for event in events) + '\n')
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, 'run', fake_run)
    return invoked


def status(directory: Path) -> str:
    """Return the last status line the detector appended to `detection.log`."""
    return (directory / 'detection.log').read_text().splitlines()[-1]


def test_verdict_splits_reasons_into_the_full_result(detection_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The uploaded result carries no model text; `conclude` recovers reasons from the full result."""
    verdict = {**CLEAN, 'secret_leak': True, 'reasons': ['  A token is printed in the comment body.  ']}
    invoked = answer(monkeypatch, 'Analysis done.\nTHREAT_DETECTION_RESULT:' + json.dumps(verdict))

    gh_aw_opencode_detect.main()

    assert json.loads((detection_dir / 'detection_result.json').read_text()) == {**verdict, 'reasons': []}
    assert json.loads((detection_dir / 'detection_result_full.json').read_text()) == {
        **verdict,
        'reasons': ['A token is printed in the comment body.'],
    }
    assert status(detection_dir) == 'THREAT_DETECTION_STATUS: reason=result_recorded exit=0'
    [call] = invoked
    assert call['args'] == [
        'opencode',
        'run',
        '--standalone',
        '--model',
        'opencode-go/deepseek-v4.1-flash',
        '--agent',
        'detector',
        '--format',
        'json',
    ]
    assert call['cwd'] == detection_dir
    assert isinstance(call['input'], str)
    assert call['input'].startswith('# Threat Detection Analysis\n')
    assert 'repository workspace are unavailable' in call['input']
    env = TypeAdapter(dict[str, str]).validate_python(call['env'])
    assert env['OPENCODE_API_KEY'] == 'fake-key'
    assert 'GH_TOKEN' not in env


@pytest.mark.parametrize(
    'text',
    [
        json.dumps(CLEAN),
        'THREAT_DETECTION_RESULT:not json',
        'THREAT_DETECTION_RESULT:' + json.dumps({**CLEAN, 'prompt_injection': 'false'}),
        'THREAT_DETECTION_RESULT:' + json.dumps({**CLEAN, 'confidence': 'high'}),
        'THREAT_DETECTION_RESULT:' + json.dumps({key: value for key, value in CLEAN.items() if key != 'reasons'}),
        'THREAT_DETECTION_RESULT:' + json.dumps({**CLEAN, 'reasons': [' \n']}),
        'THREAT_DETECTION_RESULT:' + json.dumps({**CLEAN, 'reasons': ['x' * 2001]}),
        'THREAT_DETECTION_RESULT:' + json.dumps({**CLEAN, 'reasons': ['reason'] * 21}),
    ],
)
def test_invalid_report_concludes_as_parse_error(
    detection_dir: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """No verdict is written, and the status line maps the missing result to `parse_error`."""
    answer(monkeypatch, text.replace('"reasons": []', '"reasons": ["::error::injected"]', 1))

    with pytest.raises(SystemExit) as exc_info:
        gh_aw_opencode_detect.main()

    assert not (detection_dir / 'detection_result.json').exists()
    assert not (detection_dir / 'detection_result_full.json').exists()
    assert status(detection_dir) == 'THREAT_DETECTION_STATUS: reason=invalid_report_exhausted exit=1'
    assert 'injected' not in str(exc_info.value.code)


def test_engine_failure_concludes_as_agent_failure(detection_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenCode failure leaves no verdict and reports an engine error."""

    def failed_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, 'run', failed_run)

    with pytest.raises(SystemExit):
        gh_aw_opencode_detect.main()

    assert not (detection_dir / 'detection_result.json').exists()
    assert status(detection_dir) == 'THREAT_DETECTION_STATUS: reason=engine_error exit=2'


def test_escaping_symlink_stops_before_opencode_runs(detection_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenCode checks paths lexically, so a link out of the inputs directory would widen its reads."""
    invoked = answer(monkeypatch, 'THREAT_DETECTION_RESULT:' + json.dumps(CLEAN))
    (detection_dir / 'comment-memory').mkdir()
    (detection_dir / 'comment-memory' / 'link.md').symlink_to(detection_dir.parent / 'prompt.txt')

    with pytest.raises(SystemExit):
        gh_aw_opencode_detect.main()

    assert invoked == []
    assert status(detection_dir) == 'THREAT_DETECTION_STATUS: reason=engine_error exit=2'
