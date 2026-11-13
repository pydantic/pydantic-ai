"""Verify the OpenCode threat detector records only verdicts `threat-detect conclude` accepts."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

import douwebot_opencode_run
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
    (tmp_path / 'github-output').touch()
    monkeypatch.setattr(gh_aw_opencode_detect, 'DETECTION_DIR', directory)
    monkeypatch.setenv('GH_AW_PROMPT', str(prompt))
    monkeypatch.setenv('RUNNER_TEMP', str(runner_temp))
    monkeypatch.setenv('OPENCODE_API_KEY', 'fake-key')
    # Detection stays on Go unless a test configures a Coding Plan account.
    monkeypatch.delenv('ZAI_API_KEY', raising=False)
    monkeypatch.delenv('ZAI_BACKUP_API_KEY', raising=False)
    monkeypatch.setenv('GITHUB_RUN_NUMBER', '3188')
    monkeypatch.setenv('GH_AW_HARNESS_MAX_RETRIES', '3')
    monkeypatch.setenv('GH_TOKEN', 'fake-github-token')
    monkeypatch.setenv('GITHUB_OUTPUT', str(tmp_path / 'github-output'))
    monkeypatch.delenv('LOGFIRE_TOKEN', raising=False)
    return directory


def answer(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    *,
    incomplete_runs: int = 0,
    refused_runs: int = 0,
    clock: list[float] | None = None,
) -> list[dict[str, object]]:
    """Make OpenCode stop normally with `text` as its final message, recording each launch.

    The first `refused_runs` launches fail as OpenCode Go's usage limit refusing the first request does, each
    advancing `clock` by a minute. The next `incomplete_runs` exit without writing a stream, as an interrupted run
    does.
    """
    invoked: list[dict[str, object]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        invoked.append({'args': args, **kwargs})
        stdout = kwargs['stdout']
        assert isinstance(stdout, io.TextIOBase)
        if len(invoked) <= refused_runs:
            error = {'type': 'provider.go-limit', 'message': 'OpenCode Go weekly usage limit reached.', 'status': 429}
            stdout.write(json.dumps({'type': 'step_start', 'part': {}}) + '\n' + json.dumps({'type': 'error', 'error': error}) + '\n')
            if clock is not None:
                clock[0] += 60
            raise subprocess.CalledProcessError(1, args)
        if len(invoked) <= refused_runs + incomplete_runs:
            return subprocess.CompletedProcess(args, 0)
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


def outputs(directory: Path) -> str:
    """Return the step outputs the detector wrote, which name the plan of its last OpenCode run."""
    return (directory.parent / 'github-output').read_text()


GO_OUTPUTS = 'provider=opencode-go\nplan=OpenCode Go\n'
ZAI_OUTPUTS = 'provider=zai-coding-plan\nplan=the primary Coding Plan\n'


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
    assert outputs(detection_dir) == GO_OUTPUTS
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
    assert isinstance(call['timeout'], float)
    assert 0 < call['timeout'] <= gh_aw_opencode_detect.DEADLINE_SECONDS
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
    invoked = answer(monkeypatch, text.replace('"reasons": []', '"reasons": ["::error::injected"]', 1))

    with pytest.raises(SystemExit) as exc_info:
        gh_aw_opencode_detect.main()

    assert len(invoked) == 1

    assert not (detection_dir / 'detection_result.json').exists()
    assert not (detection_dir / 'detection_result_full.json').exists()
    assert status(detection_dir) == 'THREAT_DETECTION_STATUS: reason=invalid_report_exhausted exit=1'
    assert 'injected' not in str(exc_info.value.code)


def test_engine_failure_concludes_as_agent_failure(detection_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenCode failure that outlasts the retries leaves no verdict and reports an engine error."""
    attempts: list[list[str]] = []

    def failed_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        attempts.append(args)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, 'run', failed_run)

    with pytest.raises(SystemExit):
        gh_aw_opencode_detect.main()

    assert len(attempts) == 4
    assert not (detection_dir / 'detection_result.json').exists()
    assert status(detection_dir) == 'THREAT_DETECTION_STATUS: reason=engine_error exit=2'


def test_retry_recovers_from_an_incomplete_stream(
    detection_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that stops without a final message is retried, and the job log says why; an invalid verdict is not."""
    invoked = answer(monkeypatch, 'THREAT_DETECTION_RESULT:' + json.dumps(CLEAN), incomplete_runs=1)

    gh_aw_opencode_detect.main()

    assert len(invoked) == 2
    assert 'OpenCode threat detection attempt 1 failed: OpenCode review did not complete' in (
        capsys.readouterr().out.splitlines()
    )
    streams = (detection_dir.parent / 'runner-temp').glob('gh-aw-opencode-detector-*.ndjson')
    assert sorted(stream.name for stream in streams) == [
        'gh-aw-opencode-detector-1.ndjson',
        'gh-aw-opencode-detector-2.ndjson',
    ]
    assert json.loads((detection_dir / 'detection_result.json').read_text()) == CLEAN


def test_go_refusal_runs_detection_on_the_coding_plan(detection_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection that Go refused before its first step runs on the Coding Plan within the same attempt and deadline."""
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    # Go takes a minute to refuse, on the clock the launcher reads.
    clock = [1000.0]
    invoked = answer(monkeypatch, 'THREAT_DETECTION_RESULT:' + json.dumps(CLEAN), refused_runs=1, clock=clock)
    monkeypatch.setattr(douwebot_opencode_run, 'time', SimpleNamespace(monotonic=lambda: clock[0]))

    gh_aw_opencode_detect.main()

    go, zai = invoked
    assert isinstance(go['args'], list) and isinstance(zai['args'], list)
    assert (go['args'][4], zai['args'][4]) == ('opencode-go/deepseek-v4.1-flash', 'zai-coding-plan/glm-5.3-flash')
    env = TypeAdapter(dict[str, str]).validate_python(zai['env'])
    assert env['ZHIPU_API_KEY'] == 'zai-key'
    assert 'OPENCODE_API_KEY' not in env and 'ZAI_API_KEY' not in env
    assert isinstance(go['timeout'], float) and isinstance(zai['timeout'], float)
    assert zai['timeout'] == pytest.approx(go['timeout'] - 60)
    streams = (detection_dir.parent / 'runner-temp').glob('gh-aw-opencode-detector-*.ndjson')
    assert [stream.name for stream in streams] == ['gh-aw-opencode-detector-1.ndjson']
    assert json.loads((detection_dir / 'detection_result.json').read_text()) == CLEAN
    assert status(detection_dir) == 'THREAT_DETECTION_STATUS: reason=result_recorded exit=0'
    assert outputs(detection_dir) == ZAI_OUTPUTS


def test_coding_plan_failure_names_its_error_type(
    detection_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Detection that the Coding Plan fails names its error type in the job log, then concludes as an engine error."""
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('GH_AW_HARNESS_MAX_RETRIES', '0')

    def failed_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = kwargs['stdout']
        assert isinstance(stdout, io.TextIOBase)
        go = args[4].startswith('opencode-go/')
        error = {'type': 'provider.go-limit' if go else 'provider.auth', 'message': 'Refused', 'status': 429 if go else 401}
        stdout.write(json.dumps({'type': 'error', 'error': error}) + '\n')
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, 'run', failed_run)

    with pytest.raises(SystemExit):
        gh_aw_opencode_detect.main()

    assert capsys.readouterr().out.splitlines()[-3:] == [
        'OpenCode run on zai-coding-plan/glm-5.3-flash on the primary Coding Plan failed with error type provider.auth, status 401.',
        'OpenCode threat detection attempt 1 failed: CalledProcessError',
        'THREAT_DETECTION_STATUS: reason=engine_error exit=2',
    ]
    assert outputs(detection_dir) == ZAI_OUTPUTS


@pytest.mark.parametrize('deadline_seconds', [gh_aw_opencode_detect.DEADLINE_SECONDS, 0])
def test_deadline_stops_retries_and_reports_a_timeout(
    detection_dir: Path, monkeypatch: pytest.MonkeyPatch, deadline_seconds: int
) -> None:
    """A run that reaches the deadline is not retried, and a spent deadline starts no run at all."""
    monkeypatch.setattr(gh_aw_opencode_detect, 'DEADLINE_SECONDS', deadline_seconds)
    attempts: list[list[str]] = []

    def slow_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        attempts.append(args)
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(subprocess, 'run', slow_run)

    with pytest.raises(SystemExit):
        gh_aw_opencode_detect.main()

    assert len(attempts) == (1 if deadline_seconds else 0)
    assert status(detection_dir) == 'THREAT_DETECTION_STATUS: reason=engine_timeout exit=2'
    assert outputs(detection_dir) == (GO_OUTPUTS if deadline_seconds else '')


def test_escaping_symlink_stops_before_opencode_runs(detection_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenCode checks paths lexically, so a link out of the inputs directory would widen its reads."""
    invoked = answer(monkeypatch, 'THREAT_DETECTION_RESULT:' + json.dumps(CLEAN))
    (detection_dir / 'comment-memory').mkdir()
    (detection_dir / 'comment-memory' / 'link.md').symlink_to(detection_dir.parent / 'prompt.txt')

    with pytest.raises(SystemExit):
        gh_aw_opencode_detect.main()

    assert invoked == []
    assert status(detection_dir) == 'THREAT_DETECTION_STATUS: reason=engine_error exit=2'
