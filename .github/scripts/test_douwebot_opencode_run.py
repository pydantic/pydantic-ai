"""Verify the OpenCode fallback's checkout boundary."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

import douwebot_opencode_run
from douwebot_opencode_run import DEEPINFRA, OPENCODE_GO, Provider, check_symlinks

# What OpenCode 2.2.5 writes when OpenCode Go's usage limit refuses the first request: a step starts, none finishes.
LIMIT = 'OpenCode Go weekly usage limit reached. Switch to a free model, or turn on Extra Usage on the Go page.'
REFUSED: list[dict[str, object]] = [
    {'type': 'step_start', 'part': {'messageID': 'msg_1', 'type': 'step-start'}},
    {
        'type': 'error',
        'error': {'type': 'provider.go-limit', 'message': LIMIT, 'status': 429, 'response': {'body': '{}'}},
    },
]
FINISHED = [
    {'type': 'text', 'part': {'messageID': 'msg_1', 'text': '{}'}},
    {'type': 'step_finish', 'part': {'messageID': 'msg_1', 'reason': 'stop'}},
]


def test_internal_symlinks_are_allowed(tmp_path: Path) -> None:
    """Existing links between files and directories in a checkout are safe."""
    root = tmp_path / 'checkout'
    (root / 'directory').mkdir(parents=True)
    (root / 'directory' / 'target').write_text('content')
    (root / 'file-link').symlink_to('directory/target')
    (root / 'directory-link').symlink_to('directory', target_is_directory=True)
    check_symlinks(root)


@pytest.mark.parametrize('kind', ['external', 'dangling', 'cycle', 'ancestor'])
def test_unsafe_symlink_fails_closed(tmp_path: Path, kind: str) -> None:
    """A review cannot start with links that escape or cannot resolve safely."""
    root = tmp_path / 'checkout'
    root.mkdir()
    if kind == 'external':
        (tmp_path / 'outside').write_text('secret')
        (root / 'link').symlink_to('../outside')
    elif kind == 'dangling':
        (root / 'link').symlink_to('missing')
    elif kind == 'cycle':
        (root / 'link').symlink_to('other')
        (root / 'other').symlink_to('link')
    else:
        (root / 'link').symlink_to('.', target_is_directory=True)
    with pytest.raises(ValueError, match=r'[Ss]ymlink'):
        check_symlinks(root)


def test_pr_checkout_is_checked_separately(tmp_path: Path) -> None:
    """An untrusted head cannot link into the surrounding base checkout."""
    root = tmp_path / 'base'
    head = root / 'pr-head'
    head.mkdir(parents=True)
    (root / 'trusted-file').write_text('base')
    (head / 'link').symlink_to('../trusted-file')
    check_symlinks(root, skip_pr_head=True)
    with pytest.raises(ValueError, match='escapes'):
        check_symlinks(head)


@pytest.fixture
def review_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stage the douwebot review layout and environment, and return the job's output file."""
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'pr-head').mkdir()
    runner_temp = tmp_path / 'runner-temp'
    runner_temp.mkdir()
    workflow_path = Path(__file__).parent.parent / 'workflows' / 'bots.yml'
    workflow = yaml.load(workflow_path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    review_steps = workflow['jobs']['review']['steps']
    prompt = next(step['with']['prompt'] for step in review_steps if step.get('id') == 'review')
    github_output = tmp_path / 'github-output'
    github_output.touch()
    monkeypatch.setenv('GITHUB_WORKSPACE', str(workspace))
    monkeypatch.setenv('RUNNER_TEMP', str(runner_temp))
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    monkeypatch.setenv('REVIEW_PROMPT', prompt)
    monkeypatch.setenv('OPENCODE_API_KEY', 'go-key')
    monkeypatch.setenv('DEEPINFRA_API_KEY', 'deepinfra-key')
    monkeypatch.setenv('GH_TOKEN', 'fake-github-token')
    return github_output


class OpenCodeConfig(BaseModel):
    """The parts of the generated OpenCode config these tests check."""

    agents: dict[str, dict[str, object]]
    policies: list[dict[str, str]]


@dataclass
class Launch:
    """One OpenCode launch: its command, config, environment, prompt and timeout."""

    args: list[str]
    config: OpenCodeConfig
    env: dict[str, str]
    input: str
    timeout: float | None


def fake_opencode(
    monkeypatch: pytest.MonkeyPatch, *runs: tuple[Sequence[Mapping[str, object] | str], BaseException | None]
) -> list[Launch]:
    """Make each OpenCode launch write its event lines and then raise its error, recording each launch."""
    invoked: list[Launch] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = TypeAdapter(dict[str, str]).validate_python(kwargs['env'])
        config = (Path(env['XDG_CONFIG_HOME']) / 'opencode' / 'opencode.json').read_text(encoding='utf-8')
        prompt, timeout = kwargs['input'], TypeAdapter(float | None).validate_python(kwargs['timeout'])
        assert isinstance(prompt, str)
        invoked.append(Launch(args, OpenCodeConfig.model_validate_json(config), env, prompt, timeout))
        events, error = runs[len(invoked) - 1]
        stdout = kwargs['stdout']
        assert isinstance(stdout, io.TextIOBase)
        stdout.write(''.join((event if isinstance(event, str) else json.dumps(event)) + '\n' for event in events))
        if error is not None:
            raise error
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, 'run', fake_run)
    return invoked


def assert_runs_on(launch: Launch, provider: Provider) -> None:
    """The run names `provider`'s model, allows only that provider, and holds only its key."""
    assert launch.args[launch.args.index('--model') + 1] == provider.model
    assert launch.config.agents['title']['model'] == provider.model
    assert launch.config.agents['compaction']['model'] == provider.model
    assert [policy for policy in launch.config.policies if policy['action'] == 'provider.use'] == [
        {'action': 'provider.use', 'resource': '*', 'effect': 'deny'},
        {'action': 'provider.use', 'resource': provider.name, 'effect': 'allow'},
    ]
    keys = {OPENCODE_GO.key_variable: 'go-key', DEEPINFRA.key_variable: 'deepinfra-key'}
    assert {name: value for name, value in launch.env.items() if name in keys} == {
        provider.key_variable: keys[provider.key_variable]
    }
    assert 'GH_TOKEN' not in launch.env


def test_launcher_sends_shared_review_and_restricts_environment(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fallback keeps the review rubric and does not pass GitHub credentials to OpenCode."""
    invoked = fake_opencode(monkeypatch, (FINISHED, None))
    douwebot_opencode_run.main()
    [call] = invoked
    assert call.args == [
        'opencode',
        'run',
        '--standalone',
        '--model',
        'opencode-go/deepseek-v4.1-flash',
        '--agent',
        'reviewer',
        '--format',
        'json',
    ]
    assert_runs_on(call, OPENCODE_GO)
    assert 'What to look for' in call.input
    assert 'Live GitHub and Git commands are unavailable' in call.input
    assert 'Before returning the structured result, run `gh pr view' not in call.input
    assert 'Use the `gh` CLI only when' not in call.input
    assert review_env.read_text(encoding='utf-8') == 'engine=opencode\n'
    assert capsys.readouterr().out == 'OpenCode run served by opencode-go/deepseek-v4.1-flash.\n'


@pytest.mark.parametrize('kind', ['provider.go-limit', 'provider.quota'])
def test_go_refusal_reruns_the_review_on_deepinfra(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    """A run Go refused before its first step did no work, so DeepInfra reviews instead and the receipt says so."""
    refused = [REFUSED[0], {'type': 'error', 'error': {'type': kind, 'message': LIMIT, 'status': 429}}]
    invoked = fake_opencode(monkeypatch, (refused, subprocess.CalledProcessError(1, 'opencode')), (FINISHED, None))
    douwebot_opencode_run.main()
    assert invoked[1].input == invoked[0].input
    assert_runs_on(invoked[0], OPENCODE_GO)
    assert_runs_on(invoked[1], DEEPINFRA)
    assert review_env.read_text(encoding='utf-8') == 'engine=opencode-deepinfra\n'
    assert capsys.readouterr().out == (
        f'OpenCode Go refused the run before its first step ({kind}); retrying on DeepInfra.\n'
        'OpenCode run served by deepinfra/deepseek-ai/DeepSeek-V4.1-Flash.\n'
    )


@pytest.mark.parametrize(
    ('deepinfra_key', 'events', 'error'),
    [
        pytest.param(None, REFUSED, subprocess.CalledProcessError(1, 'opencode'), id='no-deepinfra-key'),
        pytest.param('', REFUSED, subprocess.CalledProcessError(1, 'opencode'), id='empty-deepinfra-key'),
        pytest.param(
            'deepinfra-key', [*FINISHED, REFUSED[1]], subprocess.CalledProcessError(1, 'opencode'), id='after-a-step'
        ),
        pytest.param('deepinfra-key', [], subprocess.CalledProcessError(1, 'opencode'), id='no-error-event'),
        pytest.param(
            'deepinfra-key',
            [REFUSED[0], {'type': 'error', 'error': {'type': 'provider.auth', 'message': 'Invalid', 'status': 401}}],
            subprocess.CalledProcessError(1, 'opencode'),
            id='rejected-key',
        ),
        pytest.param(
            'deepinfra-key',
            [REFUSED[0], {'type': 'error', 'error': {'type': ['provider.go-limit'], 'status': 429}}],
            subprocess.CalledProcessError(1, 'opencode'),
            id='error-type-not-a-string',
        ),
        pytest.param(
            'deepinfra-key',
            [REFUSED[0], '{"type": "text", "part"'],
            subprocess.CalledProcessError(1, 'opencode'),
            id='cut-off-event',
        ),
        pytest.param('deepinfra-key', REFUSED[:1], subprocess.TimeoutExpired('opencode', 1), id='timeout'),
    ],
)
def test_go_failure_stays_on_go(
    review_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    deepinfra_key: str | None,
    events: Sequence[Mapping[str, object] | str],
    error: BaseException,
) -> None:
    """Only a refusal before the first step falls back, and only with a DeepInfra key; the job fails as before."""
    if deepinfra_key is None:
        monkeypatch.delenv('DEEPINFRA_API_KEY')
    else:
        monkeypatch.setenv('DEEPINFRA_API_KEY', deepinfra_key)
    invoked = fake_opencode(monkeypatch, (events, error))
    with pytest.raises(type(error)):
        douwebot_opencode_run.main()
    [call] = invoked
    assert_runs_on(call, OPENCODE_GO)
    assert review_env.read_text(encoding='utf-8') == ''
