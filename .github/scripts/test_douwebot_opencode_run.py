"""Verify the OpenCode fallback's checkout boundary."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pytest
import yaml
from pydantic import BaseModel, TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

import category_classify_opencode
import douwebot_opencode_run
from douwebot_opencode_run import OPENCODE_GO, ZAI, ZAI_BACKUP, Provider, check_symlinks

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
WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'bots.yml'


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
    workflow = yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    review_steps = workflow['jobs']['review']['steps']
    prompt = next(step['with']['prompt'] for step in review_steps if step.get('id') == 'review')
    github_output = tmp_path / 'github-output'
    github_output.touch()
    monkeypatch.setenv('GITHUB_WORKSPACE', str(workspace))
    monkeypatch.setenv('RUNNER_TEMP', str(runner_temp))
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    monkeypatch.setenv('REVIEW_PROMPT', prompt)
    monkeypatch.setenv('OPENCODE_API_KEY', 'go-key')
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('ZAI_BACKUP_API_KEY', 'zai-backup-key')
    monkeypatch.setenv('GITHUB_RUN_NUMBER', '3188')
    monkeypatch.setenv('GH_TOKEN', 'fake-github-token')
    monkeypatch.delenv('LOGFIRE_TOKEN', raising=False)

    def urlopen(request: urllib.request.Request, timeout: int) -> contextlib.nullcontext[None]:
        raise AssertionError('The review exported spans without LOGFIRE_TOKEN')

    monkeypatch.setattr(douwebot_opencode_run.urllib.request, 'urlopen', urlopen)
    return github_output


@pytest.fixture
def exported(review_env: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, object]]]:
    """Give the review a Logfire token and its workflow run; return the spans of each export it sends."""
    environment = {
        'LOGFIRE_TOKEN': 'logfire-token',
        'LOGFIRE_URL': 'https://logfire-eu.pydantic.info/',
        'GITHUB_REPOSITORY': 'pydantic/pydantic-ai',
        'GITHUB_RUN_ID': '43168376242',
        'GITHUB_RUN_ATTEMPT': '2',
        'GITHUB_JOB': 'douwebot-opencode',
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    exports: list[list[dict[str, object]]] = []

    def urlopen(request: urllib.request.Request, timeout: int) -> contextlib.nullcontext[None]:
        assert (request.full_url, request.get_header('Authorization')) == (
            'https://logfire-eu.pydantic.info/v1/traces',
            'logfire-token',
        )
        assert isinstance(request.data, bytes)
        # Neither the provider's message nor its response body leaves the job.
        assert b'Rate limit reached' not in request.data and LIMIT.encode() not in request.data
        [resource] = json.loads(request.data)['resourceSpans']
        assert resource['resource'] == {'attributes': [{'key': 'service.name', 'value': {'stringValue': 'douwebot'}}]}
        [scope] = resource['scopeSpans']
        exports.append(scope['spans'])
        return contextlib.nullcontext()

    monkeypatch.setattr(douwebot_opencode_run.urllib.request, 'urlopen', urlopen)
    return exports


def span_attributes(span: dict[str, object]) -> dict[str, object]:
    """The span's attributes as plain values."""
    values: dict[str, object] = {}
    for attribute in TypeAdapter(list[dict[str, object]]).validate_python(span['attributes']):
        [value] = TypeAdapter(dict[str, object]).validate_python(attribute['value']).values()
        values[str(attribute['key'])] = value
    return values


class OpenCodeConfig(BaseModel):
    """The parts of the generated OpenCode config these tests check."""

    agents: dict[str, dict[str, object]]
    policies: list[dict[str, str]]


@dataclass
class Launch:
    """One OpenCode launch: its command, config, environment, prompt, timeout and working directory."""

    args: list[str]
    config: OpenCodeConfig
    env: dict[str, str]
    input: str
    timeout: float | None
    cwd: Path


def fake_opencode(
    monkeypatch: pytest.MonkeyPatch, *runs: tuple[Sequence[Mapping[str, object] | str | bytes], BaseException | None]
) -> list[Launch]:
    """Make each OpenCode launch write its event lines and then raise its error, recording each launch."""
    invoked: list[Launch] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = TypeAdapter(dict[str, str]).validate_python(kwargs['env'])
        config = (Path(env['XDG_CONFIG_HOME']) / 'opencode' / 'opencode.json').read_text(encoding='utf-8')
        prompt, timeout = kwargs['input'], TypeAdapter(float | None).validate_python(kwargs['timeout'])
        assert isinstance(prompt, str)
        cwd = TypeAdapter(Path).validate_python(kwargs['cwd'])
        invoked.append(Launch(args, OpenCodeConfig.model_validate_json(config), env, prompt, timeout, cwd))
        events, error = runs[len(invoked) - 1]
        stdout = kwargs['stdout']
        assert isinstance(stdout, io.TextIOBase)
        # OpenCode writes bytes to the stream's descriptor, so an event can be invalid UTF-8.
        lines = [
            event if isinstance(event, bytes) else (event if isinstance(event, str) else json.dumps(event)).encode()
            for event in events
        ]
        os.write(stdout.fileno(), b''.join(line + b'\n' for line in lines))
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
    assert launch.config.agents['summary']['model'] == provider.model
    assert [policy for policy in launch.config.policies if policy['action'] == 'provider.use'] == [
        {'action': 'provider.use', 'resource': '*', 'effect': 'deny'},
        {'action': 'provider.use', 'resource': provider.name, 'effect': 'allow'},
    ]
    secrets = {'OPENCODE_API_KEY': 'go-key', 'ZAI_API_KEY': 'zai-key', 'ZAI_BACKUP_API_KEY': 'zai-backup-key'}
    names = {*secrets, OPENCODE_GO.key_variable, ZAI.key_variable}
    assert {name: value for name, value in launch.env.items() if name in names} == {
        provider.key_variable: secrets[provider.secret_variable]
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
    assert capsys.readouterr().out == 'OpenCode run served by opencode-go/deepseek-v4.1-flash on OpenCode Go.\n'
    assert not [name for name in call.env if name.startswith('OTEL_')]


def test_logfire_token_tags_each_runs_telemetry_with_its_plan(
    exported: list[list[dict[str, object]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `LOGFIRE_TOKEN`, each run exports OpenCode's telemetry, tagged with its workflow run, plan and place."""
    invoked = fake_opencode(monkeypatch, (REFUSED, subprocess.CalledProcessError(1, 'opencode')), (FINISHED, None))
    douwebot_opencode_run.main()
    run = 'github.repository=pydantic/pydantic-ai,github.run_id=43168376242,github.run_attempt=2'
    assert [{name: value for name, value in launch.env.items() if name.startswith('OTEL_')} for launch in invoked] == [
        {
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'https://logfire-eu.pydantic.info',
            'OTEL_EXPORTER_OTLP_HEADERS': 'Authorization=logfire-token',
            'OTEL_RESOURCE_ATTRIBUTES': f'{run},github.job=douwebot-opencode,'
            f'opencode_fallback.plan={quote(plan)},opencode_fallback.attempt={attempt}',
        }
        for attempt, plan in enumerate([OPENCODE_GO.plan, ZAI.plan], start=1)
    ]


@pytest.mark.parametrize(
    ('job', 'step'), [('douwebot-opencode', 'opencode'), ('category-classify-opencode', 'classify')]
)
def test_fallback_steps_hand_opencode_the_logfire_token(job: str, step: str) -> None:
    """Both `bots.yml` OpenCode fallbacks export OpenCode's telemetry, as `PR Review Checks` does."""
    workflow = yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    env = next(entry['env'] for entry in workflow['jobs'][job]['steps'] if entry.get('id') == step)
    assert (env['LOGFIRE_TOKEN'], env['LOGFIRE_URL']) == ('${{ secrets.LOGFIRE_TOKEN }}', '${{ vars.LOGFIRE_URL }}')


GO_LIMIT_LOG = 'OpenCode run on opencode-go/deepseek-v4.1-flash on OpenCode Go failed with error type provider.go-limit, status 429.'
NO_TYPE_LOG = 'OpenCode run on opencode-go/deepseek-v4.1-flash on OpenCode Go failed with no error type.'
TO_ZAI_LOG = (
    'opencode-go/deepseek-v4.1-flash refused the run before its first step; '
    'retrying on zai-coding-plan/glm-5.3-flash on the primary Coding Plan.'
)
ZAI_LOG = 'OpenCode run on zai-coding-plan/glm-5.3-flash on the primary Coding Plan'
# What OpenCode 2.2.5 writes when the Coding Plan's quota refuses the first request: Z.AI code 1308 or 1310.
PLAN_REFUSED: list[dict[str, object]] = [
    REFUSED[0],
    {'type': 'error', 'error': {'type': 'provider.quota', 'message': 'Usage limit reached for 5 hour.', 'status': 429}},
]


def zai_error(kind: str, code: object) -> dict[str, object]:
    """The `error` OpenCode 2.2.5 writes for a Z.AI 429 whose response body carries `code`."""
    message = 'Rate limit reached for requests'
    body = json.dumps({'error': {'code': code, 'message': message}})
    return {'type': kind, 'message': message, 'status': 429, 'response': {'body': body}}


@pytest.mark.parametrize('kind', ['provider.go-limit', 'provider.quota'])
def test_go_refusal_reruns_the_review_on_the_coding_plan(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    """A run Go refused before its first step did no work, so the primary Coding Plan reviews and the receipt says so."""
    refused: list[dict[str, object]] = [
        REFUSED[0],
        {'type': 'error', 'error': {'type': kind, 'message': LIMIT, 'status': 429}},
    ]
    invoked = fake_opencode(monkeypatch, (refused, subprocess.CalledProcessError(1, 'opencode')), (FINISHED, None))
    douwebot_opencode_run.main()
    assert invoked[1].input == invoked[0].input
    assert_runs_on(invoked[0], OPENCODE_GO)
    assert_runs_on(invoked[1], ZAI)
    assert review_env.read_text(encoding='utf-8') == 'engine=opencode-zai\n'
    assert capsys.readouterr().out == (
        f'OpenCode run on opencode-go/deepseek-v4.1-flash on OpenCode Go failed with error type {kind}, status 429.\n'
        f'{TO_ZAI_LOG}\n'
        'OpenCode run served by zai-coding-plan/glm-5.3-flash on the primary Coding Plan.\n'
    )


@pytest.mark.parametrize('kind', ['provider.quota', 'provider.rate-limit'])
@pytest.mark.parametrize(
    ('run_number', 'first', 'second'),
    [pytest.param('3188', ZAI, ZAI_BACKUP, id='even-run'), pytest.param('3189', ZAI_BACKUP, ZAI, id='odd-run')],
)
def test_coding_plan_refusal_reruns_the_review_on_the_other_account(
    review_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    run_number: str,
    first: Provider,
    second: Provider,
) -> None:
    """A Coding Plan account whose quota or rate limit refused the run hands it to the other account's key."""
    monkeypatch.setenv('GITHUB_RUN_NUMBER', run_number)
    refusal = subprocess.CalledProcessError(1, 'opencode')
    plan_refused = [REFUSED[0], {'type': 'error', 'error': {'type': kind, 'status': 429}}]
    invoked = fake_opencode(monkeypatch, (REFUSED, refusal), (plan_refused, refusal), (FINISHED, None))
    douwebot_opencode_run.main()
    assert [launch.input for launch in invoked] == [invoked[0].input] * 3
    assert_runs_on(invoked[1], first)
    assert_runs_on(invoked[2], second)
    assert review_env.read_text(encoding='utf-8') == 'engine=opencode-zai\n'
    model = 'zai-coding-plan/glm-5.3-flash'
    assert capsys.readouterr().out == (
        f'{GO_LIMIT_LOG}\n'
        f'opencode-go/deepseek-v4.1-flash refused the run before its first step; retrying on {model} on {first.plan}.\n'
        f'OpenCode run on {model} on {first.plan} failed with error type {kind}, status 429.\n'
        f'{model} refused the run before its first step; retrying on {model} on {second.plan}.\n'
        f'OpenCode run served by {model} on {second.plan}.\n'
    )


@pytest.mark.parametrize(
    ('run_number', 'matrix_index', 'first'),
    [
        ('3188', 0, ZAI),
        ('3188', 1, ZAI_BACKUP),
        ('3188', 2, ZAI),
        ('3189', 0, ZAI_BACKUP),
        ('3189', 1, ZAI),
    ],
)
def test_run_number_and_matrix_place_alternate_the_first_account(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, run_number: str, matrix_index: int, first: Provider
) -> None:
    """Consecutive runs of a workflow, and consecutive checks of one matrix, start on alternate accounts."""
    monkeypatch.setenv('GITHUB_RUN_NUMBER', run_number)
    invoked = fake_opencode(monkeypatch, (REFUSED, subprocess.CalledProcessError(1, 'opencode')), (FINISHED, None))
    _, provider = douwebot_opencode_run.run_with_fallback(
        lambda command, env, stream: subprocess.run(
            command, input='', text=True, cwd=review_env.parent, env=env, stdout=stream, check=True, timeout=None
        ),
        agent='reviewer',
        output=review_env.parent / 'review.ndjson',
        matrix_index=matrix_index,
    )
    assert provider == first
    assert_runs_on(invoked[1], first)


def test_go_refusal_skips_a_coding_plan_account_without_a_key(
    review_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain leaves out the primary account when only the backup key is set."""
    monkeypatch.delenv('ZAI_API_KEY')
    invoked = fake_opencode(monkeypatch, (REFUSED, subprocess.CalledProcessError(1, 'opencode')), (FINISHED, None))
    douwebot_opencode_run.main()
    assert_runs_on(invoked[1], ZAI_BACKUP)


@pytest.mark.parametrize(
    ('run_number', 'backup_key', 'events', 'described'),
    [
        pytest.param('3188', None, PLAN_REFUSED, 'error type provider.quota, status 429', id='no-backup-key'),
        # Without the backup account, an odd run starts on the primary, as an even one does.
        pytest.param('3189', None, PLAN_REFUSED, 'error type provider.quota, status 429', id='no-backup-key-odd-run'),
        pytest.param(
            '3188',
            None,
            [REFUSED[0], {'type': 'error', 'error': zai_error('provider.rate-limit', '1302')}],
            'error type provider.rate-limit, status 429, code 1302',
            id='no-backup-key-rate-limit',
        ),
        pytest.param('3188', '', PLAN_REFUSED, 'error type provider.quota, status 429', id='empty-backup-key'),
        pytest.param(
            '3188',
            'zai-backup-key',
            [*FINISHED, PLAN_REFUSED[1]],
            'error type provider.quota, status 429',
            id='quota-after-a-step',
        ),
        # Z.AI codes 1302 and 1305: OpenCode retried the 429 within the run before it failed.
        pytest.param(
            '3188',
            None,
            [*FINISHED, {'type': 'error', 'error': zai_error('provider.rate-limit', '1305')}],
            'error type provider.rate-limit, status 429, code 1305',
            id='no-backup-key-rate-limit-after-a-step',
        ),
    ],
)
def test_coding_plan_failure_stays_on_the_primary_account(
    review_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_number: str,
    backup_key: str | None,
    events: list[dict[str, object]],
    described: str,
) -> None:
    """Only a quota or rate-limit refusal before the first step, or a rate limit after it, reaches the backup.

    The backup needs its key. The job fails otherwise.
    """
    monkeypatch.setenv('GITHUB_RUN_NUMBER', run_number)
    if backup_key is None:
        monkeypatch.delenv('ZAI_BACKUP_API_KEY')
    else:
        monkeypatch.setenv('ZAI_BACKUP_API_KEY', backup_key)
    refusal = subprocess.CalledProcessError(1, 'opencode')
    invoked = fake_opencode(monkeypatch, (REFUSED, refusal), (events, refusal))
    with pytest.raises(subprocess.CalledProcessError):
        douwebot_opencode_run.main()
    assert len(invoked) == 2
    assert review_env.read_text(encoding='utf-8') == ''
    assert capsys.readouterr().out == f'{GO_LIMIT_LOG}\n{TO_ZAI_LOG}\n{ZAI_LOG} failed with {described}.\n'


RATE_LIMITED_AFTER_A_STEP: list[Mapping[str, object]] = [
    *FINISHED,
    {'type': 'error', 'error': zai_error('provider.rate-limit', '1302')},
]


RUN = {
    'github.repository': 'pydantic/pydantic-ai',
    'github.run_id': '43168376242',
    'github.run_attempt': '2',
    'github.job': 'douwebot-opencode',
}


@pytest.mark.parametrize(
    ('last_run', 'raised', 'failure'),
    [
        pytest.param(FINISHED, None, {}, id='served'),
        pytest.param(
            RATE_LIMITED_AFTER_A_STEP,
            subprocess.CalledProcessError,
            {
                'opencode_fallback.error_type': 'provider.rate-limit',
                'opencode_fallback.error_status': '429',
                'opencode_fallback.error_code': '1302',
                'opencode_fallback.step_finish': True,
            },
            id='failed',
        ),
    ],
)
def test_logfire_token_exports_a_span_per_run(
    exported: list[list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
    last_run: list[Mapping[str, object]],
    raised: type[BaseException] | None,
    failure: dict[str, object],
) -> None:
    """Each run exports a span with its plan, its place and how it failed, a run the chain moved past included."""
    refusal = subprocess.CalledProcessError(1, 'opencode')
    handover = [REFUSED[0], {'type': 'error', 'error': zai_error('provider.rate-limit', '1302')}]
    fake_opencode(monkeypatch, (REFUSED, refusal), (handover, refusal), (last_run, refusal if raised else None))
    with pytest.raises(raised) if raised else contextlib.nullcontext():
        douwebot_opencode_run.main()
    [runs] = exported
    assert [span['name'] for span in runs] == ['OpenCode run'] * 3
    assert len({span['traceId'] for span in runs}) == 1
    assert not [span for span in runs if 'parentSpanId' in span]
    assert [span.get('status') for span in runs] == [{'code': 2}, {'code': 2}, {'code': 2} if raised else None]
    assert [span_attributes(span) for span in runs] == [
        {
            **RUN,
            'opencode_fallback.plan': 'OpenCode Go',
            'opencode_fallback.attempt': '1',
            'opencode_fallback.provider': 'opencode-go',
            'opencode_fallback.error_type': 'provider.go-limit',
            'opencode_fallback.error_status': '429',
            'opencode_fallback.step_finish': False,
        },
        {
            **RUN,
            'opencode_fallback.plan': 'the primary Coding Plan',
            'opencode_fallback.attempt': '2',
            'opencode_fallback.provider': 'zai-coding-plan',
            'opencode_fallback.error_type': 'provider.rate-limit',
            'opencode_fallback.error_status': '429',
            'opencode_fallback.error_code': '1302',
            'opencode_fallback.step_finish': False,
        },
        {
            **RUN,
            'opencode_fallback.plan': 'the backup Coding Plan',
            'opencode_fallback.attempt': '3',
            'opencode_fallback.provider': 'zai-coding-plan',
            **failure,
        },
    ]


@pytest.mark.parametrize(
    ('run_number', 'first', 'second'),
    [pytest.param('3188', ZAI, ZAI_BACKUP, id='even-run'), pytest.param('3189', ZAI_BACKUP, ZAI, id='odd-run')],
)
def test_coding_plan_rate_limit_after_a_step_restarts_the_review_on_the_other_account(
    review_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_number: str,
    first: Provider,
    second: Provider,
) -> None:
    """A Coding Plan account that rate-limits the review after it took steps hands the review to the other account.

    The restart is its own attempt, and its log line tells it from a refusal before the first step.
    """
    monkeypatch.setenv('GITHUB_RUN_NUMBER', run_number)
    refusal = subprocess.CalledProcessError(1, 'opencode')
    invoked = fake_opencode(monkeypatch, (REFUSED, refusal), (RATE_LIMITED_AFTER_A_STEP, refusal), (FINISHED, None))
    attempts: list[douwebot_opencode_run.Attempt] = []
    provider = douwebot_opencode_run.run_read_only(
        'Review.',
        cwd=review_env.parent,
        agent='reviewer',
        output=review_env.parent / 'review.ndjson',
        attempts=attempts,
    )
    assert provider == second
    assert [launch.input for launch in invoked] == ['Review.'] * 3
    assert_runs_on(invoked[1], first)
    assert_runs_on(invoked[2], second)
    assert [(attempt.provider, attempt.error_type, attempt.code, attempt.finished) for attempt in attempts] == [
        (OPENCODE_GO, 'provider.go-limit', None, False),
        (first, 'provider.rate-limit', '1302', True),
        (second, None, None, None),
    ]
    model = 'zai-coding-plan/glm-5.3-flash'
    assert capsys.readouterr().out == (
        f'{GO_LIMIT_LOG}\n'
        f'opencode-go/deepseek-v4.1-flash refused the run before its first step; retrying on {model} on {first.plan}.\n'
        f'OpenCode run on {model} on {first.plan} failed with error type provider.rate-limit, status 429, code 1302.\n'
        f'{model} rate-limited the run after it took steps; restarting the run on {model} on {second.plan}.\n'
        f'OpenCode run served by {model} on {second.plan}.\n'
    )


@pytest.mark.parametrize(
    'first_events',
    [
        pytest.param([REFUSED[0], {'type': 'error', 'error': zai_error('provider.rate-limit', '1302')}], id='handover'),
        pytest.param(RATE_LIMITED_AFTER_A_STEP, id='restart'),
    ],
)
def test_last_coding_plan_account_rate_limit_after_a_step_fails_the_job(
    review_env: Path, monkeypatch: pytest.MonkeyPatch, first_events: list[Mapping[str, object]]
) -> None:
    """Each account runs the review at most once, so a rate limit after steps on the last account fails the job."""
    refusal = subprocess.CalledProcessError(1, 'opencode')
    invoked = fake_opencode(
        monkeypatch, (REFUSED, refusal), (first_events, refusal), (RATE_LIMITED_AFTER_A_STEP, refusal)
    )
    with pytest.raises(subprocess.CalledProcessError):
        douwebot_opencode_run.main()
    assert len(invoked) == 3
    assert_runs_on(invoked[1], ZAI)
    assert_runs_on(invoked[2], ZAI_BACKUP)
    assert review_env.read_text(encoding='utf-8') == ''


@pytest.mark.parametrize(
    ('zai_key', 'events', 'error', 'log'),
    [
        pytest.param(None, REFUSED, subprocess.CalledProcessError(1, 'opencode'), GO_LIMIT_LOG, id='no-zai-key'),
        pytest.param('', REFUSED, subprocess.CalledProcessError(1, 'opencode'), GO_LIMIT_LOG, id='empty-zai-key'),
        pytest.param(
            'zai-key',
            [*FINISHED, REFUSED[1]],
            subprocess.CalledProcessError(1, 'opencode'),
            GO_LIMIT_LOG,
            id='after-a-step',
        ),
        pytest.param(
            'zai-key',
            [*FINISHED, {'type': 'error', 'error': {'type': 'provider.rate-limit', 'status': 429}}],
            subprocess.CalledProcessError(1, 'opencode'),
            'OpenCode run on opencode-go/deepseek-v4.1-flash on OpenCode Go failed with error type provider.rate-limit, status 429.',
            id='rate-limit-after-a-step',
        ),
        pytest.param('zai-key', [], subprocess.CalledProcessError(1, 'opencode'), NO_TYPE_LOG, id='no-error-event'),
        pytest.param(
            'zai-key',
            [REFUSED[0], {'type': 'error', 'error': {'type': 'provider.auth', 'message': 'Invalid', 'status': 401}}],
            subprocess.CalledProcessError(1, 'opencode'),
            'OpenCode run on opencode-go/deepseek-v4.1-flash on OpenCode Go failed with error type provider.auth, status 401.',
            id='rejected-key',
        ),
        pytest.param(
            'zai-key',
            [REFUSED[0], {'type': 'error', 'error': {'type': ['provider.go-limit'], 'status': 429}}],
            subprocess.CalledProcessError(1, 'opencode'),
            'OpenCode run on opencode-go/deepseek-v4.1-flash on OpenCode Go failed with no error type, status 429.',
            id='error-type-not-a-string',
        ),
        pytest.param(
            'zai-key',
            [REFUSED[0], '{"type": "text", "part"'],
            subprocess.CalledProcessError(1, 'opencode'),
            NO_TYPE_LOG,
            id='cut-off-event',
        ),
        pytest.param(
            'zai-key',
            [REFUSED[0], b'\xff'],
            subprocess.CalledProcessError(1, 'opencode'),
            NO_TYPE_LOG,
            id='undecodable',
        ),
        pytest.param('zai-key', REFUSED[:1], subprocess.TimeoutExpired('opencode', 1), None, id='timeout'),
    ],
)
def test_go_failure_stays_on_go(
    review_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    zai_key: str | None,
    events: Sequence[Mapping[str, object] | str | bytes],
    error: BaseException,
    log: str | None,
) -> None:
    """Only a refusal before the first step falls back, and only with a Coding Plan key; the job fails as before.

    A failed process names its error type in the job log. A timeout is not a failed process.
    """
    monkeypatch.delenv('ZAI_BACKUP_API_KEY')
    if zai_key is None:
        monkeypatch.delenv('ZAI_API_KEY')
    else:
        monkeypatch.setenv('ZAI_API_KEY', zai_key)
    invoked = fake_opencode(monkeypatch, (events, error))
    with pytest.raises(type(error)):
        douwebot_opencode_run.main()
    [call] = invoked
    assert_runs_on(call, OPENCODE_GO)
    assert review_env.read_text(encoding='utf-8') == ''
    assert capsys.readouterr().out == ('' if log is None else log + '\n')


@pytest.mark.parametrize(
    ('error', 'described'),
    [
        pytest.param(
            {'type': 'provider.auth', 'message': 'Invalid key', 'status': 401, 'response': {'body': 'Invalid key'}},
            'error type provider.auth, status 401',
            id='type-and-status',
        ),
        pytest.param({'type': 'unknown', 'message': 'Model not found'}, 'error type unknown', id='no-status'),
        pytest.param(
            {'type': 'provider.auth: Invalid key', 'status': 401},
            'no error type, status 401',
            id='type-not-shaped-like-opencodes',
        ),
        pytest.param(zai_error('unknown', 1305), 'error type unknown, status 429, code 1305', id='numeric-code'),
        pytest.param(zai_error('unknown', '1302: sk-key'), 'error type unknown, status 429', id='code-not-all-digits'),
        pytest.param(zai_error('unknown', '1302130213'), 'error type unknown, status 429', id='code-too-long'),
        pytest.param(zai_error('unknown', True), 'error type unknown, status 429', id='code-not-a-number'),
        pytest.param(
            {'type': 'unknown', 'status': 429, 'response': {'body': 'code 1302'}},
            'error type unknown, status 429',
            id='body-not-json',
        ),
    ],
)
def test_coding_plan_failure_names_its_error_type(
    review_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: dict[str, object],
    described: str,
) -> None:
    """A failed Coding Plan run fails the job and names its error type and status, never its message or body."""
    invoked = fake_opencode(
        monkeypatch,
        (REFUSED, subprocess.CalledProcessError(1, 'opencode')),
        ([REFUSED[0], {'type': 'error', 'error': error}], subprocess.CalledProcessError(1, 'opencode')),
    )
    with pytest.raises(subprocess.CalledProcessError):
        douwebot_opencode_run.main()
    assert_runs_on(invoked[1], ZAI)
    assert review_env.read_text(encoding='utf-8') == ''
    assert capsys.readouterr().out == f'{GO_LIMIT_LOG}\n{TO_ZAI_LOG}\n{ZAI_LOG} failed with {described}.\n'


def classifier_answer(text: str) -> list[dict[str, object]]:
    """The events of a completed OpenCode run whose final message is `text`."""
    return [
        {'type': 'text', 'part': {'messageID': 'msg_1', 'text': text}},
        {'type': 'step_finish', 'part': {'messageID': 'msg_1', 'reason': 'stop'}},
    ]


@pytest.fixture
def classify_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stage the prefetched context and the `Category Classify (OpenCode)` environment; return the job's output file."""
    runner_temp = tmp_path / 'runner-temp'
    (runner_temp / 'category-context').mkdir(parents=True)
    workflow = yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    steps = workflow['jobs']['category-classify-opencode']['steps']
    github_output = tmp_path / 'github-output'
    github_output.touch()
    monkeypatch.setenv('RUNNER_TEMP', str(runner_temp))
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    monkeypatch.setenv(
        'CLASSIFY_PROMPT', next(step['env']['CLASSIFY_PROMPT'] for step in steps if step.get('id') == 'classify')
    )
    monkeypatch.setenv('OPENCODE_API_KEY', 'go-key')
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('ZAI_BACKUP_API_KEY', 'zai-backup-key')
    monkeypatch.setenv('GITHUB_RUN_NUMBER', '3188')
    monkeypatch.setenv('GH_TOKEN', 'fake-github-token')
    return github_output


def test_classifier_reads_only_the_prefetched_context(
    classify_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenCode gets the Claude prompt, runs in the prefetched context without GitHub credentials, and sets the output."""
    invoked = fake_opencode(monkeypatch, (classifier_answer('{"category": "chore"}'), None))
    category_classify_opencode.main()
    [call] = invoked
    assert_runs_on(call, OPENCODE_GO)
    assert call.args[call.args.index('--agent') + 1] == 'classifier'
    assert call.cwd == tmp_path / 'runner-temp' / 'category-context'
    workflow = yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    claude = next(step for step in workflow['jobs']['category-classify']['steps'] if step.get('id') == 'classify')
    assert call.input == claude['with']['prompt'] + category_classify_opencode.TOOL_GUIDANCE
    assert classify_env.read_text(encoding='utf-8') == 'category=chore\n'


@pytest.mark.parametrize(
    'text',
    [
        pytest.param('The PR fixes a crash.\n```json\n{"category": "bug"}\n```', id='fenced'),
        pytest.param('The PR fixes `f"{x}"`.\n\n{"category": "bug"}', id='prose-before-bare-json'),
    ],
)
def test_classifier_reads_an_answer_after_prose(classify_env: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Prose before the JSON object, bare or fenced, passes, as it does for `PR Review Checks`."""
    fake_opencode(monkeypatch, (classifier_answer(text), None))
    category_classify_opencode.main()
    assert classify_env.read_text(encoding='utf-8') == 'category=bug\n'


@pytest.mark.parametrize(
    ('events', 'reason'),
    [
        pytest.param(classifier_answer('{"category": "refactor"}'), 'validation failed: literal_error', id='unknown'),
        pytest.param(
            classifier_answer('{"category": "bug", "reason": "fixes a crash"}'),
            'validation failed: extra_forbidden',
            id='extra-field',
        ),
        pytest.param(classifier_answer('The category is bug.'), 'validation failed: json_invalid', id='prose'),
        pytest.param(REFUSED[:1], 'OpenCode review did not complete', id='incomplete'),
    ],
)
def test_classifier_rejects_any_other_answer(
    classify_env: Path, monkeypatch: pytest.MonkeyPatch, events: list[dict[str, object]], reason: str
) -> None:
    """Only one allowed category and nothing else reaches `category-apply`; the job log names why, never the text."""
    fake_opencode(monkeypatch, (events, None))
    with pytest.raises(SystemExit) as raised:
        category_classify_opencode.main()
    assert raised.value.code == f'OpenCode returned an invalid category: {reason}'
    assert classify_env.read_text(encoding='utf-8') == ''
