"""Pull requests leave the built `clai2` web UI bundle to `main`, and everything that ships it builds it first (#11554)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import NotRequired

import pytest
import yaml
from pydantic import TypeAdapter
from typing_extensions import TypedDict

WORKFLOWS = Path(__file__).parents[1] / 'workflows'
BUILD_ACTION = './.github/actions/build-clai2-web-ui'
STATIC = 'src/pydantic_clai2/pydantic_clai2/builtin_plugins/web/static'
SOURCE = 'src/pydantic_clai2/web-ui/src/App.tsx'
ARTIFACT = 'clai2-web-static'


class Job(TypedDict):
    steps: list[dict[str, object]]
    env: NotRequired[dict[str, str]]


class Workflow(TypedDict):
    on: dict[str, object]
    jobs: dict[str, object]


WORKFLOW_ADAPTER: TypeAdapter[Workflow] = TypeAdapter(Workflow)
JOB_ADAPTER: TypeAdapter[Job] = TypeAdapter(Job)
STEP_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])


def _workflow(name: str) -> Workflow:
    loaded: object = yaml.load((WORKFLOWS / name).read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    return WORKFLOW_ADAPTER.validate_python(loaded)


def _job(workflow: str, job: str) -> Job:
    return JOB_ADAPTER.validate_python(_workflow(workflow)['jobs'][job])


def _steps(workflow: str, job: str) -> list[dict[str, object]]:
    return _job(workflow, job)['steps']


def _step(workflow: str, job: str, name: str) -> dict[str, object]:
    return next(step for step in _steps(workflow, job) if step.get('name') == name)


def _git(repo: Path, *args: str) -> str:
    git = shutil.which('git')
    assert git is not None
    return subprocess.run(
        [git, '-c', 'user.name=t', '-c', 'user.email=t@t', *args], cwd=repo, text=True, capture_output=True, check=True
    ).stdout


def _write(repo: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')


def _repo(tmp_path: Path, committed: dict[str, str]) -> Path:
    """A repository whose `HEAD^1` is the base branch and `HEAD` adds `committed`, like a pull request's merge commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '--quiet')
    _write(repo, {SOURCE: 'base\n', f'{STATIC}/index.html': 'page\n', f'{STATIC}/assets/app.js': 'base build\n'})
    _git(repo, 'add', '--all')
    _git(repo, 'commit', '--quiet', '--message', 'base')
    _write(repo, committed)
    _git(repo, 'add', '--all')
    _git(repo, 'commit', '--quiet', '--allow-empty', '--message', 'pull request')
    return repo


def _bash(script: str, *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    bash = shutil.which('bash')
    assert bash is not None
    return subprocess.run([bash, '-e', '-c', script], cwd=cwd, env=env, text=True, capture_output=True, check=False)


GUARD = 'Check the pull request leaves the built web UI to `main`'


@pytest.mark.parametrize(
    ('committed', 'built', 'error'),
    [
        pytest.param({SOURCE: 'change\n'}, {}, None, id='source only'),
        pytest.param({}, {}, None, id='no change'),
        pytest.param(
            {SOURCE: 'change\n', f'{STATIC}/assets/app.js': 'rebuilt\n'},
            {f'{STATIC}/assets/app.js': 'rebuilt\n'},
            'Do not commit the built web UI',
            id='source and its rebuilt bundle',
        ),
        pytest.param(
            {f'{STATIC}/assets/app.js': 'rebuilt\n'},
            {f'{STATIC}/assets/app.js': 'rebuilt\n'},
            None,
            id='bundle refresh matching the build',
        ),
        pytest.param(
            {f'{STATIC}/assets/app.js': 'rebuilt\n'},
            {f'{STATIC}/assets/app.js': 'other build\n'},
            'is not the build',
            id='bundle refresh differing from the build',
        ),
        pytest.param(
            {f'{STATIC}/assets/app.js': 'rebuilt\n'},
            {f'{STATIC}/assets/app.js': 'rebuilt\n', f'{STATIC}/assets/chunk.js': 'new\n'},
            'is not the build',
            id='bundle refresh missing a built file',
        ),
    ],
)
def test_pull_requests_leave_the_bundle_to_main(
    tmp_path: Path, committed: dict[str, str], built: dict[str, str], error: str | None
):
    step = _step('clai2-web.yml', 'web-ui', GUARD)
    repo = _repo(tmp_path, committed)
    # The workflow builds the web UI into the working tree before this step runs.
    _write(repo, built)

    git = shutil.which('git')
    assert git is not None
    env = {'PATH': f'{Path(git).parent}:/usr/bin:/bin', **TypeAdapter(dict[str, str]).validate_python(step['env'])}
    process = _bash(str(step['run']), cwd=repo, env=env)

    if error is None:
        assert process.returncode == 0, process.stdout + process.stderr
    else:
        assert process.returncode == 1
        assert error in process.stdout


def test_the_guard_runs_on_pull_requests_after_the_build():
    steps = _steps('clai2-web.yml', 'web-ui')
    names = [step.get('uses') or step.get('name') for step in steps]
    assert names.index(BUILD_ACTION) < names.index(GUARD)
    assert _step('clai2-web.yml', 'web-ui', GUARD)['if'] == (
        "github.event_name == 'pull_request' || github.event_name == 'merge_group'"
    )


def _provides_the_build(step: dict[str, object]) -> bool:
    """Whether the step builds the web UI, or downloads the build a job without credentials made."""
    if step.get('uses') == BUILD_ACTION:
        return True
    options = STEP_ADAPTER.validate_python(step.get('with', {}))
    return str(step.get('uses', '')).startswith('actions/download-artifact@') and options.get('name') == ARTIFACT


@pytest.mark.parametrize(
    ('workflow', 'job', 'packaging'),
    [
        ('ci.yml', 'release-build', 'uv build --all-packages'),
        ('clai2-bleeding.yml', 'publish', 'build_bleeding.sh'),
        ('clai2-web.yml', 'web-ui', 'uv build --no-sources --package pydantic-clai2'),
    ],
)
def test_packages_ship_a_bundle_built_from_source(workflow: str, job: str, packaging: str):
    """The committed bundle lags a web UI change until its refresh merges, so packaging never relies on it."""
    steps = _steps(workflow, job)
    build = next(index for index, step in enumerate(steps) if _provides_the_build(step))
    package = next(index for index, step in enumerate(steps) if packaging in str(step.get('run', '')))
    assert build < package
    checks = [index for index, step in enumerate(steps) if 'check_web_assets.py' in str(step.get('run', ''))]
    assert checks and checks[0] >= package


@pytest.mark.parametrize(
    ('workflow', 'job'),
    [('ci.yml', 'release-build'), ('clai2-bleeding.yml', 'publish'), ('clai2-web-bundle.yml', 'pull-request')],
)
def test_a_downloaded_build_replaces_the_committed_bundle(workflow: str, job: str):
    steps = _steps(workflow, job)
    download = next(index for index, step in enumerate(steps) if _provides_the_build(step))
    assert STEP_ADAPTER.validate_python(steps[download]['with'])['path'] in (STATIC, '${{ env.STATIC }}')
    assert str(steps[download - 1]['run']) in (f'rm -rf {STATIC}', 'rm -rf "$STATIC"')


@pytest.mark.parametrize(('workflow', 'job'), [('clai2-web.yml', 'web-ui'), ('clai2-web-bundle.yml', 'build')])
def test_the_build_is_handed_over_as_an_artifact(workflow: str, job: str):
    steps = _steps(workflow, job)
    build = next(index for index, step in enumerate(steps) if step.get('uses') == BUILD_ACTION)
    upload = next(
        index for index, step in enumerate(steps) if str(step.get('uses', '')).startswith('actions/upload-artifact@')
    )
    options = STEP_ADAPTER.validate_python(steps[upload]['with'])
    assert build < upload
    assert options['name'] == ARTIFACT
    assert options['path'] in (STATIC, '${{ env.STATIC }}')
    assert options['overwrite'] == 'true'
    if workflow == 'clai2-web.yml':
        # `release-build` downloads it, also on a rerun of a failed release days later.
        assert options['retention-days'] == "${{ github.ref_type == 'tag' && 30 || 1 }}"


def test_no_job_that_runs_npm_holds_credentials():
    """npm dependencies run in the build; a later step in the same job could reach any credential (#11554)."""
    builds: list[tuple[str, str]] = []
    for path in sorted(WORKFLOWS.glob('*.yml')):
        if path.name.endswith('.lock.yml'):
            continue
        loaded = _workflow(path.name)
        default = TypeAdapter(dict[str, str]).validate_python(
            yaml.load(path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader).get('permissions', {})
        )
        for name, raw in loaded['jobs'].items():
            job = STEP_ADAPTER.validate_python(raw)
            steps = TypeAdapter(list[dict[str, object]]).validate_python(job.get('steps', []))
            if not any(step.get('uses') == BUILD_ACTION for step in steps):
                continue
            builds.append((path.name, name))
            permissions = TypeAdapter(dict[str, str]).validate_python(job.get('permissions', default))
            assert 'write' not in permissions.values(), (path.name, name)
            assert 'secrets.' not in yaml.safe_dump(raw), (path.name, name)
    assert sorted(builds) == [('clai2-web-bundle.yml', 'build'), ('clai2-web.yml', 'web-ui')]


def test_releases_do_not_restore_an_npm_cache():
    build = next(step for step in _steps('clai2-web-bundle.yml', 'build') if step.get('uses') == BUILD_ACTION)
    assert STEP_ADAPTER.validate_python(build['with']) == {'npm-cache': 'false'}
    # `clai2-bleeding.yml` and the release build publish the build their call to `clai2-web.yml` made.
    build = next(step for step in _steps('clai2-web.yml', 'web-ui') if step.get('uses') == BUILD_ACTION)
    assert STEP_ADAPTER.validate_python(build['with']) == {'npm-cache': '${{ inputs.npm-cache }}'}
    for workflow, job, npm_cache in (
        ('clai2-bleeding.yml', 'validate', 'false'),
        ('ci.yml', 'test-clai2-web', "${{ github.ref_type == 'tag' && 'false' || 'true' }}"),
    ):
        call = STEP_ADAPTER.validate_python(_workflow(workflow)['jobs'][job])
        assert call['uses'] == './.github/workflows/clai2-web.yml'
        assert STEP_ADAPTER.validate_python(call['with']) == {'npm-cache': npm_cache}, (workflow, job)


def test_the_bundle_is_rebuilt_after_each_web_ui_change_on_main():
    workflow = _workflow('clai2-web-bundle.yml')
    push = STEP_ADAPTER.validate_python(workflow['on']['push'])
    assert push['branches'] == ['main']
    assert 'src/pydantic_clai2/web-ui/**' in TypeAdapter(list[str]).validate_python(push['paths'])
    for job in ('build', 'pull-request'):
        assert _job('clai2-web-bundle.yml', job).get('env', {})['STATIC'] == STATIC


def _run_rebuild(tmp_path: Path, *, built: dict[str, str], open_pr: str = '', auto_merge: str = 'false'):
    """Run the build job's check and the pull request job's push against a stub `gh` and a `git` that records pushes."""
    job_env = {
        **_job('clai2-web-bundle.yml', 'build').get('env', {}),
        **_job('clai2-web-bundle.yml', 'pull-request').get('env', {}),
    }
    check = _step('clai2-web-bundle.yml', 'build', 'Check whether the committed bundle is current')
    push = _step('clai2-web-bundle.yml', 'pull-request', 'Push the rebuilt bundle and queue its pull request')
    repo = _repo(tmp_path, {})
    _write(repo, built)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    git = shutil.which('git')
    assert git is not None
    calls = tmp_path / 'calls'
    (bin_dir / 'git').write_text(
        f"""#!/bin/bash
if [[ "$1" == push ]]; then printf 'git %s\\n' "$*" >> {calls}; exit 0; fi
exec {git} "$@"
""",
        encoding='utf-8',
    )
    (bin_dir / 'gh').write_text(
        f"""#!/bin/bash
args="$*"
printf 'gh %s\\n' "${{args//$'\\n'/ }}" >> {calls}
case "$1 $2" in
  "pr list") printf '%s\\n' "$GH_OPEN_PR" ;;
  "pr view") printf '%s\\n' "$GH_AUTO_MERGE" ;;
  "pr create"|"pr merge") ;;
  *) exit 2 ;;
esac
""",
        encoding='utf-8',
    )
    for stub in ('git', 'gh'):
        (bin_dir / stub).chmod(0o755)
    output = tmp_path / 'github-output'
    env = {
        'PATH': f'{bin_dir}:{Path(git).parent}:/usr/bin:/bin',
        'GITHUB_OUTPUT': str(output),
        'GITHUB_SHA': 'a' * 40,
        'GH_TOKEN': 'token',
        'GH_REPO': 'fixture/repository',
        'GH_OPEN_PR': open_pr,
        'GH_AUTO_MERGE': auto_merge,
        **job_env,
    }
    checked = _bash(str(check['run']), cwd=repo, env=env)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    stale = output.read_text(encoding='utf-8').strip() == 'stale=true'
    if stale:
        pushed = _bash(str(push['run']), cwd=repo, env=env)
        assert pushed.returncode == 0, pushed.stdout + pushed.stderr
    recorded = calls.read_text(encoding='utf-8').splitlines() if calls.exists() else []
    return stale, recorded, repo


def test_a_current_bundle_opens_nothing(tmp_path: Path):
    stale, calls, _ = _run_rebuild(tmp_path, built={})

    assert not stale
    assert calls == []


def test_a_stale_bundle_is_pushed_and_queued(tmp_path: Path):
    stale, calls, repo = _run_rebuild(tmp_path, built={f'{STATIC}/assets/app.js': 'rebuilt\n'})

    assert stale
    assert calls[0] == (
        'git push --force https://x-access-token:token@github.com/fixture/repository.git HEAD:refs/heads/clai2-web-bundle'
    )
    assert calls[1].startswith('gh pr list --head clai2-web-bundle')
    assert calls[2].startswith('gh pr create --base main --head clai2-web-bundle')
    assert calls[3:] == [
        'gh pr view clai2-web-bundle --json autoMergeRequest --jq .autoMergeRequest != null',
        'gh pr merge clai2-web-bundle --auto --squash',
    ]
    assert _git(repo, 'show', '--name-only', '--format=%s', 'HEAD').split() == [
        'Rebuild',
        'the',
        'clai2',
        'web',
        'UI',
        'bundle',
        'for',
        'a' * 9,
        f'{STATIC}/assets/app.js',
    ]


def test_an_open_queued_refresh_is_only_replaced(tmp_path: Path):
    stale, calls, _ = _run_rebuild(
        tmp_path, built={f'{STATIC}/assets/app.js': 'rebuilt\n'}, open_pr='12345', auto_merge='true'
    )

    assert stale
    assert [call.split()[:3] for call in calls] == [
        ['git', 'push', '--force'],
        ['gh', 'pr', 'list'],
        ['gh', 'pr', 'view'],
    ]
