from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter
from typing_extensions import TypedDict

WORKFLOW = Path(__file__).parents[1] / 'workflows' / 'ci.yml'
DEFINITION = 'src/pydantic_ai_harness/gh-aw/pydantic.md'


class Workflow(TypedDict):
    jobs: dict[str, object]


class PinBumpStep(TypedDict, total=False):
    name: str
    run: str


class PinBumpJob(TypedDict):
    steps: list[PinBumpStep]


WORKFLOW_ADAPTER: TypeAdapter[Workflow] = TypeAdapter(Workflow)
PIN_BUMP_JOB_ADAPTER: TypeAdapter[PinBumpJob] = TypeAdapter(PinBumpJob)


def _run_bump(
    tmp_path: Path,
    pinned: str,
    released: str,
    *,
    tag: str = 'v0.55.0',
    stale_pinned: str = '0.55.0',
    snapshot_version: str | None = None,
    existing_title: str = '',
    branch_exists: bool = False,
    branch_content: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], str, str, str]:
    workflow: object = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    jobs = WORKFLOW_ADAPTER.validate_python(workflow)['jobs']
    job = PIN_BUMP_JOB_ADAPTER.validate_python(jobs['gh-aw-engine-pin-bump'])
    guard_run = next(
        step['run']
        for step in job['steps']
        if step.get('name') == 'Check whether the release moved past the definition' and 'run' in step
    )
    bump_run = next(
        step['run'] for step in job['steps'] if step.get('name') == 'Open the pin bump pull request' and 'run' in step
    )
    snapshot = snapshot_version or pinned
    definition = tmp_path / DEFINITION.rsplit('/', maxsplit=1)[-1]
    definition.write_text(
        f"""---
name: Pydantic
engine:
  id: pydantic-ai
  version: {snapshot_version or f'"{snapshot}"'}
runtimes:
  python:
    version: "3.13"
---
engine:
  version: "9.9.9"
""",
        encoding='utf-8',
    )
    branch_definition = tmp_path / 'pydantic.branch.md'
    branch_definition.write_text(branch_content or definition.read_text(encoding='utf-8'), encoding='utf-8')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in ('grep', 'awk', 'cmp', 'cat', 'cp', 'tee', 'tr'):
        source = shutil.which(name)
        assert source is not None
        (bin_dir / name).symlink_to(source)
    base64 = shutil.which('base64')
    assert base64 is not None
    (bin_dir / 'base64').write_text(
        """#!/bin/bash
if [[ "${1:-}" == -w0 ]]; then
  shift
  if [[ $# -gt 0 ]]; then
    "$BASE64_REAL" -i "$1" | tr -d '\\n'
  else
    "$BASE64_REAL" | tr -d '\\n'
  fi
elif [[ "${1:-}" == -d ]]; then
  shift
  "$BASE64_REAL" -d "$@"
else
  "$BASE64_REAL" "$@"
fi
""",
        encoding='utf-8',
    )
    (bin_dir / 'base64').chmod(0o755)
    ledger = tmp_path / 'gh-calls'
    gh = bin_dir / 'gh'
    gh.write_text(
        """#!/bin/bash
set -e
printf '%s\\n' "$*" >> "$GH_CALLS"
case "$*" in
  "pr list "*) if [[ -n "$GH_EXISTING_TITLE" ]]; then printf '%s\\n' "$GH_EXISTING_TITLE"; fi ;;
  "api repos/fixture/repository/git/ref/heads/main --jq .object.sha") printf 'base-sha\\n' ;;
  *"/contents/src/pydantic_ai_harness/gh-aw/pydantic.md?ref=base-sha"*) cat "$GH_SNAPSHOT" ;;
  "api repos/fixture/repository/git/ref/heads/gh-aw-engine-pin-"*)
    if [[ "$GH_BRANCH_EXISTS" == true ]]; then printf 'branch-sha\\n'; else exit 1; fi
    ;;
  "api repos/fixture/repository/git/refs "*) ;;
  *"/contents/src/pydantic_ai_harness/gh-aw/pydantic.md"*"--jq .sha"*) printf 'file-sha\\n' ;;
  *"/contents/src/pydantic_ai_harness/gh-aw/pydantic.md?ref=gh-aw-engine-pin-"*)
    cat "$GH_BRANCH_CONTENT" | tee "$GH_BRANCH_READ"
    ;;
  *"-X PUT repos/fixture/repository/contents/src/pydantic_ai_harness/gh-aw/pydantic.md"*)
    has_expected_sha=false
    for argument in "$@"; do
      case "$argument" in
        sha=file-sha) has_expected_sha=true ;;
        content=*) printf '%s' "${argument#content=}" | base64 -d > "$GH_UPDATED_CONTENT" ;;
      esac
    done
    [[ "$has_expected_sha" == true ]]
    ;;
  "pr create "*) cp "$GH_BODY" "$GH_CREATED_BODY" ;;
  *) exit 2 ;;
esac
""",
        encoding='utf-8',
    )
    gh.chmod(0o755)
    assert shutil.which('gh', path=str(bin_dir)) == str(gh)
    runner_temp = tmp_path / 'runner-temp'
    runner_temp.mkdir()
    output = tmp_path / 'github-output'
    env = {
        'PATH': str(bin_dir),
        'GITHUB_OUTPUT': str(output),
        'RUNNER_TEMP': str(runner_temp),
        'GITHUB_REF_NAME': tag,
        'GITHUB_REPOSITORY': 'fixture/repository',
        'GITHUB_SERVER_URL': 'https://github.com',
        'GITHUB_RUN_ID': '123',
        'GH_REPO': 'fixture/repository',
        'DEFAULT_BRANCH': 'main',
        'RELEASED': released,
        # Keep the stale job output present so the regression catches the old equality guard.
        'PINNED': stale_pinned,
        'GH_CALLS': str(ledger),
        'GH_SNAPSHOT': str(definition),
        'GH_BODY': str(runner_temp / 'pin-bump.md'),
        'GH_CREATED_BODY': str(tmp_path / 'created-body.md'),
        'GH_UPDATED_CONTENT': str(tmp_path / 'updated-content.md'),
        'BASE64_REAL': base64,
        'GH_EXISTING_TITLE': existing_title,
        'GH_BRANCH_EXISTS': str(branch_exists).lower(),
        'GH_BRANCH_CONTENT': str(branch_definition),
        'GH_BRANCH_READ': str(tmp_path / 'branch-read.md'),
    }
    shell = shutil.which('bash')
    assert shell is not None
    guard = subprocess.run([shell, '-e', '-c', guard_run], env=env, text=True, capture_output=True, check=False)
    should_bump = output.exists() and 'bump=true' in output.read_text(encoding='utf-8').splitlines()
    if guard.returncode == 0 and should_bump:
        process = subprocess.run([shell, '-e', '-c', bump_run], env=env, text=True, capture_output=True, check=False)
    else:
        process = guard
    calls = ledger.read_text(encoding='utf-8').splitlines() if ledger.exists() else []
    created_body = (
        (tmp_path / 'created-body.md').read_text(encoding='utf-8') if (tmp_path / 'created-body.md').exists() else ''
    )
    updated_content = (
        (tmp_path / 'updated-content.md').read_text(encoding='utf-8')
        if (tmp_path / 'updated-content.md').exists()
        else ''
    )
    branch_read = (
        (tmp_path / 'branch-read.md').read_text(encoding='utf-8') if (tmp_path / 'branch-read.md').exists() else ''
    )
    return process, calls, created_body, updated_content, branch_read


@pytest.mark.parametrize(('pinned', 'released'), [('0.55.0', '0.54.0'), ('0.55.0', '0.55.0')])
def test_older_or_equal_release_does_not_write_even_when_job_pin_is_stale(tmp_path: Path, pinned: str, released: str):
    process, calls, _, _, _ = _run_bump(tmp_path, pinned, released, stale_pinned='0.53.0')

    assert process.returncode == 0, process.stderr
    assert not any('-X PUT' in call or call.startswith('pr create ') for call in calls)
    assert not any('/git/refs ' in call for call in calls)


def test_exact_open_pull_request_title_stops_after_listing(tmp_path: Path):
    released = '0.55.0'
    title = f'Bump the gh-aw engine pin to `{released}`'

    process, calls, _, updated_content, branch_read = _run_bump(tmp_path, '0.54.0', released, existing_title=title)

    assert process.returncode == 0, process.stderr
    assert len(calls) == 1
    assert calls[0].startswith('pr list ')
    assert updated_content == ''
    assert branch_read == ''


def test_existing_release_branch_with_old_file_is_updated_without_ref_creation(tmp_path: Path):
    process, calls, body, updated_content, branch_read = _run_bump(tmp_path, '0.54.0', '0.55.0', branch_exists=True)
    expected_content = branch_read.replace('version: "0.54.0"', 'version: "0.55.0"', 1)
    puts = [call for call in calls if '-X PUT' in call]

    assert process.returncode == 0, process.stderr
    assert branch_read.startswith('---\n') and 'version: "0.54.0"' in branch_read
    assert not any('/git/refs ' in call for call in calls)
    assert len(puts) == 1
    assert 'sha=file-sha' in puts[0]
    assert updated_content == expected_content
    assert sum(call.startswith('pr create ') for call in calls) == 1
    assert 'from `0.54.0` to `0.55.0`' in body


def test_existing_release_branch_with_current_file_skips_put_and_creates_pull_request(tmp_path: Path):
    current_content = """---
name: Pydantic
engine:
  id: pydantic-ai
  version: "0.55.0"
runtimes:
  python:
    version: "3.13"
---
engine:
  version: "9.9.9"
"""
    process, calls, _, updated_content, branch_read = _run_bump(
        tmp_path, '0.54.0', '0.55.0', branch_exists=True, branch_content=current_content
    )

    assert process.returncode == 0, process.stderr
    assert branch_read == current_content
    assert not any('/git/refs ' in call or '-X PUT' in call for call in calls)
    assert updated_content == ''
    assert sum(call.startswith('pr create ') for call in calls) == 1


@pytest.mark.parametrize(
    ('pinned', 'released'),
    [
        ('0.54.0', '0.54.1'),
        ('0.54.0', '0.55.0'),
        ('0.9.9', '0.10.0'),
        ('0.9.99', '0.10.0'),
        ('1.99.99', '2.0.0'),
    ],
)
def test_newer_release_bumps_from_fetched_snapshot_and_uses_numeric_ordering(
    tmp_path: Path, pinned: str, released: str
):
    if pinned == '0.9.9':
        version_literal = f"'{pinned}'"
    elif pinned == '0.9.99':
        version_literal = pinned
    else:
        version_literal = None
    process, calls, body, updated_content, _ = _run_bump(
        tmp_path, pinned, released, stale_pinned=released, snapshot_version=version_literal
    )

    assert process.returncode == 0, process.stderr
    assert sum('-X PUT' in call for call in calls) == 1
    assert sum(call.startswith('pr create ') for call in calls) == 1
    assert f'from `{pinned}` to `{released}`' in body
    original = f"""---
name: Pydantic
engine:
  id: pydantic-ai
  version: "{pinned}"
runtimes:
  python:
    version: "3.13"
---
engine:
  version: "9.9.9"
"""
    updated = original.replace(f'  version: "{pinned}"', f'  version: "{released}"', 1)
    assert updated_content == updated


@pytest.mark.parametrize('version', ['0.55.0rc1', '0.55.0.dev1', '0.55.0+local'])
def test_non_final_releases_do_not_call_gh(tmp_path: Path, version: str):
    process, calls, _, _, _ = _run_bump(tmp_path, '0.54.0', '0.55.0', tag=f'v{version}')

    assert process.returncode == 0, process.stderr
    assert calls == []


@pytest.mark.parametrize(('pinned', 'released'), [('bad', '0.55.0'), ('0.54.0', '0.55.0rc1')])
def test_unexpected_version_shape_fails_closed_before_writes(tmp_path: Path, pinned: str, released: str):
    process, calls, _, _, _ = _run_bump(tmp_path, pinned, released)

    assert process.returncode != 0
    assert 'numeric' in process.stderr.lower() or 'numeric' in process.stdout.lower()
    assert not any('-X PUT' in call or call.startswith('pr create ') or '/git/refs ' in call for call in calls)
