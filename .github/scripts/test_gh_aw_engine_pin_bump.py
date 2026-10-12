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
) -> tuple[subprocess.CompletedProcess[str], list[str], str, str]:
    workflow: object = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    jobs = WORKFLOW_ADAPTER.validate_python(workflow)['jobs']
    job = PIN_BUMP_JOB_ADAPTER.validate_python(jobs['gh-aw-engine-pin-bump'])
    guard_run = next(
        step['run']
        for step in job['steps']
        if step.get('name') == 'Check whether the release moved past the definition'
    )
    bump_run = next(step['run'] for step in job['steps'] if step.get('name') == 'Open the pin bump pull request')
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
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in ('grep', 'awk', 'cmp', 'cat', 'cp', 'tr'):
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
  "$BASE64_REAL" -D "$@"
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
  "pr list "*) ;;
  "api repos/fixture/repository/git/ref/heads/main --jq .object.sha") printf 'base-sha\\n' ;;
  *"/contents/src/pydantic_ai_harness/gh-aw/pydantic.md?ref=base-sha"*) cat "$GH_SNAPSHOT" ;;
  "api repos/fixture/repository/git/ref/heads/gh-aw-engine-pin-"*) exit 1 ;;
  "api repos/fixture/repository/git/refs "*) ;;
  *"/contents/src/pydantic_ai_harness/gh-aw/pydantic.md?ref=gh-aw-engine-pin-"*) cat "$GH_SNAPSHOT" ;;
  *"/contents/src/pydantic_ai_harness/gh-aw/pydantic.md"*"--jq .sha"*) printf 'file-sha\\n' ;;
  *"-X PUT repos/fixture/repository/contents/src/pydantic_ai_harness/gh-aw/pydantic.md"*)
    for argument in "$@"; do
      case "$argument" in
        content=*) printf '%s' "${argument#content=}" | base64 -d > "$GH_UPDATED_CONTENT" ;;
      esac
    done
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
    return process, calls, created_body, updated_content


@pytest.mark.parametrize(('pinned', 'released'), [('0.55.0', '0.54.0'), ('0.55.0', '0.55.0')])
def test_older_or_equal_release_does_not_write_even_when_job_pin_is_stale(tmp_path: Path, pinned: str, released: str):
    process, calls, _, _ = _run_bump(tmp_path, pinned, released, stale_pinned='0.53.0')

    assert process.returncode == 0, process.stderr
    assert not any('-X PUT' in call or call.startswith('pr create ') for call in calls)
    assert not any('/git/refs ' in call for call in calls)


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
    process, calls, body, updated_content = _run_bump(
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
    process, calls, _, _ = _run_bump(tmp_path, '0.54.0', '0.55.0', tag=f'v{version}')

    assert process.returncode == 0, process.stderr
    assert calls == []


@pytest.mark.parametrize(('pinned', 'released'), [('bad', '0.55.0'), ('0.54.0', '0.55.0rc1')])
def test_unexpected_version_shape_fails_closed_before_writes(tmp_path: Path, pinned: str, released: str):
    process, calls, _, _ = _run_bump(tmp_path, pinned, released)

    assert process.returncode != 0
    assert 'numeric' in process.stderr.lower() or 'numeric' in process.stdout.lower()
    assert not any('-X PUT' in call or call.startswith('pr create ') or '/git/refs ' in call for call in calls)
