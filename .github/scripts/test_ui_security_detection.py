from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter
from typing_extensions import TypedDict

WORKFLOW = Path(__file__).parents[1] / 'workflows' / 'pydantic-ai-ui-security-review.md'


class DetectStep(TypedDict):
    """A step in the detector job."""

    name: str
    run: str


class DetectJob(TypedDict):
    """The detector job definition."""

    steps: list[DetectStep]


class Workflow(TypedDict):
    """The workflow frontmatter fields used by these tests."""

    jobs: dict[str, object]


WORKFLOW_ADAPTER: TypeAdapter[Workflow] = TypeAdapter(Workflow)
DETECT_JOB_ADAPTER: TypeAdapter[DetectJob] = TypeAdapter(DetectJob)


def _detect(
    tmp_path: Path,
    changed_files: list[tuple[str, str]],
    *,
    files_api_fails: bool = False,
) -> str:
    source = WORKFLOW.read_text(encoding='utf-8')
    frontmatter = source.split('---', maxsplit=2)[1]
    workflow = WORKFLOW_ADAPTER.validate_python(yaml.safe_load(frontmatter))
    detect_job = DETECT_JOB_ADAPTER.validate_python(workflow['jobs']['detect'])
    run = next(step['run'] for step in detect_job['steps'] if step['name'] == 'Detect UI security paths in the PR')

    gh = tmp_path / 'gh'
    gh.write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *"pulls/$PR_NUMBER/files"* ]]; then
  [[ "$*" == *"--paginate"* ]] || exit 3
  [[ "${GH_FILES_API_FAILURE:-false}" != true ]] || exit 1
  if [[ "$*" == *"@tsv"* ]]; then
    cat "$GH_CHANGED_FILES_FILE"
  else
    while IFS=$'\t' read -r filename previous; do
      [[ -n "$filename" ]] || continue
      printf '%s\n' "$filename"
      if [[ -n "$previous" ]]; then
        printf '%s\n' "$previous"
      fi
    done < "$GH_CHANGED_FILES_FILE"
  fi
else
  exit 2
fi
""",
        encoding='utf-8',
    )
    gh.chmod(0o755)

    output = tmp_path / 'github-output'
    changed_files_file = tmp_path / 'changed-files.tsv'
    changed_files_file.write_text('\n'.join(f'{new}\t{old}' for new, old in changed_files) + '\n', encoding='utf-8')
    process = subprocess.run(
        ['bash', '-e', '-c', run],
        env={
            **os.environ,
            'PATH': f'{tmp_path}:{os.environ["PATH"]}',
            'GITHUB_OUTPUT': str(output),
            'GH_TOKEN': 'test-token',
            'PR_NUMBER': '123',
            'REPO': 'pydantic/pydantic-ai',
            'GH_CHANGED_FILES_FILE': str(changed_files_file),
            'GH_FILES_API_FAILURE': str(files_api_fails).lower(),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        assert not output.exists()
        return 'error'
    return output.read_text(encoding='utf-8').strip().split('=', maxsplit=1)[1]


@pytest.mark.parametrize(
    ('changed_files', 'expected'),
    [
        ([('pydantic_ai_slim/pydantic_ai/ui/_adapter.py', '')], 'true'),
        ([('src/pydantic_clai2/web-ui/src/App.tsx', '')], 'true'),
        ([('src/pydantic_clai2/pydantic_clai2/builtin_plugins/web/app.py', '')], 'true'),
        ([('src/pydantic_clai2/pydantic_clai2/cli/web.py', '')], 'true'),
        ([('src/pydantic_clai2/scripts/check_web_assets.py', '')], 'true'),
        ([('tests/clai2/test_web_keys.py', '')], 'true'),
        ([('tests/clai2/browser/test_security.py', '')], 'true'),
        ([('docs/harness/clai2.md', '')], 'true'),
        ([('docs/web.md', '')], 'true'),
        ([('src/pydantic_clai2/web-ui-old/src/App.tsx', '')], 'false'),
        ([('tests/clai2/test_cli.py', '')], 'false'),
        ([('README.md', '')], 'false'),
        ([('src/renamed.py', 'src/pydantic_clai2/web-ui/src/App.tsx')], 'true'),
        ([('src/renamed.py', 'pydantic_ai_slim/pydantic_ai/ui/_adapter.py')], 'true'),
    ],
)
def test_detects_ui_security_paths_and_renames(
    tmp_path: Path, changed_files: list[tuple[str, str]], expected: str
) -> None:
    """Detect covered paths under current and previous filenames."""
    assert _detect(tmp_path, changed_files) == expected


def test_detection_fails_closed_when_pull_request_files_cannot_be_read(tmp_path: Path) -> None:
    """Fail the detector job when GitHub cannot return the changed-file list."""
    assert _detect(tmp_path, [('README.md', '')], files_api_fails=True) == 'error'


def test_large_changed_file_list_still_detects_ui_security_paths(tmp_path: Path) -> None:
    """Drain large file lists after finding a matching path to avoid SIGPIPE."""
    changed_files = [('src/pydantic_clai2/web-ui/src/App.tsx', '')] + [
        (f'unrelated/{index:04d}/' + 'x' * 90, '') for index in range(2000)
    ]

    assert _detect(tmp_path, changed_files) == 'true'
