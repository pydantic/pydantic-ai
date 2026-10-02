#!/usr/bin/env python3
"""Collect pinned test-quality evidence and publish its advisory check."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import dropwhile
from pathlib import Path
from typing import Literal

import tomllib
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

_API = 'https://api.github.com'
_CHECK_NAME = 'Test Quality Review'
_MAX_CHECK_SUMMARY_BYTES = 65_535
_TEST_NAME = re.compile(r'(?:test_.*|.*_test)\.py\Z')
_PYTHON_COMMAND = re.compile(r'python(?:3(?:\.\d+)?)?\Z')
_UV_VALUE_OPTIONS = {
    '--directory',
    '--extra',
    '--group',
    '--package',
    '--project',
    '--python',
    '--with',
    '--with-editable',
    '--with-requirements',
}
_INVENTORY = Path('.test-quality-context/candidate-inventory.json')
_CI_EVIDENCE = Path('.test-quality-context/ci-evidence.json')
_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_OBJECTS_ADAPTER = TypeAdapter(list[object])


class ReportEntry(BaseModel):
    """One evidenced guarantee classification supplied by the agent."""

    model_config = ConfigDict(extra='forbid', strict=True)

    path: str
    guarantee: str
    outcome: Literal[
        'useful_protection_added',
        'preserved_or_strengthened',
        'justified_removal',
        'changes_needed',
        'inconclusive',
    ]
    evidence: str
    action: str

    @field_validator('path', 'guarantee', 'evidence', 'action')
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('must not be empty')
        return value


class Report(BaseModel):
    """The complete report envelope passed through the safe output."""

    model_config = ConfigDict(extra='forbid', strict=True)

    entries: list[ReportEntry] = Field(min_length=1)


class Candidate(BaseModel):
    """A changed file with a potential test-protection or test-selection signal."""

    model_config = ConfigDict(extra='forbid', strict=True)

    path: str
    status: str
    previous_filename: str | None = None
    relevant_paths: list[str]


class ReviewContext(BaseModel):
    """Trusted identity and evidence keys pinned before inference."""

    model_config = ConfigDict(extra='forbid', strict=True)

    version: Literal[1]
    complete: bool
    reason: str
    repository: str
    pr_number: int
    base_sha: str
    head_sha: str
    merge_base_sha: str
    workflow_version: str
    ci_run_url: str
    review_run_url: str
    candidates: list[Candidate]


def _path_is_test_candidate(path: str) -> bool:
    parts = path.split('/')
    if any(part in {'tests', 'test', 'cassettes', 'snapshots'} for part in parts[:-1]):
        return True
    if Path(path).name == 'conftest.py' or _TEST_NAME.fullmatch(Path(path).name) is not None:
        return True
    if path.startswith('.github/scripts/') and path.endswith('.py'):
        return True
    return path in {
        '.github/workflows/ci.yml',
        'Makefile',
        'pytest.ini',
        'tox.ini',
    }


def _mapping(value: object) -> dict[str, object] | None:
    """Validate an untrusted JSON object without leaving unknown key or value types."""
    if not isinstance(value, dict):
        return None
    try:
        return _MAPPING_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return None


def _objects(value: object) -> list[object] | None:
    """Validate an untrusted JSON array with elements exposed only as objects."""
    if not isinstance(value, list):
        return None
    try:
        return _OBJECTS_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return None


def _relevant_pyproject_changed(before: str, after: str) -> bool:
    """Return whether pytest or coverage configuration changed in a valid TOML file."""
    old_data = tomllib.loads(before)
    new_data = tomllib.loads(after)
    old_tool = _mapping(old_data.get('tool', {}))
    new_tool = _mapping(new_data.get('tool', {}))
    if old_tool is None or new_tool is None:
        return False
    for section in ('pytest', 'coverage'):
        if old_tool.get(section) != new_tool.get(section):
            return True
    return False


def _indented_yaml_block(lines: list[str], start: int, parent_indent: int) -> tuple[list[str], int]:
    """Return a YAML block scalar's indented body and the next unconsumed line."""
    block: list[str] = []
    while start < len(lines):
        line = lines[start]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped and indent <= parent_indent:
            break
        block.append(line)
        start += 1
    return block, start


def _workflow_may_select_tests(source: str) -> bool:
    """Find test commands and local reusable-workflow calls that may select tests."""
    lines = source.splitlines()
    commands: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if not stripped or stripped.startswith('#'):
            index += 1
            continue
        stripped = stripped.removeprefix('- ')
        is_uses = stripped.startswith('uses:') and (len(stripped) == 5 or stripped[5].isspace())
        if is_uses:
            value = stripped[5:].split('#', 1)[0].strip().strip('\'"')
            if value.startswith('./.github/workflows/') and value.endswith(('.yml', '.yaml')):
                return True
            index += 1
            continue
        is_run = stripped.startswith('run:') and (len(stripped) == 4 or stripped[4].isspace())
        if not is_run:
            _, separator, value = stripped.partition(':')
            if separator and value.strip().startswith(('|', '>')):
                _, index = _indented_yaml_block(lines, index + 1, indent)
                continue
            index += 1
            continue
        value = stripped[4:].strip()
        index += 1
        if value.startswith(('|', '>')):
            block, index = _indented_yaml_block(lines, index, indent)
            commands.extend(block)
        else:
            commands.append(value)

    pending = ''
    for line in [*commands, '']:
        command = line.strip()
        if not command or command.startswith('#'):
            continue
        if pending:
            command = f'{pending} {command}'
            pending = ''
        if command.endswith('\\'):
            pending = command[:-1].rstrip()
            continue
        for segment in re.split(r'&&|\|\||[;|]', command):
            parts = list(dropwhile(lambda token: re.match(r'[A-Za-z_][A-Za-z0-9_]*=', token), segment.strip().split()))
            command_index = 2 if parts[:2] == ['uv', 'run'] else 0
            while command_index < len(parts) and parts[command_index].startswith('--'):
                option, separator, _ = parts[command_index].partition('=')
                command_index += 1 + int((option in _UV_VALUE_OPTIONS) & (not separator))
            command_name = parts[command_index] if command_index < len(parts) else ''
            target = parts[command_index + 1] if command_index + 1 < len(parts) else ''
            is_test_command = command_name in {'pytest', 'tox', 'nox', 'unittest'}
            is_test_command |= parts[command_index : command_index + 4] == ['coverage', 'run', '-m', 'pytest']
            is_test_command |= bool(_PYTHON_COMMAND.fullmatch(command_name)) & (
                parts[command_index + 1 : command_index + 3] in (['-m', 'pytest'], ['-m', 'unittest'])
            )
            is_test_command |= (command_name == 'make') & (
                target in {'test', 'testcov'} or target.startswith(('test-', 'integration-'))
            )
            if is_test_command:
                return True
    return False


def build_candidates(
    files: Iterable[Mapping[str, object]],
    *,
    pyproject_before: str | None = None,
    pyproject_after: str | None = None,
    workflow_contents: Mapping[str, tuple[str | None, str | None]] | None = None,
) -> tuple[list[Candidate], bool, str]:
    """Select possible test or test-selection changes, failing closed on incomplete metadata."""
    candidates: list[Candidate] = []
    seen_paths: set[str] = set()
    for file_data in files:
        path = file_data.get('filename')
        status = file_data.get('status')
        previous = file_data.get('previous_filename')
        if not isinstance(path, str) or not path or not isinstance(status, str) or not status:
            return [], False, 'pull-request file metadata is incomplete'
        if previous is not None and not isinstance(previous, str):
            return [], False, 'pull-request rename metadata is incomplete'
        names = [path]
        if isinstance(previous, str) and previous:
            names.insert(0, previous)
        relevant_paths = list(dict.fromkeys(name for name in names if _path_is_test_candidate(name)))
        workflow_paths = [
            name
            for name in names
            if name.startswith('.github/workflows/')
            and name.endswith(('.yml', '.yaml'))
            and not name.endswith('.lock.yml')
            and name != '.github/workflows/ci.yml'
        ]
        for workflow_path in workflow_paths:
            contents = workflow_contents.get(workflow_path) if workflow_contents is not None else None
            if contents is None:
                return [], False, 'pinned workflow content could not be compared'
            if contents == (None, None):
                return [], False, 'pinned workflow content is missing from both revisions'
            if any(content is not None and _workflow_may_select_tests(content) for content in contents):
                relevant_paths.append(workflow_path)
        if path == 'pyproject.toml' or previous == 'pyproject.toml':
            if pyproject_before is None or pyproject_after is None:
                return [], False, 'pinned pyproject.toml sections could not be compared'
            try:
                if _relevant_pyproject_changed(pyproject_before, pyproject_after):
                    relevant_paths.append('pyproject.toml')
            except (tomllib.TOMLDecodeError, TypeError):
                return [], False, 'pinned pyproject.toml could not be parsed'
        if relevant_paths and path not in seen_paths:
            candidates.append(
                Candidate(
                    path=path,
                    status=status,
                    previous_filename=previous,
                    relevant_paths=list(dict.fromkeys(relevant_paths)),
                )
            )
            seen_paths.add(path)
    return candidates, True, 'candidate inventory complete'


def validate_report(raw_report: str | None, candidate_paths: Iterable[str]) -> Report:
    """Validate the model report against the immutable candidate allowlist."""
    if raw_report is None or not raw_report.strip():
        raise ValueError('report is missing')
    report = Report.model_validate_json(raw_report)
    allowed = set(candidate_paths)
    reported = {entry.path for entry in report.entries}
    unknown = reported - allowed
    missing = allowed - reported
    if unknown:
        raise ValueError(f'report contains unknown candidate paths: {", ".join(sorted(unknown))}')
    if missing:
        raise ValueError(f'report omits candidate paths: {", ".join(sorted(missing))}')
    return report


def summarize(report: Report) -> tuple[str, str]:
    """Aggregate outcomes and render each guarantee without hiding mixed results."""
    outcomes = {entry.outcome for entry in report.entries}
    title = (
        'Changes needed'
        if 'changes_needed' in outcomes
        else 'Inconclusive'
        if 'inconclusive' in outcomes
        else 'Protection accounted for'
    )
    lines = ['## Test Quality Review', '']
    for entry in report.entries:
        lines.extend(
            [
                f'### `{entry.path}` — {entry.outcome.replace("_", " ")}',
                '',
                f'**Guarantee:** {entry.guarantee}',
                f'**Evidence:** {entry.evidence}',
                f'**Action:** {entry.action}',
                '',
            ]
        )
    return title, '\n'.join(lines).rstrip()


def inconclusive_report(context: ReviewContext, reason: str) -> tuple[str, str]:
    """Produce an explicit neutral result when evidence or model output is unavailable."""
    lines = ['## Test Quality Review', '', f'Inconclusive: {reason}', '']
    for candidate in context.candidates:
        lines.extend(
            [
                f'### `{candidate.path}` — inconclusive',
                '',
                '**Guarantee:** Not assessed because a complete report was unavailable.',
                f'**Evidence:** {reason}',
                '**Action:** No change; rerun with complete evidence.',
                '',
            ]
        )
    if not context.candidates:
        lines.append('No candidate inventory was available for assessment.')
    return 'Inconclusive', '\n'.join(lines).rstrip()


def check_payload(
    *,
    head_sha: str,
    title: str,
    summary: str,
    details_url: str,
    base_sha: str,
    workflow_version: str,
) -> dict[str, object]:
    """Build a neutral Check Run pinned to the trusted head SHA."""
    identity = f'<!-- test-quality-review:v1 base={base_sha} head={head_sha} workflow={workflow_version} -->'
    rendered_summary = f'{identity}\n\n{summary}'
    if len(rendered_summary.encode('utf-8')) > _MAX_CHECK_SUMMARY_BYTES:
        title = 'Inconclusive'
        rendered_summary = (
            f'{identity}\n\n'
            'The complete review exceeds the Check Run summary limit, so no guarantee classifications are shown here. '
            f'The complete validated result is retained in the [review run and its artifact]({details_url}).'
        )
    return {
        'name': _CHECK_NAME,
        'head_sha': head_sha,
        'status': 'completed',
        'conclusion': 'neutral',
        'details_url': details_url,
        'output': {'title': title, 'summary': rendered_summary},
    }


def has_completed_report(
    check_runs: Iterable[Mapping[str, object]], *, base_sha: str, head_sha: str, workflow_version: str
) -> bool:
    """Recognize only a completed substantive result for the same pinned identity."""
    marker = f'<!-- test-quality-review:v1 base={base_sha} head={head_sha} workflow={workflow_version} -->'
    for check in check_runs:
        output = _mapping(check.get('output'))
        if (
            check.get('name') == _CHECK_NAME
            and check.get('status') == 'completed'
            and output is not None
            and output.get('title') in {'Protection accounted for', 'Changes needed', 'No candidate changes'}
            and marker in str(output.get('summary', ''))
            and check.get('conclusion') == 'neutral'
        ):
            return True
    return False


class GitHub:
    """Small GitHub REST client used only by the trusted host controller."""

    def __init__(self, token: str, repository: str):
        self.token = token
        self.repository = repository

    def request(self, path: str, *, method: str = 'GET', body: Mapping[str, object] | None = None) -> object:
        url = f'{_API}/repos/{self.repository}/{path.lstrip("/")}'
        payload = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            url,
            data=payload,
            method=method,
            headers={
                'Accept': 'application/vnd.github+json',
                'Authorization': f'Bearer {self.token}',
                'X-GitHub-Api-Version': '2022-11-28',
                **({'Content-Type': 'application/json'} if payload is not None else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'GitHub API request failed: {method} {path}: {exc}') from exc

    def paginated(self, path: str, *, collection: str | None = None) -> list[object]:
        results: list[object] = []
        page = 1
        while True:
            query = urllib.parse.urlencode({'per_page': 100, 'page': page})
            response = self.request(f'{path}?{query}')
            if collection is None:
                items = _objects(response)
                if items is None:
                    raise ValueError(f'expected a list from GitHub API: {path}')
            else:
                container = _mapping(response)
                items = _objects(container.get(collection)) if container is not None else None
                if items is None:
                    raise ValueError(f'expected a {collection} list from GitHub API: {path}')
            results.extend(items)
            if len(items) < 100:
                return results
            page += 1


def _git_text(*args: str) -> str:
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def _context_from_file(path: Path = _INVENTORY) -> ReviewContext:
    return ReviewContext.model_validate_json(path.read_text(encoding='utf-8'))


def _check_runs(github: GitHub, head_sha: str) -> list[Mapping[str, object]]:
    values = github.paginated(f'commits/{head_sha}/check-runs', collection='check_runs')
    check_runs: list[Mapping[str, object]] = []
    for value in values:
        if (check_run := _mapping(value)) is not None:
            check_runs.append(check_run)
    return check_runs


def _set_output(name: str, value: str) -> None:
    output_path = os.environ.get('GITHUB_OUTPUT')
    if output_path:
        with Path(output_path).open('a', encoding='utf-8') as output:
            output.write(f'{name}={value}\n')


def _check_or_skip(
    github: GitHub,
    *,
    head_sha: str,
    title: str,
    summary: str,
    details_url: str,
    base_sha: str,
    workflow_version: str,
) -> None:
    github.request(
        'check-runs',
        method='POST',
        body=check_payload(
            head_sha=head_sha,
            title=title,
            summary=summary,
            details_url=details_url,
            base_sha=base_sha,
            workflow_version=workflow_version,
        ),
    )


def _object_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f'GitHub response is missing {key}')
    return value


def _pr_number_from_context() -> int | None:
    event_name = os.environ.get('EVENT_NAME', '')
    if event_name == 'workflow_dispatch':
        try:
            context = json.loads(os.environ.get('AW_CONTEXT', '{}'))
        except json.JSONDecodeError:
            return None
        context = _mapping(context)
        if context is None or context.get('item_type') != 'pull_request':
            return None
        number = context.get('item_number')
        return number if isinstance(number, int) and not isinstance(number, bool) else None
    if event_name != 'workflow_run':
        return None
    try:
        return int(os.environ.get('RUN_ID', ''))
    except ValueError:
        return None


def _resolve_pr_number(github: GitHub) -> int | None:
    if os.environ.get('EVENT_NAME') == 'workflow_dispatch':
        return _pr_number_from_context()
    if os.environ.get('EVENT_NAME') != 'workflow_run':
        return None
    trigger_sha = os.environ.get('RUN_HEAD_SHA', '')
    branch = os.environ.get('RUN_HEAD_BRANCH', '')
    if not trigger_sha or not branch:
        return None
    matches = github.paginated(f'commits/{trigger_sha}/pulls')
    numbers: set[int] = set()
    for value in matches:
        item = _mapping(value)
        head = _mapping(item.get('head')) if item is not None else None
        number = item.get('number') if item is not None else None
        if item is not None and item.get('state') == 'open' and head is not None and head.get('ref') == branch:
            if isinstance(number, int) and not isinstance(number, bool):
                numbers.add(number)
    return next(iter(numbers)) if len(numbers) == 1 else None


def _find_ci_run(github: GitHub, head_sha: str) -> Mapping[str, object] | None:
    response = _mapping(
        github.request(f'actions/workflows/ci.yml/runs?head_sha={head_sha}&event=pull_request&per_page=100')
    )
    raw_runs = _objects(response.get('workflow_runs')) if response is not None else None
    if raw_runs is None:
        raise ValueError('CI workflow run response is incomplete')
    matches: list[dict[str, object]] = []
    for value in raw_runs:
        run = _mapping(value)
        if (
            run is not None
            and run.get('head_sha') == head_sha
            and run.get('event') == 'pull_request'
            and run.get('status') == 'completed'
            and run.get('conclusion') == 'success'
        ):
            matches.append(run)
    return max(matches, key=lambda item: str(item.get('updated_at', ''))) if matches else None


def _ci_jobs(github: GitHub, run_id: int) -> list[dict[str, object]]:
    raw_jobs = github.paginated(f'actions/runs/{run_id}/jobs', collection='jobs')
    jobs: list[dict[str, object]] = []
    for value in raw_jobs:
        job = _mapping(value)
        if job is None:
            raise ValueError('CI job metadata is incomplete')
        steps = _objects(job.get('steps'))
        if steps is None:
            raise ValueError('CI step metadata is incomplete')
        step_data = [_mapping(step) for step in steps]
        jobs.append(
            {
                'name': _object_string(job, 'name'),
                'conclusion': _object_string(job, 'conclusion'),
                'html_url': _object_string(job, 'html_url'),
                'steps': [
                    {
                        'name': _object_string(step, 'name'),
                        'conclusion': _object_string(step, 'conclusion'),
                    }
                    for step in step_data
                    if step is not None
                ],
            }
        )
    return jobs


def _fetch_git_objects(repository: str, base_sha: str, head_sha: str) -> str:
    url = f'https://github.com/{repository}.git'
    subprocess.run(['git', 'fetch', '--no-tags', url, base_sha, head_sha], check=True, capture_output=True, text=True)
    return _git_text('merge-base', base_sha, head_sha).strip()


@dataclass(frozen=True)
class PinnedReview:
    """Trusted PR and CI identity resolved before any agent work."""

    pr_number: int
    pr: Mapping[str, object]
    head_sha: str
    base_sha: str
    base_ref: str
    run_id: int
    ci_run: Mapping[str, object]
    ci_run_url: str
    review_run_url: str
    workflow_version: str


def _skip_review(
    github: GitHub,
    reason: str,
    *,
    review_run_url: str,
    workflow_version: str,
    marker_sha: str = '',
) -> None:
    """Record a pre-inference skip, with a neutral check when its head is known."""
    print(reason)
    if marker_sha:
        _check_or_skip(
            github,
            head_sha=marker_sha,
            title='Skipped',
            summary=f'Test quality assessment skipped: {reason}',
            details_url=review_run_url,
            base_sha='',
            workflow_version=workflow_version,
        )
    _set_output('eligible', 'false')
    _set_output('reason', reason)


def _pin_review(github: GitHub) -> PinnedReview | None:
    """Resolve one same-repository PR and its successful current-head CI run."""
    repository = os.environ['REPOSITORY']
    event_name = os.environ.get('EVENT_NAME', '')
    original_sha = os.environ.get('RUN_HEAD_SHA', '') if event_name == 'workflow_run' else ''
    ci_run_url = os.environ.get('CI_RUN_URL', '')
    review_run_url = os.environ.get('REVIEW_RUN_URL', '')
    workflow_version = os.environ.get('WORKFLOW_VERSION', '')

    def skip(reason: str, marker_sha: str = '') -> None:
        _skip_review(
            github,
            reason,
            review_run_url=review_run_url,
            workflow_version=workflow_version,
            marker_sha=marker_sha,
        )

    if event_name == 'workflow_run' and (
        os.environ.get('RUN_EVENT') != 'pull_request'
        or os.environ.get('RUN_CONCLUSION') != 'success'
        or os.environ.get('RUN_HEAD_REPOSITORY') != repository
    ):
        skip('triggering CI did not succeed for a same-repository pull request')
        return None
    pr_number = _resolve_pr_number(github)
    if pr_number is None:
        skip('could not resolve exactly one pull request from trusted trigger context')
        return None

    pr = _mapping(github.request(f'pulls/{pr_number}'))
    if pr is None:
        skip('pull-request response is incomplete', original_sha)
        return None
    head, base = _mapping(pr.get('head')), _mapping(pr.get('base'))
    if head is None or base is None:
        skip('pull-request head or base metadata is incomplete', original_sha)
        return None
    head_sha, base_sha = _object_string(head, 'sha'), _object_string(base, 'sha')
    base_ref = _object_string(base, 'ref')
    head_repo, base_repo = _mapping(head.get('repo')), _mapping(base.get('repo'))
    if pr.get('state') != 'open' or pr.get('draft') is not False:
        skip('pull request is closed or draft', original_sha or head_sha)
        return None
    if (
        head_repo is None
        or head_repo.get('full_name') != repository
        or base_repo is None
        or base_repo.get('full_name') != repository
    ):
        skip('fork pull requests are outside this workflow scope', original_sha)
        return None
    if original_sha and original_sha != head_sha:
        skip(f'CI ran on {original_sha}, but the pull request now points to {head_sha}', original_sha)
        return None

    if event_name == 'workflow_run':
        try:
            run_id = int(os.environ.get('RUN_ID', ''))
        except ValueError:
            skip('triggering CI run id is missing', original_sha)
            return None
        ci_run: Mapping[str, object] = {
            'id': run_id,
            'head_sha': original_sha,
            'html_url': ci_run_url,
            'conclusion': 'success',
        }
    else:
        ci_run = _find_ci_run(github, head_sha) or {}
        run_id = ci_run.get('id')
        if not ci_run:
            skip(f'no successful CI run exists for current head {head_sha}', head_sha)
            return None
        if not isinstance(run_id, int):
            skip('successful CI run metadata is incomplete', head_sha)
            return None
        ci_run_url = _object_string(ci_run, 'html_url')
    if not head_sha or not base_sha or not workflow_version or not ci_run_url or not review_run_url:
        skip('trusted review identity is incomplete', original_sha or head_sha)
        return None
    return PinnedReview(
        pr_number=pr_number,
        pr=pr,
        head_sha=head_sha,
        base_sha=base_sha,
        base_ref=base_ref,
        run_id=run_id,
        ci_run=ci_run,
        ci_run_url=ci_run_url,
        review_run_url=review_run_url,
        workflow_version=workflow_version,
    )


def _matches_pinned_pr(pr: object, pinned: PinnedReview, repository: str) -> bool:
    """Reject a PR whose state, head, or base changed after eligibility was pinned."""
    pr = _mapping(pr)
    if pr is None or pr.get('state') != 'open' or pr.get('draft') is not False:
        return False
    head, base = _mapping(pr.get('head')), _mapping(pr.get('base'))
    if head is None or base is None:
        return False
    head_repo, base_repo = _mapping(head.get('repo')), _mapping(base.get('repo'))
    return (
        head.get('sha') == pinned.head_sha
        and base.get('sha') == pinned.base_sha
        and base.get('ref') == pinned.base_ref
        and head_repo is not None
        and head_repo.get('full_name') == repository
        and base_repo is not None
        and base_repo.get('full_name') == repository
    )


def _candidate_inventory(
    github: GitHub,
    pinned: PinnedReview,
    repository: str,
) -> tuple[list[dict[str, object]], list[Candidate], str]:
    """Build the complete changed-file inventory and collect CI job evidence."""
    jobs = _ci_jobs(github, pinned.run_id)
    raw_files = github.paginated(f'pulls/{pinned.pr_number}/files')
    files = [file for value in raw_files if (file := _mapping(value)) is not None]
    if len(files) != len(raw_files):
        raise ValueError('pull-request file metadata is incomplete')
    expected_file_count = pinned.pr.get('changed_files')
    if not isinstance(expected_file_count, int) or expected_file_count != len(files):
        raise ValueError('pull-request file listing is incomplete')
    merge_base_sha = _fetch_git_objects(repository, pinned.base_sha, pinned.head_sha)
    before = after = None
    if any(
        file.get('filename') == 'pyproject.toml' or file.get('previous_filename') == 'pyproject.toml' for file in files
    ):
        before = _git_text('show', f'{merge_base_sha}:pyproject.toml')
        after = _git_text('show', f'{pinned.head_sha}:pyproject.toml')
    changed_workflows = {
        path
        for file in files
        for path in (file.get('previous_filename'), file.get('filename'))
        if isinstance(path, str)
        and path.startswith('.github/workflows/')
        and path.endswith(('.yml', '.yaml'))
        and not path.endswith('.lock.yml')
        and path != '.github/workflows/ci.yml'
    }
    workflow_contents: dict[str, tuple[str | None, str | None]] = {}
    for path in changed_workflows:
        revisions: list[str | None] = []
        for revision in (merge_base_sha, pinned.head_sha):
            tree_paths = _git_text('ls-tree', '-r', '--name-only', revision, '--', path).splitlines()
            revisions.append(_git_text('show', f'{revision}:{path}') if path in tree_paths else None)
        if revisions == [None, None]:
            raise ValueError('changed workflow could not be found in either pinned revision')
        workflow_contents[path] = (revisions[0], revisions[1])
    candidates, complete, reason = build_candidates(
        files,
        pyproject_before=before,
        pyproject_after=after,
        workflow_contents=workflow_contents,
    )
    if not complete:
        raise ValueError(reason)
    return jobs, candidates, merge_base_sha


def _record_inconclusive_context(
    github: GitHub,
    pinned: PinnedReview,
    repository: str,
    reason: str,
    *,
    merge_base_sha: str = '',
    candidates: list[Candidate] | None = None,
) -> None:
    """Persist incomplete trusted evidence and its neutral pre-inference check."""
    context = ReviewContext(
        version=1,
        complete=False,
        reason=reason,
        repository=repository,
        pr_number=pinned.pr_number,
        base_sha=pinned.base_sha,
        head_sha=pinned.head_sha,
        merge_base_sha=merge_base_sha,
        workflow_version=pinned.workflow_version,
        ci_run_url=pinned.ci_run_url,
        review_run_url=pinned.review_run_url,
        candidates=candidates or [],
    )
    _write_json(_INVENTORY, context.model_dump(mode='json'))
    _check_or_skip(
        github,
        head_sha=pinned.head_sha,
        title='Inconclusive',
        summary=f'Candidate evidence is incomplete: {reason}\n\nCI run: {pinned.ci_run_url}',
        details_url=pinned.review_run_url,
        base_sha=pinned.base_sha,
        workflow_version=pinned.workflow_version,
    )
    _skip_review(
        github,
        f'candidate context is incomplete: {reason}',
        review_run_url=pinned.review_run_url,
        workflow_version=pinned.workflow_version,
    )


def prepare_context() -> int:
    """Resolve the authorized PR, pin evidence, and create the agent inventory."""
    repository = os.environ['REPOSITORY']
    github = GitHub(os.environ['GITHUB_TOKEN'], repository)
    pinned = _pin_review(github)
    if pinned is None:
        return 0
    try:
        if os.environ.get('EVENT_NAME') == 'workflow_run' and has_completed_report(
            _check_runs(github, pinned.head_sha),
            base_sha=pinned.base_sha,
            head_sha=pinned.head_sha,
            workflow_version=pinned.workflow_version,
        ):
            _skip_review(
                github,
                f'a completed Test Quality Review already exists for {pinned.head_sha}',
                review_run_url=pinned.review_run_url,
                workflow_version=pinned.workflow_version,
            )
            return 0
        jobs, candidates, merge_base_sha = _candidate_inventory(github, pinned, repository)
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        _record_inconclusive_context(github, pinned, repository, str(exc))
        return 0
    if not candidates:
        latest_pr = github.request(f'pulls/{pinned.pr_number}')
        if not _matches_pinned_pr(latest_pr, pinned, repository):
            _skip_review(
                github,
                'pull request changed while candidate inventory was gathered',
                review_run_url=pinned.review_run_url,
                workflow_version=pinned.workflow_version,
                marker_sha=pinned.head_sha,
            )
            return 0
        _check_or_skip(
            github,
            head_sha=pinned.head_sha,
            title='No candidate changes',
            summary=f'The pull request changes no in-scope test or test-selection files.\n\nCI run: {pinned.ci_run_url}',
            details_url=pinned.review_run_url,
            base_sha=pinned.base_sha,
            workflow_version=pinned.workflow_version,
        )
        _skip_review(
            github,
            'no in-scope candidate changes',
            review_run_url=pinned.review_run_url,
            workflow_version=pinned.workflow_version,
        )
        return 0

    try:
        script = Path('scripts/gather-pydantic-ai-review-context.sh')
        gathered = (
            script.is_file()
            and subprocess.run(
                ['bash', str(script), str(pinned.pr_number), repository, pinned.head_sha, pinned.base_sha],
                check=False,
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        current_pr = github.request(f'pulls/{pinned.pr_number}')
        if not _matches_pinned_pr(current_pr, pinned, repository):
            _skip_review(
                github,
                'pull request changed while candidate evidence was gathered',
                review_run_url=pinned.review_run_url,
                workflow_version=pinned.workflow_version,
                marker_sha=pinned.head_sha,
            )
            return 0
        if not gathered:
            raise ValueError('pinned discussion or diff collection failed')
        _write_json(_CI_EVIDENCE, {'run': pinned.ci_run, 'jobs': jobs})
        context = ReviewContext(
            version=1,
            complete=True,
            reason='candidate inventory and pinned context are complete',
            repository=repository,
            pr_number=pinned.pr_number,
            base_sha=pinned.base_sha,
            head_sha=pinned.head_sha,
            merge_base_sha=merge_base_sha,
            workflow_version=pinned.workflow_version,
            ci_run_url=pinned.ci_run_url,
            review_run_url=pinned.review_run_url,
            candidates=candidates,
        )
        _write_json(_INVENTORY, context.model_dump(mode='json'))
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        _record_inconclusive_context(
            github,
            pinned,
            repository,
            str(exc),
            merge_base_sha=merge_base_sha,
            candidates=candidates,
        )
        return 0

    _set_output('eligible', 'true')
    _set_output('reason', 'eligible')
    _set_output('pr_number', str(pinned.pr_number))
    _set_output('head_sha', pinned.head_sha)
    _set_output('base_sha', pinned.base_sha)
    _set_output('merge_base_sha', merge_base_sha)
    _set_output('ci_run_url', pinned.ci_run_url)
    return 0


def _safe_output_report(agent_output_path: Path) -> str | None:
    try:
        outer = json.loads(agent_output_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    outer = _mapping(outer)
    items = _objects(outer.get('items')) if outer is not None else None
    if items is None:
        return None
    matching: list[str] = []
    for value in items:
        item = _mapping(value)
        if item is None or item.get('type') != 'record_test_quality_review':
            continue
        report = item.get('report')
        if not isinstance(report, str):
            return None
        matching.append(report)
    return matching[0] if len(matching) == 1 else None


def validate_tool_output() -> int:
    """Validate the model output and preserve either the report or its failure."""
    context = _context_from_file()
    report_text = _safe_output_report(Path(os.environ.get('GH_AW_AGENT_OUTPUT', '')))
    try:
        report = validate_report(report_text, (candidate.path for candidate in context.candidates))
    except (ValidationError, ValueError) as exc:
        result: dict[str, object] = {'valid': False, 'reason': str(exc)}
    else:
        result = {'valid': True, 'entries': [entry.model_dump(mode='json') for entry in report.entries]}
    result['base_sha'] = context.base_sha
    result['head_sha'] = context.head_sha
    result['workflow_version'] = context.workflow_version
    _write_json(Path(os.environ.get('RESULT_PATH', 'validated-result.json')), result)
    return 0


def publish_result() -> int:
    """Publish one validated or explicitly inconclusive neutral Check Run."""
    context = _context_from_file()
    token = os.environ['GITHUB_TOKEN']
    github = GitHub(token, context.repository)
    try:
        result = json.loads(Path(os.environ.get('RESULT_PATH', 'validated-result.json')).read_text(encoding='utf-8'))
        result = _mapping(result)
        if result is None:
            raise ValueError('validated report artifact is malformed')
        if (
            not context.complete
            or result.get('valid') is not True
            or result.get('base_sha') != context.base_sha
            or result.get('head_sha') != context.head_sha
            or result.get('workflow_version') != context.workflow_version
        ):
            reason = result.get('reason')
            raise ValueError(reason if isinstance(reason, str) else 'validated report artifact is missing')
        raw_entries = result.get('entries')
        if not isinstance(raw_entries, list):
            raise ValueError('validated report artifact has no entries')
        report = Report.model_validate({'entries': raw_entries})
        candidate_paths = {candidate.path for candidate in context.candidates}
        if not candidate_paths.issubset({entry.path for entry in report.entries}):
            raise ValueError('validated report artifact omits candidate paths')
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        title, summary = inconclusive_report(context, str(exc))
    else:
        title, summary = summarize(report)
    summary = f'{summary}\n\nCI run: {context.ci_run_url}'
    github.request(
        'check-runs',
        method='POST',
        body=check_payload(
            head_sha=context.head_sha,
            title=title,
            summary=summary,
            details_url=context.review_run_url,
            base_sha=context.base_sha,
            workflow_version=context.workflow_version,
        ),
    )
    return 0


def main() -> int:
    """Run the requested trusted-controller phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'publish'))
    args = parser.parse_args()
    if args.command == 'prepare':
        return prepare_context()
    if args.command == 'validate':
        return validate_tool_output()
    if args.command == 'publish':
        return publish_result()
    return 2


if __name__ == '__main__':
    sys.exit(main())
