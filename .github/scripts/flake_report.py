"""File each test that failed and then passed in one CI job as a `flake` issue.

A job that reruns the tests it failed, as a merge-queue test job does through `pytest_rerun_once.sh` and the nightly
shuffled run does (`tests-shuffled-nightly.yml`), writes one JUnit report per pytest session and uploads them as one
`junit-<job>-<index>` artifact per leg. A test that one report of a leg lists as failed or errored and another lists
as passed gave two outcomes on the same commit: it is flaky. A test that failed in one leg and passed in another is
not, because legs differ in Python and install.

With `--fails-on-rerun`, which the nightly passes, a test that two reports of a leg list as failed and none as passed
is filed the same way, as one that fails on rerun too. The merge queue passes no flag: there a test that fails its
rerun fails its job, and is no flake to file.

Each flaky test gets one open issue titled `Flaky <node id>` and labeled `flake`, the convention of "Flake claims"
in `.agents/skills/pushing-commits-to-the-repo/SKILL.md`:

- A test with an open issue of that exact title and label gets one comment per run.
- A test without one gets a new issue. A run files at most `_NEW_ISSUE_LIMIT`, in node id order; the rest are logged
  and wait for a later run that sees them flake.

The reports come from jobs that ran the code under test, so they are untrusted. A node id keeps a test's path only
when it is a file under `tests/` in this checkout, and its class and test names only when they are identifier-shaped.
Failure text reaches issues only fenced and truncated.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from pydantic import BaseModel, TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))
import issue_pr_attention_monitor as attention
import latest_versions_canary_report as junit

_LABEL = 'flake'
_TITLE_PREFIX = 'Flaky '
_NEW_ISSUE_LIMIT = 3
_DEFERRED_LISTING_LIMIT = 20
_OPEN_PAGE_LIMIT = 10
# GitHub caps an issue title at 256 characters.
_NODE_ID_LIMIT = 256 - len(_TITLE_PREFIX)
# The name each test job gives its JUnit artifact: `junit-<job id>-<strategy.job-index>`.
_ARTIFACT = re.compile(r'junit-([a-z0-9][a-z0-9-]{0,99})-(\d{1,4})', re.ASCII)
_PATH = re.compile(r'tests/(?:[\w-]{1,100}/){0,10}[\w-]{1,100}\.py', re.ASCII)
_NAME = re.compile(r'[A-Za-z_]\w{0,99}', re.ASCII)
_TEST = re.compile(r'(?P<name>[A-Za-z_]\w{0,99})(?P<params>\[.*\])?', re.ASCII | re.DOTALL)
# A parametrization id stays in the node id only when it holds nothing GitHub reads as markup, a mention or a reference.
_PARAMS = re.compile(r'\[[\w .+=,:/-]{1,100}\]', re.ASCII)


class Issue(BaseModel):
    """The fields this script reads from a GitHub issue."""

    number: int
    title: str = ''
    pull_request: object = None


_ISSUES = TypeAdapter(list[Issue])


@dataclass
class Flake:
    """How a test flaked: each leg it flaked in, mapped to the first line of its failure there."""

    legs: dict[str, str]
    # It failed its rerun too, in every one of `legs`.
    fails_on_rerun: bool = False


@dataclass
class Run:
    """The workflow run whose reports are being filed."""

    repo: str
    run_id: int
    server_url: str
    # The `pytest-randomly` seed the run's first pytest session shuffled the tests with, when it shuffled them.
    seed: int | None = None

    @property
    def link(self) -> str:
        return f'[run {self.run_id}]({self.server_url}/{self.repo}/actions/runs/{self.run_id})'


def collect(reports: Path, root: Path, *, fails_on_rerun: bool = False) -> dict[str, Flake]:
    """Each flaky test's node id, mapped to how it flaked; with `fails_on_rerun`, also each test that failed its rerun.

    A test that flaked in one leg and failed its rerun in another is filed as flaky, with the legs it flaked in.
    """
    flakes: dict[str, Flake] = {}
    failed_on_rerun: dict[str, dict[str, str]] = {}
    artifacts = sorted(path for path in reports.iterdir() if path.is_dir()) if reports.is_dir() else []
    for artifact in artifacts:
        if (match := _ARTIFACT.fullmatch(artifact.name)) is None:
            continue
        failed: dict[str, str] = {}
        # How many reports list each test as failed: a test that fails in call and in teardown is listed twice in one.
        failures: Counter[str] = Counter()
        passed: set[str] = set()
        for report in sorted(artifact.rglob('*.xml')):
            failed_here: set[str] = set()
            try:
                for case in junit.testcases(report):
                    if (node_id := issue_node_id(case.get('classname', ''), case.get('name', ''), root)) is None:
                        continue
                    outcomes = [result for result in case if result.tag in ('failure', 'error', 'skipped')]
                    failure = next((result for result in outcomes if result.tag != 'skipped'), None)
                    if not outcomes:
                        passed.add(node_id)
                    elif failure is not None:
                        failed.setdefault(node_id, junit.first_line(failure.get('message') or failure.text or ''))
                        failed_here.add(node_id)
            except (OSError, ElementTree.ParseError) as error:
                # A report cut short still names the outcomes before the cut.
                print(f'::warning::{junit.first_line(f"{report.relative_to(reports)} read only in part: {error}")}')
            failures.update(failed_here)
        leg = f'{match.group(1)} leg {match.group(2)}'
        for node_id in sorted(failed.keys() & passed):
            flakes.setdefault(node_id, Flake({})).legs[leg] = failed[node_id]
        if fails_on_rerun:
            for node_id in sorted(failed.keys() - passed):
                if failures[node_id] > 1:
                    failed_on_rerun.setdefault(node_id, {})[leg] = failed[node_id]
    for node_id, legs in failed_on_rerun.items():
        flakes.setdefault(node_id, Flake(legs, fails_on_rerun=True))
    return flakes


def issue_node_id(classname: str, name: str, root: Path) -> str | None:
    """The node id a flake issue names for the test a report names, or `None` when the report's names can't make one.

    A parametrization id that `_PARAMS` rejects is left out, so every such case of the test shares one issue.
    """
    node_id = junit.collected_node_id(classname, name, root)
    if node_id is None:
        return None
    path, *classes, test = node_id.split('::')
    match = _TEST.fullmatch(test)
    if match is None or not _PATH.fullmatch(path) or not all(_NAME.fullmatch(part) for part in classes):
        return None
    base = '::'.join([path, *classes, match.group('name')])
    params = match.group('params')
    if params and _PARAMS.fullmatch(params) and len(base + params) <= _NODE_ID_LIMIT:
        return base + params
    return base if len(base) <= _NODE_ID_LIMIT else None


def _evidence(run: Run, flake: Flake) -> str:
    text = junit.fenced([f'{leg}: {message}' for leg, message in flake.legs.items()])
    if run.seed is not None:
        text += (
            f'\n\nThe first run shuffled the tests with `-p randomly --randomly-seed={run.seed}` and the rerun kept '
            'their default order'
        )
        text += '.' if flake.fails_on_rerun else ', so the failure can depend on the order.'
    return text


def reconcile(client: attention.GitHubClient, run: Run, flakes: Mapping[str, Flake]) -> list[str]:
    """Comment on each flaky test's open issue, and file one for each test that has none; return a log."""
    pages, listed_all = client.first_pages(
        f'/repos/{run.repo}/issues?state=open&labels={_LABEL}', count=_OPEN_PAGE_LIMIT
    )
    if not listed_all:
        raise RuntimeError('Too many open flake issues to rule out an existing one')
    issues = _ISSUES.validate_python(pages)
    open_issues: dict[str, int] = {issue.title: issue.number for issue in issues if issue.pull_request is None}
    log: list[str] = []
    new: list[str] = []
    for node_id, flake in sorted(flakes.items()):
        if (number := open_issues.get(f'{_TITLE_PREFIX}{node_id}')) is None:
            new.append(node_id)
            continue
        outcome = (
            f'Failed in {run.link}, and fails on rerun too'
            if flake.fails_on_rerun
            else f'Failed and then passed again in {run.link}'
        )
        body = f'{outcome}.\n\n{_evidence(run, flake)}\n'
        client.post(f'/repos/{run.repo}/issues/{number}/comments', {'body': body})
        log.append(f'#{number}: noted that `{node_id}` flaked again')
    for node_id in new[:_NEW_ISSUE_LIMIT]:
        flake = flakes[node_id]
        outcome = (
            f'failed in {run.link}, and fails on rerun too'
            if flake.fails_on_rerun
            else f'failed and then passed when its job reran it, in {run.link}'
        )
        body = (
            f'`{node_id}` {outcome}.\n\n'
            f'{_evidence(run, flake)}\n\n'
            'Claim this issue before you fix the flake, as "Flake claims" in '
            '`.agents/skills/pushing-commits-to-the-repo/SKILL.md` says.\n'
        )
        payload = {'title': f'{_TITLE_PREFIX}{node_id}', 'body': body, 'labels': [_LABEL]}
        created = Issue.model_validate(client.post(f'/repos/{run.repo}/issues', payload))
        log.append(f'#{created.number}: filed `{node_id}`')
    if deferred := new[_NEW_ISSUE_LIMIT:]:
        listed = ', '.join(f'`{node_id}`' for node_id in deferred[:_DEFERRED_LISTING_LIMIT])
        more = f' and {len(deferred) - _DEFERRED_LISTING_LIMIT} more' if len(deferred) > _DEFERRED_LISTING_LIMIT else ''
        log.append(f'{len(deferred)} flaky tests wait for a later run (limit {_NEW_ISSUE_LIMIT}): {listed}{more}')
    return log


def main(argv: list[str] | None = None) -> int:
    """File the flaky tests in the reports of the run the GitHub Actions environment names."""
    parser = argparse.ArgumentParser(description='File each test that failed and then passed in one CI job.')
    parser.add_argument('--reports', type=Path, required=True, help='directory of downloaded `junit-*` artifacts')
    parser.add_argument('--seed', type=int, help='the `pytest-randomly` seed the first pytest session shuffled with')
    parser.add_argument(
        '--fails-on-rerun', action='store_true', help='also file each test that failed its rerun too, as a flake'
    )
    parser.add_argument('--dry-run', action='store_true', help='list the flaky tests without reading or writing GitHub')
    args = parser.parse_args(argv)
    reports: Path = args.reports
    seed: int | None = args.seed
    dry_run: bool = args.dry_run
    fails_on_rerun: bool = args.fails_on_rerun
    flakes = collect(reports, Path(os.environ.get('GITHUB_WORKSPACE', '.')), fails_on_rerun=fails_on_rerun)
    if dry_run:
        log = [
            f'`{node_id}` {"failed its rerun too" if flake.fails_on_rerun else "flaked"} in {", ".join(flake.legs)}'
            for node_id, flake in flakes.items()
        ]
    else:
        token = os.environ.get('GITHUB_TOKEN')
        if not token:
            print('GITHUB_TOKEN is required', file=sys.stderr)
            return 1
        run = Run(
            repo=os.environ['GITHUB_REPOSITORY'],
            run_id=int(os.environ['GITHUB_RUN_ID']),
            server_url=os.environ.get('GITHUB_SERVER_URL', 'https://github.com'),
            seed=seed,
        )
        log = reconcile(attention.GitHubClient(token), run, flakes)
    for line in log or ['no flaky test']:
        print(line)
    if summary := os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(summary).open('a', encoding='utf-8') as file:
            file.writelines(f'- {line}\n' for line in log)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
