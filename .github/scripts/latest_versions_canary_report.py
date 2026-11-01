"""File scheduled latest-versions canary failures as deduplicated issues.

The canary installs the newest versions our declared dependency ranges allow, so
a red scheduled run means a fresh install of our packages is broken today. Each
canary leg uploads its step outcomes, a JUnit report, and the dependencies that
moved past `uv.lock`. This script reduces those reports to failure signatures
and reconciles them with the issues it filed before, one issue per signature:

- A signature that no open issue carries gets a new issue labeled for priority
  and Pydanty triage. A run files at most `_NEW_ISSUE_LIMIT`: failures that stopped a
  leg's suite first, then those with the most failing tests. The rest are logged
  and wait for a later run.
- A signature that an open issue carries gets one comment on that issue per run.
- A run in which every leg finished its test session shows which signatures pass.
  Each issue records a pass once; closing stays a maintainer decision.
- A signature whose last issue was closed gets a new issue only if a run
  recorded it passing after its last failure. A close is not proof of recovery:
  automated triage or an incomplete fix can close an issue while the canary
  keeps failing, and refiling every day would loop issues through triage. That
  closed issue gets one notice instead. A close as not planned or as a
  duplicate keeps the signature quiet.
- A re-run of an older run leaves alone every issue a newer run already updated.

The reports come from a job that ran freshly published third-party code, so they
are untrusted. A signature is built only from paths that exist in this checkout
and identifier-shaped names. Comments carry nothing else, because Pydanty reads
`github-actions` comments as vetted input. Free text reaches issue bodies only
fenced and truncated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import unicodedata
import urllib.parse
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree

from pydantic import BaseModel, TypeAdapter, ValidationError

sys.path.insert(0, str(Path(__file__).parent))
import issue_pr_attention_monitor as attention
from triage_models import Account

_MARKER = re.compile(r'<!-- pydantic-ai-latest-versions-canary:v1 (\{[^\n]*\}) -->')
_TITLE_PREFIX = 'Latest versions canary'
_LABELS = ('p:1-highest', 'pydanty:triage-issue')
_BOT = 'github-actions[bot]'
_NEW_ISSUE_LIMIT = 3
_DEFERRED_LISTING_LIMIT = 20
_LISTED_TEST_LIMIT = 10
_MESSAGE_LIMIT = 200
_MOVED_LINE_LIMIT = 80
_BODY_LIMIT = 60_000
_REPORT_FILE_LIMIT = 50_000_000
_OPEN_PAGE_LIMIT = 10
_CLOSED_SEARCH_LIMIT = 30
# A maintainer decided these closes; the canary neither refiles them nor reports their passes.
_SETTLED_CLOSES = frozenset({'not_planned', 'duplicate'})
_LEG = re.compile(r'latest-versions-canary-(3\.\d{1,2})')
# pytest's traceback location lines: `path:line: ` closing a long entry's source, `path:line: in function`
# opening a short one, and `path:line: ExceptionName` closing the last entry of each chained traceback.
_LOCATION = re.compile(r'^(\S+\.py):\d+: (.*)$', re.MULTILINE)
_RAISED = re.compile(r'^E +(([A-Za-z_][\w.]{0,99}):.*)$', re.MULTILINE)
_EXCEPTION_NAME = re.compile(r'[A-Za-z_][\w.]{0,99}')
_SHORT_ENTRY = re.compile(r'in ([A-Za-z_]\w{0,99})')
_DEFINITION = re.compile(r'^\s*(?:async\s+)?def ([A-Za-z_]\w{0,99})\(', re.MULTILINE)
_TEST_MODULE = re.compile(r'test_\w*\.py')
# How pytest-xdist reports a lost worker and a collection mismatch between workers, each as the raw text
# of a JUnit error. The session can stop after either, leaving tests unrun. They are different causes.
_XDIST_STOPS = (
    (re.compile(r"worker '[^']*' crashed while running "), 'a pytest-xdist worker crashes'),
    (re.compile(r'Different tests were collected between '), 'pytest-xdist workers collect different tests'),
)
_PACKAGE = re.compile(r'/site-packages/([A-Za-z_][\w-]{0,99})')

StepOutcome = Literal['success', 'failure', 'cancelled', 'skipped']
Status = Literal['failing', 'passing']


class LegOutcome(BaseModel):
    """The `steps.<id>.outcome` of each stage one canary leg ran."""

    resolve: StepOutcome
    install: StepOutcome
    test: StepOutcome
    # `strategy.job-total`: how many legs the run had, so a missing one is noticed.
    legs: int
    # pytest's exit status, when the test step ran: 1 means tests failed and the session finished.
    pytest_exit: int | None = None


class Marker(BaseModel):
    """The hidden state on each issue body and comment this script writes."""

    run_id: int
    status: Status
    # On an issue body only: the signature the issue tracks.
    signature: str = ''


class Comment(BaseModel):
    """The fields this script reads from a GitHub issue comment."""

    body: str | None = None
    user: Account | None = None
    created_at: dt.datetime | None = None


class Issue(BaseModel):
    """The fields this script reads from a GitHub issue."""

    number: int
    body: str | None = None
    user: Account | None = None
    created_at: dt.datetime | None = None
    closed_at: dt.datetime | None = None
    state_reason: str | None = None
    pull_request: object = None


class _SearchResult(BaseModel):
    items: list[Issue]


_ISSUES = TypeAdapter(list[Issue])
_COMMENTS = TypeAdapter(list[Comment])


@dataclass
class Failure:
    """Everything one run observed under one failure signature."""

    pythons: set[str] = field(default_factory=set[str])
    # Test node ID -> first line of its failure. Untrusted text, for issue bodies only.
    tests: dict[str, str] = field(default_factory=dict[str, str])
    # First lines of failures tied to no test, such as the node a crashed worker was running. Untrusted too.
    details: list[str] = field(default_factory=list[str])


@dataclass
class Observation:
    """What one run showed."""

    # Failures that stopped a leg's suite first, then most failing tests first: the order a capped run files.
    failures: dict[str, Failure]
    # Every leg finished its test session, so a signature missing from `failures` passed.
    complete: bool


@dataclass
class Run:
    """The scheduled canary run being reported."""

    repo: str
    run_id: int
    sha: str
    server_url: str
    # When the tested commit was made: a close after it may hold a fix this run did not test.
    tested_at: dt.datetime

    @property
    def link(self) -> str:
        url = f'{self.server_url}/{self.repo}'
        return (
            f'scheduled [canary run]({url}/actions/runs/{self.run_id}) at [`{self.sha[:12]}`]({url}/commit/{self.sha})'
        )


@dataclass
class Tracked:
    """A canary issue this script filed."""

    client: attention.GitHubClient
    repo: str
    issue: Issue
    body: Marker

    @property
    def signature(self) -> str:
        return self.body.signature

    @cached_property
    def markers(self) -> list[tuple[Marker, dt.datetime | None]]:
        """Every marker on the issue with the time it was posted, newest first."""
        # Two pages: the newest page alone can hold a single comment.
        path = f'/repos/{self.repo}/issues/{self.issue.number}/comments'
        comments = _COMMENTS.validate_python(self.client.last_pages(path, count=2))
        markers = [
            (marker, comment.created_at)
            for comment in reversed(comments)
            if _by_bot(comment) and (marker := _marker(comment.body))
        ]
        return [*markers, (self.body, self.issue.created_at)]


def collect(reports: Path, root: Path) -> Observation:
    """Reduce every leg's uploaded report to failure signatures."""
    legs = [(match.group(1), leg) for leg in _children(reports) if (match := _LEG.fullmatch(leg.name))]
    failures: dict[str, Failure] = {}
    totals: set[int] = set()
    complete = bool(legs)
    for python, leg in legs:
        try:
            outcome = LegOutcome.model_validate_json(_read(leg / 'outcome.json'))
        except (OSError, ValidationError):
            outcome = None
        found = _leg_failures(leg, python, outcome, root)
        # A failed session counts only if it finished: an interrupted, internally failed or worker-starved
        # one writes JUnit for just the tests it reached, and the absent rest must not read as passes.
        # Failures not tied to a test, xdist's included, carry no test ID.
        tested = outcome is not None and (
            outcome.test == 'success' or (outcome.pytest_exit == 1 and all(test for _, test, _ in found))
        )
        complete = complete and tested
        totals.add(outcome.legs if outcome else 0)
        for signature, test, message in found:
            failure = failures.setdefault(signature, Failure())
            failure.pythons.add(python)
            if test:
                failure.tests.setdefault(test, message)
            elif message and message not in failure.details:
                failure.details.append(message)
    ranked = sorted(
        failures,
        key=lambda signature: (bool(failures[signature].tests), -len(failures[signature].tests), signature),
    )
    return Observation(
        failures={signature: failures[signature] for signature in ranked},
        complete=complete and totals == {len(legs)},
    )


def _children(directory: Path) -> list[Path]:
    return sorted(directory.iterdir()) if directory.is_dir() else []


def _leg_failures(leg: Path, python: str, outcome: LegOutcome | None, root: Path) -> list[tuple[str, str, str]]:
    if outcome is None:
        return [(f'canary job on Python {python} left no readable report', '', '')]
    for stage, result in (('resolve', outcome.resolve), ('install', outcome.install), ('test', outcome.test)):
        if result == 'success':
            continue
        if result == 'skipped':
            return [(f'canary job on Python {python} fails before its {stage} step', '', '')]
        if result == 'cancelled':
            return [(f'{stage} step times out or is cancelled on Python {python}', '', '')]
        if stage == 'resolve':
            # One universal resolution serves every interpreter, so its failure has no version.
            return [('dependency resolution fails', '', '')]
        if stage == 'install':
            return [(f'install fails on Python {python}', '', '')]
        tests = list(_test_failures(leg / 'junit.xml', root))
        return tests or [(f'pytest fails on Python {python} with no failing test reported', '', '')]
    return []


def _test_failures(junit: Path, root: Path) -> Iterator[tuple[str, str, str]]:
    try:
        suite = ElementTree.fromstring(_read(junit))
    except (OSError, ElementTree.ParseError):
        return
    for case in suite.iter('testcase'):
        for result in case:
            if result.tag not in ('failure', 'error'):
                continue
            text = result.text or ''
            if stop := next((signature for pattern, signature in _XDIST_STOPS if pattern.match(text)), None):
                yield stop, '', _first_line(text)
                continue
            locations = list(_LOCATION.finditer(text))
            frames = [
                (index, path)
                for index, location in enumerate(locations)
                if (path := _repo_path(location.group(1), root))
            ]
            test_file = _repo_path(case.get('file', ''), root)
            site = test_file or 'an unrecognized location'
            if frames:
                index, site = frames[-1]
                # A library function is a call path many tests share, so it tells causes in one file apart.
                # A test module's functions are the tests themselves, so naming one would split a cause per test.
                if not _TEST_MODULE.fullmatch(Path(site).name):
                    location = locations[index]
                    short = _SHORT_ENTRY.fullmatch(location.group(2))
                    # A long entry prints its function's source above its location line, `def` first.
                    start = locations[index - 1].end() if index else 0
                    definition = _DEFINITION.search(text, start, location.start())
                    function = short.group(1) if short else definition.group(1) if definition else ''
                    site += f'::{function}' if function else ''
            # Where a third-party package raised tells causes in one function apart. The standard library
            # does not: the same failure raises in Python code on one interpreter and in C on another.
            if locations and (package := _PACKAGE.search(locations[-1].group(1))):
                site += f' (raised in {package.group(1)})'
            names = [location.group(2) for location in locations if _EXCEPTION_NAME.fullmatch(location.group(2))]
            # The last `E` line is the final exception. JUnit's own message for a collection error is
            # just "collection failure".
            raised = list(_RAISED.finditer(text))
            exception = names[-1] if names else raised[-1].group(2) if raised else result.tag
            evidence = raised[-1].group(1) if raised else result.get('message') or text
            yield f'{exception} in {site}', _node_id(case, test_file), _first_line(evidence)


def _read(path: Path) -> bytes:
    with path.open('rb') as file:
        content = file.read(_REPORT_FILE_LIMIT + 1)
    if len(content) > _REPORT_FILE_LIMIT:
        raise OSError(f'{path.name} exceeds {_REPORT_FILE_LIMIT} bytes')
    return content


def _repo_path(raw: str, root: Path) -> str | None:
    """The checkout-relative path `raw` names, if it is a file in this checkout."""
    path = Path(raw)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            return None
    if '..' in path.parts or not path.parts or path.parts[0] == '.venv':
        return None
    resolved = (root / path).resolve()
    if not resolved.is_file() or not resolved.is_relative_to(root.resolve()):
        return None
    return path.as_posix()


def _node_id(case: ElementTree.Element, test_file: str | None) -> str:
    if test_file is None:
        return _first_line(f'{case.get("classname", "")}::{case.get("name", "")}')
    module = test_file.removesuffix('.py').replace('/', '.')
    # A collection error names the module itself; a test names its class path and function.
    classname = case.get('classname', '')
    parts = [*classname.removeprefix(module).lstrip('.').split('.'), case.get('name', '')] if classname else []
    return _first_line('::'.join([test_file, *filter(None, parts)]))


def _first_line(text: str) -> str:
    line = next((line.strip() for line in text.splitlines() if line.strip()), '')
    printable = ''.join(' ' if unicodedata.category(char).startswith('C') else char for char in line)
    return printable if len(printable) <= _MESSAGE_LIMIT else f'{printable[: _MESSAGE_LIMIT - 1]}…'


def _fenced(lines: list[str]) -> str:
    text = '\n'.join(lines)
    fence = '`' * max([3, *(len(run) + 1 for run in re.findall(r'`+', text))])
    return f'{fence}text\n{text}\n{fence}'


def read_moved(reports: Path) -> dict[str, list[str]]:
    """Each leg's report of the dependencies that moved past `uv.lock`."""
    moved: dict[str, list[str]] = {}
    for leg in _children(reports):
        if not (match := _LEG.fullmatch(leg.name)):
            continue
        try:
            lines = _read(leg / 'moved.txt').decode(errors='replace').splitlines()
        except OSError:
            continue
        kept = [_first_line(line) for line in lines[:_MOVED_LINE_LIMIT]]
        if len(lines) > _MOVED_LINE_LIMIT:
            kept.append(f'... and {len(lines) - _MOVED_LINE_LIMIT} more')
        moved[match.group(1)] = kept
    return moved


def _marker(text: str | None) -> Marker | None:
    match = _MARKER.search(text or '')
    if match is None:
        return None
    try:
        return Marker.model_validate_json(match.group(1))
    except ValidationError:
        return None


def _hidden(marker: Marker) -> str:
    return f'<!-- pydantic-ai-latest-versions-canary:v1 {marker.model_dump_json(exclude_defaults=True)} -->'


def _by_bot(item: Issue | Comment) -> bool:
    return item.user is not None and item.user.login == _BOT


def _counts(failure: Failure) -> str:
    pythons = ', '.join(sorted(failure.pythons))
    count = len(failure.tests)
    tests = f'; {count} failing test{"" if count == 1 else "s"}' if count else ''
    return f'Python {pythons}{tests}'


def _track(client: attention.GitHubClient, repo: str, issue: Issue) -> Tracked | None:
    """The canary issue `issue` is, or `None` if this script did not file it."""
    if issue.pull_request is not None or not _by_bot(issue) or (body := _marker(issue.body)) is None:
        return None
    return Tracked(client, repo, issue, body)


def _comment(client: attention.GitHubClient, run: Run, tracked: Tracked, status: Status, text: str) -> None:
    marker = Marker(run_id=run.run_id, status=status)
    path = f'/repos/{run.repo}/issues/{tracked.issue.number}/comments'
    client.post(path, {'body': f'{_hidden(marker)}\n{text}\n'})
    tracked.markers.insert(0, (marker, None))


def reconcile(
    client: attention.GitHubClient, run: Run, observation: Observation, moved: Mapping[str, list[str]]
) -> list[str]:
    """Comment on the canary issues and file the new signatures; return a log."""
    failures = observation.failures
    pages, listed_all = client.first_pages(
        f'/repos/{run.repo}/issues?state=open&creator={urllib.parse.quote(_BOT)}', count=_OPEN_PAGE_LIMIT
    )
    if not listed_all:
        raise RuntimeError('Too many open github-actions issues to rule out an existing canary issue')
    log: list[str] = []
    carried: set[str] = set()
    for issue in sorted(_ISSUES.validate_python(pages), key=lambda issue: issue.number):
        if (tracked := _track(client, run.repo, issue)) is not None:
            carried.add(tracked.signature)
            log.extend(_update_open(client, run, observation, tracked))

    new = [signature for signature in failures if signature not in carried]
    if not new and not observation.complete:
        return log
    closed = _closed_issues(client, run.repo)
    if observation.complete:
        log.extend(_record_recoveries(client, run, observation, closed))
    filed: list[str] = []
    for signature in new:
        if (last := closed.get(signature)) is None:
            filed.append(signature)
            continue
        marker, posted_at = last.markers[0]
        number, reason, closed_at = last.issue.number, last.issue.state_reason, last.issue.closed_at
        if reason in _SETTLED_CLOSES:
            log.append(f'`{signature}`: #{number} was closed as {reason}; not refiled')
        elif marker.run_id >= run.run_id:
            log.append(f'#{number}: already updated by this run or a later one')
        elif marker.status == 'passing':
            filed.append(signature)
        elif closed_at is not None and closed_at >= run.tested_at:
            log.append(f'`{signature}`: #{number} was closed after the tested commit; waiting for the next run')
        elif posted_at is not None and closed_at is not None and posted_at > closed_at:
            log.append(f'`{signature}`: still failing since #{number} was closed; already noted there')
        else:
            _comment(
                client,
                run,
                last,
                'failing',
                f'This issue was closed, but the {run.link} still fails ({_counts(failures[signature])}).'
                '\n\nNo canary run has passed it since its last failure, so no new issue is filed. '
                'Reopen this issue if the fix is incomplete.',
            )
            log.append(f'#{number}: noted that `{signature}` still fails after the close')

    for signature in filed[:_NEW_ISSUE_LIMIT]:
        body = _issue_body(run, signature, failures[signature], closed, moved)
        created = Issue.model_validate(
            client.post(
                f'/repos/{run.repo}/issues',
                {'title': f'{_TITLE_PREFIX}: {signature}', 'body': body, 'labels': list(_LABELS)},
            )
        )
        log.append(f'#{created.number}: filed `{signature}`')
    if deferred := filed[_NEW_ISSUE_LIMIT:]:
        listed = ', '.join(f'`{signature}`' for signature in deferred[:_DEFERRED_LISTING_LIMIT])
        more = f' and {len(deferred) - _DEFERRED_LISTING_LIMIT} more' if len(deferred) > _DEFERRED_LISTING_LIMIT else ''
        log.append(f'{len(deferred)} new signatures wait for a later run (limit {_NEW_ISSUE_LIMIT}): {listed}{more}')
    return log


def _update_open(client: attention.GitHubClient, run: Run, observation: Observation, tracked: Tracked) -> list[str]:
    number, signature = tracked.issue.number, tracked.signature
    failure = observation.failures.get(signature)
    if failure is None and not observation.complete:
        return []
    marker = tracked.markers[0][0]
    # Run IDs only grow, so an older one is a re-run: it must not overwrite what a newer run reported.
    if marker.run_id >= run.run_id:
        return [f'#{number}: already updated by this run or a later one']
    if failure is not None:
        verb = 'Fails again after passing' if marker.status == 'passing' else 'Still fails'
        _comment(client, run, tracked, 'failing', f'{verb} in the {run.link} ({_counts(failure)}).')
        return [f'#{number}: noted that `{signature}` {verb.lower()}']
    if marker.status == 'passing':
        return []
    _comment(
        client, run, tracked, 'passing', f'Passes in the {run.link}. Close this issue if the fix is meant to last.'
    )
    return [f'#{number}: noted that `{signature}` passes']


def _closed_issues(client: attention.GitHubClient, repo: str) -> dict[str, Tracked]:
    """The most recently closed canary issue of each signature among the latest closed ones."""
    query = f'repo:{repo} is:issue is:closed author:app/github-actions in:title "{_TITLE_PREFIX}"'
    params = urllib.parse.urlencode({'q': query, 'sort': 'created', 'order': 'desc', 'per_page': _CLOSED_SEARCH_LIMIT})
    result = _SearchResult.model_validate(client.get(f'/search/issues?{params}'))
    closed: dict[str, Tracked] = {}
    closes = [(issue.closed_at, issue) for issue in result.items if issue.closed_at is not None]
    for _, issue in sorted(closes, key=lambda close: close[0]):
        if (tracked := _track(client, repo, issue)) is not None:
            closed[tracked.signature] = tracked
    return closed


def _record_recoveries(
    client: attention.GitHubClient, run: Run, observation: Observation, closed: Mapping[str, Tracked]
) -> list[str]:
    log: list[str] = []
    for signature, tracked in closed.items():
        if signature in observation.failures or tracked.issue.state_reason in _SETTLED_CLOSES:
            continue
        marker = tracked.markers[0][0]
        if marker.run_id >= run.run_id:
            log.append(f'#{tracked.issue.number}: already updated by this run or a later one')
            continue
        if marker.status == 'passing':
            continue
        _comment(client, run, tracked, 'passing', f'Passes in the {run.link}. A later failure files a new issue.')
        log.append(f'#{tracked.issue.number}: recorded that `{tracked.signature}` passes')
    return log


def _issue_body(
    run: Run, signature: str, failure: Failure, closed: Mapping[str, Tracked], moved: Mapping[str, list[str]]
) -> str:
    sections = [
        _hidden(Marker(run_id=run.run_id, status='failing', signature=signature)),
        f'The {run.link} failed on `main`: `{signature}` ({_counts(failure)}).',
        'The canary installs the newest versions our declared dependency ranges allow, so a fresh install of our '
        'packages hits this today. Migrate and raise the floor, or cap the affected requirement until the '
        'migration lands. Do not skip or suppress the failing tests.',
    ]
    if (last := closed.get(signature)) is not None:
        sections.append(f'This recurs after #{last.issue.number} was closed and a later canary run passed it.')
    if failure.tests:
        listed = [f'{test}\n    {message}' for test, message in list(failure.tests.items())[:_LISTED_TEST_LIMIT]]
        if len(failure.tests) > _LISTED_TEST_LIMIT:
            listed.append(f'... and {len(failure.tests) - _LISTED_TEST_LIMIT} more in the run')
        sections.append(f'<details><summary>Failing tests</summary>\n\n{_fenced(listed)}\n</details>')
    if failure.details:
        sections.append(_fenced(failure.details[:_LISTED_TEST_LIMIT]))
    for python, lines in moved.items():
        sections.append(
            f'<details><summary>Dependencies past <code>uv.lock</code> on Python {python}</summary>\n\n'
            f'{_fenced(lines or ["none"])}\n</details>'
        )
    body = ''
    for section in sections:
        if len(body) + len(section) > _BODY_LIMIT:
            return f'{body}The rest of the evidence is in the canary run.\n'
        body += f'{section}\n\n'
    return body


def main() -> int:
    """Report the scheduled canary run named by the GitHub Actions environment."""
    parser = argparse.ArgumentParser(description='File scheduled latest-versions canary failures as issues.')
    parser.add_argument('--reports', type=Path, required=True, help='directory of downloaded leg artifacts')
    parser.add_argument('--result', required=True, help="the canary job's `needs.<job>.result`")
    parser.add_argument('--tested-at', type=dt.datetime.fromisoformat, required=True, help='tested commit time')
    args = parser.parse_args()
    result: str = args.result
    if result not in ('success', 'failure'):
        print(f'canary result {result!r}: nothing to report')
        return 0
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print('GITHUB_TOKEN is required', file=sys.stderr)
        return 1
    run = Run(
        repo=os.environ['GITHUB_REPOSITORY'],
        run_id=int(os.environ['GITHUB_RUN_ID']),
        sha=os.environ['GITHUB_SHA'],
        server_url=os.environ.get('GITHUB_SERVER_URL', 'https://github.com'),
        tested_at=args.tested_at,
    )
    observation = collect(args.reports, Path(os.environ.get('GITHUB_WORKSPACE', '.')))
    if result == 'success':
        # Only a green run proves every leg passed; a leftover report cannot contradict it.
        observation = Observation(failures={}, complete=True)
    elif not observation.failures:
        # Every leg that reported claims success, so a missing leg failed: nothing here passed.
        observation = Observation(failures={'canary run fails with no readable report': Failure()}, complete=False)
    log = reconcile(attention.GitHubClient(token), run, observation, read_moved(args.reports))
    for line in log or ['nothing to report']:
        print(line)
    if summary := os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(summary).open('a', encoding='utf-8') as file:
            file.writelines(f'- {line}\n' for line in log)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
