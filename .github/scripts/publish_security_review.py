"""Gate and publish the `Security Review` commit status from trusted default-branch code."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

from pr_gates import Commit, GateError, GhCli, Label
from pydantic import BaseModel, Field, TypeAdapter

CONTEXT = 'Security Review'
RUN_LABEL = 'security-review:run'
ACCEPTED_LABEL = 'security-review:accepted'
MARKER = re.compile(r'\n\n<!-- security-review run=([1-9][0-9]*) head=([a-f0-9]{40}) -->\Z')
# `.permission` maps maintain and custom roles onto these base levels.
TRUSTED_PERMISSIONS = frozenset({'admin', 'write'})

Severity = Literal['critical', 'high', 'medium']
BLOCKING: frozenset[Severity] = frozenset({'critical', 'high'})
CommitStatus = Literal['pending', 'success', 'failure']


class SecurityFinding(BaseModel):
    """One inline finding on a file its check reviews."""

    path: str
    line: int = Field(gt=0)
    side: Literal['LEFT', 'RIGHT']
    severity: Severity
    body: str = Field(min_length=1)


class CheckResult(BaseModel):
    """What a review job hands the publisher; it never carries the private report."""

    check: str
    files: list[str]
    outcome: Literal['COMPLETE', 'INCOMPLETE']
    findings: list[SecurityFinding]
    coverage: str
    steps: int
    tokens: int
    """Telemetry only: input, output, reasoning and cache tokens summed over every step."""
    seconds: int


class Actor(BaseModel):
    """A GitHub account."""

    login: str
    type: str


class LivePullRequest(BaseModel):
    """The live state an event's payload can no longer vouch for."""

    state: Literal['open', 'closed']
    head: Commit
    user: Actor
    labels: list[Label]


class _EventLabel(BaseModel):
    name: str


class _IssueEvent(BaseModel):
    event: str
    actor: Actor | None
    label: _EventLabel | None = None


class SecurityGitHub(Protocol):
    """The GitHub reads and writes the gate and the publisher need."""

    def pull_request(self) -> LivePullRequest: ...
    def permission(self, login: str) -> str: ...
    def labeler(self, label: str) -> Actor | None: ...
    def remove_label(self, label: str) -> None: ...
    def status(self, head: str, state: CommitStatus, description: str, url: str) -> None: ...
    def post_review(self, payload: str) -> str | None: ...


def gate(github: SecurityGitHub, env: Mapping[str, str]) -> bool:
    """Return whether this event reviews the head; every other outcome writes its own status."""
    url = run_url(env)
    if group := env.get('MERGE_GROUP_SHA'):
        github.status(group, 'success', 'Admission-only: every pull request head is reviewed before the queue.', url)
        return False
    head = env['HEAD_SHA']
    pull = github.pull_request()
    if pull.state != 'open' or pull.head.sha != head:
        print('PR closed or moved; the event for its new head decides.')
        return False
    labels = {label.name for label in pull.labels}
    action, label, sender = env['EVENT_ACTION'], env.get('EVENT_LABEL', ''), env['EVENT_SENDER']
    if action == 'labeled' and label == ACCEPTED_LABEL:
        if ACCEPTED_LABEL not in labels:
            raise GateError(f'`{ACCEPTED_LABEL}` is no longer present.')
        if env['EVENT_SENDER_TYPE'] != 'User' or github.permission(sender) not in TRUSTED_PERMISSIONS:
            raise GateError(f'Only a user with write access can apply `{ACCEPTED_LABEL}`.')
        github.status(head, 'success', f'Findings accepted by @{sender} for this head.', url)
        return False
    # An acceptance covers only the head it was given for.
    if action == 'synchronize' and ACCEPTED_LABEL in labels:
        github.remove_label(ACCEPTED_LABEL)
    author_trusted = pull.user.type == 'User' and github.permission(pull.user.login) in TRUSTED_PERMISSIONS
    if action == 'labeled' and label == RUN_LABEL and author_trusted:
        print(f'The author has write access, so `{RUN_LABEL}` changes nothing.')
        return False
    if not author_trusted and not _run_label_trusted(github, labels):
        github.status(head, 'failure', f'Awaiting maintainer: a user with write access adds `{RUN_LABEL}`.', url)
        return False
    github.status(head, 'pending', 'Reviewing this head.', url)
    return True


def _run_label_trusted(github: SecurityGitHub, labels: set[str]) -> bool:
    """The run label counts only when its latest application came from a user with write access."""
    if RUN_LABEL not in labels:
        return False
    labeler = github.labeler(RUN_LABEL)
    return labeler is not None and labeler.type == 'User' and github.permission(labeler.login) in TRUSTED_PERMISSIONS


def load_results(directory: Path, checks: list[str]) -> dict[str, CheckResult]:
    """Read each selected check's result, rejecting a finding outside the files that check reviewed."""
    results: dict[str, CheckResult] = {}
    for path in sorted(directory.glob('*.json')) if directory.is_dir() else []:
        result = CheckResult.model_validate_json(path.read_text(encoding='utf-8'))
        if result.check not in checks or path.stem != result.check:
            raise GateError(f'Unexpected result file {path.name}')
        if outside := sorted({finding.path for finding in result.findings} - set(result.files)):
            raise GateError(f'{result.check} reported findings outside its files: {outside}')
        results[result.check] = result
    return results


def verdict(checks: list[str], results: Mapping[str, CheckResult]) -> tuple[CommitStatus, str]:
    """Fail on a blocking finding, an incomplete review, or a check that did not report."""
    blocking = [
        check for check in checks if check in results and any(f.severity in BLOCKING for f in results[check].findings)
    ]
    incomplete = [check for check in checks if check in results and results[check].outcome == 'INCOMPLETE']
    missing = [check for check in checks if check not in results]
    problems = [
        f'{label}: {", ".join(names)}'
        for label, names in (('Blocking findings', blocking), ('Incomplete', incomplete), ('No result', missing))
        if names
    ]
    if problems:
        return 'failure', '; '.join(problems)[:140]
    if not checks:
        return 'success', 'No file this review covers changed.'
    advisory = sum(len(results[check].findings) for check in checks)
    noted = f'; {advisory} non-blocking findings' if advisory else ''
    return 'success', f'Reviewed by {", ".join(checks)}{noted}.'[:140]


def review_payload(results: list[CheckResult], head: str, run_id: int, *, inline: bool) -> str:
    """One `COMMENT` review for every check's findings, bound to this run and head."""
    body = '## Security Review\n\n' + '\n'.join(
        f'- `{result.check}` ({result.outcome.lower()}): {result.coverage}' for result in results
    )
    body += '\n\n<details><summary>Usage</summary>\n\n' + usage_table(results) + '\n\n</details>'
    comments: list[dict[str, object]] = []
    for result in results:
        for finding in result.findings:
            text = f'**{finding.severity}** · `{result.check}` security check\n\n{finding.body}'
            if inline:
                comments.append({'path': finding.path, 'line': finding.line, 'side': finding.side, 'body': text})
            else:
                body += f'\n\n### `{finding.path}:{finding.line}` (`{finding.side}`)\n\n{text}'
    body += f'\n\n<!-- security-review run={run_id} head={head} -->'
    return json.dumps({'commit_id': head, 'event': 'COMMENT', 'body': body, 'comments': comments})


def publish(github: SecurityGitHub, env: Mapping[str, str]) -> None:
    """Post the findings and the final status for the head this run reviewed."""
    head = env['HEAD_SHA']
    url = run_url(env)
    checks: list[str] = _CHECKS.validate_json(env['SECURITY_CHECKS'] or '[]')
    pull = github.pull_request()
    if pull.state != 'open' or pull.head.sha != head:
        print('PR closed or moved while it was reviewed; the event for its new head decides.')
        return
    if env['DETECT_RESULT'] != 'success':
        github.status(head, 'failure', 'Could not select the checks for this head; rerun the workflow.', url)
        return
    results = load_results(Path(env['RESULTS_DIR']), checks)
    state, description = verdict(checks, results)
    ordered = [results[check] for check in checks if check in results]
    if any(result.findings for result in ordered):
        run_id = int(env['GITHUB_RUN_ID'])
        review_url = github.post_review(review_payload(ordered, head, run_id, inline=True))
        if review_url is None:
            print('GitHub rejected the inline review; retrying with findings in the review body.')
            review_url = github.post_review(review_payload(ordered, head, run_id, inline=False))
        if review_url is None:
            raise GateError('GitHub rejected the review with its findings in the body.')
        url = review_url
    github.status(head, state, description, url)
    summary = f'## {CONTEXT}: {state}\n\n{description}\n\n{usage_table(ordered)}'
    print(summary)
    if summary_path := env.get('GITHUB_STEP_SUMMARY'):
        with open(summary_path, 'a', encoding='utf-8') as file:
            file.write(summary + '\n')


def usage_table(results: list[CheckResult]) -> str:
    """Each check's steps, tokens and duration, for tuning its budget."""
    rows = [
        f'| `{r.check}` | {r.outcome} | {len(r.findings)} | {r.steps} | {r.tokens:,} | {r.seconds} |' for r in results
    ]
    return '\n'.join(['| check | outcome | findings | steps | tokens | seconds |', '|---|---|---|---|---|---|', *rows])


def run_url(env: Mapping[str, str]) -> str:
    """The workflow run that wrote a status."""
    return f'{env["GITHUB_SERVER_URL"]}/{env["GITHUB_REPOSITORY"]}/actions/runs/{env["GITHUB_RUN_ID"]}'


_CHECKS = TypeAdapter(list[str])
_EVENT_PAGES = TypeAdapter(list[list[_IssueEvent]])
_PERMISSION = TypeAdapter(dict[str, object])


class _PostedReview(BaseModel):
    html_url: str


class SecurityGhCli(GhCli):
    """Pull request state, permissions, labels, reviews and statuses through `gh api`."""

    def pull_request(self) -> LivePullRequest:
        return LivePullRequest.model_validate_json(self._api(self._pull))

    def permission(self, login: str) -> str:
        """Fail closed: an account the endpoint cannot resolve, such as an app, has no write access."""
        try:
            permission = _PERMISSION.validate_json(self._api(f'{self._repo}/collaborators/{login}/permission'))
        except GateError as error:
            print(f'No repository permission for @{login}: {error}')
            return 'none'
        value = permission.get('permission')
        return value if isinstance(value, str) else 'none'

    def labeler(self, label: str) -> Actor | None:
        pages = _EVENT_PAGES.validate_json(self._api('--paginate', '--slurp', f'{self._issue}/events?per_page=100'))
        applied = [
            event.actor
            for page in pages
            for event in page
            if event.event == 'labeled' and event.label is not None and event.label.name == label
        ]
        return applied[-1] if applied else None

    def remove_label(self, label: str) -> None:
        self._api('--method', 'DELETE', f'{self._issue}/labels/{label}')

    def status(self, head: str, state: CommitStatus, description: str, url: str) -> None:
        payload = {'context': CONTEXT, 'state': state, 'description': description, 'target_url': url}
        self._api('--method', 'POST', '--input', '-', f'{self._repo}/statuses/{head}', input=json.dumps(payload))

    def post_review(self, payload: str) -> str | None:
        """Return the review's URL, or None when GitHub rejects an inline position (HTTP 422)."""
        try:
            posted = self._api('--method', 'POST', '--input', '-', f'{self._pull}/reviews', input=payload)
        except GateError as error:
            if '(HTTP 422)' in str(error):
                return None
            raise
        return _PostedReview.model_validate_json(posted).html_url


def main() -> None:
    """`gate` decides whether this event reviews its head; `publish` posts the result."""
    env = os.environ
    github = SecurityGhCli(env['GITHUB_REPOSITORY'], int(env.get('PR_NUMBER') or '0'))
    if sys.argv[1] == 'gate':
        review = gate(github, env)
        with open(env['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
            output.write(f'review={str(review).lower()}\n')
    else:
        publish(github, env)


if __name__ == '__main__':
    main()
