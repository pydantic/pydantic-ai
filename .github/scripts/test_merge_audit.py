"""Audit `main` after the fact, replaying recorded GitHub responses."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).parent))

import _commands
import merge_audit
from _command_fixtures import Replay

FIXTURES = Path(__file__).parent / 'fixtures' / 'commands'


class _Ruleset(TypedDict):
    rules: list[dict[str, object]]


class _RequiredCheck(TypedDict):
    context: str


class _RequiredChecks(TypedDict):
    required_status_checks: list[_RequiredCheck]


@pytest.fixture(autouse=True)
def _repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixtures were recorded against this repository, whatever repository runs the tests."""
    monkeypatch.setattr(merge_audit, 'REPO', 'pydantic/pydantic-ai')


def test_recorded_windows_report_each_finding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each recorded window, in recorded order, prints what the audit finds in it."""
    replay = Replay(FIXTURES / 'merge_audit_windows.json')
    monkeypatch.setattr(_commands, 'run', replay.run)
    reports: list[list[str]] = []
    for since, until in [
        ('2026-10-21T20:00:00Z', '2026-10-21T21:00:00Z'),
        # The head of https://github.com/pydantic/pydantic-ai/pull/12236 failed `check`, but the merge queue's run on
        # the merge commit passed before the merge.
        ('2026-11-02T17:00:00Z', '2026-11-02T18:00:00Z'),
        ('2026-11-11T19:00:00Z', '2026-11-11T19:30:00Z'),
        # The one dismissal in this window links its issue.
        ('2026-11-05T16:00:00Z', '2026-11-05T16:15:00Z'),
    ]:
        merge_audit.main(['--since', since, '--until', until])
        reports.append(capsys.readouterr().out.rstrip('\n').splitlines()[3:])

    assert reports == [
        [
            '### Merged before `check` passed',
            '',
            '- https://github.com/pydantic/pydantic-ai/pull/10930 Fix flaky BoxLite conformance test by signalling the '
            "fake box's script before its command",
            # `check` passed on its head 9 minutes after the merge.
            "- https://github.com/pydantic/pydantic-ai/pull/10928 Match the unknown-tool retry by `outcome='retried'` "
            'in the `unless_tool_kind` tests',
            '- https://github.com/pydantic/pydantic-ai/pull/10925 Detect output-tool retries by '
            '`ToolReturnPart.outcome` in the native-or-local selection test',
        ],
        [
            '### Merged with a review context not green',
            '',
            'The ruleset does not require these contexts.',
            '',
            '- https://github.com/pydantic/pydantic-ai/pull/12212: Gate: clean douwebot approval: failure',
        ],
        [
            '### CodeQL alerts dismissed without a linked issue',
            '',
            *(
                f'- https://github.com/pydantic/pydantic-ai/security/code-scanning/{number} ({reason})'
                for number, reason in [
                    (50, 'false positive'),
                    (49, 'false positive'),
                    (45, 'used in tests'),
                    (40, 'false positive'),
                    (39, 'false positive'),
                ]
            ),
        ],
        [],
    ]
    assert replay.unconsumed == []


def test_a_ruleset_that_drifted_from_the_committed_one_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live ruleset recorded in a quiet window differs from a committed one that no longer requires `check`."""
    committed = TypeAdapter(dict[str, object]).validate_json(merge_audit.RULESET.read_bytes())
    ruleset = tmp_path / 'main.json'
    ruleset.write_text(TypeAdapter(dict[str, object]).dump_json({**committed, 'rules': []}).decode(), encoding='utf-8')
    monkeypatch.setattr(merge_audit, 'RULESET', ruleset)
    monkeypatch.setattr(_commands, 'run', Replay(FIXTURES / 'merge_audit_windows.json').run)

    report = merge_audit.report(datetime(2026, 11, 5, 16, tzinfo=UTC), datetime(2026, 11, 5, 16, 15, tzinfo=UTC))

    assert report is not None
    assert report.splitlines()[3:] == [
        '### `main` ruleset',
        '',
        'The live ruleset differs from `.github/rulesets/main.json` in `rules`.',
    ]


def test_a_window_without_a_utc_offset_is_refused() -> None:
    """A naive time cannot be compared with GitHub's, and would be read as UTC in the search query."""
    with pytest.raises(SystemExit):
        merge_audit.main(['--since', '2026-10-21T20:00:00'])


def test_ruleset_drift_names_each_changed_field() -> None:
    """A changed enforcement or rule list is drift, and the committed ruleset against itself is none."""
    committed = TypeAdapter(dict[str, object]).validate_json(merge_audit.RULESET.read_bytes())
    live: dict[str, object] = {**committed, 'enforcement': 'evaluate', 'rules': []}

    assert merge_audit.ruleset_drift(committed, committed) == []
    assert merge_audit.ruleset_drift(committed, live) == ['enforcement', 'rules']


def test_the_committed_ruleset_requires_only_the_check_the_audit_reads() -> None:
    """The audit reads `check` as the one required context, so it must be the one the ruleset requires."""
    rules = TypeAdapter(_Ruleset).validate_json(merge_audit.RULESET.read_bytes())['rules']
    [required] = [rule for rule in rules if rule['type'] == 'required_status_checks']

    assert TypeAdapter(_RequiredChecks).validate_python(required['parameters']) == {
        'required_status_checks': [{'context': merge_audit.REQUIRED_CHECK}]
    }


def test_publish_opens_the_rolling_issue_when_none_is_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no open audit issue, the report opens one; the write's output is not read."""
    replay = Replay(FIXTURES / 'merge_audit_no_open_issue.json')
    writes: list[list[str]] = []

    def run(args: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
        if list(args[1:3]) == ['issue', 'list']:
            return replay.run(args, check=check)
        writes.append(list(args))
        return subprocess.CompletedProcess(list(args), 0, '', '')

    monkeypatch.setattr(_commands, 'run', run)
    body = f'{merge_audit.MARKER}\n## Merge audit'

    merge_audit.publish(body)

    assert writes == [
        ['gh', 'issue', 'create', '--repo', 'pydantic/pydantic-ai', '--title', merge_audit.ISSUE_TITLE, '--body', body]
    ]
    assert replay.unconsumed == []
