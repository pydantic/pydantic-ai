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


def test_a_recorded_window_lists_each_merge_before_check_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a `check` run that passed by the merge counts, on the head or on the commit the merge queue tested."""
    replay = Replay(FIXTURES / 'merge_audit_windows.json')
    monkeypatch.setattr(_commands, 'run', replay.run)

    red = merge_audit.report(datetime(2026, 10, 21, 20, tzinfo=UTC), datetime(2026, 10, 21, 21, tzinfo=UTC))
    # The head of https://github.com/pydantic/pydantic-ai/pull/12236 failed `check`, but the merge queue's run on the
    # merge commit passed before the merge.
    queued = merge_audit.report(datetime(2026, 11, 2, 17, 30, tzinfo=UTC), datetime(2026, 11, 2, 18, tzinfo=UTC))

    assert red is not None
    assert red.splitlines()[3:] == [
        '### Merged before `check` passed',
        '',
        '- https://github.com/pydantic/pydantic-ai/pull/10930 Fix flaky BoxLite conformance test by signalling the '
        "fake box's script before its command",
        # `check` passed on its head 9 minutes after the merge.
        "- https://github.com/pydantic/pydantic-ai/pull/10928 Match the unknown-tool retry by `outcome='retried'` in "
        'the `unless_tool_kind` tests',
        '- https://github.com/pydantic/pydantic-ai/pull/10925 Detect output-tool retries by `ToolReturnPart.outcome` '
        'in the native-or-local selection test',
    ]
    assert queued is None
    assert replay.unconsumed == []


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
