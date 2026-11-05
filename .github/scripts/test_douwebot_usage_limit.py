"""The douwebot OpenCode fallback runs only for the subscription-limit rejection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

from douwebot_usage_limit import is_usage_limit, is_usage_limit_file, main

WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'bots.yml'


def _limit_event(window: str = 'five_hour', status: str = 'rejected') -> dict[str, object]:
    # Shape documented by the Agent SDK version pinned by claude-code-action.
    return {'type': 'rate_limit_event', 'rate_limit_info': {'status': status, 'rateLimitType': window}}


@pytest.mark.parametrize('window', ['five_hour', 'seven_day', 'seven_day_opus', 'seven_day_sonnet'])
def test_rejected_subscription_window(window: str) -> None:
    assert is_usage_limit([_limit_event(window)])


@pytest.mark.parametrize('window', ['overage', 'seven_day_overage_included', 'unknown'])
def test_other_limits_do_not_match(window: str) -> None:
    assert not is_usage_limit([_limit_event(window)])


@pytest.mark.parametrize('status', ['allowed', 'allowed_warning', 'unknown'])
def test_non_rejected_limit_does_not_match(status: str) -> None:
    assert not is_usage_limit([_limit_event(status=status)])


def test_missing_and_malformed_events_fail_closed() -> None:
    assert not is_usage_limit([{'type': 'result', 'is_error': True, 'total_cost_usd': 0}])
    assert not is_usage_limit([{'type': 'rate_limit_event', 'rate_limit_info': {'status': 'rejected'}}])
    assert not is_usage_limit([{'type': 'rate_limit_event', 'rate_limit_info': None}])
    assert not is_usage_limit(
        [{'type': 'rate_limit_event', 'rate_limit_info': {'status': 'rejected', 'rateLimitType': []}}]
    )
    assert not is_usage_limit(None)
    assert not is_usage_limit(_limit_event())
    assert not is_usage_limit([_limit_event(), _limit_event(status='allowed')])


def test_execution_file(tmp_path: Path) -> None:
    path = tmp_path / 'claude-execution-output.json'
    path.write_text(json.dumps([_limit_event()]), encoding='utf-8')
    assert is_usage_limit_file(str(path))
    assert not is_usage_limit_file('')
    assert not is_usage_limit_file(str(tmp_path / 'missing.json'))
    path.write_text('{', encoding='utf-8')
    assert not is_usage_limit_file(str(path))


def test_main_writes_fallback_only_for_a_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    execution = tmp_path / 'claude-execution-output.json'
    execution.write_text(json.dumps([_limit_event()]), encoding='utf-8')
    output = tmp_path / 'github-output.txt'
    monkeypatch.setenv('EXECUTION_FILE', str(execution))
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))

    main()
    assert output.read_text(encoding='utf-8') == 'fallback=true\n'

    execution.write_text(json.dumps([_limit_event(status='allowed')]), encoding='utf-8')
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1
    assert output.read_text(encoding='utf-8') == 'fallback=true\n'


def test_main_fails_closed_without_github_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    execution = tmp_path / 'claude-execution-output.json'
    execution.write_text(json.dumps([_limit_event()]), encoding='utf-8')
    monkeypatch.setenv('EXECUTION_FILE', str(execution))
    monkeypatch.delenv('GITHUB_OUTPUT', raising=False)
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1


_OBJECT = TypeAdapter(dict[str, object])
_OBJECTS = TypeAdapter(dict[str, dict[str, object]])
_STEPS = TypeAdapter(list[dict[str, object]])


def _workflow() -> dict[str, object]:
    # BaseLoader keeps `on` as a string. SafeLoader reads it as boolean true.
    loaded: object = yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    return _OBJECT.validate_python(loaded)


def _job(name: str) -> dict[str, object]:
    jobs = _OBJECTS.validate_python(_workflow()['jobs'])
    return jobs[name]


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    return _STEPS.validate_python(job['steps'])


def _step(job: dict[str, object], step_id: str) -> dict[str, object]:
    for step in _steps(job):
        if step.get('id') == step_id:
            return step
    raise AssertionError(step_id)


def test_workflow_only_publishes_the_successful_review() -> None:
    """A fallback failure cannot publish the primary's missing result as an approval."""
    review = _job('review')
    fallback = _job('douwebot-opencode')
    publish = _job('publish-review')
    assert _step(review, 'usage-limit')['if'] == "steps.review.outcome == 'failure'"
    assert fallback['needs'] == 'review'
    assert fallback['if'] == "needs.review.result == 'success' && needs.review.outputs.fallback == 'true'"
    assert 'douwebot-opencode' in publish['needs']
    publish_step = _step(publish, 'publish')
    assert "needs.douwebot-opencode.result == 'success'" in publish_step['if']
    assert 'needs.douwebot-opencode.outputs.result' in _OBJECT.validate_python(publish_step['env'])['REVIEW_RESULT']
