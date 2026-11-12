"""Validate OpenCode review completion and output before publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent))

from douwebot_opencode_result import review_result

_RESULT = '{"verdict":"APPROVE","comments":[],"discussion":""}'
_STOP: dict[str, object] = {'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'stop'}}


def _write_events(tmp_path: Path, events: list[dict[str, object]]) -> Path:
    """Write an OpenCode event stream the way `opencode run --format json` does: one event per LF-ended line."""
    stream = tmp_path / 'review.ndjson'
    stream.write_text(''.join(json.dumps(event, ensure_ascii=False) + '\n' for event in events), encoding='utf-8')
    return stream


def _final_text(text: str) -> dict[str, object]:
    return {'type': 'text', 'part': {'messageID': 'msg_final', 'text': text}}


def _stream(tmp_path: Path, text: str) -> Path:
    """A normally completed stream whose final assistant text is `text`."""
    return _write_events(tmp_path, [_final_text(text), _STOP])


def test_completed_review_result(tmp_path: Path) -> None:
    """Accept the final JSON only when the stream stopped normally."""
    stream = _write_events(tmp_path, [_final_text('intermediate analysis'), _final_text(_RESULT), _STOP])
    assert review_result(stream).verdict == 'APPROVE'


def test_line_separator_in_model_text_stays_in_its_event(tmp_path: Path) -> None:
    """OpenCode's `JSON.stringify` leaves U+2028 raw, and only LF ends an event."""
    discussion = 'First point.\u2028Second point.'
    result = json.dumps({'verdict': 'APPROVE', 'comments': [], 'discussion': discussion}, ensure_ascii=False)
    assert review_result(_stream(tmp_path, result)).discussion == discussion


@pytest.mark.parametrize(
    'text',
    [
        f'```json\n{_RESULT}\n```',
        f'```\n{_RESULT}\n```',
        f'\n  ```json\n{json.dumps(json.loads(_RESULT), indent=2)}\n```\n',
        '```json\n{"verdict":"APPROVE","comments":[],"discussion":"Use:\\n```py\\nx = 1\\n```"}\n```',
        f'Here is the review:\n```json\n{_RESULT}\n```',
    ],
)
def test_code_fence_that_ends_the_answer_is_unwrapped(tmp_path: Path, text: str) -> None:
    """Accept a final JSON object in the Markdown code fence that ends the answer."""
    assert review_result(_stream(tmp_path, text)).verdict == 'APPROVE'


@pytest.mark.parametrize(
    'text',
    [
        f'```json\n{_RESULT}\n```\nLet me know if you need more.',
        f'```json\n```json\n{_RESULT}\n```\n```',
        f'```json\n{_RESULT}',
        f'```json {_RESULT} ```',
    ],
)
def test_fence_with_other_text_still_fails(tmp_path: Path, text: str) -> None:
    """Text after the fence, nested or unterminated fences stay invalid."""
    with pytest.raises(ValidationError):
        review_result(_stream(tmp_path, text))


# The answers glm-5.3-flash returned in runs 43266067725 and 43265435601 began with this prose and ended with this
# JSON, in `json.dumps`'s default separators; the job log kept only both ends.
_PROSE = "I've completed a thorough review of the changed files."
_FINDINGS = json.dumps(
    {
        'verdict': 'REQUEST_CHANGES',
        'comments': [{'path': 'a.py', 'line': 1, 'side': 'RIGHT', 'body': 'Handle the empty case.'}],
        'discussion': '',
    }
)


@pytest.mark.parametrize('separator', ['\n\n', ' '])
def test_prose_before_bare_json_is_dropped(tmp_path: Path, separator: str) -> None:
    """Accept a bare JSON object that ends the answer after prose."""
    result = review_result(_stream(tmp_path, _PROSE + separator + _FINDINGS))
    assert result.verdict == 'REQUEST_CHANGES'


def test_fence_heavy_answer_is_rejected_in_linear_time(tmp_path: Path) -> None:
    """A model-controlled answer of a million fence characters cannot stall validation."""
    with pytest.raises(ValidationError):
        review_result(_stream(tmp_path, '```\n' * 250_000 + 'x'))


@pytest.mark.parametrize(
    'events',
    [
        [],
        [_final_text(_RESULT)],
        [{'type': 'error'}, _STOP],
        [{'type': 'text', 'part': {'messageID': 'msg_old', 'text': _RESULT}}, _STOP],
        [{'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'error'}}],
        [_final_text('not json'), _STOP],
        [_final_text('{"verdict":"APPROVE"}'), _STOP],
    ],
)
def test_incomplete_or_invalid_review_fails(tmp_path: Path, events: list[dict[str, object]]) -> None:
    """Never pass a partial stream, error, or invalid review to the publisher."""
    with pytest.raises((ValueError, ValidationError)):
        review_result(_write_events(tmp_path, events))
