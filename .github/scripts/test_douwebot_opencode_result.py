"""Validate OpenCode review completion and output before publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent))

from douwebot_opencode_result import review_result


def test_completed_review_result(tmp_path: Path) -> None:
    """Accept the final JSON only when the stream stopped normally."""
    stream = tmp_path / 'review.ndjson'
    events = [
        {'type': 'text', 'part': {'messageID': 'msg_final', 'text': 'intermediate analysis'}},
        {
            'type': 'text',
            'part': {'messageID': 'msg_final', 'text': '{"verdict":"APPROVE","comments":[],"discussion":""}'},
        },
        {'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'stop'}},
    ]
    stream.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')
    assert review_result(stream).verdict == 'APPROVE'


_RESULT = '{"verdict":"APPROVE","comments":[],"discussion":""}'


def _stream(tmp_path: Path, text: str) -> Path:
    stream = tmp_path / 'review.ndjson'
    events = [
        {'type': 'text', 'part': {'messageID': 'msg_final', 'text': text}},
        {'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'stop'}},
    ]
    stream.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')
    return stream


@pytest.mark.parametrize(
    'text',
    [
        f'```json\n{_RESULT}\n```',
        f'```\n{_RESULT}\n```',
        f'\n  ```json\n{json.dumps(json.loads(_RESULT), indent=2)}\n```\n',
        '```json\n{"verdict":"APPROVE","comments":[],"discussion":"Use:\\n```py\\nx = 1\\n```"}\n```',
    ],
)
def test_one_surrounding_code_fence_is_stripped(tmp_path: Path, text: str) -> None:
    """Accept a final JSON object that the model wrapped in one Markdown code fence."""
    assert review_result(_stream(tmp_path, text)).verdict == 'APPROVE'


@pytest.mark.parametrize(
    'text',
    [
        f'Here is the review:\n```json\n{_RESULT}\n```',
        f'```json\n{_RESULT}\n```\nLet me know if you need more.',
        f'```json\n```json\n{_RESULT}\n```\n```',
        f'```json\n{_RESULT}',
        f'```json {_RESULT} ```',
    ],
)
def test_fence_with_other_text_still_fails(tmp_path: Path, text: str) -> None:
    """Only a bare fence is unwrapped; prose, nested or unterminated fences stay invalid."""
    with pytest.raises(ValidationError):
        review_result(_stream(tmp_path, text))


@pytest.mark.parametrize(
    'events',
    [
        [],
        [
            {
                'type': 'text',
                'part': {'messageID': 'msg_final', 'text': '{"verdict":"APPROVE","comments":[],"discussion":""}'},
            }
        ],
        [{'type': 'error'}, {'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'stop'}}],
        [
            {
                'type': 'text',
                'part': {'messageID': 'msg_old', 'text': '{"verdict":"APPROVE","comments":[],"discussion":""}'},
            },
            {'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'stop'}},
        ],
        [{'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'error'}}],
        [
            {'type': 'text', 'part': {'messageID': 'msg_final', 'text': 'not json'}},
            {'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'stop'}},
        ],
        [
            {'type': 'text', 'part': {'messageID': 'msg_final', 'text': '{"verdict":"APPROVE"}'}},
            {'type': 'step_finish', 'part': {'messageID': 'msg_final', 'reason': 'stop'}},
        ],
    ],
)
def test_incomplete_or_invalid_review_fails(tmp_path: Path, events: list[dict[str, object]]) -> None:
    """Never pass a partial stream, error, or invalid review to the publisher."""
    stream = tmp_path / 'review.ndjson'
    stream.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')
    with pytest.raises((ValueError, ValidationError)):
        review_result(stream)
