"""Validate OpenCode review completion and output before publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

sys.path.insert(0, str(Path(__file__).parent))

from category_classify_opencode import Classification
from douwebot_opencode_result import main, parse_answer
from publish_douwebot_review import ReviewResult
from run_pr_review_check import ModelResult

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
    assert parse_answer(ReviewResult, stream).verdict == 'APPROVE'


def test_line_separator_in_model_text_stays_in_its_event(tmp_path: Path) -> None:
    """OpenCode's `JSON.stringify` leaves U+2028 raw, and only LF ends an event."""
    discussion = 'First point.\u2028Second point.'
    result = json.dumps({'verdict': 'APPROVE', 'comments': [], 'discussion': discussion}, ensure_ascii=False)
    assert parse_answer(ReviewResult, _stream(tmp_path, result)).discussion == discussion


class _Answer(BaseModel):
    """The text around the JSON object in one answer layout, and the pull request that reported or tested it."""

    model_config = ConfigDict(extra='forbid')

    id: str
    source: str
    # Whether `source` reports a real model answer with this layout.
    observed: bool
    accepted: bool
    before: str
    after: str
    indent: int | None


# Job logs never print an answer, so the corpus holds the layouts pull requests reported or tested, not captured answers.
_CORPUS = TypeAdapter(list[_Answer]).validate_json(
    (Path(__file__).parent / 'fixtures' / 'opencode_answers.json').read_bytes()
)
# One answer per runner that reads one: douwebot, `PR Review Checks` and `Category Classify`.
_ANSWERS: dict[type[BaseModel], dict[str, object]] = {
    ReviewResult: {
        'verdict': 'REQUEST_CHANGES',
        'comments': [{'path': 'a.py', 'line': 1, 'side': 'RIGHT', 'body': 'Handle the empty case.'}],
        'discussion': '',
    },
    ModelResult: {
        'outcome': 'COMPLETE',
        'findings': [{'path': 'a.py', 'line': 3, 'side': 'RIGHT', 'severity': 'high', 'body': 'finding'}],
        'coverage': 'Reviewed a.py.',
        'private_report': '',
    },
    Classification: {'category': 'bug'},
}


@pytest.mark.parametrize('model', _ANSWERS, ids=lambda model: model.__name__)
@pytest.mark.parametrize('layout', _CORPUS, ids=lambda layout: layout.id)
def test_every_runner_reads_a_corpus_answer_alike(tmp_path: Path, layout: _Answer, model: type[BaseModel]) -> None:
    """Every runner accepts the JSON object that ends the answer, bare or fenced, after any prose, and nothing else."""
    answer = _ANSWERS[model]
    stream = _stream(tmp_path, layout.before + json.dumps(answer, indent=layout.indent) + layout.after)
    if layout.accepted:
        assert parse_answer(model, stream) == model.model_validate(answer)
    else:
        with pytest.raises(ValidationError):
            parse_answer(model, stream)


def test_braces_and_escapes_inside_json_strings_stay_in_the_answer(tmp_path: Path) -> None:
    """Braces and escaped quotes inside JSON strings, and a string that ends with an escaped backslash."""
    result = json.dumps(
        {
            'verdict': 'REQUEST_CHANGES',
            'comments': [{'path': 'a.py', 'line': 1, 'side': 'RIGHT', 'body': 'Write `f"{x}"`, not `"}"`.'}],
            'discussion': 'Ends with a backslash: \\',
        }
    )
    review = parse_answer(ReviewResult, _stream(tmp_path, f'Checked `{{`.\n\n{result}'))
    assert (review.comments[0].body, review.discussion) == ('Write `f"{x}"`, not `"}"`.', 'Ends with a backslash: \\')


@pytest.mark.parametrize(
    'text',
    [
        pytest.param('```\n' * 250_000 + 'x', id='fences'),
        pytest.param('"' * 1_000_000 + '}', id='quotes'),
        pytest.param('{' * 500_000 + '}' * 500_000, id='braces'),
    ],
)
def test_adversarial_answer_is_rejected_in_linear_time(tmp_path: Path, text: str) -> None:
    """A model-controlled answer of a million fence, quote or brace characters cannot stall validation."""
    with pytest.raises(ValidationError):
        parse_answer(ReviewResult, _stream(tmp_path, text))


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
        parse_answer(ReviewResult, _write_events(tmp_path, events))


def test_rejected_review_exits_with_its_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job log says why the answer was rejected without repeating it, and no result reaches the publisher."""
    output = tmp_path / 'github-output'
    output.touch()
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    monkeypatch.setattr(sys, 'argv', ['douwebot_opencode_result.py', str(_stream(tmp_path, 'Private finding.'))])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 'OpenCode returned an invalid review: validation failed: json_invalid'
    assert output.read_text(encoding='utf-8') == ''
