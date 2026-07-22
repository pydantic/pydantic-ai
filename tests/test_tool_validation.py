"""Unit tests for the tool-call argument validation-failure classifier.

These are unit tests rather than VCR/public-API tests: the classifier is a small pure
function whose branches (unknown keys / type mismatch / missing / mixed / unparsable JSON)
can't be reliably or exhaustively provoked through a real provider, and its output is
definitory internal behavior that the instrumentation layer (and a future argument-repair
layer) depends on, so it's worth pinning directly against drift.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from pydantic_ai._tool_validation import (
    ToolArgsValidationFailure,
    classify_tool_args_validation_failure,
)


class _Args(BaseModel):
    model_config = ConfigDict(extra='forbid')

    a: int
    b: str


_args_ta = TypeAdapter(_Args)
_nested_ta = TypeAdapter(dict[str, _Args])


def _error_from_python(value: object) -> ValidationError:
    try:
        _args_ta.validate_python(value)
    except ValidationError as e:
        return e
    raise AssertionError('expected validation to fail')  # pragma: no cover


def _error_from_json(value: str) -> ValidationError:
    try:
        _args_ta.validate_json(value)
    except ValidationError as e:
        return e
    raise AssertionError('expected validation to fail')  # pragma: no cover


def _error_from_nested(value: object) -> ValidationError:
    try:
        _nested_ta.validate_python(value)
    except ValidationError as e:
        return e
    raise AssertionError('expected validation to fail')  # pragma: no cover


def test_unknown_keys_only():
    error = _error_from_python({'a': 1, 'b': 'x', 'bogus': 2, 'other': 3})
    assert classify_tool_args_validation_failure(error) == ToolArgsValidationFailure(
        kind='unknown_keys_only', unknown_keys=('bogus', 'other')
    )


def test_unknown_keys_are_bounded():
    # Nine distinct extra keys; only the first five are recorded, in first-seen order.
    payload = {'a': 1, 'b': 'x', **{f'k{i}': i for i in range(9)}}
    error = _error_from_python(payload)
    failure = classify_tool_args_validation_failure(error)
    assert failure.kind == 'unknown_keys_only'
    assert failure.unknown_keys == ('k0', 'k1', 'k2', 'k3', 'k4')


def test_nested_unknown_key_uses_leaf_name():
    error = _error_from_nested({'entry': {'a': 1, 'b': 'x', 'bogus': 2}})
    assert classify_tool_args_validation_failure(error) == ToolArgsValidationFailure(
        kind='unknown_keys_only', unknown_keys=('bogus',)
    )


def test_repeated_leaf_key_across_nested_objects_is_deduplicated():
    # Two sibling objects each carry an extra key with the same leaf name; it's recorded once.
    error = _error_from_nested({'e1': {'a': 1, 'b': 'x', 'bogus': 2}, 'e2': {'a': 1, 'b': 'x', 'bogus': 3}})
    assert classify_tool_args_validation_failure(error) == ToolArgsValidationFailure(
        kind='unknown_keys_only', unknown_keys=('bogus',)
    )


def test_type_mismatch_only():
    error = _error_from_python({'a': 'not-an-int', 'b': 'x'})
    assert classify_tool_args_validation_failure(error) == ToolArgsValidationFailure(kind='type_mismatch_only')


def test_missing_required():
    error = _error_from_python({'a': 1})
    assert classify_tool_args_validation_failure(error) == ToolArgsValidationFailure(kind='missing_required')


def test_mixed_missing_and_unknown():
    error = _error_from_python({'bogus': 2})
    assert classify_tool_args_validation_failure(error) == ToolArgsValidationFailure(kind='mixed')


def test_mixed_type_and_unknown():
    error = _error_from_python({'a': 'not-an-int', 'b': 'x', 'bogus': 2})
    assert classify_tool_args_validation_failure(error) == ToolArgsValidationFailure(kind='mixed')


@pytest.mark.parametrize('raw', ['{not valid json', '', 'null-ish {'])
def test_not_json(raw: str):
    error = _error_from_json(raw)
    assert classify_tool_args_validation_failure(error) == ToolArgsValidationFailure(kind='not_json')
