import pytest

from pydantic_ai._utils import validate_empty_kwargs
from pydantic_ai.exceptions import UserError


def test_validate_empty_kwargs_empty():
    """Test that empty dict passes validation."""
    validate_empty_kwargs({})


def test_validate_empty_kwargs_with_unknown():
    """Test that unknown kwargs raise UserError."""
    with pytest.raises(UserError, match='Unknown keyword arguments: `unknown_arg`'):
        validate_empty_kwargs({'unknown_arg': 'value'})


def test_validate_empty_kwargs_multiple_unknown():
    """Test that multiple unknown kwargs are properly formatted."""
    with pytest.raises(UserError, match='Unknown keyword arguments: `arg1`, `arg2`'):
        validate_empty_kwargs({'arg1': 'value1', 'arg2': 'value2'})


def test_validate_empty_kwargs_message_format():
    """Test that the error message format matches expected pattern."""
    with pytest.raises(UserError) as exc_info:
        validate_empty_kwargs({'test_arg': 'test_value'})

    assert 'Unknown keyword arguments: `test_arg`' in str(exc_info.value)


def test_validate_empty_kwargs_preserves_order():
    """Test that multiple kwargs preserve order in error message."""
    kwargs = {'first': '1', 'second': '2', 'third': '3'}
    with pytest.raises(UserError) as exc_info:
        validate_empty_kwargs(kwargs)

    error_msg = str(exc_info.value)
    assert '`first`' in error_msg
    assert '`second`' in error_msg
    assert '`third`' in error_msg
