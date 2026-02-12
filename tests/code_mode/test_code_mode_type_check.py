"""Tests for Monty-specific type checking and signature generation."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

try:
    from pydantic_monty import Monty

    from pydantic_ai.runtime.monty import (
        MontyRuntime,
        _build_type_check_prefix,  # pyright: ignore[reportPrivateUsage]
    )
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic-monty is not installed', allow_module_level=True)

from pydantic_ai._python_schema_types import _to_pascal_case  # pyright: ignore[reportPrivateUsage]
from pydantic_ai._python_signature import Signature
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets.code_mode import _dedup_typeddicts  # pyright: ignore[reportPrivateUsage]

from .conftest import build_code_mode_toolset, run_code_with_tools

pytestmark = pytest.mark.anyio


def add(*, x: int, y: int) -> int:
    """Add two integers."""
    return x + y


async def test_type_error_raises_model_retry():
    """Type errors in generated code raise ModelRetry so the LLM can fix them."""
    with pytest.raises(ModelRetry) as exc_info:
        await run_code_with_tools('add(x="hello", y="world")', MontyRuntime(), (add, False))

    assert str(exc_info.value) == snapshot("""\
Type error in generated code:
main.py:1:5: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["hello"]`
main.py:1:16: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["world"]`
""")


async def test_generated_signatures_are_valid_python():
    """Generated signatures must be valid Python that Monty can parse and type check."""
    code_mode, _tools = await build_code_mode_toolset(MontyRuntime(), (add, False))

    assert code_mode._cached_signatures is not None  # pyright: ignore[reportPrivateUsage]
    prefix = _build_type_check_prefix(code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]

    # `...` and `pass` are not valid for Monty/ty type checking — ty is intentionally
    # stricter than pyright here. See https://github.com/astral-sh/ty/issues/1922
    assert prefix == snapshot('''\
from typing import Any, TypedDict, NotRequired, Literal

async def add(*, x: int, y: int) -> int:
    """Add two integers."""
    raise NotImplementedError()\
''')
    # Verify Monty can parse and type check code using this prefix
    m = Monty('add(x=1, y=2)', external_functions=['add'])
    m.type_check(prefix_code=prefix)  # Should not raise


async def test_signatures_use_ellipsis_monty_converts_for_type_check():
    """Signatures use '...' body; Monty converts to 'raise NotImplementedError()' for type checking."""
    code_mode, tools = await build_code_mode_toolset(MontyRuntime(), (add, False))

    # LLM-facing description should have '...'
    description = tools['run_code_with_tools'].tool_def.description or ''
    assert '...' in description
    assert 'raise NotImplementedError()' not in description

    assert code_mode._cached_signatures is not None  # pyright: ignore[reportPrivateUsage]

    # But when Monty builds the type-check prefix, it converts to 'raise NotImplementedError()'
    prefix = _build_type_check_prefix(code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]
    assert 'raise NotImplementedError()' in prefix
    assert '    ...' not in prefix


def test_dedup_typeddicts_substring_names():
    """Renaming 'User' must not corrupt 'UserMeta' in the same signature."""
    sig1 = Signature(
        name='tool_a', params=['user: User'], return_type='Any', typeddicts=['class User(TypedDict):\n    name: str']
    )
    sig2 = Signature(
        name='tool_b',
        params=['user: User', 'meta: UserMeta'],
        return_type='UserMeta',
        typeddicts=['class User(TypedDict):\n    id: int', 'class UserMeta(TypedDict):\n    role: str'],
    )
    _dedup_typeddicts([sig1, sig2])
    # UserMeta must be untouched despite User being renamed
    assert sig2.params == ['user: tool_b_User', 'meta: UserMeta']
    assert sig2.return_type == 'UserMeta'


def test_to_pascal_case_digit_prefix():
    """PascalCase of a name starting with digits gets a leading underscore."""
    assert _to_pascal_case('123_tool') == '_123Tool'
