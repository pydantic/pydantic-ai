"""Tests for Monty-specific type checking and signature generation."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

pydantic_monty = pytest.importorskip('pydantic_monty')
from pydantic_monty import Monty  # noqa: E402

from pydantic_ai.exceptions import ModelRetry  # noqa: E402
from pydantic_ai.runtime.monty import (  # noqa: E402  # pyright: ignore[reportPrivateUsage]
    MontyRuntime,
    _build_type_check_prefix,
)

from .conftest import build_code_mode_toolset, run_code_with_tools  # noqa: E402

pytestmark = pytest.mark.anyio


def add(x: int, y: int) -> int:
    """Add two integers."""
    return x + y


async def test_type_error_raises_model_retry():
    """Type errors in generated code raise ModelRetry so the LLM can fix them."""
    with pytest.raises(ModelRetry) as exc_info:
        await run_code_with_tools('add("hello", "world")', MontyRuntime(), (add, False))

    assert str(exc_info.value) == snapshot("""\
Type error in generated code:
main.py:1:5: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["hello"]`
main.py:1:14: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["world"]`
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

async def add(x: int, y: int) -> int:
    """Add two integers."""
    raise NotImplementedError()\
''')
    # Verify Monty can parse and type check code using this prefix
    m = Monty('add(1, 2)', external_functions=['add'])
    m.type_check(prefix_code=prefix)  # Should not raise


async def test_signatures_use_ellipsis_monty_converts_for_type_check():
    """Signatures use '...' body; Monty converts to 'raise NotImplementedError()' for type checking."""
    code_mode, tools = await build_code_mode_toolset(MontyRuntime(), (add, False))

    # LLM-facing description should have '...'
    description = tools['run_code'].tool_def.description or ''
    assert '...' in description
    assert 'raise NotImplementedError()' not in description

    # Cached signatures should also have '...' (not NotImplementedError)
    assert code_mode._cached_signatures is not None  # pyright: ignore[reportPrivateUsage]
    assert all('...' in sig for sig in code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]
    assert not any('raise NotImplementedError()' in sig for sig in code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]

    # But when Monty builds the type-check prefix, it converts to 'raise NotImplementedError()'
    prefix = _build_type_check_prefix(code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]
    assert 'raise NotImplementedError()' in prefix
    assert '    ...' not in prefix
