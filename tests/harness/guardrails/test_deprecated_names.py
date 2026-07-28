"""Tests for the deprecated `*Guard` aliases in `pydantic_ai_harness.guardrails`."""

from __future__ import annotations

import pytest

import pydantic_ai_harness
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.guardrails import (
    GuardrailResult,
    InputGuardrail,
    InputGuardrailFunc,
    OutputGuardrail,
    OutputGuardrailFunc,
)


def test_old_names_warn_and_resolve() -> None:
    import pydantic_ai_harness.guardrails as guardrails

    expected = {
        'GuardResult': GuardrailResult,
        'InputGuard': InputGuardrail,
        'InputGuardFunc': InputGuardrailFunc,
        'OutputGuard': OutputGuardrail,
        'OutputGuardFunc': OutputGuardrailFunc,
    }
    for old_name, new_object in expected.items():
        with pytest.warns(
            HarnessDeprecationWarning, match=f'`pydantic_ai_harness.guardrails.{old_name}` has been renamed'
        ):
            assert getattr(guardrails, old_name) is new_object


def test_old_names_resolve_from_package_root() -> None:
    with pytest.warns(HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.guardrails.InputGuardrail`'):
        assert pydantic_ai_harness.InputGuard is InputGuardrail


def test_new_names_resolve_from_package_root_without_warning() -> None:
    assert pydantic_ai_harness.InputGuardrail is InputGuardrail
    assert pydantic_ai_harness.OutputGuardrail is OutputGuardrail
    assert pydantic_ai_harness.GuardrailResult is GuardrailResult


def test_unknown_attribute_raises() -> None:
    import pydantic_ai_harness.guardrails as guardrails

    with pytest.raises(AttributeError, match='has no attribute'):
        _ = guardrails.DoesNotExist
