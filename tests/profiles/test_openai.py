"""Tests for OpenAI model profiles.

Tests verify model profile detection for different OpenAI models, particularly:
- `openai_supports_reasoning`: Whether the model supports reasoning (o-series, GPT-5, GPT-5.1+)
- `openai_supports_reasoning_effort_none`: GPT-5.1+ models support sampling params when reasoning_effort='none'
- `OpenAIJsonSchemaTransformer`: strict-mode key handling
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


@dataclass
class SamplingParamsCase:
    model: str
    supports_reasoning: bool = False
    supports_reasoning_effort_none: bool = False


SAMPLING_PARAMS_CASES = [
    # o-series: reasoning enabled, no effort_none
    SamplingParamsCase(model='o1', supports_reasoning=True),
    SamplingParamsCase(model='o1-mini', supports_reasoning=True),
    SamplingParamsCase(model='o3', supports_reasoning=True),
    SamplingParamsCase(model='o3-mini', supports_reasoning=True),
    SamplingParamsCase(model='o4-mini', supports_reasoning=True),
    # gpt-5 (not 5.1+): reasoning enabled, no effort_none
    SamplingParamsCase(model='gpt-5', supports_reasoning=True),
    SamplingParamsCase(model='gpt-5-pro', supports_reasoning=True),
    SamplingParamsCase(model='gpt-5-turbo', supports_reasoning=True),
    # gpt-5.1+: reasoning + effort_none
    SamplingParamsCase(model='gpt-5.1', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.1-turbo', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.1-mini', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.1-codex-max', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.2', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.2-turbo', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.2-mini', supports_reasoning=True, supports_reasoning_effort_none=True),
    # no reasoning
    SamplingParamsCase(model='gpt-5-chat'),
    SamplingParamsCase(model='gpt-4o'),
    SamplingParamsCase(model='gpt-4o-mini'),
    SamplingParamsCase(model='gpt-4o-2024-08-06'),
]


@pytest.mark.parametrize('case', SAMPLING_PARAMS_CASES, ids=lambda c: c.model)
def test_sampling_params_support(case: SamplingParamsCase):
    """Test reasoning capability flags for OpenAI models."""
    profile = openai_model_profile(case.model)
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.openai_supports_reasoning is case.supports_reasoning
    assert profile.openai_supports_reasoning_effort_none is case.supports_reasoning_effort_none


class TestOpenAIJsonSchemaTransformerStrictIncompatibleKeys:
    """Regression tests for issue #4438.

    The 8 JSON Schema keywords below were missing from _STRICT_INCOMPATIBLE_KEYS,
    so they were neither stripped in strict=True mode nor detected as incompatible
    in strict=None mode. This caused pydantic-ai to incorrectly mark schemas with
    Pydantic Field() constraints (ge, le, pattern, min_length, etc.) as
    strict-compatible, and to emit invalid schemas to the OpenAI API when
    strict=True.
    """

    def _transform(self, schema: dict[str, Any], *, strict: bool | None = None) -> dict[str, Any]:
        """Run the transformer and return the result."""
        t = OpenAIJsonSchemaTransformer(schema, strict=strict)
        return t.walk()

    def _is_strict_compatible(self, schema: dict[str, Any]) -> bool:
        t = OpenAIJsonSchemaTransformer(schema, strict=None)
        t.walk()
        return t.is_strict_compatible

    # ------------------------------------------------------------------ #
    # strict=None: incompatible keys should make the schema incompatible  #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        'key,value',
        [
            ('pattern', r'^\d{3}$'),
            ('minimum', 0),
            ('maximum', 100),
            ('exclusiveMinimum', 0),
            ('exclusiveMaximum', 100),
            ('multipleOf', 5),
            ('minItems', 1),
            ('maxItems', 10),
        ],
    )
    def test_missing_keys_mark_schema_as_incompatible(self, key: str, value: Any) -> None:
        """Each previously-missing key must render the schema non-strict-compatible."""
        schema = {
            'type': 'object',
            'properties': {'x': {'type': 'number', key: value}},
            'required': ['x'],
            'additionalProperties': False,
        }
        assert not self._is_strict_compatible(schema), (
            f"Schema with '{key}' should not be strict-compatible, but was marked as compatible"
        )

    # ------------------------------------------------------------------ #
    # strict=True: incompatible keys must be stripped from the schema     #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        'key,value',
        [
            ('pattern', r'^\d{3}$'),
            ('minimum', 0),
            ('maximum', 100),
            ('exclusiveMinimum', 0),
            ('exclusiveMaximum', 100),
            ('multipleOf', 5),
            ('minItems', 1),
            ('maxItems', 10),
        ],
    )
    def test_missing_keys_are_stripped_in_strict_mode(self, key: str, value: Any) -> None:
        """Each previously-missing key must be stripped when strict=True."""
        # Build a simple schema that contains the incompatible keyword inside a property
        schema = {
            'type': 'object',
            'properties': {'x': {'type': 'number', key: value}},
            'required': ['x'],
            'additionalProperties': False,
        }
        result = self._transform(schema, strict=True)
        assert key not in result['properties']['x'], (
            f"Key '{key}' should have been stripped in strict=True mode but was still present"
        )

    def test_pattern_stripped_from_string_property(self) -> None:
        """Verify 'pattern' is stripped from a string property in strict mode."""
        schema = {
            'type': 'object',
            'properties': {'code': {'type': 'string', 'pattern': r'^\d{3}$'}},
            'required': ['code'],
            'additionalProperties': False,
        }
        result = self._transform(schema, strict=True)
        prop = result['properties']['code']
        assert 'pattern' not in prop
        # The constraint value should be preserved in the description
        assert 'pattern' in prop.get('description', '')

    def test_multiple_incompatible_keys_all_stripped_and_noted(self) -> None:
        """Multiple incompatible keys must all be stripped and noted in description."""
        schema = {
            'type': 'object',
            'properties': {
                'score': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 100,
                    'multipleOf': 5,
                }
            },
            'required': ['score'],
            'additionalProperties': False,
        }
        result = self._transform(schema, strict=True)
        prop = result['properties']['score']
        for key in ('minimum', 'maximum', 'multipleOf'):
            assert key not in prop, f"Key '{key}' should have been stripped"
        # All constraints should be noted in description
        description = prop.get('description', '')
        assert 'minimum' in description
        assert 'maximum' in description
        assert 'multipleOf' in description


class TestEncryptedReasoningContent:
    """Tests for encrypted reasoning content support."""

    def test_reasoning_models_support_encrypted_content(self):
        """Models with reasoning support encrypted reasoning content."""
        for model in ['o1', 'o3', 'gpt-5', 'gpt-5.1', 'gpt-5.2']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_encrypted_reasoning_content is True

    def test_non_reasoning_models_no_encrypted_content(self):
        """Models without reasoning don't support encrypted reasoning content."""
        for model in ['gpt-4o', 'gpt-4o-mini', 'gpt-5-chat']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_encrypted_reasoning_content is False
