"""Tests for OpenAI model profiles.

Tests verify model profile detection for different OpenAI models, particularly:
- `openai_supports_reasoning`: Whether the model supports reasoning (o-series, GPT-5, GPT-5.1+)
- `openai_supports_reasoning_effort_none`: GPT-5.1+ models support sampling params when reasoning_effort='none'
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile

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


class TestStrictIncompatibleKeys:
    """Regression tests for strict-incompatible JSON Schema keys.

    See https://github.com/pydantic/pydantic-ai/issues/4438
    """

    def test_numeric_constraints_detected_as_incompatible(self):
        """Schema with minimum/maximum must be detected as strict-incompatible."""
        from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

        schema = {
            'type': 'object',
            'properties': {
                'age': {'type': 'integer', 'minimum': 0, 'maximum': 120},
            },
            'required': ['age'],
            'additionalProperties': False,
        }
        t = OpenAIJsonSchemaTransformer(schema.copy(), strict=None)
        t.walk()
        assert not t.is_strict_compatible

    def test_numeric_constraints_stripped_in_strict_mode(self):
        """strict=True must strip minimum/maximum from the schema."""
        from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

        schema = {
            'type': 'object',
            'properties': {
                'age': {'type': 'integer', 'minimum': 0, 'maximum': 120},
            },
            'required': ['age'],
            'additionalProperties': False,
        }
        t = OpenAIJsonSchemaTransformer(schema.copy(), strict=True)
        result = t.walk()
        age_schema = result['properties']['age']
        assert 'minimum' not in age_schema
        assert 'maximum' not in age_schema

    def test_pattern_detected_as_incompatible(self):
        """Schema with pattern must be detected as strict-incompatible."""
        from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

        schema = {
            'type': 'object',
            'properties': {
                'email': {'type': 'string', 'pattern': r'^[\w.-]+@[\w.-]+\.\w+$'},
            },
            'required': ['email'],
            'additionalProperties': False,
        }
        t = OpenAIJsonSchemaTransformer(schema.copy(), strict=None)
        t.walk()
        assert not t.is_strict_compatible

    def test_all_new_keys_stripped_in_strict_mode(self):
        """All newly added strict-incompatible keys must be stripped."""
        from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

        schema = {
            'type': 'object',
            'properties': {
                'value': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 100,
                    'exclusiveMinimum': 0,
                    'exclusiveMaximum': 100,
                    'multipleOf': 5,
                },
                'name': {
                    'type': 'string',
                    'pattern': r'^\w+$',
                },
                'tags': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'minItems': 1,
                    'maxItems': 10,
                },
            },
            'required': ['value', 'name', 'tags'],
            'additionalProperties': False,
        }
        t = OpenAIJsonSchemaTransformer(schema.copy(), strict=True)
        result = t.walk()
        value_schema = result['properties']['value']
        for key in ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf'):
            assert key not in value_schema, f'{key} should have been stripped'
        assert 'pattern' not in result['properties']['name']
        tags_schema = result['properties']['tags']
        assert 'minItems' not in tags_schema
        assert 'maxItems' not in tags_schema
