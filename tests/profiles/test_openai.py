"""Tests for OpenAI model profiles.

Tests verify model profile detection for different OpenAI models, particularly:
- `openai_supports_reasoning`: Whether the model supports reasoning (o-series, GPT-5, GPT-5.1+)
- `openai_supports_reasoning_effort_none`: GPT-5.1+ models support sampling params when reasoning_effort='none'
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.profiles.openai import (
        OpenAIJsonSchemaTransformer,
        OpenAIModelProfile,
        openai_model_profile,
        validate_openai_profile,
    )

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
    SamplingParamsCase(model='gpt-5.3-codex', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.3-mini', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.4', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.4-mini', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.4-nano', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.4-pro', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.5', supports_reasoning=True, supports_reasoning_effort_none=True),
    SamplingParamsCase(model='gpt-5.5-pro', supports_reasoning=True, supports_reasoning_effort_none=True),
    # no reasoning
    SamplingParamsCase(model='gpt-5.3-chat-latest'),
    SamplingParamsCase(model='gpt-5-chat'),
    SamplingParamsCase(model='gpt-4o'),
    SamplingParamsCase(model='gpt-4o-mini'),
    SamplingParamsCase(model='gpt-4o-2024-08-06'),
]


@pytest.mark.parametrize('case', SAMPLING_PARAMS_CASES, ids=lambda c: c.model)
def test_sampling_params_support(case: SamplingParamsCase):
    """Test reasoning capability flags for OpenAI models."""
    profile = openai_model_profile(case.model)
    assert isinstance(profile, dict)
    assert profile.get('openai_supports_reasoning', False) is case.supports_reasoning
    assert profile.get('openai_supports_reasoning_effort_none', False) is case.supports_reasoning_effort_none


class TestEncryptedReasoningContent:
    """Tests for encrypted reasoning content support."""

    def test_reasoning_models_support_encrypted_content(self):
        """Models with reasoning support encrypted reasoning content."""
        for model in ['o1', 'o3', 'gpt-5', 'gpt-5.1', 'gpt-5.2', 'gpt-5.3-codex', 'gpt-5.4', 'gpt-5.5']:
            profile = openai_model_profile(model)
            assert isinstance(profile, dict)
            assert profile.get('openai_supports_encrypted_reasoning_content', False) is True

    def test_non_reasoning_models_no_encrypted_content(self):
        """Models without reasoning don't support encrypted reasoning content."""
        for model in ['gpt-4o', 'gpt-4o-mini', 'gpt-5-chat', 'gpt-5.3-chat-latest']:
            profile = openai_model_profile(model)
            assert isinstance(profile, dict)
            assert profile.get('openai_supports_encrypted_reasoning_content', False) is False


def test_send_back_thinking_parts_field_requires_thinking_field():
    with pytest.raises(
        UserError,
        match='If `openai_chat_send_back_thinking_parts` is "field", `openai_chat_thinking_field` must be set to a non-None value.',
    ):
        validate_openai_profile(OpenAIModelProfile(openai_chat_send_back_thinking_parts='field'))


def test_json_schema_transformer_keeps_supported_patterns():
    class MyModel(BaseModel):
        simple_pattern: Annotated[str, Field(pattern='^my-pattern$')]

    schema_transformer = OpenAIJsonSchemaTransformer(MyModel.model_json_schema(), strict=None)

    assert schema_transformer.walk() == snapshot(
        {
            'properties': {'simple_pattern': {'pattern': '^my-pattern$', 'type': 'string'}},
            'required': ['simple_pattern'],
            'type': 'object',
            'additionalProperties': False,
        }
    )
    assert schema_transformer.is_strict_compatible is True

    escaped_schema_transformer = OpenAIJsonSchemaTransformer(
        {
            'properties': {'escaped_literal': {'pattern': '\\(?=USD', 'type': 'string'}},
            'required': ['escaped_literal'],
            'type': 'object',
        },
        strict=None,
    )
    assert escaped_schema_transformer.walk() == snapshot(
        {
            'properties': {'escaped_literal': {'pattern': '\\(?=USD', 'type': 'string'}},
            'required': ['escaped_literal'],
            'type': 'object',
            'additionalProperties': False,
        }
    )
    assert escaped_schema_transformer.is_strict_compatible is True


def test_json_schema_transformer_removes_unsupported_regex_lookarounds():
    json_schema: dict[str, Any] = {
        'properties': {
            'before': {'pattern': '(?<=USD)\\d+', 'type': 'string'},
            'after': {'pattern': '\\d+(?=USD)', 'type': 'string'},
            'negative_before': {'pattern': '(?<!USD)\\d+', 'type': 'string'},
            'negative_after': {'pattern': '\\d+(?!USD)', 'type': 'string'},
        },
        'required': ['before', 'after', 'negative_before', 'negative_after'],
        'type': 'object',
    }

    schema_transformer = OpenAIJsonSchemaTransformer(json_schema, strict=None)

    assert schema_transformer.walk() == snapshot(
        {
            'properties': {
                'before': {'pattern': '(?<=USD)\\d+', 'type': 'string'},
                'after': {'pattern': '\\d+(?=USD)', 'type': 'string'},
                'negative_before': {'pattern': '(?<!USD)\\d+', 'type': 'string'},
                'negative_after': {'pattern': '\\d+(?!USD)', 'type': 'string'},
            },
            'required': ['before', 'after', 'negative_before', 'negative_after'],
            'type': 'object',
            'additionalProperties': False,
        }
    )
    assert schema_transformer.is_strict_compatible is False

    assert OpenAIJsonSchemaTransformer(json_schema, strict=True).walk() == snapshot(
        {
            'properties': {
                'before': {'type': 'string', 'description': 'pattern=(?<=USD)\\d+'},
                'after': {'type': 'string', 'description': 'pattern=\\d+(?=USD)'},
                'negative_before': {'type': 'string', 'description': 'pattern=(?<!USD)\\d+'},
                'negative_after': {'type': 'string', 'description': 'pattern=\\d+(?!USD)'},
            },
            'required': ['before', 'after', 'negative_before', 'negative_after'],
            'type': 'object',
            'additionalProperties': False,
        }
    )
