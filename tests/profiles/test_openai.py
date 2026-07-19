"""Tests for OpenAI model profiles.

Tests verify model profile detection for different OpenAI models, particularly the full desired
reasoning-flag matrix per model version: `openai_supports_reasoning`,
`openai_reasoning_enabled_by_default`, `openai_supports_reasoning_effort_none`, and
`openai_responses_supports_reasoning_mode`.
"""

from __future__ import annotations as _annotations

import re
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
class ReasoningCase:
    """One row of the desired reasoning matrix, mirroring `_REASONING_SUPPORT_BY_PREFIX`."""

    model: str
    enabled_by_default: bool = False
    """Reasoning is on when `reasoning_effort` is omitted."""
    can_be_disabled: bool = False
    """The model accepts `reasoning_effort='none'`, which also allows sampling params."""
    supports_mode: bool = False
    """The Responses API accepts `reasoning.mode` ('standard' | 'pro')."""


# Every cell verified against the live Responses API (2026-07): "enabled by default" = sampling
# params rejected with no `reasoning.effort` set; "can be disabled" = `effort='none'` accepted.
REASONING_CASES = [
    # o-series: always reasons, no off switch
    ReasoningCase(model='o1', enabled_by_default=True),
    ReasoningCase(model='o1-mini', enabled_by_default=True),
    ReasoningCase(model='o3', enabled_by_default=True),
    ReasoningCase(model='o3-mini', enabled_by_default=True),
    ReasoningCase(model='o4-mini', enabled_by_default=True),
    # gpt-5 (not 5.x): always reasons, no off switch
    ReasoningCase(model='gpt-5', enabled_by_default=True),
    ReasoningCase(model='gpt-5-pro', enabled_by_default=True),
    ReasoningCase(model='gpt-5-codex', enabled_by_default=True),
    ReasoningCase(model='gpt-5-turbo', enabled_by_default=True),
    # gpt-5.1..5.4 mainline: reasoning off by default, opt-in via effort
    ReasoningCase(model='gpt-5.1', can_be_disabled=True),
    ReasoningCase(model='gpt-5.1-turbo', can_be_disabled=True),
    ReasoningCase(model='gpt-5.1-mini', can_be_disabled=True),
    ReasoningCase(model='gpt-5.2', can_be_disabled=True),
    ReasoningCase(model='gpt-5.2-turbo', can_be_disabled=True),
    ReasoningCase(model='gpt-5.2-mini', can_be_disabled=True),
    ReasoningCase(model='gpt-5.3-codex', can_be_disabled=True),
    ReasoningCase(model='gpt-5.3-mini', can_be_disabled=True),
    ReasoningCase(model='gpt-5.4', can_be_disabled=True),
    ReasoningCase(model='gpt-5.4-mini', can_be_disabled=True),
    ReasoningCase(model='gpt-5.4-nano', can_be_disabled=True),
    # -pro and gpt-5.1 codex variants: always reason, no `effort='none'`
    ReasoningCase(model='gpt-5.1-codex', enabled_by_default=True),
    ReasoningCase(model='gpt-5.1-codex-max', enabled_by_default=True),
    ReasoningCase(model='gpt-5.2-pro', enabled_by_default=True),
    ReasoningCase(model='gpt-5.4-pro', enabled_by_default=True),
    ReasoningCase(model='gpt-5.5-pro', enabled_by_default=True),
    # gpt-5.1+ chat variants: always reason at a fixed 'medium' effort (sampling params rejected)
    ReasoningCase(model='gpt-5.1-chat-latest', enabled_by_default=True),
    ReasoningCase(model='gpt-5.2-chat-latest', enabled_by_default=True),
    ReasoningCase(model='gpt-5.3-chat-latest', enabled_by_default=True),
    # gpt-5.5: reasons by default AND can be turned off, like gpt-5.6 but without `reasoning.mode`
    ReasoningCase(model='gpt-5.5', enabled_by_default=True, can_be_disabled=True),
    # gpt-5.6: reasons by default AND can be turned off; the only family with `reasoning.mode`
    ReasoningCase(model='gpt-5.6-sol', enabled_by_default=True, can_be_disabled=True, supports_mode=True),
    ReasoningCase(model='gpt-5.6-terra', enabled_by_default=True, can_be_disabled=True, supports_mode=True),
    ReasoningCase(model='gpt-5.6-luna', enabled_by_default=True, can_be_disabled=True, supports_mode=True),
    # no reasoning
    ReasoningCase(model='gpt-5-chat'),
    ReasoningCase(model='gpt-4o'),
    ReasoningCase(model='gpt-4o-mini'),
    ReasoningCase(model='gpt-4o-2024-08-06'),
]


@pytest.mark.parametrize('case', REASONING_CASES, ids=lambda c: c.model)
def test_reasoning_matrix(case: ReasoningCase):
    """Pin the reasoning-flag matrix for every OpenAI model version, including the derived flags."""
    supports_reasoning = case.enabled_by_default or case.can_be_disabled

    profile = openai_model_profile(case.model)
    assert isinstance(profile, dict)
    assert profile.get('openai_supports_reasoning', False) is supports_reasoning
    assert profile.get('openai_reasoning_enabled_by_default', False) is case.enabled_by_default
    assert profile.get('openai_supports_reasoning_effort_none', False) is case.can_be_disabled
    assert profile.get('openai_responses_supports_reasoning_mode', False) is case.supports_mode
    assert profile.get('supports_thinking', False) is supports_reasoning
    assert profile.get('thinking_always_enabled', False) is (case.enabled_by_default and not case.can_be_disabled)


class TestEncryptedReasoningContent:
    """Tests for encrypted reasoning content support."""

    def test_reasoning_models_support_encrypted_content(self):
        """Models with reasoning support encrypted reasoning content."""
        for model in [
            'o1',
            'o3',
            'gpt-5',
            'gpt-5.1',
            'gpt-5.2',
            'gpt-5.3-codex',
            'gpt-5.3-chat-latest',
            'gpt-5.4',
            'gpt-5.5',
        ]:
            profile = openai_model_profile(model)
            assert isinstance(profile, dict)
            assert profile.get('openai_supports_encrypted_reasoning_content', False) is True

    def test_non_reasoning_models_no_encrypted_content(self):
        """Models without reasoning don't support encrypted reasoning content."""
        for model in ['gpt-4o', 'gpt-4o-mini', 'gpt-5-chat']:
            profile = openai_model_profile(model)
            assert isinstance(profile, dict)
            assert profile.get('openai_supports_encrypted_reasoning_content', False) is False


def test_send_back_thinking_parts_field_requires_thinking_field():
    with pytest.raises(
        UserError,
        match=re.escape(
            'If `openai_chat_send_back_thinking_parts` is "field", `openai_chat_thinking_field` must be set to a non-None value.'
        ),
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


@pytest.mark.parametrize(
    'model_name',
    [
        'gpt-5.6-sol',
        'gpt-5.5',
        'gpt-5.4',
        'gpt-5.3-codex',
        'gpt-oss-120b',
        'o3',
    ],
)
def test_vendor_prefixed_model_id_matches_bare_profile(model_name: str):
    """A vendor-namespaced id (e.g. Bedrock Mantle's `openai.<model>`) resolves
    to the same profile as the underlying bare OpenAI model id.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/6517:
    previously the `openai.` prefix shadowed the real family (so e.g.
    `openai.gpt-5.6-sol` lost `openai_supports_phase`) and collided with the
    `o`-series prefix check (so e.g. `openai.gpt-oss-120b` was wrongly flagged
    as a reasoning model).
    """
    prefixed = openai_model_profile(f'openai.{model_name}')
    bare = openai_model_profile(model_name)
    assert dict(prefixed) == dict(bare)
