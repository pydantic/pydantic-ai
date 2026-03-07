"""Tests for OpenAI model profiles.

Reasoning capability flags (supports_reasoning, supports_reasoning_effort_none)
are now tested via VCR ground-truth probes in test_openai_capabilities_vcr.py.
This file retains tests for other profile behaviors.
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


class TestEncryptedReasoningContent:
    """Tests for encrypted reasoning content support.

    These verify the derived flag: models with reasoning support also
    support encrypted reasoning content.
    """

    def test_reasoning_models_support_encrypted_content(self):
        for model in ['o1', 'o3', 'gpt-5', 'gpt-5.1', 'gpt-5.2', 'gpt-5.3-chat-latest', 'gpt-5.4']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_encrypted_reasoning_content is True

    def test_non_reasoning_models_no_encrypted_content(self):
        for model in ['gpt-4o', 'gpt-4o-mini', 'gpt-5-chat-latest']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_encrypted_reasoning_content is False
