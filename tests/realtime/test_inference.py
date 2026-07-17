from __future__ import annotations as _annotations

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.realtime import infer_realtime_model

from ..conftest import TestEnv


def test_infer_realtime_models(env: TestEnv) -> None:
    env.set('OPENAI_API_KEY', 'test')
    env.set('XAI_API_KEY', 'test')
    env.set('GOOGLE_API_KEY', 'test')

    assert infer_realtime_model('openai:gpt-realtime').model_name == 'gpt-realtime'
    assert infer_realtime_model('xai:grok-voice-latest').model_name == 'grok-voice-latest'
    assert (
        infer_realtime_model('google:gemini-2.5-flash-native-audio-latest').model_name
        == 'gemini-2.5-flash-native-audio-latest'
    )
    assert infer_realtime_model('bedrock:amazon.nova-2-sonic-v1:0').model_name == 'amazon.nova-2-sonic-v1:0'


def test_infer_realtime_model_unknown_provider() -> None:
    with pytest.raises(UserError, match='Supported providers are `openai`, `xai`, `google`, and `bedrock`'):
        infer_realtime_model('anthropic:voice')
