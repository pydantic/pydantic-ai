import pytest

from pydantic_ai import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from tests.conftest import TestEnv


def test_infer_str_openai(env: TestEnv):
    env.set('OPENAI_API_KEY', 'via-env-var')
    m = infer_model('openai:gpt-3.5-turbo')
    assert isinstance(m, OpenAIModel)
    assert m.name() == 'openai:gpt-3.5-turbo'

    m2 = infer_model(m)
    assert m2 is m


def test_infer_str_gemini(env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')
    m = infer_model('gemini-1.5-flash')
    assert isinstance(m, GeminiModel)
    assert m.name() == 'gemini-1.5-flash'


def test_infer_str_unknown():
    with pytest.raises(UserError, match='Unknown model: foobar'):
        infer_model('foobar')  # pyright: ignore[reportArgumentType]
