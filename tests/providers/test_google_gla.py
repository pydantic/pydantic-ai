# pyright: reportDeprecated=false
import re

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from ..conftest import TestEnv

pytestmark = [
    pytest.mark.filterwarnings('ignore:`GoogleGLAProvider` is deprecated.:DeprecationWarning'),
]


def test_api_key_arg(env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')
    provider = GoogleGLAProvider(api_key='via-arg')
    assert provider.client.headers['x-goog-api-key'] == 'via-arg'
    assert provider.client.base_url == 'https://generativelanguage.googleapis.com/v1beta/models/'


def test_api_key_env_var(env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')
    provider = GoogleGLAProvider()
    assert 'x-goog-api-key' in dict(provider.client.headers)


def test_api_key_not_set(env: TestEnv):
    env.remove('GEMINI_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `GEMINI_API_KEY` environment variable or pass it via `GoogleGLAProvider(api_key=...)`'
        ),
    ):
        GoogleGLAProvider()


def test_api_key_empty(env: TestEnv):
    env.set('GEMINI_API_KEY', '')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `GEMINI_API_KEY` environment variable or pass it via `GoogleGLAProvider(api_key=...)`'
        ),
    ):
        GoogleGLAProvider()
