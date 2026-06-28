from __future__ import annotations as _annotations

import re

import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.google import GoogleProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


def test_google_provider_need_api_key_mentions_supported_env_vars(env: TestEnv) -> None:
    env.remove('GOOGLE_API_KEY')
    env.remove('GEMINI_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable '
            'or pass it via `GoogleProvider(api_key=...)` to use the Gemini API.'
        ),
    ):
        GoogleProvider()  # pyright: ignore[reportCallIssue]
