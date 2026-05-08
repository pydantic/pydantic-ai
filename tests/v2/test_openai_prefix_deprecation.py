"""Card 23: deprecation of the bare `'openai:'` prefix default.

In 1.x, `infer_model('openai:gpt-5')` resolves to `OpenAIChatModel` and emits
a `DeprecationWarning` because v2 will flip the default routing to the OpenAI
Responses API. The new `'openai-chat:'` prefix routes to `OpenAIChatModel`
without any warning, and the existing `'openai-responses:'` prefix continues
to route to `OpenAIResponsesModel` without any warning.
"""

from __future__ import annotations

import os
import warnings
from unittest.mock import patch

import pytest

from pydantic_ai.models import infer_model
from tests.conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

if not imports_successful():
    pytest.skip('openai package was not installed', allow_module_level=True)  # pragma: lax no cover


_OPENAI_RESPONSES_FLIP_MATCH = r"'openai:' will resolve to the OpenAI Responses API"
_OPENAI_ENV = {'OPENAI_API_KEY': 'mock-api-key'}


def test_bare_openai_prefix_resolves_to_chat_and_warns():
    with (
        patch.dict(os.environ, _OPENAI_ENV),
        pytest.warns(DeprecationWarning, match=_OPENAI_RESPONSES_FLIP_MATCH),
    ):
        model = infer_model('openai:gpt-5')
    assert isinstance(model, OpenAIChatModel)


def test_explicit_openai_chat_prefix_resolves_to_chat_silently():
    with patch.dict(os.environ, _OPENAI_ENV), warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        model = infer_model('openai-chat:gpt-5')
    assert isinstance(model, OpenAIChatModel)


def test_explicit_openai_responses_prefix_resolves_to_responses_silently():
    with patch.dict(os.environ, _OPENAI_ENV), warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        model = infer_model('openai-responses:gpt-5')
    assert isinstance(model, OpenAIResponsesModel)
