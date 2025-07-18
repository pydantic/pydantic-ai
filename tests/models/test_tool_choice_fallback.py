"""Ensure `tool_choice='required'` is downgraded to `'auto'` when the profile says so."""

from __future__ import annotations

import types
from typing import Any

import pytest

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


@pytest.mark.anyio()
async def test_tool_choice_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'dummy')

    model = OpenAIModel('stub', provider='openai')

    # Make profile report lack of `tool_choice='required'` support but keep sampling
    def fake_from_profile(_p: Any) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            openai_supports_tool_choice_required=False,
            openai_supports_sampling_settings=True,
        )

    monkeypatch.setattr(OpenAIModelProfile, 'from_profile', fake_from_profile, raising=True)

    captured: dict[str, Any] = {}

    async def fake_create(*_a: Any, **kw: Any) -> dict[str, Any]:
        captured.update(kw)
        return {}

    # Patch chat completions create
    monkeypatch.setattr(model.client.chat.completions, 'create', fake_create, raising=True)

    params = ModelRequestParameters(function_tools=[ToolDefinition(name='x')], allow_text_output=False)

    await model._completions_create(  # pyright: ignore[reportPrivateUsage]
        messages=[],
        stream=False,
        model_settings={},
        model_request_parameters=params,
    )

    assert captured.get('tool_choice') == 'auto'
