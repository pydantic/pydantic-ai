"""How an adapter's `compaction_mode` drives the pre-boundary trim.

The trim's *behavior* per mode is covered end to end by the provider suites
(`test_anthropic_trims_before_latest_compaction`, `test_openai_responses_trims_before_latest_compaction`
and the standing-prompt tests around them). What those can't reach is which mode each adapter
declares, and what an adapter that declares none does — so both are pinned here.
"""

from __future__ import annotations

import pytest

from pydantic_ai import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.messages import CompactionPart
from pydantic_ai.models import CompactionMode, Model
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider


def _build_model(model_id: str) -> Model:
    """Build the adapter under test.

    Called from inside the test rather than from `parametrize`, whose arguments are evaluated at
    collection time — before `skipif` applies — so touching the optional imports there would raise
    on the lanes that don't install them.
    """
    if model_id == 'openai-responses':
        return OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key='test'))
    elif model_id == 'openai-chat':
        return OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(api_key='test'))
    else:
        assert model_id == 'anthropic'
        return AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test'))


@pytest.mark.skipif(not imports_successful(), reason='openai or anthropic not installed')
@pytest.mark.parametrize(
    'model_id, expected',
    [
        ('openai-responses', 'encrypted'),
        # Shares the same `OpenAIModelProfile` as the Responses adapter, and must still be `None`:
        # Chat Completions has no compaction surface to round-trip a `CompactionPart` to.
        ('openai-chat', None),
        ('anthropic', 'text'),
    ],
)
def test_compaction_mode_declared_per_adapter(model_id: str, expected: CompactionMode | None):
    assert _build_model(model_id).compaction_mode == expected


def test_no_compaction_mode_leaves_history_untouched():
    """An adapter that declares no mode never trims, even with a boundary in the history.

    Unreachable through a provider adapter — both that implement the trim declare a mode — so the
    safe default every other adapter inherits is pinned directly.
    """

    def return_text(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('ok')])  # pragma: no cover

    model = FunctionModel(return_text)
    assert model.compaction_mode is None

    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt('before the boundary'),
        ModelResponse(parts=[CompactionPart(content='summary', provider_name='function')], provider_name='function'),
        ModelRequest.user_text_prompt('after the boundary'),
    ]

    assert model._trim_before_compaction(messages) == messages  # pyright: ignore[reportPrivateUsage]
