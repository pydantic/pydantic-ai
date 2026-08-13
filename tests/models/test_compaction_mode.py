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

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai or anthropic not installed')


def _models() -> list[tuple[str, Model, CompactionMode | None]]:
    openai_provider = OpenAIProvider(api_key='test')
    return [
        ('openai-responses', OpenAIResponsesModel('gpt-5.2', provider=openai_provider), 'encrypted'),
        # Shares the same `OpenAIModelProfile` as the Responses adapter above, and must still be
        # `None`: Chat Completions has no compaction surface to round-trip a `CompactionPart` to.
        ('openai-chat', OpenAIChatModel('gpt-5.2', provider=openai_provider), None),
        (
            'anthropic',
            AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test')),
            'text',
        ),
        ('function', FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('ok')])), None),
    ]


@pytest.mark.parametrize('label, model, expected', _models(), ids=lambda value: value if isinstance(value, str) else '')
def test_compaction_mode_declared_per_adapter(label: str, model: Model, expected: CompactionMode | None):
    assert model.compaction_mode == expected


def test_no_compaction_mode_leaves_history_untouched():
    """An adapter that declares no mode never trims, even with a boundary in the history.

    Unreachable through a provider adapter — both that implement the trim declare a mode — so the
    safe default for every other adapter is pinned directly.
    """

    def return_text(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('ok')])  # pragma: no cover

    model = FunctionModel(return_text)
    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt('before the boundary'),
        ModelResponse(parts=[CompactionPart(content='summary', provider_name='function')], provider_name='function'),
        ModelRequest.user_text_prompt('after the boundary'),
    ]

    assert model._trim_before_compaction(messages) == messages  # pyright: ignore[reportPrivateUsage]
