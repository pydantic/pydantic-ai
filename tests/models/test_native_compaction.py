"""`NativeCompaction` and `Agent.compact()` against the providers' own compaction APIs.

Each test has a two-turn conversation compacted, once during a run and once with `Agent.compact()`, then asks a
question only the compacted history can answer. The `FunctionModel` tests in
`tests/test_capability_native_compaction.py` cover the mapping in detail.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import Agent, CompactionPart, ModelMessage, ModelResponse, RunContext
from pydantic_ai.capabilities import AbstractCapability, NativeCompaction
from pydantic_ai.models import Model

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.vcr,
    pytest.mark.moves_cache_prefix(reason='compaction replaces the history the next request is built on'),
]


@dataclass
class RequestOnce(AbstractCapability[Any]):
    """Requests compaction for the run's first request."""

    async def before_run(self, ctx: RunContext[Any]) -> None:
        ctx.request_compaction(reason='threshold')


@pytest.fixture(
    params=[
        pytest.param('openai', marks=pytest.mark.skipif(not openai_available(), reason='openai not installed')),
        pytest.param(
            'anthropic', marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
        ),
    ]
)
def compacting_model(request: pytest.FixtureRequest, openai_api_key: str, anthropic_api_key: str) -> Model:
    if request.param == 'openai':
        return OpenAIResponsesModel('gpt-5.4-nano', provider=OpenAIProvider(api_key=openai_api_key))
    return AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))


def compaction_part(messages: list[ModelMessage]) -> CompactionPart:
    """The provider's compaction part the compacted history starts with."""
    first = messages[0]
    assert isinstance(first, ModelResponse)
    [part] = first.parts
    assert isinstance(part, CompactionPart)
    return part


async def test_native_compaction_in_a_run(allow_model_requests: None, compacting_model: Model):
    agent = Agent(compacting_model, instructions='Answer in one short sentence.', capabilities=[NativeCompaction()])
    first = await agent.run('My name is Bob and my favorite color is teal.')

    second = await agent.run(
        'What is my favorite color?', message_history=first.all_messages(), capabilities=[RequestOnce()]
    )

    assert 'teal' in second.output.lower()
    messages = second.all_messages()
    assert compaction_part(messages).provider_name == compacting_model.system
    [record] = second.compactions
    assert (record.handler, record.reason, record.messages_before, record.messages_after, record.kept_from) == snapshot(
        ('native', 'threshold', 3, 2, 1)
    )
    assert record.summarized == first.all_messages()
    assert len(messages) == 3


async def test_agent_compact(allow_model_requests: None, compacting_model: Model):
    agent = Agent(compacting_model, instructions='Answer in one short sentence.', capabilities=[NativeCompaction()])
    first = await agent.run('My name is Bob and my favorite color is teal.')

    compaction = await agent.compact(first.all_messages())

    assert compaction is not None
    assert compaction.part == compaction_part(compaction.messages)
    assert compaction.usage.output_tokens > 0
    assert compaction.record is not None
    assert (
        compaction.record.handler,
        compaction.record.reason,
        compaction.record.model_name,
        compaction.record.messages_before,
        compaction.record.messages_after,
        compaction.record.kept_from,
    ) == (
        'native',
        'manual',
        compacting_model.model_name,
        2,
        1,
        1,
    )
    assert compaction.record.summarized == first.all_messages()

    second = await agent.run('What is my favorite color?', message_history=compaction.messages)
    assert 'teal' in second.output.lower()


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_stateless_openai_compaction_without_the_profile_flag(allow_model_requests: None, openai_api_key: str):
    """`OpenAICompaction(stateless=True)` compacts on a Responses model whatever its profile's generic flag says.

    An OpenAI-compatible provider may not set `supports_compaction_on_demand`, which `NativeCompaction` relies on.
    """
    model = OpenAIResponsesModel(
        'gpt-5.4-nano',
        provider=OpenAIProvider(api_key=openai_api_key),
        profile={'supports_compaction_on_demand': False},
    )
    agent = Agent(
        model, instructions='Answer in one short sentence.', capabilities=[OpenAICompaction(message_count_threshold=2)]
    )
    first = await agent.run('My name is Bob and my favorite color is teal.')
    second = await agent.run('What is my favorite color?', message_history=first.all_messages())

    assert 'teal' in second.output.lower()
    assert compaction_part(second.all_messages()).provider_name == 'openai'
    assert [record.handler for record in second.compactions] == ['native']
