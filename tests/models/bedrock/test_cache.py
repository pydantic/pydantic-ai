"""Bedrock prompt-caching regression tests.

Feature-centric file for Bedrock caching tests that pin wire-level invariants — the cacheable
tools-array prefix, cache_write/cache_read token activity across repeated requests, and the
interaction between `bedrock_cache_tool_definitions` and `toolChoice` shapes.
"""

from __future__ import annotations as _annotations

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent, ModelRequest, RunContext, ToolReturnPart
from pydantic_ai.settings import ModelSettings

from ...cassette_utils import get_bedrock_tool_config_from_cassette, get_bedrock_tool_names_from_cassette
from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelName, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
]


@pytest.mark.parametrize(
    'model_name,cache_tool_definitions',
    [
        pytest.param('us.anthropic.claude-sonnet-4-5-20250929-v1:0', True, id='anthropic'),
        pytest.param('us.amazon.nova-lite-v1:0', False, id='nova'),
    ],
)
async def test_bedrock_single_tool_choice_preserves_cache(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
    model_name: BedrockModelName,
    cache_tool_definitions: bool,
    vcr: Cassette,
):
    """Regression test for https://github.com/pydantic/pydantic-ai/issues/5672.

    Bedrock single-tool forcing must use native `toolChoice.tool` while preserving the full
    tools array, so prompt caching can read the same cacheable prefix on repeated requests.
    """
    model = BedrockConverseModel(model_name, provider=bedrock_provider)
    agent = Agent(
        model,
        system_prompt='YOU MUST USE TOOLS WHEN REQUIRED AND THEN ANSWER WITH ONLY THE TOOL RESULT.\n' * 100,
        model_settings=BedrockModelSettings(
            bedrock_cache_instructions=True,
            bedrock_cache_tool_definitions=cache_tool_definitions,
        ),
    )

    def force_catalog_lookup_before_result(ctx: RunContext[None]) -> ModelSettings:
        called = any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'catalog_lookup'
            for message in ctx.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if called:
            return ModelSettings()
        return BedrockModelSettings(tool_choice=['catalog_lookup'])

    @agent.tool_plain
    def catalog_lookup() -> str:
        return '21'

    @agent.tool_plain
    def diagnostics() -> str:  # pragma: no cover
        return 'diagnostics-ok'

    prompt = 'Call `catalog_lookup`, then answer with only its return value.'
    first = await agent.run(prompt, model_settings=force_catalog_lookup_before_result)
    assert '21' in first.output
    # Either a write or a read is acceptable on first run: the cassette may have been recorded
    # while Bedrock's prefix cache was already warm.
    assert first.usage.cache_write_tokens + first.usage.cache_read_tokens > 0

    second = await agent.run(prompt, model_settings=force_catalog_lookup_before_result)
    assert '21' in second.output
    assert second.usage.cache_read_tokens > 0

    tool_config = get_bedrock_tool_config_from_cassette(vcr)
    assert tool_config['toolChoice'] == {'tool': {'name': 'catalog_lookup'}}
    assert get_bedrock_tool_names_from_cassette(vcr) == ['catalog_lookup', 'diagnostics']
    assert any('cachePoint' in tool for tool in tool_config['tools']) is cache_tool_definitions
