"""Tests for Anthropic's strict-schema complexity limits on Claude through Bedrock's Converse API.

Converse compiles Claude's strict schemas with the same limits as Anthropic's own API:
https://platform.claude.com/docs/en/build-with-claude/structured-outputs#schema-complexity-limits
"""

from __future__ import annotations as _annotations

from typing import Any

import pytest
from cassetter import Cassette
from pydantic import BaseModel, create_model

from pydantic_ai import Tool
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelHTTPError, ModelRequestRejectedError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import NativeOutput, OutputObjectDefinition

from ..._inline_snapshot import snapshot
from ...cassette_utils import request_json
from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.vcr,
]

_MODEL_NAME = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'


def _model(name: str, fields: dict[str, Any]) -> type[BaseModel]:
    return create_model(name, **fields)


# 17 fields of type `int | None`, one more union-typed parameter than Anthropic allows in strict schemas.
Survey = _model('Survey', {f'answer_{i}': (int | None, None) for i in range(17)})


async def test_bedrock_default_output_over_limits_uses_non_strict_tool_output(
    allow_model_requests: None, bedrock_provider: BedrockProvider, vcr: Cassette
):
    """A bare `output_type` on a request that thinks uses Native Output, unless its schema is over the limits.

    The output tool is sent instead, without `strict` and without forcing the call, as the request thinks.
    """
    agent = Agent(BedrockConverseModel(_MODEL_NAME, provider=bedrock_provider), output_type=Survey)

    result = await agent.run(
        'Fill in a survey answer with `answer_0` set to 1 and every other field left empty.',
        model_settings={'thinking': True},
    )

    assert isinstance(result.output, Survey)
    assert result.output.model_dump(exclude_none=True) == {'answer_0': 1}
    body = request_json(vcr.requests[0])
    assert {
        'outputConfig': 'outputConfig' in body,
        'tools': [(tool['toolSpec']['name'], tool['toolSpec'].get('strict')) for tool in body['toolConfig']['tools']],
        'toolChoice': body['toolConfig'].get('toolChoice'),
    } == snapshot({'outputConfig': False, 'tools': [('final_result', None)], 'toolChoice': {'auto': {}}})
    assert [
        part.tool_name
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ] == ['final_result']


async def test_bedrock_explicit_strict_over_limits_raises_before_sending(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """A `NativeOutput` that Bedrock would reject raises a `ModelRequestRejectedError` before a request is sent, so there's no cassette."""
    agent = Agent(BedrockConverseModel(_MODEL_NAME, provider=bedrock_provider), output_type=NativeOutput(Survey))

    with pytest.raises(ModelRequestRejectedError, match='17 parameters with union types, over the limit of 16'):
        await agent.run('Fill in a survey answer.')


async def test_bedrock_strict_tools_over_limits_raise_before_sending(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """Strict function tools count toward the limits too.

    The error is raised before a request is sent, so there's nothing to record.
    """

    def lookup(key: str) -> str:
        return key  # pragma: no cover

    agent = Agent(
        BedrockConverseModel(_MODEL_NAME, provider=bedrock_provider),
        tools=[Tool(lookup, name=f'lookup_{i}', strict=True) for i in range(21)],
    )

    with pytest.raises(ModelRequestRejectedError, match='21 strict tools, over the limit of 20'):
        await agent.run('Look something up.')


def test_bedrock_non_claude_model_isnt_checked(bedrock_provider: BedrockProvider):
    """The limits are Anthropic's, so another model family's over-limit `NativeOutput` is prepared for sending.

    A unit test, since what's under test is that preparing the request doesn't raise.
    """
    model = BedrockConverseModel('qwen.qwen3-32b-v1:0', provider=bedrock_provider)
    params = ModelRequestParameters(
        output_mode='native', output_object=OutputObjectDefinition(json_schema=Survey.model_json_schema())
    )

    _, prepared = model.prepare_request(None, params)

    assert prepared.output_mode == 'native'
    assert prepared.output_object is not None and prepared.output_object.strict is True


async def test_bedrock_schema_too_complex_error_has_hint(allow_model_requests: None, bedrock_provider: BedrockProvider):
    """A schema within the explicit limits can still be too complex to compile; the 400 then points at the docs."""
    form = _model('Form', {f'field_{i}': (str, '') for i in range(13)})
    agent = Agent(BedrockConverseModel(_MODEL_NAME, provider=bedrock_provider), output_type=NativeOutput(form))

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Fill in the form.')

    error = exc_info.value
    assert isinstance(error, ModelRequestRejectedError)
    assert error.status_code == 400
    assert error.hint is not None
    assert error.hint.endswith('See https://pydantic.dev/docs/ai/models/anthropic/#schema-complexity-limits')


def test_bedrock_strict_tools_under_tool_choice_none_arent_counted(bedrock_provider: BedrockProvider):
    """`tool_choice='none'` sends Converse no tools until the history has tool blocks, so its strict tools aren't counted.

    A unit test, since what's under test is that preparing the request doesn't raise.
    """

    def lookup(key: str) -> str:
        return key  # pragma: no cover

    model = BedrockConverseModel(_MODEL_NAME, provider=bedrock_provider)
    params = ModelRequestParameters(
        function_tools=[Tool(lookup, name=f'lookup_{i}', strict=True).tool_def for i in range(21)]
    )

    _, prepared = model.prepare_request({'tool_choice': 'none'}, params)

    assert len(prepared.function_tools) == 21
