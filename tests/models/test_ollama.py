from __future__ import annotations as _annotations

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserError,
    UserPromptPart,
)
from pydantic_ai.output import NativeOutput, PromptedOutput

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.ollama import OllamaProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


OLLAMA_CLOUD_BASE_URL = 'https://ollama.com/v1'
OLLAMA_LOCAL_BASE_URL = 'http://localhost:11434/v1'


class CityLocation(BaseModel):
    city: str
    country: str


class Pet(BaseModel):
    name: str
    animal: str
    age: int


# ---------- Unit tests for profile and capability wiring ----------


def _get_profile(model: OllamaModel) -> OpenAIModelProfile:
    return OpenAIModelProfile.from_profile(model.profile)


def test_local_ollama_supports_json_schema_output(ollama_api_key: str) -> None:
    """Self-hosted Ollama enforces `response_format` with `json_schema` via llama.cpp's
    grammar-constrained decoder, so the profile should leave `supports_json_schema_output`
    on."""
    provider = OllamaProvider(base_url=OLLAMA_LOCAL_BASE_URL, api_key=ollama_api_key)
    model = OllamaModel('qwen3:0.6b', provider=provider)
    profile = _get_profile(model)

    assert model.profile.supports_json_schema_output is True
    assert model.profile.supports_json_object_output is True
    assert profile.openai_supports_strict_tool_definition is False


def test_ollama_cloud_base_url_disables_json_schema_output(ollama_api_key: str) -> None:
    """Ollama Cloud accepts `response_format` with `json_schema` but does not enforce the
    schema upstream (see https://github.com/pydantic/pydantic-ai/issues/4917 and
    https://github.com/ollama/ollama/issues/12362), so the profile should advertise no
    native JSON schema support when the base URL is Ollama Cloud."""
    provider = OllamaProvider(base_url=OLLAMA_CLOUD_BASE_URL, api_key=ollama_api_key)
    model = OllamaModel('gpt-oss:20b', provider=provider)
    profile = _get_profile(model)

    assert model.profile.supports_json_schema_output is False
    assert model.profile.supports_json_object_output is True
    assert profile.openai_supports_strict_tool_definition is False


def test_local_ollama_cloud_suffix_disables_json_schema_output(ollama_api_key: str) -> None:
    """A local Ollama daemon forwards `-cloud` suffixed models to the same upstream that
    the direct Cloud path hits, so the same capability downgrade must apply."""
    provider = OllamaProvider(base_url=OLLAMA_LOCAL_BASE_URL, api_key=ollama_api_key)
    model = OllamaModel('gpt-oss:20b-cloud', provider=provider)

    assert model.profile.supports_json_schema_output is False
    assert model.profile.supports_json_object_output is True


def test_ollama_explicit_profile_overrides_cloud_detection(ollama_api_key: str) -> None:
    """Passing an explicit `profile` must win: an advanced user who knows their Cloud
    path actually enforces schemas can opt back in."""
    provider = OllamaProvider(base_url=OLLAMA_CLOUD_BASE_URL, api_key=ollama_api_key)
    override = OpenAIModelProfile(supports_json_schema_output=True)
    model = OllamaModel('gpt-oss:20b', provider=provider, profile=override)

    assert model.profile.supports_json_schema_output is True


async def test_ollama_cloud_native_output_raises(allow_model_requests: None, ollama_api_key: str) -> None:
    """`NativeOutput` against an Ollama Cloud model must raise a clear
    [`UserError`][pydantic_ai.exceptions.UserError] rather than silently falling into a
    retry loop (which is what used to happen per #4917). The error fires in
    `customize_request_parameters` before any HTTP is sent."""
    provider = OllamaProvider(base_url=OLLAMA_CLOUD_BASE_URL, api_key=ollama_api_key)
    model = OllamaModel('gpt-oss:20b', provider=provider)
    agent = Agent(model, output_type=NativeOutput(CityLocation))

    with pytest.raises(UserError, match='Native structured output is not supported'):
        await agent.run('What is the capital of France?')


async def test_ollama_local_cloud_suffix_native_output_raises(allow_model_requests: None, ollama_api_key: str) -> None:
    provider = OllamaProvider(base_url=OLLAMA_LOCAL_BASE_URL, api_key=ollama_api_key)
    model = OllamaModel('gpt-oss:20b-cloud', provider=provider)
    agent = Agent(model, output_type=NativeOutput(CityLocation))

    with pytest.raises(UserError, match='Native structured output is not supported'):
        await agent.run('What is the capital of France?')


def test_ollama_provider_name_routes_through_ollama_model() -> None:
    """`Agent('ollama:<model>')` should resolve to an [`OllamaModel`][pydantic_ai.models.ollama.OllamaModel]
    (not a bare [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]), so the Cloud detection
    kicks in automatically for users who configure Ollama via the `OLLAMA_BASE_URL` env var."""
    import os

    prev_base = os.environ.get('OLLAMA_BASE_URL')
    prev_key = os.environ.get('OLLAMA_API_KEY')
    os.environ['OLLAMA_BASE_URL'] = OLLAMA_CLOUD_BASE_URL
    os.environ['OLLAMA_API_KEY'] = 'test-key'
    try:
        agent = Agent('ollama:gpt-oss:20b')
        assert isinstance(agent.model, OllamaModel)
        assert agent.model.profile.supports_json_schema_output is False
    finally:
        if prev_base is None:
            os.environ.pop('OLLAMA_BASE_URL', None)
        else:
            os.environ['OLLAMA_BASE_URL'] = prev_base
        if prev_key is None:
            os.environ.pop('OLLAMA_API_KEY', None)
        else:
            os.environ['OLLAMA_API_KEY'] = prev_key


# ---------- VCR integration tests against live Ollama ----------


@pytest.fixture(scope='module')
def vcr_config():
    """Override the repo-wide `vcr_config` fixture so that localhost traffic is recorded.

    The local Ollama tests need to replay `http://localhost:11434` cassettes in CI where
    there is no running daemon; the repo default ignores localhost so those requests
    would pass through live.
    """
    return {
        'ignore_localhost': False,
        'filter_headers': ['authorization', 'x-api-key'],
        'decode_compressed_response': True,
    }


@pytest.mark.vcr
async def test_ollama_local_native_output_uses_json_schema(allow_model_requests: None, ollama_api_key: str) -> None:
    """Self-hosted Ollama with a local model should accept `NativeOutput` and produce
    schema-valid output in a single request, via llama.cpp's grammar-constrained decoder."""
    provider = OllamaProvider(base_url=OLLAMA_LOCAL_BASE_URL, api_key=ollama_api_key)
    model = OllamaModel('qwen3:0.6b', provider=provider)
    agent = Agent(model, output_type=NativeOutput(CityLocation))

    result = await agent.run('What is the capital of France?')

    assert result.output == snapshot(CityLocation(city='Paris', country='France'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="Okay, the user is asking for the capital of France. I know that France's capital is Paris. But I need to make sure I'm correct here. Let me think. Paris is indeed the capital since it's where the government is based. It's a major city, so I should mention that it's a key city. Also, maybe add some context like its role in the country. I should keep it simple and direct since the user is just asking the question. Double-checking my knowledge to avoid any mistakes. Yep, I'm confident the answer is Paris.\n",
                        id='reasoning',
                        provider_name='ollama',
                    ),
                    TextPart(content='{ "city": "Paris", "country": "France" }'),
                ],
                usage=RequestUsage(input_tokens=136, output_tokens=15),
                model_name='qwen3:0.6b',
                timestamp=IsDatetime(),
                provider_name='ollama',
                provider_url='http://localhost:11434/v1/',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr
async def test_ollama_cloud_prompted_output(allow_model_requests: None, ollama_api_key: str) -> None:
    """Ollama Cloud supports `response_format: {type: "json_object"}` (free-form JSON);
    [`PromptedOutput`][pydantic_ai.output.PromptedOutput] routes through that path."""
    provider = OllamaProvider(base_url=OLLAMA_CLOUD_BASE_URL, api_key=ollama_api_key)
    model = OllamaModel('gpt-oss:20b', provider=provider)
    agent = Agent(model, output_type=PromptedOutput(Pet))

    result = await agent.run('Generate a pet: a 3 year old black cat named Loki')

    assert result.output == snapshot(Pet(name='Loki', animal='cat', age=3))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Generate a pet: a 3 year old black cat named Loki', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='We need to output JSON with name, animal, age. It\'s a 3 year old black cat named Loki. So name: "Loki". animal: "cat" or "black cat"? Schema: animal is string. Use "cat". age: 3. Output JSON. No markdown.',
                        id='reasoning',
                        provider_name='ollama',
                    ),
                    TextPart(content='{"name":"Loki","animal":"cat","age":3}'),
                ],
                usage=RequestUsage(input_tokens=172, output_tokens=88),
                model_name='gpt-oss:20b',
                timestamp=IsDatetime(),
                provider_name='ollama',
                provider_url='https://ollama.com/v1/',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr
async def test_ollama_cloud_tool_output(allow_model_requests: None, ollama_api_key: str) -> None:
    """The default [`ToolOutput`][pydantic_ai.output.ToolOutput] path works on Ollama Cloud;
    this is the fallback users get when they drop `NativeOutput`."""
    provider = OllamaProvider(base_url=OLLAMA_CLOUD_BASE_URL, api_key=ollama_api_key)
    model = OllamaModel('gpt-oss:20b', provider=provider)
    agent = Agent(model, output_type=CityLocation)

    result = await agent.run('What is the capital of France?')

    assert result.output == snapshot(CityLocation(city='Paris', country='France'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='We need to answer question: "What is the capital of France?" The answer: Paris. The user didn\'t request the output to be in a specific format, but the system instructions specify that the tool should be used only if user requests. The user didn\'t request a tool. So we can produce a direct answer. The instruction: "When I need to: use tool, say I need it with a brief statement" - that\'s for tool usage. We don\'t need the tool. Just give the answer. Ensure no extra marks. Just plain text.',
                        id='reasoning',
                        provider_name='ollama',
                    ),
                    TextPart(content='Paris.'),
                ],
                usage=RequestUsage(input_tokens=134, output_tokens=122),
                model_name='gpt-oss:20b',
                timestamp=IsDatetime(),
                provider_name='ollama',
                provider_url='https://ollama.com/v1/',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            {
                                'type': 'json_invalid',
                                'loc': (),
                                'msg': 'Invalid JSON: expected value at line 1 column 1',
                                'input': 'Paris.',
                                'ctx': {'error': 'expected value at line 1 column 1'},
                            }
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The conversation: user asked "What is the capital of France?" assistant responded "Paris." That is fine. But they expect a JSON output? They saw a validation error: "Expected JSON: invalid JSON." The system is set up such that the assistant must respond using the function "final_result" JSON object with keys city and country. The previous response was not JSON; that's why validator error. So we must call function final_result with city = "Paris" and country = "France". We need to produce JSON accordingly. The function signature: final_result({city: string, country: string}). We need to output the JSON. Using the functions interface. Let's produce: { "city": "Paris", "country": "France" } as JSON.

We need to provide answer in JSON format. Let's do that.\
""",
                        id='reasoning',
                        provider_name='ollama',
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city":"Paris","country":"France"}',
                        tool_call_id='call_o2vnpxrw',
                    ),
                ],
                usage=RequestUsage(input_tokens=206, output_tokens=194),
                model_name='gpt-oss:20b',
                timestamp=IsDatetime(),
                provider_name='ollama',
                provider_url='https://ollama.com/v1/',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_o2vnpxrw',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
