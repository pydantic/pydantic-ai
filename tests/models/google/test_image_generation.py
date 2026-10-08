"""Tests for image generation on `GoogleModel`.

Gemini image models return images alongside or instead of text; `ImageGenerationTool` only configures them
(size and aspect ratio, plus output format and compression on Vertex AI). On a text model an `ImageGeneration`
capability runs its local fallback instead.
"""

from __future__ import annotations as _annotations

import re
from datetime import UTC
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    BinaryImage,
    FilePart,
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    UserPromptPart,
)
from pydantic_ai.capabilities import ImageGeneration, NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import ImageGenerationTool, WebSearchTool
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

from ..._inline_snapshot import snapshot
from ...conftest import IsDatetime, IsInstance, IsNow, IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.vcr,
]


async def test_google_image_generation_text_model_runs_local_fallback(
    allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture
):
    """A Gemini text model has no native image generation, so `ImageGeneration` sends its local tool instead."""
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=30))
    prompts: list[str] = []

    def generate_image(prompt: str) -> str:
        """Generate an image from a text prompt."""
        prompts.append(prompt)
        return 'The image was generated and shown to the user.'

    agent = Agent(
        GoogleModel('gemini-2.5-flash', provider=provider), capabilities=[ImageGeneration(local=generate_image)]
    )
    await agent.run('Generate an image of an axolotl.')

    assert prompts == snapshot(['axolotl'])
    body = request_capture.body(':generateContent')
    assert body['tools'] == snapshot(
        [
            {
                'functionDeclarations': [
                    {
                        'description': 'Generate an image from a text prompt.',
                        'name': 'generate_image',
                        'parameters_json_schema': {
                            'additionalProperties': False,
                            'properties': {'prompt': {'type': 'string'}},
                            'required': ['prompt'],
                            'type': 'object',
                        },
                    }
                ]
            }
        ]
    )
    assert body['generationConfig'] == snapshot({'responseModalities': ['TEXT']})


@pytest.mark.parametrize(
    ('model_name', 'profile', 'supports_native'),
    [
        ('gemini-2.5-flash', None, False),
        ('gemini-3.1-flash-image', None, True),
        ('gemini-2.5-flash', ModelProfile(supports_image_output=True), True),
    ],
)
def test_google_image_generation_tool_follows_supports_image_output(
    gemini_api_key: str, model_name: str, profile: ModelProfile | None, supports_native: bool
):
    """`ImageGenerationTool` is supported exactly when the resolved `supports_image_output` is true.

    Pinned on the profile because the flag is resolved there, including a user `profile=` override. The
    request paths are recorded in `test_google_image_generation_text_model_runs_local_fallback` (flag off)
    and `test_google_image_or_text_output` (flag on).
    """
    model = GoogleModel(model_name, provider=GoogleProvider(api_key=gemini_api_key), profile=profile)
    assert (ImageGenerationTool in model.profile.get('supported_native_tools', frozenset())) is supports_native


def test_google_optional_image_generation_tool_dropped_on_text_model(gemini_api_key: str):
    """An optional `ImageGenerationTool` on a text model is dropped instead of raising.

    Not a VCR test: the tool is resolved in `prepare_request`, before any request is built.
    """
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=gemini_api_key))
    _, params = model.prepare_request(None, ModelRequestParameters(native_tools=[ImageGenerationTool(optional=True)]))
    assert params.native_tools == []


async def test_google_image_generation_tool(allow_model_requests: None, gemini_api_key: str):
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(model=model, capabilities=[NativeTool(ImageGenerationTool())])

    with pytest.raises(
        UserError,
        match=re.escape(
            "`ImageGenerationTool` is not supported by model 'gemini-2.5-flash'. "
            "Use a model with 'image' in the name, or `ImageGeneration(local=...)` for a local fallback."
        ),
    ):
        await agent.run('Generate an image of an axolotl.')


async def test_google_image_generation(allow_model_requests: None, gemini_api_key: str):
    m = GoogleModel('gemini-3-pro-image-preview', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(m, output_type=BinaryImage)

    result = await agent.run('Generate an image of an axolotl.')
    messages = result.all_messages()

    assert result.output == snapshot(IsInstance(BinaryImage))
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=10,
                    output_tokens=1304,
                    input_text_tokens=10,
                    output_image_tokens=1120,
                    details={'thoughts_tokens': 115, 'text_prompt_tokens': 10, 'image_candidates_tokens': 1120},
                    output_reasoning_tokens=115,
                    cost=Decimal('0.136628'),
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    result = await agent.run('Now give it a sombrero.', message_history=messages)
    assert result.output == snapshot(IsInstance(BinaryImage))
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Now give it a sombrero.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=276,
                    output_tokens=1374,
                    input_text_tokens=18,
                    input_image_tokens=258,
                    output_image_tokens=1120,
                    details={
                        'thoughts_tokens': 149,
                        'text_prompt_tokens': 18,
                        'image_prompt_tokens': 258,
                        'image_candidates_tokens': 1120,
                    },
                    output_reasoning_tokens=149,
                    cost=Decimal('0.138000'),
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_google_image_generation_stream(allow_model_requests: None, gemini_api_key: str):
    m = GoogleModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(m, output_type=BinaryImage)

    async with agent.run_stream('Generate an image of an axolotl') as result:
        assert await result.get_output() == snapshot(IsInstance(BinaryImage))

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Generate an image of an axolotl.') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot(IsInstance(BinaryImage))
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='Here you go! '),
                    FilePart(content=IsInstance(BinaryImage)),
                ],
                usage=RequestUsage(
                    input_tokens=10,
                    output_tokens=1295,
                    input_text_tokens=10,
                    output_image_tokens=1290,
                    details={'text_prompt_tokens': 10, 'image_candidates_tokens': 1290},
                    cost=Decimal('0.0387155'),
                ),
                model_name='gemini-2.5-flash-image',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=TextPart(content='Here you go!')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' ')),
            PartEndEvent(index=0, part=TextPart(content='Here you go! '), next_part_kind='file'),
            PartStartEvent(
                index=1,
                part=FilePart(content=IsInstance(BinaryImage)),
                previous_part_kind='text',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
        ]
    )


async def test_google_image_generation_with_text(allow_model_requests: None, gemini_api_key: str):
    m = GoogleModel('gemini-3-pro-image-preview', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(m)

    result = await agent.run('Generate an illustrated two-sentence story about an axolotl.')
    messages = result.all_messages()

    assert result.output == snapshot(
        """\
A little axolotl named Archie lived in a beautiful glass tank, but he always wondered what was beyond the clear walls. One day, he bravely peeked over the edge and discovered a whole new world of sunshine and potted plants.

"""
    )
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an illustrated two-sentence story about an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
A little axolotl named Archie lived in a beautiful glass tank, but he always wondered what was beyond the clear walls. One day, he bravely peeked over the edge and discovered a whole new world of sunshine and potted plants.

""",
                        provider_name='google',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=14,
                    output_tokens=1457,
                    input_text_tokens=14,
                    output_image_tokens=1120,
                    details={'thoughts_tokens': 174, 'text_prompt_tokens': 14, 'image_candidates_tokens': 1120},
                    output_reasoning_tokens=174,
                    cost=Decimal('0.138472'),
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_google_image_or_text_output(allow_model_requests: None, gemini_api_key: str):
    m = GoogleModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key=gemini_api_key))
    # ImageGenerationTool is listed here to indicate just that it doesn't cause any issues, even though it's not necessary with an image model.
    agent = Agent(m, output_type=str | BinaryImage, capabilities=[NativeTool(ImageGenerationTool(size='1K'))])

    result = await agent.run('Tell me a two-sentence story about an axolotl, no image please.')
    assert result.output == snapshot(
        'In a hidden cave, a shy axolotl named Pip spent its days dreaming of the world beyond its murky pond. One evening, a glimmering portal appeared, offering Pip a chance to explore the vibrant, unknown depths of the ocean.'
    )

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(IsInstance(BinaryImage))


async def test_google_image_and_text_output(allow_model_requests: None, gemini_api_key: str):
    m = GoogleModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(m)

    result = await agent.run('Tell me a two-sentence story about an axolotl with an illustration.')
    assert result.output == snapshot(
        'Once, in a hidden cenote, lived an axolotl named Pip who loved to collect shiny pebbles. One day, Pip found a pebble that glowed, illuminating his entire underwater world with a soft, warm light. '
    )
    assert result.response.files == snapshot([IsInstance(BinaryImage)])


async def test_google_image_generation_with_tool_output(allow_model_requests: None, gemini_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = GoogleModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(model=model, output_type=Animal)

    with pytest.raises(UserError, match=re.escape('Tool output is not supported by this model.')):
        await agent.run('Generate an image of an axolotl.')


async def test_google_image_generation_with_native_output(allow_model_requests: None, gemini_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = GoogleModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(model=model, output_type=NativeOutput(Animal))

    with pytest.raises(UserError, match=re.escape('Native structured output is not supported by this model.')):
        await agent.run('Generate an image of an axolotl.')

    model = GoogleModel('gemini-3-pro-image-preview', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(model=model, output_type=NativeOutput(Animal))

    result = await agent.run('Generate an image of an axolotl and then return its details.')
    assert result.output == snapshot(Animal(species='Ambystoma mexicanum', name='Axolotl'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl and then return its details.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=15,
                    output_tokens=1334,
                    input_text_tokens=15,
                    output_image_tokens=1120,
                    details={'thoughts_tokens': 131, 'text_prompt_tokens': 15, 'image_candidates_tokens': 1120},
                    output_reasoning_tokens=131,
                    cost=Decimal('0.136998'),
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "species": "Ambystoma mexicanum",
  "name": "Axolotl"
} \
""",
                        provider_name='google',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=295,
                    output_tokens=222,
                    input_text_tokens=37,
                    input_image_tokens=258,
                    details={'thoughts_tokens': 196, 'text_prompt_tokens': 37, 'image_prompt_tokens': 258},
                    output_reasoning_tokens=196,
                    cost=Decimal('0.003254'),
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_google_image_generation_with_prompted_output(allow_model_requests: None, gemini_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = GoogleModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(model=model, output_type=PromptedOutput(Animal))

    with pytest.raises(UserError, match=re.escape('JSON output is not supported by this model.')):
        await agent.run('Generate an image of an axolotl.')


async def test_google_image_generation_with_tools(allow_model_requests: None, gemini_api_key: str):
    model = GoogleModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(model=model, output_type=BinaryImage)

    @agent.tool_plain
    async def get_animal() -> str:
        return 'axolotl'  # pragma: no cover

    with pytest.raises(UserError, match=re.escape('Tools are not supported by this model.')):
        await agent.run('Generate an image of an animal returned by the get_animal tool.')


async def test_google_image_generation_with_web_search(allow_model_requests: None, gemini_api_key: str):
    model = GoogleModel('gemini-3-pro-image-preview', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(model=model, output_type=BinaryImage, capabilities=[NativeTool(WebSearchTool())])

    result = await agent.run(
        'Visualize the current weather forecast for the next 5 days in Mexico City as a clean, modern weather chart. Add a visual on what I should wear each day'
    )
    assert result.output == snapshot(IsInstance(BinaryImage))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Visualize the current weather forecast for the next 5 days in Mexico City as a clean, modern weather chart. Add a visual on what I should wear each day',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['', 'current 5-day weather forecast for Mexico City and what to wear']},
                        tool_call_id=IsStr(),
                        provider_name='google',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'domain': None,
                                'title': 'accuweather.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQElsvx97FT3Kr__tvs8zIgS3C1znKqEOvuHdjyLe2WZZsJpbDDqn9gdF6rKV8KMZytsiWXCDcNwD5m0WvZzGWY6eVbnz0lxftYNTSNdXTiv1AtLrmw-NUcnITjEScK_JHJgnr9xmFapH9DXMGWWYKRSfcT3iy96J1gZeWjCBph5Sci23DAhzA==',
                            },
                            {
                                'domain': None,
                                'title': 'weather-and-climate.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGlGJX9f12rrKOYrY71rszTFf5KghgToVKZckqRWzT-cjW-mYE_PV3xRbk0JxQxJS18rkCt-y8qwpB41BMYEuxLnkCSBapX5s-4-0pwPUimTjHK4W65OdkVtjTU5-wlHsAppBwdwXNDSmzXZNUYLE1N0R9SKhLeHVVj-2BYYeoO9GPH',
                            },
                            {
                                'domain': None,
                                'title': '',
                                'uri': 'https://www.google.com/search?q=time+in+Mexico+City,+MX',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=33,
                    output_tokens=2309,
                    input_text_tokens=33,
                    output_image_tokens=1120,
                    details={
                        'thoughts_tokens': 529,
                        'text_prompt_tokens': 33,
                        'image_candidates_tokens': 1120,
                        'web_search_requests': 1,
                    },
                    output_reasoning_tokens=529,
                    web_searches=1,
                    cost=Decimal('0.148734'),
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_google_image_generation_tool_aspect_ratio(gemini_api_key: str) -> None:
    model = GoogleModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key=gemini_api_key))
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(aspect_ratio='16:9')])

    tools, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools == []
    assert image_config == {'aspect_ratio': '16:9'}


async def test_google_image_generation_resolution(gemini_api_key: str) -> None:
    """Test that resolution parameter from ImageGenerationTool is added to image_config."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=GoogleProvider(api_key=gemini_api_key))
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(size='2K')])

    tools, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools == []
    assert image_config == {'image_size': '2K'}


async def test_google_image_generation_resolution_with_aspect_ratio(gemini_api_key: str) -> None:
    """Test that resolution and aspect_ratio from ImageGenerationTool work together."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=GoogleProvider(api_key=gemini_api_key))
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(aspect_ratio='16:9', size='4K')])

    tools, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools == []
    assert image_config == {'aspect_ratio': '16:9', 'image_size': '4K'}


async def test_google_image_generation_unsupported_size_raises_error(gemini_api_key: str) -> None:
    """Test that unsupported size values raise an error."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=GoogleProvider(api_key=gemini_api_key))
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(size='1024x1024')])

    with pytest.raises(UserError, match='Google image generation only supports `size` values'):
        model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]


async def test_google_image_generation_auto_size_raises_error(gemini_api_key: str) -> None:
    """Test that 'auto' size raises an error for Google since it doesn't support intelligent size selection."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=GoogleProvider(api_key=gemini_api_key))
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(size='auto')])

    with pytest.raises(UserError, match='Google image generation only supports `size` values'):
        model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]


async def test_google_image_generation_tool_output_format(vertex_client_google_provider: GoogleProvider) -> None:
    """Test that ImageGenerationTool.output_format is mapped to ImageConfigDict.output_mime_type on Vertex AI."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=vertex_client_google_provider)
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(output_format='png')])

    tools, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools == []
    assert image_config == {'output_mime_type': 'image/png'}


async def test_google_image_generation_tool_unsupported_format_raises_error(
    vertex_client_google_provider: GoogleProvider,
) -> None:
    """Test that unsupported output_format values raise an error on Vertex AI."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=vertex_client_google_provider)
    # 'gif' is not supported by Google
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(output_format='gif')])  # pyright: ignore[reportArgumentType]

    with pytest.raises(UserError, match='Google image generation only supports `output_format` values'):
        model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]


async def test_google_image_generation_tool_output_compression(
    vertex_client_google_provider: GoogleProvider,
) -> None:
    """Test that ImageGenerationTool.output_compression is mapped to ImageConfigDict.output_compression_quality on Vertex AI."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=vertex_client_google_provider)

    # Test explicit value
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(output_compression=85)])
    tools, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools == []
    assert image_config == {'output_compression_quality': 85, 'output_mime_type': 'image/jpeg'}

    # Test None (omitted)
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(output_compression=None)])
    tools, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert image_config == {}


async def test_google_image_generation_tool_compression_validation(
    vertex_client_google_provider: GoogleProvider,
) -> None:
    """Test compression validation on Vertex AI: range and JPEG-only."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=vertex_client_google_provider)

    # Invalid range: > 100
    with pytest.raises(UserError, match='`output_compression` must be between 0 and 100'):
        model._get_native_tools(  # pyright: ignore[reportPrivateUsage]
            ModelRequestParameters(native_tools=[ImageGenerationTool(output_compression=101)])
        )

    # Invalid range: < 0
    with pytest.raises(UserError, match='`output_compression` must be between 0 and 100'):
        model._get_native_tools(  # pyright: ignore[reportPrivateUsage]
            ModelRequestParameters(native_tools=[ImageGenerationTool(output_compression=-1)])
        )

    # Non-JPEG format (PNG)
    with pytest.raises(UserError, match='`output_compression` is only supported for JPEG format'):
        model._get_native_tools(  # pyright: ignore[reportPrivateUsage]
            ModelRequestParameters(native_tools=[ImageGenerationTool(output_format='png', output_compression=90)])
        )

    # Non-JPEG format (WebP)
    with pytest.raises(UserError, match='`output_compression` is only supported for JPEG format'):
        model._get_native_tools(  # pyright: ignore[reportPrivateUsage]
            ModelRequestParameters(native_tools=[ImageGenerationTool(output_format='webp', output_compression=90)])
        )


async def test_google_image_generation_tool_all_fields(vertex_client_google_provider: GoogleProvider) -> None:
    """Test that all ImageGenerationTool fields are mapped correctly on Vertex AI."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=vertex_client_google_provider)
    params = ModelRequestParameters(
        native_tools=[ImageGenerationTool(aspect_ratio='16:9', size='2K', output_format='jpeg', output_compression=90)]
    )

    tools, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools == []
    assert image_config == {
        'aspect_ratio': '16:9',
        'image_size': '2K',
        'output_mime_type': 'image/jpeg',
        'output_compression_quality': 90,
    }


async def test_google_image_generation_silently_ignored_by_gemini_api(gemini_api_key: str) -> None:
    """Test that output_format and compression are silently ignored by the Gemini API (google)."""
    model = GoogleModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key=gemini_api_key))

    # Test output_format ignored
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(output_format='png')])
    _, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert image_config == {}

    # Test output_compression ignored
    params = ModelRequestParameters(native_tools=[ImageGenerationTool(output_compression=90)])
    _, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert image_config == {}

    # Test both ignored when None
    params = ModelRequestParameters(native_tools=[ImageGenerationTool()])
    _, image_config = model._get_native_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert image_config == {}


async def test_google_vertexai_image_generation_with_output_format(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    """Test that output_format works with Vertex AI."""
    model = GoogleModel('gemini-2.5-flash-image', provider=vertex_provider)
    agent = Agent(
        model,
        capabilities=[NativeTool(ImageGenerationTool(output_format='jpeg', output_compression=85))],
        output_type=BinaryImage,
    )

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output.media_type == 'image/jpeg'


async def test_google_vertexai_image_generation(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    model = GoogleModel('gemini-2.5-flash-image', provider=vertex_provider)

    agent = Agent(model, output_type=BinaryImage)

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(IsInstance(BinaryImage))
