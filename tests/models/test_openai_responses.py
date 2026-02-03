import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import (
    BinaryContent,
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    FinalResultEvent,
    ImageGenerationTool,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UnexpectedModelBehavior,
    UserError,
    UserPromptPart,
    capture_run_messages,
)
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool, FileSearchTool, ImageAspectRatio, MCPServerTool, WebSearchTool
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, RunUsage

from ..conftest import IsDatetime, IsFloat, IsInstance, IsInt, IsNow, IsStr, TestEnv, try_import
from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types.responses import ResponseFunctionWebSearch
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText
    from openai.types.responses.response_reasoning_item import (
        Content as ReasoningContent,
        ResponseReasoningItem,
        Summary,
    )
    from openai.types.responses.response_usage import ResponseUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
        _resolve_openai_image_generation_size,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]


async def _cleanup_openai_resources(file: Any, vector_store: Any, async_client: Any) -> None:  # pragma: lax no cover
    """Helper function to clean up OpenAI file search resources if they exist."""
    if file is not None:
        await async_client.files.delete(file.id)
    if vector_store is not None:
        await async_client.vector_stores.delete(vector_store.id)
    await async_client.close()


def test_openai_responses_model(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    model = OpenAIResponsesModel('gpt-4o')
    assert model.model_name == 'gpt-4o'
    assert model.system == 'openai'
    assert model.base_url == 'https://api.openai.com/v1/'
    assert model.client.api_key == 'test'


async def test_openai_responses_model_simple_response(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_openai_responses_image_detail_vendor_metadata(allow_model_requests: None):
    c = response_message(
        [
            ResponseOutputMessage(
                id='output-1',
                content=cast(list[Content], [ResponseOutputText(text='done', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model=model)

    image_url = ImageUrl('https://example.com/image.png', vendor_metadata={'detail': 'high'})
    binary_image = BinaryContent(b'\x89PNG', media_type='image/png', vendor_metadata={'detail': 'high'})

    result = await agent.run(['Describe these inputs.', image_url, binary_image])
    assert result.output == 'done'

    response_kwargs = get_mock_responses_kwargs(mock_client)
    image_parts = [
        item
        for message in response_kwargs[0]['input']
        if message.get('role') == 'user'
        for item in message['content']
        if item['type'] == 'input_image'
    ]
    assert image_parts
    assert all(part['detail'] == 'high' for part in image_parts)


@pytest.mark.parametrize(
    ('aspect_ratio', 'explicit_size', 'expected_size'),
    [
        ('1:1', 'auto', '1024x1024'),
        ('2:3', '1024x1536', '1024x1536'),
        ('3:2', 'auto', '1536x1024'),
    ],
)
def test_openai_responses_image_generation_tool_aspect_ratio_mapping(
    aspect_ratio: ImageAspectRatio,
    explicit_size: Literal['1024x1024', '1024x1536', '1536x1024', 'auto'],
    expected_size: Literal['1024x1024', '1024x1536', '1536x1024'],
) -> None:
    tool = ImageGenerationTool(aspect_ratio=aspect_ratio, size=explicit_size)
    assert _resolve_openai_image_generation_size(tool) == expected_size


def test_openai_responses_image_generation_tool_aspect_ratio_invalid() -> None:
    tool = ImageGenerationTool(aspect_ratio='16:9')

    with pytest.raises(UserError, match='OpenAI image generation only supports `aspect_ratio` values'):
        _resolve_openai_image_generation_size(tool)


def test_openai_responses_image_generation_tool_aspect_ratio_conflicts_with_size() -> None:
    tool = ImageGenerationTool(aspect_ratio='1:1', size='1536x1024')

    with pytest.raises(UserError, match='cannot combine `aspect_ratio` with a conflicting `size`'):
        _resolve_openai_image_generation_size(tool)


def test_openai_responses_image_generation_tool_unsupported_size_raises_error() -> None:
    tool = ImageGenerationTool(size='2K')
    with pytest.raises(UserError, match='OpenAI image generation only supports `size` values'):
        _resolve_openai_image_generation_size(tool)


async def test_openai_responses_model_simple_response_with_tool_call(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    agent = Agent(model=model)

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        return 'Potato City'

    result = await agent.run('What is the capital of PotatoLand?')
    assert result.output == snapshot('The capital of PotatoLand is Potato City.')


async def test_openai_responses_output_type(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class MyOutput(TypedDict):
        name: str
        age: int

    agent = Agent(model=model, output_type=MyOutput)
    result = await agent.run('Give me the name and age of Brazil, Argentina, and Chile.')
    assert result.output == snapshot({'name': 'Brazil', 'age': 522})


async def test_openai_responses_reasoning_effort(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='low'))
    result = await agent.run(
        'Explain me how to cook uruguayan alfajor. Do not send whitespaces at the end of the lines.'
    )
    assert [line.strip() for line in result.output.splitlines()] == snapshot(
        [
            'Alfajores uruguayos are delicious sandwich cookies filled with dulce de leche and rolled in coconut or covered with chocolate. Follow these steps to prepare them:',
            '',
            'Ingredients:',
            '• 300 g cornstarch',
            '• 200 g all‐purpose flour',
            '• 100 g unsalted butter, cold and diced',
            '• 100 g powdered sugar',
            '• 2 egg yolks',
            '• Zest of 1 lemon',
            '• 1 teaspoon vanilla extract',
            '• 1 pinch of salt',
            '• Dulce de leche for filling',
            '• Optional: desiccated coconut or melted chocolate for coating',
            '',
            'Instructions:',
            '1. In a large bowl, sift together the cornstarch, flour, powdered sugar, and salt to avoid lumps.',
            '2. Add the cold diced butter, lemon zest, and vanilla extract. Work the ingredients with your fingertips or a pastry cutter until the mixture resembles coarse crumbs.',
            '3. Add the egg yolks and gently mix until the dough comes together. Avoid overworking the dough for a tender texture.',
            '4. Shape the dough into a disc, wrap it in plastic wrap, and refrigerate for at least 1 hour to relax the gluten and firm the dough.',
            '5. Preheat your oven to 180°C (350°F). Line baking sheets with parchment paper.',
            '6. On a lightly floured surface, roll out the dough to about 0.5 cm thickness. Use a round cookie cutter (approximately 5 cm in diameter) to cut out circles.',
            '7. Place the cookies on the prepared baking sheets and bake for 8-10 minutes until the edges are just set. The cookies should remain pale.',
            '8. Remove the cookies from the oven and allow them to cool on a rack completely.',
            '9. Once cooled, spread a generous layer of dulce de leche on the flat side of one cookie and sandwich it with another. Repeat with the rest.',
            '10. For finishing touches, optionally roll the alfajores in desiccated coconut or gently dip them in melted chocolate. Place them on a rack to let the coating set.',
            '',
            'Enjoy your homemade alfajores with a cup of mate or tea!',
            '',
            'Note: Clean up all utensils and surfaces promptly and store any leftovers in an airtight container to preserve freshness.',
        ]
    )


async def test_openai_responses_reasoning_generate_summary(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('computer-use-preview', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        model=model,
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_summary='concise',
            openai_truncation='auto',
        ),
    )
    result = await agent.run('What should I do to cross the street?')
    assert result.output == snapshot("""\
To cross the street safely, follow these steps:

1. **Use a Crosswalk**: Always use a designated crosswalk or pedestrian crossing whenever available.
2. **Press the Button**: If there is a pedestrian signal button, press it and wait for the signal.
3. **Look Both Ways**: Look left, right, and left again before stepping off the curb.
4. **Wait for the Signal**: Cross only when the pedestrian signal indicates it is safe to do so or when there is a clear gap in traffic.
5. **Stay Alert**: Be mindful of turning vehicles and stay attentive while crossing.
6. **Walk, Don't Run**: Walk across the street; running can increase the risk of falling or not noticing an oncoming vehicle.

Always follow local traffic rules and be cautious, even when crossing at a crosswalk. Safety is the priority.\
""")


async def test_openai_responses_system_prompt(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, system_prompt='You are a helpful assistant.')
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_openai_responses_model_retry(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, I only know about "London".')

    result = await agent.run('What is the location of Londos and London?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the location of Londos and London?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name":"Londos"}',
                        tool_call_id=IsStr(),
                        id='fc_0a4779586464cbd900697ccca057908195ae0f71a641babecf',
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name":"London"}',
                        tool_call_id=IsStr(),
                        id='fc_0a4779586464cbd900697ccca05de88195ae0972ae4dd26603',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=21, output_tokens=48, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0a4779586464cbd900697ccc9f35388195b56656b6141ff1cf',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, I only know about "London".',
                        tool_name='get_location',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
I was unable to find a location named "Londos," but here's the information for London:

- **London**: Located at latitude 51 and longitude 0.\
""",
                        id='msg_0d7ae142b967b8f500697ccca13aa88197b285c8cf2e5b3d9f',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=115, output_tokens=37, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0d7ae142b967b8f500697ccca0b9948197bc26a3697805a4aa',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, image_content: BinaryContent, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool?'])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What fruit is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_image',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_052d4bce83a89de800697ccc9d8968819e9cdbcd70b5ac82ba',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=40, output_tokens=11, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_052d4bce83a89de800697ccc9c9d6c819eb584867c766fc9bb',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 241a70',
                        tool_call_id='call_steJXNgwhlXh2G0FkkTPyGyF',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content=['This is file 241a70:', image_content], timestamp=IsDatetime()),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The fruit in the image is a kiwi.',
                        id='msg_05920e5ace9f274500697ccc9f448c8196aeacf9cdac7ca867',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=839, output_tokens=11, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_05920e5ace9f274500697ccc9de1dc8196adfe7c5e228888fd',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot('The fruit in the image is a kiwi.')


async def test_openai_responses_audio_as_binary_content_input(
    allow_model_requests: None, audio_content: BinaryContent, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    with pytest.raises(NotImplementedError):
        await agent.run(['Whose name is mentioned in the audio?', audio_content])


async def test_openai_responses_document_as_binary_content_input(
    allow_model_requests: None, document_content: BinaryContent, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['What is in the document?', document_content])
    assert result.output == snapshot(
        "It sounds like you're referring to a file named \"Dummy PDF file.\" Without specific content details, it's hard to say what's inside. Typically, dummy PDFs are used as placeholders and might contain sample text, random characters, or lorem ipsum to simulate a document's layout or design. If you need help creating or analyzing the content of a dummy PDF, let me know!"
    )


async def test_openai_responses_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'It sounds like you\'re referring to a placeholder document often used for testing purposes. These "dummy" PDFs typically contain random text (like "Lorem Ipsum") or repeated characters, and they help in testing layout, formatting, and file handling. If you have specific details or need further assistance, feel free to provide more context!'
    )


async def test_openai_responses_text_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    with pytest.raises(ModelHTTPError, match=r'Please try again with a pdf'):
        await agent.run(['What is the main content on this document?', text_document_url])


async def test_openai_responses_image_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot("Hello! That's a nice potato. How can I assist you today?")


async def test_openai_responses_stream(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        return 'Paris'

    output_text: list[str] = []
    async with agent.run_stream('What is the capital of France?') as result:
        async for output in result.stream_text():
            output_text.append(output)
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[
                            TextPart(
                                content='The capital of France is Paris.',
                                id='msg_0565aca8c997b45400697ccca3fb9c8197a61c6abf40e44aca',
                                provider_name='openai',
                            )
                        ],
                        usage=RequestUsage(input_tokens=62, output_tokens=9, details={'reasoning_tokens': 0}),
                        model_name='gpt-4o-2024-08-06',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                        provider_url='https://api.openai.com/v1/',
                        provider_details={
                            'finish_reason': 'completed',
                            'timestamp': IsDatetime(),
                        },
                        provider_response_id='resp_0565aca8c997b45400697ccca39bb08197ba2c6c7e7808a63a',
                        finish_reason='stop',
                    )
                )

    assert output_text == snapshot(['The capital of France is Paris.'])


async def test_openai_include_raw_annotations_streaming(allow_model_requests: None, openai_api_key: str):
    prompt = 'What is the tallest mountain in Alberta? Provide one sentence with a citation.'
    instructions = 'Use web search and include citations in your answer.'

    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, instructions=instructions, builtin_tools=[WebSearchTool()])

    settings = OpenAIResponsesModelSettings(openai_include_raw_annotations=True)

    events = [event async for event in agent.run_stream_events(prompt, model_settings=settings)]
    annotation_event = next(
        event
        for event in events
        if isinstance(event, PartDeltaEvent)
        and isinstance(event.delta, TextPartDelta)
        and event.delta.provider_details
        and 'annotations' in event.delta.provider_details
    )
    assert annotation_event.delta.provider_details == snapshot(
        {
            'annotations': [
                {
                    'type': 'url_citation',
                    'start_index': 77,
                    'end_index': 162,
                    'title': 'Mount Columbia | mountain, Alberta, Canada | Britannica',
                    'url': 'https://www.britannica.com/place/Mount-Columbia?utm_source=openai',
                }
            ]
        }
    )

    model2 = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key=openai_api_key))
    agent2 = Agent(model2, instructions=instructions, builtin_tools=[WebSearchTool()])
    events2 = [event async for event in agent2.run_stream_events(prompt)]
    assert not any(
        (
            isinstance(event, PartDeltaEvent)
            and isinstance(event.delta, TextPartDelta)
            and event.delta.provider_details
            and 'annotations' in event.delta.provider_details
        )
        or (
            isinstance(event, PartEndEvent)
            and isinstance(event.part, TextPart)
            and event.part.provider_details
            and 'annotations' in event.part.provider_details
        )
        for event in events2
    )

    model3 = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key=openai_api_key))
    agent3 = Agent(model3, instructions='Answer directly.')
    settings3 = OpenAIResponsesModelSettings(openai_include_raw_annotations=True)
    events3 = [event async for event in agent3.run_stream_events('What is 2+2?', model_settings=settings3)]
    assert not any(
        (
            isinstance(event, PartDeltaEvent)
            and isinstance(event.delta, TextPartDelta)
            and event.delta.provider_details
            and 'annotations' in event.delta.provider_details
        )
        or (
            isinstance(event, PartEndEvent)
            and isinstance(event.part, TextPart)
            and event.part.provider_details
            and 'annotations' in event.part.provider_details
        )
        for event in events3
    )


async def test_openai_responses_model_http_error(allow_model_requests: None, openai_api_key: str):
    """Set temperature to -1 to trigger an error, given only values between 0 and 1 are allowed."""
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(temperature=-1))

    with pytest.raises(ModelHTTPError):
        async with agent.run_stream('What is the capital of France?'):
            ...  # pragma: lax no cover


async def test_openai_responses_model_builtin_tools_web_search(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_builtin_tools=[{'type': 'web_search'}])
    agent = Agent(model=model, model_settings=settings)
    result = await agent.run('Give me the top 3 news in the world today')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Give me the top 3 news in the world today',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aaa4efb081958605d7b31e838366',
                        signature='gAAAAABoxKrgd0uCWxLjgCiIWj3ei9eYp9sdRdHLVNWOpZvOS6TS_8hF6IEgz5acjqUiaGnXfLl3kn78UERavEItdZ-6PupaB2V7M8btQ2v76ZJCPXR5DGvXe3K2y_zrSLC-qbX4ui3hPfGG01qGiftAM7m04zuCdJ33SVDyOasB8uzV7vSqFzM4CkcAeN0jueQtuGDJ9U5Qq9blCXo6Vxx4BVOVPYnCONMQvwJXlbZ7i_s3VmUFFDf2GlNYtkT07Z1Uc5ESVUVDYfVC2qlOWWp2MLh20tbsUMqHPYzO0R7Y1lmwAqNxaT4HIhhlQ0xVer1qBRgUfLn1fGXX0vBb4rN0N_w7c2w-iwY-4XAvhAr-Y3pejueHfepmv76G67cJVQjzgM37wlQFdl_UmDfkVDIxmAE62QjOjPs8TweVPEXUXAK4itTDQiS7M42dS6QzxivPVvzoMkNOjJ58vUy83DCr-Obw8SMfFGB5sd1hGg9enLYiGxN_Qzs9IGegBU4cH1wpCvARmuVP10-CJe0jzSFy0OI76JUgGMVido_cEgrAF5eEOS-3vkel6L07Q9Sl_f8C-ZW04zF40ZIvCZ4RJfRAKr2bfXH6IVNhu528-ilQTCoCeFy_CG6UYlUY2jws_DRuTsAVb6691hPRI8mG28NCPXNGV5h8sVgypbeqWyBNZEnSgqFcNVplAPTxDNqlcFps5bEND4Q0SLSNTZv9vFbRvfyrf-4s3UWqn-SI4QAmGzKRRuTumEpldsTuZgv69Nu2qA7px1ZNu-hN7S0E7ONGDs2fCaUG4X-Xp3j2fizfaTkZpOC_sdTK5e10lIG019zKGngXSrBy_sOWyTIsjiRGdr0Va-RjDw2ruFr3ewQcH5vZ8LgUwTzijfqLqbkF1zgZopHTnz1Gpt42AbZiyP30S9BQuDODD8RmtZQ5oB1NKmISeGkLCJRd6dZKGibFskFFMFr53YvUfVZx4mRpxSjuadceNKPhTVkbGPYE6XrZbChCxDL9aJJ37ctRxf91r9QAXMqeFZR-4HR13_Pp0AyN_H7gqBR2yVuGbXkhs1QwkEhl-6_keNsJYUaRSSf5QN9gRjsuWchWEsTr8AqTbIApGO24a5Rr4GDnZ_6ICYBr-IhUesv0VJKQF3DcNFaOQCLtLTKCC4G4SqURt60V0zkQKWBdUdUGFkxDUN5gtcKrR0F4J5hvZ6OMV3XaP6kpgx62TL_gd9g_QyV8QDFwXuDDrGyXi6l68veZXOElkZ4lpVAjfeXnysK401DRt3vF0z99wUc-QVMjZG0wVZUr5rYHjKKaB2vG85n_onMrddThz2_a1NG_THQZ3L1rprThcQY7FdPtw1JXWfXWeS7ZuOOZCZvjyCrVhevaxTl5UKNbkguqYhNJQfx5X8IkwJWVRObA3QxFD0ZEgW9OKt-v-g_EAsjtftPbeeqaDfPBwqVguYJUEZqPPwcsG2cv8Xu5sCc6h7J8fvwTK-MY847JS5Q5CSDe4GDFvJn4Tk4aIOeGlr-VlrgwOS_yaKd1GogBIDzjh8pXIXXSDP2UkEOd2T0zSoa0u8oewPf8Pwmd7pmVb10Y9tHPgEo44ZQRiyVCe9S36BVjf1iZgTYetfBfq9JJom1Ksz-WUf74sHYfLkUY96lOlSvziyFFmTXxFgssLFgtBuWNaehKeuJ0QiQm2r4jEvX3n7dvUj09tWw_boLWGUJqL5YkxVadlw8wF1KRFJjGIAvEvO7YNoEoyolmS9616ZBvWNlBg54A5DITXEfIMloXVYNmYomoBloM74USiV7AjQE5hPIIqO97dW4btd2zMx9Nbr8G-nZsLgCqrqzDVz0UorAHTgaThtp9BW6VJZJ9q3Ew_z_494P7GNv9ehuK6m3fT-MXIq-t0Bo28YGgGhiFjoYSSYUd1adlHQdPHZCxZojt4-DxgD3iFoWQGc7BBRU3f9rRVRzbDvlHpaLRUQUFXiaB6rQ=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'top world news September 12, 2025 Reuters', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aaab56508195a1effa9583720d20',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aaab56508195a1effa9583720d20',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aaaef4b481959dfd7d8031662152',
                        signature='gAAAAABoxKrgnD1AQrb0I2TSTLa6DiSpAwMbkF6gxz_b8tkkns4MZ4Rr6a8ejwmX7aGMoXEgOO2nkuLFeKoQBzQBrfNZIhmCy68QZMQQKZfKBUv1k8OAKzz4A1dO-xNH6xLMS-3cG4ev4zqjQEOBSGoZNKcZMU9L3B0VCvZsBU7S50g7zCcVwEk6H0wx4HO6IuUEOzgqqx8NYHmOkudSv3ikiHn1xhLc1JEzXkupTyRxyw1O81jJEpNzLlEUIFeu0vkAJrlwQzAHeEzxFMMQMoru3pKwnzujgljefGG8RY34jsAc6XcbJSstAa5GnKn24ehA_CQu80ICcibs7LBKsa3oO8wWWHXgDhMCPJn0N322MZcHfH77PhgEr-T1YSIRrSMPXcxoPaptN0O4ceK9BYN4FDRddaR1jXzWdZ3VhYBNbRrQEuO6z0TOWsPmzIlDql1a20jiOteGNQgIX94Af4PB5g_DYWzJW8YVffnhKXJEmU7BmYuctQgyewLj_CoQYfQ9HtGcae6ZElUEP96lo1ID3AW2iMa3iP4C2xULWDVh-8rWf0D2fgS1toexXXCtWbXn8XlYMGWVjq3WX5q16Kq0KyInuCZleABTeFRuzh0MTx1GaYhDTwHxG8BRPYUxz0bHHESz-h_UGmhGu8-a49YdBpLe36_Z1wprXJ82Yg7KvJy68VwKnLeH1Zm56aMHviJl143iZYgiZaVmRBIRExMvnI9LVAT5pv0Y3CdCCSq8Bs2jSbhU0xe26HAqfZZnAsE0LpPAfW1tMCiKzqhtzoKR6yauAYCXP5YtnX6BqFr-J8px6owPJhepjyrSVCObyya7v7_rV81BkYOtLQSwCUUhOjbawgI6XDQ_FK0hye5lFVKckFNM3cVpgRcZymeqx-XoQeoFOR8uLtcXv2DIoo0TfP7RxgBvAvdohv8vZx7xJSXlrYKqLEK1ASQDcc36gIfNQuNXM24WuXForXTO2l_sTeos58eX5FGxWJFDghhrNa_ia1dL7towjcegQzf9LtLjLlnqUGpEte-o23DKKQQEiFfMpLlvGu2cOVwYUuoeOpEBe7QpDbJGdBjq0hOKdakHGl6KwBw6vCkRp_wtW4R7QBuncdYyRT6AJ1_Z_byBP7kH1A2-P6QMVycBVcXlUgc0BzuGlkt51l__O3CM4z-PmI8zR5cL6ZCXoQzG2Yp-OhQ-n-3hgMaCfBGca6J3wP1vgQpR2AF0',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'Nepal protests September 12 2025 Reuters', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aab0c534819593df0190332e7aa3',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aab0c534819593df0190332e7aa3',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aab3df148195bdec4fe186f23877',
                        signature='gAAAAABoxKrgZN3V5pGaqoIM8EEiFso41O_kxOTWpzAh5Nlj3pqqDjIGrFH2zcmDyURpUmXdExY8L9K6KcwGOAlF6okEgQojeTxysBi4-gDVFNCVfp6c6K4tAtCBrvq5wC2g22Ny1pU2OMyxVU2GCxIIehCZiPQio_7IS8WY_VWkwLOag7bT4FBGn-aVFyoEfDDpIPF-4Zpcal6bAvdjD2hYGl6_-8alwh36ttUkJroo2qG-Mn0LsAWJ7YEzfrHgoPTDF7TB3Mfvvc5M_eP3pzY8O4WhZKMLBSnM92iIt5J3nSJYhRoiwEjaCamIM4vK0cnJR0oX87u_XtGvnNBX93ttrIrXDKK-mh-LIoe_sK1dViFINxk6rJHZvkFK12J6UXMK4me-C3uQ_qGygpw4uYvWhYk7LDR9Zgxfv1OoDg13DCYWWrHX7Oa1ALXPotk1Uw_Tof-Wc_wDqE16Elm1a5TP-ISH45v9W_Xl1IXo7J_jwOlAjkXvrh2a8YNljWQqBFCca-M2hSWvKuX8JuNF_tkI2q2E7jIDNt77jGd2yavqb1W2WoB_s7jqyAWomT91E2gZQtGJa4X2ydeTPQ_oWv2hgdTUynV0nbOKWA6suZixvxVDLLedhYHRnKY6EOtyso9MZav1qhr_DpHExn1_woquJXtS7c3Fe3Rs_YrU6PpRx5_DEVjVKme-3XjLJNclx6NF-rbXYqhXXExqPk-od7n-YMyrYhpfVP8lmLCewwyzVRb1koOEcCqnuhqM9DWyazKAcdvejM7VEM1AEk8ugT02cTiF7CfLefYFsLSYVBM0Ox47Ceh4BOA82jdlf1pZNvGqgHi8kKm9HLVh-yM_DAhD8O5Ub-SCd3bNi8735XPDWVIm6sKMdg1bcgVehz_R4iEBr_pguKfZUJLcckUTI6fitAQ6YSLpLAfRA0nMDBfM6p43jqsSCP8Ovjx58TwAPElgpme4ENBCozS_VaxmqawpfUfvnD60xia57wtSBYr5s1j-FUUjBsFTInjHdKcp0EBd3Pv-mpVE-Yj0MYExbn1upi3RxWN6jwVeYc603HQBjsjqsb-op9Tb0GZxf5Z4DpZ_eeb4IBTWNf3FTLIbsVg18Oyl128Std9CkMGak8iI_dFCvm1ZQQ6u3CyLEwxGsMZnkZl6OhSKDlnHDvRsF0F0OcRtFV5i7j92kMs9_qJ2JLdb5LzdqOBnFfKOcUCXBOflL58PYIav',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={
                            'query': 'UN Security Council condemns attack in Doha September 12 2025 Reuters',
                            'type': 'search',
                        },
                        tool_call_id='ws_0e3d55e9502941380068c4aab597f48195ac7021b00e057308',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aab597f48195ac7021b00e057308',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aab85e7c81958c4b4ae526792e49',
                        signature='gAAAAABoxKrghWofTkZCzljg86Akl6ch3bNwR70Wz37t6mBpLah1wZ-7U6isPixPCn0t66fA6xKxGX75bmjRu8Gts4cIaYpm78c8n6R44UULYfQoDC9ZEyGgImbQUKoHFU63nSbsjuPTTFtLdLHhEebDE_t6AfqIWBlyZKRqYlXS_8mTZ_NwM5_JgJun1Xz-I3Pb0X5ZgX8RTP_Kh7Kk79PStvg0-qcVoxMtFsK4ZN3fzQBOSUwvkMhglIweiS3s9CpTbtOs0PYqFCOIjKEYZ2-Rt_7SKhOGaWEMuvuWggMLeO_Wkl8HyIHre5JolVFR9M-43XByZXQxrvBxFzzwHubiyCs-WHFicgMyZcAF8e2KR9KdUJxAwQ3acCi3zBc7e5q1jgc8-Csm-vZQJMTyABDu4yuLena6rF777C8jq-naUe5M-bBpiimK1nbpg5YDiwx7-TbZz5eiTpptHL3P6izhgEOXuEvLhlrhxPBKTezDkiwu-wjs0tHguRYbOIMf-3NZGHuYnOcGfC2wJKkE9DmRvbicnChrLqzHmiXWblYhPwsH9wt-QDvrz3tgCH4B3ri9APreQjBmxtZEVGQAtfdm1qpgiDcWqEijrj05rvr4HxbSReCFszZJDYAufNhJSPhuJXl4e7EHRLyVd2uJA264ONj-MxT2WRr4MGzubSXtPd1QJn7IEkCCuPZxbLf9q27DTSpAvS1oZVs1Ad0J4lbRV5tS_sG54JLvpXf4jtYHD-R2CG0vkL1i0273IJroXScLaPELp0iJMn-WzAkbEjjMsX8gmZlV2X06XuvSjry-dh2sU9Yldqw4NHMLM8rpZIfKbsm6w0ub5Icmu19E856R57JM3K3Pjm3fdO3HR-adVsJTAaIusyUVX3SOiTY53-X6UbqBJh5H3WOORqkwW2nGbNur6B_tyRjlegD3CGJzC-A9rNxMWrecALmCEJBwnXxOuvpsGkSgjP8vjnY9JJNj53hxAirHFIxknDMrKt5qlsRHxGlCdN9H7YuTGdTSgPWH_L9C4BtZrr2Qk41osiDCpacMwBeUDwo1YwYWd1SO0DEzm2qGlXSYeuAQ6Fvyc7sZHCkOsl-bINhCuY1aEBOLzXS7kcu0YAIuEZGVp5wUrr2L6YssdrzpzQ_KENFI7LiB2v5CrF1wZN85H2dkwaGciOXznAa0Su1fWD3BUdpyR0h_mVIcHUxmeoCywWbWO-Do3LFu70MMxKmfSzVfL9hlU2B2jo1aqJ5HesWsWbsbslW5FfREayeUzK7hxkrjliDePhN6gkfy0HOYQijPN6dko4TNEeKFO6Q-aw7c4X5IF3WBCYd_IszlLBK-vTX4EX2J5QtaLRfwFgRwz_K2fkOTT64eknQ6R3fFJpgeyLBZ5ut7j2o7xhEuHeE4KPm2T_AJi8yRScMU-ZsDcUZ8IVYAduy2TGov51AM7K2WojgvqWi62AwSLd16eEnd7SUD8fiCwtRN3zTdmh3MenUogxtKG2YL4hUvSN6Ia1STXpfU4ToLvBnPS5FoY2GuOG-EdEAHdKfYsSUZmSauAlQy7sT43STLkDE42lOKWqtSNHOygkGUodv1GNR0sA6CIg_gVAOyUG-o20rMsfANynNokpoKxJBPJScf1Mbivm-7wJFRipf2-Ay4HzXhXZ4RTkpoq2MMC7cZkHkEprUlLshEhCIHF_6sb1Uhqg4E3UPCCNZ-X0epbQ2GmhtaaIt6BCnWz4SccN5qTks5XpQarlyTW1HubLoLjjXmwJ5DImdUGZkitiJw6ermiOFAFhLfhug-XVKBcTBZOG_CHjrR_2j5TPn6FNLHbYpLYS5hkrUWCJy4U_1xebGl3F6VdQDy3LHZehxuKPowPtdYFenqdJ-naK_A2ygjDUdGBoB2-QFaq8ZPTAti5_Ca6LgiZPvzZdGZ712BED-Opges0mwyAhhsgKRvjjztcsiZ21QpfUaSGLS0vO7J-NcRVvCDyBisMRKfRcWk0PFa4LKcqx9_FNU9nqXH1RXYh_WNAJRVLJDR3WzpNzDv7xMcPOYUUx0wuAYAWcGbc3i5mkVRlzRW_WymBibPF_Y9Yf5yt7plmai5dzlg6aoRdrzSwT9Lphrf79QI3LfYzOV4sXmRGEnN1ud0FyfVB4aLHSsc59_eiPswLL-xg8XT0L27IU_Gja0VuE3zBlErtlQB4uPq778Ojs8hucNTD0rjxs2qqA==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={
                            'query': 'Israel airstrikes Yemen Sanaa September 10 2025 Reuters death toll',
                            'type': 'search',
                        },
                        tool_call_id='ws_0e3d55e9502941380068c4aabe83a88195b7a5ec62ec10a26e',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aabe83a88195b7a5ec62ec10a26e',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aac1c2e081959628b721b679ddfb',
                        signature='gAAAAABoxKrgINXOfVTRxYfQpc8ZZGXsBHdv43DhHkpUjfExhAS41ACM9vHyRgDNfC9E62QVMWRCWPuz5QX9ks0NtD76PYS8n5bessBYeBtYgbiMtl0piW1gE5dlw-BeKLiijMhIVwytWhF3JTzoxoA60FjPK_sA8mFk6wDCNKDXlaLWsLaECxUwCtdktN9SQnQFgxKNemRKQTyRTNKsurCZSSt0tHyd4lxO0Ei3F2mO3WB4Oq28BeVG7RKlcZ9BmLRdBhFQX5eoLxTBHwC_qgSIGzoVCiyClW1OzFzXzmaCUCm3oUDQjooYIZtQqK1b8FBArzN9seOJ4vuxu2qqdtF-JC1vAi-_9J61EwELhN5gYvld83zGCSPg_asjeKeoA6qnA5RFtYwh5kmMSFo9VzGp9MlCmb4_-L-iux3JKc7Kz-jvF1sXSH7YfKgBvcn8HcOdXGjU-aBJTmdP3hCZSL9ko-NNsUO31667QwMZsQTlVoTCAfWS_xDEI0QgmV2kFReKhKanzMmOToUECPPQHQfofCGxwxjbGllSyhpSZHIdyjXpHBmwFALBflPAfeM8wUbqQbNyWbWTdx4Uz62Z4j0OGfcMpgMlDb6BON8vvpIjmlV-fOqRlzkP97klPBygPKeRyT-UezEN5Vj5t00nmB-cV2kNj1WYmL8-eBuJPs3LOU_4Q3ysb90AxYxRJGOsl74lEBqfUKb6b4JWff9JFv11EVJ-puIpE7MA3DPM4NcgGfDZYyDvLS589wbTVxSngBqEOIOEcAZF5Tae93Drajy_x8fXm9uWc8daMf5kqUeq_vwr-ZqEz5ZBUvhvGPL7xkYfTfn-RrQXBx2JfyDRakf4X4D1W6jaO_LXfExH922e9hQ1vH8VA_GPdOIqL5BTiIeO3qFjDSRxMi94XWPPRm87yStxEjx8bse00Bzi3grZ1c6M5dEUXNaHrnvEdJZECT6lz365_Qbl73_Ma_2CLYZhLhtqZRZ6Tycfpprg7rWxqTftOKq4twUgCzzv7kg0e1f_JM_om5loPP6r4MOeAL9O1p49tWmj1kQt_nmYcX1WFTQOgRuB_h3t6ZeOsDb3-VYjIjK0pvj_X_VArrT2suBVitTBXumnG2dXg_z2k5t4KTbWVe-aaGhije0VNxgPWCcu1RlIxOaz',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'typhoon September 12 2025', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac378308195aca61a302c5ebae6',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac378308195aca61a302c5ebae6',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aac5d2bc8195b79e407fee2673cc',
                        signature='gAAAAABoxKrgkfoWE9D7tW3LtG9Hb8kBR9vHgjhSKvDrW0_FUU34LIByJhhBiwZOr5RfbqX9mBwahQKrIVAev3WGtBfgtJF0kP67CIXXRjA1-RHuY-4QXL_w-t9gttak5Dje2NU5hNyp-LyW0plO7DZwZDkFUgeW5plxMzcAFNTdflBSC_-zYqBFl9p-11YKOslzKYxkfQrDiodarFGFhDOJr97qwo-l4BhSg8jywQwgFOTSrOjJMlZRSrTkHd8CUaSF5rUaLKpY4AZWtpiR71otchA9N-d0AaVwnnzJbe53PXJpe4fGUkmkcZt-ZOcNTQlIpifirDsXln2Sc3jxSM05fteSPKoUeUFIIqbCaZwBPau45DKq54PvkVQ4Fpv8JtfqKEuQtJ6EVlNJALuDlskdxM2H3Z7XJsXkcNCVAKmpA80yYwh3eApMr_cERl2bLS9jJpGt8QN3z1yRe5oCPCNWj2_NTgtzjknxcFy8HdT-pcTzLDOhLJPYyl66psc0Hn8V_GFIFkRBa8tWb7CTLt77a3pW3Ifnxov5ANAaaJLM9gGiH_DgkkuNZMR3dz2sVnHzAG5TxmSQteu-uYQgIYanBH_D2BN24JfBFxckpT0z-kGHbJnL5q_wBeyy7o2puohaH3MNIluzWARcDWaFa1tGkzeZg59woqrrddAdWLRNULpnX9fzr7aAWXr1U5-XkSjyfWa4nmIFtchwPSC-12wHRNFDzdZiUvQDdJ2ENGoIXeYpob_O4Wa5zx4zZj_qHXoQWXLELyEMJZCVADjAjO8uy2gXDxZKcUxyDgi17hIyFtC9Z_4rxDbV_S_JJ68s1qHBZljuH0mrkLU0KXmYi5ZgB_z1CEaz9KkL32FGBt0YXuFoR0LjnrdpOTa9ifWC82ZhDfjz1E4y9FUoGPVl-QYQ5ihDY0LswB1x_FJfvwRLvLRtMeeGqNYEwnkX-XAcVa72acijnRJVxd5WjV5nolIrtq55l941oeun2ThZJZWujP7eMDuQ8SycBOx_6Bz6wECDbnCrfyxypwpVhKSPGuI1IoP_8fCeFDWzZZhD2bTbH2Uw6nzm9SLODQ47GqYlZ6ZtTIgNBlGpiSUrqXhtj9_1hkGZuGv6AE9UAjFNqAWX25db2I2uH1MXdsYRPLZFhYan9G60cozj6N0ekasNkbaAod39JQ7zL6Np2O_qz85s3bcJSS1_aIxW4YFSEv5IYFlztQrhnlyE_gloA8eRntHAinUaGbL9IKTmuj4w74Al1sN7ELITivL6aZ-EM-F7vvFM6Rt4gL0NvlfTYsafoUL99EfBTh3Rfl7pIwOQWXxg_p-51s13BQ1-HWOQxu1lyxbZdJHmhi-tIzk9iyQh1tbkCZJeh_qF-eGH6voxUlcz07gvTckVKR147UPjIrfSm6EO5zXBgva0Zk3nvGFCZshZSau3tLQrAnB7hQ3AAyQT8_6eFBHtsscuApVGtRYIw3vi9decgXmFdvSEg4Iq6JNObTilSq6a3zmUt8fop_M5qYzq-0ctNsXN5lkqi9iB19lLw9EyHNDgClaTAviXWh6aDdbWP-atkQQ82PXBnKJAiP7luW1qf-YVHtKkwNadbMy82CT-dMNu9c-chRSx3g0tdwTex6tgwKMdBRbPWa8NVZreuTy8x2yarHskXhHM21jrexM0pMbk',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'Nepal protests September 12 2025 BBC', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac9b92081958054d2ec8fabe63f',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac9b92081958054d2ec8fabe63f',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aaccad0c8195bc667f0e38c64c2b',
                        signature='gAAAAABoxKrgxBU1Y3g_B0Eo5nVBHYxLC3Lgh2vNx7AcpSm-o7XiHfQvzLqaLvkI-Cc3F15mQexU0OTvx9FePdIKbwkMNm_X_s_K7YazPjZUTQ0TEod2VereH-Ebh6Xjq3bHm7mh5PWWGnY2SqVMCdKGtrXkoMzBraxedlv2-Tz8o0p6SYuyzM8yHecIfkG6Zd40AdZSiDzsnRNg7gA0zCddrDrRcOpeMTzSPw1z74UZtng-_pPeiv-TGCgwdlmBv8RRr2cuQTYE-yhcp6doCMKqemL8ShuIyfJz0KhQPwYE1zM1CB8sFc_TuArJJD3V2U-Bl3o8anIA8X7YclTlzz_N7HROtVI5qFQjSNhSrbxZKUBFDfAayrpQBEOyIRu7J42uAiBmoyms1WG1E2UtO69nx2ELSJs5yheEuVy4cTXyndBJr2sCs8VkVvcX7xvYkfKeChvkAbUfCotc991qAiyVNzhncM2Z31IEXDEDypeo2IFSwAcKuuXgePFFPBiJxmNQAQmErqbSoB3Woe1j5XjAzJ2eY5YEBZ-68GI3B5wmiZOLsPla_L4iBrczHI1iwGASgtMsuHPj5KVzwef093kg9QBlt-7pZHM3yoU1l5DFSJ5C168MdMdNGF3hn0T2Q3teUmJ5khgcKMKz4_ZVUjEDq8bPwp8DiaWlFgTv-Y-I8etik4o35EFmmmZbIZ7tk69xlBrGizm_KlcYWHBQ5BfuNyZDXZ13MKDyn4uyYxRvkHq4z4jPFEiZ3xX79mlNP3-B0T9g8CsqX1G1prKI7lde6oAHcWPFSWqZmM_JxvYXDBbck2DpEpx4xTuE_iJfGnKiNzanqV4EdOXiCTBVLZhMvXj9rAbwnhttvz5WhIeYAdsKEE0M1MUHuSWuWFVtClp6lPKSLtHQCBtE6mpPDyzUuaw6S1DoixZ6f33Sr8DB-EwF_deHRa95kEN9w4i_LqNbl5QQPF_1je6spo-yQTDpHc5wUidI0fBEQzM57rr9XH0F2afZtrQv9HcLfWKVufBTdd7ScpyOaKj70zgqTAq08Te-Yrj9eo3tbDt698U1fKEYW_uqP48ZKmnSNtFzKOoBzkPpKcwA5AQUiFOYH4-iDPDTOH23SYx8vlymoRiK1imCdPwWYI3miMURxPr9-zCHoM7AiB8cnJlD--zk-j1vQqcf3AntIKPwqycSEuJ7MWb9iN5Ybd1YE25_ZiXKJNVg8wnmTueelRdeM-2JVzAQwth1_3gnsemXn5v0uDVNpxvXoRtR1w8L_zQzKzag8kZMvfESnLCAEwYsCcrP-ngO97iKVvUQnII4RUtG_mSPV4V6Ses_cMUVqyHiM_W_frIosY-7dXnlox89-SPWrRwyC1jlGRA_LE1fpPZ2cZU7Gcyzrxp6yBuTCx8BHr9FJvqgbqtAUeYDpr_Sv-RsG8-w4IulSNZLH5Bh8TyvBGDhi8_lUbDCFTS3KI1ZJ8KJwbNLxF4YUI156zkWIN5yU0WDVlwoxpJD0naMPZzR0sQadMuaXEvLXTFm9Gtb667B2cjdzJqbb8z6NkAx3txRRD6EoezoYADq_ZR_LYha0iwv3bHvg4HIblhU_GVhnU-a-lQGQhTJ5Mh4OmrnTGUVD2Is1OVI0EmNscUuaVc7M1_ga5KbOgyff6bYS0ARh3Io5ekKQKkPVyBLgjjKlej4tB-vSEgitDhEJ-PD__ouuFaogm6twZy7hWVn9cgJmt-RHDZ6gOZm4QP8dWqRpuyEAtTpWR2TLTQVgM05hWpDqDL5AvBjAQ_GWkHCvdCvUINyyl5TsyXUcL207shrLUDCpBe_kESpF5dpAVng8_Zfu1dt3c04cCG1eg40e9JcO5iA9-upTrEPIPrXnAKy4vw-vbhQyL1r2jZWRVga9Do2idmzVf-c7yQ_AHGmf62SHGm-qqbljw0sXJe1rdPt2IHxzYXkhxpqqoaUueQk-pXLUvpMFeMcH97sK3toeCO3oiWQPG-nev0B0b__U8ntgI5m9df6n4IA97iS2zSylSY-F-XEJmLM2TKuSEdgAx1EBL_jyRQKB_8PW-0hSQGJLT70SQqDUJexwyrKABkApv3FuSH4FO0rXZ9TGN3GsnJSkIrTrzE2NG4OXK4syrmtBCb8DjsiicvjAvQhcouOM1xMZ89aSG9Psx5HRnViy6M73TIhYmWO71BRNEayMJaOMgUlgpl5alvV1YFBsChL6mxLVAJWUFuv2YPNaaDRqZEXYHWljhwSn24ASetweLc5GhnehdiT4JVJ_nfT3bygPIjEzvvIa7bbJSeL_bcY-qGAgsuR5m70BdjIH6xLmuqn3lEqulh9n6IPaDciryWqRr1OwxZJQ0-x3u6-G1wrbtrhVMK2Z6cyNUX6MvIMz39B_782X4JcLMrVm9Jgt6qzmfbJPnGA_NK3e9dlz6hP_AYoY-Je-IZEtpv4wyXAYE8v7QXsZbf6DetAM2LzGmxkEI647-pwVPQua-L-84L56GoAw9yDeoXxgyxyf40sbaPIiVLgl_3A4Nghl7uOnOX_1VnZL2X85zCkOZbmm5pZbuSeKesBYbX002PN-_P-P5xRv5b8dZzD0utGv4GUuZJXKJPhbpv8cuBUR0BYHKBQkmOzOBxgCFCDtX84VkZcrFwmQHcS7zmjgqEl39UNrqq6NZXW6HZDyi_SSvEYV7eJfJfxnUUF7RJ49RtSbC9n0AkzorBi0mSMnCC_A1zhamNLjT1-tj4E2a1zI9YsBZ8lPv3t7a6U85iMYjl3kCPiAXkRIDVBihBK4ki_OEa4v6kNBEgXNMuFmd1l8O3WTqZRSTLek4yH95V_uE5DQ9NH52pkgrN7QOe0QXxZ0aErqjkSQRbbhFVVRYp2VN7QpvMGZIAtu_mGssA5Id3X1ZsLEU9zGNibIzAmJdBjS98fVj2MsD-4qZmzlWiCGcC5ko2bbpTrFGtr4r3-SNc4UMOa3dsdyrRlnK3o_tbXbPN7c1H44oneAsqWuekfUVFGvCRm3yA0X7njFB2l8tSXkAuophgRUlWnzp4mEMcpFRwEX3WEnK9hPqXEhdirLtC18yupkKYBtIpCIT98zgJNb5TRbfwRplInEG1E8dk4gCbwyXCNu67QEI2NM2yqCHc4P5rWhwTGAl30tmDQ064ba920L9ZV8d6PgpBHZmUxpJ-JUZuYMzXfCFdlBQANdjtuxCy3-Pi0-cO7UEA84WN-keYB-kHck3aPpeTG7-lv3je0N-407H_A1TKUqkSknjlmwVdL3h41bbGmqxFGizNXfq-uCGUD2tWaZ-cdmZZtGXxgEQ2z7_tLur28eS1tlx43y9CKtKPPJruJm_7BljMOCMPnSmOJDI0JnoGpjNRqzKbSuZFTihaQSBo_Vc-NxRpFwM4xJgq3z5eShb_WamKw9uYrjCBEEwYFTW2QjmiQJtM9eVHBuLkfOVa66YZowcCvL8aCccsuPbe7KBMCD21IGzH4nlhfgUKa1cTAUiWjRSgn6SO5Wqahxs7dEf44F5HvPG6XUy9HFOe-d61ZE-tJQsHZgssQWqV1UfPsccqgyWIc2yv9aK4pPpu2lcrlGu8aDZDz7pBD-dPUG_B9XWt5c0CQj4CCnURDATNWqH8J8VvKap6Zn7pBHW_PxNSJ3f0z_l-GjBlx7U4w6XmOMBtJK8lE_Y8CuuQY9dNVnTGMPibCeJt7M_Q9-IYcqhriUh7Q5WkCvDVu8157gIRwwUAvgqsWcD2msXtO9svRkXKxNxYFdW7KolF-y8oxXRPwVJy1bf89pAOa8djb21ovJuJmbvrRzplFGYNj8rGZ2hXenxDoYiKv71LGALVU63mS9q-Y1zfTHCPpA-Rw7oR6T5G_Q35H-elaA_u-vkgh64mQNP5sgc_kpwbVlM0wSl79RcExnmBTpA-kn7B4w_QPwt185WD9jQRjhh3LMQa_crf4nCWLlsYcDCyB07TU0vXQiQ3nynqsX2MstUc2DaiseVG1SO0UEv8oobwLhnSvl3n8zWMWq93NSuISAsaWmqriNhM74aSHw4CVPoO68RSSdNrpxaKGf8kuO9Xy6iLr3VPE_vyMJDq65q42AEvKqP0TCoFUzXA28Tkrg0tsMLsXIhuT5MGtO3O8RpLnthF9vT0lM64jMp9_QSH2BuWYtwgok7xk3gRX5yBQeksAos3c7Jn2bLM9VNrV9dLi7MH_mRl5C64b0Lgj6Zi1USCyyPhL95ZJIvdxLWHSII2RFbL9ToCThKp_cgPZklLAVJXBeIOqG09pIQ==',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0e3d55e9502941380068c4aada6d8c8195b8b6f92edbb53b4f',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=115886,
                    cache_read_tokens=92160,
                    output_tokens=1720,
                    details={'reasoning_tokens': 1472},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 23, 19, 54, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0e3d55e9502941380068c4aa9a62f48195a373978ed720ac63',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_model_instructions(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The capital of France is Paris.',
                        id='msg_67f3fdfe15b881918d7b865e6a5f4fb1003bc73febb56d77',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=24, output_tokens=8, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 4, 7, 16, 31, 57, tzinfo=timezone.utc),
                },
                provider_response_id='resp_67f3fdfd9fa08191a3d5825db81b8df6003bc73febb56d77',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_model_web_search_tool(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.', builtin_tools=[WebSearchTool()])

    result = await agent.run('What is the weather in San Francisco today?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_028829e50fbcad090068c9c82f1b148195863375312bf5dc00',
                        signature='gAAAAABoycg20IcGnesrSxQLRh2yjCjaCx-O9xA4RVYLpo0E7n_6m0T1IUyes5d6U4gDzUNRWbxasFx_3NEhFIuRx4ymcqI_K-nZ6QNsq3V4CgwBbWBXRcBEDVzXSZZ4IoFASBzHpQGbs80RvZkgqmJkk8UzBw0ikt1q9jlUrwMKf1iGdH-S0fIgZn_uEbli1yGWRDryyS2YQWDKNTYuaER_WHVg8DadL6_ltUTwJ9dMzaXyFEenPfuLdDgmba8DP_-WYFMbggATUfdMNfM0O4YqnTmjR5ZnSA6kAbXvnp9sBoC-t8e2mWiCXzvy8iIJozNPo_NE_O1IcMdj1lsaY3__yWzoyLOFCgkrZEnB-_WQNCSx-sVcWWLZO_Tqxw2Afw9sWAvFR6CvTTKdigzDpbmRlvlAJCiOkFQCMrQeEiyGEu0SSfqmx6ptOukfJn4HtQguvigLDWUctpjmNPutwP880S1YwAcd7A-3xp611erVJtYFf6oxGDXKKb63QAff_nZ57-7LdlzSSUr6VaJa5dneGwCgKl-9J3H0Mo-cOns-8ahZOL8Qlpj8Z2vZLS5_JQrNgtmDaaoze13ONE5R84e6fcgHK8eRhBNTULgSD13F59Xx7ww3chlqWeiYfHFwmOkNZp0iNO7RJ-s7crs79n2l6Ppxx5kd4abA0c58k1AZj7avFrexN_t7snuYqCNPsUHMUK_1fSq1toGa7hTVX5b8A56WFSdMlFD51AuzeIzgaEqBtGvq51murGbghqUmOy9g-6_vHz-WOPZeE1M2p13VB1n5fIh3-V7nd9PAXLX1kLLKiS2ox5tODYvkxf6oqjgR56n5KCuWtF9WzCwikaSMN8pwC3ewW6nkkSCPhTBASEJ7BK9a7lDlV60T6gikDbZGHcAfSKDZ5mBBwSBRpDfH3F0MI0Uo4oQ83J63J8a5r3JKy4KVa-5eNsNZsCgxO-7xx_fan1MH9zT85SLwocpvryGSbIDD9itBHK7Yo7REFRV6_U_cdi5RhDpEc13QETSsFT6CaeoL4GAwvJDCrcKjW5u64StH8l-Z4XDAtChG-znHeme6WlJNElY5unp9L-IolqqypTS6lybk7bfUtGPBDeuZp6CD80qFkyd46M16vP1mudv8rMC_ZEdFvCoHDmUg6_KxBxdVbYi-jaXtXYY9D8G6SlfVkeBcNiDCWjsDXSlhE1ibI2pHHN2E-kJLRaHA_Pse0Gknu6ZecQLaUCKWr_mKh3axV9d-pkvxpCcVVakOF08By0bUe8h5ORELsRe5zzMpfbYGaUVhB360OxwqzizyISXmqhW3Q7FHcgZQOCZQVfpuk6ccAYpZwgZbft2YZWqw7_1MyK6TitpdyIwdLFnt2t81JNoJ8zWLveZGpuKABxW6krhjQ0_qJCnLHm03o_D-9BximrLUCs0PbleK5mu4Le8lCCs4eoVjeDHQs4xMm-VtJk_3KMT6EVe4nrb41ddSKX8hH9rh9l2NlPpmPh5UTledwhbtQYdJdQBNFkGei5gpAQ1oHaLkSOYRqrRmy-VIBobxAVBaQWNKcv8CrGx8RIMxrAiU8JoyRsU7Vsobwt1Jboo=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: San Francisco, CA', 'type': 'search'},
                        tool_call_id='ws_028829e50fbcad090068c9c8306aec8195ae9451d32175ed69',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_028829e50fbcad090068c9c8306aec8195ae9451d32175ed69',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_028829e50fbcad090068c9c83115608195bd323499ece67477',
                        signature='gAAAAABoycg2MBei1jlOMd9YfezZ45PArjJAExhzJt4YG36vuQT_e4K6W78Awn6mrJEueCnEAbciBRoPBd8n0YMXbqTiKdgeceqAoZu_UJAVWxgY7tVDlkg4e8BgJ_SrAumbi0yL4Ttwy5yZNU8g1aICCSdjGqfI0cmVbJpXEyCU8Wt4UKV_912jaG62vA6Tlqii0ikc8UItcrgk94TEGpOEQXlG1HXsWyAryCvOMSM2F785Q4Jx2XOrNv4klRPEZGUeIbp4ReTVXVi0JT-cjc6O3gKNxN6vxzUbvPhmcyTa9UogLuCTHjv3KpcIvBOw-_pF3Z02oQE0GaJKBpP4SJLE2yZsIII4uMls7Lw07EuHZjsZoCQRg12dRle6rwba7IeRw0RJWYEp9aavT0Ttrj69dO0e20NpispmeAXLh0xxrRCKcjxAn6c5XtEbJP54_ka1FUSVY4x8IaU_pCKI85fGmHIx-HarXBtWzZO9B5O1K4Pqr3BE7LELTXaMwWQ2SU-RGsvgmDpmUZjwifQ2YgamjIJPt0UcuGWb8BTwssP81XT5mQ2Tsq1YjQmgfzeF28yeb7XhkEaBUNejSou3SuEXZ9aEuSaMz62gzPSpsSrr51QoBJpMBF9Jd7LXuFJwaQV7jP9NJawF9GT-CMWj2IOXgVca7cL_d99IMSR94vNyg8yPzDsncJZ9Dw3HXFsPfdGHtO2FaFUB3RRZAVKoHy7S1NTNfLxdtB-p0eDuu1JbcsgtULWC71E6TbPxg8OguiEgAPTXJviUAed6udruUrSMlZQv-AgRYfxYPPMXLeUIWTTUo6PKICy_PO3U5CF6VBkaNUvCLf317L47FCeEAJNTb9Uj_S67ZqoAnEG0tQG7tVPuN13cy12xO2-8xFQSpO7gg0DzF8vCD1cAcKAvo0FUEnIeXOVHVQxThLHDiXOmB_ZpoT-qJYb88RTLNoAq5oI0ZuZYvPHJ63EhVjaANKwNe4DrfAvoPpf0qWiBOH2vHxnlIJc84pRh33ixB-azK7arhetqwIuLhDo4u9REcD2avxew8rDEOTqb5Tk02hhCKX9drLYCriNdkQh3mrC3KYzOWZ9aebwOR1c-s54KbvGDHAjTNPCLlROf30MmTON3jb-NW15YyzQrVFfV1c-egUiWRwMVE3KeWi4wmicK_QGMZkdyEqZMSzNcgOZMFfUWxdUKxACHY5J_7lUZltrz9JnhsfuM7KMuEW3GMASIP8f8WmR03nleJTi7k21oLtX-xz1gjble9WzSzd5pTz9GrFw4KWatCyrLXtKWw9fAqm_k5HpIJdya9KK3jNve6MirP6jdetIUNIbN3MGkMJ8lfavyTaa6-t4hsQSmyTQn6OKwhK_PA8-KTluNMW-dpqZU2YPFYk_QHYW6EJe_Kw5aOq-zpKR3hGgoHm75Ossr23QERsVgP0LChljPzR4OQlce1GMDtRNqLX0wGu1RO7OdM9R_lqJWMlIaAa5wfvdH5LznaQV1vuGPrfpzGL4mlocKDv8ASvrxA4bm5fWBoqsfzcLu-H8uz069vLDyHgrPNse6W4Ex1BVY6By0K_f7sidbmc1FxwP3ypVv4nX_lncg6RiZzaQTHTxXJFmvVO8_L9XBHJcGkQGpEuEjx2aMTWZGJNxfaO2fKJ8U3XflYVXJkSg5b5ixTHuvDYjCOELs3fTVAy50CuMXMoCEgyZlqZNg_EJXEmz5niLNQnwQPRWUbe3kicaLzJqvZrtrvPOPcTM31Ph2-_dfEOeKNOIE2B0pvMgTaFRck_xOc7s5J2tWAEYszDz6aMXvnvzm1WH9cXYLbgZPyJmMUxeGZ70DdnueVbrNr8VA5bzvjkgjEkhks_BQprXEAZL1lSL2s0O9G8ekgFnt75JBJmSFGT0twl-t1ia1BFkRtMGXLIj91xWJb2GsF6ZN9Uknfm0Akfk1STtRbxFIeBRlwQsix5rQ7EstyhfsBXiBILky2rSfj0UJwH1NjDskXjFxxpy-FEE7KRYwMws9rKKuMQMyURUK-DbLvMmQoxekYvqu7bJfWqxj3lndGwD1sQL78cpVVPVfJeqnlAw7k_xd6QdHg9DwSlGNb4OCYdFWT4xaaltFIJfo6g1Pay7HD8gWTrrgUzHgEWfbJxcKIXs1etHx1lxYVTmm9TFkXshmsbKptL7kAaxBy9JknSsGsh9gZXf3YFkocEj1xa8f8Xcuf3zatefAeFFh1Q629b0Sc-GzfXnu-KfuSyJzAZulrP1IQ0jlOiGP5hKnvzePVL_JZGTNJrJxmtWXejLodY-JzLzUjIeALKtyUsu1ELFtwDxyadPSsFW8qvMeolLcVDysGm8NkmRgLzQTBDGR4AcipdozZmElDRTm5P6JArLlqdZCxXpiOH2x4juPIYUfRrrTT2g6emTXHz_AurjFgYn55G6xv1YGSuM5tNBXc_WP5ya9cdpBIEYj1i05DIMsvUPsNAkt0MIeTiVSPPDMgpT4lLsR1ezwBMx2kQBJI6E7rmH9f3Abn5H6yeKQLZckAAru1SLkVwoDxcTTJZqD3sZt6RhBDuuMWX5ZoB21K-zkE3Tde6caBupWLK-W2eGJSJ_oOaG2YGQxL56irxU6DIVxLuMWUTOVH5vpqeo2RlrGpXu-lJkg3tC69gXlNd55233uIkchhihakwSIxFF1Ka-hcBlKtn0Kz7CXrXam4B0sSWjc9xGRfSOaQ6LiameoozXfhj8r_GSOwoV8EMa2vIBFggFGrPEzaczNkOKBiA-xTQtdEPqmfQNznuZ-B-VX-s0E0Ew2EopP4ljZ4QMW8k6pbNX1aegBBxbxkNc5ugJhBBoSVJeEAC2Lw3iCZUnX_leWUJBp2up09oJtRWlnGG4mLAu7nYsI7blues0ZLZE4C49v2eYBmfkeyq1DBAGXu0RC1qMz5729tzLPUEPYpKS1H7w2iGHQ9P1jBBWAAfFoqgn1lYtBF1ioxL7ry6YMrvCgTlqvVRXB7zmAUlsJdPq-CTWpF79YSco4fAhrDVCmxdS6Y4arD7p26YWk8PioCDt9ranaUi7--wlyh2OTdJPHAUHW2-o5NaXXfhqaIVfCqH1sbVmNwP0BRiAmUlwK7GB_m7dtEztYz1sHl5sXmXEDcFjJtr6uozFDjEA42F48AVuZMlQfQ3eJNSRqHEThYeyzbtCdYZ6J6ntg2XS0uDHISgM4zi1mDeur6-ZCw4rGwUXvB1BWXifFeh2miEGtvRzw3sa1zBKBCGtYtRsl4Iz5Plo9RNN8eQ_vvwmfDk2F-5YWsDZbpJuSXQXy1hjDvyM7TVGj4uL9gxFQ-ZCxFl9cufUeqfEGgHX38mZoJAT2emXbe4A4byFYvWfM-NxjpbNA67ZkOWgcDPtY853Y6dKoBihh49ZAzvmEjmPixKp2rBuNX26jJzhW2OJH91GpsncHGwJ3ajWht88XbKBp4Lb8sNVxYD3hK4c-mB95WYYaUKe5_ugc-PhC4FGu-FYNLYTX2ZxLKpk_T4uEG64zBQ0NbS9y8WWiTojeQ7b4-MBG_j3VJr5Pi0T0meC623J2ldwud3DRBZXB5q5rKgofFF6WqvwhIDi8YLL7CVUJ9aOE57SkUKVrYYD48Cv8Wv9piI2hbTgXwWkCpg_tVROBjl4RYfYVlOBV4pM1G5AK73PXfDGsPdiCxhmxHlvzanAm30eVKIctRaS1xlcBqLp8CUPkgnPDlPVclMagd1CjIlN4igMnFN9gDPOUckrA0-VBlg-EKsHG3o_jNMbsvgfXg8BuApc=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_028829e50fbcad090068c9c8362ef08195a8a69090feef1ac8',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9299, cache_read_tokens=8448, output_tokens=577, details={'reasoning_tokens': 512}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 16, 20, 27, 26, tzinfo=timezone.utc),
                },
                provider_response_id='resp_028829e50fbcad090068c9c82e1e0081958ddc581008b39428',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run(user_prompt='how about Mexico City?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about Mexico City?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_028829e50fbcad090068c9c83c3bc88195a040e3b9fa2b3ee4',
                        signature='gAAAAABoychCJ5ngp_Es2S-gVyH01AJS0uQK5F4HdAkVFJcXPiZanGRjmEPXNGx45nUyTonk62hht1dZ8siaE7v0SCE-oBFoP3du4UqNqtCmJ_0_EmkXG7sHh3pR_zuS_iEDGae9S_qM-vcVXyqFYbEtEVD9ZimiQGtLEU7WFyQq4UeLuD-U4vRhpFreMCAfen1DkV9txJijEPRL_2cTUGT47rpi2HYyuN1CzYKzRrn2qbHsgDjnPtZ8cY-QGTm5Mm0LHV9GeDh4MmRY5Lgxt0slssKI7vy3OqTWR3OCESp-5VmMR3fbyVNxkeogT9XqPfnl_9maf5jYLv57tVGVRJUEx50QvMJ9V20qbUzIAuMw5d11s8q627IyyFu-bD8QmjGsaBj_wsjdMe6adDF8hzOau3svjuouGf066I73I2euw2NpokdNA8fbI3bAHfqyXpFDADKXg7WL_zYB0eyREbWe3n2mo3KL2sLW2908ScYEvsv9VlAo6q1vByI0wfGmnkqkgBvh04Fe15ljjSkvLy7iRnOFL_CCPakpDcViIOD-yRSDk-MSHpQsK1sP2GgxHHy8jGO2g_ef2bOH4FkcYZK1oJLIUGqhLJI0LurXFnLZ3zcUML01aV0rMFyweQwbdIjpivIGaAg1BUPU1Tc8nCNmZC5aRcbixMzzu21HtW1SWnMziebhKHyN66b5skUXl_RHrCoKhFyJxSJJjxHeuUKHQ5VxvJDJSylZjHvMkX0KQ-Vn78pv-Be5ETRxR2G3Agp-a-iX0zM4HbwVyoF5l5t7g07pTrfEMP0WFJu4_OG_tsy4u53JGMQwQLB_RNYcd2n1yXPCpZYHuq8Vkt6-A7kYHW3wvUmI2cSyZGBNpwt-pL7kqdPaGyqnfhMTDzTS_CTXBBrCjjQg-RsWGu9hYon5iKgHFv-w_qGykzyPtEzZt_VWUrVm0WFOinLqLXTQgiKm0sypDdGRht69Rbfe9WqP3fhFychLwcP22IvDQsh_OenHiF5ytB1XTI90VB4e890QUI2CzsnH-8fFkQT9Bj7ou-MstjIeOQrCwDGAPRnxP8PWoCg3uYk0DuAWuJY0lYq6isqGKc57Lz1bLaGRG3oYpWH0MC6b-D2y7c4cAgOYMhOzYq2ufblZDinvBLrr9TV5jtog21xrBy22o7dbVEgIJ2T2HI2XOmjG-l7qrchcAykaosXQkW3ASIv0OpfG-SSd9UU1_1dOUFzOXGej5UMxZidzQa_dW3XPLCqVqgiDW9HCu_XCmSZo36DY95I2hofXq5mXUHT4qxdZ48y7KGiM6mllFudcdyXu1w8ZGFlU0BfzKDOfbhEJz7MRLuXL6GO0bCHqgFo5WHJrsTNrXuHNNTe2LxPPIpejVl6kvE_1LtHy2jKffOR_BcBCS1c_KLIIbl7U10__OWglq3KpDXuupMa9-fXXSn0Ko8rRybTLQpXIn1D6phbi8hhS93EkaVE-9zZZGBvgcYhPP2fa0XniiexQcX-VDQ==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: Mexico City, Mexico', 'type': 'search'},
                        tool_call_id='ws_028829e50fbcad090068c9c83e3a648195a241c1a97eddfee8',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_028829e50fbcad090068c9c83e3a648195a241c1a97eddfee8',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_028829e50fbcad090068c9c83eeb7c8195ad29c9e8eaf82127',
                        signature='gAAAAABoychC16oV3bX-2fpfsHcxFWoRnoEx69lgG3ff6IIJemlYvbM8OVbf4e41oydklk1kkBbyWg2spj4puPrSV9w19NOknK00NJ170UxqM5jqvtZHcvAAdeBjA9XSyRObTamXE16W3KtXvRvyRBmBpqpC6pQGX15fxxdAESZUV6uUexSQIYZEfCT2q2aRj7YV4kCGXUQoMRvjzFE9YLE4LDNrQykcrIytjZ95hz6czjxsd95qmYtGdjMU-s4BlOvs34pE-d-H7cR3a3cQHI8SkpaQrL7bCOxZk2fYws-t0YBXEsOIRNCpX3uEany3iGgq_8jn-ggeZtwvnA6oRFtIkzpscLaU4kwhlZbYHNI_RinezdR5ByRjwSdc2UHvqoLb4a2rYmHSLLpSmvr1f9UesAz2M5AexJYlk4sDmGhMD5DoiLy05lbnbo86osBDmRpwXhb4F0pSVgPxUEadMvvr_l69Mv_VAhTJdr_iLFn3E15HCLPCFND9TcROgxPzhW7aeDrt8fJPwEZZ4fZ3BAphxP5sOzzmd3-6uwCHLZxB-51ILHGMkBVmGxFSXB3u5mr7TtaDafh7bxWQv2bpLoV3Y5QD1lRvBj6sx95B6J-CWgw0WeOd7jSgHR2Y6nDzD6XAGgg-aEK5Jk3CDGLsSqv6SxYMoY9MvT16syFsNuEki6XDx3cF252VeOHIPNPQiqBB5NRgf0Vx1zAMgAn8EYWarg8bWsJrazh_nSKWmM4gCFFAUK3Tqi2rfbx6eCPlPBYHxX73GdiHrypeAA50pqVySFxXzXgeRKghzGEQetBPzNMPykyUmiDuq3oPc_bliFQu_15-rDhEfmJcfS65DpL-_tLdtTFV4-BeAjVNsdPjX-7I1bTHdZzyuBiMr5sltxKzmHd4fLWLKv_ZsAustyfUmQnO5_reR0T3SwlY2Ytg4wJo96dtx-XUqJxWgZ9tAW8_rhwgejaH2H8zTM2wczgWVXJZxlsIl_U1xY4pSgxosqBq8a5EPrAqJFnpcZqj9ctCImVN5oElb8o4474pOhSeY0qFQgL5iol5d6QB1gNTKugU_rCgAPbHwBAvnONLJ0v3hQXncgcuIJgQw8BjpOgS6KTXLmf-5uH6CyXum-oE3JJy8EMBjvyerecMMQl6dpeJxYHlB6B0RUUzTI5bHFaoJeSGetoKH7t-L2lUwgcL7F84Wf1ZU3EUkCPWl6DdUq99aLfYLWPqd3bQ2JCvWiMVrlwuHZr_8l_N3gCWuy2t43N2nAKBBc3HWoWRJPgHCmkj0MIMdnZBiUD7IXz-b9jO_1ASYT0NhOPc3gqipzP_9lFE0EojjvqUXV1P_OiAX-Cl2cFpn7ACDQpxAGyW0yr-lgffzLI0GA6dP47DMYs0P6dQBD6XJFbvlxigcl_9GURApvAb66ITpFWMeQAJOCGdMMPZF2CahK8Riq9b0RtkSmgmmEL9SUNaMpEJBlk6j41_IdZnxnO4Qm0Fqos6RFKFbwqfxEopy9rVWvkbjFzRS_B7gAc0kH9AbFx0CZ61NZYNVnQcN1qpr0iuJtSGG-DW9EjT56IFtnt_clgrjfFuFj3cwX5ZcKMrN_RTQNgY5QhAPShSXUB4MDstvHgFhBObn-4rDl3TIFJiIgNY9lBz5egE8YZZXg8XxW7nFZpP0fmQD9a0CPdA1BhafzNcvCbReTjddrVeJcHhflTNjy0YiXrXUyJmlmjO1y0opcXkS8R3E-Md73KKEW9wJUOuEFDDr9PAaocHUsvqWPTNb_Lu90knDMKEi_NnlB8SHf2Agg6FkyMo4Z_k-T_51IGYfFJHPuGRZ8-CqK-qI8-6BRIDpnei_UIi2K9ALXGOuYrcG9A4YexW_vPg2qmoVgishgzr-ddFGOuWr_j05j7AKffDc6wqK0PNBTEqpnMKSVICOdOEBcilXsncLhjFm_JmS3JfxaM0Ly83tKhZqjP83hxrL_JvBjBQRuW7LwyYuFbE_8dAysUMI5jYwqPd40mGPALADFca0U1rolFD41tdX6LijA7Wz9JjYpfuphLiXNH5cGqTe4T_ReZAN29DffISVS08dRiQUEnw2-OMBYz_nY2qe1vyEItwYmUe2fjOgec4ClJPdRDXBW0HWVS6ei1sgOOD6FvA0moRFpSJypcEC2R1PiRqN_FEoTXzRsSAPF6pXoQIlgXxudLwitpW5xSZS4v_DZTlGa7GgHnq_dhDRdSw5GzCvqPU3CSlP7GmvxZKA_9WoiHNd6JdOSVJg6x8BGpxDjvJy9T-XB8SIKyNx2ymCVKaEhnNTh9UefBGcEXR32oYiRa6GOLtVLt_7OJ_YOqSU4XB9OEjoWlWisBxCrvnAI6URp-wxVLLkLzAPhX-O1sbjcOkCillvnJWyDbnL12JkI0NsvenYonUdprMbVKcX68KdkkpgmKyMICY7eUKpZfWy32E5stRQFUE1GMZ6wYKGOBFa8a5QiIwIx_4IAU44BZCqBDaV57H9KAlsHhqY0K9PJa2fetDVGb2MKohfcEmF4lAzmHKiu22OINYHBYX1LZulsVrcQUj6zSA7r3GEEP6K6wBmk6i1SuLgf4ze9WC2pyb9zemaZ7dHbb3btZw_xAk5a-RVoNb2hIXfiX9clN3BkMw5V2vbpDHaNM80N8z_3VC5uXkQ_v1543ZFWvxbdvEVHlR8P9JyG_Asts0VrwDnFAo6rTGmPj52GJcmhLVAgZ0KPDrujpGHu9HTV7sO-3KvqxOMHYuKG34GvpjfZzlgV8GzbXtpsRk2E-GJPKLfLN9KIHYMxdfkaWBurYvea7iMYe954Gcwehfvlk83foG1ez6FtysZ2V4eLjg9IcVJVAWucdnUWyIIgYMocgpS6ESkO2wRs6pUz4mg8MT8q-h03BJXmWiJIi-4_3TOhz0owLKMza_1IljVaMAUIHp6Kd9yEPohWQo3uyGulXU-vEsSeSkId_sVxLphe9yuimK3CtzU7FBjewoGhaj9vnTdv5_abDRZ13Glp_b4vpfUrr37CBAX_RwJ_mTqGhbv-mPuFRVD6ESjlg-JrJDCUY605dcyU_0hyvjSFepiHQ4FCEHzL6GNSfR',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='Today (Tuesday, September 16, 2025) in Mexico City: mostly cloudy, around 73°F (23°C) now. High near 74°F (23°C), low around 57°F (14°C) tonight.',
                        id='msg_028829e50fbcad090068c9c8422f108195b9836a498cc32b98',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9506, cache_read_tokens=8576, output_tokens=439, details={'reasoning_tokens': 384}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_028829e50fbcad090068c9c83b9fb88195b6b84a32e1fc83c0',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_model_web_search_tool_with_user_location(
    allow_model_requests: None, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool(user_location={'city': 'Utrecht', 'region': 'NL'})],
    )

    result = await agent.run('What is the weather?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0b385a0fdc82fd920068c4aaf48ec08197be1398090bcf313b',
                        signature='gAAAAABoxKsLzYlqCMk-hnKu6ctqpIydzdi403okaNJo_-bPYc4bxS_ICvnX_bpkqwPT_LEd0bZq6X-3L2XLY0WNvH_0SrcNCc-tByTcFHNFkQnE_K7tiXA9H5-IA_93M-2CqB2GVOASFmCMBMu5KwzsYIYlAb-swkkEzltsf5JEmn1Fx9Dqh5V0hxkZI6cz35VsK0LEJSYpkJjAMcfoax1mXlnTMo7tDQ_eBtoCa_O2IQqdxwPnzAalvnO_F4tlycUBM5JQkGfocppxsh9CQWpr7bZysWq0zGfuUvtuOi1YJkHVlrqdeWJGDZN7bgBuTAHMiuzx68N-ZrNgQ2fvot0aoVYBnBDxJFbr82VJexB-Kuk_Zf3_FVm-MGcQfiMxvwHgEYsnaJBvMA56__KLlc3G4nL91fibIXbh3AZ24p3j1Dl1V3D03LaEdU3x6RF7fF47y5eyaFWyWkmPl1RwiEaYy9Pi7WHuh-6n69ADGYWbv0m4mgvECbmvbBIIkZWr4y0UK0B8hbC-Oqz776Taww73OmchIzgkg09rIz9CfoKcGMXgvzbpIBa4sME5BQ3mQtfIdPLY7uUIwya4o_g5wVy583MQva75jNsR4A6sRVW9SgVEWusMJPHv6NLzHCdWehp6SBcKuovxZayoM4KQrIvUMNlUkrSR-euoBaa_WNc1HeY8ikKolX6emm2LhRzXH5HssCgH0g8GUvWilYx7U-UFSB0r6yoy44_DzsyH85pXN1ivsSU5dGIBQgG7WiN3bfk6oBGSrz4XkBLiHJiBX9ZUe270TeDNfpgjmKO34_k35zviIUd7-kVY4EsJGGijEhjbkInFwhilyH08EdKvYDzrzpKJIHT235drt3eLTKXKEA-g3iW-qOMqH15KPk-slzPNkE8yahWEkLrYsqGsjwdHVXiKF77-i8rwvDWOf-pOs9d3bBxily3t-22D6RsOL6wFYQS6BsuroKdlO3b_0Ju5E2Kq4P3jxtZ8jnG9D2--XEcEB5x9yX_brfdFuFHrF3C4mYVWTrNN3_S9V8zUp4CdIh3EqAuSs_QJPJuN-RNlorK3bwYqOymgNlcezKIqxhWnqtS1vxuxC7msRlJRmzTN_Lg6XuLRNS1uIp8jmx7TcCnDx62ynYn2oGCOCLSspK_T_LVTG6js4Oiw9ZB5A_I3TfDLrtnLRh7pGJnAv9nVnfYd4Y1czSjhPui5LF-FvLOlzWxSu_1Mo56QA1BIerB9lCQsDjPOkLF_XHOFLWGLQANx5nQ2wlbgBNyMcPacQowRyn3NncjfzlSLyaPijEZ0HROyL_Hff5JXCMu5-6muvxQz1TirmbyjBbLjtv93JpXrVvby14mdXdNs97dMATIiqpwF2r0873_dijDKRxIDMZxqFB2ZBaHJc80khjG_NaA_jxv1GEqVWmllBXBz-wUDbUJKtNtI86YmcZboZIA71V416UW94-TXbtyQpGlB8tj_764sn9fKitg3vCqC42mr5Kj_aTzAN34BXLykkFWYl_AfVL5PRbJXc0Uh0GW0xTH8eD0hvqd2Xsr9eCoP0nGM6TBNMCl4T82wOhRy7jelWMpt8LBxAYkw3nAlVVOi2puCoYRaRFWNQnLcO5iYBF8_rg9oX-cUsBFepGGDmoOfwUmWLlYqNZDho3AJ_SL3azAVJz7lqa3vcFubrRMFiGcee6sHj0HJI_2N2mZqBO77kEbXrJ6SiUV0EXX5vrjZGzpU_wZ9G8AUz9Tdgistq8XLVsMC0uZWlbRdqD6-UjmnsJW7XINzH6MnkQwPvbduRKF4ywViUUbKVs5XRVFUQF5gTdVMTK8mIIppJx6fQRfZBju1NuNrdTDjd-5P9_QNBQj89_Y_N1fow_676bSvYrhlrIXVuLGy0-RuWezuqEwenIZ_U5wSTp9remqWzeuolwKnF7xG_QlcxGOgCivkRvqAyDxWiqlBhUtC-oPEQtychFa_W9uLHyBhm4bcSUz9KvOlUTt9fNYgvDWFciGCE7B5iPz2s-lCS-Onq0ZvUiZY5nB63htK1bIMzB5lc4N7XVh6COcSIArGBnXKARHdIenJ9vYBSmB4XBrKOIU6SmNNM4fq3ZFoWIc4gsS8L5LZyhTX_qlmY2L6znek3XT0Z7kjEHs5qQ87_sw9ho2KaqNSjMalbUEp7L0JlU73szrtdpMkmBk3BK0of4Nl_v_CCbmYWW9z_rsNpTpPQgUHNVn1s38DX3cesMqlzlBOky-rpLAj2-sS-Xj6WZWBs_8n0lLFS7FL3IpKzveOXE9eV4zjJSZ0y74b_g7u5US3dT8EgSEeHa_pGOMn3t3J37oz1pZcSufD8vjyG7wtGxYUGn8L9U3zJHN1VdOR9id5VYOo3OLtMjCrSqPO',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: Utrecht, Netherlands', 'type': 'search'},
                        tool_call_id='ws_0b385a0fdc82fd920068c4aaf7037081978e951ac15bf07978',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0b385a0fdc82fd920068c4aaf7037081978e951ac15bf07978',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0b385a0fdc82fd920068c4aaf877808197aaba6a27b43b39aa',
                        signature='gAAAAABoxKsLhc5YCXvcJidIAJvFyzs2T3IwW0fie9oMN3Nk5fAcAP3apWArzw8DdWjWjR0tn8Fpw_H_xATFGktsCeA5nzkcKvdc0Bbu2bwMo2QUkQZfFcLHqlcNnAcvrw49XpolGFl-mu7hAyP38LGGtjbTBNRh4dHkd-hYZzy3nYd56JQi5GLS_KuxdU78xUW3gNOtAvrseTx1fcY2eseUcNLm8uDi8a_qDw16nFvuY31ZkrmuVCESawkppmxrhGFVg0Y99dgyufnSVXXMKyE89tmXMc60yZiaB1i5cIJQcZMkwupod7yZNGqmr1GtFru5uJq-bJfGx7nAEs50jUcu-rP-_ZbvptkuADDC-bfzFjaeq13wCih8wCXqDWqnGjqIHlFkBM6agn6VKOcuDC18L3caqcH3KEYT4f3TGwg_ZZjsiRDdBC-saqIduaAjjMDqMKx9XpmreRq5BLfC7fPjRykpUcWQQYbQ07J9pe0EW2VhZwoGtd1u96fmz55MzryX4VOWIwDsUTEZAoCzULvVrEBnzFqnfvQwejBxJX2XU4fIlOtT_XpOcI2afolh8KgitzHHpJ8Dr9ELI-Be2KEd6enxmdaPhgYUif2D8ZCVfOoXZEmrFBMQTRyuxtp9H0U3zGamEYuUxRavxkQD77HhmqWOSr1Agm8pWzAN97jxJSxxY4BEnjtrgp1mavtv4G7VHjrpNWrL-smZEWmnCPGKVxP9afrdSZYL-HXKY9yO6__0PR6DdX1o0JvUq1KFPx2dzag4eXDxb56HI5MKNr6J5P8Smmxxwoelx6UXEKw_hyFWMmPUHYD5Yw5dxrXeYmAiomYKFpG0bxVbuAb4_iAVliHkdIsOBcWoix0KLxmS-4RJnikZPMvDwLDWfENZ2sh9_RrQbuMBAgjHwlfWM_tww0ufm_aVdDZ1CULJ5Ki3ZxH_0oIRRyyB-a25q3DARnVzutgo32H9X6qjMb06ExMn--ndCinBglTTGvj1QOIJews6UMrcKj5ZPTc7GyPbHXvdPmPdIrtJ0wCqFj4cgNRuxjiaZDSCqmEQERYyX9Fxu8tY4f7-Fxje6A_zflqrIyhLfzo1iMaoNbba4HNkzRMWba1L1fC8St8MO4ZuZTGs_60FwzSUmBDW4Gl0CcRAdY39BE65uEpKGZeRqDfxvLUelG9YlJTowqN8hzAYShzcPPkgWk_s1AtY0RT_roregPuQ8PQayvHcJzKqnijOIhRA9k6LjF6cnHj90d6fSzTYn8F27rhufLySe56n9SA2WDWhVcjsFEFAcsL461tjiQ5U0mjaFdBQ5H__s09dhp0NzhE35I4q0pzM2KI1YWgLnwlyPFnnfce9bbL81jvbXw8DDC2KfZVOGU-ZDdqIqF0UmwNyBaMYb4SonrG8vrj5bFmCMPSFsEeuDPv_bmD8HRx8536b30RmYD0K38Wf6-UoatMxzgMpgmwsBP6Wh0HCpFeIhjRsJLxYXeoafypcKJPQgKXJwuXVLi4iejXkrbjBdc2Sq2dqIVzzUhULLJSPBYouyjeyVSbYYp9WPoBNWj67uQsX7OUbQN1_qxopsPJdqqQynJIAtULNHjKrDA0GKpyZ3OUV660OkogPAWoxTVevRemwkIJZbr2hXyy0Nx6Xc1Vf9xC0nPclJ6VXapdnjK69bIDHxDUZGCh8UZt6DbcA7azBrugcXlbaMJzoHWkzmusJoTh_2UXRjrS3B33jsxf6LQnUl0s1ETo3Tif868zLvkTEtfo6btbND0FPDFFQrdeVlW4mUWEOJhPeOmwnDeLsafTfRCI_V_xTzgkpQxx7pVZt6mkYZ2qDTE--NhqgFfHPlw-nC4zU6klRdbaO8284QGlbJvHmdsmHi4AtMSWAf-_jegocmaneM1wUquNKoy6hnbkZFul9qV2c-_L077uC4nZYNjRay3lT_3giVH6Ra6WnBovt9ocCYIwSeygVAyqBHxo5EJpfyJhNCtak3bl-CIz2TraYqqUCiB0h1fyxIF7M0uENZKALtwqRVHOtEsN5JVotgv-8YzaBRFs3qvtjQn7eEcw-zrIg5fwMP7tDi8O3TXl6qPVWTCHMa1wkfb7OkfuwXREognLvO-3qdRgxinodvKyHn9XbsUcQMQjPPFMLOs4wpEhTJpcIFPqtR6tArjTT3P-T21mc8B56K1wXfEDvpU64XQ0HnfZWaqS1TbDyfL2i12ddhhnxbCV-0f3lUGnZVsfeGEc4FlST7iqUguhwPGb4mBpjBVFu2dv3DMCIPHew1v92gZH1OJqZJJVDUpu0vvFGTqxHz31LSX6lWa4gn2l6hvkT1e4aXkjHg93iy0ZXMpB0JqJbbWseZY0LDYzpH9noHq626Q9H4ZEKPo_MYBWSS_yH-V2_cN6a4HarqhcRwD9oT1QJ4_4AzWeFIrCZlClYbA-84H1CbBfQjgtRh6zTZLDHM2In2M8mKGyFSfeIhMHIcfPBTpG4flLBmTNrwwbuOP-0ss_bb5gxLeDsgU5xjwfaUzOWXudPJOEorz4t6Oc88MiRH42troV2fun6Uf7e7j1OQSGtTQ1kXf0rroz2ykDfVIXCefX_3io_xJ7ev9dH54CNlARSF6cVpTqzbyLWkA0BJeAVYcX2JW_AT-9VYTOo1Vixja7KtMAmMMk1E08japeGnoAd_a_4-bEfklFTChseUDgZhOt5_XtBiuQdPvJDorSQWQl8VCPKdMATr-EdUiZN54GSM46pdBr6p-Dg7LvB-zBAbTlm_6SET0O0k4RkkHxUCtgRMZQ52aC4brcym771djtWC-BbaR5CefibOoSo-i-BP2Zf-RVaS_MuFar0dT03zXdb0XuC2vuhbVPPF-7gsJez2dufEiU9LBhV3__zTDlFc-rGwwf04Fh5KuleNzr1QNyVPH9GZSS8jZkja6EcRfGn0X-oBr2oRLyxuL5vWgOdPadBOJGjIoRnMhCAxGla_gD_5m0qwF9CtWWv7ugW7YpATe62zE0O1icYDPwaXGovzTOeRDRn4BfJzgzwLRkP3-zOgF_09X41umrq0TCnCujXe-JOhFuIcYx8IxOb_cCcfGRqGXeZYP7z',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0b385a0fdc82fd920068c4ab0996c08197a1adfce3593080f0',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9463, cache_read_tokens=8320, output_tokens=660, details={'reasoning_tokens': 512}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 23, 21, 23, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0b385a0fdc82fd920068c4aaf3ced88197a88711e356b032c4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert result.output == snapshot(IsStr())


async def test_openai_responses_model_web_search_tool_with_allowed_domains(
    allow_model_requests: None, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool(allowed_domains=['wikipedia.org'])],
    )

    result = await agent.run('Search Google for the current population of Tokyo prefecture. Give me just the number.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Search Google for the current population of Tokyo prefecture. Give me just the number.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id=IsStr(),
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={
                            'query': 'Tokyo population prefecture current site:wikipedia.org',
                            'type': 'search',
                            'queries': [
                                'Tokyo population prefecture current site:wikipedia.org',
                                'Tokyo Metropolis population site:wikipedia.org',
                                'Demographics of Tokyo population site:wikipedia.org',
                            ],
                        },
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id=IsStr(),
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={
                            'query': 'Tokyo Metropolis population 2025 site:wikipedia.org',
                            'type': 'search',
                            'queries': [
                                'Tokyo Metropolis population 2025 site:wikipedia.org',
                                'Tokyo Prefecture population site:wikipedia.org',
                            ],
                        },
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id=IsStr(),
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={
                            'type': 'open_page',
                            'url': 'https://en.wikipedia.org/wiki/List_of_Japanese_prefectures_by_population',
                        },
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id=IsStr(),
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'type': 'open_page', 'url': 'https://en.wikipedia.org/wiki/Demographics_of_Tokyo'},
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id=IsStr(),
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(content='14195730', id=IsStr(), provider_name='openai'),
                ],
                usage=RequestUsage(input_tokens=22013, output_tokens=1737, details={'reasoning_tokens': 1728}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert result.output == snapshot('14195730')


async def test_openai_responses_model_web_search_tool_with_invalid_region(
    allow_model_requests: None, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool(user_location={'city': 'Salvador', 'region': 'BRLO'})],
    )

    result = await agent.run('What is the weather?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0b4f29854724a3120068c4ab0be6a08191b495f9009b885649',
                        signature='gAAAAABoxKsml4Y3hqqolEa8BSvPr6mIoOyAbWRJz9FeLHoqX03v4b6Kni2j9HxfifAm2cHD_m9-b2nOHcwDPOeJA28LQpl_BfOakn7h4saDElA_yz3WgVfy8ZN_oTLQz2ONqptBxdxCaLMGOADqBJ1tJ93B5s8bsFNZUdGXe382lPpCNX0aKPGxd0e-UBAICRmjGVnKd9cVzB8jhtQWBrITvMOLBvi6bE_TqnpXWf8-rhed78mFVMRweh6zAzukkJPMAjD7QfUAiODvD6oynwU6G04UOJoFTItUsAULPfyAw-YZqRwfcfMxoiLAE0rOOj9V7-eyp_J7DYu2uF16jaOopnrehFDJr-0pIGMFRxMSyFp7Ze7z3gWvcCOB4VwpSFao12nozedMeinybf71wo0750TNXXQ9Uye6qsUxxMamqcNiB02LjCM3nyBQ6FpWa59TD5O5UytT5FPOWSflYEhuiTFknt_JRHbKoeqVTfe_CTeSVlYBtiW8ouhkTHAAVI5lXi_mgvUMHINTYw5MEilzBSPunuMRquopRjt_07YMKuwPDQ8o__s1NlyrDAYKLA0gPzse4tWMkKREcfxuvU948pEJwVN9RuKS-NNXI2KiKKOAtPoXLbflAEtpx9N9PpPdwvz_z3yhF6S1_D_9P8OrSdxd8ldqvnqec75Jwt-a0fuQvRTSC3GsYuhk1Cb1aBvZdBtfcwBd2CXRuDUEdtzbLZ5AUNBy3f0mC3ITHG9aSpuD4GUHQDTjF_10-Qr4Rzygnj4-qubY5ibVxGtHlXkI0QzvGMVf7obhHMNxEQNaJ4k2dKddRJEhrSFWmAVYdWbKiZp-Dwx8veUSlpwMu8kLfGUq64MBQOApf-Srtry0eJAr3cTBqzmUIU5OOPg2C8j9SbAuTLbbcR6XeWizp5fbxdcVipVRqqp_PJptIJhaAUpHaaOB9u1nZbtlKWFJhJbrZzdktth5DNim4ayYBbBX1VAefwCugReld4C6QtB5Q-j_Tt3dug3Jh9TJmkhS8pJE4aHURzbCikFohJHAukZYgMY7wCuLWlahQ8snlIj8kbhPP-l-iH-e0xM2vFDF8rZnfYblnDLZYQBezfiZ4GtvO64SB5apQuRXkxExfZyBd6Kv-WhAxhPGoQdmTXfVEXePJLvbzAJcAXYpmmzt1STxoxR9cnaeLL13fFXZ4DGXe4j68-R7xCC52jfoV-l8JZjI0NDRJ3Mx1R26bp-lnvoertQBs1c18QHVShluHtH5c6V3j4yOMgG6cA2aVM25i6sjhUV3iltijuRv3E19ZlzgVTtrypeCVH7ab0PQ3Qki28mFI9s5M1z1TSuFis1qhHwf3r0kkmjLXIUbXAnfJkcv50tlcweXRTLKs0ZX0nxsxiZptBo95wxqBf4VaqfOY4NUNAWVoZ4AS5oSIgjGfUZtfrLisWmX8NjDWiOiENLmn9fCCq9nxDDsaucnwNhsMZo9jJqJS_99kryMXi0yGX4GManClCTe31Fj5zOrtRIezlEILiTla6fZwvD6vcl8GWO2wuyEY9zsEvfjyuvcU6Ernvw9S5HFPnQ-FnDxNtSTe1A8IHTspfEROnuSNVCMs6j02eFZMbXFKMaVi6LNDD2i7SYn3dMbN7aOfubtjeilMpIZ20U-J3uBUsc0rr8s4b-szDB1lkmiMvRDVY8YKNqH3iJFCToE3OibVwHeaUnMmEHJkIvJvBOX4hSwmAMxjZArusTnlYnLE2raAD707H_Q5JhpWXwtgFPj5ra6HFtOjtbPtDWrDn5_M180klxvF-JxfSxSl6U6y2FYeou36ttPRprWJynfcPSPY_sdrB9ZupHDR5zZy01Uby1J7XXOZt5an91kuHr0qU4bQJsq6AigFQ72C_YxpDNmQXcy5awJDBlXv9SoLiXRcTxpoXgii9alV8MeorRbc23O0fP_O6XKUso-lp-e7Q6bOqzV0c9K3imYUDzM9cqlvEyUGMDLlWzEvVGSwpag1CsLCNQ5bPc31W8hc-2WXrlltP6JZ9gYpcueL5AIud6RUTSJWg4Li6Th4ZGNs5cqh6Nk6oSu07P4Ie2JJ5bt1tAJbE4EupK3NVzUpzYzFdPrQkBY-VQ-klCFq4icnvlpD3pajYv9OoCpo0z8GfsdLeJlefIQ1NejuMg3EwbGRA_OEWn7sJzR2RFCYkt3YIuWRJb2UzIzvWhZsLxr4UpihrsieNKggGBh7nDpOXeAZhS8pGrNSlKjfvWtvmWG9NKXSpx79dNLSkumiD3FsQjk-L1Ov-K5WksY0yJTgc3ipgO2UpN9zolpXhXum9Uy8UeKLlB35cCtte15t_HSogTh2HDkc9SuCq4d3adSdstdXodr9jLbST50cHYn-F9qmkKiqV2nBzxW-9A4BB9WB_tWEoazKWYHtIdmjRm6O9NxvOxYuWIwhMmRf-OE6MHOeH0emhuTFaeuZ4zjbM0T9peRh9shiUw6T1NT0doCgfyRAq1NL1rG7iSc4jxrc5ahP0gN',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: Brazil, Bahia, Salvador', 'type': 'search'},
                        tool_call_id='ws_0b4f29854724a3120068c4ab0f070c8191b903ff534320cb64',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0b4f29854724a3120068c4ab0f070c8191b903ff534320cb64',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0b4f29854724a3120068c4ab1022f88191b2e3e726df9f635e',
                        signature='gAAAAABoxKsmZCctfduUbipds6REy8FkoOiADLcLER75WMHyO7PtQt26NIhcGkiXReZWucbDdEBRKk7_g9PUuu9g-zEBe6kIQwm4lwjxCGPy-rQdmfJpueznyPJ14Ood-wazqT9a8ab_BMFS7VLonsOjZR_b1gxcx5yO62oLvv1GnZfkEykIgRbGIBSYDWX6I55Sfwkf0JRaiOFgeHoOvQ6f2mdb5UetdJwbIFgRh9Bk-_l6goC-ONyElqvPxrh8zlLxqEhL-KtTVw6TPNW67QeYxekA4vdXseYT4W2oJMcMKp8aIfxYr3-ZWSy81UqGPD2PAfs1DoOYkWMHxt_VnZjLQs0qkO-JBPsWBFWEofZC1GxOIT6gd_dDvExBXkaFdNH7xf0OxsxSMWfyKSXMlq3kmVsDIN3hKImwfZQ171mkFEwgwIBeo4XiY58YJXmzyNXSs3c82gAeGpS8cOQw5shjC449uJZkixSaXmwOKwtm0z1MOVAp17QLGeD_2YVa-DZUy1z6xTqStuZWnwLDOz8HPL_rW3MXGcmC63kWmcTCsFngwR_IArcTd8lsRAXJghnEdOZDYgrU7uc7bbqO8W_PyzPDDnrAbcwo0InMqJ2BZErMXOmy2dEm1jlJPEn23PL26k8r_sKNCZpg-I-q8epjbF225NJ9S8g_vvqLsyCzo-WnHPHaFDMfUhRxU-ylSReCZO3pcjNJXAfmsNiBs3g272BtvNWn7GpDqlJL9aB9Erc79CpLghKKV9JiVRsr79aW-JSzn9gJET3JteU2MMCvRxv3ePPkmZUvQdKOmzZQMwQ8j0FQHd--4qMkXDdAz-lsUjitCKK0z2ES0oSnWOVVPoR5AVIUCSfg-yGwBWhKv6qIkMTsCYaCaR86j_hGlCSxNqYdbMwy7sr6nwqDmqgmcsiNkAVUAUeU7LLXmVfGDR9InNL3lNCICpmcHMd8YJO5A1wFMPHFgfXt3o4CZP1ZSjQjQuQ-Oh2AfLaAYSNbU4y8JDtKPiini_rWIqH1yykwV0Xt__QvQtj600ksUqij_zxbKnZKy_u3Ud5E04bNgTZ0Mq9ihVtPBlcDCtWSsp5U8Sm6JL0ZXV5XaT3CVG3T7Mj-kKs4yHHOLNoR2rKAGPTA6VRzaJDNO4goMeE7aIqWKhFTYMBcKJEGD-B2J2J36iZ2RNGo9JbxmUw4ZPMVaPPulSfpLvDptYEN3LX0D6L4Xu5iaW900EQ_Ym60siMB257NRxfVPb5Sg8hqxGeKKgg6NGa6y-qyVXvqjy4HA-ODvHLbiT2n75fTD_OE2CX1FpLgmpmKkSopjT5G1vv5qtXqdhigDy-l_b9Qxwvbd7XXD72EUVPzDVwMDBZNeJkylcCecaRVJZRnhmOMkGbV4WFrMxjy7eoYrIBQ6zytutBFXNkAb6a6UXdTrlOlzclPP4P81sp3J6BytVSaLJXCIpZ3pAM9aWVzfavRW22R-rIMbmCWT9hq-1ZDfjdglHN7yowAF_rjVGrgl02wsh8IlLKfJreh7ughi9vSk1WMinlsiZfZynp33IfB3ayv00a_huU4oSKXstf1KaeQ1Z8L-ReCdPRwDYaLbP1ZT7BQAbXKgIjUsLdSiU3MmW8FVBdevLQq8AUUKsXxfQLS4TsjMYTNZ_8LkMcVeuwTDQTBYkBdyTl7jawXy2jujxDJe5mK3ZvvS_70sWokuPXkCApVFkJpNRDdcvBuoLG3g_KZ7dA0oQW9QHkKpd_-FEuUZFnL6-ZhjR7pe-EmR6gqJbuQVs19N2qho2pnNEe21WqAN-anBb4H7QN2V1ODJkW6vDDRH5sV8Ya7YYUScSI3TUASWH3MWapL1_-lRiXtVIM9Q8leFFIO_qkr8DFXoDOHp29HNa3gpQkjOqAFqX0VLg1Ub6X6C-kUbXWMcYIUoKNvQx5-Yhy5Lo0N6izxdE4Zw6U6Lfu90rA2DWeQ5-iae79H9yUy74jZw3bclkJFzGkydXWIP4OkKnDPemIKmsh28ovmfgtz_gJ99SlQDBmI6paH6P8wmHd7QvDQkMBnuACOnTnTud_MqdNUR4-qtcnPoNkFPXoTfYJNDDBkxvaEIXylqKK0wPf9aBsICsvB0N96nPpQTYuV2YHfIr8PagOi8wWC9ceUmDib8fMq3xgClujOcXOPk2Hh4Xuslecn315m-SoLjRg-dIdmTjuIyT9CrSdXMto5Jp7vcPTsRPebw41Tf4iR78BOTuGhbe_B7_WDm5FH10EptF1e3GZ0eO--VdgqLY3T3ivuoxtXIkTvDHvLHqNwFJIvH4ULUAIx3UGqJwE84_OqGwKBRT4UuQRm5wwZUZ0teyzOQx0cp7aKhsOkBzKY8jVFMmTBKin52ioD1inMiyBUYICYwYUngdYRmE5Qx7qzqB6Mg5CSW_7TaXuZFNVuVnitQp5uw2RrOlookLqyKYIQhruNjaUAvvDnhhIrTjh_Bi7f-wv7znhbJDE7YWy_zC_ufQj9VfxJcz6eXKu3fXr4EKlLayk2nwO5BkwaijetPdBNs4SOroEo6WfvFgVtbt-c6kkEfY5abo5zK6OPVHrpBVyew-A53SA0bQNptBVMNkZDiPczaviF3H3fnkMQH59RhIhMV9knjfCbAhP5BTmBFyFIXjX_ErOJgb3RtUObwjnifMNwN2hIE_-eMqk8K-jxMrT7xNoojwqcCgmzcY5w8hbmA77xW4ZnlBuTZORjFhppokfhLPcoVCcbt1AEWLc3oFYhquugqG9WZbS_7p_pI8C_zB4Q4x8MTn7lO9RZFufBeI9iTm6JP95asBuEafpQxP91ZAhfiU93UybWsoaKQb78PvjqwwK2D-LRumK6ftSMU3LNn1MBmiFowwzOLPxrkN4dzqF89rXEXJCuqS3jl9fEwKOdCvhpXyVRN6Kx5VBxSrY8KO9ItwWkrjHF4cWCTRVNePbw92TzRnzgLB4aEZ9T5TkIvdNgOyCQYSaOZ1TMSgO3a-i03avh9KisZcyt-gUbD11-EJmt_KOSeK5o-Jn3GmUKnZJJX9hKCOWCmN00qv8DzYCfIO9Bd6kfOXAqJJ0RFDHn6a4VHv4NrZNyXQWrX12_V3H4oHVZhDurhlhhak-6xoSC6KWeHFFlU39xzKx-2BfggTfghpTj4x8WiObhHvg7I6OY67vzfyRtJoA4muFzqq0c-RJ1QMvOXLGDEMJMSmuXxT0GOux0GvkB6VB4snKw5ZWdzTdm-maT6LBL9POZ8f2psW9CtE9tuzs1EfrBS9SHn9s_B6NHRCahEwwaIRFePU0v9mT3hhQoq_CawOykzNVGAPPAKyA8PNZr5GGmdmV7v0fWppgHUZA_sQPbq0XuxgoQFLJttwnCEf_mkS1zPYMYBv16U9G-kZQ25-rdHBFyZG-Wa6nBCSk7lm6ZNkDKSN7L-lBAVgpPgzDvXlCHaklZmQXwtNnBSPOZ3yO2-MBcDmSyoDbXpdM0zYZhMCyv0vMf2mKhEP91a2xD4tsp-Og6gAo0AXgk6Ge_be4zhMaUxm_NdPGg65mkaSaOZqCuevYVh0En18B7x2erzzUAMuJoo5C8ab1yLVGZSKNda3z8j40JeqcaYLN-yS4RaGaNdva_pmCq0dXYadIjaoivy4TqnHig9uJtboQqBevHPq2xXdsSutQOyEEexxjYbEz1USu25bTvog4tJs5okxNWDnL_0vBXZTpYCGdVo2WcMJgwqNBp-CPoZjMxCQ9IM6iS3KKETc9U46ksBbN95ZSeRUoUUtO_i0AoBsxE9A4NFbK9Uox2RGcJxOlC9HM2n5D6LmOyIO5KaYl16sfmURTRlcNpgTYAvat5HbfDYMFrH9EgSxu0y735-2wvZSuD0credILM3XFTyBmM7-278If-6-QaDX7zV9JxJaXrXx92T-srNH2Z5DLBOJDkl7oo1lVGKcFAmEgHjnkT_rPt8DvU4tlh0eI8HzSe7B35oA02GJE17hiWk-_VOUG2zNaOaesGK437EOzcCcc1dMZAtN206qPtzDZsNPhQNEBUx9Ta_jPG6waGpwihNxVfhwVvrR0zFUy1IspR9B1ONXttsi7nQ0YAtDSJaBuUgwwtYk2KL4QqRAixv_KSma8mOfuxs0th-sTyFGQ5f77q71ZcLUeYqVqrsjcDsh0K9pDvj4-KXcQXgd6EzY8zfh7VvXOHIr2aHBcHk1tw9zjYAR19sP87lo7YdVNrYlB09IkCICT9N1RSWJHUsszCvP0oBSmdNPfelx1CvHlClrc2qNGcyalsF8hc4wnG3mrYIC0rb4sHLc6Xp47g7vWnXH1ud169K4dB5YwnLam08lPwSYJwqculJw5d_L2egSoNIdYGvlvH-4prN6EkkyiqmZCHXYSNoKorU-ce7cRpc6mbxxU6CLCS_1FhlgfG_mZFP-KAZ3b-lQVdimYcudQeCgtjaydeAcUP4raEP_Wa3bhMB-GK90eskPs0cZgeRDvwohATR8ynHvxFCAeoiQcL-3bQgdOhZxY6r8dn6HF3RWWaeA6o4xS0XTlxecl4rOXs4nJAvn3jGZ4VmU9qkYcoVBW44IkLnbx0q07n4rRiurI4596rknVRJwbeb--_d9l9gSqn_ZwIHHyO4tk9np7I8yMTGp0j3ea_GbKrss2_8gU-XDU57ihgCQyOrAcyyfljyHTE6m-upNK0glJ-2m9r0ktOToCN-6ve4H3trSNvRL26rmH_WV8d-gwsF76cPYdlCZu46pC3Ib_R4sHUeBjg39ilY0IxUTOsLz-34NuMeKKnaViX68pZw1XzMLb7ZJOYhe0AKKO4Yrrkwpwlqvbpgd369PENtcqdakdbn44wKOfp49d9czQYQcYlRK3L08MhGsHXuDTlUcqqEYSDpwM_D2__AicfRazviJzdWQQMNJHA_0COIuhQ4c0dbPOOZqCMM9BxQe69fNlTfZEpFL2Axh_6-TqEXdqU8CO2fYScvQfuXZ2AMbmit46qlhUJMj5082R_XYNwIR_b-QMqm0e6aI_vZRVw8MwdJHG73Z_u4whBIR36VHrrK1qUYLxC2pYyLOwHlPEYlyN7HlTs6i_iJ9z4TQuK_mk_b1bc4-1XfgQUU8ZfjYPNoQNII_Dtym-9k7Ukv-pU5Nk1lItlLk07wiCcKMlui8Y-23K9mb03O38x9ZhN051SusVM9ItehAp684sy-kb6MymRW0LsXXIPdRc9LxI85RZ3aANfAtMaHbRov2jpVvZT4OQhTQIJLg3656y_NG32DJvFQoBLEgfFCTKYQgpKWmbxj1gRsVDrdk8EBF3rz1ohyUfxqyrHSYM39YGs2bnk9TkvaOaHOluV_ZoY-qIDysJ_p1eKxJVdpF2VCxZ1ctwuKCbVx6pl6XLuN-g2KaJnpgxVcVbrnxsgLrh5OGeDuXiBFYeLYaF09wFBHTHF0naw63TgB8jy61c5r7_y4DVAiicoSJ3B8SJxEmB5qgXVse_vwmKOxvULXcgU9XLaONbYYIUulkSNOSK_x_xWnVRL7yWHj9xMjWTvBXgVcux1CmehPPQ7dGhooXgzCoipDZ_y_sRl43wYZiaqG7Nl79ciyfdwi6xKUb0CgLQp1D2Q90bHKRUV1Y1IdcIUl-atTUcMGYDyLKmYQQ0BWvqXeaZtHra_yDzoIlB7rR9Hg9agchVJsUA46egTwwvlHdiYPIxJidKAQFgpDospYReegQxCIZHg_PI0FPVfXBfNR2Vc8fIrXiNwzPi4jvj83YmDTvTJ1xBLYDao7QzDQUjkpl09EnP4UoGlvFYlrXH0Ev1sWz_svhFVAduqJzHke7BW5b7gYipmIqQCvPgehCMuD8-NkaEAtE613V6BLPTu51IPtkvFoS_zSRCkLnspDFVTeDToBKQlN0-u1LlMF9f1dQDPxBE8ZLacKFP2F6lezHhikzuoJTyfCzF0xT4nn8alqzDzRV3K0wAl_4NKjhwSHz9i8MRxPo1WEfO8Xpt1aKa6WIbZ2rr5ayhX3H4ASPQ7UDoMNrRZP82lcAerRb_j7wyL57W6oE7VetxnmbexD15h_7LukUqUNSSgg6D0zxX2C23EhpBaQ7Bw4Va_costesVZBuYwEig3VR5Y-9WvmN0CuaeE1oZkXJ5zBCBgO5F_hIESxHP9zx9Z4fs7fswQDJHaick1xpSSZNDbBghUqlswGvI4TTtUWGPc5R1mf9dLQDF6j5wTo1kycMpfXIUF6hVqZRlKHgP4DRetOCsAgb_WMW0b_GCVyK8JyeZsTSXN547g8Q6WMRYikbZDP25hglrI5hU03GLf3m2WLJAd4eKB5e1nlDhIqAGn289gdttwfe8rUzB5BhdSZ6BcaWAEVp64EHYFmtco1aBleXa0RVlSDS6gt7U7ozAp0YxkBW7YlqXxfM8A8y-Dn8LkKewv5p7q7yL5Bkun5Cy7rZ_FPQ_4ktHUr_RzqpQbgSgtXwOSyCfoDKqIPNg4AhjaI33nD93HuRQeV_mhxYwXN5GNTq-7SxkulMwTSgg7b2UhmOSu87pX_FMk5nFaglzYzHKpoZA3QuNxwHzTVInF8Ufu6fAIOPT5fEuhfilDU3uxCkpC-us4yeLwm8e36ICJZFfcqa5dXHkFezEXPKvFbhpVgjTO-TI2EH_vb4QcYNQxtQGWUqFcuQ7IaIgYChVS7ifjkPc65wR9ffjTEEqFAt6e-_mviI4ltyiTLTNTWY68JV64SnjeMQ9qR9gPYmefUp_E_LyOdwfetRYKBJ81jAMz2piWNoJHwHbFjBxeZj8iZ34TnirgvWRltUi20aN09b8TN_IbFNPFjkI1UwshqMwLY9GXT4eq0QaIdvhW9CE90--KNVjGvqyRLodo0gsGTpmTcoTPDgF_AuaeDlaBrbAnW-pFr1HOV5YqUGja5_vkDvi9mdKooFrlSau-Dt1HmZf81izJ8odFR-tHl0u-wT66G0aEkk1DS81IXvSLLNAQlIpj5FoZYx2RPFWyw1WBlY8iSa4r6HyN5YKW9taJ7ljUliA8KClax8VM282lqYL5Fd-wtYu5Iceez8jGGj4cZ7JetWp6X-wjLHeo6SDUGjNO7k7h3ODmCRnIKJZVtbx6qJEVX1u8J9mIAXEjdArqa_7YiUBTuka0W7IxVXZUx9R96h5f',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0b4f29854724a3120068c4ab22122081918f25e06f1368274e',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9939, cache_read_tokens=8320, output_tokens=1610, details={'reasoning_tokens': 1344}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 23, 21, 47, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0b4f29854724a3120068c4ab0b660081919707b95b47552782',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_model_web_search_tool_stream(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool()],
        model_settings=OpenAIResponsesModelSettings(openai_include_web_search_sources=True),
    )

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the weather in San Francisco today?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_00a60507bf41223d0068c9d2fc927081a088e0b920cdfe3866',
                        signature='gAAAAABoydMADQ6HaJB8mYQXlwd-4MrCfzmKqMHXUnSAXWV3huK1UrU1h3Do3pbK4bcD4BAvNiHTH-Pn27MGZDP_53IhKj_vB0egVf6Z_Y2uFPtzmyasYtTzrTkGSfAMR0xfI4wJk99aatk3UyLPNE7EO_vWYzN6CSX5ifJNNcmY3ArW1A7XnmsnMSBys05PsWqLMHZOUFuBvM2W37QUW6QOfBXZy0TamoO5UknNUfZb_TwvSnMEDpa-lXyDn4VuzfxreEGVHGdSyz5oLN0nBr3KwHIfxMRZIf9gi9-hKCnxX7i-ZktNIfTgd_WEmNKlaPO-qjKHPlO_XPKbEfpBdMv5b2P9BIC20ZG3m6qnEc4OqafWZa1iC2szi4eKOEa6neh2ltVLsLS3MlurF4sO-EHQT4O9t-zJ-738mZsOgjsI9rTrLm_aTAJrntSSWRLcP6PI6_ILHyeAl_aN4svtnwQJZhv4_Qf62q70SZQ5fSfqoqfO1YHLcXq6Op99iH3CfAhOjH-NcgThFLpT4-VLYABl8wiWBTsWzdndZoPmvMLEOaEGJOcM6_922FC0Q-fUio3psm_pLcElaG-XIkyn4oNuk6OJQonFE-Bm6WS_1I9sMF0ncSD4gH1Ey-5y2Ayxi3Kb3XWjFvs1RKW17KFXj8sthF3vY5WHUeRKA14WtN-cHsi4lXBFYJmn2FiD3CmV-_4ErzXH8sIMJrDDsqfCoiSbHwih25INTTIj7KAPL2QtIpU6A8zbzQIK-GOKqb0n4wGeOIyf7J4C2-5jhmlF2a6HUApFXZsRcD8e3X1WqSjdTdnRu_0GzDuHhPghRQJ3DHfGwDvoZy6UK55zb2MaxpNyMHT149sMwUWkCVg0BruxnOUfziuURWhT-VJWzv5mr3Z765TFB1PfHJhznKPFiZN0MTStVtqKQlOe8nkwLevCgZY4oT1Mysg7YJhcWtkquKILXe-y6luJBHzUy_aFAgFliUbcrOhkoBk5olAbSz8Y4sSz5vWugYA1kwlIofnRm4sPcvoIXgUD_SGGI3QNsQyRWQEhf7G5mNRrxmLhZZLXAcBAzkw10nEjRfew2Fri7bdvyzJ1OS_af9fHmeqCZG5ievKIX6keUkIYQo_qm4FQFkXZSl9lMHsUSF-di4F6ws31vM0zVLMmH52u12Z3SZhvAFzIV5Vtyt_IfrMV3ANMqVF4SmS4k2qUlv1KuPQVgqGCVHvfeE1oSyYgYF6oFX8ThXNB79wxvi4Oo8fWEZLzZMFH9QEr2c7sOWHYWk-wUMP1auXTQNExEVz22pBxueZGZhRyLdpcA12v8o6vJkVuBj-2eR8GRI7P6InJdQAO9TIBhM7NtJU2NUpeP_84js3RTBVktqBT74nWPaHIddGMSfW2aGmFJovvshhxGMLtN_6XMh4wRKW0IE_-Rfbhk8_-xHKI5McYI048N_TMYOS8KqPPAmGVklRGqPZ5xXMNvQEVweThDTYTo3NoAsS0fN2yMmSwrjRYBHsgYMtil4pd6ddp8dvF_XSJUkW0nF8t6ciI_k47sug3gyw4usqspWxY9Hwbzb4OFzzrgtO_7Ll6lFFFUx2oHy8AO9sJ97Y3Fg6luuew7ZRDzA_4XMrT7mNW6YuT-o2DunaZw-jvQezNHjPN2WhaTS7fkisyhFSFTMBYE-H4psfj_sizutv-LjwbumTcX2mnYE9SZhVr8dL0c7sgwHP1831RxTSSl3ql_obE3ICDooyuM8PYE56Jx0HOOGbEeJd3w91SzNHPG_3SQfXszrZlw4BGWrEUHBbtVY2ZEnsyGNAx6vKO8lz9D-6yZ618foDJSH-Ilk56a5rhr0beWjSd9mYMsr3zpVz6HcpTLYGEgHfPxpT2eaYaC1H_znw7y1eMKamwudYmtz_azX5LrOtwc0p-pXH-kdoNe248pSz9qsmHcXA41fuj2weKQNrmBcghwtfM95B060tnmebJ_B_KkLXL4cNF-hZqi0wAHrHYrZ_WM0Dy90AFH-b7iiWuWz5M1EhZXo179iEdybM-1PgccFJ0zvOqODl7FNxSgWVyNS1k9R42aZx2PzFAfAbBtJ-KVMhUayAvGLNmi35EAT0G6FK65VBEe7A6zPFqzrrAiG8dy3Z0I0253WzIblHPNMpmxI_ca5tIx3u8Za6Nu9rx8mi0CY2jsRSKnqb7RZvLuB78Uj32lb_9jbq5_gL9_y7Bt7U7i7FospyqMFzEYQLvdyrtfNrfY0rB4zr4Mo0tDn_4YOD_d_nP5axUh9_ruqXZ_d3eVdNmlITjQZj8ALe1EfidP8a-Dl62t6STVv8d2y8v9-jy3J7wReLJbJ6gDDnygJllY7NrIVXSjR45FXiCDnpaRonu--I_0b_LRJFOoJUJX0S9YMaXAkKyHSEj-UWjiuk8cIBNcXxwlxnqqNMezvvV113MAOEbfHygDnphzjzZQxteAVbSy0ucGDR2FPi30d6z51NxGnXNS_sM7wnjBMNp4Li0hhttOp6PgvDKPSMAcgUtKLFKE8iWQAvERoUVxw5Et20hNTNXf_0sXOyh0bF0URPGDxSYz9uZI6-nlwVlo1aobdEnn7STSq2_tuTDIrQyfBGZzhv8OB0H3cj9mBs=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: San Francisco, CA', 'type': 'search'},
                        tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'sources': [{'type': 'api', 'name': 'oai-weather'}],
                            'status': 'completed',
                        },
                        tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_00a60507bf41223d0068c9d300b23481a0b77a03d911213220',
                        signature='gAAAAABoydMLww_4DcIPiCy5zW1t-Dtx57JodsdP9YkyDCvY9r0nBfoD-bwcBl8FfcFxRuq5nK5ndf90J6Xdxmvl9kGVbCUYSFaOd-kWeE4tgwXM8BwwE3jVs6ZMG3BdiWyXS3alnUO5jcE6kuXeun1LufAdZ6rWtJl3zSmqPycnTh9zoQ4cBBxLDq_qcVS1fgU4WjsWgjCZw6ZWRPWwZk8hywmU7ykCLH7SXItM4oH1m_GCEounJAS8YR4ajUh5KAdN6a1wngfGnXhYdzov98yiNLP6Nrkgr--K4vMTTqWXLTcR6fbWgkijanIeKmfSErCjMT6k5TrAbkFx0rgblHbdQii7zj8seV1BWZse92_k4sltxfc5Ocpyho1YSHhgxyGL7g442xMUEibjPCv6kwPMcW9yGu9wPMWsfPYCXpBbG6kQibQPNFJ_bEubwBRaqdSDq93Aqr1YkTYBja7Tewn8UfzZ8YYaGe5y_K4ZD47lfvDp019dOdXmOuZGC1ECRrMqKzSFYVG1CFY1VhjGdPmzobDoMcpZcLn25s1pg6lnNqNQwOk_IA4MvUcCU5HHD5YjmFkEy5-i_iRoDVu5coK0zyEMvPJ_h10y_ByszcfzS9e0ht5CSilckkFdxTBkZ5epp0YIg1e-PrZ790P-I35Ucquam9OXyULV1Y5bn9ohZa93Tv0JZRxUeTDG72_28xRj8tkJaBAZjoCC7VICw39KVmz-ZkuVN6IIX1WdNzyC4d808-2Tz4UZaU42-wxEWDnSDMD7iZu1Bi9fKKwAYBJt_OcEsJwpW63ZaUSG2PVFfm7a3wRcSMxMTUTTJB7L1Keu1hmNepif5tavn3P35nSq28D_IJyAqAgX7ZyROk2bJqjzSE4A0MddqAoBFFqKBi68n49KH09vDtDXIoh8jVWuIgowgVGr8pN3kuhLI9cir4Pr_WES0tPD7yWHPTzrD7OIJCfQbr_4Y4dEza4ixNi0RTADWzMUZBfr7bvwIsgvg6ZNuQlx_d71Go5VDsT2KI8H8AldiRvNWoLyYTFGyK9Kot97YsS5sEmSYgNAH48NU7pgnM0jNDQU1G39nTNFEjL_ziDwjDT5g3jm4S_gbQfwx-XFT3Pv-JYR-E71AqR--Lg71OsASq49rrlULfl5OENfiT-NB6x8MqnfUI6NpcCsOWLp8XfRbgqmZFutLIi43pcnxEe3cXHLWGF77qJXP6dFb-G5Ide7n9tAOoEgfsVu7hCDPEQ_xrIYRdc2DzDPUMCtXBai24E0AnQF8kxsEtlDW_YmAgGNTl9Gx0tFSGdDuUCsNx__c7v-_LOMWycXUKmH3iEr_su83oGIMapNp2PnLccN4iOxspdZQq0C6WBaR6SrdnGzK-0KwRPRoyKDLNWS8zfluR5bIgKlqd3Sbv_7eL-WO4LQXMvdKP3KS-DBt1HbA-gmyFW03iX2smPQbtVmRLWi1vG329R_07-tHMJSO9OQy6_6aiyO8Rgpbl_CHa1Q9BEkI2csonayDJRPvEXBPuk9-NPUP4VLNPB7npWBLlAqes5ZmhagnC7srTL0fFiLGLJiAxWo1f0BBiIlXjwqHdlgBjTw0KryCnEU8Ic8ATzrqEXXhs-FTBCcWInf3Bt5bzUhy20g7cTtYP-VCbsku-lXQ6wceWrfQVFtjKKICD8I4g9QusAIAvgCUm7J2rR3TLkzwOKngdTFPGQrQ1TYzlkA7q_Ew1uZpaPRckMaEioZYC6Sv_B0rgW0nyBJ0GLrB3AUN60hDrOFntyFHp0FM-Zh1SY-GKGBwZwVetOzM0ZAJ-NreFg1XVgyLTYDNjUrYJjRhr_JARsZ5t0pU4_yI6dPqM5jKO5_k4UpZspfQon6d2-NlWX0EDmz6G4CMTx0TScehYHrQZtPzpVnivc8h_pmXV3jO5GLzNeLWoB70SDPTETo1Of4txiEUaC2komu5B7MN9aR4c7VBOTv1NIjoiZcrd1HFACzZ7r1qAE-G38j1f1YhfZ0_TiMmtfR1cqjAKcFkyRM7rZMyMvvnsH7NFq59gFgWZt0dy0aAdw03XWXFNT67lrw58OYC3NcVozH4SKlmleu7TfjHNWSnJVjJ66riLn9DZWVxPeTk4zuISZn0yyaoXcdW8OMn_mJ9vP-8L1wElMyxKbtBRz-0cW7MshmJ3YXmHWDKbnqETSbDMtqcN_QyRJovopwlptJ8VzL7biuURRFw-l63Kc9vKP72Z-QWOUIPLB4q4nX4yb-IV0mkWFxIUlfv5Cze2anf7zDFyGzeU9xG0onfhJE4HFKcoUT8MzfrHZ0dDZtnEYeL5Xem3GuHpwEVGCxRE_J1joTmJfeWxSVnr2Vey9gaPmXCyRrdKS75v9xSXJFfHvcOO8Qp35Dzk-yFqL3dSOJfOEwDZbEf6QnV7VU1EhJvW4XmRS-wsRLMLCYcLrOx96NHEwb2h2l6gNfbCVJoQrMhMg68qBPnoSYLhML2ho7hWkSNZFy61yX5I-oEJV5XdtjFcBkyurmUD6uYTkJSqXyxLexQiPbT-uv49Yp9cAfFBG23sC9lUQ=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='San Francisco weather today (Tuesday, September 16, 2025): Mostly sunny and pleasant. Current conditions around 71°F; expected high near 73°F and low around 58°F. A light jacket is useful for the cooler evening. ',
                        id='msg_00a60507bf41223d0068c9d30b055481a0b0ee28a021919c94',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9463,
                    cache_read_tokens=8320,
                    output_tokens=582,
                    details={'reasoning_tokens': 512},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 16, 21, 13, 32, tzinfo=timezone.utc),
                },
                provider_response_id='resp_00a60507bf41223d0068c9d2fbf93481a0ba2a7796ae2cab4c',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_00a60507bf41223d0068c9d2fc927081a088e0b920cdfe3866',
                    signature='gAAAAABoydMADQ6HaJB8mYQXlwd-4MrCfzmKqMHXUnSAXWV3huK1UrU1h3Do3pbK4bcD4BAvNiHTH-Pn27MGZDP_53IhKj_vB0egVf6Z_Y2uFPtzmyasYtTzrTkGSfAMR0xfI4wJk99aatk3UyLPNE7EO_vWYzN6CSX5ifJNNcmY3ArW1A7XnmsnMSBys05PsWqLMHZOUFuBvM2W37QUW6QOfBXZy0TamoO5UknNUfZb_TwvSnMEDpa-lXyDn4VuzfxreEGVHGdSyz5oLN0nBr3KwHIfxMRZIf9gi9-hKCnxX7i-ZktNIfTgd_WEmNKlaPO-qjKHPlO_XPKbEfpBdMv5b2P9BIC20ZG3m6qnEc4OqafWZa1iC2szi4eKOEa6neh2ltVLsLS3MlurF4sO-EHQT4O9t-zJ-738mZsOgjsI9rTrLm_aTAJrntSSWRLcP6PI6_ILHyeAl_aN4svtnwQJZhv4_Qf62q70SZQ5fSfqoqfO1YHLcXq6Op99iH3CfAhOjH-NcgThFLpT4-VLYABl8wiWBTsWzdndZoPmvMLEOaEGJOcM6_922FC0Q-fUio3psm_pLcElaG-XIkyn4oNuk6OJQonFE-Bm6WS_1I9sMF0ncSD4gH1Ey-5y2Ayxi3Kb3XWjFvs1RKW17KFXj8sthF3vY5WHUeRKA14WtN-cHsi4lXBFYJmn2FiD3CmV-_4ErzXH8sIMJrDDsqfCoiSbHwih25INTTIj7KAPL2QtIpU6A8zbzQIK-GOKqb0n4wGeOIyf7J4C2-5jhmlF2a6HUApFXZsRcD8e3X1WqSjdTdnRu_0GzDuHhPghRQJ3DHfGwDvoZy6UK55zb2MaxpNyMHT149sMwUWkCVg0BruxnOUfziuURWhT-VJWzv5mr3Z765TFB1PfHJhznKPFiZN0MTStVtqKQlOe8nkwLevCgZY4oT1Mysg7YJhcWtkquKILXe-y6luJBHzUy_aFAgFliUbcrOhkoBk5olAbSz8Y4sSz5vWugYA1kwlIofnRm4sPcvoIXgUD_SGGI3QNsQyRWQEhf7G5mNRrxmLhZZLXAcBAzkw10nEjRfew2Fri7bdvyzJ1OS_af9fHmeqCZG5ievKIX6keUkIYQo_qm4FQFkXZSl9lMHsUSF-di4F6ws31vM0zVLMmH52u12Z3SZhvAFzIV5Vtyt_IfrMV3ANMqVF4SmS4k2qUlv1KuPQVgqGCVHvfeE1oSyYgYF6oFX8ThXNB79wxvi4Oo8fWEZLzZMFH9QEr2c7sOWHYWk-wUMP1auXTQNExEVz22pBxueZGZhRyLdpcA12v8o6vJkVuBj-2eR8GRI7P6InJdQAO9TIBhM7NtJU2NUpeP_84js3RTBVktqBT74nWPaHIddGMSfW2aGmFJovvshhxGMLtN_6XMh4wRKW0IE_-Rfbhk8_-xHKI5McYI048N_TMYOS8KqPPAmGVklRGqPZ5xXMNvQEVweThDTYTo3NoAsS0fN2yMmSwrjRYBHsgYMtil4pd6ddp8dvF_XSJUkW0nF8t6ciI_k47sug3gyw4usqspWxY9Hwbzb4OFzzrgtO_7Ll6lFFFUx2oHy8AO9sJ97Y3Fg6luuew7ZRDzA_4XMrT7mNW6YuT-o2DunaZw-jvQezNHjPN2WhaTS7fkisyhFSFTMBYE-H4psfj_sizutv-LjwbumTcX2mnYE9SZhVr8dL0c7sgwHP1831RxTSSl3ql_obE3ICDooyuM8PYE56Jx0HOOGbEeJd3w91SzNHPG_3SQfXszrZlw4BGWrEUHBbtVY2ZEnsyGNAx6vKO8lz9D-6yZ618foDJSH-Ilk56a5rhr0beWjSd9mYMsr3zpVz6HcpTLYGEgHfPxpT2eaYaC1H_znw7y1eMKamwudYmtz_azX5LrOtwc0p-pXH-kdoNe248pSz9qsmHcXA41fuj2weKQNrmBcghwtfM95B060tnmebJ_B_KkLXL4cNF-hZqi0wAHrHYrZ_WM0Dy90AFH-b7iiWuWz5M1EhZXo179iEdybM-1PgccFJ0zvOqODl7FNxSgWVyNS1k9R42aZx2PzFAfAbBtJ-KVMhUayAvGLNmi35EAT0G6FK65VBEe7A6zPFqzrrAiG8dy3Z0I0253WzIblHPNMpmxI_ca5tIx3u8Za6Nu9rx8mi0CY2jsRSKnqb7RZvLuB78Uj32lb_9jbq5_gL9_y7Bt7U7i7FospyqMFzEYQLvdyrtfNrfY0rB4zr4Mo0tDn_4YOD_d_nP5axUh9_ruqXZ_d3eVdNmlITjQZj8ALe1EfidP8a-Dl62t6STVv8d2y8v9-jy3J7wReLJbJ6gDDnygJllY7NrIVXSjR45FXiCDnpaRonu--I_0b_LRJFOoJUJX0S9YMaXAkKyHSEj-UWjiuk8cIBNcXxwlxnqqNMezvvV113MAOEbfHygDnphzjzZQxteAVbSy0ucGDR2FPi30d6z51NxGnXNS_sM7wnjBMNp4Li0hhttOp6PgvDKPSMAcgUtKLFKE8iWQAvERoUVxw5Et20hNTNXf_0sXOyh0bF0URPGDxSYz9uZI6-nlwVlo1aobdEnn7STSq2_tuTDIrQyfBGZzhv8OB0H3cj9mBs=',
                    provider_name='openai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_00a60507bf41223d0068c9d2fc927081a088e0b920cdfe3866',
                    signature='gAAAAABoydMADQ6HaJB8mYQXlwd-4MrCfzmKqMHXUnSAXWV3huK1UrU1h3Do3pbK4bcD4BAvNiHTH-Pn27MGZDP_53IhKj_vB0egVf6Z_Y2uFPtzmyasYtTzrTkGSfAMR0xfI4wJk99aatk3UyLPNE7EO_vWYzN6CSX5ifJNNcmY3ArW1A7XnmsnMSBys05PsWqLMHZOUFuBvM2W37QUW6QOfBXZy0TamoO5UknNUfZb_TwvSnMEDpa-lXyDn4VuzfxreEGVHGdSyz5oLN0nBr3KwHIfxMRZIf9gi9-hKCnxX7i-ZktNIfTgd_WEmNKlaPO-qjKHPlO_XPKbEfpBdMv5b2P9BIC20ZG3m6qnEc4OqafWZa1iC2szi4eKOEa6neh2ltVLsLS3MlurF4sO-EHQT4O9t-zJ-738mZsOgjsI9rTrLm_aTAJrntSSWRLcP6PI6_ILHyeAl_aN4svtnwQJZhv4_Qf62q70SZQ5fSfqoqfO1YHLcXq6Op99iH3CfAhOjH-NcgThFLpT4-VLYABl8wiWBTsWzdndZoPmvMLEOaEGJOcM6_922FC0Q-fUio3psm_pLcElaG-XIkyn4oNuk6OJQonFE-Bm6WS_1I9sMF0ncSD4gH1Ey-5y2Ayxi3Kb3XWjFvs1RKW17KFXj8sthF3vY5WHUeRKA14WtN-cHsi4lXBFYJmn2FiD3CmV-_4ErzXH8sIMJrDDsqfCoiSbHwih25INTTIj7KAPL2QtIpU6A8zbzQIK-GOKqb0n4wGeOIyf7J4C2-5jhmlF2a6HUApFXZsRcD8e3X1WqSjdTdnRu_0GzDuHhPghRQJ3DHfGwDvoZy6UK55zb2MaxpNyMHT149sMwUWkCVg0BruxnOUfziuURWhT-VJWzv5mr3Z765TFB1PfHJhznKPFiZN0MTStVtqKQlOe8nkwLevCgZY4oT1Mysg7YJhcWtkquKILXe-y6luJBHzUy_aFAgFliUbcrOhkoBk5olAbSz8Y4sSz5vWugYA1kwlIofnRm4sPcvoIXgUD_SGGI3QNsQyRWQEhf7G5mNRrxmLhZZLXAcBAzkw10nEjRfew2Fri7bdvyzJ1OS_af9fHmeqCZG5ievKIX6keUkIYQo_qm4FQFkXZSl9lMHsUSF-di4F6ws31vM0zVLMmH52u12Z3SZhvAFzIV5Vtyt_IfrMV3ANMqVF4SmS4k2qUlv1KuPQVgqGCVHvfeE1oSyYgYF6oFX8ThXNB79wxvi4Oo8fWEZLzZMFH9QEr2c7sOWHYWk-wUMP1auXTQNExEVz22pBxueZGZhRyLdpcA12v8o6vJkVuBj-2eR8GRI7P6InJdQAO9TIBhM7NtJU2NUpeP_84js3RTBVktqBT74nWPaHIddGMSfW2aGmFJovvshhxGMLtN_6XMh4wRKW0IE_-Rfbhk8_-xHKI5McYI048N_TMYOS8KqPPAmGVklRGqPZ5xXMNvQEVweThDTYTo3NoAsS0fN2yMmSwrjRYBHsgYMtil4pd6ddp8dvF_XSJUkW0nF8t6ciI_k47sug3gyw4usqspWxY9Hwbzb4OFzzrgtO_7Ll6lFFFUx2oHy8AO9sJ97Y3Fg6luuew7ZRDzA_4XMrT7mNW6YuT-o2DunaZw-jvQezNHjPN2WhaTS7fkisyhFSFTMBYE-H4psfj_sizutv-LjwbumTcX2mnYE9SZhVr8dL0c7sgwHP1831RxTSSl3ql_obE3ICDooyuM8PYE56Jx0HOOGbEeJd3w91SzNHPG_3SQfXszrZlw4BGWrEUHBbtVY2ZEnsyGNAx6vKO8lz9D-6yZ618foDJSH-Ilk56a5rhr0beWjSd9mYMsr3zpVz6HcpTLYGEgHfPxpT2eaYaC1H_znw7y1eMKamwudYmtz_azX5LrOtwc0p-pXH-kdoNe248pSz9qsmHcXA41fuj2weKQNrmBcghwtfM95B060tnmebJ_B_KkLXL4cNF-hZqi0wAHrHYrZ_WM0Dy90AFH-b7iiWuWz5M1EhZXo179iEdybM-1PgccFJ0zvOqODl7FNxSgWVyNS1k9R42aZx2PzFAfAbBtJ-KVMhUayAvGLNmi35EAT0G6FK65VBEe7A6zPFqzrrAiG8dy3Z0I0253WzIblHPNMpmxI_ca5tIx3u8Za6Nu9rx8mi0CY2jsRSKnqb7RZvLuB78Uj32lb_9jbq5_gL9_y7Bt7U7i7FospyqMFzEYQLvdyrtfNrfY0rB4zr4Mo0tDn_4YOD_d_nP5axUh9_ruqXZ_d3eVdNmlITjQZj8ALe1EfidP8a-Dl62t6STVv8d2y8v9-jy3J7wReLJbJ6gDDnygJllY7NrIVXSjR45FXiCDnpaRonu--I_0b_LRJFOoJUJX0S9YMaXAkKyHSEj-UWjiuk8cIBNcXxwlxnqqNMezvvV113MAOEbfHygDnphzjzZQxteAVbSy0ucGDR2FPi30d6z51NxGnXNS_sM7wnjBMNp4Li0hhttOp6PgvDKPSMAcgUtKLFKE8iWQAvERoUVxw5Et20hNTNXf_0sXOyh0bF0URPGDxSYz9uZI6-nlwVlo1aobdEnn7STSq2_tuTDIrQyfBGZzhv8OB0H3cj9mBs=',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta={'query': 'weather: San Francisco, CA', 'type': 'search'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                ),
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'weather: San Francisco, CA', 'type': 'search'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={'status': 'completed', 'sources': [{'type': 'api', 'name': 'oai-weather'}]},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=3,
                part=ThinkingPart(
                    content='',
                    id='rs_00a60507bf41223d0068c9d300b23481a0b77a03d911213220',
                    signature='gAAAAABoydMLww_4DcIPiCy5zW1t-Dtx57JodsdP9YkyDCvY9r0nBfoD-bwcBl8FfcFxRuq5nK5ndf90J6Xdxmvl9kGVbCUYSFaOd-kWeE4tgwXM8BwwE3jVs6ZMG3BdiWyXS3alnUO5jcE6kuXeun1LufAdZ6rWtJl3zSmqPycnTh9zoQ4cBBxLDq_qcVS1fgU4WjsWgjCZw6ZWRPWwZk8hywmU7ykCLH7SXItM4oH1m_GCEounJAS8YR4ajUh5KAdN6a1wngfGnXhYdzov98yiNLP6Nrkgr--K4vMTTqWXLTcR6fbWgkijanIeKmfSErCjMT6k5TrAbkFx0rgblHbdQii7zj8seV1BWZse92_k4sltxfc5Ocpyho1YSHhgxyGL7g442xMUEibjPCv6kwPMcW9yGu9wPMWsfPYCXpBbG6kQibQPNFJ_bEubwBRaqdSDq93Aqr1YkTYBja7Tewn8UfzZ8YYaGe5y_K4ZD47lfvDp019dOdXmOuZGC1ECRrMqKzSFYVG1CFY1VhjGdPmzobDoMcpZcLn25s1pg6lnNqNQwOk_IA4MvUcCU5HHD5YjmFkEy5-i_iRoDVu5coK0zyEMvPJ_h10y_ByszcfzS9e0ht5CSilckkFdxTBkZ5epp0YIg1e-PrZ790P-I35Ucquam9OXyULV1Y5bn9ohZa93Tv0JZRxUeTDG72_28xRj8tkJaBAZjoCC7VICw39KVmz-ZkuVN6IIX1WdNzyC4d808-2Tz4UZaU42-wxEWDnSDMD7iZu1Bi9fKKwAYBJt_OcEsJwpW63ZaUSG2PVFfm7a3wRcSMxMTUTTJB7L1Keu1hmNepif5tavn3P35nSq28D_IJyAqAgX7ZyROk2bJqjzSE4A0MddqAoBFFqKBi68n49KH09vDtDXIoh8jVWuIgowgVGr8pN3kuhLI9cir4Pr_WES0tPD7yWHPTzrD7OIJCfQbr_4Y4dEza4ixNi0RTADWzMUZBfr7bvwIsgvg6ZNuQlx_d71Go5VDsT2KI8H8AldiRvNWoLyYTFGyK9Kot97YsS5sEmSYgNAH48NU7pgnM0jNDQU1G39nTNFEjL_ziDwjDT5g3jm4S_gbQfwx-XFT3Pv-JYR-E71AqR--Lg71OsASq49rrlULfl5OENfiT-NB6x8MqnfUI6NpcCsOWLp8XfRbgqmZFutLIi43pcnxEe3cXHLWGF77qJXP6dFb-G5Ide7n9tAOoEgfsVu7hCDPEQ_xrIYRdc2DzDPUMCtXBai24E0AnQF8kxsEtlDW_YmAgGNTl9Gx0tFSGdDuUCsNx__c7v-_LOMWycXUKmH3iEr_su83oGIMapNp2PnLccN4iOxspdZQq0C6WBaR6SrdnGzK-0KwRPRoyKDLNWS8zfluR5bIgKlqd3Sbv_7eL-WO4LQXMvdKP3KS-DBt1HbA-gmyFW03iX2smPQbtVmRLWi1vG329R_07-tHMJSO9OQy6_6aiyO8Rgpbl_CHa1Q9BEkI2csonayDJRPvEXBPuk9-NPUP4VLNPB7npWBLlAqes5ZmhagnC7srTL0fFiLGLJiAxWo1f0BBiIlXjwqHdlgBjTw0KryCnEU8Ic8ATzrqEXXhs-FTBCcWInf3Bt5bzUhy20g7cTtYP-VCbsku-lXQ6wceWrfQVFtjKKICD8I4g9QusAIAvgCUm7J2rR3TLkzwOKngdTFPGQrQ1TYzlkA7q_Ew1uZpaPRckMaEioZYC6Sv_B0rgW0nyBJ0GLrB3AUN60hDrOFntyFHp0FM-Zh1SY-GKGBwZwVetOzM0ZAJ-NreFg1XVgyLTYDNjUrYJjRhr_JARsZ5t0pU4_yI6dPqM5jKO5_k4UpZspfQon6d2-NlWX0EDmz6G4CMTx0TScehYHrQZtPzpVnivc8h_pmXV3jO5GLzNeLWoB70SDPTETo1Of4txiEUaC2komu5B7MN9aR4c7VBOTv1NIjoiZcrd1HFACzZ7r1qAE-G38j1f1YhfZ0_TiMmtfR1cqjAKcFkyRM7rZMyMvvnsH7NFq59gFgWZt0dy0aAdw03XWXFNT67lrw58OYC3NcVozH4SKlmleu7TfjHNWSnJVjJ66riLn9DZWVxPeTk4zuISZn0yyaoXcdW8OMn_mJ9vP-8L1wElMyxKbtBRz-0cW7MshmJ3YXmHWDKbnqETSbDMtqcN_QyRJovopwlptJ8VzL7biuURRFw-l63Kc9vKP72Z-QWOUIPLB4q4nX4yb-IV0mkWFxIUlfv5Cze2anf7zDFyGzeU9xG0onfhJE4HFKcoUT8MzfrHZ0dDZtnEYeL5Xem3GuHpwEVGCxRE_J1joTmJfeWxSVnr2Vey9gaPmXCyRrdKS75v9xSXJFfHvcOO8Qp35Dzk-yFqL3dSOJfOEwDZbEf6QnV7VU1EhJvW4XmRS-wsRLMLCYcLrOx96NHEwb2h2l6gNfbCVJoQrMhMg68qBPnoSYLhML2ho7hWkSNZFy61yX5I-oEJV5XdtjFcBkyurmUD6uYTkJSqXyxLexQiPbT-uv49Yp9cAfFBG23sC9lUQ=',
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-return',
            ),
            PartEndEvent(
                index=3,
                part=ThinkingPart(
                    content='',
                    id='rs_00a60507bf41223d0068c9d300b23481a0b77a03d911213220',
                    signature='gAAAAABoydMLww_4DcIPiCy5zW1t-Dtx57JodsdP9YkyDCvY9r0nBfoD-bwcBl8FfcFxRuq5nK5ndf90J6Xdxmvl9kGVbCUYSFaOd-kWeE4tgwXM8BwwE3jVs6ZMG3BdiWyXS3alnUO5jcE6kuXeun1LufAdZ6rWtJl3zSmqPycnTh9zoQ4cBBxLDq_qcVS1fgU4WjsWgjCZw6ZWRPWwZk8hywmU7ykCLH7SXItM4oH1m_GCEounJAS8YR4ajUh5KAdN6a1wngfGnXhYdzov98yiNLP6Nrkgr--K4vMTTqWXLTcR6fbWgkijanIeKmfSErCjMT6k5TrAbkFx0rgblHbdQii7zj8seV1BWZse92_k4sltxfc5Ocpyho1YSHhgxyGL7g442xMUEibjPCv6kwPMcW9yGu9wPMWsfPYCXpBbG6kQibQPNFJ_bEubwBRaqdSDq93Aqr1YkTYBja7Tewn8UfzZ8YYaGe5y_K4ZD47lfvDp019dOdXmOuZGC1ECRrMqKzSFYVG1CFY1VhjGdPmzobDoMcpZcLn25s1pg6lnNqNQwOk_IA4MvUcCU5HHD5YjmFkEy5-i_iRoDVu5coK0zyEMvPJ_h10y_ByszcfzS9e0ht5CSilckkFdxTBkZ5epp0YIg1e-PrZ790P-I35Ucquam9OXyULV1Y5bn9ohZa93Tv0JZRxUeTDG72_28xRj8tkJaBAZjoCC7VICw39KVmz-ZkuVN6IIX1WdNzyC4d808-2Tz4UZaU42-wxEWDnSDMD7iZu1Bi9fKKwAYBJt_OcEsJwpW63ZaUSG2PVFfm7a3wRcSMxMTUTTJB7L1Keu1hmNepif5tavn3P35nSq28D_IJyAqAgX7ZyROk2bJqjzSE4A0MddqAoBFFqKBi68n49KH09vDtDXIoh8jVWuIgowgVGr8pN3kuhLI9cir4Pr_WES0tPD7yWHPTzrD7OIJCfQbr_4Y4dEza4ixNi0RTADWzMUZBfr7bvwIsgvg6ZNuQlx_d71Go5VDsT2KI8H8AldiRvNWoLyYTFGyK9Kot97YsS5sEmSYgNAH48NU7pgnM0jNDQU1G39nTNFEjL_ziDwjDT5g3jm4S_gbQfwx-XFT3Pv-JYR-E71AqR--Lg71OsASq49rrlULfl5OENfiT-NB6x8MqnfUI6NpcCsOWLp8XfRbgqmZFutLIi43pcnxEe3cXHLWGF77qJXP6dFb-G5Ide7n9tAOoEgfsVu7hCDPEQ_xrIYRdc2DzDPUMCtXBai24E0AnQF8kxsEtlDW_YmAgGNTl9Gx0tFSGdDuUCsNx__c7v-_LOMWycXUKmH3iEr_su83oGIMapNp2PnLccN4iOxspdZQq0C6WBaR6SrdnGzK-0KwRPRoyKDLNWS8zfluR5bIgKlqd3Sbv_7eL-WO4LQXMvdKP3KS-DBt1HbA-gmyFW03iX2smPQbtVmRLWi1vG329R_07-tHMJSO9OQy6_6aiyO8Rgpbl_CHa1Q9BEkI2csonayDJRPvEXBPuk9-NPUP4VLNPB7npWBLlAqes5ZmhagnC7srTL0fFiLGLJiAxWo1f0BBiIlXjwqHdlgBjTw0KryCnEU8Ic8ATzrqEXXhs-FTBCcWInf3Bt5bzUhy20g7cTtYP-VCbsku-lXQ6wceWrfQVFtjKKICD8I4g9QusAIAvgCUm7J2rR3TLkzwOKngdTFPGQrQ1TYzlkA7q_Ew1uZpaPRckMaEioZYC6Sv_B0rgW0nyBJ0GLrB3AUN60hDrOFntyFHp0FM-Zh1SY-GKGBwZwVetOzM0ZAJ-NreFg1XVgyLTYDNjUrYJjRhr_JARsZ5t0pU4_yI6dPqM5jKO5_k4UpZspfQon6d2-NlWX0EDmz6G4CMTx0TScehYHrQZtPzpVnivc8h_pmXV3jO5GLzNeLWoB70SDPTETo1Of4txiEUaC2komu5B7MN9aR4c7VBOTv1NIjoiZcrd1HFACzZ7r1qAE-G38j1f1YhfZ0_TiMmtfR1cqjAKcFkyRM7rZMyMvvnsH7NFq59gFgWZt0dy0aAdw03XWXFNT67lrw58OYC3NcVozH4SKlmleu7TfjHNWSnJVjJ66riLn9DZWVxPeTk4zuISZn0yyaoXcdW8OMn_mJ9vP-8L1wElMyxKbtBRz-0cW7MshmJ3YXmHWDKbnqETSbDMtqcN_QyRJovopwlptJ8VzL7biuURRFw-l63Kc9vKP72Z-QWOUIPLB4q4nX4yb-IV0mkWFxIUlfv5Cze2anf7zDFyGzeU9xG0onfhJE4HFKcoUT8MzfrHZ0dDZtnEYeL5Xem3GuHpwEVGCxRE_J1joTmJfeWxSVnr2Vey9gaPmXCyRrdKS75v9xSXJFfHvcOO8Qp35Dzk-yFqL3dSOJfOEwDZbEf6QnV7VU1EhJvW4XmRS-wsRLMLCYcLrOx96NHEwb2h2l6gNfbCVJoQrMhMg68qBPnoSYLhML2ho7hWkSNZFy61yX5I-oEJV5XdtjFcBkyurmUD6uYTkJSqXyxLexQiPbT-uv49Yp9cAfFBG23sC9lUQ=',
                    provider_name='openai',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=4,
                part=TextPart(
                    content='San Francisco',
                    id='msg_00a60507bf41223d0068c9d30b055481a0b0ee28a021919c94',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' weather'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' today'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' ('),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='Tuesday'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=','),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' September'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' '),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='16'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=','),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' '),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='202'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='5'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='):'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' Mostly'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' sunny'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' and'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' pleasant'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='.'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' Current'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' conditions'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' around'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' '),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='71'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='°F'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=';'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' expected'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' high'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' near'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' '),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='73'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='°F'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' and'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' low'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' around'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' '),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='58'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='°F'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='.'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' A light jacket'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' is useful'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' for the'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta=' cooler evening'),
            ),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(content_delta='. '),
            ),
            PartEndEvent(
                index=4,
                part=TextPart(
                    content='San Francisco weather today (Tuesday, September 16, 2025): Mostly sunny and pleasant. Current conditions around 71°F; expected high near 73°F and low around 58°F. A light jacket is useful for the cooler evening. ',
                    id='msg_00a60507bf41223d0068c9d30b055481a0b0ee28a021919c94',
                    provider_name='openai',
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'weather: San Francisco, CA', 'type': 'search'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={'sources': [{'type': 'api', 'name': 'oai-weather'}], 'status': 'completed'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )

    result = await agent.run(user_prompt='how about Mexico City?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about Mexico City?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_00a60507bf41223d0068c9d316accc81a096fd539b77c931cd',
                        signature='gAAAAABoydMovnl5STyQJfKyyT-LV6102tn7M3ppFZHklPnA1LWETYbnDdCSLgeh1OqOXicuil2GTd-peiKj033k_NL0ZF5mCymWY-g5qoovU8OauQyb2uR9zmLe-cjghlOuiIJjiZC1_DCbwY1MHObzuME-Hn5WiSlfTTcdKfZqaQpzIKVKgbx6cSDDyS5j29ClLw-M6GQUDVDsjkclLEcc8pdoAwvuWDoARgMYXwcS-7Ajl46_9oA92RP-64VjrO6Wxzz9HjKcnBTcSDUcyJxsdolHq6G0TjZFwECg4RWvzcpijO53OF58a4_SfgUqbupni7o-tMzITyF1lwE5Xq9fluUFHXmbH0QCrk_7lGRjeiFqY9tTv_VKbNeHSVj5obUnA5HyAYb5jEqgy9M-CgdN1DJeODMTq3Ncu1y81_p7sXqxpbh1c-2eHkGj6yMFjO-dF9LpX_GUZZgAoPXN-J0k3_6VFWc6FjwOGbPU_weslCBpBnS0USfiif9y8nzH2xg0VrHCUEliBOkN-QLqq68edZOBAmYgG8iRDx-yG762TzOBri-0EdFHGWnMij_onb0y4f0UOXD-qSqHvBj8WKasOSRkBpJmIkDViKXYab3nhOtUb4Y3jNhSh6KYEW1QETK9oOMc1zd0Osk-z0QBLQdGtMuFiR00Bs1M_E4T0lMYEsFRqQ8TZmM5-hmrAkBVx3u1f9-ccBZE0ANOiNWH-G75LozwgZhYrOwbuDSnG3wq2M0L7F1mkseg5lOGKgyaxkaifO6WyS6JCHMwDZUF4gZKyHItg3x3PACmTdUy_Wda55J5oIFklWtjFGbU-dY7vr8wvyF0Q0jEeMp8tFvMpGOGTVlydMBq6SCWrZAz8uDoMRxuNLecaHj3bSQHbfeC3hs8uKCLOMr0X_ZCQ8ATXSSjjml3onzNvqChlsspKcwtEKKSwHNTMUJbY6cyy45EQdYhbKg75k-ZL7Y6BXMRjCc5CJd-4uuD8_cXHi4ikmkpHmgZLHcQPOdFflXeDlpYVTF9-Hyblg4SsxvLX9Vp5h4T4J_RcalfwPsIAwIEn8RSutJyMAIm0tYsEzq5i4usmLMxyEBbekCgP5DlHbeWvj3B8h0WoPE7C4cA1m29A_7bRDcJiL06D2T13r9zh17W7UYucDtTcJF7dtKHJTFK_C9m6wW-rHhXi1CgTFU8acDLYGK_VhZhQmTD7tM5JX7IEw_yokWzqyZzWFHmN4mgvAn3imeOXliVLY2YxD7I8-6xAgez6tVyX6plXIpE4KL-GLnFXyqORwIhH4F4EvEm6AcurW8pPWBXXVOY8Ml25-3D1tSu6sQ4PFzgvE5FWiwkBUpLSKwBjZqfg3_aG3NQe4exExztofsCD1l12US7OTx76h7utifDiu_FuzSZHOq0sM0kWfsrzoaPW79T7CT0Ew97HqEJTvYvhkdmzgtA-57zYK-8kc2bUTmTNdl_nUovO-xRhvwamIjMTzgqo3FXjLAtj4QZYWIHInkGj8GIxLluow315yWxARpfTehrpgvwYbd-tJ0UFyCZ1J0RwXQ8QmBu7UV-qPxj88d8cuY9sn8xba3kFCLifxlohEOupJcDDNHjta5eunNYoE127ap0Pv5KdJHWaOUcpScrXz3dIEXBlax12ySZNkghKGgGqYzOyQBKvkAgcV2rHaUQjuAkEbV3uQuE7iG3413fqfRVyAOKHKv3ig0jUM2DqBfhK9Tmxdbh-5VI5H5r5dgw3GmTQtSZVd0Q3mIMCeghrfHeCW4Ms1lRjcwEbn1Uyffs7KylhabOdqmiRTUPavLgKZmSrh7q0Vrkmb3s-nZEcfnVL6o2OpuQrdm83K-aI0Pvnsf9V9U_qoW1HWf61ENQUhnMECD2P70EsSmXLnQ_7f3v4Nyw-MCWCPpdzJvCh0TrpcTpY4WcflgbkNxm9xorCEiTlnEaeGSYj0MDcNm8sJYZbWzNQoNmbj58XS4IgnfCIYcoyu6PTceMcE7o_w50MPC3LcMTzZWKSYnGA7xDrvfeD7boqfj-Xd37SDYSTp9OAifiwiTXZyl7FqVTk1Y-1RCYTvIPPpnhXedT4ehYPRL9_fYmTgVISPLK8IQyNHpme86nG1-0FOJoitzwOa94MICeNKJArYvZ4Kj9WlP5-cTjP6zoDlaYxXXuln6DRmOnqL5CDVqf3f-7Dg-n8ARgNFwaAuvLXhCxuuRdcnNN5gx1z5vnvusq2sMCZx-eRqaGQsRoAoWo1VsrW5bwPGHwZN9Ip97KeORMAV8ExDttxjS4DXO-nB5fVZ2KToAsglOjLfvoXi7ArwK4Du3u7N_kzERB8lVT25jOltMdhOISXCGzY-ORQr6WhS_fgM8s8wHJSAtEl2w5VaFku57kEgWmfmasDNz5O1iMlqKOzVGpd9qNUtWaqYDK9DIxaL-O1pQGbzzuCsq332tez68SMNdbjNaf5RS3MHgAKHmI0I2RaGdBcaXjlap3sEMANG7keCNYSrtU-vfoMfb708dt2Ux2dDktmtSMFwZyzbOnGOshGhxsW5O98Uo-I-PZLsHSj4ZJSD5yIayNiuf8bZ0_REJ-9I-5xdfyUDstO7xj4IRjwwnsF9Td8CUycBKxr4gsttwfOoo04LVLOg7mDbK1GtoLEP2e-nXBHsFsOObaW3bOTx7TZwQf5DLggHsEfqdArl1-MqhRllSJNFtBLV3T8bRIvDl-YCV_LYjvWqRvo0RsR3oxrrPGwHM5ROy0WdfHixv2t5voksrS40VJI-KVXqgvF4ixUTMCjpL_pKpBq3pVZEnsJc4yZgK-C-sz72NZNKFHZviJhcdPDuwd4dX7oiI9X2KbnRfoo67xMqTuQCryLeiF7FpFoBHIjH2OhMzk2HbJR5YK9Q8blsWHpAdy',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: Mexico City, Mexico', 'type': 'search'},
                        tool_call_id='ws_00a60507bf41223d0068c9d31b6aec81a09d9e568afa7b59aa',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'sources': [{'type': 'api', 'name': 'oai-weather'}],
                            'status': 'completed',
                        },
                        tool_call_id='ws_00a60507bf41223d0068c9d31b6aec81a09d9e568afa7b59aa',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_00a60507bf41223d0068c9d31c935881a0b835341209f6ac8b',
                        signature='gAAAAABoydMoKoctyyCO6gsPILkjEnvCX0VL-9Gqk9qAmNEdWKNRPvxRIBVCxX4hGZ4m5fZJmuSIjjrA-nU-cUj_XIsARJsJywo2ka8IDmGRF8m7lm5atgcSJQjytRVpIA6s7sz0Sw3iAKrjtQcbymz2sUViTiOn7OqUStKtW0h98UIubdU6d19hu3iDwNddCuAC4QDy8cg3qJhjq9QTtovoBwFpibBJ12ISJqoPLSs43YvWK26o-evCMfzVbkuqJ7Gqie14gZ0oQChxGj7-bopeml1MCaDAz0EUxD5EDfjSdgjB_JABqF13kTTFdAVJu8gY1WgjFt0m1CONQGlM2oQA7cywjU7NnGWSNOqZp_NSDeTBYsKykAmyJP_lTzIDhhG37GBW7PwvBwuUYbvPcMmsRR9FDXxcMeVcpZPmaDjXhRAkJ-Am48Xz676pYl5Sx732-Pv9w503O66ARt6jwQYB4ZW5GgJAnqoqugbmJoGfOV4TaF0glOfKB5XPNQx--_hARpmXuQX3M_Xg1zLa6n7xGmf9pv__Gnhk3V0OlEnTD5HPZzc13F2hKX1PZ8E4ykq4843ZHDV3vpc5WsNCp6C6Cq8STXq58_QAU8P9vpqEP8khnYt3EJTjzbweiqVrMj6cSoUS9C32z8dFcA0rQrTmt_tEMTaoTN1Q5nTboSm0jX1arXqGh3RhcDkqddBDLfI6PdTVulEPVnBkmZJmCFqdfm_aD9FCSCVJdKE5pktBFqtmGFRJ6RVeGbc_YB6XG9najhjXNhhXIpy176CIPLZbeXkxcgsJQBdDGm4PpUePHZAGKxOpFCNv7kZMyGcsd-Ye-envhfdGhJ5dMOqRq-1KtjopdvNFfmxASkrT8f33YFj6n07fXOOfY02pTl9Dyv7fp0gk_3DR6zKFZRwv-Y3u0sTjQTkk7xTZsuEb0iP_zpqMNcj834fq4FZFvmhJ_siVVOQUPMaP0OFJnYFTteQR8S8JXud4Er1jEZlVojHugyJ3K4yMoj5c16jIQLaFn1_Jk1G97LCO-WZjSxpDD5niEXmYEoC1cw5zweUE7MjkzG1cBU2Wgjw_K0zt0Ko9DxYMDDDS-ZphpCJFPKBiX7pDcpKDpkQnDkEpzIIyDQ3mEKoKvYAXLveKuhOnNnVpUVN28hvW5_QfhD3C1WEBTzz2-dfxLpiS_MHI9NVUZdIue_ThGAM8TFY9MqDrTfAMRMD_mdQHW8XE_QdxighLLuG56AqufuA4CutwifYdbMiAE_mWtApqG4U6dx8cMnmIxnN_lrerv3IQR9_rk6vgPG-MfyJ0drDmSaJGMKyBexYau6sCzyMZYzFO-YgPDa0Yz4DYwhjTnGqtoMSE94ciYiJWZV473WIcyvJ8lE2mQD735nf1OKk7FHsai2mmQzk6NHyyEvvltkTPN8ply0fqmxLksng1bKD43zkHjnP_wUU5uInfAPIGMtIXuwJJXUziMTFRcCawC0KcUUP1J9GK9nrIMeO2B-yM5GXwfvMq3TiI4VFHD9Dav18T5BufMsjIY6uOUuWKNHSOpSQ6VHoql3k7fh2NVGOWqq3juBo2P3BNwXpP6mPr_6diYK4ciukrh4MiUd3pkLZnaW_iv4XYoq0Wix4ENU4zI1kMj5ObFAQOEbeoqdC6u4I5MIOXU6Pep-kaFl6P3yb37Ce95GyPq6xx8q4G29DK6Rx9Qowha8x9BIphuSL01Z6snFTewQW9rqAP7GyEltkso456vXzay08wtzG0dGpxoCIc87mAhx7-ulTj1Wti0qekLhsavem7GPfNKqso4CPsiXMxtTBBoIHk0xAvXcpZcw33pY_71-SHpMafrMrkS-Rp2T6YztbX2u_Nx__O8NAD2V0T0l69gR4S0khT_z-rttSPuCfx0-C4_hz7mCjVPMlLGDzxahOxG25Z9LHst6NPvlfg0xxX5rQ80XAS9GtLJ5uKMEwMxoGCatV3VL2zT2M0SpNiZKLZpH2tHfm0j_2dFcsLWN0a9MAooVZQ1Rlnq_7r0QrAPqcca_Y1Q7Jlzx2dgiEylYfFzNlNU2JTtinZg25gq3A7WayuWE5iBV5dhPijkcgEQbDETKg0eRa584q_cd68Rlm7qYeID3pc8gAbZ4zdqz6SfcQqoZS_EN43Z4Mc-t_HKN-9BwgXFNfvzbLoNekhoCiTrcEUikzXjVKqTbcuczAtH-uie_bfQkwfljFn7J8t7A3SeP961mvpx7iE-yJ4HXTeFhJI2TlBm4JB3OKMCoJSFdEiHjx82bX7TEPvq9g940TgPaooWUD2mEJ_f9ByY84L4EywrGFhtj-DxA1igkbWnCgWlxEquBcvmkRHkbTylkJz6kyz-_-5EPUEJLHqGsDHgotxYWXsxCalzDktH_GivrkeTYqhy1SikEJw93-X5SPMLD7EdQUS_K3XIe9p4T9lpn__zs_tCqssrun7ZQEpY9ULoYiMn2ENU9rK4IYpDoV0beXs4Xa24nj3qgrzbuzbLeKKbm8Y8RxNStogi4E4pK_difBVb_1oTIxfPrLnAJibQ8H-Tb9v20L2Zd3RWXtKi46-XJizKe9r-_JI2HmZ4QM2JOaBhHdybeBrwnu1Z36WhPk4m7YyK8-0K-kIPd-mW_ZF29tHBVhLifqPOq7D3HkJbnBH--KJum-F3v5LLqmeBN-3LWv6bk9-jqQNum9pm2WHtUkOMvH3zw0h8yiBjK3Qov7XHAP9dKHKs3B1eVqiVFGNbuB3Ss07ZzXQrSxgNFP2z64-HtdLJdsSXu3BGc7BqFrnF1tUVeu-KDXKXxJ0SFYaxnLqThuQ4b8CUXYWd8fnhCbhu3OE9Pd2aKWr-4bj73DTDcHLnYmy53mgNKtItsJBfA7m5Dzf6WKREmictNl5nMUWWlEay0nvE6so39zkRlc7wihRthJTEMDbMUdARJw7o1F8JBUPY3cIJchDnq0ZiGkrCA-OyPx-rkxbrQq9usJoTT7XUZNVZ5u7mXH8dY6uY4opcJmV02W2eJms-VtTxgkXuh_HLz_VPmCRMGfACFMwigpShdnr_j3T70ixy80FLcY6ILu1EbuZeLeqo4L8Z5fznYZ1',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='Mexico City weather today (Tuesday, September 16, 2025): Cloudy. Current around 73°F; high near 74°F and low around 56°F. Showers return midweek. ',
                        id='msg_00a60507bf41223d0068c9d326034881a0bb60d6d5d39347bd',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9703,
                    cache_read_tokens=8576,
                    output_tokens=638,
                    details={'reasoning_tokens': 576},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 16, 21, 13, 57, tzinfo=timezone.utc),
                },
                provider_response_id='resp_00a60507bf41223d0068c9d31574d881a090c232646860a771',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


def test_model_profile_strict_not_supported():
    my_tool = ToolDefinition(
        name='my_tool',
        description='This is my tool',
        parameters_json_schema={'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
        strict=True,
    )

    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='foobar'))
    tool_param = m._map_tool_definition(my_tool)  # type: ignore[reportPrivateUsage]

    assert tool_param == snapshot(
        {
            'name': 'my_tool',
            'parameters': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
            'type': 'function',
            'description': 'This is my tool',
            'strict': True,
        }
    )

    # Some models don't support strict tool definitions
    m = OpenAIResponsesModel(
        'gpt-4o',
        provider=OpenAIProvider(api_key='foobar'),
        profile=replace(openai_model_profile('gpt-4o'), openai_supports_strict_tool_definition=False),
    )
    tool_param = m._map_tool_definition(my_tool)  # type: ignore[reportPrivateUsage]

    assert tool_param == snapshot(
        {
            'name': 'my_tool',
            'parameters': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
            'type': 'function',
            'description': 'This is my tool',
            'strict': False,
        }
    )


async def test_reasoning_model_with_temperature(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, model_settings=OpenAIResponsesModelSettings(temperature=0.5))
    with pytest.warns(UserWarning, match='Sampling parameters.*temperature.*not supported when reasoning is enabled'):
        result = await agent.run('What is the capital of Mexico?')
    assert result.output == snapshot(
        'The capital of Mexico is Mexico City. It serves as the political, cultural, and economic heart of the country and is one of the largest metropolitan areas in the world.'
    )


async def test_gpt5_pro(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5-pro', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)
    result = await agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('Mexico City (Ciudad de México).')


async def test_tool_output(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_08ff313743aa5a6200697ccca50118819f9f50f8853c7c1ebb',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=62, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_08ff313743aa5a6200697ccca45810819f9e5ef7bc8b8a5d1e',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_Bm3vvu0Y5L5gEYzStXNReZ94',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city":"Mexico City","country":"Mexico"}',
                        tool_call_id='call_7bZyjNnz9xsPUFHTcdnNKwDb',
                        id='fc_0b35d9cb374e54a700697ccca5fb7c819dadd8dbf17e545af9',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=81, output_tokens=20, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0b35d9cb374e54a700697ccca54538819dbb37ad4ada16529d',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_7bZyjNnz9xsPUFHTcdnNKwDb',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_text_output_function(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot('THE LARGEST CITY IN MEXICO IS MEXICO CITY.')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id='call_2KOvRpMHLjrlWgf3O0GaNT1k',
                        id='fc_0e72e194bd8a114f00697ccca77f1881a09797e17c0a84fbd6',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=36, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0e72e194bd8a114f00697ccca6e88c81a0a184b8d6a8696988',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_2KOvRpMHLjrlWgf3O0GaNT1k',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The largest city in Mexico is Mexico City.',
                        id='msg_06e59ea7bd8d01b700697ccca915dc81928358b6746724aa62',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=55, output_tokens=11, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_06e59ea7bd8d01b700697ccca871bc8192bd6fdf70fcd63490',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_native_output(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_0832bea62c53a02600697ccca9bc2881a3ae7755db09e0bdd0',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=73, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0832bea62c53a02600697ccca96cc481a38cb13a92f58c87d7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_Uog29U6F8U0yDj7TKqzN9Zkm',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"city":"Mexico City","country":"Mexico"}',
                        id='msg_0d9cda63e0df6f8400697cccaa73ac8191ae9802f1a191a44d',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=92, output_tokens=16, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0d9cda63e0df6f8400697cccaa03f08191912457f35e032578',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_native_output_multiple(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=NativeOutput([CityLocation, CountryLanguage]))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_0194c6208474c48c00697ccca7642c8193847d79b96145e416',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=142, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0194c6208474c48c00697ccca7009481938b922cc028479565',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_UgLxgDyY8kJc9hTrnXN8CWXd',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result":{"kind":"CityLocation","data":{"city":"Mexico City","country":"Mexico"}}}',
                        id='msg_0ee8ed22d03afdd500697ccca83440819ead369b6ce161efba',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=161, output_tokens=26, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0ee8ed22d03afdd500697ccca7b7ec819ebea5ece5c562b17e',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_prompted_output(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_0dce1d99089e937500697ccca98340819f9f3dc70adf128bf3',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=114, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0dce1d99089e937500697ccca8ebfc819f9ec20f40988ca782',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_OU3eMY1qbWDJ4UZTXLjEUVx3',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"city":"Mexico City","country":"Mexico"}',
                        id='msg_09edffa02192a81c00697cccaa3d748196902e046649fa3e59',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=133, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_09edffa02192a81c00697ccca9cd4c8196910b110c2eefe971',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_prompted_output_multiple(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=PromptedOutput([CityLocation, CountryLanguage]))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_09232c544b13ca7d00697cccab636c8190bcd3c0ea92d4d307',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=266, output_tokens=30, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_09232c544b13ca7d00697cccaac4d88190af5732ffc2d78f23',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_t2ZBySHadRskjx1WQCV4UqhH',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result":{"kind":"CityLocation","data":{"city":"Mexico City","country":"Mexico"}}}',
                        id='msg_06e8c8752e648e5700697cccad01a08197ab77911a45e122e0',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=285, output_tokens=22, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_06e8c8752e648e5700697cccacae688197bf0b3e56af187f7c',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_verbosity(allow_model_requests: None, openai_api_key: str):
    """Test that verbosity setting is properly passed to the OpenAI API"""
    # Following GPT-5 + verbosity documentation pattern
    provider = OpenAIProvider(
        api_key=openai_api_key,
        base_url='https://api.openai.com/v1',  # Explicitly set base URL
    )
    model = OpenAIResponsesModel('gpt-5', provider=provider)
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(openai_text_verbosity='low'))
    result = await agent.run('What is 2+2?')
    assert result.output == snapshot('4')


async def test_openai_previous_response_id(allow_model_requests: None, openai_api_key: str):
    """Test if previous responses are detected via previous_response_id in settings"""
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    result = await agent.run('The secret key is sesame')
    settings = OpenAIResponsesModelSettings(openai_previous_response_id=result.all_messages()[-1].provider_response_id)  # type: ignore
    result = await agent.run('What is the secret code?', model_settings=settings)
    assert result.output == snapshot('The secret code is "sesame."')


async def test_openai_previous_response_id_auto_mode(allow_model_requests: None, openai_api_key: str):
    """Test if invalid previous response id is ignored when history contains non-OpenAI responses"""
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The first secret key is sesame',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Open sesame! What would you like to unlock?'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bd97025c8195b443af591ca2345c08cb6072affe6099',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The second secret key is olives',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Understood'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b',
        ),
    ]

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    with pytest.raises(
        ModelHTTPError,
        match="Previous response with id 'resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b' not found.",
    ):
        await agent.run('what is the first secret key', message_history=history, model_settings=settings)


async def test_openai_previous_response_id_mixed_model_history(allow_model_requests: None, openai_api_key: str):
    """Test if invalid previous response id is ignored when history contains non-OpenAI responses"""
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The first secret key is sesame',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Open sesame! What would you like to unlock?'),
            ],
            model_name='claude-sonnet-4-5',
            provider_name='anthropic',
            provider_response_id='msg_01XUQuedGz9gusk4xZm4gWJj',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='what is the first secret key?',
                ),
            ],
        ),
    ]

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    previous_response_id, messages = model._get_previous_response_id_and_new_messages(history)  # type: ignore
    assert not previous_response_id
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='The first secret key is sesame', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content='Open sesame! What would you like to unlock?')],
                usage=RequestUsage(),
                model_name='claude-sonnet-4-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_response_id='msg_01XUQuedGz9gusk4xZm4gWJj',
            ),
            ModelRequest(parts=[UserPromptPart(content='what is the first secret key?', timestamp=IsDatetime())]),
        ]
    )


async def test_openai_previous_response_id_same_model_history(allow_model_requests: None, openai_api_key: str):
    """Test if message history is trimmed when model responses are from same model"""
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The first secret key is sesame',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Open sesame! What would you like to unlock?'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bd97025c8195b443af591ca2345c08cb6072affe6099',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The second secret key is olives',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Understood'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='what is the first secret key?',
                ),
            ],
        ),
    ]

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    previous_response_id, messages = model._get_previous_response_id_and_new_messages(history)  # type: ignore
    assert previous_response_id == 'resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b'
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='what is the first secret key?', timestamp=IsDatetime())]),
        ]
    )


async def test_openai_responses_usage_without_tokens_details(allow_model_requests: None):
    c = response_message(
        [
            ResponseOutputMessage(
                id='123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ],
        # Intentionally use model_construct so that input_tokens_details and output_tokens_details will not be set.
        usage=ResponseUsage.model_construct(input_tokens=14, output_tokens=1, total_tokens=15),
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='4', id='123', provider_name='openai')],
                usage=RequestUsage(input_tokens=14, output_tokens=1, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )

    assert result.usage() == snapshot(
        RunUsage(input_tokens=14, output_tokens=1, details={'reasoning_tokens': 0}, requests=1)
    )


async def test_openai_responses_model_thinking_part(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(m, model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42cb1aaec819cb992bd92a8c7766007460311b0c8d3de',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=2199, details={'reasoning_tokens': 1920}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 22, 8, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68c42c902794819cb9335264c342f65407460311b0c8d3de',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
**Analyzing river crossing instructions**

The user is asking for analogous instructions on crossing a river, similar to those for crossing a street. Safety is key here since crossing a river can be risky due to factors like current, depth, and weather conditions. I need to break this down into steps, such as choosing a safe spot, preferably using bridges or ferries, and checking for hazards. It's also essential to have the right equipment and to know how to manage it safely.\
""",
                        id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de',
                        signature='gAAAAABoxCzZ6TwL-z6tYv85YPC9CwMFJRrSrWgYUyOhSFYLkdfGwRUIbrgBnmAW0qtzE-shPNPNfbUltLnBS4uU8kMuSMPxotFngVcFFnZ8qmzDUJiEf2CsnqYqH5d2RmuvIXUQTau20SjGte8vOLksGsz0sXRUQnFs7ZoT_2DSy7tKVCCN7fbvkxm2l9D-IibdvGiflbP6lGGgYWczEl3_QyEyJGhDTcL_nUk511C2hjvqI6YkEjujvYjEGMh-BdhFrJeLFL2wCbn6beWaDB2IXYVo7LqhlQUtKksY2PQ4ybBgPhERNvWWf2N-K4NxEQuv6vdB_5ni_BlUKFMabeeeI4wVr5M3HKE6cSgnM8cpi93o_rGu9uSh5fbN5zZcIYumnjkGJ-ULwku7TVtpw_Vha8bJl67qCLvRe90MaCCgJxFKp-IsxakpnNCXVrAlN70cIchPEjOWW3VKrGZ-omyQLgs3EzddhaPmSqDELbmYkxCUuRYdLvZPzhI5VvNiW_j-WGBKEMJLFz0k33oPbH3Q6Fn9HcJSruSsQ8Lc-QWQsibGoABEY_auM7LDLjbvpEYcdsP5BJOBUS8wpyOD7N5jEQL2PYPT3tsP6mvWy7csRIkwEbh1x1sXp6NL3skS6QwpqNiWalP4OQvs8jDi8Gz-NrtNvmX3RR4A0MIuX4q0Tb1VI9PilRUvOvg9hF-fk3bqIh7wW7_iD2eDoXjvI99M-UZ2QZGZlPa1O039QVGLBTi-KwrLuhh-D5q3N-3-DyXY0sjsdFzM5Qf__lVaxtRhZRKyxSONZsTt8mGatlOH4Q-MIPgCXsrLu25gWn6Jr6vt_iQ1eZG30cox98F4aU11Hp-o4tFKuHL8Kbgl9Eetpd4ggfJrTe9QxQfWfgCHAKnhjVRJOepHPEFzSRFKiHrOgATIw_x5xNRacsF_tO1WkpeBdBJhpAUozarHgPQi2A71LCcmd5Zzy7FAL6R_5QtHW-H7NE2PjqqXnf2ZMiMM-oYLHKhfu8tZu9EV7V21xKkovIGH-0OQ-lk44ww8-j3rBNXjWIwp8SO73O85PP7vzE_zJJCF8riB_rOpazJpMCaDP_HUOZq5XhcfHseiS1SownunaXj2wOmgLJ103BowC18-i83n8RVTtRQxLsiCRzMwRH3eEbEClqZ1Fqt4V6NKqICKK3kF3_R_oPtX_-7xYBpl8LGR2Ei_2hXkEIuQ8cCRz1zmuyfjLEWIbm6Rm9wgYafXIXugOqGl4lUc2NSeeJ5T_-ziLEX9-Z6LZafWo9PMvxbxpm86-e5ZqTBlsbukWWIePrKZ6BjElMcxTOUr2XYg-vMAc7QRx11Ld11DwesN_Xj62Kc0TUQRfREPp2AF0gAlYYKLvl7XKwU12Rg6UlWBfJWjX-hUvPY83B7H1saGyAv6V4RR9PODI0p8duptI4mC6u3cYPUk8pOVwFvX1PYCTM3I2yik_XG5ZyI6BIGBNAGL5SoAyMfpqYu0Da4duOIve6tQ3FWkMYnU9sSjxvS00mRJ3JscA-pNm2B_dUXBRATIAD6XVg1m19-pV6OLG58tuDtbjE7nBHUHuS2sm2gbm4PFjpGKFMS07LUSfYax19Ombld0hV_0kwaIjl-4UTcRcqqLK-Tc2E3NPwmdJuzb95-lUoAJG9Jp8UCoyXQRcZk33IGXMYtNquee5K-Oc-ajKjbbevpi6rIEOVjwKrZYwTC5ddl8FBcCBTm0NunKUzo9TkDeEIF6kce9zYdKo9GwYc8Go2S9lxbb6tlizDql2Uv_ZiTz0d5RUvShsMH6RzKiXu8eFzv5sUraMbLUtm3VtZsz9Ld0iztNIqV7q6glrqD2UuLgWxKsa7EbDRvrlBnAOP1iMRtg6ICr2YHYtlwU6R5RzKllG4LGMccZ3x6HsZL6MMaMJc8rLMTtQYqE067kCnXAIBz_-FJS0CzwCwnbh70tc-ytaiTm404Jp3CIBTlGwDwTvXUqzzNAdFRHrilKUPNgtKb6uQAyPhJZwbFIfrFevaMdOGZ9UhJoKtQtsSj1qM-dhXIkn4T_SL0o72wgb08zYSjsemhfJdbNO6jPG_GfWV6FDAhY9vNyTT8XlqgsePnALnV60AIvRZPI7mfh8D7F8SI0yZU4AJVtEHKKgStPvCuixxtjTt9nSWm-cXOItJdKYxMjVtWCmcCI9LbPolXXkewgNhmUx-b0XlY_BN9cX2j0COic5Ml6LS0BZ6ikG65a-7kIl_yofLeat8q9-4OI-lhKYOxwMhQq8yWNCBuJZCGg5n3DM_QgEQt7cqWYuTuTMTgMnRIvSefK_Q-ZSY-ZfzPcNp13-v56KgA_yUqYOHvQDu4ZuZCI3KvlO1WSO17SiqiNZvo3iaM2BrrgzOAJYcSx4TUzl7ZuFE_6LkokGlxguZH82W9dpewDyN83yLLwakOhgLfbTlMCFs8NCLzMr17eBFNX8YqP9bh88f98KtLSYr09naEesCwW3v-M-I1MUdR7oEC64ixrAWEwpSI-Jcc60oUgvLBVbwkeoQWI29NqZCS2JYGsxjXfNmhWWB3kW7_7ik5RHk3oNRWXiEQ_N_K7c97Ui1YkomyAAZv4t3zQYHwsLBmnf2DCEMBViQn2nXXjGwBxEf9KY2dlT1ihIbcPBmyzI17UrKYXvYKkfE25c_R0VjQtRSiXrlEWkBgLwRzWt18_62P00dJRAeJOEq6etGKoSh0PsrmtCGfnOf1EJhksjZPyObdHWm_yrNiX-h3myewuYT4BIanBFdUQxQvpNyvoQJIMtKH3gAh9oCJY8os-vS1Mx7h4Zh0gEdQZhIVn7KoJwqbx09F8hGokOaVXx6H9gMD5kVZkXkX56AFC78ykpMLSsolvyUhAI_qxcpTVGSgDIzfK7BzQsMd4TawOtHObxfvVr9qojhx__eQEipXplbfOVh8vjM8TFY7eb8k5EvxaU_pBuEmSagGoV0fiPUmMTzipVTMQ-cpaF6xQSVYZK3CvmUwxHfcX2r8lPPH_SyeiVFKRdqTmpIGYw8drhwc9GFgFM_N1O_iJuhjSt-8iLt9BExKYJZftIcXPYl4eGgC1yC6oUBurdmvQUOZD2tv_isd28fMNeoiWAy7HCDz57TAkJg9nJsfULk_z5ClBqz7rPP9Qq9CZrYEsz6Lggj19AXh6PmA9Skl9nUSgsqn2GvDI3I5E8YmxhlP58XrZyYKLkYLqqAMDy4tqI2eiQMe4z1OhMGpVxoEt7ACxw8-YaQTKgItzSNNskrXkRsGvWa2m9XcaZNwbZFeqek7OZ0vxLMsQJ3InnvY1lj9aMUUGukolxZEZ0KolEPOsIKhwH4iMg79yNlxeqdM2e-OoxrUyIBvBqC5uNemtqnulwarV_0K4XYtimpR2ONcwjzSO-eY9untVEmJRdw-OYEJiKJoRRNEjmr77UuJLDhPqjtxmcF2LHRTbpQ8b73Vp-mt-1okyP2yOr8Rd6LbsJuntv77VWE6Mt2JBzJpwJZab3mKd-2c6VkTswJX8p8Lf3g6m3uPwBRrvyCuuxztoxG2rDrh-3krcjDFnwU5tZjpZ3G3pq6KXbcmsX3kvJhJc6er38kO6uR9SkJUcRkbQPZ5nUKCtRZ3AkFukiT_YHtUamsUlA0nX5DI0CcrrNtu71HD7jSkwF9-P_QNX3_c9k31cSfHQ8BCMHcy7DO0vPYuzj2BiStEdTymAXfMeTFVO3wUvQf9KyuqVnKFMtFwP2DaospAJCd4TaHeDCm-uczeg1of-R2SB0f5XorCQ5MljHFk1VTHmWG3LHgCTEmAtx2KKbdMO8zXsBop6LAnFj3dNEBv1grO5lMCZRqKRo-csAJKwqXn9kuIa6GXi-NMZWBGLvxWhJpWdCeGnaFjCk-7uzPQ7PN_JhtPjeI3W-9A5bY8RvzYLrKKIEOEMGyHb-77dcYPhhLgn10-HpglS0_uJw5f0STZE60rxVXH-sb7vrpyAyBqMUh-OdoGAqR4r9dflKk-jECG0sggRuzx8jLGGfO-dNZSznD8JbaCsRDdLuHWiFSZKJswXwX5PS9rmVpYRrH-0QniMhPiMOuINwp9GvTtzSvXNIPBWtNMsY_jN_O2gIKxL-KnyoQkyUFWHN0Anuyv5pCHQk1lhfcR0alSvDLVmVVgi1tQdjsdAl19qMdfm43ViBTRLDwzP8iauhipIfCpxFjak0Ka8uFaHfEhKstcmUdxEnzIsN3SQXrPud-p0YJ3O1MsyUDjzMROP43FKMhVCSnM5PsXVvuPZKRmuEFbJyj0dwwBnSH-rk8tuK5SYoX8PIrJXyAySgj2VouGdzvkqa9WPSHU42AxDZNQT2c5pEI3gUS7VzcGCTwoznmFtmHKGOJj8HxKswzB2z0AumI_ht8zx6kehnvKK4cJfKjdJfO--2o_gzVR0DkA1HcenMD4ugxj6rWQDiJvo6cFdfkXpMfLAe4NQRLVLNv4UdIehCxCHMsF1RZJf93Z6sdxW_2PumLLcfdb2XIkouuFN0mECzZlWgM4XLUErlaaqDc8oU_gRrgHEJbjZqF7by2nHtqCjyWOZC9n7vF18iSRJES5DX71nY84a8ydPR2_KEYfHHgqF51Nz9zMo2Gv6ramAFs6lZC5rjOUR1VmpLd2pKs-sPQJpYNncwbDe8dG_NDJI1wIgs2K42eKPE4lEueCtp359ifPU58DpJA0JiB_FW3jBBh_9MphsnQjuso9S-YSaSFt4Llc_Zf0MJ3CbiEsk-IwfgsLysFQg-vJq9hHR48ZoW0kx7CaIAUgkiJg7-7m_dVNsfaUM-OSHtfryYXWrUsotLJ6OTuicPOfb_LfaYz5OoasG0o-4vdtF8e4VX6ZpR_Krje-dZaCph1XjgGLfFlRRkywl7i8IEUirHpQJaoCPmc9q61hjETcuPCEVDy-dJMwPLiNzXY1mkJI1lQQ1LL8Ke5EgtAcC-2K_5Nwh4h4bafWA44XgnisMfYDv66CzsL35i0XtZHibfLqPwdLyHqInlLPtiFyhJZifcJ_GsiQ0U4SZO3WHmvSG9AHy0YnA2jC3O4x7yj73RmroELKFvT-_CSO5LVz3qRRVdlyL0CQr3lkHUXJbFNpOz7Uek4q6MjB8Cn0xAKzt9ztcClLdAA9NkR88Dy3rOP5yPjRJJVVG6SbelsE1SnPDaKQqu2ZN1zdtGUY9AcULc2dGFb4mezcbJsC_eL3DD0LwfwcSAILDfDIrqHACR0XmI7jmF_p_PzmBhMsocLnblzKTho5ufo4htqhTEjVdFywm9WIjw1N_9PWge3i3J8mSdILp0HQ1QvbRlMt13qPyLNwAmY75m4Zw7EIlk4cyxMjcTSY02TRCC3cBKX2e6xl0DIAygChReVOxL16BT21sEgT2ps24Mljsvht-fokU3JPtj3fY8wojksHAWNv2hlF62xKrOv9h9AqM_Q9JlghHHNTrzxvT8SVeLBhEvysbhErrgtTPcgvKfMUCT1JMWolxLZUSFEFKNR9cl2GiSqMIhdTL3JvA0-etZ7ykeIbUEU0jlRZAw4a3vY1R1zW1-c6XRSNiS9VDPAfcN2OD4bXL8Hi2Un7Vc_pRyWn32kWQqQuw7v6J4RB-sr3Bkb4OzUljytF3ixqGkMIpuA-LjJwdBJ-XgR8Fj_DONbguNct7MTkMtQqWU9twa8XF_sot8iEYc1UOkZQVJvTtNclIkklJhP8VNj1aY18Y1Gtnn0fvfaOFfY4wa7HIq0U7paT2Fqb0xtY62bqNM6rRXLsAQex5HvMvHQvL_9JbQngBOD2ehk7zIAJKZWzbAKqKCgH-_v4kaGBVrciIxffYF6WnWq35pQwbsvNqQG4uJR0qhLEXfc5akEon1sL2fjxim_GFcgunyCqL28lMB8yhnzA48QESW0sotVss_GIeeftSYS74zPaHXiuSx0RoAmw90kERXPDGVCRkwTTwaW2BkshCJCx2JOSDEWVRMjxngzaoroybl9A1VzTrDEbodeq2O_kVyZfrZE58BO8DDd6ZH07m4zuiL_NKXLw23NmC4CuAN0ufn7HFtX_ow4ZJILfZI9PczBi16liV0LWrBwvadZ2Lf4o3sV65CLg1VKujttEiOhtpYjmB9zIRLfAHViz5vgTonL6H_9N4D2dXLzUWlaDPNGJNnt7EbHS64f3jSXGG1fb7LGKhNtTHqbfBJnMpmPjFQGb4ae2kKv_V_L8xgXNpQRi5I_QNTzDzfBs4avowydppTpEE1b8WNITLc7qvMa6gwpDxp2N91fIVC9wyd0k2DXaz3VbDn-2v7W72D2UPa0L00NKvIKYecd9FMpYmkEyxJfFFr7b9hKA1ZXdqlejjPJPjVYC9XTEZLEwzdeoG6yrxtYMrl7o72M37uIRNZAj1Jf0B8s0qhPaaBfUVKtKl7bd2Ub2503LZ7ZvJ_1segaIT4S8nt17hjfep3re_4rEKA2rrxMbWoLhM4qkAPZvdjyAFF1YNB7u-Uex7l4STRYSpzl4eQIMSl7N0pGlKk58Z1FxTKHsAZTkZYtj40lptma1AfcwRPViNPceCFA36XfVjaratWPYgDhcjK8nNnAhxvL7t25nPbCXazPK1D16ZkY2SwcX7sMx8QklNQdzukZwGyP51yhU8o0PT-u3xGDaXnP6Wh_o35cCIRnar6Id9h2ovViZXJRUzNa8aXedxiFp7hLScSQfIPrPBcqN45EvLjEmtLvzzxNgHY7zw6gzj5QFgv64u2fI9zAY00B9U37bHIZPQnFAhge0466vyXNENzYO3acUGo5HhyFDlcuiBCbIZ413bmfbtRbj5W1FZMY58l1plTmNljfcwQ7Kd0PUGvGEb_jwKlmyKni1HzyyUmElGHdRkyS5Yk7aja6AJOk6z12WUM5snqG-uRVkByxCXQ2KaFR6Qm-IOHMFqzrPINkjyEJgG6KyKfkF5ScVfQaThHBuMrb7ETWMnlk0L28ZszB-TLt3HfaxINsoD5pndSSqBpHB1z_2kEweik3y_PdxIchV3CyBLnmrlqBPBIarCoM8VwzForh3RzQmgNsOGkJXy3Lk0rIr3t-BE7qgXWsrDolobtH6MMFO96Taa6MYIYYYbYvzQYcDuQwD21yGTLwKAnynBDn99bGYEg65LomsJbTSKCYQRcj2Rh2AaScsq8JiHYJEGAnRkgsXNivZnioTlkI9_5XqXBJt3GKMY37YL6qIU3XJ7HapKELorbX4fH_JlOIZLIOIaU80295GZcFdx1I2La9lp_UvAtALEDJklGNvbcDG51NNn2mj_P85vIfzsVdrfX7N_AftR8t1kQx7bdbVLyP4ls_qKnXXigfNJajWtPVgwGeroQq2jtaQw6Tj2Wsi3Itmo5QNg3N_ja3erlEHEdTo8lm4XZHjbl2bAH3n_wuEdYKTt2B3jhj4ddhO8atc1fryqbRfn61YFKUnR3f5vQ1qUq79uisQi0cvHr6DQ_12FpKzFvFckAkQmbk5fcS5Ri5S4L_JtFAexj8n2f_oYuG_9CYYplOU5i1Q84MyXUZ2fZyDhRPRiyjvPJ07_IAiL2ny1btmIFtVPeQSHVwfun-m5Lba3KgnZIJ6CRFqxR-o6F8e5K78d-bAotgA5vOcfeC_fBQfaom-EX0mJyzf7_Wjgam4IA_W3PO_wGoVULHfr7mwUw2Qc-VzITtjajZvRoT-SkVREyJVnKBxHFj2cNTcTtooSxCu115KmbtePf2OwgJocB2jyPT2GK-kIVZtuGdv4Dns8m7Z7FpT7PvjCVjidQqVsJxFQUs7ZzcTXX4vdChdcVK1_OiUwCJKiYN9ZKGA_OmwQJjhnV6gutHFYanu3Gm0m5PQLIM6Pip7yXCw8-Lhq0sxFTl2f-7DeYlfrWpYJyc2wR3T2u-kHl1g52CxX5XoHCV1LsBIU75L23Igsx5_v9ZNeewK2Qh8yWwlWZ18qnXOh5ZJqJUpYtxy3sUhHMG8fe5RKVGbbJyHltosuW26yJ9eUQXqJOSj4wv0YN6a-BezPdkkCqdiUGDHllrgAi96kkeM4jES6iZV60rTgdmfOS03JL0eqCwLnH0gP9t5Xq7l0-GUKp7mrsUjgN4Utb89AVPyIDFWxfJTuFJK-g6Z1EHexEvDY4iTsuOdq9M49_zV6iL8ChB4p5l0lUOiFzIH1Qj-gNMqpU2p0H73cMd3MNeecFhMsCfRadVCO8D4UN39W2mEXDxLBlTN2f4m6aDKjLHi58zyDiJ4aga_ntZRnv56KJpWob0A-4U6_GRJI3OynwKPfb0wpAzK8Gm6Ak1OpxvD-8TmbKUKVA5m55PMoBgGbmvtrwC9Q9Exg-zdotpZ5-iWM1ssnKLXobFa2SmYJeOcGbEjh_yvRvjBRNHxWmq_sP1ohPCTce4Pt97WVQs0Kc_Peb85RqZEnCyjRfV4C3nFUkTxJhFJbmIBPG-VPJDfKEMVTtUe7xB_w55BKViE87CTrd9AAveBFKkBrL3x4bVba3SK5HLU3tOgUPMDnN6Q-waVbjTXsNPGPwJjvbDAN-xwm1V19e-gYcI9sExy4hH9fCkx7Y93e3EMo_EvPA8ZFO8C-KKKlKttqg66BlOp_WYj4GnTpurNDHjVtBQZI6FN5HGl72n7ZL0me0cbfiDkFh7mIfHJCc3EcRCp__gjeBYQaGaR3hh7hLrNeuJB_oAfW5mz-2mIgUYz7vGxRrTeo9CyNYjr5CNwd-hp06pklYw1Bmce-1ZqG87pwFGjfWaAABwqNElf9Wdhgzll31WmopAxDC12Q8pzpAiFfkWWeHsPgvx_Hfm6B50TiIn_LCi8hjTaVUkvtNxZgiKJdAdL5btiUvPwVD10EkysT4dfAEg6l9Z64sZT6oss0KlH55VrNoLOybfx9-TGJZOCB5Zf7JU0KuA1Dpfp8I8TlCOUBwybFvNbECafzsrKB8XI6cmvDtsG0NqNzifoYUPzZDv4eNQhb4QAxlWx_xjiD_G33rgZ27DfvecgVkhDUc317uWUTrWt0f4WJItv4A8P4fj2CDa4cAjfe9CCWM8oehppoxScG5nBFdeebs1m3efdBi0VlYbJTEFQC92q0Zdr7kdfo0FbR1GQ05x66u5b0qLRqqsgF6n72p8Vo_izkLs1N3HKOz9AvcoTmD_tD70rZC6SVJ0jQcKtOMre7VK7lJQJwkuuUL-TRcVYVFeKln1CDs0zpqwza1iCVw3LcUdh96iPsN3XcHrH9pqUxSEY_T6Bq8aSTHdf-iqucTB2u2zsq5BgU40_zXUqvfvQRhcHbaNl0NkdbWKVr8AA3KkoGHMe-H_FBRmyBFX2KL7F4naGngOz7crLfNoSqvPF6cBudn3Q_PPxutj_F5d3FxcfHSHizhfZ3sC39DiwyKgWMU4rUm4gsyYpnKQ90aXtF9nuLWDyNZEb1hsHiVyIoNkUx4IznUY94rrMNzlobnMyqtaic819_vgKqTo9jYA-i5JgMoTPy2DTtIZIpU_p-TggdpEActJ7qncHdQjM0mCrZGYo78NvZ0697fexOgo-dQjfUmNNKNtLp1UQiMBllHwIDZzRePAgOeisXx-E10lLdPUm6AKQcg6IYbvrPDil1_EaQyV3JXqEgNCZwXvkDIh2cnaBLUzyWsep_iupiypilShz_QWLsa467qw5dJ_Q0vR1ulAzMWokaCmTTiJQ6RL71J4WBiGAnobFDrluwJtRr4BTRNDmzDBh8_G3EKTS5mr6W1EQnFnMAqIrW_TOvKasBlQqlLVcFDlllp7X2akyW6Z23NBe7JCnYi4m1HkM0_gnHqnFbnT3cS4umAd22ugbPbo8gn0KRoFxAZK4_4JPHbz_T2itrnDcl4z6_50F-I0Ynp9nHSKnlG5x5V6IGd47vfF7sIks8Vfpq8LUKgQu2g9XHzwf-q_jmwH-eokmnUlznsJnZWnz_GnHyeEMsdQbBKDGK9rjNuMC_QZGuiaARSlSvP7hMyowcczvtuKrN7sZc5nTWPD31UV-TBzrDnzmmsKZ_3BdPebNj71vt31hatE4Mc6OJJUZ-HPPUieICZJFiYXltJBqApmSrscPjDUy9fUoD0iJudrtDu6UJ31SFMkaapXqHG2FqMOwkEk_T5wGmdZ3Y6I8QKKiI4bXRcjPexjVQUJdHnGLRZ4L7tyKU1mFJHMEIMwAer1LiYsnFLHBt3XoAQbNu7LewtB3Cd1Qzeo16USsORAgO8DYtAiu9T0yfTOMAn6lfMgEIdqdJdY5KiPC2TWro08m4pn3mhDlfIWB4XzyBsaO8JiRlF_7Dfg0UpDEnk2AN7ACqsjeUoMk5478ov9JeTPOnCDw2W3x4-w3VhTndvrAD3OrnDoxsc7OcoQjni96fsh-E72g_FKaHJVqH591bROjD5nAPTj5rseGJYDTJkAlHauG67pXgIpMLSLgTUUeplBgjDDTWPdww6dYp8yW9CJNOXCrqmAunZDJe5d0n2BmzGLiOHxiDLKT4-7XoycRXq9adUn2uWgHUooKJbs4BI9sF_28FhxsbN5E2KT4b51aZsK91802OXcSbLCF1K-HqxFNeELpvrlnYT4Cefr_gQydbzRvzSdpTSpUTLtCspJ8RZzJZzQT6x6DnOT8KVt95Sx2OsaE6zCOEBUf5EX85M3Xw630kFQPAjAXx85FxEos_Fl5NtgSssugztP0Q6dbXJTuQrsSgSb6TWsJWaC6agqNudU2lkI3lWpxcxjnYdlDisZE2CVtPXIbMc0fP81Oz9g-LybenNrYDUdIL_FvJ9FM2ZSAleLrEgMM6IyDUXBC8PXA7QP-DukpVBDGkt6GRaNqiqMFN6ceHu1IcTRy8RBucg2ahZqS9BgLlbj2xGANaeZKjzvNKBbtqawx-RkwsHludZ4I6WaLC6-fCnqE0GuK3kw8mbym6gAbfHouwbWarAdtesd3tjoa620dCNg9QwQ1Hzgs4Fd0pjsSgMFB7ahRSDy78GWbdPXZqPollN22uUgQ1zWEva_VxrPK2C05iQMPE4Cg5X8XQbxGH6GY7wuOnc23Pvh9qe6po8voXKaUAPVb3oUGgYx7lgsWZIW-2c861ip10CSjcfFm6AVrS5Iaf7m0OZvUvL52q2r5Py_LWxLbgf0ubrDXMtHbhEGXYmiRaHcmuOxoFUtaqCB6kr6970im-84hk-SiCoL_96yg6Bna4isurYJb2JQgGktUX_SjcB4TshvDDx-G4H-lUYId4AaqP1zNJ0ryUPOlQQXlKtc6s0tUfEZu5SagAE6h6iTUkCr4hJ9hLanj_mMSMJiCjBCFUYMeZ4LCxzQagyR5qWxq90n2HlHhmPy5eCj16DTBIL49ScuUqyN5x8fWt23H-B0VjTUdFEVk8FFaPsSFv0SddM9LEvrlUcA-RcMpyZ1a2ifF7N_HwvRlMslVSzW2BUZRibnB0xpeZlUbqy2CDkr10azuxJYajE6giMCnSHfiKM2EPMuFl5-sX6FiyfsTIA3CmcdoHvd83KNQbMuDRp_nDCAl6LNcj8CXPyenI56CAX5owOddnszaI3C7fEfW7MavfzLE4KlNBPYeFmQLL7w0Rw6uZfqjBKLFLNtXNDBnJW3GSWYKFB0QiAMjf_9_nfOU0w3cwgnfsdmUU0BG_9bXy-qvzYvnWs1eLfoCfruH7gt5Bvb4iGPr37EFMI8yv-df__xbRHgDxVtAd-A5kL8N8DR4U6eSsw5UH8--p3ZpLyrh9vklTf4LDr0xt62qvZjUKzcuMZjQpcBvDHmKRx55aZ8BEkKhIhsWlZJMsE5prZOz3pwT5_pAp1h9u7sw60NnWisoTdBM08hP-JD06OWB_OFPa3I0x1boTojwlzVpiVGUdNeMgpWKa9wQj6rVsEl3-UFIyrKNpgir2df5m5q9vxZ3TSKawBZUAQJnQp9aYgysMtCnUCNrZ-zaC6mojA6cBLk-g5rCcXsyd1WqkHrFwYoPAu5593tQA-ttSVSFCUQxCILqKUZx1-fw2NIchE-NUak3dAZ4RTWm6EwMGXh6rmanI2swBjkglVxw3CfmOAsL7yXEhm2tOofu5EoADa_lEdy8ihHgsCPoJiq4hxlwFXrg9y-Pww0xsrldPuFcc6AAIpPR9UrKHoEdxoDnANqi6zsBTsp-VdyaNj5nFWgpm5mxgnuWUrBGif-U5ldingoapB0hJ50--CucJyHaLHHV2m7WcVDTXiykeHhjIp8qnKBaXwoyezqmLgzIOmhrJgqBH-drr_Usl2Mtzq5lyA2if9MFtKcktldky2GXDiMlwLWSf70Jj5d2Tq8_4P0C-StEZ66V_PkVarTSrFahni4g1OqUJbxxsK7oV932wEtpHJPo6MYnkebOFGlGQlxxk6OlqoZwyP6xbJCwu1_36FuIYcl09Vy11EXK58g62eDyg9eGaAAe1mU4fTByEowy94v67qTC9otmjGlB6K3pyqsGnpWEGz5FujtRPO_ChDNN84qZRMSk5QMU3jx1t3-_0Mg48WnuWGiDDveAhZVxNz9HThbvMviECVk2E70W0nj9XEybPhNLcKQxKMekE6_3qJxUTk8Q7KJpoDysmWznz0Brg-GT6NSJG0pFFfgdNCAK_K5G9SkdnS7Pr2S7DcDIasmdvIciCzw0Lv7Z2rZR8eiica4lrrz52lwO-rD2380r-UgaLsbX5PhZDc57JSkIKnAfS5qeqxK9koKyD4UyR-6gfA2SsoRbxayLigQOk4O6bIDoJ0ECvEoyaJTohselYUhhQDuwaYQ3UVBsxyNzFxXv4i_ldHItOjckkNBLTyz0rfh0bAbqP5ROsXG7Q9OIMglRSRyDjZdlMaRVO1-lXy_zQ8AoBxuAwGOyc2CQm4zvxMjmiAyMymBdZS5w3-aa6ILoaqvhMzYTEyP8pojxTER7-swPs4J_zroRVV-USiheF9qLRr8C7WtZnzaJXYhAap3HrTtfCVMAy8bHYFjk5zYLmVYw09C04sFxQ3apXWUc6A-30D_Ix_x-Nh0LmnhBXc-BiCPdaGA-iqaCScX94nyMmCC5-26KpbcKZL9qH6ZFUez72d2gmu0oX59AgiJkLaVqmWkC2fKMc-GxO7SI-tzO8AGmi2G5vVeYHIwc2zfRhHuNn8znS7SSRKD1uzTLvJzt0Bonu6NLukVUamhLf5SYmHR3Rd9Qb6NNprHYCyJkim-szlYjso9-EhfRqt3TZtuNx7zbXIoJJ_anSIBQKXrWpLJHr4CMexa0xlnVrB_K6-RPVBfKMZWdcRc2cltFzo-w1VvqOzmK_-B_9FkVhCdObJ-aak26s2CjdzF4SmyetdbUt1U6zDx3lb2InlGz544dNY6OjjQX0X0ImPVTlFAFWFO4ONwnUp5GQmqBjYQZ7CYKqx7RZexDqbmGsW33jnf17hPsToLphEo8M6_p9-I4-PSity9OSX3Gw35Lon-myQxV9Nq_Ui9Mtql08C_Uyhdqwk2FH8wESENlRPoM8Dq044Yj0gHqwUj4xqUht8j9Hk-au6fYM_fLPRyXv9jqjT_KklnmfaeAqZOb4hHRIFBxXA9Q-1rreVfYFWVMz-FRiyf3cWd7QQdMciuMBulWeJjgOUoZK3vQIS5f4AZzwYkNOt6F96hwRtdI8bU6tKJU2afAeqNQ6xiSa85tx8_YGzMsrdKdWAPQIKe66QmQLlQP04k_EoSLH_f9dc51NZYfGF8YwzJuvclpqtAeMNguhPuGj-7V5wFMYhni7osiiPQ==',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Providing river crossing safety tips**

When crossing a river as a non-expert, there are some critical guidelines to follow. First, avoid using ropes or attempting to cross in flood conditions or alone. Look for the widest and shallowest sections, and be aware of obstacles like strainers. If fording, link arms with partners for stability, face upstream, and use trekking poles. Always keep three points of contact and evaluate the current's speed before crossing, especially if the water level is above your knees. It's best to err on the side of caution here!\
""",
                        id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Outlining river crossing strategies**

I need to provide some general heuristics for crossing a river, similar to crossing a street. First, choose a safe method like a bridge or ferry. If none are available, assess if crossing is necessary. Then, evaluate conditions such as flow, depth, and width, along with weather and recent rainfall. Choose a spot that's wide, shallow, and free of hazards, and prepare by wearing a PFD, securing gear, loosening backpack straps, and possibly using a wading staff or trekking poles for stability.\
""",
                        id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Detailing river crossing techniques**

When wading across a river, ensure you unbuckle your hip belt, face upstream at a slight angle, and maintain three points of contact while shuffling your feet sideways. Be on the lookout for safe exits and avoid slippery rocks to prevent foot entrapment. If you slip, assume a defensive swimming position by floating on your back with feet upstream. For boats, wear a PFD, cross at a right angle to the flow, and be cautious of obstructions. Special considerations apply for tidal and icy rivers, where experienced guidance is essential.\
""",
                        id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Emphasizing river safety precautions**

It's crucial to mention that you should never tie a rope to yourself when crossing a current, as it could pin you dangerously. Also, avoid running handlines unless you're trained and using proper anchors. If safety ropes are necessary, opt for throw bags and quick-release chest harnesses, again requiring training. If wearing waders, unbuckle the waist belt to reduce drag. I think a good structure would be to mirror street-crossing steps, ensuring everyone is ready and aware, and then ask for more info to tailor the advice further.\
""",
                        id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de',
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
Short version: default to a bridge or ferry; only ford if conditions are clearly safe and you're prepared. Here's a street-crossing-style checklist for rivers.

- Pick the safest "crosswalk": a bridge, ferry, or established ford. If none, ask if you really need to cross or can reroute.
- Stop and assess conditions:
  - Recent rain, snowmelt, tide, or dam releases can spike flow—check local gauges/reports if possible.
  - Watch the current with a leaf/stick. Lots of debris, standing waves, or muddy, opaque water = higher risk.
  - Rule of thumb: fast water above knee depth is often too dangerous to wade.
- Choose your spot and exit:
  - Look for a wide, shallow, straight section with a firm, non-slippery bottom.
  - Avoid rapids, chutes, waterfalls, strainers (downed trees), undercut banks, and blind bends.
  - Identify a primary exit and a backup eddy or beach downstream.
- Prepare yourself and your gear:
  - Wear a PFD if you have one. Keep shoes on for traction; use a wading staff/trekking poles.
  - Loosen or unclip backpack hip and sternum straps so you can shed it if you fall; dry-bag valuables.
  - Do not tie a rope to yourself or stretch a "hand line" across the current unless you're trained and properly anchored.
- If crossing solo on foot:
  - Face slightly upstream; keep three points of contact (two feet and a pole).
  - Shuffle sideways, feeling each step; don't jump rock to rock.
  - Move at a slight downstream angle toward your exit; keep your eyes on where you want to go.
- If crossing as a group:
  - Link up: strongest person upstream, others braced behind/alongside; move in sync and communicate.
  - Alternatively, send one person at a time with spotters ready downstream.
- If you lose your footing:
  - Don't try to stand up in fast, deep current. Float on your back, feet up and pointed downstream, angle toward shore.
  - Avoid strainers; if you're being pushed under or pinned, ditch the pack immediately.
  - Stand only in slow, shallow water.
- Cold water and timing:
  - Cold shock and hypothermia are real. Test water, minimize immersion time, and have dry layers ready.
  - Levels often drop overnight or midday in some systems; waiting can turn a no-go into a safe ford.
- Alternatives:
  - Walk upstream/downstream for a safer braid or gravel bar.
  - Use a bridge, call a local ferry/boat, or turn back. "Not crossing" is often the safest choice.

Tell me your context (on foot vs. in a boat, river width/depth/clarity, current speed, gear you have, group size, weather), and I can tailor a go/no-go and the exact technique.\
""",
                        id='msg_68c42cd36134819c800463490961f7df07460311b0c8d3de',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=314, output_tokens=2737, details={'reasoning_tokens': 2112}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_68c42cb3d520819c9d28b07036e9059507460311b0c8d3de',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_part_from_other_model(
    allow_model_requests: None, anthropic_api_key: str, openai_api_key: str
):
    m = AnthropicModel(
        'claude-sonnet-4-0',
        provider=AnthropicProvider(api_key=anthropic_api_key),
        settings=AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024}),
    )
    agent = Agent(m)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=42,
                    output_tokens=291,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 42,
                        'output_tokens': 291,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_0114iHK2ditgTf1N8FWomc4E',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=OpenAIResponsesModel(
            'gpt-5',
            provider=OpenAIProvider(api_key=openai_api_key),
            settings=OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed'),
        ),
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42d0b5e5c819385352dde1f447d910ad492c7955fc6fc',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=306, output_tokens=3134, details={'reasoning_tokens': 2496}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 23, 30, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68c42ce277ac8193ba08881bcefabaf70ad492c7955fc6fc',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_part_iter(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    responses_model = OpenAIResponsesModel('o3-mini', provider=provider)
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(responses_model, model_settings=settings)

    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for _ in request_stream:
                        pass

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                        signature='gAAAAABoxC0m_QWpOlSt8wyPk_gtnjiI4mNLOryYlNXO-6rrVeIqBYDDAyMVg2_ldboZvfhW8baVbpki29gkTAyNygTr7L8gF1XK0hFovoa23ZYJKvuOnyLIJF-rXCsbDG7YdMYhi3bm82pMFVQxNK4r5muWCQcHmyJ2S1YtBoJtF_D1Ah7GpW2ACvJWsGikb3neAOnI-RsmUxCRu-cew7rVWfSj8jFKs8RGNQRvDaUzVniaMXJxVW9T5C7Ytzi852MF1PfVq0U-aNBzZBtAdwQcbn5KZtGkYLYTChmCi2hMrh5-lg9CgS8pqqY9-jv2EQvKHIumdv6oLiW8K59Zvo8zGxYoqT--osfjfS0vPZhTHiSX4qCkK30YNJrWHKJ95Hpe23fnPBL0nEQE5l6XdhsyY7TwMom016P3dgWwgP5AtWmQ30zeXDs=',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42d26866c819da8d5c606621c911608fbf9b1584184ff',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=1680, details={'reasoning_tokens': 1408}),
                model_name='o3-mini-2025-01-31',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 12, 14, 24, 15, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68c42d0fb418819dbfa579f69406b49508fbf9b1584184ff',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_with_tool_calls(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel(
        model_name='gpt-5',
        provider=provider,
        settings=OpenAIResponsesModelSettings(openai_reasoning_summary='detailed', openai_reasoning_effort='low'),
    )
    agent = Agent(model=m)

    @agent.instructions
    def system_prompt():
        return (
            'You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually '
            "update it as you make progress against the user's prompt"
        )

    @agent.tool_plain
    def update_plan(plan: str) -> str:
        return 'plan updated'

    prompt = (
        'Compose a 12-line poem where the first letters of the odd-numbered lines form the name "SAMIRA" '
        'and the first letters of the even-numbered lines spell out "DAWOOD." Additionally, the first letter '
        'of each word in every line should create the capital of a country'
    )

    result = await agent.run(prompt)

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Compose a 12-line poem where the first letters of the odd-numbered lines form the name "SAMIRA" and the first letters of the even-numbered lines spell out "DAWOOD." Additionally, the first letter of each word in every line should create the capital of a country',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions="You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually update it as you make progress against the user's prompt",
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0c4304d16a4bf24200697cccab25d081a1969a2adbce35d18a',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0c4304d16a4bf24200697cccab25d081a1969a2adbce35d18a',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0c4304d16a4bf24200697cccab25d081a1969a2adbce35d18a',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0c4304d16a4bf24200697cccab25d081a1969a2adbce35d18a',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0c4304d16a4bf24200697cccab25d081a1969a2adbce35d18a',
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='update_plan',
                        args=IsStr(),
                        tool_call_id='call_MZPoaCDhIYRyG4kUPSdtdINs',
                        id='fc_0c4304d16a4bf24200697cccd20f7c81a18d8f8d981765e27e',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=124, output_tokens=2286, details={'reasoning_tokens': 2112}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0c4304d16a4bf24200697cccaacc9081a19d741d6b31bbb790',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_plan',
                        content='plan updated',
                        tool_call_id='call_MZPoaCDhIYRyG4kUPSdtdINs',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions="You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually update it as you make progress against the user's prompt",
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content=IsStr(),
                        id='msg_0c4304d16a4bf24200697cccd335fc81a1bd645d03f7a68f42',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=2423, cache_read_tokens=2304, output_tokens=103, details={'reasoning_tokens': 0}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0c4304d16a4bf24200697cccd2d57481a1992bbe58a5da3992',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_without_summary(allow_model_requests: None):
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[],
                type='reasoning',
                encrypted_content='123',
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', id='rs_123', signature='123', provider_name='openai'),
                    TextPart(content='4', id='msg_123', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )

    _, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        result.all_messages(),
        model_settings=cast(OpenAIResponsesModelSettings, model.settings or {}),
        model_request_parameters=ModelRequestParameters(),
    )
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is 2+2?'},
            {'id': 'rs_123', 'summary': [], 'encrypted_content': '123', 'type': 'reasoning'},
            {
                'role': 'assistant',
                'id': 'msg_123',
                'content': [{'text': '4', 'type': 'output_text', 'annotations': []}],
                'type': 'message',
                'status': 'completed',
            },
        ]
    )


async def test_openai_responses_thinking_with_multiple_summaries(allow_model_requests: None):
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[
                    Summary(text='1', type='summary_text'),
                    Summary(text='2', type='summary_text'),
                    Summary(text='3', type='summary_text'),
                    Summary(text='4', type='summary_text'),
                ],
                type='reasoning',
                encrypted_content='123',
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='1', id='rs_123', signature='123', provider_name='openai'),
                    ThinkingPart(content='2', id='rs_123', provider_name='openai'),
                    ThinkingPart(content='3', id='rs_123', provider_name='openai'),
                    ThinkingPart(content='4', id='rs_123', provider_name='openai'),
                    TextPart(content='4', id='msg_123', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )

    _, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        result.all_messages(),
        model_settings=cast(OpenAIResponsesModelSettings, model.settings or {}),
        model_request_parameters=ModelRequestParameters(),
    )
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is 2+2?'},
            {
                'id': 'rs_123',
                'summary': [
                    {'text': '1', 'type': 'summary_text'},
                    {'text': '2', 'type': 'summary_text'},
                    {'text': '3', 'type': 'summary_text'},
                    {'text': '4', 'type': 'summary_text'},
                ],
                'encrypted_content': '123',
                'type': 'reasoning',
            },
            {
                'role': 'assistant',
                'id': 'msg_123',
                'content': [{'text': '4', 'type': 'output_text', 'annotations': []}],
                'type': 'message',
                'status': 'completed',
            },
        ]
    )


async def test_openai_responses_thinking_with_modified_history(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='low', openai_reasoning_summary='detailed')
    agent = Agent(m, model_settings=settings)

    result = await agent.run('What is the meaning of life?')
    messages = result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the meaning of life?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_07e4c22b60f9810000697ce6d2ba1481a387699e728e73b334',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_07e4c22b60f9810000697ce6d61d5c81a3beb80cd29b725a8f',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=315, details={'reasoning_tokens': 64}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_07e4c22b60f9810000697ce6d2179081a38850cbe01092e148',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts, list)
    response.parts[1] = TextPart(content='The meaning of life is 42')

    with pytest.raises(
        ModelHTTPError,
        match=r"Item '.*' of type 'reasoning' was provided without its required following item\.",
    ):
        await agent.run('Anything to add?', message_history=messages)

    result = await agent.run(
        'Anything to add?',
        message_history=messages,
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort='low',
            openai_reasoning_summary='detailed',
            openai_send_reasoning_ids=False,
        ),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Anything to add?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_03fe771535c7d3c700697ce6da3374819599dada8b8c8884c2',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_03fe771535c7d3c700697ce6df89e88195ba6e046677f0dbc8',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=134, output_tokens=332, details={'reasoning_tokens': 128}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_03fe771535c7d3c700697ce6d9b56c81959aab90d95bd190c4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_with_code_execution_tool(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel(
        model_name='gpt-5',
        provider=provider,
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_summary='detailed',
            openai_reasoning_effort='low',
            openai_include_code_execution_outputs=True,
        ),
    )
    agent = Agent(model=m, builtin_tools=[CodeExecutionTool()])

    result = await agent.run(user_prompt='what is 65465-6544 * 65464-6+1.02255')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='what is 65465-6544 * 65464-6+1.02255',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68cdba57390881a3b7ef1d2de5c8499709b7445677780c8f',
                        signature='gAAAAABozbpoKwjspVdWvC2skgCFSKx1Fiw9QGDrOxixFaC8O5gPVmC35FfE2jaedsn0zsHctrsl2LvPt7ELnOB3N20bvDGcDHkYzjSOLpf1jl2IAtQrkPWuLPOb6h8mIPL-Z1wNrngsmuoaKP0rrAcGwDwKzq8hxpLQbjvpRib-bbaVQ0SX7KHDpbOuEam3bIEiNSCNsA1Ot54R091vvwInnCCDMWVj-9u2fn7xtNzRGjHorkAt9mOhOBIVgZNZHnWb4RQ-PaYccgi44-gtwOK_2rhI9Qo0JiCBJ9PDdblms0EzBE7vfAWrCvnb_jKiEmKf2x9BBv3GMydsgnTCJdbBf6UVaMUnth1GvnDuJBdV12ecNT2LhOF2JNs3QjlbdDx661cnNoCDpNhXpdH3bL0Gncl7VApVY3iT2vRw4AJCU9U4xVdHeWb5GYz-sgkTgjbgEGg_RiU42taKsdm6B2gvc5_Pqf4g6WTdq-BNCwOjXQ4DatQBiJkgV5kyg4PqUqr35AD05wiSwz6reIsdnxDEqtWv4gBJWfGj4I96YqkL9YEuIBKORJ7ArZnjE5PSv6TIhqW-X9mmQTGkXl8emxpbdsNfow3QEd_l8rQEo4fHiFOGwU-uuPCikx7v6vDsE-w_fiZTFkM0X4iwFb6NXvOxKSdigfUgDfeCySwfmxtMx67QuoRA4xbfSHI9cctr-guZwMIIsMmKnTT-qGp-0F4UiyRQdgz2pF1bRUjkPml2rsleHQISztdSsiOGC2jozXNHwmf1b5z6KxymO8gvlImvLZ4tgseYpnAP8p_QZzMjIU7Y7Z2NQMDASr9hvv3tVjVCphqz1RH-h4gifjZJexwK9BR9O98u63X03f01NqgimS_dZHZUeC9voUb7_khNizA9-dS-fpYUduqvxZt-KZ7Q9gx7kFIH3wJvF-Gef55lwy4JNb8svu1wSna3EaQWTBeZOPHD3qbMXWVT5Yf5yrz7KvSemiWKqofYIInNaRLTtXLAOqq4VXP3dmgyEmAZIUfbh3IZtQ1uYwaV2hQoF-0YgM7JLPNDBwX8cRZtlyzFstnDsL_QLArf0bA8FMFNPuqPfyKFvXcGTgzquaUzngzNaoGo7k6kPHWLoSsWbvY3WvzYg4CO04sphuuSHh9TZRBy6LXCdxaMHIZDY_qVB1Cf-_dmDW6Eqr9_xodcTMBqs6RHlttLwFMMiul4aE_hUgNFlzOX7oVbisIS2Sm36GTuKE4zrbkvsA==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'container_id': 'cntr_68cdba56addc81918f656db25fd0a6800d6da575ea4fee9b',
                            'code': """\
# compute the value
65465 - 6544 * 65464 - 6 + 1.02255
""",
                        },
                        tool_call_id='ci_68cdba5af39881a393a01eebb253854e09b7445677780c8f',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed', 'logs': ['-428330955.97745']},
                        tool_call_id='ci_68cdba5af39881a393a01eebb253854e09b7445677780c8f',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68cdba63843881a3a9c585d83e4df9f309b7445677780c8f',
                        signature='gAAAAABozbpoJefk0Fp1xqQzY6ego00t7KnH2ohbIw-rR9ZgaEAQs3n0Fubka6xbgRxzb1og6Xup1BuT8hQKMS-NHFxYsYXw4b6KeSbCd5oySVO53bsITEVk0A6tgjGssDJc1xSct1ORo-nCNV24MCNZvL9MKFeGQHP-jRypOZ9Vhepje87kFWTpw9lP9j54fZJdRIBGA9G_goI9m1cPztFUufcUxtLsgorsM053oxh8yWiEccAbvBaGXRlPWSoZYktbKrWeBVwiRt2ul-jRV43Z3chB32bEM1l9sIWG1xnvLE3OY6HuAy5s3bB-bnk78dibx5yx_iA36zGOvRkfiF0okXZoYiMNzJz3U7rTSsKlYoMtCKgnYGFdrh0D8RPj4VtxnRr-zAMJSSZQCm7ZipNSMS0PpN1wri14KktSkIGZGLhPBJpzPf9AjzaBBi2ZcUM347BtOfEohPdLBn8R6Cz-WxmoA-jH9qsyO-bPzwtRkv28H5G6836IxU2a402Hl0ZQ0Q-kPb5iqhvNmyvEQr6sEY_FN6ogkxwS-UEdDs0QlvJmgGfOfhMpdxfi5hr-PtElPg7j5_OwA7pXtuEI8mADy2VEqicuZzIpo6d-P72-Wd8sapjo-bC3DLcJVudFF09bJA0UirrxwC-zJZlmOLZKG8OqXKBE4GLfiLn48bYa5FC8a_QznrX8iAV6qPoqyqXANXuBtBClmzTHQU5A3lUgwSgtJo6X_0wZqw0O4lQ1iQQrkt7ZLeT7Ef6QVLyh9ZVaMZqVGrmHbphZK5N1u8b4woZYJKe0J57SrNihO8Slu8jZ71dmXjB4NAPjm0ZN6pVaZNLUajSxolJfmkBuF1BCcMYMVJyvV7Kk9guTCtntLZjN4XVOJWRU8Db5BjL17ciWWHGPlQBMxMdYFZOinwCHLIRrtdVxz4Na2BODjl0-taYJHbKd-_5up5nysUPc4imgNawbN2mNwjhdc1Qv919Q9Cz-he9i3j6lKYnEkgJvKF2RDY6-XAI=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
Using standard order of operations (multiplication before addition/subtraction):

65465 - 6544 * 65464 - 6 + 1.02255 = -428,330,955.97745

If you intended different grouping with parentheses, let me know.\
""",
                        id='msg_68cdba6652ac81a3a58625883261465809b7445677780c8f',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1493, cache_read_tokens=1280, output_tokens=125, details={'reasoning_tokens': 64}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 20, 17, 21, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68cdba511c7081a389e67b16621029c609b7445677780c8f',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run(user_prompt='how about 2 to the power of 8?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about 2 to the power of 8?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
**Calculating a power of two**

I need to answer the question simply: 2 raised to the power of 8 equals 256. It's straightforward, and I want to keep it concise for clarity. I know users appreciate quick answers, so I'll just present the number without any extra elaboration. 256 is the answer, and it's important to communicate it efficiently!\
""",
                        id='rs_68cdba6c100481a394047de63f3e175009b7445677780c8f',
                        signature='gAAAAABozbpuOXVfjIYw7Gw6uSeadpkyaqMU1Frav7mTaf9LP8p8YuC8CWR9fYa02yZ5oYr1mqmYraD8ViOE33zqO2HBCdiWpOkVdNX-s4SGuPPB7ewyM7bDD4XbaSzo-Q5I6MgZmvVGWDGodqa3MfSKKNcGyD4aEfryQRLi4ObvHE5yuOqRo8FzGXMqe_pFdnvJXXD7njyfUofhWNvQPsLVLQFA_g_e7WKXtJJf_2JY183oi7-jNQ6rD9wGhM81HWSv0sTSBIHMpcE44rvlVQMFuh_rOPVUHUhT7vED7fYtrMoaPl46yDBc148T3MfXTnS-zm163zBOa34Yy_VXjyXw04a8Ig32y72bJY7-PRpZdBaeqD3BLvXfMuY4C911Z7FSxVze36mUxVO62g0uqV4PRw9qFA9mG37KF2j0ZsRzfyAClK1tu5omrYpenVKuRlrOO6JFtgyyE9OtLJxqvRNRKgULe2-cOQlo5S74t9lSMgcSGQFqF4JKG0A4XbzlliIcvC3puEzObHz-jArn_2BVUL_OPqx9ohJ9ZxAkXYgf0IRNYiKF4fOwKufYa5scL1kx2VAmsmEv5Yp5YcWlriB9L9Mpg3IguNBmq9DeJPiEQBtlnuOpSNEaNMTZQl4jTHVLgA5eRoCSbDdqGtQWgQB5wa7eH085HktejdxFeG7g-Fc1neHocRoGARxwhwcTT0U-re2ooJp99c0ujZtym-LiflSQUICi59VMAO8dNBE3CqXhG6S_ZicUmAvguo1iGKaKElMBv1Tv5qWcs41eAQkhRPBXQXoBD6MtBLBK1M-7jhidVrco0uTFhHBUTqx3jTGzE15YUJAwR69WvIOuZOvJdcBNObYWF9k84j0bZjJfRRbJG0C7XbU=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='256', id='msg_68cdba6e02c881a3802ed88715e0be4709b7445677780c8f', provider_name='openai'
                    ),
                ],
                usage=RequestUsage(input_tokens=793, output_tokens=7, details={'reasoning_tokens': 0}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_68cdba6a610481a3b4533f345bea8a7b09b7445677780c8f',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_thinking_with_code_execution_tool_stream(
    allow_model_requests: None, openai_api_key: str
):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel(
        model_name='gpt-5',
        provider=provider,
        settings=OpenAIResponsesModelSettings(openai_reasoning_summary='detailed', openai_reasoning_effort='low'),
    )
    agent = Agent(model=m, builtin_tools=[CodeExecutionTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt="what's 123456 to the power of 123?") as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="what's 123456 to the power of 123?",
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c3509b2ee0819eba32735182d275ad0f2d670b80edc507',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n = pow(123456, 123)\\nlen(str(n))"}',
                        tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed'},
                        tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"str(n)[:100], str(n)[-100:]"}',
                        tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed'},
                        tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n"}',
                        tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed'},
                        tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c350a75ddc819ea5406470460be7850f2d670b80edc507',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=3727, cache_read_tokens=3200, output_tokens=347, details={'reasoning_tokens': 128}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 11, 22, 43, 36, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68c35098e6fc819e80fb94b25b7d031b0f2d670b80edc507',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='', id='rs_68c3509b2ee0819eba32735182d275ad0f2d670b80edc507', provider_name='openai'
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='**Calcul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' large')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' integer')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
**

I\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' compute')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 123')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='456')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' raised')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' power')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 123')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' That')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' enormous')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' integer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' probably')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wants')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' exact')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' value')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Python')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ability')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' handle')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' big')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' integers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' output')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' will')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' likely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' extremely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' long')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' —')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' potentially')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' hundreds')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' digits')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' consider')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' prepare')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' return')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' result')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' as')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' plain')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' text')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ends')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' up')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' being')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' around')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 627')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' digits')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' go')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ahead')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' compute')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='!')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(signature_delta=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content=IsStr(),
                    id='rs_68c3509b2ee0819eba32735182d275ad0f2d670b80edc507',
                    signature='gAAAAABow1CfwMTF6GjgPzWVr8oKbF3qM2qnldMGM_sXMoJ2SSXHrcL4lsIK69rnKn43STNM_YZ3f5AcwxF4oThzCOPl1g9-u4GGFd5sISVWJYruCukTVDPaEEzdmJqCU1JMSIZvlvqo7b5PsUGyQU5ldX4KXDq8zs4NmRyLIJe-34SCmDG3BYVWR_O-CtcjH0tF9e3XnJ5T9TvxioDEGbASqXMKx5XB9P_b1ser8P9WIQk6hxZ8YX-FAmWSt-sad-zScdeTmyPcakDb7Z4NVcXmL_I-hoQYH_lu-HPFVwcXU8R7yeXU-7YF3vZBE84cmFuv25lftyojbdGq2A7uxGJZBPMCoUBDGBNG2_7mVvKyGz_ZZ6vXIO0GVDhHdW4Y012pkoDfLp6B-B9CGvANOH3ORlcbhB8aT9qN5bY773wW44JIxRU3umkmNzwF7lkbmuMCbGybHYSzqtkOrMIRgqxaXOx3bGbsreM4kGwgD3EXWqQ1PVye_K7gRkToVQpfpID5iuH4jJZDkvNjjJI09JR2yqlR6QkQayVg2x1y8VHXoMYjNdQdZeP62AguqYbgrlBRcjaUnw78KcWscQHaNsg0MfxL_5Q-pZR1OPVsFppHRTzrVK8458d05yEhDmun345oI9ScBrtXFRdHXPy0dQaayfjxM9H0grPrIogMw_zz4jAcFqWxE_C7GPMnNIJ_uEAhkPOetpNb-izd-iY4pGYKs8pmCB5czrAlKC1MXTnowrlWcwf5_kuD5SzWlzlWOoKWCeBDOZuKTDVJKXh_QCtQfftomQazDFCiCSgaQMuP7GaPcDuS1jdQoMQBcFfKuWoq-3eQBOCiEOAERH81zR4hz1x02T_910jGreSpfgxSqt4Td0pDDSmlEV6CwaUDQvrPc67d8_Wtx8YKv4eBH544_p1k9T8tHo3Q7xvgE37ZCdd_AVhC2ed1b5oUI95tM570HAVugFilcHJICa1RbFzIlRkNgI4k2JvsVWtD5_h3x6ZaEFTomwIXlochYgsegh8RJIRRCNKO9ebsvTrkdl8n1mb3hLrz7puwCkRFyUkxYBGT9zUjuKrjp_IjTvvov29v6pwYHg2Xd0nAfLP4WWWPBLNx3oV1-yOfXStRGHMZTB6iN9d0Bxi2QS7dk-rPPXml5HxrSo1TG06EdBXQ1VgrkWIxG1TF97-gK9oWWT9S5aaYKZAOdaqDvi7qO8I-4VwExtIq4Do3BHnWrgKNHfyuAobQK4H_CFMElYibJHwA9t-UGujMic07AxS-2XjXaCtjf7LnW_aXE2rQDqzHiTiLmTqT6jYHP0WHGSqFTOFkNmzqy6uVfU-TbdT91zDBeesc8XpzCXWBVKqxEzuQGdJrYk6ieZaxL76Kjs4jyo838LMJCXzhcF8enukz_llnoxAV59hTDAn0MUQvstGlDX0ToI7C8Oc0NZfZU5Pi4gs8u0He_Nw5UsoV7sA-jk4M45sFt6g3u00kJFP3gIcdvOzHcRK5z3Sfb9JF0bnvIYSbUFUidEJxSOAcRlxofOJPnkPtWCYiiv3zSVxZXX77-wtc8yrOYFzH1k_8P6CDpcfzOW7Yl1Tajgcm20nygmPlFtXF3RNFPztW1V5GwQHc99FvT4ZAex3fQ_UBDKyXnyGoySgpZbHQIvhzUhDEGm77EiYw5FoF6JgnHGGUCbfXr2EudtpbGW8MRHop2ytonb8Hq7w10yQSginBbH_w3bwtd7cwgDKcp6wIPotjpEC-N1YDsRqhPuqxVA==',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='n', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' pow', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='123', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='456', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='123', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\n', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='len', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(str', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(n', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='))', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n = pow(123456, 123)\\nlen(str(n))"}',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-return',
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='str', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='(n', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=')', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='[:', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='100', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='],', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=' str', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='(n', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=')[', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='-', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='100', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=':]', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartEndEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"str(n)[:100], str(n)[-100:]"}',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=5,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-return',
            ),
            PartDeltaEvent(
                index=5,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                ),
            ),
            PartDeltaEvent(
                index=5,
                delta=ToolCallPartDelta(
                    args_delta='n', tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=5,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507'
                ),
            ),
            PartEndEvent(
                index=5,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n"}',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=6,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=7,
                part=TextPart(
                    content='123', id='msg_68c350a75ddc819ea5406470460be7850f2d670b80edc507', provider_name='openai'
                ),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='456'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='^'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='123'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta=' equals'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta=':\n'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='180'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='302'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='106'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='304'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='044'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='807'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='508'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='140'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='927'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='865'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='938'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='572'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='807'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='342'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='688'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='638'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='559'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='680'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='488'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='440'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='159'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='857'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='958'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='502'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='360'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='813'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='732'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='502'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='197'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='826'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='969'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='863'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='225'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='730'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='871'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='630'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='436'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='419'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='794'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='758'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='932'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='074'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='350'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='380'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='367'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='697'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='649'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='814'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='626'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='542'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='926'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='602'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='664'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='707'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='275'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='874'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='269'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='201'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='777'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='743'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='912'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='313'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='197'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='516'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='323'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='690'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='221'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='274'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='713'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='845'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='895'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='457'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='748'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='735'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='309'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='484'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='337'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='191'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='373'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='255'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='527'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='928'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='271'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='785'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='206'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='382'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='967'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='998'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='984'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='330'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='482'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='105'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='350'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='942'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='229'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='970'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='677'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='054'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='940'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='838'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='210'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='936'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='952'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='303'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='939'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='401'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='656'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='756'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='127'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='607'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='778'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='599'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='667'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='243'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='702'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='814'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='072'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='746'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='219'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='431'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='942'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='293'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='005'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='416'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='411'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='635'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='076'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='021'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='296'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='045'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='493'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='305'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='133'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='645'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='615'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='566'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='590'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='735'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='965'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='652'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='587'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='934'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='290'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='425'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='473'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='827'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='719'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='935'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='012'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='870'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='093'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='575'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='987'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='789'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='431'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='818'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='047'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='013'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='404'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='691'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='795'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='773'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='170'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='405'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='764'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='614'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='646'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='054'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='949'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='298'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='846'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='184'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='678'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='296'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='813'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='625'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='595'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='333'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='311'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='611'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='385'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='251'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='735'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='244'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='505'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='448'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='443'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='050'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='050'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='547'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='161'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='779'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='229'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='749'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='134'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='489'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='643'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='622'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='579'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='100'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='908'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='331'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='839'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='817'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='426'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='366'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='854'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='332'),
            ),
            PartDeltaEvent(
                index=7,
                delta=TextPartDelta(content_delta='416'),
            ),
            PartEndEvent(
                index=7,
                part=TextPart(
                    content="""\
123456^123 equals:
180302106304044807508140927865938572807342688638559680488440159857958502360813732502197826969863225730871630436419794758932074350380367697649814626542926602664707275874269201777743912313197516323690221274713845895457748735309484337191373255527928271785206382967998984330482105350942229970677054940838210936952303939401656756127607778599667243702814072746219431942293005416411635076021296045493305133645615566590735965652587934290425473827719935012870093575987789431818047013404691795773170405764614646054949298846184678296813625595333311611385251735244505448443050050547161779229749134489643622579100908331839817426366854332416\
""",
                    id='msg_68c350a75ddc819ea5406470460be7850f2d670b80edc507',
                    provider_name='openai',
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n = pow(123456, 123)\\nlen(str(n))"}',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"str(n)[:100], str(n)[-100:]"}',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n"}',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )


async def test_openai_responses_streaming_usage(allow_model_requests: None, openai_api_key: str):
    class Result(BaseModel):
        result: int

    agent = Agent(
        model=OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key)),
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort='low',
            openai_service_tier='flex',
        ),
        output_type=Result,
    )

    async with agent.iter('Calculate 100 * 200 / 3') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as response_stream:
                    async for _ in response_stream:
                        pass
                    assert response_stream.get().usage == snapshot(
                        RequestUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448})
                    )
                    assert response_stream.usage() == snapshot(
                        RunUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448}, requests=1)
                    )
                    assert run.usage() == snapshot(RunUsage(requests=1))
                assert run.usage() == snapshot(
                    RunUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448}, requests=1)
                )
    assert run.usage() == snapshot(
        RunUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448}, requests=1)
    )


async def test_openai_responses_non_reasoning_model_no_item_ids(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4.1', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model)

    @agent.tool_plain
    def get_meaning_of_life() -> int:
        return 42

    result = await agent.run('What is the meaning of life?')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the meaning of life?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_meaning_of_life',
                        args='{}',
                        tool_call_id='call_otDUd1BvDW9o3wOIccaWfU5S',
                        id='fc_03b12740d1d3962e00697cccd4ea5c819ea3cdfdafadcdc2d2',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=36, output_tokens=15, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_03b12740d1d3962e00697cccd48a04819ea4edd6cdda5957ae',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_meaning_of_life',
                        content=42,
                        tool_call_id='call_otDUd1BvDW9o3wOIccaWfU5S',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="The meaning of life, according to Douglas Adams' famous novel \"The Hitchhiker's Guide to the Galaxy,\" is 42. This has become a humorous and philosophical answer, often cited in popular culture. If you're looking for a deeper or different perspective, feel free to ask!",
                        id='msg_024fab3cae04892000697cccd5a488819683ce43e6d5f621ef',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=61, output_tokens=59, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_024fab3cae04892000697cccd53470819699bfc0debde05e0e',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    _, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        messages,
        model_settings=cast(OpenAIResponsesModelSettings, model.settings or {}),
        model_request_parameters=ModelRequestParameters(),
    )
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is the meaning of life?'},
            {
                'name': 'get_meaning_of_life',
                'arguments': '{}',
                'call_id': 'call_otDUd1BvDW9o3wOIccaWfU5S',
                'type': 'function_call',
            },
            {'type': 'function_call_output', 'call_id': 'call_otDUd1BvDW9o3wOIccaWfU5S', 'output': '42'},
            {
                'role': 'assistant',
                'content': "The meaning of life, according to Douglas Adams' famous novel \"The Hitchhiker's Guide to the Galaxy,\" is 42. This has become a humorous and philosophical answer, often cited in popular culture. If you're looking for a deeper or different perspective, feel free to ask!",
            },
        ]
    )


async def test_openai_responses_code_execution_return_image(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
    )

    agent = Agent(model=model, builtin_tools=[CodeExecutionTool()], output_type=BinaryImage)

    result = await agent.run('Create a chart of y=x^2 for x=-5 to 5')
    assert result.output == snapshot(IsInstance(BinaryImage))
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Create a chart of y=x^2 for x=-5 to 5',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_68cdc38812288190889becf32c2934990187028ba77f15f7',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'container_id': 'cntr_68cdc387531c81938b4bee78c36acb820dbd09bdba403548',
                            'code': """\
import numpy as np\r
import matplotlib.pyplot as plt\r
\r
# Data\r
x = np.arange(-5, 6, 1)\r
y = x**2\r
\r
# Plot\r
plt.figure(figsize=(6, 4))\r
plt.plot(x, y, marker='o')\r
plt.title('y = x^2 for x = -5 to 5')\r
plt.xlabel('x')\r
plt.ylabel('y')\r
plt.grid(True, linestyle='--', alpha=0.6)\r
plt.xticks(x)\r
plt.tight_layout()\r
\r
# Save and show\r
plt.savefig('/mnt/data/y_equals_x_squared.png', dpi=200)\r
plt.show()\r
\r
'/mnt/data/y_equals_x_squared.png'\
""",
                        },
                        tool_call_id='ci_68cdc39029a481909399d54b0a3637a10187028ba77f15f7',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ci_68cdc39029a481909399d54b0a3637a10187028ba77f15f7',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed', 'logs': ["'/mnt/data/y_equals_x_squared.png'"]},
                        tool_call_id='ci_68cdc39029a481909399d54b0a3637a10187028ba77f15f7',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68cdc398d3bc8190bbcf78c0293a4ca60187028ba77f15f7',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2973, cache_read_tokens=1920, output_tokens=707, details={'reasoning_tokens': 512}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 20, 56, 34, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68cdc382bc98819083a5b47ec92e077b0187028ba77f15f7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run('Style it more futuristically.', message_history=messages)
    assert result.output == snapshot(IsInstance(BinaryImage))
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Style it more futuristically.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_68cdc39f6aa48190b5aece25d55f80720187028ba77f15f7',
                        signature='gAAAAABozcPV8NxzVAMDdbpqK7_ltYa5_uAVsbnSW9OMWGRwlnwasaLvuaC4XlgGmC2MHbiPrccJ8zYuu0QoQm7jB6KgimG9Ax3vwoFGqMnfVjMAzoy_oJVadn0Odh3sKGifc11yVMmIkvrl0OcPYwJFlxlt2JhPkKotUDHY0P2LziSsMnQB_KaVdyYQxfcVbwrJJnB9wm2QbA3zNZogWepoXGrHXL1mBRR3J7DLdKGfMF_7gQC5fgEtb3G4Xhvk8_XNgCCZel48bqgzWvNUyaVPb4TpbibAuZnKnCNsFll6a9htGu9Ljol004p_aboehEyIp6zAm_1xyTDiJdcmfPfUiNgDLzWSKf-TwGFd-jRoJ3Aiw1_QY-xi1ozFu2oIeXb2oaZJL4h3ENrrMgYod3Wiprr99FfZw9IRN4ApagGJBnWYqW0O75d-e8jUMJS8zFJH0jtCl0jvuuGmM5vBAV4EpRLTcNGOZyoRpfqHwWfZYIi_u_ajs_A6NdqhzYvxYE-FAE1aJ89HxhnQNjRqkQFQnB8sYeoPOLBKIKAWYi3RziNE8klgSPC250QotupFaskTgPVkzbYe9ZtRZ9IHPeWdEHikb2RP-o1LVVO_zFMJdC6l4TwEToqRG8LaZOgSfkxS8eylTw7ROI2p8IBSmMkbkjvEkpmIic0FSx23Ew_Q-Y6DPa9isxGZcMMS0kOPKSPSML2MGoVq5L3-zIVj6ZBcFOMSaV5ytTlH-tKqBP9fejMyujwQFl5iXawuSjVjpnd2VL83o-xKbm6lEgsyXY1vynlS2hT52OYUY3MMvGSCeW5d7xwsVReO0O1EJqKS0lLh8thEMpJvar9dMgg-9ZCgZ1wGkJlpANf2moQlOWXKPXcbBa2kU0OW2WEffr4ecqg1QwPoMFLmR4HDL-KknuWjutF5bo8FW0CAWmxObxiHeDWIJYpS4KIIwp9DoLdJDWlg8FpD6WbBjKQN6xYmewHaTLWbZQw8zMGBcnhAkkyVopjrbM_6rvrH4ew05mPjPRrq9ODdHBqDYEn1kWj9MBDR-nhhLrci_6GImd64HZXYo0OufgcbxNu5mcAOsN3ww13ui8CTQVsPJO20XHc4jfwZ2Yr4iEIYLGdp0Xgv8EjIkJNA1xPeWn9COgCRrRSVLoF6qsgZwt9IRRGGEbH6kvznO_Y7BTTqufsORG6WNKc_8DDlrczoZVy0d6rI1zgqjXSeMuEP9LBG-bJKAvoAGDPXod8ShlqGX3Eb9CmBTZtTOJZYdgAlsZHx9BZ6zHlrJDjSDhc8xvdUAn9G3JvTI3b5JWSNX0eEerZ4c0FVqlpR-mSG201qnFghtoGHTLJhlIf9Ir8Daio_AYxUTRarQbcKnJuyKHPOz1u0PX2zS0xegO-IZhFbzNaB8qwQgeBiHfP-1dP9mkttqIRMt-hMt9NMHXoGIvFxgQ-xUVw7GRWx-ffKY7nPAbZD8kwVP3i4jTVj8phhwQcDy9UmbaPjm4LBgJkfdwNfSpm3g_ePK4aLa_l7iF2WSSfy2wObb7VatDzYDcNRG0ZTMGsiHy8yzZAcec18rG7uE6QCKx32G8NI5YvcN1kbnrZEuoKTBuSb2B_ZAhvED9HxbG8mH4ZEHHioVuH3_-b2TesVUAbORab_-rG9CU6qyy_eAqP54FYiXXSWtBWNo4baVdqCzgSCiNxgpxx64WPw8y2M1bOMoV6KPGwDOjcNwbO9nQwztqTWPW0Ot_Llf0HV0p-RPC1Uy8uBB5flhJ3p5uqxCPV3kDRzXgjh28EaBEkaSw_6SZkJNvwbD_7VihlHGaO89TwlqSIYUT_gc72NZKRrj4f-Y-0NwxjaSVVGuWCoeG-TMjG6uXpSozo2J47_x_a0lr4KCT8NDYlksajyuPUbYhC7jhQ9uJakmAc7ay_VHn_LYlAWRdAA7wYvqw7aYIuSIYg2OfL6NlggCpBnhsUPEXmMRHcfj1Ctc1aeUjBcpLFVmTZ82lB0FdcKRe3bBsKRckbdKalehoK0NJtrWqNQQH7xPrS-r7or_oOWhA4EDIkRUOG9eZhdsvTXBUamxGwutJ97SdDkgppVC4M7DMK2ZGGBzQsE-JMilERvFQ8JqwVWPxExWmE_-H2-bYe-T-CguCin-mTqhLYswHVtXjtruoHBmDs2SdnkD3intwSpqxsltscCfRaoRYWTCTbchCdbctSEIc39ECpc5tL1Gnav0bwSkMYkxyaRVBiYBbmIG9JftkKIYtdZ_Ddjmq8k29QflqrcigahsVLZPye3dxVTuviqbQjRd2SPMv8RxgSebgm5RZZIpP4WposryghYZFvuA1WImRzsImnAJI9J-8dv6IhHpHsWOw9K-Neg8GlnDU1mGHUElMUbqHiLojmXqPGfhBI3iSR0Ugs7ErpeRUrSk3il2o3rysG1Fn7ePuP5qNJUt2NyBUxf3TExMOwG_zqvpIPr2V_ARr3PsfeD0IcY83Bh428S8KPzc7ASOjT9dGQtVVrdjSxHi8o5ANxGx6z3bHC5dJvDCXg8a7FIJHAd5CUqJxrBi-K4p21jf1BNqgO5JAJO1JrvtdTk4GOVe8YEfhxmGWW9oeuRg8crsIWCCCoxr2XJKgPCj2TTPkBDZ1O3Yw3_nuWaBU5sB09uEB5lTKMd0OfSHbPF4c50RWAFgQB-tHjIUss3oEcAUaZHC77r6sIYoAEBlU8Dgly983fFD0HCqtpIpKS_B_K1fTXYpWRM3uUZpPKEgbfw1Kiqp5cweKTeRKNvjlau6VxhPyVi66xPdHUCC_BcX1eeFe-zcxe6fczcJWqGZGtYyVS_S_GlWZcdA6AHvGU6c4KjG0oU_9q-pdHSRtpnrhqFu2L884m64A_HsFU71Dj34AxhmXO1Am-zSL3j9nEPPUe6lJSGyhHU9k8ApDadWagvlODdXYWaWiMCXGXcYtl_iUAm24IJozlLJ1IW9HW6RoTfKrxwQwND3pX9CLNewuPV776pVtRjvUMbLaYg8nzOu1eNT2IW9dUdzc7wqOjiT1gHuVd6RzJyTCWJb9yPwDTkB_NKkjfUPmJ9Id924xtxy6H0eDYRq-SqsSSEklr6KJc88PV35QqvaMUW1dt_tGynHgYy9PXlWXQLKw-Xphku3FS_R4BLUhJbXDsMOQq332yhizP3qQ7vjEmPm8KB4DMIWBNn_D9xFuDuTCMNPAA9AGYWgC39-L4wPbpBHpqWjDwMzijFpm0CEViPD9ghyyV8syT1uLscxJVVDlBx90u_qWLSzMnFrVWmZ60OyWa9EqG44ZU8ELLHlEDRO_yHuTVpSafCLeDe5baOG2mI6tZnDBmm_ysbYdaC2N_zNBK9rhx7g7BNLQPevl0vtZm7GVLYXiVaO5ZinHxeTyJ6dRU5b0HmSw8r7EpdgORfjUuMkUfWPwhXgTU8SbvjTZg1gJowyNDYCvacrgnmnpBG9BgNjsfWlGTwz19AcEP_GjCWRWoE-uE_5fIyq5eFEefCBUKU0Ejs0IB-Re5h8bbdc6bNV3Tnx4UfGDU6FbQrJmPzrw5wp_wCeVYjtNGRbO2MKr_m52km5xMpVMMHtthVbQ9Zsa9F9zB6Dkr-R4F7o0dITMhG3qaREHKc8mXIGoHND-WSGPZLntB43JmRIWwjlJNstv7VlVc-dU89oh6Z1biH9B88SENI1ao2wMQV-BB17E6cmfzm1JsSR-HkzSf3yoUJWwvIu4CaR4jeMZohuoNqfGvQWIJSfyyUNzq5uY5__04QUmNcRVspOTH4EOHAoXLfCV3VI7fodj4FppiIuIXKwS3N03-Qt4sQ__XQWuyDdORvhRJeCvYcK5kkyOQILcABxDItxLmk8AgdT0Hz0BAo_u1U71srS-T8a8O0-fXWsJAHxDg_rJn0LUm6zq2vXNl8zmOKwEayyb0YySbMRxI-LwLyOXGRDyAVvm_7KKJu1HHqMntLyY2G1xowFpwMVLYXlGxDbsSpE-g5kFnHWhj13FiekLxaFgMRNsMA-r5_rWbEjRa6H328FKsUJcYe9qsp2LlzdJmYZDTIMgzxupFwQ-R5F6QjWOudMBsRszb4YqnOPJ8P9YnY2WYd0B7srb5Gh7T6r6mcCl-HAb2z9QDeXOc2Lu7ujuSvGj7_Gk7PkZH-LzoAEaGG9Z-7IVJlV_hOBPif3GlJUSUhTlIwWxn75gOyoOFuMak-rQqkb0SaL5anfXS_NUTVgSh5G5JQIoykLxbVlGiyeq0M_oEvTw2wMZcWT2hhaudcQ6L912pntcD-WF2tfppgp6sN5-cq-D8Y39N5Txvs-wo-H7-vYKPozTNUKCfnzgXfvt5fOi3RBR4MZU3eHT8OZ7d1d3otho_4GVMNIFa6mxjW1BC_J42Hn27-vrNDLZI_BXdF1t2CCq9VeRwxIW1R9vadd04HzAXyhap95BAYacmbULR6BkX97TvY3hv5cMiaQFkzxg-tf-nGC_VCknvwKxu4ocoB14p9w5TPSKcJz4J26XvyQbi6AdaXbOk625ajB_clv3VJvXYz7DgvWZd408tMykYQLMEyv5lnS7qwQokeM4ilIXwM7EugiakhfefTM9ZdxaWVcvQdqGerx98wlhifCSv0FqFRpJdkqgHmV1qzrAjPDEKT5HJOjsvs5hb7gKBqHR-bYlgS94pvDUpPArQXYcGYGum6vFsCAJypefMTF3D7Zhu4hhWQQv-DzSmfcZOxSeVJFrgVeqJnIbZPtd59HCBXNIRXJa42wUYE4szNli8wKWX0rYSIhiX-ig2YYZz3ZoBE1KDOpzheuk9OMYg7tQG2UlmVq27ggaKJ2gEGuVv-GI7uD7vKxPQ97QwCf38gWKU95CjMEBm_EvmLs9eubNpSpz8Yoek8hWWgrCXUSwRsYnF-lGdG0nIkCClvzqqAGOjyPxG4qfrCXJ-4rVc4DQiJUj71_I0EAhOgxb5WYBt4a7C1aUxC__qeOTAecof-UjzNlUPTo91JgOh5xvZkRkgGFNsq1OFqOcRrrKV8U8brizYkIhDjzjwCIzScSYvEfY4S6st-oJBv5fwTqwICSs59hf6WR8GXsPFR4v3UtF0Rkt-Nrek-X6V7BCui1M5HeFRN7lcTYs1Qw2bIwu4Td5PIkZ16oHdCk9u5pEZce-n_MIwj2Yoq_Lq1BBY9f1rpG9IuaycwabFnd2MOj89-xdgC197DAij5WjZjXahooyAl0Mt3p9MrHCit7LYbxqd_dGBOmg9YRfGPhsoZ17oAmHyg_gvpooOsu21T_06ynhvySjOG0yUcphquvtHJWqQdcT6BBX0X-kGE4nA41VdMhepLhDRDXtR4HJ1m_dPFpkHeAAFIefjt5Kb782TDLFE3KuHFWqSU2K2UmlY12P21dpRvyUNz8ss_AA3rl5jFpcnC2IyJNDIZbqdJPd2z0SNlwNyBq7Vl6poenR-j2X3xzIGlCDQ9zRgs50wdWtZ3ZRWLVWMrVkhkddoVKuh1W9rlwsvxmlZbOeRk_Uh0BymAa0-4-n0jI4_-O8jqpL-YzL1Y191brY4ywLUrQXpln41UK76pxc34FojI1Nymw523SNYxAHSlpj01gNmcjPrBTFxQ9SDY7AlrSFwJia_KvWnsZ53qt6fiDHV7p62KzlG_rpz_dQSQoj-z1hZBoUxi4nqzeCIzPcB_3JqeqD6x1O-Vh3uk-6NxN_qCE8cRsizB5vV-Ur-4tqau6LIrdfIB3Db12vpgiCmD_BD4xCxOijDn-97edRZw__xYfhx9_MBEB6gYl1ZBtLJfxDN54N5UION2tiZ2U8THD_h4d8-c26H7NQv44kYppbaseMckhpVOBDh52P5gxWFwp4VGqAIkZ7KU10qAD6M3GTFx7vGth8cT8YS1s2gPDW-WcVQGlAF94gT-FE6vzAjxwRJ4m7B1rJZfYReDvMrAoLroayOVmfB8pOKVQLQEF5dUmlzAIIpeh1NAiTg4n3FXW7OXQhzhU8bmo0e2FuSEOUVimGw9Nk_Wor3kQFp-9kj_iazSC4p5VURnyY_lAirPfyw4nskpZzCjSg_EAU8Au5vvOqrdDEPjrbeT8ks0wi2rsB0AxQxhgf6jUWzp0apeZOIl9dJFH_OnyJfvwrV4YHpee3174WKYhOJIOy2-8FJbMw1MpQtVV49yWmZsIyjRNj2uLbqY7jWBo2UEeOVW5n1tdk5zAVF-RFPKyh9150MnJz_RQtgoNdUD4iLBwlHYHVGLyH4a3GJmOJP6ZC-A-8RiUjvhu5co0yC8M83aVFjLe-yob3sNgJQgdVJnEOfPz4-1DVORoDgIRrRBcZQZqvkZwADFUkyy9jy5oXdEJ5XzthnizbrOZkHk6sQsNXrP4Uadqo9w99uy7TUh62l5AMWBFcaaQhuAuFkUZCavIqoO-2k4oXIDoTeBYzbyo_HH6caMk0D0_zgEg_5i-NhT3EUPdoCBNmjbOKmN2wzf6kqEyc8-nunjfq6HOjC6B6SE6VgOVJgBrhB4cBto4CxO45eqeuCi_WCjRtSS43Bh0QFZi6xK8rRjItyQRIfBpomETElbng3mAmBLPNb_7CzfsBdhBhJQLKu9KZ__uL3YVGtrCaLcOsfwP7BXRNQJH0yN_JWfMZH3y3B8z1O__xGhR63ugExWJZyUn55KAEiODbX35_PcftWXjslq-wzsK4J2fO_HFNU8Pi4egk6ibvCUDFRUelukaAy_YHdb0VTSB6XCymTo96jK0HGjG8FaVwvQaesaUE-e0_JpdMXN3KstKFeTlDUx1o3Ny93-VxLB5rkOSd6cRjEnFRA7Q6HnturEjwPAeJjR2Ll5dsisVrdjqHMbSfSObkpd2dZ0T3LP4-_ug7qRJF60DJTjTPpx7YxeARzuwiu02TlVW0J0PrdXT8EpISHneKc1VWhtRcdD0R0spuAMzJLwELaOemihL1TJSIMBqFikbpulZCZ1k1kA_5D7I5c7pOF1g4uYBW-gJNTenfC9wYmDJAOCcnwk1W4=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'container_id': 'cntr_68cdc387531c81938b4bee78c36acb820dbd09bdba403548',
                            'code': """\
import numpy as np\r
import matplotlib.pyplot as plt\r
import matplotlib.patheffects as pe\r
\r
# Data\r
x_smooth = np.linspace(-5, 5, 501)\r
y_smooth = x_smooth**2\r
x_int = np.arange(-5, 6, 1)\r
y_int = x_int**2\r
\r
# Futuristic styling parameters\r
bg_color = '#0b0f14'          # deep space blue-black\r
grid_color = '#00bcd4'        # cyan\r
neon_cyan = '#00e5ff'\r
neon_magenta = '#ff2bd6'\r
accent = '#8a2be2'            # electric purple\r
\r
plt.style.use('dark_background')\r
plt.rcParams.update({\r
    'font.family': 'DejaVu Sans Mono',\r
    'axes.edgecolor': neon_cyan,\r
    'xtick.color': '#a7ffff',\r
    'ytick.color': '#a7ffff',\r
    'axes.labelcolor': '#a7ffff'\r
})\r
\r
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)\r
fig.patch.set_facecolor(bg_color)\r
ax.set_facecolor(bg_color)\r
\r
# Neon glow effect: draw the curve multiple times with increasing linewidth and decreasing alpha\r
for lw, alpha in [(12, 0.06), (9, 0.09), (6, 0.14), (4, 0.22)]:\r
    ax.plot(x_smooth, y_smooth, color=neon_cyan, linewidth=lw, alpha=alpha, solid_capstyle='round')\r
\r
# Main crisp curve\r
ax.plot(x_smooth, y_smooth, color=neon_cyan, linewidth=2.5)\r
\r
# Glowing integer markers\r
ax.scatter(x_int, y_int, s=220, color=neon_magenta, alpha=0.10, zorder=3)\r
ax.scatter(x_int, y_int, s=60, color=neon_magenta, edgecolor='white', linewidth=0.6, zorder=4)\r
\r
# Grid and spines\r
ax.grid(True, which='major', linestyle=':', linewidth=0.8, color=grid_color, alpha=0.25)\r
for spine in ax.spines.values():\r
    spine.set_linewidth(1.2)\r
\r
# Labels and title with subtle glow\r
title_text = ax.set_title('y = x^2  •  x ∈ [-5, 5]', fontsize=16, color=neon_cyan, pad=12)\r
title_text.set_path_effects([pe.withStroke(linewidth=3, foreground=accent, alpha=0.35)])\r
\r
ax.set_xlabel('x', fontsize=12)\r
ax.set_ylabel('y', fontsize=12)\r
\r
# Ticks\r
ax.set_xticks(x_int)\r
ax.set_yticks(range(0, 26, 5))\r
\r
# Subtle techy footer\r
footer = ax.text(0.98, -0.15, 'generated • neon-grid',\r
                 transform=ax.transAxes, ha='right', va='top',\r
                 color='#7fdfff', fontsize=9, alpha=0.6)\r
footer.set_path_effects([pe.withStroke(linewidth=2, foreground=bg_color, alpha=0.9)])\r
\r
plt.tight_layout()\r
\r
# Save and show\r
out_path = '/mnt/data/y_equals_x_squared_futuristic.png'\r
plt.savefig(out_path, facecolor=fig.get_facecolor(), dpi=200, bbox_inches='tight')\r
plt.show()\r
\r
out_path\
""",
                        },
                        tool_call_id='ci_68cdc3be6f3481908f64d8f0a71dc6bb0187028ba77f15f7',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ci_68cdc3be6f3481908f64d8f0a71dc6bb0187028ba77f15f7',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'status': 'completed',
                            'logs': [
                                """\
/tmp/ipykernel_11/962152713.py:40: UserWarning: You passed a edgecolor/edgecolors ('white') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
  ax.scatter(x_int, y_int, s=60, color=neon_magenta, edgecolor='white', linewidth=0.6, zorder=4)
""",
                                "'/mnt/data/y_equals_x_squared_futuristic.png'",
                            ],
                        },
                        tool_call_id='ci_68cdc3be6f3481908f64d8f0a71dc6bb0187028ba77f15f7',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
I gave the chart a neon, futuristic look with a dark theme, glowing curve, and cyber-style markers and grid.

Download the image: [y_equals_x_squared_futuristic.png](sandbox:/mnt/data/y_equals_x_squared_futuristic.png)

If you want different colors or a holographic gradient background, tell me your preferred palette.\
""",
                        id='msg_68cdc3d0303c8190b2a86413acbedbe60187028ba77f15f7',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=4614, cache_read_tokens=1792, output_tokens=1844, details={'reasoning_tokens': 1024}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_68cdc39da72481909e0512fef9d646240187028ba77f15f7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_code_execution_return_image_stream(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
    )

    agent = Agent(model=model, builtin_tools=[CodeExecutionTool()], output_type=BinaryImage)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Create a chart of y=x^2 for x=-5 to 5') as agent_run:
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
                        content='Create a chart of y=x^2 for x=-5 to 5',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_06c1a26fd89d07f20068dd936ae09c8197b90141e9bf8c36b1',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args="{\"container_id\":\"cntr_68dd936a4cfc81908bdd4f2a2f542b5c0a0e691ad2bfd833\",\"code\":\"import numpy as np\\r\\nimport matplotlib.pyplot as plt\\r\\n\\r\\n# Data\\r\\nx = np.linspace(-5, 5, 1001)\\r\\ny = x**2\\r\\n\\r\\n# Plot\\r\\nfig, ax = plt.subplots(figsize=(6, 4))\\r\\nax.plot(x, y, label='y = x^2', color='#1f77b4')\\r\\nxi = np.arange(-5, 6)\\r\\nyi = xi**2\\r\\nax.scatter(xi, yi, color='#d62728', s=30, zorder=3, label='integer points')\\r\\n\\r\\nax.set_xlabel('x')\\r\\nax.set_ylabel('y')\\r\\nax.set_title('Parabola y = x^2 for x in [-5, 5]')\\r\\nax.grid(True, alpha=0.3)\\r\\nax.set_xlim(-5, 5)\\r\\nax.set_ylim(0, 26)\\r\\nax.legend()\\r\\n\\r\\nplt.tight_layout()\\r\\n\\r\\n# Save image\\r\\nout_path = '/mnt/data/y_eq_x_squared_plot.png'\\r\\nfig.savefig(out_path, dpi=200)\\r\\n\\r\\nout_path\"}",
                        tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed', 'logs': ["'/mnt/data/y_eq_x_squared_plot.png'"]},
                        tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_06c1a26fd89d07f20068dd937ecbd48197bd91dc501bd4a4d4',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=2772, output_tokens=1166, details={'reasoning_tokens': 896}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 20, 47, 35, tzinfo=timezone.utc),
                },
                provider_response_id='resp_06c1a26fd89d07f20068dd9367869c819788cb28e6f19eff9b',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_06c1a26fd89d07f20068dd936ae09c8197b90141e9bf8c36b1',
                    signature=IsStr(),
                    provider_name='openai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_06c1a26fd89d07f20068dd936ae09c8197b90141e9bf8c36b1',
                    signature='gAAAAABo3ZN28TIB2hESP9n7FpWJJ4vj1KEPIVHYTNh64J3S9rOSRfmmTK_uSNB79wwlv3ur6X9Yl9sPe6moHK4nud8jgeScuOeCDq70JGXZ6xH_NBdiDWzeMis1WIDsyJrADdADGQRhjb8sXi6lz3nNvjeqXD-oZJkxTJ9FeJsCNNPBHX-ZYRIYZ7vGKLPfmi5qNS7V6VVGvwEWOBwW75ptObu5E8g2TqhPlUzsVoZsIZiczRXq6zQpDtMPAtv6Mz8puaq-o65P5-vZMywmEjyi0Dd2M9ozUfhWfhpEhCsAiItesA802-TSBQCKeP62riRAMJvfD3PEGLYL9d_7mUvJYSsiOADU0K6wfI6y8bRL-UaWUvn60KfPvqfBFm9-hwP1NS77OKoZABIuGz5sc3BuAh6ebKrJkfNHq7W0BA09S2gt3wLPzflpVl-wJ74L9UGnaKpmG3XRFogff_SNgDhO0_Cb4-1PYJi2NpqnCwTG2c8EFxXiP4trdynbpgRD5hKDj65FU46cBjR0g00bCShqwsseAzw_lAxbawcjF0zmAyz68Km2jCRKHRGgeMpbT-YQWs04IizKYsWfF-8pXX2vwSqk3Kb51OysuPN0K3gihF9v2tPnK2qFzkvNics__CDabCmafEKQlLp6TDRc5RY4ZcSHNwUM_dybJStzoH6qed1GQNt05wBhDZg39N7pJ8_dG7wXCSGHY5CRORZm19UGTd9DoZMzr8JmtxmRgJoKCHW_gavpt4__zifPVxqLUWj6GBaQRT8pR_Tym27HcsC0GbHLR1nel9hC6RzydTU5y7LWY_NoGUE4WZX5rHe5t73lFNSMwd9-6i9Qlj60_rBZ5z9oTAl_Ksywgo68AG7dFdSeI3VLnOyzhqeePn0ywaMp3HqO-FIXW3fjqtM2XMMMMn2Cje5rZhJ9JNmMqnxpltITkVdHMo7Yr1WFTkwLByEOb3M4LCq5B3dM1s1pVmqWAc9YNjpB7Fbi6fG90EAYFNEM4ubOE7y2d5E4hco0MbEKg-Fh0ubh1I2Y1kthZFEmPQLm6fFaljJKPtYojEZZ2cZ7sN3UaVg8Zpf3A7WS9kM2--lL5LuBnVDebf8Xrzv9dTmJvOtwWzJsY4RxWdnzfl_ZokHmg_HDNbeZpHsVI0gqHGr7YTlFJ0NUXW9mzZMx9e_VTrrf34XwRue3xVCqzsspRMjMIlAoDp0Rp0L2tJWAbKs_btqVpqjz8p-64CzSRq65BmSP6i86G0cJ9WLSD3gL3wR-Zt2HyvUvecHVmgKhXgY3F-RchYRO7TarJgyZY5bP2EEpHUwSWx4uWjYfzXMGYn8gNwgwl89qog-inK88qSG0DbqJQPwYNuRjS7Mu01O6eV39Zu7Njsn2io-kPc5HLRrbbhN7qCSki8yPWE_7yPtbIKlwWKOlEYx8_SGgE7waBFRem7ElsE9wvCX5KknilmN5_d9L4Sos0oT5NHAhApvVVDcygz9VGYBAmWfMOynDnOiTIpsAdjHmuZG7GJNAtUEYx7U7pNqbD2FJMIeN0L-3uqhxisRzeX64JZkVHWYL8HjeC1zHiUMZXKW1KXIvIU2_BCtqay22FtBskeMXZAReKhv3eX2oQlWL2Ps9VOk2imzjqBbFLzJgDq0iFoaHdOXGqo54GYZIxfWi10uo65s-3gOGmqPPE02FHEMjK7VHFjMh91FPhh8TmpWjOfa9QEcpEHSZJ6ipUMTVfRHHHshB6Sb74x-Jfr6Ioq2RnWd3E32GpE3kd1poqOssBi5jCqsA86tIMt0m8p_CDu_ANvMNKTiGTQdejm2rUhccpdbp8uLBPnqWxyGOCTlREglHPeh2EzjEMbtIaFp2NhHE6UlJ_nw40CDa5PA7C4lgUkn-4KtPy6rSaMu0mWM4vPO-5ksdtB3E5PkCdIB8j7htbhZH_MTv9RL7loDNkRVlJRSBiAC_qCGgVPyP4l1w4imdey-_HuVCKBD2vaXUz2l2efn-jLSlhty5vBOR-kr0EsU02_NYZtOKgBR1zIslAlnhM8lTxJWH4osSXHa4fIx9O9tyALjvxhooYww_Die_8iCH4u5cF53z3mvoK3Knzeada3jglwQyL3_uUQegcFKpvZwVAcguVMvrsbNgdR9VeKmYq8U7yBvziP-_vpj1UZcf3QxlNK_oOgDg9lxP3vsSKzxliW422svFDiyPkWPh1DWmry1xBD4Pldemf8OEvgSHSDAlegWoBnfOHljDcPf6kT0PaC-jHrKn8t1cQgWk1-1oxiW4zKIlKGoRvmo4lCcUfqGXb5EPuZM1qRFWxv4roAVoxdLV0Pz53L_Q-grQWvbKH_Rl6Dw1BysU55Klt8vn_XBL5Zw_UlbT9FrszDRjJ56F7zElzqVYunI5uJaPWTwQyO-4dvM94CqiUU59iFkfZqaSulYktZrgZeXe0lw59ecQnL_pR2xwkialTgDoqtPksIjTuWVzkiW9hIL5t9sHyCdJ9nqmwZRZU-JuTPXswmrJEJ23GhvtH9kWsswLd0qvmY5mV3cwr7hlFNWEf8_5e3LoCa9uHQgIa0uquekJ3St9dLOXpkcRv74nCpxkcjems_2ZC71DRU63NILFjKC5ffsUPOZ4NfevDMUDbYHdeyVV6E2f-_1yMYCWI_sws69fWQkWUIv33hk7Gm55NaNgLD4RYCUBTO7v1FtEZiVYAU5ab7NvvnTJ3FaEHo9G9eTzN1I_MmPzqlYX539YF_DDedh0ThnSoJl7PYD-7LhRRG1215KmsTWbqDGmtTsHePAVRSh464XHgiZ6cNPNogtMl4ym6r6nsMbzFP2krBR1f-u0tHfQFxAeLyBWij01Z1WBz4GBh3bpdLrB85AlvFeY7R46PPydAHxwwanYVyxpS0UmS7Y2S37EVRdFzai1izvoy3-wA05YKcnRiUKR-oMcLf-BmB3HHZnY77YOuqQBUZNI7OR8B6lvTARQuoJbK26ONmXEsH-VoBJR7C-hNiXMVh1jHfhuaBAj6Dg9g1Vs2kGxfoJUXB5dlFmR42mnyGcT96N8ZAIdIoQSrBzai6bQbuvOb3OAcG2lEhOZHZiwFRCzpHMfu5dctZ_wcTUhYZwgOcBNIo4WELyjv0Yx22AHSHcrUzFezOwibs-heUF_ciKWkGv9OaabaAGTaTVncfCnS7rOcD3Xum89EAVegpYiQzK0DZ_VKooPoddgHs6diYOEn4iJyvE54vaVi72NAy0Tf9poRlidKaM009FImefEtZqwD1MmaeVbjcClv5Xwyh-KCQ2hCZmrnJ2P_e0bWIsE0MAJOK8iU6Q3zxbntbZAQAKZHqqauT8kkRYxk6oBicV5BS-whqDN_GoNZrnRLTNkjk1a8mnqg_kucvC1mCQRbvP367DYqZGuAd2EQWVLSBQibHoVIUcYAFbsfRHfsQ-uiZVZsjZ-xGM-ZcTzCJ6p-hFi9IQXKqOioM_xzRl4TSY-AEbGja_RY0puxi8BeZXvSxx8eYsJ0TRtIIQwloZzKpbx1OwyK-Ibfj01PU5NIurJL10PKXcnc7ImXN-b_p8wfzEVN12lSbQ8m-Rs0tx32jfvviXyHtWYfHuNqP0eL3Xjuka6FGnuDOeOAIzy4xj1vqhXd8UN2tiFOObl4Rza5pKzF-0IcEsKX36v4iN8oYxOoCxCxLwvFw3znYiAKe6CVky4e46LxZOI3bGM6MSrypwblPMA2gC_ogfMiYViJe8gsgld9UvgQaFfj0EEgfc0BWfxVw2i6Yv3OcH3T1jaHnCVgvcDpTXI4-ZeeWKl6fhH9ukYAG4-Y2mGiJhxJ7cjSg8CwU0KDmNRwoXGB2FT0bKWovkcFYM5ueMbXFTZ4FFcgfWcOzXFZka82HFB_iqD1XvOYMFQNiz3jdtuOr8o66rtCVAjJnuoTQDmbSrWPU0-utUMJx-4QAlZM8hdtXGfNBp0JRxctMZdxR4BAzF7JH_ETYi3itZkgDLEs9JBdty6gUiM0NdR6F_7mxsHCik3rpb5bauJKP89gV03mnBQuSUQTauNxdzXqw55SPDAHMBWg8QwyffzWwmyTAjl_R1QiFsTOv31U-HditYAeYMhLAP0mIs97T0inLsTUri1s2b1s7j6-I-NLXuT4VKiBO8lqVicTbQdQwiXehHQsi18e0H6T9XM0xBQK2t1dd4Jz2oLUGroSB3XuNbcaaxsffqRQgk43KIMEw9VsUA3FOTEpdM_xYIYEFM_-ApjDQJ15JyMRspfmu7HDdd-ybcXZ-C8WASJUPV8tFEfP4xgUcZeu-mExkryebbdMExq78yj7GlwWaeqBYfEXsvG6FIOqL9iFVcc3iIelrly0oM_xJmLOB_CCkGylDmHLxZZydf5v0RDh0KOXd7J-QYepcALXYoXmToj2JPrJPkaznH-2tI5xwp_M-mktoYNOhWrOepFjceXDSF5G5ILomGd9mHLnkq514ayZJCeE437I2geH4s6upgSAaqc07IVvdU3WjorhBw9fvefI5NnYwMiUSk_LC-JiQZDJ0bMLttvwKDx0TmOnMDJqxDr06_MWXn3i0zLQlAjItS2foksr6EMeK2InZznVZtgjcbD0exqZuzjCAqKz4PLQl62xyuJx8trJe0uHbQk-NweJthN5xcj41kJTcDuXbA1bA9HerCBWMX0RW3RXAKTvltGaqyMyUsJ_uOb40D0m56SqOmxnyA-mauiV2R11KC5Hh7YSS587NxkWUx2t7G9uio6WgWyx-HvhXYVi8wejyZw51z70YEa-aUDS2G_N0e6BV2B6dMGyd3lzTkMY6Ncs127IwQmXkV4VGL0stfchFf7rhXc1CZmFm7NZOMQPgb3_Heb39gZfMa4EYUVLuvfSpuM8wHZcQa57_uj6wmGp7NBBVpcgTee9ADvJXxjlmAj6gm9TiCl_GYbBLCdoTRAgsgsy1r4WijYr2sA_zch6EbDpTjQy6ER5GINZ4zi0VDy9avZcxhGmOEHYvKzcLB5PANOAW-8FLFHGgDWvf0cEMCD0UpSLAJVIX6rMjMJC3N_cgWmmv_zbllaW-vDVNFPyZOW32zU-l7r46_5IuF9Vc5choUlWOGLADSnXReau9WC4rfGF05CAvLe5Q0dex4K14SHJTEJuBWhGTaaXzONQSGtU9LJexoI1ijcnz9X59VvXxFX0oHmLvgTAim6nN96X5kllHFvrdDjMOiZKQTXtodUI-3ZcjfA5booJk7tnFeni0H2L1sqvpGy8JDlfl0fds8hST0vtXscfD5jDC-i6btLnRgpOpRDQMebCkqRlisZScBXb0nxoHK7CHtnQy4aCQq4oCBgMXdbwHOnbBygBSAg-HCpK53YoT-R5NUdESGmGCX5uJ0qlmGaXSshFbNW_NpQItJIrD7NW3VmqfWvSB1VL-nyVLOmc_wPmUhY7dSGArYKYQFKL4cBOSfHHHuftrRXy356_mTcDeFsHzqH3RXPaXhiad_lmQ9Bcw0OD_BotHvYfvVCaETpweH3eHl3RPBiUHlc5Da4nprHbXrvQL675qwVLiwLwOvPULU4VdGU-jIfSMkRUbJhSt349C1poj4aM-aD3s5iJy-3YDRYzmqMmFFr9CoKMah6hmn6n0oKSwg0YpLOc9JRDhBfp87_NNsWdRkpNw_DC7OaIF6VNxc6o2t9jExqmAiAbyRSkW2x-UiZl6kbB3uqffgAYWNylgJDZ-UPQNki30zURQFl1anKa8xhIGOgH7piVerG2LO8X7pFxa3DlYxFm37HC6irFtBwsFbvNGicua6MfUD3dV2MhE9x-sOlG9O08DKObUwBTpTzfAe-P_jGWHnyOsLXbaiV_cwxgWkEw9rKuFpI1SPuPrdO8_iSYdH36TqIREPLVbRcSJvHrsWP2Bf-Bb04SIonHV4Olu9KEYWVCOltRx7JFjp3eVQZLAGwjtxG_vDlublMpybM6TZdg1UYaCU4ZqLKss3iWO3wBNwC2usITNSjaiiLSH96fOHpAyXMhhodFDS9X-frLB46hilqE3PwoIyiR5R1dAdM7oiWa5qD6KH_dISw5H-uO6ZrUFo6i14E4RcCtRBBKALvVnApLxA_lcpnFR9_TZkstK-6klIEiSttNhxhHhv36XJw_J6jUTHnxRBr4JyXLL3-NmDZy8mplsbS4OXl7gg0vuIOBBHarKFvCEdvZv8ikxbDeftTz2je9mrCNCAHKTeNQWKf7Q7HFfPcza_BwhSqrd64DndvGVkfLlYBrbVSZp5nxPF13qBWIw9bbXTU5z8Wna72Lh4HqL-cUDsKbKBpst1VuBgaA7Va',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68dd936a4cfc81908bdd4f2a2f542b5c0a0e691ad2bfd833","code":"',
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='import', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' numpy', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' as', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' np', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='import', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' matplotlib', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.pyplot', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' as', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' plt', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='#', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' Data', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' np', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.linspace', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(-', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='100', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='1', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='**', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='2', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='#', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' Plot', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='fig', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' plt', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.subplots', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(figsize', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=(', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='6', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='4', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='))\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.plot', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' label', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="='", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='^', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='2', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="',", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' color', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="='#", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='1', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='f', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='77', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='b', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='4', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='xi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' np', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.arange', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(-', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='6', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='yi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' xi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='**', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='2', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.scatter', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='i', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' yi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' color', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="='#", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='d', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='627', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='28', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="',", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' s', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='30', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' z', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='order', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='3', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' label', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="='", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='integer', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' points', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_xlabel', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="('", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_ylabel', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="('", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_title', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="('", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='Par', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ab', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ola', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='^', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='2', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' for', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' in', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' [-', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=']', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.grid', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(True', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' alpha', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='0', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='3', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_xlim', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(-', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_ylim', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='0', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='26', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.legend', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='()\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='plt', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.tight', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_layout', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='()\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='#', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' Save', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' image', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='out', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_path', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=" '/", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='mnt', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='/data', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='/y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_eq', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_squared', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_plot', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.png', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="'\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='fig', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.savefig', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(out', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_path', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' dpi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='200', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='out', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_path', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args="{\"container_id\":\"cntr_68dd936a4cfc81908bdd4f2a2f542b5c0a0e691ad2bfd833\",\"code\":\"import numpy as np\\r\\nimport matplotlib.pyplot as plt\\r\\n\\r\\n# Data\\r\\nx = np.linspace(-5, 5, 1001)\\r\\ny = x**2\\r\\n\\r\\n# Plot\\r\\nfig, ax = plt.subplots(figsize=(6, 4))\\r\\nax.plot(x, y, label='y = x^2', color='#1f77b4')\\r\\nxi = np.arange(-5, 6)\\r\\nyi = xi**2\\r\\nax.scatter(xi, yi, color='#d62728', s=30, zorder=3, label='integer points')\\r\\n\\r\\nax.set_xlabel('x')\\r\\nax.set_ylabel('y')\\r\\nax.set_title('Parabola y = x^2 for x in [-5, 5]')\\r\\nax.grid(True, alpha=0.3)\\r\\nax.set_xlim(-5, 5)\\r\\nax.set_ylim(0, 26)\\r\\nax.legend()\\r\\n\\r\\nplt.tight_layout()\\r\\n\\r\\n# Save image\\r\\nout_path = '/mnt/data/y_eq_x_squared_plot.png'\\r\\nfig.savefig(out_path, dpi=200)\\r\\n\\r\\nout_path\"}",
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    provider_name='openai',
                ),
                next_part_kind='file',
            ),
            PartStartEvent(
                index=2,
                part=FilePart(
                    content=IsInstance(BinaryImage),
                    id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartStartEvent(
                index=3,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed', 'logs': ["'/mnt/data/y_eq_x_squared_plot.png'"]},
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='file',
            ),
            PartStartEvent(
                index=4,
                part=TextPart(
                    content='Here', id='msg_06c1a26fd89d07f20068dd937ecbd48197bd91dc501bd4a4d4', provider_name='openai'
                ),
                previous_part_kind='builtin-tool-return',
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' chart')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' =')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' x')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='^')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' x')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' from')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='Download')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' image')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' [')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='Download')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' chart')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='](')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='sandbox')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=':/')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='mnt')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='/data')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='/y')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='_eq')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='_x')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='_squared')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='_plot')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='.png')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=')')),
            PartEndEvent(
                index=4,
                part=TextPart(
                    content=IsStr(), id='msg_06c1a26fd89d07f20068dd937ecbd48197bd91dc501bd4a4d4', provider_name='openai'
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args="{\"container_id\":\"cntr_68dd936a4cfc81908bdd4f2a2f542b5c0a0e691ad2bfd833\",\"code\":\"import numpy as np\\r\\nimport matplotlib.pyplot as plt\\r\\n\\r\\n# Data\\r\\nx = np.linspace(-5, 5, 1001)\\r\\ny = x**2\\r\\n\\r\\n# Plot\\r\\nfig, ax = plt.subplots(figsize=(6, 4))\\r\\nax.plot(x, y, label='y = x^2', color='#1f77b4')\\r\\nxi = np.arange(-5, 6)\\r\\nyi = xi**2\\r\\nax.scatter(xi, yi, color='#d62728', s=30, zorder=3, label='integer points')\\r\\n\\r\\nax.set_xlabel('x')\\r\\nax.set_ylabel('y')\\r\\nax.set_title('Parabola y = x^2 for x in [-5, 5]')\\r\\nax.grid(True, alpha=0.3)\\r\\nax.set_xlim(-5, 5)\\r\\nax.set_ylim(0, 26)\\r\\nax.legend()\\r\\n\\r\\nplt.tight_layout()\\r\\n\\r\\n# Save image\\r\\nout_path = '/mnt/data/y_eq_x_squared_plot.png'\\r\\nfig.savefig(out_path, dpi=200)\\r\\n\\r\\nout_path\"}",
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed', 'logs': ["'/mnt/data/y_eq_x_squared_plot.png'"]},
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )


async def test_openai_responses_image_generation(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=BinaryImage)

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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_68cdc3d72da88191a5af3bc08ac54aad08537600f5445fc6',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_68cdc3ed36dc8191b543d16151961f8e08537600f5445fc6',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_68cdc3ed36dc8191b543d16151961f8e08537600f5445fc6',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_68cdc3ed36dc8191b543d16151961f8e08537600f5445fc6',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='', id='msg_68cdc42eae2c81918eeacdbceb60d7fa08537600f5445fc6', provider_name='openai'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2746,
                    cache_read_tokens=1664,
                    output_tokens=1106,
                    details={'reasoning_tokens': 960},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 20, 57, 58, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_68cdc4311c948191a7fb4cb3e04f12f508537600f5445fc6',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_68cdc46a3bc881919771488b1795a68908537600f5445fc6',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_68cdc46a3bc881919771488b1795a68908537600f5445fc6',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_68cdc46a3bc881919771488b1795a68908537600f5445fc6',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='', id='msg_68cdc4c5951c8191ace8044f1e89571508537600f5445fc6', provider_name='openai'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2804,
                    cache_read_tokens=1280,
                    output_tokens=792,
                    details={'reasoning_tokens': 576},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_stream(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, output_type=BinaryImage)

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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_00d13c4dbac420df0068dd91a321d8819faab4a11031f79355',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1536',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1588,
                    output_tokens=1114,
                    details={'reasoning_tokens': 960},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_00d13c4dbac420df0068dd91a321d8819faab4a11031f79355',
                    signature=IsStr(),
                    provider_name='openai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_00d13c4dbac420df0068dd91a321d8819faab4a11031f79355',
                    signature='gAAAAABo3ZGveBi351h31WQM2aG_dbN1N74J4X3Lf1SbUrUhElKaT5odbh4N1liwG5Hjip3Ci1illQSsd4n035fOOIV3sZzAMvV3ypncux4WDBpQ9NbeuFMNSyNOPTxJLg4j66UbW2ptw3u1VP3j0vCHvV5MoDhErheYZsWKhYVtkUNkKSVLWkS_yK0pOltSwHfRy3tbrkxnqD99BuVbCjV1nWSzTAmJLicBtjDaH0NjjD_vMyFiUe83-eZRs-Q_6njWasZNCmTcOq4zlpFoJ_AGeaTbaLIC1OwDV3sNT7pXvo7YI7jmsYEhHAKa8BjZmMjzBPLDRu9TMWtXMnO6nyVYqMxsyQPdNmP-BDNfr_8Rmo_uI5egfE0qRKgAc5MrOGd1fSgtUqeKah3kbLMyCD0_-jWmVInb2Y4LfPcX0iOeTGum2IRKwy6G1tdY8C_gEOnIGAUKOT2sEF98Ythy9auV27BCbcjfCBJlH0rOir_OiQjUIXZqqY0My1kVENBbXj2-VIFIqG-CcxCldFG2Mq0NGo86h1igQIFmLItXLPTS_QnaWADSD9La8JWpg8CuWg-yB3UqYaG5_f2Cl5jRDQdIYavTBvD-lp54y8aEnGA6HksQaCtB7jHX0ZM0pqYu7LvLjeHxAJWsnF4NN0HPz3d307muS0TtrXUxqeZFdTNoqdBOxfuJ2-Ym_LmeubnEzh5wHAguJKZ6S_jcEFM3Jdb1R8Gk9dv2y7BUz1hKSFF7peXc9Ear00JjPHAlR1x0ECqONTSD9Kda80pQlDSh05ITKQ2viOy1jmCsWeSsll6EJPcGEMfAcZ1UMgHYX3sBa5Oz3DS28Ur8yk-I62nUWbcj8n7IsZmZL0CWc2qgCtj2TzFZaVEx8iumUKpU0hmML_kF3JPH2Ie8nB1ko18HZV7A_-n9XGDZzwsfPD9pu4P-fb68KNqU_qQBfe8msYvuFuljC-0kyGrQIQH2X6stwEkyme1TuJfxIZ2t2q2l02gEUVN8LN8qX98hp7DBxXepgdKvqWVOM7icvtW0mPACf1b4izSDqEgqhqx4tNsjixoHcM9M8awzss_y2_jZ3V7gY3pbPgwWKHyyTUzA1ogPfMkjxxUrVLNyHRPmnklUeQdV-vytip3BzNOq4yTUz7jVFrudSDcr_KM6Ie806OkgKF81l-W-40qzx6bGg2DAcZf5hfbTzk-ho51sRBwDp7RJrx2SXSBGXA3ArYzgq-2iat368uDLiQhhbunzKm3_6CFWggpbUO8Kp3FP7-k4Z4CRbHkg8WVT0HhH6w0ysoi-P6_ZH-IKI7XG-GT1kq4yje3qlfRUT0-0_LPsr8LyM6AbOYj4NiWHP3XJ2qa978VVOLJQtY-qG3VX9kMq13C-uU8PDOsOEidYZl2gqFtXhxkXivwACbLMnvzJayXJRev1QkoNxIg1Stl9II4D_ndHfNYeAvMvOnSNafoCOmMzCBp1klovMP_31YvR2B3af1TYanbbHoJt2UR1GRR_Aqr7G6RukNkXAl63LPlDQSYm5BB6zD9iNX9hJ8MSZ1IFIcbM0L32tAWsyKKAEWyr9MGckicDa_hES9adeXuunqqKhUctd94J1dsXLiWCGIet57YIUj5WoF_FQ6D6FY9rB00KhCDlHr1Ot5NCMmn6y-u6TYJUhpl7elEErYGXaGPhUtKUSbAIzOXzBIAKb_MiMVvo6a2VYwsxwZV14X8TYkKw_Y7w5Wt6JA_wTOoen7Cc0eFyc7FZA4NjIMkIUOXymtjzOSkFJz1eMBqp9diET9VYKGsn6GxviD8jWM6-RCWcFurewcn4d6TeTclAt7G_LZrJ9bZtMVlieSJT-3vWr9qVt8OGBUJEJRVOzpr5FBnEceqK8s7D_s8EZwTaGwyAuuaZThy2PNWJhpE4c0UeKgh0ec36Q0ZRN8DF8Khne8Epe1rehOrsfeyFFRuQ5CDGdHimhtOAIbDyg_5PPCp8fgiU4R9xqtizCVTR4ej1VPIClmebUErOl3TN-IyoSc8rv--Vi0ATn69Q8tSPweI07KVEzRJpDtxbnGcbbilPN5_liJcQrLMf5ikaWBoq42s6FXDjr-ASD0h7IlNGHxnN8q__iO6jA9-2PTywI2bbBJsie2L7OaGGehO5zv_rWv_6rbk4HLVcQafi2nC5w1GNeDaXWSz0RjiTfXxjBh98302CQxiM-e1Lvt1Pe6Mqv-pAgXlFrSHDrqw8s4NSS2YpLDTUIOcOx8UutAJOgVpyZm2sQcvtOsGsSUBIyNI_4huseO9EuXF4TUQ-yzQRsimtXaDa6VId0y6qG7dWxTP30SWZkft2iW2_Nz_56MiioY7xACIjzo4s2aGLM352ufd4nEeU-K3UQd5hvhdIWUZn6KTyCUnqgChyIlB0Sto24VwIIj74DYisSiu-d8EYsVr5gZaQ_NaW4T7M_ZB0TJ0ptlU0X_h8uLu0ro2Vc_s7D8nkIKSzhGuuHO4lOjvZ-qLsPxG-pBa6jGvv1hOyng_x99icZ0oM7G7FmDl7SjP1pdLiZAA1hMPPU9b8Uk0j8hb4AFtfoXSfwZBQ8sYlT0_QmcSBgGxfZKXv4RcFSnAEGDNUn1V-P2uNoj06MOwzroZVjTuzVy284Hqe-08Gtt_bvZDmfsHonbEw5DrthsP9SzoC62hc6pcVs_ApQE5LwHgODxT-oejDppixNCr--hJ1IYVj4rRsHsmBv33H5kJP0rwmkdJ-I8rLj66jLf_Qu_OEh02dJqf3XSYsG7io3XCVjA-d-jUhLJSqcPS_3y5thCtWUcG_ucT64ADWdtOH0EkmzN0o7HmOJ48pkGhttNScjXlQUmOdkeBV55dTdXAzAyjKZsxP5ZK1F9m_1TMWDJX6nT4rRFrzv3PQByEyc3Rje7ZUdGa3Qky1-T5uhu1dk4ty_I92CbMDCM-jGZorhg5MX10B_zZ03DFrYTrdcDILS5i_BSOlGT8Du4aSMvwvUC5FLOYQFQdM_ZNIRIGhOSWsvObmVYh0j70YKqitDudSIm1V_Yw6qsW3ZPpLDgBju176FVDJJBn1Wx-DeQ6FrYtOjFHctqJN-2mjWQi_7lAzKbTLsB-9c4iZ4_efWXsHncmAeqvt0gvglQHDhY6cM4yZurpHkrE-lb5-vDLYamv-Du7Cs0pAaynEcbT-f3_F_WOgoXFy2WYOTt2KkSQZnW6ZPHzl3gfVOsHfAkWalMJ6vXa8FuoYfMmgZJpqtee5J6AxJaUea8xQ0VlVwuXmcK8EOPcwF1pWg8w5_SweA9jZ0fh5PaFW-BNlzGDmhRR-8Up0TCUTsdnZN7bABJBlxeQ5GEcwOjgT0UBF0_zXZo5fbk34TSDoEgfdQydVLlOGda8McmvsnNzDSq77a-Vj_8BeVacM1PPG9rp9F_-PQgpM7_7YsNoWMXha4b4_H58q7vPOvMK1zxRzNrq-sm9QhQ1LkzPgt158Gf2IPq8D3rh9YCmJvg1Ju7roShfnVdV_UO73MLnDhoqaUZEdq10723KFpescGNTRpsuWDE8qBiu58rbOzjmpy7nJfuOtfrv_qSjaFRTkShLV5PW2neHjNLlvQlWy-q3yjJXq-2zM-iRehbFIxI3ATcCq-SgThDeQ1qnTg9G0Jtsx3qBNZtCIi8x1oVsyavVJcqvo36UC-IXaXA1vpjuwER1dcZ999sP9MUnXcMTO9ba-GM3dslKvDtuZ5b8x_u7eCJfawzUPItU8iwISYKDWW8wTNOS8Iukujq3-IDOEFqmCOAlkdv6-AWNc7ZVOmyvvgDCpSN5nSkjpJhWI5kP13FJtJNHkNtP4RQkhRhwRh2ei308TvNgT8YSaa4E_BJ-QWQ_9PMNBsfAYSGIl1VaQinZF0qdvNhRIlonuZMV58aEEzsLk6hS7CGlbFwBMwAzZ5Q6PANavXDFiPGeIadxTE4r-iZLQ3CdvJWUiUv3AL3lzYraXX8BGDpEVAAIqoRYZEpR2QgIUui5b3gkCSlG-YdKqJ4HZ_6VCFqpywKsgPCX_c8pVD_6eJhgt9o5Vc0ARsfc1IG_XC-nFWOV4caiARMobX0y4qXDFulrAZInqBZ9Pq5MmbbhBmLLdT-y5fdPpB5UxsIHGqb3pip4ZaKS80IqAt8t7HPXSNza7zb1TwrjNlYcO_KhbLQBB0hMmKULnEJPWLDPKf_9NeAsN3U9AWyj1WpAKjSfpjjbXn37qpTMdgd-Js9-_FDaXDFH_aOYXI0GY1AMpvSSQzx_f6Erq4qyS5TAuAtXbvUm-iVJcHaZTIy7buGJqOUBb7BC1L33KpeQEZuCg6QyAdzn4bZUKvwjXuxNykpZA9LZWaFVdx2QfwCV_yqN2TTvLFmSj5SjldGwbBndjmtHs5kkDcV2mDlm3huEfbEJqf9sdxXaYhIfmUIkFDtYTpE1C0qSol-A6Yagtx_aNfWTL7F2lFI0OusuBwnDfkNow5mPsKqGMIqx5eJA2InLcpV7GTyCxT3BjVsggtSb1-4Zz2TYzBz7iYe8NPe-rxF6XWyHf1N0nyyCY8Y0_CqJS9OPFpsd53a6qY7xlhh1kwBOM8nJWb3OEJjMVspTUfwF90O8D9fDNS293vnG8SArU6d-1L4u0LalQbKXDRzcze8W8R3KWv1N0LXrWwfArPrO1WnpdEkJnbFfc1eUHqThJ39c7RAInK66VtNe5xtUVzuNZDfPKsIfD4Ms5xqMKEOWQt8RIciRapDo9aoWv5l-YCkuTrWp4pWP4b7eu9fizM5ZuzmRCj3Ecc7ZT2uvxe9sP045dqTH6lSeBNW1eW-pmb3oQ-g_mYL6SU60NmDp_mMa5HFuTdGSAAI9jP11k8KQUX6oGGGhx24w9seLaY98N_0v-cWsiNMQSnwR_SsGs6tPYqltHguz_azu0qsQuuXTQK9B06oEDR8tyb6CTqfX8pcumXIXC_DMFYfQ3pBK5R37G_oXTtX9srpw9vSulg4z52GhuvfT09ukMmdNGoIAS0551PjpZRz7-sI_nNTJQKpGgbhiH_zvA3U5hxue7fpAnQYXd6DYxR_y7QXSleoqQhZ2iVQW90Lwqp5MIDJaAx14bn27WBmQSLcuMpgnwpothMYFMmmNMdWYnGcQ0MIjhlOoykau7DRBFsLOKZ88y_9Pke7k9ISeTmArge2IdC1Ma7-GiJ90YVwwXDBSs9ssae8F1kWgyYV9rFxNbpF4uiWdQkVvASmW-QUNWzsHAtfuvrt-TR1SQ1Z-mMP_zF8mVjC14pAP5Z4pYkolLBinwy9V7DjcN0kymIM6fwpLt2h0LgfC1eLK3sutJcJJP9fFd8tTLIskEvUly-TeEct-syQebPxjxpxae7UPmqeDrOtvPi8-JWiHeIoJrUQnnw3ik2ULXvX1VFSnzDcKBAs_xZzdtjRlCGZWD-hgPPRTmG-YWeyovXZDp5Wv06AEL-hJlk4z1ZEt3yA0H6Ni7zE8jQ0_c6zJCWk6YtPhFk0ARZfjjdYSOFwJvx6rIrteH39b5W1yE8X0bm_cdeA1Q6TluBBkwv-9liCSOGT4ctzwaK3-cb0b4ko_apEEtpYkevu2ulqZoFi1S9g1joFZ5ooBLpxYGntuXXbALvq-zZniOJOtTdbpgsFQPS6Ae9kWXddWChNeyv_CEdkwXCkM__ua4GiH_Ce9WlqCzDCEoCYFpr7PyJP3gNg9Q_vkiLQa9V9bc3VtA5z4cjWB3rU5X9fLDZ0xzwO9krtGmsK9r2gkENMMu5Yy5BxGo94n6wRef0eMY6_GTzi7QsRuQSqNQLa98UdN4QGDa2c_-uDpENkMya7_hgM1z_RyUGtqVgpCHrld-jfSIGLPUI6kKWUDZ3USldXuep47KNuO3-BEOP2QEAKgHVlS9g2viG5r6wdeMl7Njs6iMsjs1KnHaqHlfZww8egAuOxAJjFxUYPy5djKn8n5lgPdk9ISeMZfxW5LcP80kPQekLohUbHcJ_JC2rTI76ckZvwuEGDUQTwGHR0B7YonoiTVzrhOWeqndwk3EBp0cr2mIc8vsWANK1WechMxunFVn7RuwV926PZhqFrnoep4ytDP8h4nJ4Z5zr9cXQCDv624H3JdPUYBBoxJ_7-QDM0fpuFXuRArtuezy6PV0a21CHLFtNq3DCWp56o4xgGm8x_8r2NtKTXxSwUY4_5cBHWd80aXF84Z42ldtGkAXayyFsv5er4VvWTzjwfEc39qkUGDQ5feVJb3YhfsT2qFyUnhb167hdJIPkI8rud4vLu3e9eu6xNLcw-LEjHptgEtfxOqiAPrBZLfWgkhfpU-encYtxg9cing8f_bAkf4-sP1tEaczGkdkMD-0orT-aN46m-8Dyn82fQgQdvov6n7KIuQipYomIQ3mJh5mSl2BAGMFlvLY297s3dCkBD3pGbRb6AAqu-5l8yCCVtg7FvzUWoQ3gL8FcP2cK_fYoJf7Z2YbgTNI_5SHiaAb-qxWuIP8ICEsxCHJJWIOfL6UnBXXctp8B_TiSbOFfGFrQPTJDUvKPyN9_mzO4mzXlOXLXu2VRG9J4NMSYTJT6-Q269vzse4SGqnULEUnpm2zQz9b9W97ahoMFYfV8xaVeFZK5ZU8LpyaN6v4mOJuuHm_vZuVircckh7UVVEK67jvRMi5JcKv-hDQhy1EmSRNCiZ4WHmGi7wcLEcJaUVFBRi84nU50Gjjs2kTslgVAnR9MyGqL2N4xvTAjoi4o-SCvDIvgWDnRCHXSD6ghfQagEUVGldGzk3EKEQF7VO5KTdheZ9FDiSXaaJJKit9NnohzmxM651VFC-AW0Ghklj52C5yvHScmJrIpMv4IjFAKj7erMRDjvYJ0v0PZDE0guTvoUFHrZd6umnB68QFINJogoy5GeT1hUs87OjZVQzPrxZqO6rzJK9m3meI2dFvdbgyAdbUx7fJRAu4yf2LC4dh0QaS5z24wuND3y-jHEsOvjUyIklRGeoH8EdGTBI8ZJIYKXJ8Ow797VYFI3FBzKNiPxJH-VFjpw0aqTLXVrAvCxwVK3awVAoWpwWMHN5yT57TOn3kpAbnBdAXG80kwTuOAAagePIVGrzENRGWVPGhvBFi55TDrQFXyymCP6c5q01KY04VU0udmOSe2Bwd-jMk2pjT3CLHb95G4PUVgy-l-occtk0mNRX4k3P9ETjeyOuA05c2rzMDthoHcFUnMqofePnvVK3eliJjh1uoNOrbx1rJuGsDZFEGxUfkjc5z5BW9zVw5YS7mlXjACPSDgMgreTTygsKTL0xhvSPsmu18K-cGz19v8ho7ix5B1WmPDsL75qXEqKsiO0ry1Ka23z8c4omngareIMqyM6OANeslUhQ7M_4o-OSaHUKQ3kAmJ3c_iPpedZUCo8GALcrgifqgd_ckfBRBpYssZhFQkxPNKJZhncuoRkdjxeAzANinaBUCxZ-Bg5DRQI6GCHgzUiUFMIWEqi21FF5UEiq0G2PM7PTE-RRO7wu8qg==',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='image_generation',
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='image_generation',
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    provider_name='openai',
                ),
                next_part_kind='file',
            ),
            PartStartEvent(
                index=2,
                part=FilePart(
                    content=IsInstance(BinaryImage),
                    id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartStartEvent(
                index=2,
                part=FilePart(
                    content=IsInstance(BinaryImage),
                    id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                ),
                previous_part_kind='file',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolReturnPart(
                    tool_name='image_generation',
                    content={
                        'status': 'completed',
                        'background': 'opaque',
                        'quality': 'high',
                        'size': '1024x1536',
                        'revised_prompt': IsStr(),
                    },
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='file',
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='image_generation',
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='image_generation',
                    content={
                        'status': 'completed',
                        'background': 'opaque',
                        'quality': 'high',
                        'size': '1024x1536',
                        'revised_prompt': IsStr(),
                    },
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )


async def test_openai_responses_image_generation_tool_without_image_output(
    allow_model_requests: None, openai_api_key: str
):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))

    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()])

    with capture_run_messages() as messages:
        with pytest.raises(
            UnexpectedModelBehavior, match=re.escape('Exceeded maximum retries (1) for output validation')
        ):
            await agent.run('Generate an image of an axolotl.')

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_02bf8d298b645b0100697c34d7366881a0bdb502183984ee50',
                        signature='gAAAAABpfDUg__7uoiLN7UA3bQO_LBoJUM79G4uVBJBah0zLoYbC2XRIa6g4PhYFiQszB-k_bNqTpAy75QJt5tiBdExNlfmAlHojKTX1l_hnlAIJgWgsv4_dIavqx5vXS37Mfx9_ASBVbWK0DLbWZhl2_7dciPAmD142YNE5auzXPXGDWoBpVwJmloqaN8oEDUtCud0frxFI1ECmcPqtxD5gVb7H2GeWukjlZoGY4qYaaeIqpCg0ZoQSJtIUoeUreLfCdJAE5EfJDeqdCp6tD3-7TJmnnBaU9xTLU938wO29jVy6_eKH7zvQlZg9atAR8wxoMtQyXr7h5lNSKa8RZJ7Dlm-o7j5wHhHMmhi2JrVNYth7vOyrH3HBz55okJ-RvGWw8wdeMwFKkWM9J1wkT9ii5-R2xkeBmC618W6XL9rBRTi3BIOLl9HGCODF7Z4OW3G6pAxA7xTmPYwy6NFP_t7cSVT0mJYnAUzY3EV6XIc9_IMdkTF9fv8rKjOVKhs6PXoCvk2ctzYvISSOSxkKfBdLyPaXfeh7kcpTXNCuAzH70-S0gksJBQg7xwAG-GGR1QkLT_CRTcRSUfuhhRzof7xmk3nWLIY1osucy0SwefziWu3W44DIDwQS1ts4YmLnf5qmwVEJ-y74V2wSIgXd9ePUmzLKqbISE05HynS3ml370p9XyJGk7Oo_Xo-rlOhrfpzQZHEnrNnqGGWhR71pkGMlM9BYX2z8s6lcNRg-F7HBoSksdkbV4oZQLsTkNNLgr7Wk1jpK4O-onVSPU3_U9i-kO8TSlN1CvzJ7AamflebKecPESLOLlQioeb_VfaqBKSZKSO6RIBc8OAtys30wl3ZrCqwa-DHNt6QFe1mmAhsiHTfheD4LOjQLjj2b3aSnhT6vw_xqe2H_gobmwKCChpQS5c7wcQyerlmfvnnirKu4o_QqdG6ivZ4iH0kQFHoz27zclwvssRsA5mF2RHX1AB29-b7sXG3lbAzq3JbNS_F3zBiududFEkYqRBRgF1zAD6CznBsxfN2TNMOSLe1Vnivp8srso28e35Mxp8DQy_40fYkBuN2a9-rukINRK7qf8ve2bMMRn8qwwEHuFABmB389gwGso4shFyADOl5OZohtgbKr2O32Fafp2aK9Gl7XbVCT8--wyZvEVYjsHUiP31-gDTiwRRU4FJuW5Jvgob_UDGF8zGI_UpOLqgP2CwkU0u2Mol3SydUhxCImaT_vkYznQMwzolTHY4Lhvq6NMOOkJJzw8R51yT9HwlyWeaS1h8y7fOKQq1HQDFTbDb7oqHUcw5hNccD7PwTM38hkt201xdjyTsMpFTPrvi5dNYnovPwxjL9Kyv9zEMvx8Gd1U1Rtcb2j020DRcAqNudiB102LFukJYEggwya1JPqMa_9kxnj-LFHPBhsNjePGWthmroeCXYdvUV0vAWVEmXnCrKAYBgNMJFi4ZWmzc0uUiJ8xG-c_2Rg6etEFQnSWK71h1ACJISipPY5m8sXEM3P46lu8biRDxAFPFE7oKJMDvg1mbyJsBJChIp1FRr0Rqvpl2J7og7U-jCgi6-Ez4rxtn_iI7Z2fm2qp2k3ke7BU_d5B5KOGBJNmcvMr5wxCrIaJy7ugfvr9ZLunKkde19OM8LhRIGDdNEE0TZ5gChkia4ZCRyOcL7p2IjS9f_UBWyqEZERDPVsusjiWA0vkwtlfhXSe0ap2UvhfDQjjbVyRRSGMVPpIqEwnTz8_D0PM8f8N3ciT7LCDV3gGhQGxgH-Nbk23V5QH6C44xyJuB4zsv-bz0QAOOvoAelQZ78zHwd6faOeNdSsN5suMgJmKKD7JHVJqQzhjjgc04Y3hiVdxfmo6EqB23z0Mlpb9TJBSzqt4WGW3OS2jVwQ0HjPvUfw-5Q3a_TTUggAgPkxdVCLF5iJwHaUqrTOvc1cjMtm_OZmGo3CSabcmgJitL8lXF8B18-2mucopSpsisPAXeAuNW5Z75nFroVrs7Psd1jhRUn001keXTwE3Ao2DAsizTrHtIH-LFTbuo4SRHsyA0oQ7wWqpIhX-VpxdxPxGglrzPUJurdbVrzOO3N3OzPsplx4Wn1QmA0h5r_wQG0ZwgROh6pPfXif49nRQKUkj5nlc4gIN19u0EoAic7PZWdTBBiBGmP-hgYn3qCBZxHifPHohIQrwF0kQ-Kwte2pRux87dXizeNrSOygRsJqIPT8FaLI5xWX0xZRPui_lDxc0wBlkQc-PXu0Gq9cstTMJSK-fWVCQZvNbowGcGZSHqOZTVsRhuxQmq2EDmpNC0YaJQf7tHcC8RpwcuYLj3Rnt00I3rnEAv3W_uAmje4i6kf_ONFoT5E9U7xw3hArT6yYjdHKgk7NTFWj8moV58wpisyn79u38iSEly0kwTsPz8eQfyYMM0Wz8LVSg4pb9V1L-qTj-leqL9Dx7rtyexkiz8U_hfOcRMlNqkb-g6E-OEuL1P1GxKLDN1bNsKFeGt0I-paKCRpzi2WOKy2DCH1qrUUNwjZ2gcEDuaonCUztK9dr28AYzn9edZHNA4hsQ29AbDXf1_G9fc8r18wp42JtULTNM9J-O0ZO3I4efHwCHTrJWGxmTuRj4O9zxkPgHDuQxuCHcMRkrbngUVHb55ASv0kp883tqkYvis1OlqwPLmYfv8uFTdaSJi95OXlezXTg-Z0AZAgn1cwTNuvcG6n0aM3cBpW-gpfrFLKCDr_KnPFEMjwJiBVdyRu3Gg7oCr_iCsuS5t3WIZjvKqSagDFbbHOb1ohoj4F2OOvmjv4P9p-YC-MKBpOxCSg99NplzRScvwOU-CPyRdElnfl7zL0kX8f2FX-B3L_3Z58gasJNhsCfft4UEVEgOq5I2IZjZOSzhSyvVjN0jR6y96wOjktLo2bDGCE4ZLHrbxPQWkcFSAGwS3UXQ_q30oT31Ur57SCnIDvrBUo1fmNbMgCug_pTczUmuPv7Fm5DkJHjwGmGSlHjH-5SEPgkivNBQcqFnhqgIZPlDpdqLKZBcaRGmGSd9uKIin3LGXeZ8r5iC96gK-52yJn6Dqg13WCwZ0bwqnySx2z4bQScvWaa-aJEpFjnNUM1tHPxszxTJ0WbNUAQ4ygB-L5OEpC-UfJ__dumivLWqq44pKflM6NzU89Cy18yt4uwJT-hyLAczFIlYVRkWC-afOivPkGeE43gd6dfv62LT3wmwweeOtKcXowcd0fu3kgI8Bo_mPDIbjvo01EA-MGvTYUovuMLiloqCxe1NvMafJXbsXKNEBxZNb8xtD1gr9-yB5vriZZMhP6hiy2GeV76fQu_xvKzzUGLqSgj1W6_EhvW1buOPF4ItqkFHOddYA0hJDEtgWcdETBATD9Aovxs6jipEP14RnEyeopXKeKX47pU6i_cfZHTVP0KKWiYvJnanWEwGwTbrhBOOVnK0TGpY2x0VLAf8dFZprQL9995CqUGVloWCeQ0LvXp0ok-RxQy_XYJlpuKnckma_KlmMZJlVmX4oN3YS0bFej9WGV12fRSarV8oy6_vYVzroDK20WEBygIQyMu-IXoaNA8Yhe0gPjMmeAuqnS9XSOMKnPqfl5c9QhHTy8vbGGo4xTIYKyt8uV6NBWEnW2ZTUtJhX532UoivAM1Dl6Mfteqq2WXI9bVGeClCxQgNNkuzqy0FzNZ47Be8kkc10oHZzZYdM4aosWYoZbVDjKKM2DUcyEEtQ6i2V7vmpzkkUbIeWgAYO9et9ZtCzPD2mKGsDE4oEK6tjysSX6Ly5da0ESAfsnetBHIMDtXQ97qRlqC6oBdgF-L_JpqJ5ryVpiXhvT7e8qXJU45sDTO6YF5eM_4pXFM2yh4X2x2vsl-iseLotXBexZHF-glPU7kANx1-QgilWg0Vfxs5e1Mg2mqPINiv1BcIjpNqJwfvjMH2PsGrVwFY0_9BRLwHuBrfNQdwLcdflhgrIMMtbisLHRQ5buQcRdxh6UDo1wyQtzUuPZsarPcO1-Sb0e1wO6a32sUQeDcuhai_R9oVcp-zBJoxTJJnVYWy_pEBsAZr-MUS-gAX0HuHDqb38hLv7G8Y2stGqc7SMvPxUhitbf661a1y2BgnLPoR2-Enw3Z4i0V1fRAgCZOK6Jahf0bejiJkf-Ehr6SH8XQBYFhdre8vfZOSjJabBst-3r1HF7IDgNElRg5q8kgvwyqolIDLk8z5riigJ4ckKcbgBrb4tk3shnNQKUoXbra4JU8KxUVqeKzuZUyTvLCMcKZ5aJi4hnL1UrmuSH0WNseIQZI66KuORWVrp-cTh1N9cTC80ihuBgHfZDPFH_c13iMEiQL78f2P4KRTXps4QPlkv1fdX0Ppf7xi8WTR9KNTWfP6ijectlU4nIpmYi_tk8vsXY3Ml0SFobuUODO040zrPPns1MNIKFVVyU5LIvFkWkjJ8kxQQW4_egb8k3T84K_D8C8nr_ch5_fnfhUXVRaJT6dPrZOdCz_LJWw7AEUFThnzFA4CXsiPPu1tTnchTpIMwSunsMqXfWttaqNZ08V7jdBOPsbu-NQ_QCdyMTqBmyMDnD8DxqTYSm2C8zOgJmqh7zRuMEOpIOuPaUOhb7nbpgsF9b4OAfFB1qhVNRmDS8WUgZKWwmOwOd0CJbgbFkD-UiFJZx657pjMgZ3CDV7rk5BO8Cn-OCmMhSgQHSoLR1J9H-PfQMj20SMuPVzPCGJKjazKdDpHaDPsVvRdRe6_1o72awZaneMus6bZuXzwwLWIHSrDXeevsrAi3gn7Lg3Aznis1pztUPSQImo_40gXX80T2mO7YGZQq4dSKLb7bg0kYyeJHT80HPxXiiV6uIB1e7ObwqK0-eYLFqTOEsWlFu6M0kWh05nqQUhdIoJkXe5087RajtSn3PUTsnz2-dA12EEaaJtcSJWbdi8wXZixCX42IlstZZZcuymJ-7IQD1zCxRwUK_36a2x3A3wRKMm1fgEsCrYci6JjKI69aZZl5lQUCoHkb2pe9zEb3RUcemV8wcmsDYjTX_y4lOmr9CShs_MaQOf_FE6wzn1o_o29j3D_fliJyKaBuczXWpkInC6_n9C1GmqUyghKeCIHBWI3q2iMpAUzzOsTv5EWXAey8uDdocY57ek-aafv9ArSNrXsK87ZNwwfMR2xz1Re2VBmrtt2gCkcFVwtBgomJgmu6KHZsMnSRA9_0NlrkaTRFCH2G6g7wiNVZ8gBi2AEqPYuLSUi-yrOIE-uKwVAW8BFgbMVPCJvUF4WiTBsIathThM80e6dHepA2NYO8dDUXdrcydfR9rihUG0AmkoquWU1pU-z6HGwq1-GmAxokuE-i2FBBkzmwmw0uCW4AhHtoQaLyGU2zJKT2oCfM31gefM4BkGaaJ9ltWjzX-Vug9Oi7uMWyN3E-RTvKe6LxEE6j-Tqx6coRIBVqLFy4rp7_x0XADKOsfdJ_7i4jZAFf2nCTsmEYx8S7bUUMLtn-HSkwXc-_nyulbdiIeX830P73CNaSrLlDUyj7RE0w2KG-t-MDSXudXKjy3zHd7nmWWE69LSvrpTTaTeRe_yltf8GWsdNuzqdms3-QFv6akoQ56MFGZcSM_yeDTM2oAikfc0ehPo73vqFVpwbH9INApcqNSgH_Y-j9zMcgERyWqfHcyqIvmWafpncyuM8qT4X4RbwjOlyqP2lCSvXO9wC2D1_qvFsDpVKQd_UTt_xPbHrkY2qc92c9jfTy01wWnwqqVmfObyeiGdp6xyTY6ZIeurKqb3dkVDl1ItE2ypm2qUFOgwuFkY4KDI8RWfqeAxyrRwTDWTz-P24zPZD0o0DjnBUSXzbiz0_KcswBQfePr3YxoDD6sLxmvg79nwHKBGBKmQMHevNAbNhrF9rZ5iGrCQSvsOSqiICc-w09blN9kIwk2KqNzEmIUd38c6QeQa8cXTtufvniSiU29zB4PKlM6MlYKAowPYUNhafLtmyPjbOqm9XM8Lba1rlhQu8gg44ZKYILVTDnOrT1IQL2VjHZSlmwP8UTOCPiub_eQYuQje-A4nZqEptlkXO5HjawjlffoDENDTovOECB1TRFJHRSfTg28Lno5mdHAmYNwARhy7UuyKRX2XlEaPu1XjzAcUh91gxINc40GgA6WUQFAdOGAiWnZAU0oEXgcTLUePFAOJNvhSuvUIfSny6HIVqd7jaeGHqf6UzNDzhVZf1pZuOVs66H2mV7h7n_PWpy0x_ZmGLv9xPhvcgBKDj5zbcEwHupqOidBRjPJpAq7pTzkHHg59y2inPFwz03KYmGP5OXVEN1auDVoW3j6jmKicJTj-KB_7-6UNBANAkRMQNAJrnqph4m11TDG6GWhJtXyFR7D0xRrMQu0kD93DsQTRyaLodl-XKUMZayi-071FJTccAz8BeAkMFBa2fEKuH9eIaH89wq9NQcFnVZCfLaAxq5sqorcvdniKY7ABD8cTfMjjKeIM-PLMEV_MKnRtVyG7xEnK2McaL4Cp9Jdx0VlFE7awnPf7lJ0iashQHLh-9t2M7YquVfVq596x3hxrR5WeKiSe-NdH0micmebpAO8Egg2laC5ZfKqXpNUnh2PmIML5r56Gi-L828nz6WB7p9RK5L7xVfTwkINzcAjOV69US3NsQ7uvjwz5zUUJgJ-393ck94W6uQ002GEB1me9r8RJLPPXXJvNOLpLBB1fHSk1N_EC9TjAhkB3t8OR53bpPr87n7EPTYuCpag8x7e3b-AF2hFlsjZg21RsQORarfr2axCTmk9hHcrPtP0FipgrIpLyiWapZO1fEEpg1cgiEIJ3dBF1k8wAQcyg3tPgiFOqZ7ag-qgIqkNe3qJM7CinNmAojuBYJqa2f5uEp4GJY7rYRFSGY4SvnaFco3tt1L3v4Tk7VHwlO56X4hlmnMUafTFGp5ubrpMRreWvKl-myIIDvRpTHMXsPRlco7b16hoUzG_2eMuZrbBzPDP6eAhxVBrlcfPRsz4Rk51UOEXWts-TdaBT4zaXG3LZ19ec12MXC73X9pHTToZD15VLzvxW0sQXV8FWcenB04auj5UlCG2N4Wp_pzTEiyuGXm_U3eKHWJ_oIOu4tjcGX_ZiZOAuu5djUUVoUNmnz2TL7gIxG4zb_n6Ewq8UlhIjQtYJ_uukVNJcChc4TmyNhOtPkU6VuEh0cHLgEZ54DJHiM68iAQNRlpQtbdTMnEt5ZJhXk_mvTRkYd8nDBFjXBEs3p-JsxlTlQpOscGmGOwSToAobPrvE7_8jfvO2hlHiILV_FpAYEoDE0eiUiLd4EcvdqdKbEbVFS0L-r_LASnGG6nnCLN3Co8Q3zW1WvsUh_6Vmfq1dsiIHZfe9vztYjzyRBWWh5NSfMDFvCSCGK-MFo5OiHEA235jfI6hKYZItfAw3zsJ_sr4B2i3DhRWLzom4IeC-HPq7zAjjqh0igUtrTIGk9YaDDWJ7kMWkNKo_7qB_I-vALYsNLEhXfpR_tuO1CsXxEiTCbWUsM57fndF6CNkZZh_CcDTH4mxJjmQ8_RPSlQo0GSBl4bsWi9cFnh2EVT19pfywV56JpUP8kzrIP0DAz_92EoTPvudA0Qgo13e5pQMmm2QddA9k8hcblGYHGBYKdFNGUmKhVCO-uS1dH_df7Yd0jFC0ur2Z6RSNO0CC3Pl9kFYFDi5miIu6pK1nJA5qJR2rWSoWsMOZV5rrnos_R7TrKwinw5PZLsL2oCj0vYPFqJXPuqMdwY0wr0NL9DomEZlsc2xI4taFGmJ0FeexVPMM5l5rOAz1Xdt4hQ7XEW6HrQaZwMQId0Ik9eD_K0bsesWXnQ1gIFK_asigLmcPGCOEVDmbmEzt1ASDx8jIdrBvETgCPNk3mjF-LSjIRxZ2PqF3Fh7CE-2hhurmWkvApakfS4M7RobwXX9R7cDMaH60JyokeMn09sccn1uuvcoEOZt2ZP8oxw0stZpshc0TEJ-WPNcXKRt9ttluCspd7wJCuS7nHHTmby5tExI1mEfHIDFTf2H_OSqwN6aKrrT92VQi-ySxBhrTPb9jCPfiELWB8OTzZekIuxZ7dPUMoDfF7LcV9DyLMR8406j6KmKwCY8Gg_teGYflxYWqFNxR8itEPReS5R5cZMuYHZ7b1l3tD8JnFKXdpAliozs8nIWJibD41PCf0TSU0NTOv5zInzPr59UkPFaR-LUa6P_k5hSKgyNt3egahkyHQ8i5w2YuxmvFpDOquHSOkxC9-09PIPUWu_lA4mvXXJGT3VE0zcM05KSyA-CH5d-b2FxqrrwvoOzy1KMi3GFzupibraurlvgBh5PwLvrcN7gmRnPHR_CmPZby22vhCNYIjLczV6VxcnUpt3M19HKGTpwmh6hrndvAsLxG0wwi6zL6w8zWKKaGnMACDIACNUB0D304c8G0zk2pGkU8IQ9r6o3vzHYo0HmXYsIRCsxj9K6pqZ4IQh2VrRtaQbn6Op4WL90MZZnZebrwSeqmNEJ-3YKeFtOy7_cMZVUSHYSXWp63uwe49eqneHjjqQ0QnqXJCmVsLn4tVK6i-MEN1qJNRKqTHMoHjonq5WX_xSNQdtrkYN9OKdklDzYGrv2JA0wsCFlzg4Aau34tT8-iub4VgzIGr4cnt79NgKhGUBBNCDFVuKzsHr95R8L7WEiJT4DVIU9YaHmIoWeiWp_4l1k1HbWR-3nahj52f95_SEnJtz8wrKTOk3nECrkdhySS8jON72vLAxPnh1tkuQMK2Ud5cYsy5nxlP9hCuvxZ6lWpd2sqqkaBxaLQEpyZlrItdlSS0xukr_XBezP8ncr38ivuWu1aOVSw85hKiVyWCWV6358s0_AFKfJs8aHAigguyGBjgn-GL2eycqzt-mgOdFUITxKm_9sZ-I1wx-8-2A7eVDGdNaVWCCiTZY6fdwl6zEIrLJTBUgUGyB5h_697AfI7AG9rt2DRRPHELXzZ9vNntlBnbSpRNy02ipKYKrSdfkeLVS_japiQ2rSGO8BT86diJwPYcK63Y5LFng2qelGGbOh4Co-tv3PhkfjUZtepCOUJmywmmNLhwWTrjKc9e1onnpYUsmG_iEY__QZhTqj7',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_02bf8d298b645b0100697c34ecbca081a09d255983d11483df',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_02bf8d298b645b0100697c34ecbca081a09d255983d11483df',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_02bf8d298b645b0100697c34ecbca081a09d255983d11483df',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='', id='msg_02bf8d298b645b0100697c35202fa881a0aaa89ad03dd7d453', provider_name='openai'
                    ),
                ],
                usage=RequestUsage(input_tokens=1872, output_tokens=1438, details={'reasoning_tokens': 1280}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_02bf8d298b645b0100697c34d6e02881a0a3973161bc98b257',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text or call a tool.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_02bf8d298b645b0100697c3521c9d481a0842be7ffbfdee51d',
                        signature='gAAAAABpfDWTHqH7_9b7kgcIE8c08g9zwhTW0C5JXo1OPXFEneKCmAst-uMsZZ6Zwq-YS9uBDzVfNBJrlYPkg1rdFez1N7imsWXrrAnRlSFqPIOoLglPWWe2Im9z_bFgir5RaPm2DvYIiUOos57Hqw7p-V3DP9BA_ta_sL_T8D-8kieZq_Fb_o8hlYk_Gw1Cq36MyTpCBX4H4N-RXLqjEPf68oVVrW3MJLVWhs9FVT6XgK6265wusIb3DbWcXp7ySorfp1xUZACP40eupui3tKnsO4hkLuCtnGRfkmpzF9gJ0y-x2f0ZYN2yvBO14TOHtPEE-LXVmYBLIPAQAO_ApEMq90ibEibggO3EZbElJrNTJXXKVtqbrXAIYi3-jc-Iqj1z5cj1rwmMJnZX7dtbfT-iCEg3KJZVg3rtkkNS5KicADUrPB5EpGR7dOKXdR-6WVKQC0qDjctdyTHDeVR1C9dRiUA0N1pOlTuHBYfryRsByG-8zjBnMPwSkv_uCorXEpMStEI5VBLwjyq7IsHEqJWj-lMEXDaL-yudqNA6lDxAWu88dRFlhlVNYJeRQTm0Ro6XnGLR5VW1Hqr8yMx-9qsSmD1mRr0i-VrwHXFjzKnxVolweIcrgreECecOqbsYAYsteWxgrJtqsYh3p6_tG9PtnalH8RFWeALF4x7wKaeGPuNO8dkldiT--5AeP-L8Z9ay70FfctDKPX1AROSE2jT5kA3qVEuM800PgENeusru-Ngto0HqPXfpZtyFVHgdCMcLU4OHGLVHn4QjzSCKUOhZy0Ktpc2NVzojFgCUQY5DuucyC8_0Tn4rq0bV714HR830kTi-8a2fpGMVLP_gWRCwota5XDWcfXgFRWEXE0NJcYkneazxHxnZQwSTbtq1YZCCkbOZR2qFtKOXAhWf5NQcI0Vekyhm8G-5tpDmNsfM_ioTpRaUcRjVdkIGaxQ7tAAR7XZIeY9nNkuWq-XA2PEODqtmtHWMj-GNlkPA-l_0D0S5mzuADLd2Lzfzrtop18YQhs-9j3i8-pVNAhMP2Z__i9_BksDFTzfuww7otyHMMZA4w3E5iimVttc7Crgot1CeWrn8yJJTHrxwgw8kDzngkTYIGMUfG4JOiYjKJZ32UKi2uYzjvCQrcKVxPby18BCwnlP3kLebEe7A-SXhpZ0-1rI2CgYd9L6WpCXIPSqX1DKBNGVcyBS01ak_-QmUrrot6sJYbfccCNWeAhkA2yLXv3P9nATasmBnwO5O1bPkBY83a9c6Sjk9lnp7TDwt1a-UbBrb34wfnboVySud6XAl1_q960P8Ghwtsb6SrP36gVtR4eUoxFA4tqFMCb9kaQaBLsmMK8fpjpMzs6YZ_dekffgaaN5hamp0zyDKemCrB_-ILNLaWsvkV2LAqPHFuGdNgtUHf5w-22XciLWtZL3EjOiLrOMVmR8LBGT2VIy22c8BhhOoiJs1X-z9SpB8IjI3RucN7aZv5vqiKRDy7r8KBaa_M-_GUZlyoQ69YieQ-SicIMJAoP7eYdO6IWsVMq9G5ng4uxmSo1OgkxerqgTrcfPlmPByaZUQggCXUHrUu3W6b_lyq-ooOIp0GuuoJdCBpoCZBUV_uysI314nppFMBM1iCLkTxjUq3cOgqtJpKbNwaMUblJkONfm9iF2ngj8BjVbMRczw0TM4s-xY8IU-nwRRRr3lWVPvoAQ44mV3UGnC88dGT6F51lCSgphzNesvZu9sp-NFY1hrsf2Yqt7FVoIKGNUnZooLMKba_rJULg_yD28SMolHCeNmSwZ7KZ89lHy8VPmZ6t_5gbDCNDUZouFt_N6of0U7LRrco36TJp9qsi1doNK2IQxh2oIea0maGjrq9g6w-sTfUoT2A2QzyqNWckqDjU7UwTYKyGgSjgoBK-eWXM1WT9N2V2x0XD8En0fO4etI90EbDQ09O0KGlE6KtRx7hrz1qg6Wq_VUM6zUwWaPJnZrxgQmfZ0cyTVTbjRMbL76XtiWasUNNWXY-c8523HqTapdw0w0Weqyp-kUypu5ul2TDM8pNs5ex7dvtk0l2TOimLiXVMaOGqkn34-j92BywaKvDN_0hAOSNBTvWEZilVYgBVeOkwJX2SqvjXnqbh71gXNjmZrHry3AoZ397dsWWkmO0-IzKnrWeuZgLaOwJW5nha6Wx-X3gEp4pv0JCwoJ31h49DFthm6plAsjLR36wr5EJAxFTxtKkKO2z8kKE1dolhrn-QKoejpEqv7AVRbdrYHcKB9pag5yCJGOxd_i9ah1m-CD3fAIR1YaDgQbrrfIesSlpcy9omqdrzcfnVpqpjVk4zTCHPQO-E9REqrFrHMzJYfRpoGSnMmqgp0nJGPJzXPN0g_Sr4IQabFcHBcTXjEzkXE2gmtEr2JQzX5QEaI4IQ8thwduIxogzap0jJ3CRjZRyamzg60DszrSKE5YKNkKb2AQQEAQMpuxtXKtoJhW3ifpGd2TLk1fJbMivZlUzU-aulZrNkXmvPn-gRL2don64bwMtM2fjL1WjfmsiQ65KXuSjTHuEkPJkVhAskHEjlIid3KFXl1VCyHniNSS-Cxpnc4arL3nQ9susIJPK1c1pdz45JUGG0PnxXDp_s9gdhxNG4r08hMDjwDLKCgHAhBN6nVgXhSO22B1-ToKuPd_FMK6U0NCGguvrijOw7fDEFXzCrpeUUR_U-t1HmBoyYfo9RRveJUrSpAwf4IbtlUB85V2s8lNkIi6GUgh-lQQESymVnS9lu2O_fHkI8HKUJloWzJ9xg5vMHWLM00dKk6RKl0f8ul6dERpfzps4JESd7sWBYds8j-0HwQE6v7Re8VTuH1JZKJvZeyMgEP90L_kiTo35QxyisO7dbXXTn5fGM7LpLaLPYX04lFTvRdvyFrLj2I1keFyhGdD5Agl-uHTyW_C2Z4mcYbpE2yctnIlEi12sA6oyQzyZDzJXKTe69y0HO4v2_74WTycNYCjxw0a8qVtZ4eAu37f-XU9tl5z9HWlXbpaIUbYEQZ1IEetSyXLAZkK8-HLNoReAilGGbhwceAjP7QSKZO0-Lal1rn3cSBYnY82pmgsHM67P4-OKrW5_f3YllMBZndWquGIjZnZu5EtKhDC_w5hAjDLQHvo__73c_sNH3JcF-ZRtmyHTOlfuW76-ubX1zK2DC7JYD9KzHNXiWYerkf78dPgCp1zSQsr51bCN84snmLKhMWujWd09XiG68m9jKxPC8ZHvVw_In1lNyG72jQHDislPJDcFlkoMR3XjiAIYaohxeRE58rQoyfeXDiv7_9VNYbV8ImQIPxaTCdQmnSEcvkwk0z-SoFeT0gogATyRz1lthgA__u28AltiMTuoxT8tOET42pc1eTY-IcRFVf6lajOAKP5DrRVvCJU3wtEpbVvRnwq_3D4xM-y5saEt3HEGiNT37Zdpp4GNCAInjJX0Z6y9NiVq5BeYpmNemn_pk4aXDy6QNe9v50pN2eXxmU-gVR_6DJl-TxnDwKRyHS1Vezzo6p9_Az6aVQhUMZ77i3ftQ2W6XbUC9ACOffHvsl4ZEieOWzlyUcUeDYNIFlSrhW-ZVmk1KDwtu6OswtsIwb6CcgTjSTRVh8nT0BLhR_WAqbmxQ4yysJ0_ghLhM3p8u5G-5MZPUTGSFhLnlX-pRvZISfWb-OaB_VbOsp7YSyPxbbthcz3IKP1W2WkdFZ31rPrMYhJhT1hNtEbdx0N76itnE0IWYoji2QIhDHMTuq5j2s7CTM6QMMgKC44q67u7bXeKZ88-GjWMakwKkFYGl1P6kGYs8jS68sFLMf3hLizGMF8RHFpWz78OK2EgheSV7cKkFHFAT2qcT7AYZGf9o-va5wLdt64EEGNwR1a25UsjeGumNhZVJs_-RZ39axEaLyry7sTX4jx_6GyzgW1BGIZDIrDI5vy0TeIiJQcLbOFesgBqRDkmRazwm_Pc3o-1lZ8Yg_N9t9Zd5aKp-3dMfOSDUE1i_pWDXxeSbST_dSDGqYwvLeMtqTkYMZolPeuBX950tZ9B4rbPTxH3PZh1sU-8Y6DOO4MBv0TzkBxy23EwS1bWOwXsLkjonWBJmAIfBRotz5_WWxaFqz2jItX_PbMTt8vkrcNE0AaRyYQofRFNt7_q_4amnRpLEOyLLMC2rjOChpTkZ0FQVuvKMeyiFJ4BTwrMovjwz9jHZFR2Q4TpPqOkotjKydgrPM_Ua534kbAzElDaPeDgDMqFSRd5pak4Qzrilqjx_AQwRiUeGwYXxgdeNbhLJ1UYNYEaopKKRUNZN1N94mENLgy_oW7fSWTumU8DftGpsXVmkHl3od8o_4X_cD2WUGS4te5YmJhpRdsIl-kU6ZT3NrPqGiJqK5dVMK39M1fJhHB5MEiMDIPILctZ0NOBeS14o2gZ9S2GKoducHsaGRMhWi1eyHZNDIsxWh0EDjXzIUDO8fC47hMq8QVf79LZj3Ku4kbFIJPh9em-reV5hhiWuD2P-vy1ZynyAzGMrkJldAoHq4WcGT008tdY9tNYTikP-LmxNKyI5AKwkLPuPR1XjDCn65uVQgJCgKnC2oDBEsv0AHVnvj0wkMhFuNMuUohM8BqRq6RzBNl8CMp2CIQsSfoV137lLmDzolIPRdIswnuBUdEKnINiPSQInJDTfyQ5nc0xAEKC0AC_-dXNGkKTKv_Dx7P9tYHrBimw2wpP43njwjtRJCebY8_dPbglJNTaQ8pN4x9LqP0IuU_xKJAegMkUIL7C5iyiMph1d9PDfglrhgSH0ugy6jyCTOxFe8HKJoh-ioCZzvGb39_327o8nnKLvhSyt2j3QSfk2C0jkcDXaTpW3LHhM3fYEijT34WjuNQVcuOyyoH8T0hQznHSBIu96Xj8cdOQPBDf_mjdMoq0-hZsT_w8C9LD4bglv5ibbkvai9L8hvFkZu5AWp3jm1dWvQPYk5nuRLgY6osbcciWQ7ZwdIA9fDynKZjb1smOqN5yVJgX_aeGJ47jT1DCXS9u1ZOZ9BQzBnAlk9k0i8dm99RQRzH-pNeOe0h007S9dHAFan--pfXdD2kw-c6WWi_000eLulSlinmokRyiDplPtXVKvGB7G_R91YGzZa_4njZQ5N3iuMRez8OZwkORovslK8OhWGQYAOjQXdExGkaTc4KSNkKXmDlgnCKe2uGCoL7VewPMqsChcyLEO4mlxMZY-GLBqD7knCQRTPnvaYLNUt_aZlnpW634HCWVX8JD3muRThP5gDWdKAcCOq6SWYuAF9d_dz8kCNaP3SjCsXKDru85jyIAVqsDbVIOIkfSnr9MI3ozipwur8_EZEISALav8jKQBPupr6gBdF3nvArNeQgFBZe8KiKzSpRakzwerztMuJo7K4_32KWnpIFhRW4-lw_WnfFhJhZCU9-K47H3sGgaH9mVth8d_O8wHHTuXV59i8GWFNiCv0NLlLTRv-Hg2kUj_AoP97Bkcc9XJ4Zb9jtaBjs7qHEr5Ydp_reIemwHqTAE5TS46zJ4P4ZnC6miyj7ibO1K6o_GP-1K7i7-IcuS1uxram10pqVevSFBkrhbla8aZfvkZ4qa3jYpQC1A-wrEeq8eHzKJ9XheWgpDkOyhlg4wFhE2izBKQPTfCYrPvR5v9vobK27gld_pW9Pu2F5HSzIJjoCvJWqk3O9Y_ygkheZErWwQkKwtWoE_utKtCho23xPpa0uLqyKs1uQ2Z9N3idXNk6JzrEzejcEgML3h8mWT5uuOAIfzkFtLbT2u2yrxk4RiipSICUKAjcaf2wvTxH8so90XFk8hfLX7PxSrMKOB6zderU4p6nXqAVwjDZ--9Ly35GA_IfChhSx5cxVmeVK4L7X9Re1kPR8omCfuDpRmjIgHvNrv9RQwZkT4N7ZuucvGZAQsBk6owB60gMTqE9zUBKEEg1psqU00MHM0KFNzeyH-tQtwGXEmMvxVSfdYFiqmn7lojHmaKEJhCeIBuKM_zTl0bzEu2-NFoRL8KzcR9H6wZ57vlu4GMfzG8XPvNR6iIsejpw8D6IXExZH9FGL7QV6O_g0EEHIv_L6uMiNzGo80f4RbVmEP-J5XwCqp4qh2MRmcp0Qp2QvXn-RM1RXM58-A5qu0ypq0FtAgm_msiydt0N3YZ-CKEQrqRXYzWxvETuVTKxvE8VODxdV9zvX5zgDsOmFV612Pyj7N__mdk7ZrJfvV2z7Zm4cWawo-aR59Nsq8ye4pLe5sSGiyobsZB6Ku_fZWqiLfYtJG7KGpSfDXnicK6jAWooqr07stRZ5pLgCSKVb4-M76RK9IQ-711pWdHZGrKQGEuGEN5cf-Oh4V5fUKWxIPQHA49lOGMIG_fkxjuaFPD6g_joPv_lcbjjq54WLVz8_70SBw3HA8R8Qn-9xBd9MMZkl8hkRpUEBnGLKN6E-wX_L-7QltGMb-ruLzXK-lsyjF4CiyzxZ6-iDCZDzhfvmdMd2wG-oOYoz6mPDXB9zMT1qqYPKF71Y0439cct86bziVKHYFg8JDvcf-Eo1jYCAcnIlfp6PPRUThsJopom0h811D5zw26wf9rP1Xinh_aSoosen6Ezu36aZJhUF6GsMEEnrdDu-oVMJEY3prWKd9ZlaSCdUP-5aQO_L7OqeMik0hksOleRCUs_HCCzgjQS7XoX9MnNJ3Vls-RsKGv9JRQ1BSEQDGzrikw5ivubsH4TKhi6KWV3wsIDpH4jpxbC5dTj7swdqrd2Fs5KJcYqRJg9N2IaO8entkMHPhwg8T0Vt33rV60iBls6N8M3Rpkjw4e1tvG34FQIigqruzbwGxvKjHiioG62LGmJTk_DxrlwRMwZDdIVcCC6OejTK3PcpdQGqmDCGoEXMo5KndYqxw0speuoLgLvttC2ldpFAXtnQGka7-ay9X3Dfrjdq-GyZqw4R_t-Y4-r4OhURyCOp8aarlWVBN6E8NzVY0CRuHDineEjNtV7nGMqacPRHsDf-6eU6C2z2ZK4ML8Zw9fCWnXG6gyHwckvh5OrdGNUYlgqG-Ql0CckZHlAIj_5zB9T1WUeGd_QwS6n4HPvSMjE96qM6bdfJUpWeiWlZw90uxJCmWLl8QV6oGw0C9MQGKx_5sX6btq9RCPc8-aRQPnHScPuY6lp0yc5YVp-rv6FkIe87SfucRlJYkq-JrySRtFcIYII0oad4qQsQ083M9kusIjMU5poc1KOSUgSPhjA1g1_HkGruiAwnEeEYazHOyKyn23Ec298SD1KtR8XgC7eZZZuWwBpAnl2eYJy-NmuxzdOToQoapJy3WcnMS11yXTJrMYqEELRCFz0087Wa92Ghqw2VHf_7RKg8lQr3IIcGrMQnUbM6jkLy69PpDAwdadPMmITUFvkj-sZwAtDzm5aIIj3XLJKayX6BjYjdY5dPWCXSqPWd5Qgm8Mn6dQGzlulkkS1zQxyLrYqFzq3JumnI8ibjK21gMdKX3La_fGKYJSPrxcvksHE7kgpo8Dg87rxXfT4CaYdRKjydydBmzOgimdB0NPTaexfYBEUwIr8BHz9coAmJ8LU71n4eQkJw-ysjjn56b9smH932dPUj53Q9fiKwrEFuN-3PGJg9LOLeUQOkUhr594foWS8pIiKtepzOXJ1ly_kdplKBCWqm08nNDO9844tMKq7zhgAfiMkx0EhfN2SbsUzcKpLTztDWkND3SBEtpOArE1EPmH8PckCt8V38-mnqghT5xxqvUfknTaXYdoBhNoRair4dJUiu7bfeejQweTu0LTLroxX_3Qi3OrM1K0GwS3U86sr36ulBoD7Br8ZSbrbPF841k0LYDdvKdL8Q5Lda-P2ZLERMiJBqoDgHs52qKzHi7NbTXzYa94n-NtZbuxedyM-VlrItyhBJG6YtIzek7SHjeVSkW8vnAaGNQrI-jLgsRecMm_36VB8Nenn7C8JxwyY0Z8-HSSMhv8s1D5vpOma6ERpv6AbefE8V3U6Vn5nN6TfhAKIodwr2lIPTlw1JNGKTUhm7CjWqERDJdSnh98uePSLVbs4MJdtZTJbrw-wzBpzIOF5yTzK4rmHdIG6i8QYWg46gbVF4IXjVQSdTVriXJVRfvyOEd_CpcQtc3fn4Rmc2tUG1sB6p6cB3uBhkbmnGUseRk_2X3LWLiQTmlRbuqKaY9GQ7nlnr0tY7IWpyCCzh1b02NtTJRDpsa7yISmJvOBnUTY0v2FMng_o3HEa5_gbogIDly0ftYbzOxmPMkC_zJCZooF4Ci4SaXHjj4pMArYiUDPBMl9CbpBpoK7iLnEG4CWTTZwJnQ7hy_XD2eoaWrBYzZY6N0BMJBu71dsjHk7_UUNbpYJJG2dmL_VxE_2etSwuKdA9QeIJMME74vtEiQP4S16Bet2yVzvnMWU9cu-iy_T2JT0tAmHdRTyKE5B-RvbfwQUc1EK-kzNPMaTHZ3VWjkN4RGf6g2IVxZp3f7XReBZhnPVmmbY-aThNYXuJW5CbIE7ryOc6JjWmEzU52Gw-LAWf2nFOf--yXAC-QlzuvU7nhSCxSA3Jj-kFPGub0xUxRrJ2Zktx4hldOim9YuKIOnVlKEuQln5hVOfqeKmHAmcMCBGXQ70ZGLi7Pp_gXSlarhJw08k3u11BQnToE6gAs-DoiKZAQhhXGkGnmArL-j8IAZFj9Np4yWVsKHmI5CkR-8-u4vuwO47_v1acGpTJOR-Z2Gw4TFxWkFf-Vr3bUuV8eATmWFx63szxZ4ooHIpnYuP0CgVL15thrCLou76MzuPzZYgXFKNCWVqQMAMNZbn6SYn6og2oqEUmWaym9FHj99aJqIpLXINxLrZv84O9W4n0u87R7IMIOVKNYqP0LqiN67pDRvfJ5CqfOcmXDt3HBydj-TI9F_O-2FCSyw2plWDRANZrhA0VIBenJDL3cP4tgxYUZNW0UFmaeVpCa6p4NByTjOM47VfF3enF-r3AOUeoW-fYp61Zr_0wCqJuoiNoXARXn5fBF1kj_eCew64Vp45YwCt7kU2yjXUpFNy1e7hSDN3YA2whsAin8wzsqThN0nqqxgGcgrH9CgoVNenVUmH2tgmL2y419-Pl0Df84OEPLgqXnhrc_oUcvoZBlNrtDyhPQ-jjeKgKxoLT_jR6HTDV1UHb7af_8v21lkLh84snzVbOfneU25eHf7eeMuQUmMw23mSXdLUXrCKZ7n_ECDzJznxDHiejRFuhtInqAciJIQxEMGb996Z_3OhpxWOZQkxJ5hSQ-UGM1qGZQsjSlym43ThyS_hI2zLDII9H-UkIZNqrwRyL-RBQY62EQpNfTjYXJkNKb0J0eRT3oMXfD-KlcmQhJLaJ2OQNe-NQ9oyRVttTdM47GQt5DzFfSmg8yfQzQ5RCRdAUleuqGtaPGC7_nSvx4T_eMl1bXwqxDQvpFHH-jFtQf9jZgpu96BRPLCx_0ujP3vIP97yIbG9Zc1qUoHa_ny7iWuYF-KlUOVAwcwgptDzMYkt0fURgmqFPBBirt836ZrZ_2yYjMAHP0Jm2_lC0TnaZDVcHqpFhmLLY4SyngmXZmt3a7Cu3LGaiA-pfVui4bnfz4-DJAnWNeE1PbYDPYKhwSDc8YwrdOOemyaze3pu0Xk6YxZ6ooQplyQws1CAhX6-jl3CIS4OlDVh54QlzlynT86fDUPUJ3Non6_8vkuVqg2m4qqzb4yEpl7faZwLIXcV7lVO6EmLCYWjYe4OYnCEYqZzhs5lUj6As7ZmpexaTMR1ImPML4uyMYr503j2YoiPsYVkq5qvv3LV16uL4FppFps7Ry5bI5OS8nWiXBzE6oC4OcbTJxToqRdbLTI80s5o1XAwgofOHHn2FrjIgZc5THOtvn6odM7iZqhiuMfkY3DiIKdzGvaIwvkz8cGjsSRdRKNo6thZ1fAXoaUwsuM7WE_1tcoGc7D023s_vuwm8p3z8kuBfskV0Ex0Udhs8lOI7ybkHijCkD1fJ4Ou3quXO_3mMuFMm6qncXQ70lDetUtodF0omrPcfshRIM_e6_chvvN4gjtJlfgjEuyakwX6VBPNgSbx-7jZW3cCwCF_QZ-6vQyleC7PHahlGg1iE4wPb3TY8lOU3mW4MC-cSdZMreEKxoX701ub5Iwya6gP4miijnWbsJWy-P9tMZpfA-TGBzklEUHlz2V_TZMTLrkk82l3pCX3ln8Jyx7eQz_zplW2ZcB2HGlDlsIclPd7b0IjTn5rdL6Dobme7JZQHWv4a787rNSabugVlD9BYtpRSqgShZB_SlP-8Zo519Volw3gNAE-fBsE3pUHUXjNeI5zPDEI9EgRmuKxz1RnoufKfhsA_ElnVLBrec12nNXnPlQFUJoug6Dxju61LaGrdxSMnpUVZVYob-e6-r3t8KYO5Cagi3ttEz2V9EDP6T4kS3uYGMr3kZy6bNIHXYzBT2uXIgO4i4LKGkH7qExz7x2WRIoffOPLigyf_CLVb8nQOiQq6zKx7xhhYlq29qmPZhTy-FhsmadZqUghFcK_N6ib3zH5kv4qphoxmlp_kJxW0sCAJXDuNI4T2zEmBT3R9eTRzMArPBwDlMJOgOHPBwcdImLlDA60TbGLDaTwcUretpOtudq2fPADHOK6c9y613JtBpGjt8TtOr4nblkNiu4-FN3U8yY-rgGJ7CFHzTF9mxGB__lvuRywTBVrKRYgxEVvUEzwLCjguJH8skGBvhsH6ALQE9f0ftlYUbGzQVPY3pDX0BjiQanuggSOIoOm9HQmGIo3z4B9x1aal2eNaUTYAEOEcQb6C8ETuZS93SY_z0y-M2038t74C2iAO8m9q9LTXwLuEIc42dmBq801oPPgDXe5nLgCkd1ltGqd_2cfLHmwRlTY_xfC0nNGodffiP-xmThwMlS--rnzhhut1jEGwmmCt1IBL56NSDXrt4U2vDlwEJkpJ4XsSyr9Nob1u4aaIj4DfTkJG6tQEIyr8frJbEWsvT1nKZLNvk3sbDW8nXSx9qhTiE4dcK9Tz9baxJHXIHBl7PtxqWdgiXuwK5tKsufNoQKGP7sJmNiVNYerX9AKFHVFEO687BD7uGYDkovbrr_ZbcL29uCyZOZtkIyFiH5fYOQSw4LYdqEEln4OmSvAg-KdFhXcwl0thG31JY23WJDoJ72I1cDsXOaWLQ9jiA-p0Kiy2_ZxnonXGlR4WovxPwnVgCUcJug_RGXkbPeo3TjaA-Y0YSUX6jOh6OSwl-JDykeAIr7dXHpkZ4UjLWY7wVJD4UhLUVnu4TnazEVTIE8fv2cv8479bIXf0Fd6kHd8Ex6hbAIGZO-Ia-rWKeHLxgO86GmKPVY9VwM1E2V6nuZFXZ4fX44dIoQtDm3c1IhS7fJIZQyP4WWXkQ1WY1U7Lld1x_pP__k-Ye01U_0oLUP2GFf-cgCGvMdR1YKS1rvM1O8nDKC8ycO0bSCXFbdWViWqjy8-iAhOQPgh3jUeSh0ToRiGqP5M9tYRgi_L2RZDvFg2_P0GxXptyap8ZKaL1B45bhguQ2G1tBlmWlilPZ2QLnOyZDVMziWtXUh9JzfnYeh0OkVlVdWIup20sA_uF1CICyhHWK8OWTHj2L3ewU5DXiJvZP0dzQOsJEm-wqPYpMW5yLK396QM669wzYTs0H45-llEaxzfxTBaYo-r-d4hh1haZ4_Mw73WZY6SptgsqUM5SWu6-SFs1QPZeS8jWqf0JNi6suHnkClwmVGhm5w4UmwNNd7nQcynA0cSB_PihVeR40F-57khlLdHv9A0VU4ut-bfhScSrUKKLSRhKkYzON6o0as6loHBwtZPJN7337rCoLIZpu0uqvAje1pQgwAV9lL4mXBq7CST7viki17nQ8v0E4z2KhTGNZq-ToYv-oxnqt6NIkXImgA13SXTcDLyIfitLQgLBf54HSC6-_noWU6OOQ4lVo5Cu_qmyXOf8M7xzSu8ItZNj-Do0fyvnde5lbjCdqSBuoJ8cT8V4u0yj6CsrXdIezgMXS0HhI4McEz1uTSp5thkypoUxoFYEB3z-EcFugXMZ8IUIYbwIWOGpUgjY98x0T-tBu8egO_pCG1GwaK9NEL-dKuLCvTMpBtzncsB-QWTn3h-z3KWxrOVaIDItRgoAqjOsG0yF11oMVRTQLMYZAT9K7vr8LsfiMzNgQxojctbzU2QCa1NS5z3D0SSAOQNlhXEIe0K3ta0n0lBSqudfzXxm4mo3UqEakXyJ23uilnubgf4qucCG1c0tGM7wBYtfEk17iuz_gKHyt1SILbinnnckpfzgpYq0yID2y_l9uNe3ttOHeLNcvUb029mCE9wUjfHDKZIf8fePJx6Yekk7ZD7uK3g-GOs_v29HQxTLqwpTR2J8B0LSKnz20N2GIUXH9XKgq_mwiynlVDzz7Ex5Oyp5IzD-w95Gik1Nu8KLoRsOeKHTN3EWR9uGjsZBrd2cYLEjk1be6zL7KmUcqvovb29j8e4AEGH8JVpHDkDvvOZJcitjhFJ7Wqj6ED6W4R9sCvzqYEUG6VBJGLLhE1JKG1x1HZisMcFtFllozQ686-hVUeTDvC90nGgYov0TAGFtmkcySjbffQBIiffd_8Wf9M1pMmQA3Q5lXSGTkHrya5AFr_Tv3vEUSxSU3ziJWuKQQUsc5BU_Bje_244Lv5vwoa85APqokVCaD81P1oKH1TL47bxW4ODE0ebfUaLeUkkcIcKy7YHAWIaIN8vGXtltDqouDXl5UotQF1AazrjBhrbQsoIGnsnI6AKOb6LHjEYqjr0mzWz8q7f6EzOyK5HB1ur1u4wud0VqoZpzIZ8uId2TaeniW3e9Kjx6_mdlqwcfSpdSgz7UYgBj_GlwxBOuFJ-OKJKC1M0Okvtab3qTPWydIp6fdEbkzeOeQtzTbIN9mY8UkGwqf0WaMNgcFTeJU3aTI7XkdWThyD0cxRj9zfi6VxixvjsXYnw99nrSqphE9ctj7081AuYLFmnKth11o1upuNFQ0hccY4Rv35wFfltY99AYqqBEDy8w5jPVy1C_ZXrTn6rlYh0CfxqLadOwbfNySjLXU9lNpt54H7QKPdVJqcbAH-WPT5Xv2Ja8Uo5oag9KofPI4DfOUrTSYHzvJwRI_JzDGeP_He1CMGY7V0-aQH1mwZjEADDqenOEZ0rqEwspgptV0w74BZsAawz95L3Nu9gnaM5ojL_1HdO0i5gF58x6b5je0ttoEMztTc-YL4WFIAVr0Ohw8VZFy-pQuP-MnziM-9y9nkGKkam7hD3A_dF4guJUe5VDrALS45g2wF0jm3aOzHOjQ9e7EU_OR77rJ96Vkog6XWwTjiTiH5RnLcJHhasrVy_1FQvaaUdUWTg264vwwo8D_reWMTp4xih36EpjaVHdGBhure_oh-vMIrHFMRx059pt8TXGx88un9NYg6qCb4d7P1T4EnQIr9CuMbkVuJ2Ugbt2j18Ye-EpUflBrErjho0AG9D7pinDEx0GSNhf92gnXcIHlWUc95mUyjZVYQx2iM3rGrGIHI4QAvud0mlexZX3r92suOUeqUAuNuOlOCnMvXdVKhzzSihGfBM3fX6sQ1rOpukUSba8qTPW3gmHLbgul2iOlEhmYrF1QLFbsnMJCytZWH7er-71DNjLGhbkEpH6tT25-ymYN9Nv3OXZ5Gsc9Q35usnNCF8FUdez5kzgW2uQsUBVVfR4f9MoeVsiyOJfWm950F5xROiOsYJWA-EVh08WdGLlDq2TF8HJuPJdub6M1zySNNdR8fOyT1_RijSYvnAs-ofDsuqxa1haH9UR5L16SY8nP-qqUaic-Imm0D2UUYGXbrL_OKRGnwEFlC1-6iabKba7x3v8oKEb8AkZyJHcYExrFVOJQ7geviQ3eAgON_DNSIL_Mzon2UC6EBRO0xkaZhcJ0Nd7H4ZMUHsnZvsy7ikKRg_GNMXcjaOaJKW2VIRTTtK4ldld-MsttvP6t0-xk1LQFApylK81hekMWSFm3DWDvHhrUV4LHzHac2YioLv2wyG0aAt0JLi8qvf2CDVRArD_hi3aufNRD4aLEZtFpaFJPEcfGFhsctR1wbY_mV9MEJ3MvWA8302oKXwpjgQwgNw_o8ckuSeCCaIRh2a6APULynnmFSMb492o3JimapB3SyqTewFiYbbW5mfehRY06ZyzO2SwVGb1dWwcD3VWK5MaLiaSWQf_XH6AfDF7KyR_shpUD26Bo7JijpKlMRCK_xfES-koRim2aVy7PGKSArsI-AJ8XOHyS0PFl9vJgWXaw9ccPFioiHTxECCTNn-5h1HeYSC0A1vEv_vFKFgrXaiBpXkrns_Qwpg2peR4jayx_HXYtaIiis8CF_Bb4bPjaSowdM7Ry7DCBVOP-Tk4vER2ZuHC7dsDtwX21Bi_kKwzSWMC_qF22JPI-yGmK3qQaekGzuDlqDjeTGrkGBqmbBzwx9lGIF0PCsvFxGCvYNY3C1LCst2rEqJFG4dts7ebd-1o1_neMnp955eJ2cYvMQsRb5gvi4VvfYey3J1QEVDRVVynWcUbQl5hmqWdFPmVDHMWjDWoiJWMq3W8hN6g-uiYVdhJP1hu-SAFZ5sVfLuILIFR-_8xQG9WqN-Z0TjhN9KljZdYNHgwKnb58oGYAG-KP6iE8xZsdr0VQ-7RgboLTLt9QnLskRWSpIuvv2gbjQzZKMQtvaIz5mtYUgK7kYziGeCUveUbmz2fQOVnV-ng7w0FCBI_nXSQ_LG4A-r4EZuxWbYmAWLQWcWQ24vvB268tAPhCTC-8gdY9mJJbKb3my4lPQZfHNergnWDMjXp_ix25abn6BVqHv48v_54Y5SnwrB4sOdnDq1-cxnRBZrFlWYFNrSSWByNq33-GM0Ydu7-sgN77GOUvOmL8EfJ4zGvyxkdWOqYtyyM-yCJelMrmHhlEt_36NKf2l3MHtvf0-g7auAw-UihU06qe0CsQC9J2kqnfTpPPpJoXSqhe7wAVN4DLYpFtY_dr5d7txurR340KxC8gS2qaSCB_axLqIxjx7Q48uVJpyuXCZAhk_nKdvg02udqaGGhpZpp7xJDjzIdjIlHuuMaPRcSbC3ikvSnKPEZbCaZOvfQ1Kr4mSPZER2GeukJY9wJ87p_UdM_ySKWoFEBvEft2vD-qn4lgoAU6ene58cU471CwdAKBGWuOfQD6Q8ipZ_J0bZcMiRp45Y9fLFiR2zvMUnllpbRXwqJwo0lMQaA7aOvm5EXBH-D3hCdF7WBW2A901gpFHJYCyCMhRV_tIHIQeT2bXM4HecO5OYfgDM64kIGqpUwiaOntMc2IMNdx2zfU9QYOKD6hp1iOAka1gGnBYct3QzyyczUGO0Jvhh-IWfWfIJtP2SEHs1kEEwOl29DVqMqE0ntN9HJZdNkC_PKgnYnoWeeUtcsdGk1_O45DPn1trQa7YvxoXDxRIfAcnbE-TJTShWKLG8bwVx-_mciN1kCAQDZH5RRx7Hgk9s-ybHIwPWchJ0XxJPCOnS3d8jBjLUImJJSS0-7krQb-JO7km0VVeqg7j33z8RSeKD6FNYB87zD6hc4H0nAhPhowchdQpuuSZ02zBaFeX5Xw-8jJFG8iKJwEivfYU5MWO64PT2lXX1Nd0F3jlMNmrdh1Kw6f9z7pC0Y5vKXyeT1fyJrKk3pwaZvDmLBMVjRlQT3sHsIlTDj0pojhGZFC0C696DMDitplGlULE1bFwrMyDh2Nyuk316A9_E-T0JGqQcMgLdE3V6YmBxQVxGP5JpnevIhf7zvMj-EFHpGDzpnLmzPanfH06wpOWM7Hl2H_nurcflmf__O8lqZgojNiwOfjfplgU1-nq-KmN7Q7wfOdc6qqXaeap5RRDpaLxHDRIzg2PlMWfUKT7it7galNE4FUp4KSr6JjkkLUPzwqxX9ZS2SdgEUoD6WvrFvwlf-OkQRf7hncT1vVPijwUPLCDMFkCl3hTktiqBcK9-l9PtbFsIXIyNYziPEujlytjxnM5KodsUIzGq-y9OBLHPnxOcPUD9-5ZGAyAmUl8V3PuVCZLBK8R9YJ6XQUROfLhuOfKEJqIMM5Y5TTCmlgGskwRavk6lGhBqRXJuauRPUd-TKxXit7WotTuJBbpebkgZciEyrODr4hthrGmB5SAFRuCzMM1_jT0n4pks9RjGYjY_uyjZndFiTXAIHTptx7L1VkrdM15kmK7BU0a7DsKWP9MONMGGIOTlUXA67Ki_BrF5k-KTCntMI51AKPLO7MC2GXkbguukmNvlp4LcTlTbE6YS22o4bATrgpXKzflYk28zcaBDLEOqCAYQKBN_n1zUa4tFqPB57fNmhhg7q6Rn_kVFWYbw-85blbXnTQhBallw17wJNJXxTeqdpM5PPHmC_GtbUE8cYDWbvJwlwAnUUjtJu49naIinK2-UsVn4d827mABH0meBHJv582LcmRjNUbqiBClZg5ReYfMBnFJKaTTeKeNoGpTgXrU5nUohoqYj0YoqPgWJ5Pz6QdzSNhc_CF2tIt_lQYUGB_Ounwou39HNzfxhKsC9pA4RDyX-ZLadLcfiS6pWCL4WNiA_ipFQSxrQOhO6KJ-AjV19CS776PhjfFpS2in0RRDxzoBrEFnyftVk6pnyhb',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_02bf8d298b645b0100697c354a2ca081a0a0033b0b954a1a3e',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_02bf8d298b645b0100697c354a2ca081a0a0033b0b954a1a3e',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_02bf8d298b645b0100697c354a2ca081a0a0033b0b954a1a3e',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='', id='msg_02bf8d298b645b0100697c3592f2d881a0bbfe8b69634a604e', provider_name='openai'
                    ),
                ],
                usage=RequestUsage(input_tokens=3202, output_tokens=2602, details={'reasoning_tokens': 2432}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_02bf8d298b645b0100697c3521713881a0af562b3dd4ac5065',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_or_text_output(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=str | BinaryImage)

    result = await agent.run('Tell me a two-sentence story about an axolotl.')
    assert result.output == snapshot(IsStr())

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(IsInstance(BinaryImage))


async def test_openai_responses_image_and_text_output(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()])

    result = await agent.run('Tell me a two-sentence story about an axolotl with an illustration.')
    assert result.output == snapshot(IsStr())
    assert result.response.files == snapshot([IsInstance(BinaryImage)])


async def test_openai_responses_image_generation_with_tool_output(allow_model_requests: None, openai_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()], output_type=Animal)

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(Animal(species='axolotl', name='Axie'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0f6cb1c71ef2595f0069814ef35f0c81949a5357c96e5969c7',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0f6cb1c71ef2595f0069814eff87748194b17d4a1a166e9dc4',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_0f6cb1c71ef2595f0069814eff87748194b17d4a1a166e9dc4',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_0f6cb1c71ef2595f0069814eff87748194b17d4a1a166e9dc4',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='', id='msg_0f6cb1c71ef2595f0069814f2b636881949e5e4ce29ed91b29', provider_name='openai'
                    ),
                ],
                usage=RequestUsage(input_tokens=1290, output_tokens=811, details={'reasoning_tokens': 640}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0f6cb1c71ef2595f0069814ef2a36481948886cc292b5432af',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text or call one of these tools: `final_result`.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0f6cb1c71ef2595f0069814f2e81648194be9b1f97469b6c3d',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"species":"axolotl","name":"Axie"}',
                        tool_call_id='call_vRkivIBwMklwcKjvUHRfkBI4',
                        id='fc_0f6cb1c71ef2595f0069814f3a3800819494f57c71530d6a3e',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=547, output_tokens=801, details={'reasoning_tokens': 768}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0f6cb1c71ef2595f0069814f2de22c819487b1361fbdd5c05e',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_vRkivIBwMklwcKjvUHRfkBI4',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_with_native_output(allow_model_requests: None, openai_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()], output_type=NativeOutput(Animal))

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(Animal(species='Ambystoma mexicanum', name='Axolotl'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_04204928974dd26400697c355c39f481a29a7a41202b9844a4',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_04204928974dd26400697c356aad5881a28c7edd638dde83ee',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_04204928974dd26400697c356aad5881a28c7edd638dde83ee',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_04204928974dd26400697c356aad5881a28c7edd638dde83ee',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='{"species":"Ambystoma mexicanum","name":"Axolotl"}',
                        id='msg_04204928974dd26400697c35a0341481a2a4cf5d29479719a0',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=1484, output_tokens=1040, details={'reasoning_tokens': 832}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_04204928974dd26400697c355beaa881a296491fd74e4f4e79',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_with_prompted_output(allow_model_requests: None, openai_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()], output_type=PromptedOutput(Animal))

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(Animal(species='axolotl', name='Axel'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_02a6d1e4583fd25e00697c34f3483481908775dddc8e11636f',
                        signature='gAAAAABpfDUD2DOuEJCw90bVJemvdEV4ltaIAEb4D6rPEmJZJ8ANPSngxjIGfKYMYEkfiPYVC9C1FSLYSq3MnVcQwk4ECmzd7wCpKl9cWxiEoI7ePWZjHKwF568Uj5x5Zf-yyXhsgkyiPj4WVjMW5yOIE_5_a2H5Skft17usT41KJ4gwsTujF7fVM4YlLhqQUxLKH8W7PMH9lyLXamQ7oCZkQV9SkvKNRc9Gb0rq37TcWpLKwuydKIkcgAbrtpyfazWcUQWVOxyv2Zh3GkYyJnWnt-q9IUs9EqsjkOJl7QXFVAchKCDFq5BV7bH4WfaFBVdnrGkKmhqFn-1Kh53_Ho1jvlQaHNL3CZAp-PhJRhEQHiNmeb4mZ9ejAzjwaFrrM05_JLwdJ-0dT3E23VP9Vv_5Scj2X0nTQYTJp9L8kWj1IyB4uDQJ5p6dw2fWPoJlqTTnHByrVKn-dsUkWitUCHlHNpMkTaHpmklNy8FDdYElDa-d72IsRdTJZj_ggskSTirGfWIxV2A6GwC6VLlf9-IMsnSkeomzUarFpDhl8ssPBtRcVz74T2iwzSSZ6dqwmraGgYIapw9BJ31PhRfkGa2TrAJb22TdNznV9vMHOuNpp9Et6syLBjtJRx0FeTCa8uj_Q7btZqpKjAqagQ01CulxBsfyG3niKbhwLFIVWSPoz5WsBXbrJ29W7pn2jl5h0xYd6RamkqaDjLK2b0HCSixArAiegNvY-ShADw6CyREOW2c6ghS8yno0uXkpeRhWcidlJTPhA_ABxDhEbpgjOKiWTIbqqWdkv7diONJ4264C39yyf6tPy2EJrUH49EFpoAqTBI4LhyOAYn02IJttNYceVIZ5MCvPC_TL6qfq5omin4Hxw4lJjVDJU7vIqCrcoDLCV_V8eEhFSXJUqss7cCBLwID7C0EAZTey3omLANy851xm7umVD_h1wu6lhzyXJBmR439LtZWeWJiwgpuIpoUp1tVTojvKRFN1h079KV8wzXMAR_lTH07W-BZK2cpvp2THtibvUp2-rOG5tNBYLdQ3jgQnENiXnANLBp5dhVD-BCTOxY02iY1KfKsNSZq-_fmJ8-WiNDmFGKZK4aw2eLyAQNuZs9vx_GkI0RA88nsNKYb3A0J5PgAK6IOvBJdMPABZpcl0Pt0k3l4TbeJIatd5ZyWSnyHBZHUvNRr2G1GH6-ddMvdYBfAZ4m8SNUGFmsZruXMl8KREiVlZ4p80lQ8L9xHxUzkpglzsUiFmUTAkbJfmx1TmXjSbQIY5DGsJEns9qZcyBEDGbp0uK_FPx7HWcSnlWVRgvXjzoqoqDlFOTdEnA1hkXpA29olTPs38ICuGqHH-lb38JMIA35sQLc0Uq4OZz9I4tdOiNzMBYbOwhG0oDFa_VxFKIsEwwVf54ssiQMVGFfJIIrR_2u_5tJtvPtCLRAjUKycrbfvHXyUlUgYHfwHybuCO44tiimXrb9mRs7P1BsLqkR65LuZnuZBSAX97-cozYSWoBkSlUAtIVj9c_6GJ7ND828bPUbxCsv3oFkO4tCmK-3a2lKocOtxHb3Wob1g3me9MnRcrZcY6VsyNAUVC391kRgUQdF1XUjyncjl2avP06JXJULluOC1oF7bTDBPMdoDVd6xgrQ7gRz6ivNFUN543ImrQjFZWfTo7jvbL4hHMIWkVJBP0eM6pSG_I3hmpaVGyXHyzXK04KAd5QJV7XZrFNuFURnIDGEueKUDi_CPScKXMfYw7NyOdad5vtecKTAjiWl6ReOtVP4vM-wWhO7ctwUafH7ymWw9aHUtjLUf96EHw86q75ZxnLPmLADD1OaJ511XYo9MZEjsoV5PPmAPoYRZtIuLGjCAgvoxB1s3scfEKG6NdlgFEPr2_rGWdU-tRfdZX1U2HdbM3A99Roca_l-NrH5yz75fjCYtMllEGGCtyuNrcfwIU5ASvhrVRRrcKLuDtymX2dEo3m-LWIi9SjdDQUAjvHpAj1kZBa_nM_bklqggJ_v-UiZc7lVMYT8M27FB5Y8z1a2TWbKQwiu58LU9YWP8jJi-ukAyTMiV1X66doaKr29AjUx2F_4wcqzFMWeZuqYJUActfjzU9VfLgt5MT1XC2KUMluslf5-cjSeH3E5J9idQuJ-E_DYuo_vg7euCDX2-Q8aTnV4lLE-vNtXQAy34PWjssFU70MMdn9EBI10BaPNHen1nfuIg4_IOHoCLqYammaA1n7mlIibw7eVKDsgF6ibq4tVve7t8bsNfcp24yG7eZrZL6VQoZJY-p-2sB18cJidBjeJgvYPQ7ed4MQPDpTmvUhAI7_ydc6Wx80rg-ZUvykINtUrb2acrkoTH3_G8pyfiPKTEREnUhqLFvv24PuFpR3spbqlDzYEiwIjfJV9MBLYdCkvGab6lwyIZqPB1e4POSTxqcnf8GxD0ByWv2sCoKrUpu5QnV0EDA28YBj-Q88nyGz3YvwRodW258eXtEulxllXUSSBKqIvgBR51WBC1Rmob1aajtX0ddC0zk1j1siGvT8O2BbHPk1LLuZwOiCsimJON_rYAcYdqIWESQNKa2VOsTeVNJLQbVM2_-GLCRULk883bEWgaTvPG_Bu90E7JJW8u4SN5j-a4Uz_1Pg0deY9D0Ti7T43hf_MZBuPmpN7NaJLTSav-2WqnF0F_JW2tWNq4q2w9UBvUPZj-Rzcv5Yr8TnCQ7cy8p2D-7BvOE4QgPwOJbFfYstEVcMbHxU-4E29PzNLFsbjEv5XvPB_6vIm7xjeS6IwA3QTmyPgBOD9rqBOpksBbp-fMioqi0jJYnVDzMXTZZL7akWC6ekZIabEyjy-G0ETISM77dTS30g2ya85LD9-X2lC9R4dIBdCSHVyUgsxiPdlsaJzK-rfZ9J_Wtw8kdZoiJrHnnO-NR3YdN8V8NtJfYK01hn4GejMf7T4jVrsclDBSiNl84D9dGIIkHCm8ViP2ejuNnUxl0cz9GZiCkNr6j-tFPgbv8hAS6wIdiYyGc_PW4B8i9d6tKGnuOvWeRZa86QMtUFBl_NBIHNl1BuK7k5BNxFRrCqfG9mGBKfTqvcR_UtMtxn36WNmQ3eFnB4F7wDLu4bLxgHK9-Y9iJ_60lUj_jH5RuVA-C9umY9KkoALRDPbY-WsowhsiflsWsKPLSmq1Zqgdi1RLtnhgrCDdbpp-7IzmF-ZUK5_4apO3yGeXGqQLlOt-p-Bygu1XJiIlZzxWcIvAtU5su7WbBUEZeZS-SezPfY3Tc0PqctSalaQHdfkKqhIr5uoV5pPbtYaatxfNl5KcHd6SBz1I3SolteGxWUE7jxNHt-CfVmZE8IpkIkLGlUpg-1BjJhxwgIqndzcqFjkRTlEZ1zHRvi2gIATgBDHVQmZESLByndwUwcW61bf-bwqSea2uOMnR8LUbN4KMAZDeruvbChPxQN4xmTtbDho6KU3fiHoz31rhKo2RBhJ-7puqcapIPkTayAuOCEMdnWlsjjSlWRgRos5vEbqTl_G0v0nSzod6Pn9R-ddPNPKykNSEnkdrWM6JoYjhtcsYjSDeyLcLm1g-xTp4q9mStbJqRPCbO6cfSPvYAcLTP9LESIN5LzuffrcBADS1n5apjMTAMsQ4uMKUAJEN5HdYYXTteXX4nPX8bHPcvoUds4UbWUs3CHJmm7AhN71XgbxOnF8RmYv9pjoYGAZlgWSfyWlWS_53PmlYDOC7qwFfwqQrncyuFDLQmV1ANoizegHrZC4aPmhBZFd5gjGgh908RQ8IM418JDteseWSDbh8rFAAEB6RWAVBIBlxulEubW9TAVI_gC79be3qrpEi4pJmi7VxnaxDCfgidveNEP4xZ2h59ZSu3sotQQ6XOZWKarxA4kJmyihtC591jH202EAkc4Xrq_TboZ-UTsScHP3bwgC69RZTsgzwuuPvEEu-p3CMaoeoDg7v_-rbeYbITG_zUx3HR6Q6GSHpCbLxov7oKlm4cQ-1m1DvNa1y9s3pE1nacyox1Tb2dXhQmWSlobvLPhx9luqJ6-qIhve1n2HaWwKj5R6zxZzR4FVVck1UVFXiFipltMrxGKhxn_x3rRh5SYHmTX6ClFjA5EDmEOokzbMPgsixpUJ9wa5UvWPiibD6zonZ1zS0djMsKcMYQoIJfqWN8bjdrFZLb-bSPi3LlXgZnrcqnGkujZL2jTzWYd1rxMm0w_kjuLodm56mrES0bSyQaKoDl_S-MRwBB8YGFB4tAY5GL9y0CGUWRrtWmrKSmG8Mxny8YnxP25IroxZFFuHEmqqa0dyBDxPUPUxisTisLr9Q3juavgQ8x4J2CIWCzDnUjNu1Re8GrOXHS49l5iEc3a_BV0g71xVcN85pQ54KkKv2TsPhhpYISfTQuR5GQNKLHve0I3wOVAmpIDr4QJacxbUSur__zDyXfpzYy83_3Lgctm5Ia1CKY4OZ4T4gjfuEqxcts63BRbwxcNsqcCzpqtSC0XjIFCH_izYub_cHPMFsK4zwksQZm8qodFpviXgVwZPb301c5qiMPxouYcKx59Wh0fd3OMMx5AdfEdExG2UUF6U0nMf_NcWcuyhh64uZecYY-z-RLkZDYDYGtSxhDDWCZ7MyJniQUarjJccXxiEz7uX4KbV20QDM-Xp_5n21seAEtBNbof9BfzrVAPS0mJNouEVjIMaMFkX3xnPJZlVGTHl2rVfYzaCLA7lOkUBFgBeUQKs9gKBArf5dTrsXgfT0041av1yMOR3ewM9zToewENX7hwulHJUjvipTRBc9nhDs2vsGNyQ_isMXix8DHrptNtdgVh1zbxTOhlRBeN-rbYNsuUf5yTHJEuNrLLi3m7_1Sbq6tc6DQ5s8V7Vcd6-GHAceIrIx-JqW3rnEPosF_L9a_JtEI2Fu9LIs-hIDo0-BW0m1RSNyrBhWE5l5jvASmLDImqFKk0Uaa_-4lR7n36pEzRPggbhBCKs_rucYnymnCPmGM8hGpNIHfMIQEiO3blpZaiw3m0Onss__WQ0TEQWN50zTlj7gctpSiM3Tym0mxQIsp_w4XEWV6dT-fq_RmTeQY8PJsi-9WcIRQDuf3iqzHb0pTjtrVrBnE2V9kNScp5PZVLF5oYlQItsKFkcucYIps6eTuZgKaXukyn7kN4bxLl3nR_8hBe566PFmPxYwQ7EGv6xS419rGIGwL3vjc1oxgoImRD8NafdOzYegCWyD1SLGK-cEXrHi34EzSEOK-FZC73y2OsAcQFS-UAF3-tXB3UPLAas9vm7TDRozpj-qXf_ntfpMP9x7ZBDRKMm8vSFjqPvSwa5fBb9Nft0cGkuvugBdpIrbVUzn1Lk5WwAxzk6JU_usNP_dWkXmY4xYEEZVdSMkV8ldX4epIIO1tzsA_cfTk64FzSjaYObsUkVcgaPMZkD0goVg_v3htpWOGpx7mnt8Sb6h-eweuPm_hCB1sCFFNbYE1QDouFuAF9AKZMjflkVi5I4AqPy3VUa-k1YBSjb1rsN_yW3AlAYAd1k3nLadZ6EH9cKUIpUeCEH4cDqJAzav_qbsw2BtnduipOm1CJS70dvHhO8XbXf5KuFQH0E5oF-4eah1rT8BkUfGfwXAa2LJfeqhDsQN3ss9sdSQIgysIdoGt46auW-CLmllLcg_SfV2X46rPrKXq2lWFcz10_FSYC-QNIHcNABjpAguLeRXSDem3RoSAqIoivBCNPRiY95SYOgBNMPBS0QP9iHpZQodx0yUCEl2Q8ajh1ilw14LHxIUKuUEf5L1FrYkV0EB60TkoQbsFKm1nzj3dftU7Z-HYo0cDpByS7uotGFtMrn4ZihmqdjD48W9gH5hJCSvJQcu4VSgZyzvAkQpqe1nlB2HcogqIzmXztYOdl3Wfd-h-qdqOHHkvzYJLPaIXYRw2pAq7lEe9jjl88XvUc0DqqQV8cgRWDlOAPO-fPV7t7jSDAAWl97c0r6cO66aZYGbPOx6uBiRu8ALwFbI-oWgiOCP1RJuTZ1ARbEVRhMeM1b4KxyJBRXFRwz2dSC94fpfu55a3V4wVad2QXHV9NLmxmi8RMx0yh87KA2IxH0ayemI8Q1t9iDRgCqewkbY9QGMFGq1SRGPCitVwggNQ9kdSQXuyJb2JOrbBkE9d18b2qohMijzL1SDQmzOS9o4Q6dJhqlh5ZRadlAUNBt4cTT-1mF23ddJtL6jncMTIPz-OAbN7cwkho4RI3mYyAITOeJCfmTvQACXEb0imJDd2prHVlEttAHHzmX6SN0qrKauFVKcV4JZtAOZti5m2lEKQ6UBCCs-4KW7Z7R0mGDJI0PwV72YxLkF8S5SCAuwcK8K5GbPaSrie6IdBQIfQgGjWiaARQhhwAllBKCKWmTeT43tBLjayoihHjiwwW1fhhMPjYco0tlN7xSGEhAyS6i655VCNFslcSNsB3UDIjin_L091t2MKF27cbGwG6_22zQDjHi6Kou3v_2R_8TpFSdCZAUicnP4sAEaouWRA9boDav41Wm4YkQQ3U6vU8g1g69Nrqv4G8FD4g8kLoD8r8Q4d50BJDkBgrgwDWgFEcVHclhMZVYNRrB6G4LcpAGZZTcbFq-C3aJ24AE8yHWap-1MbOO90rAQfyw2BBmnuGunuw9xGrRhDF38opKWDgfDvCjnS7x3IXovywB9cC19NSU39G4JGfUDOe8xcQ1PyenpV3293TxcwfoJqAv8cslSw0jC6gv9R94ohSulLmc4Y9wHKjIxVQKoNviqDNSyPYuU6yf2yVLKbqJiX1eoiAhoL6iCueC0wCWeWHYatpWR_ml0aFlqeNW9Uf1AORpsuJD5mLQjbo8LyBAexJhZT5wgUYIsg9ZVgFMtmF7EVg_axU4EgBmN1wog8dDxdAdY5dKIuURKbmDRjTqRo5eHeezP_PZAt1XKlerc2K1ceWOvlB7JnVtBAdeV-POq1id7HC92fPu7biYB9cqQ2in4gzmJdEoiX4V5FqiZvdrePeS5e87IzPXfMXT8zGTXsvwdqgrsqk9fZC7TDenvFYbvts4gMQBCqM7xF2exJK1Fv7HNweKMiMnt3p6gcPLbHECiKjM_JSICFe6Dkoh30cat6um7eBsCB0AFfAJikcNC5l36HoitGcka57qmd7FVs_QX9-d9FKbInC-uqiJCwuo3KEE8g4Ig3jlgqAG-cUNW3yi4mbgH5vL2Tb1rbvaVb-OvllisIXAu2',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='{"species":"axolotl","name":"Axel"}',
                        id='msg_02a6d1e4583fd25e00697c35033a5c8190b26e121930e67c0c',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=434, output_tokens=973, details={'reasoning_tokens': 896}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_02a6d1e4583fd25e00697c34f2ea3881908d505a7a6572977b',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_with_tools(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=BinaryImage)

    @agent.tool_plain
    async def get_animal() -> str:
        return 'axolotl'

    result = await agent.run('Generate an image of the animal returned by the get_animal tool.')
    assert result.output == snapshot(IsInstance(BinaryImage))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of the animal returned by the get_animal tool.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_08dff3c895eb2c4b00697cccd770a48196aebf38f7d1976566',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='get_animal',
                        args='{}',
                        tool_call_id='call_uNor6MBqoGwxTPGsYAOToqR6',
                        id='fc_08dff3c895eb2c4b00697ccce1ef58819688006826fb3922d1',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=386, output_tokens=679, details={'reasoning_tokens': 640}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_08dff3c895eb2c4b00697cccd700f0819682a45c5e71d3989b',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_animal',
                        content='axolotl',
                        tool_call_id='call_uNor6MBqoGwxTPGsYAOToqR6',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_08dff3c895eb2c4b00697ccce2c59c8196b31763695b827a41',
                        signature='gAAAAABpfM00w96NRvWc2DtltCZ9FEAMB-zddVEfFViB_yKJDYDdoHwDO3YSqq-pG9EJJ5pW7z67Ftnbj6RBOXkdEqJCIive-Bl71vwo-g8zSPmIJYa1oHQ_PfbhrjCjD5MLf7sW7V2Wwnvs-1MdZ0DR66SiGjTlebTgMB8naBfgpydgYrpc6KxLQWaLu6VCCwyWE_wHbE05nnLQH4-xBwjFg_rpUdOVwzNNzUbjSrlhrGypZTAktUQosGVjDsk8-HdN5skiVy5DD_RdpyMKSBlUt172vUVR8r4Lktjxcos7fQO4z0hrugU5U_vv6whtGbGR8TfDXDroFV8rEUIlLSB_e_sztpJlDltatZufteC_lQeRYvKFs94STZlWPMvwtAHRKNcFrPMMnDFLFnQnRyzOEQ4HGyBaVLVxceiq28SnTNBGmXsNBhYPUwPeCJCWIOOjJ-2XIf-ODUkWyh9O6cf2durr8e91fwFz11Mr3u_cNHrX-pGBFZpk-hLm9tQihS7TDfrnvIjgknjy8RRgQeUclD5b92Ceb1_71v1HuAETZDSuNsrwXpkwcoQAsUoRRWdkKcRJ1WWxYbONEsiMBW9Zx-PsoRz5fLUtSrgPMi3E7OyXRfi1X6Z5TjfEFI1JLjDwUe7x1KsWIf09FWfZw2n283c5hSEh4kHMku3CjQXWLRVPRkakmRVKqiy7Pi715VV7fte9lXEQfLRHO1Co7cyj4Ke4ed9POcJFT0MGFE2Ve7Kz1X2A4Vwr8t0MDXWdJxS0qXM_2NrPA4OwlB9YYLFTwK3qJlFvmpyelz94bOC7s6bKQwHQLq4kY3DfHgtWIDA3kWMC1HTGESCvpEBLUkTHYjH3vwUMU5QH98XZSjqHe5yJA6q8AAirNvPjHH_XpxGZi2pVmHTidZPRKRy1uvv7NJeKLvBx5UVstscONgaF0aEnutSQ1TaBVOzdzCcJGA_ne5tRB-1z3Rjol9Uujh9S_nVuzmyi8eaVapCNmC4MLGbwcdTpVkUOR3vQ9bBCGVDkdDP6Q8mRNwbNzJwyOztjHhYlNmfQTfUyawJyGATYdtUwjr7oz2gTK68LELAysU-W6rCrYVGYwuVbYfqPWqXku05a8jB-OxiREYP4GqWxL0E8mrN-0m7XufZTEd6VNUzjKt7JTeYvITlNLlvYWreEfVhBp-zO7a4xRnX-uMGHpW9QNLlTuNunRTH6LP6qdWxPL-lkqO7d362nh6-wvv27oWQkibkng0ESYgfWHgqpFA5CYB9nyN3_NHb_7uAV-WYX9yBz_iVL5JlKuM88_0vtHQc7HVUUCU5FDRGSyccuOFnLbyWTYU1Z1ElEzPcN3CzpBOrXlw0eIz7Mon107zyhwA6k49x2WEXPyH2uY6v6pX7mKCpAna4AN3G1mGma70_TyWdoVl42_0cyjXLyQ_OxuRL7H7imbnFmAPMCM99Ntt7ekxFW_5GDivB8KTLzb-_BiHEieTObKA3uOzo_QQCKMl37HOXRrtfmbfrE7p6hTtxs3yKraMBNfcP2NYjrB9K-um2KpxdRQVtRDyTKM-K6gJxExpqc2Nbk6ds3Cg5Tc5-KaweWjXVOZI8OjWLrp3qnbxj3ScK0g_KLkpCUFLiSQSjWKO40I29ursq0YJkUW2FWotc5ZGOR1YEauuUWdH0u3MPjRvLVqDVrTXEaqElgEIetOElH5G8Y9kmnZUBvto4H3TkI_Mhlx9aJOt5dVIEu-KK-4gs2Awj5RbPS6tigfu-khiz23jEdL8fyYFcwk7LreQcS5lcHfirCpxC0rOUuoL8e79-3oAHCcXNV3m8XGXwe-lPMluO4hgntyDISELej0EV-67Py0NEFybgTrJJVh7JMtc2SwPx3BnFZ7SNrhITJg7NciUBWCqh3_XVQJdDdraScQ0yj5k9creDbqCmhtGaS9TlZU43j2hBmAvz5qZ2hS_mORrVDxsfg0_38e9u1IGIGTwKwD0VvjRSLoFiU7AWdjpW4vVYUb5gq8KHQ5JH5ybDfKfCPEX3DBkOqKBmXvK2-URgoZgHXFb7iVe1_G_nyQF4PSOFvxvfIA8rX9UTbLoU6YMn8tTme_UlcRyBdmxp0IuU5b_NGNm42luZylHwYug-7yArWtuF3fgFZymdD27T7AoX0sqD_RVY6fgS9rk5soof795OTTDKNHyNzd7ejVjTNe23EAgWvSaeNiGaZUxa8Ukoa1IpEF6o2AgBQ9kZFWcvy3ztJqCuwE4gZHTl3-Qfeik17IO8_OQbDiNiyR2ZQ5zaWPO4nSX0OcDjWNTsN9NweiTOfO85ndGYL4abD5iHy4HuRGAeAQn_M0OieydJDTMVeZsloycnnNwfmy2y1IDFVayZXPjAUOm4Ptubn0FRlKBbs_7Mdu7VsB2A2e4Zg5pZpHEWICOPKq-ZM7b9ugRlsJpeH4yjgY3C4FIgAQ4RzWbc874A0wV4HLP6nH59FITVzS0f96Zm72AMCqNDLJ0hduN6hiBzo6yiJ-s1_ZheViE4uwaEPFMu4bnwL578FL0i25fi5xQNSD21jEPXmqBAv1n_6Q7gEXfwOXVGVn2y14UB_jGugVx37Fi4w5OaK-QDwvdTDTnFzk-B_2pEmXWlrMxbdQZWWM6-X7W2zPsaIOfOGmUb83xwdbI8uA2Rfr1e-iFemABgxhMnZWQRC7sFdMifW2tlI9_2ag3Orc6dwq2EKJC0MiUMmcOQf3m9LsxV-JNW1Ph7QR-7sqLdJZhr8F4L6vFrsD_l3f1i4SMnhu34R8t16koN2O6IV5iGGDhyjbU_XrL05wQ4HSppSxsG24LiZdMfJTet29MW6Kl2VIGOKHJNTSglmoOp70fHbXXvv0ErJItFN84h70e0WOSj-K8ZkeySwUOHIzDDIy8G0C2QbgZaljX3mBZV1Q1qZg0Evu0R-eJmQdH5pAjCohfAujV5tkkgqsWKzeMTxc4zVePCxrp9V1R9MHZHUgaZ64_SZWeeuOnTP_ybchLSY9nnw6J4007usppEr2ACtTggQPeDxhDqN_yayuBlfU6Cd2jibPHvS1sIvUmqHc5sFsbFxxdThjFa1yPb-UOk5TjVd-ubJrZdOfsqo5-2GCXABPirqL2pMX2S6BLfGSF8SnQz63M5ssIvX4qQYd5rYn8FpFs8U_z6ul6p9K0i11ciebHtkwcU-aEkSQJPco4598jeM0xWTqbZY-bpDVvZQI9rYl0iNEkSocxOYm1c4Ovag60j2v_1zPcxEWrWdYtQCf6wtRH0Njic3NlewgNTJcsvNzN33t8MABNEQaRWTM_BKV-DcSrM1wti8v8pfvgM6HHIuxIERj394-Qq_YbaPEVi7VgaeEb5glo7xuS3iS3DHQqtp0jhcVvAsNoRGaPSegBzUy8QuCM-ngTYyNKm_HaKgvKcTTnx96k3N97JU_qeU9yjrS5eugERkqpHTxJTNQNAdFWkmTrQlI3mPQqU7OnlVMdKsUpxzeOit-dIZHaDXHZnh3OYUHNobq-FMYnerCx_huu2LjeikeRx9vSDmIMMi2E1xv06X7ErusiTZJwcS286_9_TWY_Tu6E-rlIwwZ_7IgJ8Elwg9hQX4CoN-8oQqAwwVIC8OpKaj_GBtUtRzFRoKituTT1Tt613Mt_hQiQ6P1Q2iMRSbPzBKaj6A7Q6-PfwZkkxYIgNzsuwbiCDsaZxrMtC3p4K6fKzJw4qRdMtu8JN3b_ikBVc3vd2grtSDUkf_39mCDRcKx7SQjnuDuNAfY9c9jBkfngbvpkPnLy4186VCXLWJ9LNjEdgV8bTzypS8AHfoCySmBU-7DKjHzPMk7fJAUePXxf_39QtnYw5Om0vy7shmH53cULvvxO7cTP1GKxSFl92lF4xHUe0MjQAInTjC0eKnzVZGjsjKp8IzRUEDUL2UkOEZCgwq1l3N8yUIOWBVUUzk_aH460968wBgpN5aQryM_GjH9FSwWJJL4zsX9uH1o3zB8BoIeMVl8o63OC_0VxyKKSk_2D3xZmM5T_FWTeA84tsQ6nlBewfQaZaLq2aFlpn9lA1Ko3d9GHqxKpdfA44NWz4EZt7nmnqigswev2HQ9Ss6OJvgDe1kxxFy1ovHfH4YNU_i-zfrXN2opXgw_hZtwW39xrYi7dRpdaODh2b4OxvPLEAjOGGAHtj0aEU57tnzpc0BJa1WRli5qTbmqb5cQzHVhwolwZFJvQSOab1T0zmbn6mBnrf2xSPSak6p__snLkQsmgpH6hMlVmVsZ-qsrUQD1D8-z-iK8r28sUQcPIdbQNIHjsa4Y2z5ushcVKZd33WedDnuFFhIJgEI-Dl1feqOY-Zs9Jr7Hv-ZRgnDWetEMZlbdHeE88NJd2hsS0bDoaTqAAeGHGF3EW_d3gXAFU3rqSmEIN03Sw6ycmtGgP_9yfqU6OVQYZlFO4iQcNnbKCtgrOH1BYX38KcdN7VZXzUaDWnyuCu3nKb8qGBeEKpUUt344I-mjpkJnSJJFnF-VSYevBthu3eNQO3PnI1QGU5v7u9HGeMeQw7k9D4pw0Jv7BSYKNRNyKcS-m5XkNsjVPVb3ORNm26QowcmoE4Y8k3FjX94aDdaO8Q0weEwkg43eMGg7t61vfItr_ZXf0ZZvWSX_dz0OliHFr5SQORo9cG1SxdHeeuI6_ae6j4RnbHi5ev4lT3xLaNMpLBT0gdjuBaaBLidxOUpfOGgfsl5yHyM5Fu5ha8Tgp8KrUAaQLLuQ65vOkFO2Zn4KmHjF8okWy-z2zyXrRcHEPcCstXFba5-xycVw8cHUtEIp8hAgFtjgG3YgM99k3pXEumfJxY_d6_QAVGTuscDkQ1skheXX_bhAErXCIh0ctXco1IXP1lpJvjwJFhN2bvF5_adAs8StFUhqnokoK9p8LHEobHLo0-uYnxgL3ucDWNfAQ4Nh45KbBLUPF26SN7-oRzPZKkIKX1HihzC7MAdg6hiqQIGtAj6tVXRscViKIk8kS8iNCxSxQY6YZxtNjFzZ8EwmlNtZo5etGumSPwfAjq9OF3CK0oGuH1hfpGx-R7bhxWFa4mTEj8V7lUsua8K4S7dkBgW7KMsL8vHNxuWKw4Te5JtWqTuOBOQ4R-3U0P0BqDH2-hQhKc5FQKKkjG4H7w5BSzBqBj0iUlBO-eRTaUI7ydQmlFkDiwwFkQ_E2Pzs5ejDRXglg43f2boGq2BdXsgorqvOjc_g6QhxunP7txOsw_G1ANbxFtBH7hpJ5MomaLXtU_kIZLJsE23sshlxEUqNSFhuYI2A8vMuxYRGrMibIZD_BMhp9jKmT647GxYLSMAx3H3TDp_RneZxNyUKG6Uuk-GiB-U1fQid09V-cKUv5cEtRYVReR27gfwuXaI3TkgMoL5O86om-fCLCYCCTBk8WN7x2vBCs-lB5-CcUY5IP1NEh-cmp9oVJW8QlucdNyF5bcIfFay-rZPAYbLNAq7NA5LxgMRuQNmSxyqevuc64eZNjyT_Cbf0N5hkLTsaQUu5pm__bZLxUN_r5YUxnUHe1vRFXcQp1DjQTOXy1iPQWWNEomwi-8a_8H112Lv0ik_qAEB1uVE5T1yYWmlhtGUcSVbTEildXe4hnH-eJspOegWSkre2OHA_NMtWo2yRAOz_kcKomYdCq9Ws6S6_I2cjoAFEhcB7c7PDNfAerohRNsyobv2zMyOrDYjFaHy70oHi_JhemhqWZA2nFEzgNtws-CUzGx9-l1wmXdXWG-Zvq7e_DDk7nTOBQHOo3fU3YYVBca8fgNP8gO9RvFCTQKVjbKn2Wvv6Z3nMNbLYvuVEkFPwufGGlSNoN-3P_nJgdmt6LyulRB4WJ3by998HdDJ2VN4MH6DONNVMA5yljTOzb88vyTs26DSwMMjxzENv0KeoWVGHqJ0kzXndvMwMxcKSM_cHNMWM0F9oOOcyVefzVatcYlyqboBxNGiGttEBIjVhi-VzurskdArPP-vgTQaEtdWHvWCMG7dyfLTAsiAeZo8XyvxYxl2BKonZKHKaw5OAMSsj_6cohhA0aWs-D8BPNzxlOe7_pZQm5dtZX-2gOwoe8I202a9aaS762sQoyvqsfNQbmxS0UAwh30_17AunpnL9fvkDVFULqmZ7tDyJMfu_cLPJNl-Dw69xxd7z3UdiHTtclVC5F4ZxW9diq3V8chQCNwd--E2arSBmLR6z05Vo-ccRQ5mOa6DfzlJXq5PehJueaLkag8089CwKxUkXbWUm6GNo7iIbMD9-tYnowPbqduQ42a5I6kw-QpHbBQJi-ZkXTtK4ICwu8p8i1SZim7uWhggUdo2fwTrB7bhN0NtYvY7-s5u39jfTVFsuj0Iqn7azTlkcxhPqb_9imC4b3VdULHlLAt2yxtOgY36NQb6OdfSwOwsZcKYjZQMHShC3enKkkZpZIcrMzo49KDZA6v9hg26VcMKOQOaYG5QCDAu3CkTxNFkZ1zws8tQCjfCrUUJYhSx7nMLqivXD81eu0YcVIbiZxWCfLGVwgTRdaAZ7x2umSgOwQFrYJKZzss9eJHe0trCSqbpdnQWMVWKumUPI0EDK_VEGYjnXlPSKLx9Zu-mmOQfkcq7-OHYBfTLdoe5Ll01jx-uoXeTSsBIULtgoXUfzMrsh__cHAnvmsWco7_yyy9sz5Z_94hRhbsgLUDIu_UTMzmRnLKJEVvVcXuLt5JSTzI7pChh2t-uJgNwZLfb0hVS164mxt2xvegmg4ZaRujq_SmDQdZAdAgEKEZwpunSWI6HO13No1Cf7IUk70zPhQtxZiZtkMQC792BtjBbmVAxiVSP8h8tJcaexCs5eibmvFMUEPr8o_ajRQpFChZrSRUH61FDW4zSpxmeYnTXMlgYxS9bWAbq1m0mATFjXdwpLhxaGvULQiPmS1x9twy1Et7SG9-20NPGDYYcIpTnduIPGhWpAmFaZ8dX7_0VgjpsMeBdHlEg6WxJd9u946sX1f957S3rMHqRsLKeqMCTmfAiG4h3Fzvppu6qjpRunieBHrI44LZ1t-xhHF1_gc-tos_Y6sg4uMLH41jbKeg6TjCTcE39LqGSDl6eKEccBxu65oJLLzy7FHmFwDmbokWC6BMF2QMfrGkprEiYHGBXLS8bbaC6-T6K2V__xCcAMtFPP02ZTDkkAI9o9NIq1oGQBp7mQSXISgTN-0CmlU_WO4C-aW_2bS_fTatQQqOXrKJeDFp0tNxFtaZoXOCM-tm4cG_z3TXY4UEKOOlv8BRGPZ-98TzQrRt2HHw3P8E2o_0jWGANwbNiLkB7KKzBLo7pUONIGIu4zCqkFV8AqrFpmb0wtIWXPXzGdZVlJ7VAAlEUbIQyw7-EkfRkBkug12Lt-OJ7J_obPZys_0XzdPScisWsEJJNp0m0M5UYnPbwgIURCK3ijsJ0CdHrVmMAczlHYpyJEoGfgR9Azz48PIDsEIMaNZDJr-AVjlhDkhym4ajMkH4tZKoV54bUbfyIxdL4RoCT89clR8CzZ2C9fIrJTkmOjNSW4Rfm76kic54t-3rW3Wjvz3cOLdIDyCmViJRhV5EqakScybYEi__LjfVNdA8dOkQvhW1v8I7faiMy3EMn2MlhS_nK6FVTobErVpaTbxGJMPStblHnkfXo43wlmQNKXSaKBaPYvgYPBcUXeevU-CpN5G7ejULb3-g9ACd0UaMxS3C_VcZNuBVTsQciUtKHrvE6WgffEGb_dGPoz3UhcQIIk7knv2zvyR1f3JXPDiyeRCh36pDmweafc5gQ2X9U3bLIw1k1wj4vTGq3W87oa7tU3DSHlVEAxkM3StD-F8Bx4sOzmEQyB3-qr9rBe8F3_9b52sn8t4CdctH4UL8Sms6YwsLdxBu1XUNSqV9e5rkoHKkB-LICtFDQaICkRdstk_OfU3hctErKPvtB7323avPl5vL32mg9ayoWC-wNf2R7_U1OJVMImpx0sQEi52OyaG4z0vOgamAfBVorhhvzlDWuXPNRQFg1dzqQ6V2c69tGUxSRexkctKgSAUkExQEsTcTHg4EwYrXicEPE_H9f61FrVngvewxpRm88M1Z47XocJbi73P7Wi177mmTRwDHM0SKNoaHYoIq_KZCIGDyjGZ0fVJHj68MDXTjJu2oN_PufV6BzL9GLVWFuBNzbr_KK731T021-cI-3aK4CBSa_SD8fJiOMOy2yMkZnxWwqZRHDttmYBQ8eDh9NmOQomD2g4GeY31xc_0Gfo-XckpAR9QNMn95wjOzjL7jfBBcu3S2uaXcln5KkaxfzlhJyE9gEUHtoffRWxNRpaNz5swboxIp7UwOdopvC_xlVXVwMvmVoYlb88qnDCT1Fzpbvebj-iyfSLzR0xUD44FhGJLqoN4sSSEd29vEjapVGzkn-H2JAd49WId8jhGUB7rE-7IZlYsQzsNzGrq4nryePrMn2sXzT3fRiXBd2-sarJ6sGrWMUx77Cz6NF6PtpF5mfgwqUfVnnxR3eaUUjrqyoZLY89BNUGcFOYj6bYLVjSz8jSIozZ2GdSQ5DwmZf2-HTjYtwBMFksxIsWfgO1Pf7nv81uPyr7XDt3AmVrAJOR40BY5P8yGiBPXuEuWxCMuhar8QIvjcWdI6o52rByAWsJ7vzv_pGboU0sCxX_qONDkpXsWBB-otCQ-QMR3h0ObRa_8_23Z7EGmKIlSuT6ECD71gVLcfzf0q4wHspxbJ1JKuZmnu7roblBWDjO8NhtgYpQrGqqlNfPsaAIyhd0QU1p2i9CSvUUKZdzACfCT4WZbFv0jU5SFB2bMZnOqG1yjJxZjjDldKCIgzrAR53fxSHNEF7_DxOyH1qV6zD7P9DrjRG1je6t-iXj6y4CIHhOXTI7-VYKyE_05zXfkLGFu-5yGiQB6nADId2OjhdPIWyr3kdYn4HWMtWdJECYEGa7wILnkTHgQSSwCJ405jNFdoDY3z8OOWP7XTaXqEiS37PhvTkTaFduoAn36aYhZHOcnYkgFLcJy5GidZnSe6xK2WBeQIUO7PFjRFcb1UUg5y3J56JYVmbQ_gLdW-kybPKEI9Ymj1P71DVgAe-48y-r_SPTree_MBDvpQyqEUnBp2HK1CGnkborg278DGhr21XEx8hs0LlnfJuhEwxYEvaTEndqU2JegEsxhdsM3QYGEzixSPj6PjZbbaFROmraDaydcGLFiNMSfwClEs8jjDJKzybGkatVfCzRr5S6n0fo2P3Ju2xKJzErHoNBb2-BQ6p4w-hDj8ESPEkjjN5hQDdsP5O10B-YrJuV762XKxZyakxqegOwKQtJeJ6ijF4SCZAjKkNX5pygiDxjNt_V3-X7aRZ6HOhhNE2L0X4HGkuRWNBJWDfjTm_c-4mAEK_cQB-Y7O0AzSLrFmhnAkto4eJN_Ep1VRStvxyQ0lbPAZEriOzr69FW22cR7KnEPICqGCN-gXorGAWB-ZUOCFMr8CYsazvfvGhyRZDw6VZghLqw7aK33eRH_ciRjF5Qu3dZQAKr7X21J6_GRjWu4IBINCaK4lWXQaLjrUE8M4moEOW5hWWY_PMqqHCMfJVTgUVJMRVGz_6-htRFb0kqTVWtN49FeFdSrpvJjvpg-IJsansatBUAvdWP9I37d4k_YOILRKeAbTWWG5rDr7D0oBXikm_XpfpxYg1Z7y8KVwxAvKhcbBMv4FLY5-MI97MNh3OFQwyl76e7Nhzl8N_mm7P9TStVJLhJp101CV8YKWUxOl_OWXq8-0GZhq6Qk0ZPHFY4t2ZI5-UNqjeL3g=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_08dff3c895eb2c4b00697cccff89e08196a7f44060878c033b',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_08dff3c895eb2c4b00697cccff89e08196a7f44060878c033b',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': 'Photorealistic axolotl swimming in a clear freshwater aquarium with smooth pebbles and lush green aquatic plants. Pale pink body with feathery external gills, small smiling mouth, dark beady eyes. Soft diffused lighting, shallow depth of field, high detail, 4k resolution.',
                        },
                        tool_call_id='ig_08dff3c895eb2c4b00697cccff89e08196a7f44060878c033b',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='', id='msg_08dff3c895eb2c4b00697ccd33edcc819688e3c4a08b7c2274', provider_name='openai'
                    ),
                ],
                usage=RequestUsage(input_tokens=2645, output_tokens=1485, details={'reasoning_tokens': 1344}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_08dff3c895eb2c4b00697ccce266b0819685c5e9cb0ab2f1ba',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_multiple_images(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=BinaryImage)

    result = await agent.run('Generate two separate images of axolotls.')
    # The first image is used as output
    assert result.output == snapshot(IsInstance(BinaryImage))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate two separate images of axolotls.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0b6169df6e16e9690068dd80d6daec8191ba71651890c0e1e1',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0b6169df6e16e9690068dd80f7b070819189831dcc01b98a2a',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_0b6169df6e16e9690068dd80f7b070819189831dcc01b98a2a',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_0b6169df6e16e9690068dd80f7b070819189831dcc01b98a2a',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0b6169df6e16e9690068dd8125f4448191bac6818b54114209',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_0b6169df6e16e9690068dd8125f4448191bac6818b54114209',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_0b6169df6e16e9690068dd8125f4448191bac6818b54114209',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='', id='msg_0b6169df6e16e9690068dd8163a99c8191ae96a95eaa8e6365', provider_name='openai'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2675,
                    output_tokens=2157,
                    details={'reasoning_tokens': 1984},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 19, 28, 22, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0b6169df6e16e9690068dd80d64aec81919c65f238307673bb',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_jpeg(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool(output_format='jpeg')], output_type=BinaryImage)

    result = await agent.run('Generate an image of axolotl.')

    assert result.output == snapshot(IsInstance(BinaryImage))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_08acbdf1ae54befc0068dd9cee0698819791dc1b2461291dbe',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_08acbdf1ae54befc0068dd9d0347bc8197ad70005495e64e62',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        id='ig_08acbdf1ae54befc0068dd9d0347bc8197ad70005495e64e62',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_08acbdf1ae54befc0068dd9d0347bc8197ad70005495e64e62',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='', id='msg_08acbdf1ae54befc0068dd9d468248819786f55b61db3a9a60', provider_name='openai'
                    ),
                ],
                usage=RequestUsage(input_tokens=1889, output_tokens=1434, details={'reasoning_tokens': 1280}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 21, 28, 13, tzinfo=timezone.utc),
                },
                provider_response_id='resp_08acbdf1ae54befc0068dd9ced226c8197a2e974b29c565407',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_history_with_combined_tool_call_id(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation))

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='What is the largest city in the user country?',
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='get_user_country',
                    args='{}',
                    tool_call_id='call_ZWkVhdUjupo528U9dqgFeRkH|fc_68477f0bb8e4819cba6d781e174d77f8001fd29e2d5573f7',
                )
            ],
            model_name='gpt-4o-2024-08-06',
            provider_name='openai',
            provider_response_id='resp_68477f0b40a8819cb8d55594bc2c232a001fd29e2d5573f7',
            finish_reason='stop',
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='get_user_country',
                    content='Mexico',
                    tool_call_id='call_ZWkVhdUjupo528U9dqgFeRkH|fc_68477f0bb8e4819cba6d781e174d77f8001fd29e2d5573f7',
                )
            ]
        ),
    ]

    result = await agent.run('What is the largest city in the user country?', message_history=messages)
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_001fd29e2d5573f70068ece2e816fc819c82755f049c987ea4',
                        signature='gAAAAABo7OLt_-yMcMz15n_JkwU0selGH2vqiwJDNU86YIjY_jQLXid4usIFjjCppiyOnJjtU_C6e7jUIKnfZRBt1DHVFMGpAVvTBZBVdJhXl0ypGjkAj3Wv_3ecAG9oU3DoUMKrbwEMqL0LaSfNSN1qgCTt-RL2sgeEDgFeiOpX40BWgS8tVMfR4_qBxJcp8KeYvw5niPgwcMF3UPIEjHlaVpglJH2SzZtTOdxeFDfYbnvdWTMvwYFIc0jKOREG_-hZE4AznhHdSLV2-I5nGlxuxqaI4GQCk-Fp8Cvcy15_NYYP62ii50VlR6HPp_gQZEetwgC5pThsiuuG7-n1hGOnsj8gZyjSKsMe2KpzlYzhT7ighmArDVEx8Utvp1FXikqGkEzt4RTqqPInp9kuvqQTSyd8JZ6BEetRl1EuZXT7zXrzLwFN7Vm_gqixmf6mLXZUw6vg6LqGkhSh5fo6C7akPTwwJXjVJ37Dzfejo6RiVKOT-_9sdYCHW2kZ9XfQAmRQfB97UpSZ8QrVfaKy_uRIHLexs8QrQvKuw-uHDQBAL3OEmSTzHzCQ-q7b0FHr514Z29l9etavHNVdpeleWGo6VEtLWGQyblIdIBtf946YnQvr6NYIR8uATn9Z91rr8FsFJTpJh_v5iGA2f8rfPRu27nmw-q8XnPVc_FYCZDk08r_YhdEJZn1INBi8wYSWmpib8VxNpkFO7FFRuK-F8rh3MTpYgIOqPQYbf3LCRvKukTwv1b3mjSKVpHQSm_s6s7djdD-rLuc22-3_MLd0ii4_oOT8w51TQIM61LtonGvxUqf4oKHSUFCVnrWWiT-0ttdpwpJ_iB5frnEeY2mWyU1u7sd38BI3dOzoM82IFaIm98g9fa99bmoA7Z7gI60tzyF8YbJmWF-PCwyKHJ7B1MbCBonO36NmeEM-SplrR54fGykxTmwvtbYGhd5f0cdYzD0zulRDj-AhOd96rrUB_fIgoQGTXey8L_w0whcnVTWdG6is-rx8373Sz8ZRoE5RiLWW1mfHzVXxwslphx4BedRVF0tL-1YO7sg5MXhHCf6hpw8dOht-21NMrb1F1DQadFE_fhySFl-TgOD5BlhAuupLMsqcCIa4lcXP_loyA4ERP6WSdz2Bybz7_1eOiflfVodRrNqvr_DnL0NEXD_JkYTeIn84ziarFV7U7ZnkMvRiA_p1fWdbHTsE_8lu1rsf8fcJ1e76_6ycPkOc4TrOZw8gVRb7gIbMMVrv72BT_sFhW7GkXrzCQpQaeybmRw-bjFhkMMjMDYGXkA_H0q2Zfyh3zCOoa40hl2cqRWp7n1XuafmtKG_F8e9hyWox0q7AhZr5HOOaHz8r3O3-dmNl1KP52bqA8S72rLDslAOQlDupmAQgAmkm5ApYeYcEBredN78jHQ1pviUEI2-3qr4ClXZFHPa54AJ_q4HQ-EcKXEcYQglG21mSUy_tFQF-m4X46Qu8yYWcBVW4E0CG3wbvYx0BCdbc5RhIDkJo1elxLK8XS64lpFkCWy62xLVeMuVuCj8q84-Kk7tZ7gtMtLV9PHQCdbl3s2pAzMfuNIBJog6-HPmwha2n9T0Md5qF7OqCtnYWOWUfIMmQVcdW-ECGsQy9uIUmpsOjdtH31hrX3MUEhIOUB5xErLwfp-_s22ciAY_ap3JlYAiTKGlMCxKxTzK7wWEG_nYhDXC1Afj2z-tgvYhtn9MyDf2v0aIpDM9BoTOLEO-ButzylJ06pJlrJhpdvklvwJxUiuhlwy0bHNilb4Zv4QwnUv3DCrIeKe1ne90vEXe6YlDwSMeWJcz1DZIQBvVcNlN8q2y8Rae3lMWzsvD0YXrcXp02ckYoLSOQZgNYviGYLsgRgPGiIkncjSDt7WWV6td3l-zTrP6MT_hKigmg5F5_F6tS1bKb0jlQBZd0NP-_L_TPqMGRjCYG8johd6VyMiagslDjxG39Dh2wyTI19ZW7h_AOuOpnfkt2armqiq6iGfevA3malqkNakb6mFAS04J9O0butWVAw4yiPCEcLuDNAzzi_qrqLee4gkjh0NplvfGCaE6qqYms61GJbJC4wge6vjyTakurbqWEV3YoR3y_dn-0pjQ7TOx9kkruDwg0nZIV5O6yYxaulmbuvo3fs5CZb9ptZPD0MzGZj7CZU2MDCa4a4gr0McOx2MricxSzIu6emuRUzZuC6C1JxPRC00M0TrZNMIe_WVa9fXDLV1ULEAIMwMXzNT9zV6yiYQCwhkp30Wqde3W0LlIRpSbDuJXcvT8OCbXkdPNIScccdT9LvUQQ--hU2P45kisOev3TYn7yv-pdxM3u1KFNwuFxedSArMBPg7GDz1BOxDQRzv0mfwbf_CcoFbuyj7Tf4zWO46HVdHeRNbvIE--bnaSYD-UFaKknp8ZsBQQhBU_2TEca3fKwmg81-g7Vdb28QUZEuPzgE4ekxZejkKpiKqlLC5nJYgvXrqk2H35D51mYdzPs0ST05Mc41x9MFm_YOLxSFyA0yGAKVINmD5wT6kvRflPkgoksd2ryIvo4KMw3oZQKodv5By0mSJ8iX2vhTGylxiM8wj-ICyNuOsaRFrcMSpX7tZbXcDyysApdmx217BSADoQiNZBLngF7ptxc2QGyo3CwuDjaljwmSgL9KeGthd1RJFd826M287IPpCjLM4WRquCL_E0pQryNqOMn-ZEOCAlBjE37290EhkjKbhiGBEnHUvSbhoH4nL47AmunP_Q5aqh5173VfyoyaybuS3fXjQ5WO0kyFjMdD-a7C6PVdwToCTP-TljoF2YnQKCiqUGs9gNHS9mYhQSXzY4uuGlTHLfKB4JKS5_MQHvwI9zCbTvVG854fPuo_2mzSh-y8TSzBWPokhYWI_q095Sh6tOqDIJNMGyjI2GDFRSyKpKhIFCLyU2JEo9B6l91jPlir0XI8ZOQfBd9J0I4JIqnyoj40_1bF1zUDGc014bdGfxazxwlGph_ysKAP39wV7X9DBFS3ZmeSIn-r3s-sci0HmwnJUb2r03m40rFuNTV1cJMAFP7ZY7PQQQ0TtlO_al0uedaOWylLauap_eoRqc6xGJ2rSz1e7cOevksUlAqzK5xknYKHlsW970xuDGHKOZnKPg8O9nb2PKrcjwEQF5RFPc3l8TtOUXPhhvTERZFGoEuGuSuSp1cJhzba06yPnL-wE3CstYUm3jvkaUme6kKqM4tWBCQDg-_2PYf24xXYlmkIklylskqId826Y3pVVUd7e0vQO0POPeVYU1qwtTp7Ln-MhYEWexxptdNkVQ-kWx63w6HXF6_kefSxaf0UcvL8tOV73u7w_udle9MC_TXgwJZpoW2tSi5HETjQ_i28FAP2iJmclWOm3gP08cMiXvgpTpjzh6meBdvKepnifl_ivPzRnyjz3mYCZH-UJ4LmOHIonv-8arnckhCwHoFIpaIX7eSZyY0JcbBETKImtUwrlTSlbD8l02KDtqw2FJURtEWI5dC1sTS8c2HcyjXyQDA9A25a0M1yIgZyaadODGQ1zoa9xXB',
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city":"Mexico City","country":"Mexico"}',
                        tool_call_id='call_LIXPi261Xx3dGYzlDsOoyHGk',
                        id='fc_001fd29e2d5573f70068ece2ecc140819c97ca83bd4647a717',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=103, output_tokens=409, details={'reasoning_tokens': 384}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 13, 11, 30, 47, tzinfo=timezone.utc),
                },
                provider_response_id='resp_001fd29e2d5573f70068ece2e6dfbc819c96557f0de72802be',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_LIXPi261Xx3dGYzlDsOoyHGk',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_model_mcp_server_tool(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel(
        'o4-mini',
        provider=OpenAIProvider(api_key=openai_api_key),
    )
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
                description='DeepWiki MCP server',
                allowed_tools=['ask_question'],
                headers={'custom-header-key': 'custom-header-value'},
            ),
        ],
    )

    result = await agent.run('Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={'action': 'list_tools'},
                        tool_call_id='mcpl_0083938b3a28070e0068fabd81d51081a09d4b183ced693273',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'tools': [
                                {
                                    'input_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'repoName': {
                                                'type': 'string',
                                                'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                            },
                                            'question': {
                                                'type': 'string',
                                                'description': 'The question to ask about the repository',
                                            },
                                        },
                                        'required': ['repoName', 'question'],
                                        'additionalProperties': False,
                                        '$schema': 'http://json-schema.org/draft-07/schema#',
                                    },
                                    'name': 'ask_question',
                                    'annotations': {'read_only': False},
                                    'description': 'Ask any question about a GitHub repository',
                                }
                            ],
                            'error': None,
                        },
                        tool_call_id='mcpl_0083938b3a28070e0068fabd81d51081a09d4b183ced693273',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0083938b3a28070e0068fabd84727c81a0a52c171d2568a947',
                        signature='gAAAAABo-r2bs6ChS2NtAXH6S8ZWRHzygQvAZrQGsb5ziJKg6dINF9TQnq4llBquiZh-3Ngx2Ha4S-2_TLSbgcsglradULI8c8N2CnilghcqlLE90MXgHWzGfMDbmnRVpTW9iJsOnBn4ferQtNLIsXzfGWq4Ov0Bbvlw_fCm9pQsqOavcJ5Kop2lJ9Xqb__boYMcBCPq3FcNlfC3aia2wZkacS4qKZGqytqQP13EX3q6LwFVnAMIFuwn5XLrh4lFf-S5u8UIw3C6wvVIXEUatY6-awgHHJKXxWUxqRQPJegatMb8KE-QtuKQUfdvEE0ykdHtWqT7nnC3qTY67UaSCCvJ9SdXj-t806GVei9McSUe8riU3viHnfY0R0u9GIXsVnfVthIDRnX7KzpF5ot_CpCrgbCmD9Rj2AAos5pCdSzpc08G5auUuuMZfoiWANADTHHhO2OvflSEpmO8pb-QAYfMoK9exYVQ8Oig-Nj35unupcYy7A2bDCViXzqy32aw9QHmH7rErI4v72beWQxRVdX15Z7VS2c6L1dD7cU18K35CWqlSz9hEX5AcGqEEtIDVu1TdF3m1m2u4ooc4TjYpRecjYoG8Ib-vVKoX5C65a7G1cTbCo8dO0DYKGgM8jM7ZDubxbCcZ22Sxk58f8cer7WxHyp7WRo5-6zvMwMCk8uEY44RJmg-m0Oxl_6qxdr4Md80xZah_6tCCB62agQmYwCrR75_r93xOckQAK0R_37khvQD5gWVlE5Rg-01eUTboiPGqYmIsqWvOkziMGnxgKVw_yUf8swHU1ciWr7O1EdVPHLG7YXlVQTHTE_CX3uOsE2FoZnpS_MgpxGfjb76majV50h7mJ6ySVPF_3NF3RQXx64W08SW4eVFD8JJf0yChqXDmlwu2CDZN1n99xdaE9QbMODNEOmfTQOPhQ9g-4LhstNTKCCxWDh0qiv_dq2qAd0I9Gupoit33xGpb66mndc0nuuNFe8-16iC_KzQtHBNzgasgYK-r83KFVmiYK3Jxvz_2dfdwe0M1q7NLBvbnWc6k9LIf8iDUF6Q1J-cfC7SsncCbROtzIPlKpQwxhP-M09Xy3RVxlH9dcvuk3_qqEAartUQC8ZbuLRbhiq66eE1RvQzdNd2tsoBQ85cdNs57Penio7w9zILUf1JP5O8-zCe5GPC3W3EXTIEvHR-kiuxJvhcsySijpldGmuygRx05ARNOIT7VDCZvF23RfmnRduY1X1FAqb_i_aMStK7iyHr_2ohwOWLuklpyuoG0Y1ulvq1A9-hyCZ0mpvTEF6om2tAZ9_7h8W9ksiOkey0yA-6ze17MCjfnK2XcbqmSMgOngW1PrD81oKoheMnIeJdcWgF2mk8VDqmAwaDTxMxdnXkzK74rA43a4rWk3d2bUts8dAUkuYXTwJwKQw4LfXtu-mwwgJ6BkT_GiBcBJ6ulBuPsNZfpwPuxox6PS6KpzVTQ94cKNqSIIyFCD4xZsEvPALud09-gmAEDHxdnPjqLSi2U8xd0j-6XYKN0JtZ45kwEIRsOrFu-SYLz1OcYFKI5A5P-vYlzGx1WhEnoeUlyooJBhNj6ZBfj9f63SByxm7sgh260vf1t-4OGzVTIUKFluxkI4ubigLZ-g4q4dSwiEWXn50JFPrtuPs5VxsIIz_lXbh1SrKeQ647KdDSAQZFgEfzOOt3el5K97V1x7V7gEWCCgmqDIz3yZPpwD6qmUQKqlj_p8-OQrniamGULkXrmrgbNQVfV-Qw7Hg6ELw4aHF_IZME9Qnyn7peFhH6ai_YapuNF7FK-MBtPYoMaqBf05U2-uJAVUas3VuT_-pTyHvhtFmB7vc0-qgf_CtVNIXSPq2_vXdQdEwwCVPPwW6xWm-invrzhyQR_mf3OQqZT6_zOHIMPBJUaXcQKT0KTdoBZUDamAR-ECZl8r6wdLCn0HjAEwj3ifUCNMzQ7CZHUQG46rj61YyasNWO__4Ef4kTcApKgljosuABqP4HAdmkP5eEnX-6nutrL50iv-Mms_R-T7SKtmEEf9wihTu4Meb441cU9DI4WwSyiBSnsYdGy9FJKmHwP7HD0FmpmWkOrtROkQVMlMVKQFlKK8OBtxafHYsZkWDawbA1eetzMBzQ3PP8PSvva6SJWjbgURHVm5RjXV8Hk6toIBEDx9r9vAIczSp49eDCkQbzPkGAVilO3KLQpNx2itBbZzgE36uV0neZZsVs7aqafI4qCTQOLzYA8YFDKz92yhgdIzl5VPFLFNHqRS4duPRQImQ7vb6yKSxjDThiyQQUTPBX_EXUAAR7JHwJI1i8la3V',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'ask_question',
                            'tool_args': {
                                'repoName': 'pydantic/pydantic-ai',
                                'question': 'Provide a brief summary of the repository, including purpose, main features, and status.',
                            },
                        },
                        tool_call_id='mcp_0083938b3a28070e0068fabd88db5c81a08e56f163bbc6088b',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'output': """\
Pydantic AI is a Python agent framework designed to build production-grade applications using Generative AI, emphasizing an ergonomic developer experience and type-safety . It provides type-safe agents, a model-agnostic design supporting over 15 LLM providers, structured outputs with Pydantic validation, comprehensive observability, and production-ready tooling . The project is structured as a UV workspace monorepo, including core framework components, an evaluation system, a graph execution engine, examples, and a CLI tool .

## Purpose <cite/>

The primary purpose of Pydantic AI is to simplify the development of reliable AI applications by offering a robust framework that integrates type-safety and an intuitive developer experience . It aims to provide a unified approach to interacting with various LLM providers and managing complex agent workflows .

## Main Features <cite/>

### Type-Safe Agents <cite/>
Pydantic AI agents are generic `Agent[Deps, Output]` for compile-time validation, utilizing `RunContext[Deps]` for dependency injection and Pydantic `output_type` for output validation  . This ensures that the inputs and outputs of agents are strictly typed and validated .

### Model-Agnostic Design <cite/>
The framework supports over 15 LLM providers through a unified `Model` interface, allowing developers to switch between different models without significant code changes  . Implementations for providers like OpenAI, Anthropic, and Google are available .

### Structured Outputs <cite/>
Pydantic AI leverages Pydantic for automatic validation and self-correction of structured outputs from LLMs . This is crucial for ensuring data integrity and reliability in AI applications .

### Comprehensive Observability <cite/>
The framework includes comprehensive observability features via OpenTelemetry and native Logfire integration . This allows for tracing agent runs, model requests, tool executions, and monitoring token usage and costs  .

### Production-Ready Tooling <cite/>
Pydantic AI offers an evaluation framework, durable execution capabilities, and protocol integrations .
*   **Tool System**: Tools can be registered using the `@agent.tool` decorator, with automatic JSON schema generation from function signatures and docstrings .
*   **Graph Execution**: The `pydantic_graph.Graph` module provides a graph-based state machine for orchestrating agent execution, using nodes like `UserPromptNode`, `ModelRequestNode`, and `CallToolsNode` .
*   **Evaluation Framework**: The `pydantic-evals` package provides tools for creating datasets, running evaluators (e.g., `ExactMatch`, `LLMEvaluator`), and generating reports .
*   **Integrations**: It integrates with various protocols and environments, including Model Context Protocol (MCP) for external tool servers, AG-UI for interactive frontends, and Temporal/DBOS for durable execution .

## Status <cite/>
The project is actively maintained and considered "Production/Stable"  . It supports Python versions 3.10 through 3.13  . The documentation is built using MkDocs and includes API references and examples  .

## Notes <cite/>
The repository is organized as a monorepo using `uv` for package management  . Key packages include `pydantic-ai-slim` (core framework), `pydantic-evals` (evaluation system), `pydantic-graph` (graph execution engine), `examples` (example applications), and `clai` (CLI tool) .

Wiki pages you might want to explore:
- [Overview (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#1)

View this search on DeepWiki: https://deepwiki.com/search/provide-a-brief-summary-of-the_a5712f6e-e928-4886-bcea-b9b75761aac5
""",
                            'error': None,
                        },
                        tool_call_id='mcp_0083938b3a28070e0068fabd88db5c81a08e56f163bbc6088b',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0083938b3a28070e0068fabd97008081a0ad1b2362bcb153c9',
                        signature='gAAAAABo-r2bD-v0Y3pAlyAEK1Sb8qJJcJRKSRtYwymHwLNXY-SKCqd_Q5RbN0DLCclspuPCAasGLm1WM1Q2Y_3szaEEr_OJalXTVEfRvhCJE1iTgoz2Uyf7KttZ4W92hlYjE8cjgdo5tKtSVkNyzTs4JUHKRHoDMutL2KivjZKuK_4n-lo9paJC_jmz6RWO8wUoXo3_fGxjliOGnWyRXwEPmgAcEWNOSVgCgAEO3vXerXRPLie02HegWcLMtK6WORDHd02Kr86QSK3W30bnvU7glAFX6VhSSnR8G0ceAM-ImoomQ8obEDyedX1-pYDKPOa4pZ5iTjD24ABYOwz-0L7SNziQJLycwwsr11Fj0_Au9yJph8YkNb2nAyFeiNVCRjKul51B7dZgz-UZ9juWO2ffeI0GNtQTYzf46_Y1t0qykGW6w59xjmBHTKf5SiSe0pqWxZ6LOLoPx01rX2gLaKgNZZiERSbO0iwbA4tpxb9ur-qeFVv5tS7xy8KFYOa8SPrypvFWDoY6CjSwTS3ir0vyfpbJy-n6bcYP_pTwDZxy_1aVkciim8Tmm_9wYgI0uY5kcA9VYJuyc4cg7S7ykTUxMZz7xiLMf8FoXl1gHbVJrYriyZzh2poYTWlcCuSCiUaXhQKxcxMRrt_P7WANx0n68ENQ40HkoJ6rThvWUuwtmEYqZ0ldh3XSFtyNrqha4PQ5eg_DudlU_5CxyykuzWmi_o5MEW4_XW4b9vdXg1laqx4189_jEuV_JPGNeL3Ke4EbMbKHzsiaGePRZGgNutnlERagmU4VFTeoE5bN3oHlR_Au4PeQxdb7BuBmZRDDCnnIRd2NfSWb7bgfUozkA4S6rm_089OlRBeRVoLtA8zZZinNGtOZl7MtkLnoJVIWpF1rr7D_47eWSyyegUIIS2e5UKLJfCLkNgSlWPU9VquHEzSfqeHfzoN5ccoVwrvrHmeveTjI-wIJygdfuyti5cMgOOkAtLzjWmbs4CjmlWcbZKeidtDj5YpCSmYAGFuZze-cSbNjMv4th639dCu_jmRMze-l2Y5npbRwMqEJr7VLXghmLc1vhOsaQM3gxoF0CJJlmvtR4jxPqhE3694YRva6LS1WjR4oueM6zfpVeB2kC0hQgqaL6MiwtTRYFfuCzEHi18TwA5bqqkfgrDXedmjAzlEGSZFe2EBRlF_ZtagrVVTCagHQArnH3DkVQMEDCHCqDxA_PINR_997IxeNgGPsvazVdOOBef7sO4rvAWrC94nIlt7d4aViqbTNMW-W8rqjGFOqj1swrM0yoX5y6LY5oXPc3Mu35xeitn_paqtGPkvuH6WeGzAiNZFDoQkUdLkZ4SIH2lr4ZXmMI3nuTzCrwyshwcEu-hhVtGAEQEqVrIn8J75IzYTs1UGLBvhmcpHxCfG04MFNoVf-EPI4SgjNEgV61861TYshxCRrydVhaJmbLqYh8yzLYBHK6oIymv-BrIJ0LX222LwoGbSc0gMTMaudtthlFXrHdnswKf81ubhF7viiD3Y=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0083938b3a28070e0068fabd989bb481a08c61416ab343ef49',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=1207, output_tokens=535, details={'reasoning_tokens': 320}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 23, 23, 42, 57, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0083938b3a28070e0068fabd81970881a0a1195f2cab45bd04',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run('What packages does the repo contain?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What packages does the repo contain?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0083938b3a28070e0068fabd9de42881a08fbb49a65d0f9b06',
                        signature='gAAAAABo-r2izZacxe_jVh_p3URhewxBJuyLNqkJOd0owsDPt9uCE7MXn06WHhO_mp6gLDAqcF1uhMhXCqwztJ1Nbpc0cEDAxUpUCUn2bKSgG6r8Snc_FPtKGgQWDsByvW_Nigx55CyPuNeDO_MiDgYee_WeUw7ASLPfiGOx_9YNc_BFYo1ngsb8CKZcJn3AoponMheLoxkVAPgOjMgteRVaQTr13MljTDUlBIZLIOhVbtIu_dI23saXPigbgwR4RhGn5mCHG_a9ILNkXDJUmGy5TKklIEi2HuJM3ZJ3gfoGYS3OONvzmU4AgMP2UrU17YKZAYKxUBKSpyAqigd4RJSYWzxBCoYzCTmiITwdZ6Cpsw1X9Wox_TQSGt5G2Xu0UY2TQZGRNNH8knJpWs-UQxBBV4L3alMwJuIeV-uzqeKr5fKO5rL_c9as-qQIW_EGQItjvR5z80Hi-S9VXthWCmtqZFIJkgLB5JfTYuFL86valsFVLzSavUIWJAG5qOcxag2mbZMwMRRNfvR__BBtoqBoeGIqveQAbIeZbG0ymw30PH1a2v1mmSrpkK6PB3AHYRDdpkezXLkbyGYgidyV2DAAtPaFplsubWCh_74UxmOuk4BH-9cWkE15mRUBrvtnbTb793RsPzOe7nPmkMpdgqa3nqc6RcQZ_M30lFLUViAbfpEpMVrCzz2cv1RklT1JUzpuVXBTKqQ4FxVCfnvzSgQ2INQ8K50E1X5w_7TAWhrHbNg6LetCa-4KWe9ps0GH6r1x9FWvGyVxSwa7SIdPq3sGpxjOydluPECbBOnHWFUB-3rI2DcUl4rGWYbv2FEFNeCH9Zr67uUvMc4Doi8nVMoeb1lJxFCrfziGhbEXY0FepH3zIzlj-_dXqLAL1qqhfCznT_xkDMVYg-D5gMu-_p3r2SirjJbeaz5UFmP-Dihd9v7jWgD6hx_Mq1uIdzIPE8ImGiDPR7PK64svkvwYg1Czdrc_7GmrKRuzsBL0720UXe19NQqCZfYvUJAjgbEqr3tuS_RkhuEQeeVORn88xkhkrGCEgBS0LHFpe4tcnUEXKnaYYRnoYtk5xo4EyOGVKR2yhF9ht2zrMTo83YuRAPcNT38Jk4gMtVhBaJw_GOfee-IWN_F258rpmU4p8sRV-1iSuQI3Arm4JBU66QuyjoY-KJmTcE9ft3Bfm9If3yG5W0RFRJrsVb--GjHmiiXDGWiR5Q8L1of_RnSD5QDEbXXxhn4dsDejtCXUaQXE9Ty-NvkvA7G6Ru8cMvIKqP2fXS9SmiW6ePJ2Znrlyafxx6L58pT26RF42h90BVrSldf6SjxQApK3AKZW6q8AkuJnYWTtkR9-qfIDl7W94BsgOFoEd-SDQGxWzGJV9YqAu6_SQKiNDQoZZHrJkRSOPEW_b3-BAdrpwL700I92Rye4-BdhlgeK1RwhT3w1Z-z1tvGZXJtPwdpPa3iIw2TIlesMbC1ZJ22iT3CB_r0lnlZhMtIH6o50l50UGfSDuv8HZ_RNgGnYEPqP3FW-o_VD_Yu_KBqGSA0Eb5xAJjl0vpin2vFGO1P4RdgI17eZXRsCp1KvkpWjbEQTWAvJz39yr7wFQ4BrPfgxUqMP0-ZI_h1DkdPBzWs1uKqHw-4qC77sZXgxgHGEIU1tfKosTy_fK4c-WAbdqIHNTh9VdlM1EdrUJQ4rs2rsUG8o9WXwnGTFchI9Ao64LiCFTFTiFL_dvKI4ZraNNXXprfPhxsdLBaNfgj2CIfUwBMJ9xMGmHKQKLtwZdHpQNVqi8DNm1qjvs3CxbSXGKtkl5K8UhJtI1g4OnEnbq3jDO8DGIyDl0NH-0bcCDqS2yAkh8I3IobzxTg16mqU3roXLQ4pGXnWbx26A_9zb4Y1jV7rzCq24VIfNJzMUtW4fVMYzlrp3X1l32I5hF3YP-tU2paD98xobgc2Cn2RWXd3OirrdjKAE088KhXYLZZY59y4LYRLC6MDMHSX0cbEXbBvl6mKmbaFig2_7ICiSa7rR_Ij6PpQRxIW7NfS7ZMu5w7TnhLJyg5nuwMI8A5pVxfy3gYg2L60wepuX7UUV0USaHNKi8qxbp4RJj4nO-GdE8TbLJtvPw-OzrH9Qiv7iDHVMHOe1CDPLD5IeGqmVB0tuLqlyASuIe3oPxTU7QdctyxHa1z-sO8nN6kpPnzmVmS6XK8bY-h5do28dkZvefomSquXwKeiVg9VAMWVziKLPWWg5iWp2x-spLkWcQsQle2T7xizyETaF1t6YbecXtSoVFmu90_o6ns07etU3RVK1YpQLgqUIJwwF3ZwP65MaWPwqDuWCuoQErlApdhRptxId67KE3UC4j8cAaGSoG0kXnws-jzpPyAg1GU8c-Gu_K0F-h-KFbHPMiWCrrQqzVfvoA2wLaQz3NPAqpq-kbFmrXRGkzLIeIvRVxck-sKkxQIcg3amSV5Dykl-lRCXGxlWNiFG_1SFrTSfp5VKyg7l1KjJzXUXHtqAErsPtMyhxaMmlh4An5a8NIaM9W6tafJrBXpUh85DfwZ8W92OAi1WOgoJIwWXSSeSuo6ECDstjVWW3OQQh9183jliwS7Bis3eu9jgAF3q8sYILBdwjrJRa6aAna2GirNwqZMEIg60kIlvmf1U6S2PgYaPm9UDzvMxjpzwjhXhzxHJitfU1tfl0vo-ATaTV8CxmKerNzy2AjlIZnjknG3xLyonCHbGbAe33QQTclb98y_vr5nA4WKlrls413o0a0f8GL8GjINCOd1RHVMjV',
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
The monorepo is organized into these main packages:  \n\

• pydantic-ai-slim\u2003– core agent framework (type-safe agents, model interface, tooling)  \n\
• pydantic-evals\u2003\u2003– evaluation system (datasets, metrics, evaluators, reports)  \n\
• pydantic-graph\u2003\u2003– graph-based execution engine (state-machine orchestration)  \n\
• clai\u2003\u2003\u2003\u2003\u2003\u2003\u2003– CLI for scaffolding and running agents  \n\
• examples\u2003\u2003\u2003\u2003– sample apps & demos showing real-world usage\
""",
                        id='msg_0083938b3a28070e0068fabda04de881a089010e6710637ab3',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=1109, output_tokens=444, details={'reasoning_tokens': 320}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='resp_0083938b3a28070e0068fabd9d414881a089cf24784f80e021',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_model_mcp_server_tool_stream(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('o4-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
                allowed_tools=['ask_question', 'read_wiki_structure'],
            ),
        ],
    )

    event_parts: list[Any] = []

    async with agent.iter(
        user_prompt='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        if (
                            isinstance(event, PartStartEvent)
                            and isinstance(event.part, BuiltinToolCallPart | BuiltinToolReturnPart)
                        ) or (isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta)):
                            event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={'action': 'list_tools'},
                        tool_call_id='mcpl_00b9cc7a23d047270068faa0e29804819fb060cec0408ffbcd',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'tools': [
                                {
                                    'input_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'repoName': {
                                                'type': 'string',
                                                'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                            }
                                        },
                                        'required': ['repoName'],
                                        'additionalProperties': False,
                                        '$schema': 'http://json-schema.org/draft-07/schema#',
                                    },
                                    'name': 'read_wiki_structure',
                                    'annotations': {'read_only': False},
                                    'description': 'Get a list of documentation topics for a GitHub repository',
                                },
                                {
                                    'input_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'repoName': {
                                                'type': 'string',
                                                'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                            },
                                            'question': {
                                                'type': 'string',
                                                'description': 'The question to ask about the repository',
                                            },
                                        },
                                        'required': ['repoName', 'question'],
                                        'additionalProperties': False,
                                        '$schema': 'http://json-schema.org/draft-07/schema#',
                                    },
                                    'name': 'ask_question',
                                    'annotations': {'read_only': False},
                                    'description': 'Ask any question about a GitHub repository',
                                },
                            ],
                            'error': None,
                        },
                        tool_call_id='mcpl_00b9cc7a23d047270068faa0e29804819fb060cec0408ffbcd',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_00b9cc7a23d047270068faa0e4cd5c819f8855c183ff0fe957',
                        signature='gAAAAABo-qDma-ZMjX6meVDoCLYMqgkbQoEVzx_VFnmBFRLqsq37MiF7LP1HrMpqXqtrZ0R2Knb6lUiGSKhsOjOUAn9IFNUCuJx23cPLObF2CKt86wGLb7vccbCrp8bx-I6-kUtZASjlJx7_eJnvwyr24FLZlaDyGDuqRecGA8H4tXnQSAQTT9fJqy8h8dXvxvYzNj5rgOUWgRGn1NBph164KpiEzVWHADzZ_K0l4fX-DFHgtNFssPDYqOKLs_nU0XO8xaIZOgJ8QTf0XmHYF02GA_KciV6sIlSzVricQkwmu1XfJbjpME8XmRMIzlnLRqC8SAJs2kiaYnA8ObfI-s0RbRd3ztIUrzmAsdeo13ualD3tqC1w1_H6S5F47BB47IufTTbpwe_P6f5dLGpOzcrDPbtfHXv-aAW5YEsGyusXqxk51Wp7EONtADmPmVLJffFbRgnwfvPslbxxpNGfxNkN2pIs3U1FW7g1VvmxUfrF84LJpPKvs3xOaWXGorrPBY5nUyeRckhDFt6hGdS59VICmVy8lT4dL_LNswq7dVRS74HrrkfraXDDm2EhL2rtkwhiMqZtuYFsyIK2ys0lZuhNAkhtfgIoV8IwY6O4Y7iXbODxXUr48oZyvLdgV2J2TCcyqIbWClh3-q8MXMmP5wUJdrqajJ8lMVyhQt0UtMJKyk6EWY1DayGpSEW6t8vkqmuYdhyXQOstluONd31LqnEq58Sh8aHCzrypjcLfjDRo5Om1RlxIa-y8S-6rEIXahcJCX_juSg8uYHzDNJffYdBbcLSVQ5mAVl6OM9hE8gHs7SYqw-k-MCeoYsZwt3MqSV7piAu91SMZqB0gXrRDD67bdhmcLBYKmZYKNmLce60WkLH0eZMPSls-n2yyvmwflJA---IZQZOvYXpNUuS7FgMrh3c7n9oDVp15bUgJ8jDx6Mok4pq9E-MHxboblGUpMlFCJDH3NK_7_iHetcqC6Mp2Vc5KJ0OMpDFhCfT3Bvohsee5dUYZezxAkM67qg0BUFyQykulYLHoayemGxzi1YhiX1Of_PEfijmwV2qkUJodq5-LeBVIv8Nj0WgRO-1Y_QW3AWNfQ80Iy6AVa8j9YfsvQU1vwwE9qiAhzSIEeN1Pm2ub8PaRhVIFRgyMOLPVW7cDoNN8ibcOpX-k9p_SfKA9WSzSXuorAs80CTC9OwJibfcPzFVugnnBjBENExTQRfn4l7nWq-tUQNrT4UNGx-xdNeiSeEFCNZlH50Vr5dMaz5sjQQEw_lcTrvxKAV5Zs1mtDf6Kf29LkqhuUEdlMLEJwnAdz2IHLIy41zWLQctSnzBl9HB3mkw8eHZ1LdaRBQRFH4o7Rumhb3D1HdIqDLWeE3jkA6ZBAh2KadGx1u3AIIh4g3dHUS6UREkmzyRIuImbdTsoin1DrQbuYbaqZwIqU4TTIEmA8VeohMfff0rIL5yyFy7cfgGYurgAyMhARPGAAMAoTrR8ldWwymzPkGOJ_SQlzfNGV8weHOEYUl2BgQe57EDX4n1Uk294GIbvGR7eLRL_TLBUyHQErCaOCi8TkBNlLXIobw4ScN_jqqtURmC0mjRDVZeBi6hfrVShWChpQR8A2HxxHrcuHi2hi_2akgUea3zz6_zbUYVoIRdOa9DvZuN015E8ZSL-v_1_vOzUGvt0MuWPazjiRDWgpgcISYzT8N-Xzu_EbwO1OsaOFIeUqrD8mZ6MKOuBQts68og0DWo8KQaHmCaWi4O-c8-5fbB2q3H6oiIoZtSJIoowAmFGOwyWxn_OPS9svDgEaeFYEYhXZ5wZDphxoHkjJ703opxrWoEfQw==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args='{"action":"call_tool","tool_name":"ask_question","tool_args":{"repoName":"pydantic/pydantic-ai","question":"What is the pydantic/pydantic-ai repository about?"}}',
                        tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'error': None,
                            'output': """\
The `pydantic/pydantic-ai` repository is a Python agent framework designed to simplify the development of production-grade applications using Generative AI . It aims to bring the ergonomic developer experience and type-safety philosophy of Pydantic and FastAPI to AI agent development .

## Core Purpose and Features

The framework focuses on providing a robust and type-safe environment for building AI agents . Key features include:

*   **Type-safe Agents**: Agents are generic `Agent[Deps, Output]` for compile-time validation, leveraging Pydantic for output validation and dependency injection .
*   **Model-agnostic Design**: It supports over 15 LLM providers through a unified `Model` interface, allowing for easy switching between different models and providers  .
*   **Structured Outputs**: Automatic Pydantic validation and reflection/self-correction ensure structured and reliable outputs from LLMs .
*   **Comprehensive Observability**: Integration with OpenTelemetry and native Logfire provides real-time debugging, performance monitoring, and cost tracking  .
*   **Production-ready Tooling**: This includes an evaluation framework (`pydantic-evals`), durable execution capabilities, and various protocol integrations like MCP, A2A, and AG-UI  .
*   **Graph Support**: It provides a way to define graphs using type hints for complex applications .

## Framework Architecture

The framework is structured as a UV workspace monorepo, containing several packages .

### Core Packages

*   `pydantic-ai-slim`: Contains the core framework components such as `Agent`, `Model`, and tools .
*   `pydantic-ai`: A meta-package that includes all optional extras .

### Supporting Packages

*   `pydantic-graph`: Provides the graph execution engine with `Graph` and `BaseNode` .
*   `pydantic-evals`: An evaluation framework for datasets and evaluators .
*   `examples`: Contains example applications .
*   `clai`: Provides a CLI interface .

## Agent Execution Flow

The `Agent` class serves as the primary orchestrator . Agent execution is graph-based, utilizing a state machine from `pydantic_graph.Graph` . The execution involves three core node types:

*   `UserPromptNode`: Processes user input and creates initial `ModelRequest` .
*   `ModelRequestNode`: Calls `model.request()` or `model.request_stream()` and handles retries .
*   `CallToolsNode`: Executes tool functions via `RunContext[Deps]` .

The `Agent` provides methods like `run()`, `run_sync()`, and `run_stream()` for different execution scenarios .

## Model Provider Support

The framework offers a unified `Model` abstract base class for various LLM providers . This includes native support for providers like OpenAI, Anthropic, Google, Groq, Mistral, Cohere, and Bedrock . Additionally, many OpenAI-compatible providers can be used with `OpenAIChatModel` .

## Tool System

Tools are registered using the `@agent.tool` decorator . The system automatically generates JSON schemas from function signatures and docstrings, validates tool call arguments, and provides context injection via `RunContext[Deps]` .

## Observability Integration

Pydantic AI integrates with OpenTelemetry, allowing for instrumentation of agent runs, model requests, and tool executions . It has native integration with Pydantic Logfire for enhanced monitoring and visualization .

## Evaluation Framework

The `pydantic-evals` package provides a framework for systematically testing and evaluating AI systems . It supports defining datasets with `Case` objects and using various evaluators, including built-in and custom ones .

## Integration Ecosystem

Pydantic AI supports various integrations for development and production:

*   **Model Context Protocol (MCP)**: For external tool server access .
*   **AG-UI Protocol**: For interactive application frontends .
*   **Agent2Agent (A2A)**: For multi-agent communication and workflows .
*   **Temporal**: For durable workflow execution .
*   **DBOS**: For database-backed execution and state persistence .

## Notes

The `CLAUDE.md` file provides guidance for Claude Code when working with the repository, including development commands and an overview of core components and design patterns . The `mkdocs.yml` file defines the structure and content of the project's documentation, including navigation, plugins, and watch directories for various packages  . The `docs/install.md` file details how to install the `pydantic-ai` package and its optional components, including a "slim" installation option for specific model dependencies .

Wiki pages you might want to explore:
- [Overview (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#1)

View this search on DeepWiki: https://deepwiki.com/search/what-is-the-pydanticpydanticai_e234e9cf-d4aa-4c67-a257-56034816dd56
""",
                        },
                        tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_00b9cc7a23d047270068faa0f4ff54819f9fb9ff25bebe7f5f',
                        signature='gAAAAABo-qD2WTMmhASwWVtFPlo7ILZP_OxHfRvHhda5gZeKL20cUyt0Np6wAHsJ6pyAsXCkLlKBVz3Vwm52JrJuUbqmw-zlXL19rbpvTPRMkiv_GdSfvmxKKNJvSm417OznBDVjsIAqmes2bMq03nRf6Pq2C0oUJnIbpbMwtWzs3jMQqUb0IwyopqXGhn3MWKctLPKZS89nyL4E9kJAx_TyWTQvME8bf8UrV8y2yrNz9odjSQQyZq5YXrlHzpOJjDTfLofVFjsEzM8J29SdLcWnqlv4djJ8xeMpP2ByXuHRnTEyNNuxpYJB7uQbYT0T_eLhwcLv2ZzDZ_hf2Msv7ZdyuPc7Yxc5YWlChB0iaHqQ_8UuMjIVurfgSIjSq2lTvJwdaA365-ZoBMpo4mG04jQDP3XM-0xEM6JTFWc4jZ1OjIXVpkjaXxdOOkYq3t3j8cqBQH69shFCEQr5tnM8jOEl3WHnkvaBg4xEMcd61hiLOKnWbQiYisbFucA8z5ZNbdohUZd-4ww0R8kSjIE5veiyT66gpIte0ItUnTyhIWy8SZYF9bnZGeS-2InDhv5UgjF2iXzgl6dmUrS-_ITgJkwu4Rdf9SBDJhji3_GUO9Za0sBKW8WohP142qY0Tbq4I6-7W1wJ3_gHJqiXVwDLcY90ODSyyC5_I3MgaALRC1wt55sHSeSsDjmNGmiH-m0snaqsI0JnAZwycnWCK17NamjQ9SxVM5tTqJgemkGFQNH1XhZPWvVj56mlj74KKbCJALQpdXD27C8LfdrlBd0v_zEmF1dh7e12I95fYeAlO51xOglBaMCgcMWSDHMGHsJBbJ04eVQSwYTl72rmkASTMaybD-aAm1m8qZnKU-f3xQradhs9l1x9eOfQDIsfWMr1aVMiZi59--VsrgYCbqBj7AGf8n6VNbQWkhO2etozwYZcdGIyiu4TaULX1Xp89Gb28M-tVkIrkQoHO_Z7wzKU1HRBViES1wRKUJ-Sa6wc8UP5orDxeOTFPUr7JL-qaj49cpKzvdlfuoIdbYwpsNvAg69sNbFI3w4jLxOT4yxS6thra1Bit6SY5wAEfrrjtzofLeg49aFqFVGIHeJ8kE3spc1rctpETkdHNyP9fEjZaM3mxR4yz0tPmEgUsd-sdw5BbOKDAVzwconmbeGBmf9KLXMEpRRH7-qSIWUscCi5qIdHXGYoQkStsNGrnhucn_hwqZCSti3Kbzfosud3zQPjW6NyuJCdeTxbDbsnrV7Lkge5j92pyxCHw9j0iuzofRW55_KToBtIvRoPr_37G_6d6TxK42mKqdbgk9GHrcXf27mXszCEzX-VfRVTxyc6JLfEy1iikdo-J2AzXPd4m3zE-zazBU3Z5ey596g8gxwXMkHakLrvwp4_-fQfcvs7sIH34xkEhz7BRdNok3Aqbu_zCt2np69jjHqfPQWZzAy1C-bmMuhAaItPYkkw-LgSu-YP6L89zNofK9Q_S3JwVsLN-fq-9OwhSjy_rQu22Gn4KD6saAu61QMXBPa6z0QJSFUZHJQ_megq1tENfB6wRVtQ0DdAvUwhUsMwx6yE9CT20bma4CloGW__aZuD9gikdQrQ1DCHOvTrfEpvHkl6-wuCImeNjsCvbRFAkx6Xgpc6fdbq4j6WyEVW_4VePNknFWYZ1cw795ka5uJMLc3hVughVlGwDbw60Q3utsjHPbu03pxPle5pdcVEYSQWa0WbFDCrF4ysK0lpmlF7',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_00b9cc7a23d047270068faa0f63798819f83c5348ca838d252',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=1401, output_tokens=480, details={'reasoning_tokens': 256}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 23, 21, 40, 50, tzinfo=timezone.utc),
                },
                provider_response_id='resp_00b9cc7a23d047270068faa0e25934819f9c3bfdec80065bc4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:deepwiki',
                    args={'action': 'list_tools'},
                    tool_call_id='mcpl_00b9cc7a23d047270068faa0e29804819fb060cec0408ffbcd',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:deepwiki',
                    content={
                        'tools': [
                            {
                                'input_schema': {
                                    'type': 'object',
                                    'properties': {
                                        'repoName': {
                                            'type': 'string',
                                            'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                        }
                                    },
                                    'required': ['repoName'],
                                    'additionalProperties': False,
                                    '$schema': 'http://json-schema.org/draft-07/schema#',
                                },
                                'name': 'read_wiki_structure',
                                'annotations': {'read_only': False},
                                'description': 'Get a list of documentation topics for a GitHub repository',
                            },
                            {
                                'input_schema': {
                                    'type': 'object',
                                    'properties': {
                                        'repoName': {
                                            'type': 'string',
                                            'description': 'GitHub repository: owner/repo (e.g. "facebook/react")',
                                        },
                                        'question': {
                                            'type': 'string',
                                            'description': 'The question to ask about the repository',
                                        },
                                    },
                                    'required': ['repoName', 'question'],
                                    'additionalProperties': False,
                                    '$schema': 'http://json-schema.org/draft-07/schema#',
                                },
                                'name': 'ask_question',
                                'annotations': {'read_only': False},
                                'description': 'Ask any question about a GitHub repository',
                            },
                        ],
                        'error': None,
                    },
                    tool_call_id='mcpl_00b9cc7a23d047270068faa0e29804819fb060cec0408ffbcd',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:deepwiki',
                    tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='{"action":"call_tool","tool_name":"ask_question","tool_args":',
                    tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='{"repoName":"pydantic/pydantic-ai","question":"What is the pydantic/pydantic-ai repository about?"}',
                    tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='}', tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac'
                ),
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:deepwiki',
                    content={
                        'error': None,
                        'output': """\
The `pydantic/pydantic-ai` repository is a Python agent framework designed to simplify the development of production-grade applications using Generative AI . It aims to bring the ergonomic developer experience and type-safety philosophy of Pydantic and FastAPI to AI agent development .

## Core Purpose and Features

The framework focuses on providing a robust and type-safe environment for building AI agents . Key features include:

*   **Type-safe Agents**: Agents are generic `Agent[Deps, Output]` for compile-time validation, leveraging Pydantic for output validation and dependency injection .
*   **Model-agnostic Design**: It supports over 15 LLM providers through a unified `Model` interface, allowing for easy switching between different models and providers  .
*   **Structured Outputs**: Automatic Pydantic validation and reflection/self-correction ensure structured and reliable outputs from LLMs .
*   **Comprehensive Observability**: Integration with OpenTelemetry and native Logfire provides real-time debugging, performance monitoring, and cost tracking  .
*   **Production-ready Tooling**: This includes an evaluation framework (`pydantic-evals`), durable execution capabilities, and various protocol integrations like MCP, A2A, and AG-UI  .
*   **Graph Support**: It provides a way to define graphs using type hints for complex applications .

## Framework Architecture

The framework is structured as a UV workspace monorepo, containing several packages .

### Core Packages

*   `pydantic-ai-slim`: Contains the core framework components such as `Agent`, `Model`, and tools .
*   `pydantic-ai`: A meta-package that includes all optional extras .

### Supporting Packages

*   `pydantic-graph`: Provides the graph execution engine with `Graph` and `BaseNode` .
*   `pydantic-evals`: An evaluation framework for datasets and evaluators .
*   `examples`: Contains example applications .
*   `clai`: Provides a CLI interface .

## Agent Execution Flow

The `Agent` class serves as the primary orchestrator . Agent execution is graph-based, utilizing a state machine from `pydantic_graph.Graph` . The execution involves three core node types:

*   `UserPromptNode`: Processes user input and creates initial `ModelRequest` .
*   `ModelRequestNode`: Calls `model.request()` or `model.request_stream()` and handles retries .
*   `CallToolsNode`: Executes tool functions via `RunContext[Deps]` .

The `Agent` provides methods like `run()`, `run_sync()`, and `run_stream()` for different execution scenarios .

## Model Provider Support

The framework offers a unified `Model` abstract base class for various LLM providers . This includes native support for providers like OpenAI, Anthropic, Google, Groq, Mistral, Cohere, and Bedrock . Additionally, many OpenAI-compatible providers can be used with `OpenAIChatModel` .

## Tool System

Tools are registered using the `@agent.tool` decorator . The system automatically generates JSON schemas from function signatures and docstrings, validates tool call arguments, and provides context injection via `RunContext[Deps]` .

## Observability Integration

Pydantic AI integrates with OpenTelemetry, allowing for instrumentation of agent runs, model requests, and tool executions . It has native integration with Pydantic Logfire for enhanced monitoring and visualization .

## Evaluation Framework

The `pydantic-evals` package provides a framework for systematically testing and evaluating AI systems . It supports defining datasets with `Case` objects and using various evaluators, including built-in and custom ones .

## Integration Ecosystem

Pydantic AI supports various integrations for development and production:

*   **Model Context Protocol (MCP)**: For external tool server access .
*   **AG-UI Protocol**: For interactive application frontends .
*   **Agent2Agent (A2A)**: For multi-agent communication and workflows .
*   **Temporal**: For durable workflow execution .
*   **DBOS**: For database-backed execution and state persistence .

## Notes

The `CLAUDE.md` file provides guidance for Claude Code when working with the repository, including development commands and an overview of core components and design patterns . The `mkdocs.yml` file defines the structure and content of the project's documentation, including navigation, plugins, and watch directories for various packages  . The `docs/install.md` file details how to install the `pydantic-ai` package and its optional components, including a "slim" installation option for specific model dependencies .

Wiki pages you might want to explore:
- [Overview (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#1)

View this search on DeepWiki: https://deepwiki.com/search/what-is-the-pydanticpydanticai_e234e9cf-d4aa-4c67-a257-56034816dd56
""",
                    },
                    tool_call_id='mcp_00b9cc7a23d047270068faa0e67fb0819fa9e21302c398e9ac',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
        ]
    )


async def test_openai_responses_model_mcp_server_tool_with_connector(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel(
        'o4-mini',
        provider=OpenAIProvider(api_key=openai_api_key),
    )
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='google_calendar',
                url='x-openai-connector:connector_googlecalendar',
                authorization_token='fake',
                description='Google Calendar',
                allowed_tools=['search_events'],
            ),
        ],
    )

    result = await agent.run('What do I have on my Google Calendar for today?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='What do I have on my Google Calendar for today?', timestamp=IsDatetime())
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:google_calendar',
                        args={'action': 'list_tools'},
                        tool_call_id='mcpl_0558010cf1416a490068faa0f9679481a082dc4ac08889f104',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:google_calendar',
                        content={
                            'tools': [
                                {
                                    'input_schema': {
                                        'properties': {
                                            'calendar_id': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'description': "The ID of the calendar to search. Default one is 'primary'",
                                                'title': 'Calendar Id',
                                            },
                                            'max_results': {'default': 50, 'title': 'Max Results', 'type': 'integer'},
                                            'next_page_token': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'title': 'Next Page Token',
                                            },
                                            'query': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'title': 'Query',
                                            },
                                            'time_max': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'description': "Time in the ISO-8601 format. You can also use 'now' or leave null.",
                                                'title': 'Time Max',
                                            },
                                            'time_min': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'description': "Time in the ISO-8601 format. You can also use 'now' or leave null.",
                                                'title': 'Time Min',
                                            },
                                            'timezone_str': {
                                                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                                'default': None,
                                                'description': "Timezone of the event. Default is 'America/Los_Angeles'",
                                                'title': 'Timezone Str',
                                            },
                                        },
                                        'title': 'search_events_input',
                                        'type': 'object',
                                    },
                                    'name': 'search_events',
                                    'annotations': {'read_only': True},
                                    'description': 'Look up Google Calendar events using various filters.',
                                }
                            ],
                            'error': None,
                        },
                        tool_call_id='mcpl_0558010cf1416a490068faa0f9679481a082dc4ac08889f104',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0558010cf1416a490068faa0fb684081a0a0b70f55d8194bb5',
                        signature='gAAAAABo-qEE669V-_c3vkQAeRtSj9pi72OLJweRJe4IRZkLcFfnuwdxSeJM5DVDLzb3LbfzU0ee6a4KAae0XsETU3hELT1hn3LZPwfFku5zl7CVgsc1DmYBf41Qki1EPHFyIlMj937K8TbppAAqMknfLHHwV1FLb8TapccSEhJbzGutqD3c2519P9f6XHKcuDa8d-sjyUejF0QuSjINFcjifJ8DiU40cL_-K6OJotlx6e0FqOivz6Nlj13QZxQ0I3FiiSi03mYKy240jYMpOpjXr7yPmEXLdCJdP5ycmTiJLxf4Bugww6u4F2uxy22978ACyFGSLHBiQyjczj_can7qKXAkMwYJKcGNjaNi8jG5iTIwsGswRjD1hvY-AGUotMFbPCszX3HW1M_ar-livaheiZauCfKV-Uc1ZeI3gijWEwtWQ0jye29FyQPCCpOBvT6RbUvFEpfqpwcMQuUhOyEfgzli2dpuOAgkSjCPE6ctoxjbYa62YzE-yrXAGc5_ptQy_2vw7t0k3jUzSo2Tv0aKnqvvKcj9SIilkZV4Nf-TL_d2E7d48bBJDlqbAv7fkhhd2YlkLqwdR1MqZtygcR1Jh8p2Y1pFAa4mSj7hh4M-zfSu--6dij2iKIbnKQ4DbXyGpMZXBAqTHMe9PPOwGxWKShlN5a5T89B04d_GwJYBDJx2ctecqZxDMjkTn3wVGl_5wuDnrEgd0I91vmAoYuWldR_h8M_FjDFiHefdbZjw1TxVKjkp6wk6zQiXCvvCZYJa9XkhytcllWvUI4C0gbxHrEzZRy9Vii3buqnbiIM9Qj0VPx-Q-FKM_usZBBmlvmk9PMQ8rH9vVT8dRFNQEj-aqudB5yUcTx8XaUFwYAts04OObGBqXoazYtxh6WvHwrf09pb_g0dwzE_rlcQdYxcFLOpYD-AentRAjOuIr4bLRM9BMERBxPvvPCxZ2Mva8YqV2TIOtxzMY08freim6du1IuYprO6CoejPaBdULhct-nsPubOdjLBikZt_bwumvmqGXnxI_uu51b9HtzPeDpWIjF6pi88bcsOk0qglA9GAu3wwX-iIdaV19VdVCO4KJjxiVrbTY1IVgWSdz98Alb_HzpXsoS6i2PRAjjsYOe4RBX3etxjsY07XXLlmXAM_vuYXc8Y6STxvBk4ST4OkaCvUk9DoZbVL5KmVcT6TaFpbVCOB_eHkHIvMjXc35kzxCdqEMG3FpRzL_UkY8pPridvq2z1Xw0al2KEBvdKPlInB8-zX5ANGeRkMGZ6ZfyX1zCIdYLe3wrC8xqr5nUZ-ueWmtqYLavSg8mQKphp4QyVaiwtbxEt5GEiVG7_LR754mGQYPdr9Shh3ECAp8wmSfDVO8MHaLmzgo3RXeqlqFldRjQzDHtCaGhjD9bHKF3yWF2LtH4gUN-Sf--86lcq7iwHDSDm656P_FBfYmE7rA0svH-m3hQoBhza4CKJ7s7f7ZymEhcHAfH7SPImZ3Y-kT_Sy1mbCCf3Yg8uitrpX7ukO6_bIANS_R4oiOPcuLixbWY0ZSyq8ERB5fa5EsIUm7PpGxbO96nmk5rPkewyB4gCtslwJI0Ye7zHtqrDBz1j1nsjIKsRCfFWlUdRF8J1JPiiBSvP8SraQ_94cnKBCsl34BGsVm-R1_ULbuyahBzSHq2Kwr0XQuNLdGChyLKS_FZVT58kbRFsvjZnbalAZ-k9alMeZ-pdWX5f9nSn3w7fz675zOxnBaqiZmoWHXFNOBVGH7gkz05ynJ2B8j_RpdRNJKXUN8pAvf595HGl2IPdaDhqoeS2_3jixO5mmxZuPEdzopoBFRarWud99mxH-mYxWJzKiA1pLNqj7SO93p2-jB-jtsCfZfk6bVEWpRRkIEz0XvxffFTVuGUCqpGS7FiFZc4pQU24pCrdpg2w3xeDSrmfHDAx2vUvv0iRBnQxTTWx2-de2TQQTpR5tjFNyOhYGVn1OXqkbkNtIUHdnNGA1QBCU0Qs0471Ss1CrxXIeeNVSTd00jiu4_ELk6nJYgSpmS8G_crrDza8mRLV5Yk0ItRrZj6pwKUOEaYeyM-RHyhrjf09yaf7Qc3sAozQF0aXFCQjSYiVb98DuGH28HLUxW9ulmSKKR4pYKlCOLNGm0h_gWCpSa0H1HXCgEoPn68HyaJogv_xH3k4ERYyJnxu8zVbVPMGoa9q9nNRQQ9Ks2AvxYRQeGFSCTACBmuookvHsO1zjYfHNuSCD7pCLRFE76KlmSiAX6l9LNOq_xe9Oos-1AvcZHkmVsuh-mjTVkBOjG6zmnHiNJirBpORs_UWL5lmlQBeaXgdHxcb4tHIn8XYXFkQiC4b4pw==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:google_calendar',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'search_events',
                            'tool_args': {
                                'time_min': '2025-10-23T00:00:00',
                                'time_max': '2025-10-23T23:59:59',
                                'timezone_str': 'America/Los_Angeles',
                                'max_results': 50,
                                'query': None,
                                'calendar_id': 'primary',
                                'next_page_token': None,
                            },
                        },
                        tool_call_id='mcp_0558010cf1416a490068faa0fdf64481a085a3e5b8f7d6559a',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:google_calendar',
                        content={
                            'output': None,
                            'error': {
                                'code': 500,
                                'message': 'An unknown error occurred while executing the tool.',
                                'type': 'http_error',
                            },
                        },
                        tool_call_id='mcp_0558010cf1416a490068faa0fdf64481a085a3e5b8f7d6559a',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0558010cf1416a490068faa0ff5c9081a0b156a84d46e5d787',
                        signature='gAAAAABo-qEE72KCH4RlulMdH6cOTaOQwFy4of4dPd8YlZ-zF9MIsPbumWO2qYlZdGjIIXDJTrlRh_5FJv2LtTmMbdbbECA20AzFMwE4pfNd2aNLC5RhcHKa4M9acC1wYKAddqEOPP7ETVNBj-GMx-tMT_CY8XnBLWvSwPpcfde9E--kSrfsgvRn1umqDsao4sLlAtV-9Gc6hmW1P9CSJDQbHWkdTKMV-cjQ-wZHFCly5kSdIW4OKluFuFRPkrXs7kVmlGnMr8-Q5Zuu1ZOFR9mPvpu2JdxAFohjioM-ftjeBuBWVJvOrIF4nV-yIVHVT-_psAZaPUUB5cyPAtqpoxxIV3iPKPU8DHctP03g_0R6pSWWHhggvO5PBw3zyPwtBwOrHBipc4nQEWEMxZxLH5SYJauTKwHNOx9NyCq8JUjZXM_v4xsGxNa4cAp7GuXqR2YyW2sx7syRUiDwtebh0xk_YOQtkv8tAjzCofmaz3n8FJ2nGSXkilaV5Q8LUNO-9-D2tsAaScDVMuLMMAHFNp_GPplWrmGES4mTCNtTXWyF1GLcQBw8dYYctV66Ocy2_zxyDoB7SsR5htlV77nJ6u1Hbp3tk26LutDrhAhe55xcki8iblHbXNY9MRzR1SS5Zk3-dv0ex4QOzC663NvS9aK3olQbKYko5TvM7Pq4MFYfaxwFTVFVEdaskoDJieVyikz0ZzBjTsItIwL-Q2BVN2F_P_wgCV5hyDclNMPEGTMxajxfIFv-oEunmHY1_RJavl47iXWS8H3JWAvp-9YYQdTS4Aa6m5zPndvHOvEV355UawLHRPctHFUS7rE7rYmcU6KQaqC96JRM0KRfXNIgYtNfw6cxgnyqGxzTF7qeeVzObOqoQmz59Rh0U9ti37vqHb8Ca43-q2Gx2KaVZFj7MBQK8UodfaDRIEuyMB3XNfckxCefwHs7FeAj5NuNDBrm0uDcwJjs2JfY2i54gAES8kAPLGJgRpq_qdjVXqpO6W0H9E1vBdRem7zLPYbA8OOo-KCkRW4AFCVbgCpgIvo4GDNvFOMksl-d8zgQU2qroUWJRu58j1bdaar7Zlfxk0UR33nROmJpXGb_R-RCNAN1ZxJTdEU_dVfyLCeuIXPsnO-FlfO8J6Un3WWPNLuN_bDS5RocniI_ms71qLsisJQiPTs-JDFl-eMM2Hk3QqSCC6OT0CLG9XMmI_zva9yp2joQ8HdGMddE3FDCbLejRrx8fV-9Nd0tZ7SYjFG78_fre8IfL0L67CK1JIPYzhgRZgCb-FFwUy-stR_BstIn0sRr_tDCoHdxuoVCh0dZfTY1p27xbKQ50svHxp1caNp3uze0wLXP9STNouFjFpdIHMsDRaGfO9R9mMmUsFcmBMK3aikuHTpebyL1CeZsIzH2cbZLPRx3pN2IqJ-5h6-cORHuMqf3ysEEFCjXnqmzvWPuBjYDsxnxA1awaGkYKsKhqchgakrfplOjdG5tSkklggBJA93iRaUWIR-4oV6HkkrnpdK1w7BL_VT8upqZmkpHZtZCDSgINk5S5hoYPLBTtS3dcCmQIbLvPXPuGzdAZxl0bhD4Rm3GPDFszaDoFK0Jszcjlaf4SJqyZABKEf71dDbi1as-2Qwr4fxBiQIOsF8ChbYo6Z2iFtUpBnbruFUIwB5QyKfWnwEZbOgf4UbIvIqNMkTzMc8tJgz6Ddqfih8VeNH3v8_84J6vHU0SVm_gvkgQ6P6N_6r5LwNdlAEff0hFwn-aTHWZ3s8MICckUZj97lKoZxAl91WlsKa0yrLw24dxvJ6bhZf0FsOitUJGd7vFPx0TxSobUkzE2RrbQ3hziPxw2Gins4aI6YG3M1gfumd3MgdH-fYBvZulJ9vmw0ZC1Dqh6BkCWHOFKsnpQvHmYuyTzUmnYuJf8N5j_b9XNw0krmxouOCPQClFmIOBLw8XPbe3xf0F5JP7BC0PpjlPT33A5Z6Za5zlA5O-DE_Wp0WG885-GaKtZI-zBZW3R0lc9A4s0HbxqA3lqH8leXOCe6WO46Z_iTQlALpTR-7oaHqzTegq0KSmEjCFO-jLSrVZnBOQ4ddTvLj4ASsQbj-o6TFUFVZAKSLI3FtWovHw02Gc_D0luFz9TbfaXM-EapEQYajkG0_b_nSCoPq0T9HSyvU4oCxXyQvhwIgzbijR-BheN6a_l6hiqZCw9L1c8MdPRtjpbHtEwWkpQ62s8XdydeJnV5vJYp9ezBbS_vWQ7Nz1siai6epJTdzDkRm-dudVhKzdohwg-FOQ-5gSrvoPS_MF4lZvah3iXY1g4uePO4eNDWGJ74YPybiy',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:google_calendar',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'search_events',
                            'tool_args': {
                                'time_min': '2025-10-23T00:00:00Z',
                                'time_max': '2025-10-23T23:59:59Z',
                                'timezone_str': None,
                                'max_results': 50,
                                'query': None,
                                'calendar_id': 'primary',
                                'next_page_token': None,
                            },
                        },
                        tool_call_id='mcp_0558010cf1416a490068faa102400481a09fa35e84bc26c170',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:google_calendar',
                        content={
                            'output': None,
                            'error': {
                                'code': 500,
                                'message': 'An unknown error occurred while executing the tool.',
                                'type': 'http_error',
                            },
                        },
                        tool_call_id='mcp_0558010cf1416a490068faa102400481a09fa35e84bc26c170',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0558010cf1416a490068faa102d89481a0b74cca04bcb8f127',
                        signature='gAAAAABo-qEECuiSxfvrR92v1hkqyCTCWyfmpHSaW-vVouk5mOTIFDvaBZdVTFH8-dJfpwEG3MCejRKh9V-I8mrYAjhudVr1ayHo8UYOOU1cfVc6w3wsrkL8hXljjE-amiJhBSjvRc2nwwGtgYpDxOfWTqJkaUvFnMD6MrS4CwMrCBbDOLYZgM1cQbidtrrtpP7D5u42tR6coC_PCOqwPzDN4f0RggrxVxh0038p81VUmlkUeA2jWzRyFpeDGRjXFk84Og73rXAp7EWQv7TmzgVXBjCVwwzJNU8HCZ_gkwh5dvL94QxBx32lEmfOOKcqA3hN3FLwDqXlZ8f7jEqYInnpILQgX5XMdM9OrCyXmDCr_eIy00cjvxnTcXhCnZBOaKCKmTP74yUpGNdLbQcr4BalTiviNYEeCAhJyRo4KnhUZbBoT7MB5NULf-kqhRo1gEGKjWiLdV47PhR7Z8i4BK7zBceganMKpLtzIMW5a6JAujC4Z9FYxcpJZI_CD9NHsPr4SjKgIwv89d6BYo89-xfflF6ZUZBkuDUnL2-Nc9CKgGuKlcDunvYLr38pzA278OFYzh9T42u4SbS8KkSXKjGU3H8LfpMnBEZigriixLt5vj7qnWmZvCFarzxT4U4qqR1ITp5rkO6G9kYvBEfS7wu768mteDBgAajUaeOMQEfjJRErC4wfzbB89YCsXPJz0JE90QZ5LeiP5ZlVezTTaddG9JmiGsBCPckqUb1LWdpvekCfPkePF_uDMVWyJpQ4ZBzQsZx8sHf5spygsiQjlzTiriqwhoTcPuXoONoCr9HeFX1Qy8SGOm87siRPAD7FHJdDxbJwq8tOlMpx8MH1dqEY07lwoxZB0GQ9XbB7QJXfQR_27nkpqBYFkrbqChNJLO2x8gNFClbB0mgYQE1CRy64y6yOrG3CtS53RK5VGrF1GnqwuWdZ452VgShT5nAmPFRlRk1S9px4eMUTAozT0QAYrlHQC7b6I6K3m_Qe3kXGpnn_87i2eGG8mHmXG2FvFChkgf2OU7-LRy_Wl_u-ataICeoBwfngBFMppvUW6tJP009HK7mUE8P1KJntN3ExKLIBhmKhV6ziBpIi1bSTmd8leYqfSaf648c7-sVuDRx7DzxTp19l3fwVFa67GdiagZFs7xaU1HxMnMc3uy5VKWAH_qcv-Mga3VCTtTPpMTjvB95nsLeOFjS2FtpPvaP0N6o5kkkzW7cteWpOHhSX0z7AQA7CqgOCQLfLUc7ltVxnOH4WdHoeZFah_q_Ue6caf0kNo4YsTfbRDdzsW70o8P5Agr-Pgttg19vTDA_eBFur9GDKIRT0vYMWPpykwJBDTgJKOFW6uyNkqNWk_RAAvleE9pAyOoSmgomyrMcnnpdeYHNxeNxvTWFC3mcKSjJIB316wypPvaGTJyaK_pxJScD7CtLrIPkgwPpOsJnDySF6wGe-fGsUMt3zxJrc-S6fp24mYVfTRZbjUsP0fJgLmCohJiAtEg_xvlQ8sPyuLoLdOdossTQ7ufl0CwVn4f_ol4q__gpTvYVaoGsWl3QmHul5zj7OUAn7of6iBfCSlXbrauJvMyNYt4x_dLM8SXTRNPe-ZMDmER9DOw0KJXcUrpl6uw4TphKmUOK6KrxqshujXdN9VDgOwD7eKqIHpvC_6a2R6sS6ZHcebmh2o3bic-Hctomrbv03OQ==',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0558010cf1416a490068faa103e6c481a0930eda4f04bb3f2a',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=1065, output_tokens=760, details={'reasoning_tokens': 576}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 23, 21, 41, 13, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0558010cf1416a490068faa0f945bc81a0b6a6dfb7391030d5',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_requires_function_call_status_none(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        profile=replace(openai_model_profile('gpt-5'), openai_responses_requires_function_call_status_none=True),
    )
    agent = Agent(model)

    @agent.tool_plain
    def get_meaning_of_life() -> int:
        return 42

    result = await agent.run('What is the meaning of life?')
    messages = result.all_messages()

    _, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        messages,
        model_settings=cast(OpenAIResponsesModelSettings, model.settings or {}),
        model_request_parameters=ModelRequestParameters(),
    )
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is the meaning of life?'},
            {
                'id': 'rs_00109b6f3427f09d00697ccd36c3e881a1b25dda264adab7b4',
                'summary': [],
                'encrypted_content': 'gAAAAABpfM080Vlq1Ozupd3MDgnppHzbs25KXa0hICWmqtEZ_eZBJ5SYExC7UThfti52XkjVrCdmKJ0PTNQivIA0oTJzM2LkvGtlhATwWeU_oFqUjqWc0rSEt95WFbEgJMr_ECBlPjj6TZb_acbrBbDR8cANz-qvW8GwS_BwUgX7EQrlEnKKeCV25U4EjLPmKnBhrBSqPmZfkGWppXRsT0BQVX5mY1dxP6QYYcXkiw6j3RnkV1Ooov1dMecEQWcslOUITwW943iahfwHZIDZ7lD4oceQXLptV6mFN6iVMtrfOtGPHcnxY2invRvVrROeYeahXJw1bLFPzuMnb5n0eOjQM0qBo_GpGRNnib1-53SSJxlJgDcL0Q662UVcIW_iW-k7ePz9gl8aebvmxARfwP5EjKGivDKEcM2uHhmbWFXQ3znHLz1eNj_bilDAMOuCa7o9bOj1ZGkJyXGQfpg9xwDL_abCJqJR9oJBZyMSu_Q6kvTE2CXC4mmrsTuDszM-PoyZeZM1hK6zVcrH7-gDQ-WQAjId4Z-D6M3Ra4VZ0f95g0qRRPpfXBHPghsmR3CNZ2Q1nQylfWQhP63Yzp--ktJi1s-_kF3mvXa4gwkHw5a-Qf3AtH8i921yAxsqz-8dH6SAX3c-fcbBTlYYuwikS_hld-1pHzHTa097z7sWz17NiUeC3tYm9PGYsQHrMYgzOZBSTm-OrJ11CRIGH6-ItURj2HoiQIq8lXR6n_Dq4zD6bXZsBlWqCSK4_jpLBlwmvmwis28a2dNde0PXGXw09XccQjhRtK9o--c3as_H6Evw-TgBAA2lQ91PWVNrv4Tu5xO-YY3UfIx3MbBV_Rns_B-jIzZ4mwUu7yAG6BC1yj934H7_U7-5G38gCWZJQMrHbUJTaajNXAcoFzUmmGFr2BOidr8jFsO9RE9D78jjnPOSwiTcBRWsxeOoKssdtAMdsS3BLJ1gmFFuOi9h281tW926txM341LVoJkiq5clnmqt_MIluiFqyd17LqfCoX9wz6yywD_K6hsaz5Op2o__BuPenJcLXOPF5-i22NY8180iwjFNVoVeyfcfa2JXdvf5rupZWxoSl7kpYjTzlcRQVLgLujabiBg86Jt64rG931lOHRfYOFTzPpsUL3TS7T7mThrz6q_hN8uqqTKqhlBPCKN5icLnN1v_GOs0F7iZa5AVBfHFxAqvLFjelxV0gQwXFKBWcT1Ca_nk4WEJ8AgrtmdWS83V743B3aLtT5OLxosM2Xg8xGqEMmF2C8mWtkR1HnAGNND3K7ehatAZjrTF5Z7wMNt6vgxzhaUo70cz5_IOdTTRlbDzHZHS3Nf7_qshOklywVQ96zFwMY6WJJ48GS-qNco_F1qUmezYpTWPsE3wOj-Q6y0XhEmWtFZMs9zO5sUQ9wL3uXUMJlIQvFq6KGXEm056w8mctmfJevMJkAczho0R4468OmnOGrVq656rXs3I1JrxTS8BKY3Q1A9NHBkbHKtR411s5xiUK-au-ivNOApkCVIlHMBhdT1tm6VvcHVxBAMh9xgLZ121DAy3fbZ2tUNoKiEQymSf7victIZR3NnW0E3nrujJCGYrZNpBXBofRwHyzMSO5jnV-lYsVHrR5qtUfq3AkpkMrfV8WkyhgxCCerX7JmGiWsgUxbh1nq8IxGDmgE_Dk2Pa-GlbW232O-XcK6YzwYaPJCt4XHus94eu-l7dRlAR_dNBdvirORkqwfd2d6ZWTn4e3fzFQq3tIMDPS-xEtPTQKmskAaMCpj-ecixlYSfTUeEl6zgzkaXpgiB44CJp6FjyjvY6lAXc8Mf7yarXxauDK5krlvo4I-sGwWLJoyVwPwzs3r1Bu6CUtcJt_Lo09V-3Qcnbk7vmMrow6Emmkd8uCG3NcNg4K0Pav6NRA4LYlK76Oy7mZzT0yfo6bbOeruZq813qAh24GK1fEvKXQ5UznRYBeQpg-v9dgBXzaVk-y69ZyoABCAZ9jWHFUAcJK-b5tOTsvfxWX8DZuJNUBm9Cyr9DM9b2EYA31PecLJR5lrtnxLeZrLlf8d69v_faVCBLXXfbu8DX073ZlLmvOCQCtSvV0e8LhOzEub1BukhXJlnEftyd3seV6gTmbqGFh84VoNxd5VMwmzL6-wf0iow6AQ85dUqEDPg9rA4226E=',
                'type': 'reasoning',
            },
            {
                'name': 'get_meaning_of_life',
                'arguments': '{}',
                'call_id': 'call_mEugxkUNfk5wWeN6oJldMcJG',
                'type': 'function_call',
                'status': None,
                'id': 'fc_00109b6f3427f09d00697ccd3c219881a1a6123b2bcbb220fd',
            },
            {'type': 'function_call_output', 'call_id': 'call_mEugxkUNfk5wWeN6oJldMcJG', 'output': '42'},
            {
                'role': 'assistant',
                'id': 'msg_00109b6f3427f09d00697ccd3d3da881a1b7be8155b26b7a3e',
                'content': [{'text': '42', 'type': 'output_text', 'annotations': []}],
                'type': 'message',
                'status': 'completed',
            },
        ]
    )


async def test_openai_responses_raw_cot_only(allow_model_requests: None):
    """Test raw CoT content from gpt-oss models (no summary, only raw content)."""
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[],
                type='reasoning',
                content=[
                    ReasoningContent(text='Let me think about this...', type='reasoning_text'),
                    ReasoningContent(text='The answer is 4.', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_123',
                        provider_name='openai',
                        provider_details={'raw_content': ['Let me think about this...', 'The answer is 4.']},
                    ),
                    TextPart(content='4', id='msg_123', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_raw_cot_with_summary(allow_model_requests: None):
    """Test raw CoT content with summary from gpt-oss models.

    When both summary and raw content exist, raw content is stored in provider_details
    while summary goes in content.
    """
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[Summary(text='Summary of thinking', type='summary_text')],
                type='reasoning',
                encrypted_content='encrypted_sig',
                content=[
                    ReasoningContent(text='Raw thinking step 1', type='reasoning_text'),
                    ReasoningContent(text='Raw thinking step 2', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='Summary of thinking',
                        id='rs_123',
                        signature='encrypted_sig',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw thinking step 1', 'Raw thinking step 2']},
                    ),
                    TextPart(content='4', id='msg_123', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_multiple_summaries(allow_model_requests: None):
    """Test reasoning item with multiple summaries.

    When a reasoning item has multiple summary texts, each should become a separate ThinkingPart.
    """
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[
                    Summary(text='First summary', type='summary_text'),
                    Summary(text='Second summary', type='summary_text'),
                    Summary(text='Third summary', type='summary_text'),
                ],
                type='reasoning',
                encrypted_content='encrypted_sig',
                content=[
                    ReasoningContent(text='Raw thinking step 1', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='Done', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('Test multiple summaries')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Test multiple summaries',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='First summary',
                        id='rs_123',
                        signature='encrypted_sig',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw thinking step 1']},
                    ),
                    ThinkingPart(content='Second summary', id='rs_123', provider_name='openai'),
                    ThinkingPart(content='Third summary', id='rs_123', provider_name='openai'),
                    TextPart(content='Done', id='msg_123', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_raw_cot_stream_openrouter(allow_model_requests: None, openrouter_api_key: str):
    """Test streaming raw CoT content from gpt-oss via OpenRouter.

    This is a live test (with cassette) that verifies the streaming raw CoT implementation
    works end-to-end with a real gpt-oss model response.
    """
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    model = OpenAIResponsesModel('openai/gpt-oss-20b', provider=OpenRouterProvider(api_key=openrouter_api_key))
    agent = Agent(model=model)
    async with agent.run_stream('What is 2+2?') as result:
        output = await result.get_output()
    assert output == snapshot('4')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_tmp_2kbe7x16sax',
                        provider_name='openrouter',
                        provider_details={
                            'raw_content': [
                                'The user asks: "What is 2+2?" They expect a straightforward answer: 4. Just answer 4.'
                            ]
                        },
                    ),
                    TextPart(content='4', id='msg_tmp_8cjof4f6zpw', provider_name='openrouter'),
                ],
                usage=RequestUsage(
                    input_tokens=78,
                    output_tokens=37,
                    details={'is_byok': 0, 'reasoning_tokens': 22},
                ),
                model_name='openai/gpt-oss-20b',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 11, 27, 17, 43, 31, tzinfo=timezone.utc),
                },
                provider_response_id='gen-1764265411-Fu1iEX7h5MRWiL79lb94',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_raw_cot_sent_in_multiturn(allow_model_requests: None):
    """Test that raw CoT and summaries are sent back correctly in multi-turn conversations.

    Tests three distinct cases across turns:
    - Turn 1: Only raw content (no summary) - gpt-oss style
    - Turn 2: Summary AND raw content - hybrid case
    - Turn 3: Only summary (no raw content) - official OpenAI style
    """
    # Track messages sent to OpenAI
    sent_openai_messages: list[Any] = []

    # Turn 1: Only raw content, no summary
    c1 = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[],
                type='reasoning',
                content=[
                    ReasoningContent(text='Raw CoT step 1', type='reasoning_text'),
                    ReasoningContent(text='Raw CoT step 2', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )

    # Turn 2: Summary and raw content
    c2 = response_message(
        [
            ResponseReasoningItem(
                id='rs_456',
                summary=[
                    Summary(text='First summary', type='summary_text'),
                    Summary(text='Second summary', type='summary_text'),
                ],
                type='reasoning',
                encrypted_content='encrypted_sig_abc',
                content=[
                    ReasoningContent(text='More raw thinking', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_456',
                content=cast(list[Content], [ResponseOutputText(text='9', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )

    # Turn 3: Only summary, no raw content
    c3 = response_message(
        [
            ResponseReasoningItem(
                id='rs_789',
                summary=[
                    Summary(text='Final summary', type='summary_text'),
                ],
                type='reasoning',
                encrypted_content='encrypted_sig_xyz',
                content=[],
            ),
            ResponseOutputMessage(
                id='msg_789',
                content=cast(list[Content], [ResponseOutputText(text='42', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )

    mock_client = MockOpenAIResponses.create_mock([c1, c2, c3])
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    # Hook into model to capture sent messages
    original_map_messages = model._map_messages  # pyright: ignore[reportPrivateUsage]

    async def capture_messages(*args: Any, **kwargs: Any) -> Any:
        result = await original_map_messages(*args, **kwargs)
        sent_openai_messages.append(result[1])  # result is (instructions, messages)
        return result

    model._map_messages = capture_messages  # type: ignore[method-assign]

    agent = Agent(model=model)

    # Turn 1: Only raw content, no summary
    result1 = await agent.run('What is 2+2?')
    assert result1.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_123',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw CoT step 1', 'Raw CoT step 2']},
                    ),
                    TextPart(content='4', id='msg_123', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )

    # Turn 2: Summary and raw content
    result2 = await agent.run('Add 5 to that', message_history=result1.all_messages())
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_123',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw CoT step 1', 'Raw CoT step 2']},
                    ),
                    TextPart(content='4', id='msg_123', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Add 5 to that',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='First summary',
                        id='rs_456',
                        signature='encrypted_sig_abc',
                        provider_name='openai',
                        provider_details={'raw_content': ['More raw thinking']},
                    ),
                    ThinkingPart(content='Second summary', id='rs_456', provider_name='openai'),
                    TextPart(content='9', id='msg_456', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )

    # Turn 3: Only summary, no raw content
    result3 = await agent.run('What next?', message_history=result2.all_messages())
    assert result3.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_123',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw CoT step 1', 'Raw CoT step 2']},
                    ),
                    TextPart(content='4', id='msg_123', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Add 5 to that',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='First summary',
                        id='rs_456',
                        signature='encrypted_sig_abc',
                        provider_name='openai',
                        provider_details={'raw_content': ['More raw thinking']},
                    ),
                    ThinkingPart(content='Second summary', id='rs_456', provider_name='openai'),
                    TextPart(content='9', id='msg_456', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What next?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='Final summary',
                        id='rs_789',
                        signature='encrypted_sig_xyz',
                        provider_name='openai',
                    ),
                    TextPart(content='42', id='msg_789', provider_name='openai'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )

    # Verify what was sent to the API in each turn
    assert len(sent_openai_messages) == 3

    # Turn 2 messages: should contain raw-only reasoning item from turn 1
    turn2_messages = sent_openai_messages[1]
    turn2_reasoning = [msg for msg in turn2_messages if msg.get('type') == 'reasoning']
    assert len(turn2_reasoning) == 1
    assert turn2_reasoning[0] == snapshot(
        {
            'type': 'reasoning',
            'summary': [],
            'encrypted_content': None,
            'id': 'rs_123',
            'content': [
                {'type': 'reasoning_text', 'text': 'Raw CoT step 1'},
                {'type': 'reasoning_text', 'text': 'Raw CoT step 2'},
            ],
        }
    )

    # Turn 3 messages: should contain both reasoning items
    turn3_messages = sent_openai_messages[2]
    turn3_reasoning = [msg for msg in turn3_messages if msg.get('type') == 'reasoning']
    assert len(turn3_reasoning) == 2
    assert turn3_reasoning[0] == snapshot(
        {
            'type': 'reasoning',
            'summary': [],
            'encrypted_content': None,
            'id': 'rs_123',
            'content': [
                {'type': 'reasoning_text', 'text': 'Raw CoT step 1'},
                {'type': 'reasoning_text', 'text': 'Raw CoT step 2'},
            ],
        }
    )
    assert turn3_reasoning[1] == snapshot(
        {
            'type': 'reasoning',
            'summary': [
                {'type': 'summary_text', 'text': 'First summary'},
                {'type': 'summary_text', 'text': 'Second summary'},
            ],
            'encrypted_content': 'encrypted_sig_abc',
            'id': 'rs_456',
            'content': [
                {'type': 'reasoning_text', 'text': 'More raw thinking'},
            ],
        }
    )


async def test_openai_responses_model_file_search_tool(tmp_path: Path, allow_model_requests: None, openai_api_key: str):
    async_client = AsyncOpenAI(api_key=openai_api_key)

    test_file_path = tmp_path / 'file.txt'
    test_file_path.touch()
    test_file_path.write_text('Paris is the capital of France. It is known for the Eiffel Tower.')

    file = None
    vector_store = None
    try:
        file = await async_client.files.create(file=test_file_path, purpose='assistants')

        vector_store = await async_client.vector_stores.create(name='test-file-search')
        await async_client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

        m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=async_client))
        agent = Agent(
            m,
            instructions='You are a helpful assistant.',
            builtin_tools=[FileSearchTool(file_store_ids=[vector_store.id])],
        )

        result = await agent.run('What is the capital of France?')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    instructions='You are a helpful assistant.',
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={'queries': ['What is the capital of France?']},
                            tool_call_id=IsStr(),
                            provider_name='openai',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content={'status': 'completed'},
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='openai',
                        ),
                        TextPart(content='The capital of France is Paris.', id=IsStr(), provider_name='openai'),
                    ],
                    usage=RequestUsage(input_tokens=870, output_tokens=30, details={'reasoning_tokens': 0}),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

        messages = result.all_messages()
        result = await agent.run(user_prompt='Tell me about the Eiffel Tower.', message_history=messages)
        assert result.new_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Tell me about the Eiffel Tower.',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    instructions='You are a helpful assistant.',
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={'queries': ['Eiffel Tower']},
                            tool_call_id=IsStr(),
                            provider_name='openai',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content={'status': 'completed'},
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='openai',
                        ),
                        TextPart(
                            content='The Eiffel Tower is a famous landmark in Paris, the capital of France. It is widely recognized and serves as an iconic symbol of the city.',
                            id=IsStr(),
                            provider_name='openai',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=1188, output_tokens=55, details={'reasoning_tokens': 0}),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    finally:
        await _cleanup_openai_resources(file, vector_store, async_client)


def test_map_file_search_tool_call():
    from openai.types.responses.response_file_search_tool_call import ResponseFileSearchToolCall

    from pydantic_ai.models.openai import _map_file_search_tool_call  # pyright: ignore[reportPrivateUsage]

    item = ResponseFileSearchToolCall.model_validate(
        {
            'id': 'test-id',
            'queries': ['test query'],
            'status': 'completed',
            'results': [
                {
                    'id': 'result-1',
                    'title': 'Test Result',
                    'url': 'https://example.com',
                    'score': 0.9,
                }
            ],
            'type': 'file_search_call',
        }
    )

    call_part, return_part = _map_file_search_tool_call(item, 'openai')
    assert (call_part, return_part) == snapshot(
        (
            BuiltinToolCallPart(
                tool_name='file_search',
                args={'queries': ['test query']},
                tool_call_id='test-id',
                provider_name='openai',
            ),
            BuiltinToolReturnPart(
                tool_name='file_search',
                content={
                    'status': 'completed',
                    'results': [
                        {
                            'attributes': None,
                            'file_id': None,
                            'filename': None,
                            'id': 'result-1',
                            'text': None,
                            'title': 'Test Result',
                            'url': 'https://example.com',
                            'score': 0.9,
                        }
                    ],
                },
                tool_call_id='test-id',
                timestamp=IsDatetime(),
                provider_name='openai',
            ),
        )
    )


async def test_openai_responses_model_file_search_tool_stream(
    tmp_path: Path, allow_model_requests: None, openai_api_key: str
):
    async_client = AsyncOpenAI(api_key=openai_api_key)

    test_file_path = tmp_path / 'file.txt'
    test_file_path.touch()
    test_file_path.write_text('Paris is the capital of France. It is known for the Eiffel Tower.')

    file = None
    vector_store = None
    try:
        file = await async_client.files.create(file=test_file_path, purpose='assistants')

        vector_store = await async_client.vector_stores.create(name='test-file-search-stream')
        await async_client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

        m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=async_client))
        agent = Agent(
            m,
            instructions='You are a helpful assistant.',
            builtin_tools=[FileSearchTool(file_store_ids=[vector_store.id])],
        )

        event_parts: list[Any] = []
        async with agent.iter(user_prompt='What is the capital of France?') as agent_run:
            async for node in agent_run:
                if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as request_stream:
                        async for event in request_stream:
                            event_parts.append(event)

        assert agent_run.result is not None
        messages = agent_run.result.all_messages()
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    instructions='You are a helpful assistant.',
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={'queries': ['What is the capital of France?']},
                            tool_call_id=IsStr(),
                            provider_name='openai',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content={'status': 'completed'},
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='openai',
                        ),
                        TextPart(content='The capital of France is Paris.', id=IsStr(), provider_name='openai'),
                    ],
                    usage=RequestUsage(input_tokens=1177, output_tokens=37, details={'reasoning_tokens': 0}),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

        assert event_parts == snapshot(
            [
                PartStartEvent(
                    index=0,
                    part=BuiltinToolCallPart(
                        tool_name='file_search',
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    ),
                ),
                PartDeltaEvent(
                    index=0,
                    delta=ToolCallPartDelta(
                        args_delta={'queries': ['What is the capital of France?']},
                        tool_call_id=IsStr(),
                    ),
                ),
                PartEndEvent(
                    index=0,
                    part=BuiltinToolCallPart(
                        tool_name='file_search',
                        args={'queries': ['What is the capital of France?']},
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    ),
                    next_part_kind='builtin-tool-return',
                ),
                PartStartEvent(
                    index=1,
                    part=BuiltinToolReturnPart(
                        tool_name='file_search',
                        content={'status': 'completed'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    previous_part_kind='builtin-tool-call',
                ),
                PartStartEvent(
                    index=2,
                    part=TextPart(content='The', id=IsStr(), provider_name='openai'),
                    previous_part_kind='builtin-tool-return',
                ),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' capital')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' of')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' France')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' is')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' Paris')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='.')),
                PartEndEvent(
                    index=2,
                    part=TextPart(content='The capital of France is Paris.', id=IsStr(), provider_name='openai'),
                ),
                BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                    part=BuiltinToolCallPart(
                        tool_name='file_search',
                        args={'queries': ['What is the capital of France?']},
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    )
                ),
                BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                    result=BuiltinToolReturnPart(
                        tool_name='file_search',
                        content={'status': 'completed'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    )
                ),
            ]
        )

    finally:
        await _cleanup_openai_resources(file, vector_store, async_client)


async def test_openai_responses_model_file_search_tool_with_results(
    tmp_path: Path, allow_model_requests: None, openai_api_key: str
):
    """Test that openai_include_file_search_results setting includes file search results in the response."""
    async_client = AsyncOpenAI(api_key=openai_api_key)

    test_file_path = tmp_path / 'file.txt'
    test_file_path.touch()
    test_file_path.write_text('Paris is the capital of France. It is known for the Eiffel Tower.')

    file = None
    vector_store = None
    try:
        with open(test_file_path, 'rb') as f:
            file = await async_client.files.create(file=f, purpose='assistants')

        vector_store = await async_client.vector_stores.create(name='test-file-search-with-results')
        await async_client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

        m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=async_client))
        agent = Agent(
            m,
            instructions='You are a helpful assistant.',
            builtin_tools=[FileSearchTool(file_store_ids=[vector_store.id])],
        )

        # Use the openai_include_file_search_results setting to include search results
        result = await agent.run(
            'What is the capital of France?',
            model_settings=OpenAIResponsesModelSettings(openai_include_file_search_results=True),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    instructions='You are a helpful assistant.',
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={'queries': ['What is the capital of France?']},
                            tool_call_id=IsStr(),
                            provider_name='openai',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content={
                                'status': 'completed',
                                'results': [
                                    {
                                        'attributes': {},
                                        'file_id': IsStr(),
                                        'filename': IsStr(),
                                        'score': IsFloat(),
                                        'text': 'Paris is the capital of France. It is known for the Eiffel Tower.',
                                        'vector_store_id': IsStr(),
                                    }
                                ],
                            },
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='openai',
                        ),
                        TextPart(content=IsStr(), id=IsStr(), provider_name='openai'),
                    ],
                    usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt(), details={'reasoning_tokens': 0}),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    finally:
        await _cleanup_openai_resources(file, vector_store, async_client)


async def test_openai_responses_runs_with_instructions_only(
    allow_model_requests: None,
    openai_api_key: str,
):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, instructions='Generate a short article about artificial intelligence in 3 sentences.')

    # Run with only instructions, no explicit input messages
    result = await agent.run()

    # Verify we got a valid response
    assert result.output
    assert isinstance(result.output, str)
    assert len(result.output) > 0


async def test_web_search_call_action_find_in_page(allow_model_requests: None):
    """Test for https://github.com/pydantic/pydantic-ai/issues/3653"""
    c1 = response_message(
        [
            ResponseFunctionWebSearch.model_construct(
                id='web-search-1',
                action={
                    'type': 'find_in_page',
                    'pattern': 'test',
                    'url': 'https://example.com',
                },
                status='completed',
                type='web_search_call',
            ),
        ]
    )
    c2 = response_message(
        [
            ResponseOutputMessage(
                id='output-1',
                content=cast(list[Content], [ResponseOutputText(text='done', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )
    mock_client = MockOpenAIResponses.create_mock([c1, c2])
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model=model)

    result = await agent.run('test')

    assert result.all_messages()[1] == snapshot(
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'type': 'find_in_page', 'pattern': 'test', 'url': 'https://example.com'},
                    tool_call_id='web-search-1',
                    provider_name='openai',
                ),
                BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={'status': 'completed'},
                    tool_call_id='web-search-1',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ],
            model_name='gpt-4o-123',
            timestamp=IsDatetime(),
            provider_name='openai',
            provider_url='https://api.openai.com/v1',
            provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
            provider_response_id='123',
            run_id=IsStr(),
        )
    )

    response_kwargs = get_mock_responses_kwargs(mock_client)
    assert response_kwargs[1]['input'][1] == snapshot(
        {
            'id': 'web-search-1',
            'action': {'type': 'find_in_page', 'pattern': 'test', 'url': 'https://example.com'},
            'status': 'completed',
            'type': 'web_search_call',
        }
    )


async def test_openai_responses_system_prompts_ordering(allow_model_requests: None):
    """Test that system prompts are correctly ordered in mapped messages."""
    c = response_message(
        [
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='ok', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System prompt 1'),
                SystemPromptPart(content='System prompt 2'),
                UserPromptPart(content='Hello'),
            ],
            instructions='Instructions content',
        ),
    ]

    instructions, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        messages,
        model_settings=cast(OpenAIResponsesModelSettings, {}),
        model_request_parameters=ModelRequestParameters(),
    )

    # Verify instructions are returned separately
    assert instructions == 'Instructions content'

    # Verify system prompts are in order, followed by user message
    assert openai_messages == snapshot(
        [
            {'role': 'system', 'content': 'System prompt 1'},
            {'role': 'system', 'content': 'System prompt 2'},
            {'role': 'user', 'content': 'Hello'},
        ]
    )


async def test_reasoning_summary_auto(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, instructions='You are a helpful coding assistant.')
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='auto')

    result = await agent.run(
        'Write a Python function that calculates the factorial of a number. Think step by step.',
        model_settings=settings,
    )
    assert result.response.thinking == snapshot("""\
**Generating factorial function**

I need to respond with a Python function for calculating the factorial. The user wants me to think step-by-step, but I need to keep my reasoning brief. I'll provide a brief explanation of how the function works and include some input validation. I could choose either an iterative or recursive approach. I'll keep the details high-level, showing only the essential steps before presenting the final code to the user.\
""")


async def test_openai_include_raw_annotations_non_streaming(allow_model_requests: None, openai_api_key: str):
    """Test that text annotations are included in provider_details when the setting is enabled."""
    prompt = 'What is the tallest mountain in Alberta? Provide one sentence with a citation.'
    instructions = 'Use web search and include citations in your answer.'

    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, instructions=instructions, builtin_tools=[WebSearchTool()])

    # Test with annotations enabled
    settings = OpenAIResponsesModelSettings(openai_include_raw_annotations=True)
    result = await agent.run(prompt, model_settings=settings)

    messages = result.all_messages()
    assert messages[0] == ModelRequest(
        parts=[UserPromptPart(content=prompt, timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        instructions=instructions,
        run_id=IsStr(),
    )
    response = cast(ModelResponse, messages[1])
    assert response.provider_name == 'openai'
    assert response.provider_url == 'https://api.openai.com/v1/'
    assert response.finish_reason == 'stop'

    tool_call = next(part for part in response.parts if isinstance(part, BuiltinToolCallPart))
    assert tool_call.tool_name == 'web_search'
    assert isinstance(tool_call.args, dict)
    assert tool_call.args.get('query') == 'tallest mountain in Alberta'
    assert tool_call.args.get('type') == 'search'

    text_part = next(part for part in response.parts if isinstance(part, TextPart))
    assert text_part.provider_details and 'annotations' in text_part.provider_details

    # Test with annotations disabled (default)
    model2 = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key=openai_api_key))
    agent2 = Agent(model2, instructions=instructions, builtin_tools=[WebSearchTool()])
    result2 = await agent2.run(prompt)

    messages2 = result2.all_messages()
    assert messages2[0] == ModelRequest(
        parts=[UserPromptPart(content=prompt, timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        instructions=instructions,
        run_id=IsStr(),
    )
    response2 = cast(ModelResponse, messages2[1])
    text_part2 = next(part for part in response2.parts if isinstance(part, TextPart))
    assert not (text_part2.provider_details or {}).get('annotations')
