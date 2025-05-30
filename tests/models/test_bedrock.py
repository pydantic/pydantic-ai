from __future__ import annotations as _annotations

import datetime
from typing import Any

import pytest
from dirty_equals import IsInstance
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage

from ..conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_bedrock_model(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    assert model.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'
    agent = Agent(model=model, system_prompt='You are a chatbot.')

    result = await agent.run('Hello!')
    assert result.output == snapshot(
        "Hello! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here to help."
    )
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=7, response_tokens=30, total_tokens=37))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a chatbot.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='Hello!',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="Hello! How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here to help."
                    )
                ],
                usage=Usage(requests=1, request_tokens=7, response_tokens=30, total_tokens=37),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_bedrock_model_structured_response(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', retries=5)

    class Response(TypedDict):
        temperature: str
        date: datetime.date
        city: str

    @agent.tool_plain
    async def temperature(city: str, date: datetime.date) -> str:
        """Get the temperature in a city on a specific date.

        Args:
            city: The city name.
            date: The date.

        Returns:
            The temperature in degrees Celsius.
        """
        return '30°C'

    result = await agent.run('What was the temperature in London 1st January 2022?', output_type=Response)
    assert result.output == snapshot({'temperature': '30°C', 'date': datetime.date(2022, 1, 1), 'city': 'London'})
    assert result.usage() == snapshot(Usage(requests=2, request_tokens=1236, response_tokens=298, total_tokens=1534))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful chatbot.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='What was the temperature in London 1st January 2022?',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> To find the temperature in London on 1st January 2022, I will use the "temperature" tool. I need to provide the date and the city name. The date is already provided as "1st January 2022" and the city name is "London". I will call the "temperature" tool with these parameters.</thinking>\n'
                    ),
                    ToolCallPart(
                        tool_name='temperature',
                        args={'date': '2022-01-01', 'city': 'London'},
                        tool_call_id='tooluse_5WEci1UmQ8ifMFkUcy2gHQ',
                    ),
                ],
                usage=Usage(requests=1, request_tokens=551, response_tokens=132, total_tokens=683),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='temperature',
                        content='30°C',
                        tool_call_id='tooluse_5WEci1UmQ8ifMFkUcy2gHQ',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> I have received the result from the "temperature" tool. The temperature in London on 1st January 2022 was 30°C. Now, I will use the "final_result" tool to provide this information to the user.</thinking> '
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'date': '2022-01-01', 'city': 'London', 'temperature': '30°C'},
                        tool_call_id='tooluse_9AjloJSaQDKmpPFff-2Clg',
                    ),
                ],
                usage=Usage(requests=1, request_tokens=685, response_tokens=166, total_tokens=851),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='tooluse_9AjloJSaQDKmpPFff-2Clg',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_bedrock_model_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_output()
    assert data == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, among many other attractions.'
    )


async def test_bedrock_model_anthropic_model_with_tools(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0})

    @agent.tool_plain
    async def get_current_temperature(city: str) -> str:
        """Get the current temperature in a city.

        Args:
            city: The city name.

        Returns:
            The current temperature in degrees Celsius.
        """
        return '30°C'  # pragma: no cover

    # TODO(Marcelo): Anthropic models don't support tools on the Bedrock Converse Interface.
    # I'm unsure what to do, so for the time being I'm just documenting the test. Let's see if someone complains.
    with pytest.raises(Exception):
        await agent.run('What is the current temperature in London?')


async def test_bedrock_model_anthropic_model_without_tools(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    model = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('Paris is the capital of France.')


async def test_bedrock_model_retry(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(
        model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0}, retries=2
    )

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        raise ModelRetry('The country is not supported.')

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful chatbot.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='<thinking> To find the capital of France, I will use the available tool "get_capital". I will input the country name "France" into the tool. </thinking>\n'
                    ),
                    ToolCallPart(
                        tool_name='get_capital',
                        args={'country': 'France'},
                        tool_call_id='tooluse_F8LnaCMtQ0-chKTnPhNH2g',
                    ),
                ],
                usage=Usage(requests=1, request_tokens=417, response_tokens=69, total_tokens=486),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='The country is not supported.',
                        tool_name='get_capital',
                        tool_call_id='tooluse_F8LnaCMtQ0-chKTnPhNH2g',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
<thinking> It seems there was an error in retrieving the capital of France. The tool returned a message saying "The country is not supported." This indicates that the tool does not support the country France. I will inform the user about this limitation and suggest alternative ways to find the information. </thinking>

I'm sorry, but the tool I have does not support retrieving the capital of France. However, I can tell you that the capital of France is Paris. If you need information on a different country, please let me know!\
"""
                    )
                ],
                usage=Usage(requests=1, request_tokens=509, response_tokens=108, total_tokens=617),
                model_name='us.amazon.nova-micro-v1:0',
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_bedrock_model_max_tokens(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'max_tokens': 5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is')


async def test_bedrock_model_top_p(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'top_p': 0.5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_performance_config(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(bedrock_performance_configuration={'latency': 'optimized'})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_guardrail_config(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(
        bedrock_guardrail_config={'guardrailIdentifier': 'guardrailv1', 'guardrailVersion': 'v1', 'trace': 'enabled'}
    )
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_other_parameters(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(
        bedrock_prompt_variables={'leo': {'text': 'aaaa'}},
        bedrock_additional_model_requests_fields={'test': 'test'},
        bedrock_request_metadata={'test': 'test'},
        bedrock_additional_model_response_fields_paths=['test'],
    )
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, known for its significant cultural, political, and economic influence both within the country and globally. It is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among many other historical and architectural treasures.'
    )


async def test_bedrock_model_iter_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'top_p': 0.5})

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        return 'Paris'  # pragma: no cover

    @agent.tool_plain
    async def get_temperature(city: str) -> str:
        """Get the temperature in a city.

        Args:
            city: The city name.
        """
        return '30°C'

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the temperature of the capital of France?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=TextPart(content='<thinking')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='> To find')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of the capital of France,')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' I need to first')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' determine the capital')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of France and')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' then get')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the current')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' temperature in')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' that city. The')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital of France is Paris')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='. I')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' will use')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the "get_temperature"')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' tool to find the current temperature')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' in Paris.</')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='thinking')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='>\n')),
            PartStartEvent(
                index=1, part=ToolCallPart(tool_name='get_temperature', tool_call_id='tooluse_lAG_zP8QRHmSYOwZzzaCqA')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='{"city":"Paris"}', tool_call_id='tooluse_lAG_zP8QRHmSYOwZzzaCqA'),
            ),
            IsInstance(FunctionToolCallEvent),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_temperature',
                    content='30°C',
                    tool_call_id='tooluse_lAG_zP8QRHmSYOwZzzaCqA',
                    timestamp=IsDatetime(),
                ),
                tool_call_id='tooluse_lAG_zP8QRHmSYOwZzzaCqA',
            ),
            PartStartEvent(index=0, part=TextPart(content='The')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' current temperature in Paris, the')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital of France,')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' is 30°C')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='.')),
        ]
    )


@pytest.mark.vcr()
async def test_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, bedrock_provider: BedrockProvider
):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot(
        'The image features a fruit that is round and has a green skin with brown dots. The fruit is cut in half, revealing its interior, which is also green. Based on the appearance and characteristics, the fruit in the image is a kiwi.'
    )


@pytest.mark.vcr()
async def test_video_as_binary_content_input(
    allow_model_requests: None, video_content: BinaryContent, bedrock_provider: BedrockProvider
):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot(
        'The video shows a camera set up on a tripod, pointed at a scenic view of a rocky landscape under a clear sky. The camera remains stationary throughout the video, capturing the same view without any changes.'
    )


@pytest.mark.vcr()
async def test_image_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot(
        'The image shows a potato. It is oval in shape and has a yellow skin with numerous dark brown patches. These patches are known as lenticels, which are pores that allow the potato to breathe. The potato is a root vegetable that is widely cultivated and consumed around the world. It is a versatile ingredient that can be used in a variety of dishes, including mashed potatoes, fries, and potato salad.'
    )


@pytest.mark.vcr()
async def test_video_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video',
            VideoUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/small_video.mp4'),
        ]
    )
    assert result.output == snapshot(
        'The video shows a camera set up on a tripod, pointed at a scenic view of a rocky landscape under a clear sky. The camera remains stationary throughout the video, capturing the same view without any changes.'
    )


@pytest.mark.vcr()
async def test_document_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'Based on the provided XML data, the main content of the document is "Dummy PDF file". This is contained in the <document_content> tag for the document with index="1".'
    )


@pytest.mark.vcr()
async def test_text_document_url_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('anthropic.claude-v2', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot("""\
Based on the text in the <document_content> tag, the main content of this document appears to be:

An example text describing the use of "John Doe" as a placeholder name in legal cases, hospitals, and other contexts where a party's real identity is unknown or needs to be withheld. It provides background on how "John Doe" and "Jane Doe" are commonly used in the United States and Canada for this purpose, in contrast to other English speaking countries that use names like "Joe Bloggs". The text gives examples of using John/Jane Doe for legal cases, unidentified corpses, and as generic names on forms. It also mentions how "Baby Doe" and "Precious Doe" are used for unidentified children.\
""")


@pytest.mark.vcr()
async def test_text_as_binary_content_input(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    text_content = BinaryContent(data=b'This is a test document.', media_type='text/plain')

    result = await agent.run(['What is the main content on this document?', text_content])
    assert result.output == snapshot("""\
The document you're referring to appears to be a test document, which means its primary purpose is likely to serve as an example or a placeholder rather than containing substantive content. Test documents are commonly used for various purposes such as:

1. **Software Testing**: To verify that a system can correctly handle, display, or process documents.
2. **Design Mockups**: To illustrate how a document might look in a particular format or style.
3. **Training Materials**: To provide examples for instructional purposes.
4. **Placeholders**: To fill space in a system or application where real content will eventually be placed.

Since this is a test document, it probably doesn't contain any meaningful or specific information beyond what is necessary to serve its testing purpose. If you have specific questions about the format, structure, or any particular element within the document, feel free to ask!\
""")


@pytest.mark.vcr()
async def test_bedrock_model_instructions(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)

    def instructions() -> str:
        return 'You are a helpful assistant.'

    agent = Agent(m, instructions=instructions)

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The capital of France is Paris. Paris is not only the political and economic hub of the country but also a major center for culture, fashion, art, and tourism. It is renowned for its rich history, iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its influence on global culture and cuisine.'
                    )
                ],
                usage=Usage(requests=1, request_tokens=13, response_tokens=71, total_tokens=84),
                model_name='us.amazon.nova-pro-v1:0',
                timestamp=IsDatetime(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_bedrock_empty_system_prompt(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(m)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris, officially known as "Ville de Paris," is not only the capital city but also the most populous city in France. It is located in the northern central part of the country along the Seine River. Paris is a major global city, renowned for its cultural, political, economic, and social influence. It is famous for its landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées, among many other historic and modern attractions. The city has played a significant role in the history of art, fashion, gastronomy, and science.'
    )


@pytest.mark.vcr()
async def test_bedrock_multiple_documents_in_history(
    allow_model_requests: None, bedrock_provider: BedrockProvider, document_content: BinaryContent
):
    m = BedrockConverseModel(model_name='us.anthropic.claude-3-7-sonnet-20250219-v1:0', provider=bedrock_provider)
    agent = Agent(model=m)

    result = await agent.run(
        'What is in the documents?',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content=['Here is a PDF document: ', document_content])]),
            ModelResponse(parts=[TextPart(content='foo bar')]),
            ModelRequest(parts=[UserPromptPart(content=['Here is another PDF document: ', document_content])]),
            ModelResponse(parts=[TextPart(content='foo bar 2')]),
        ],
    )

    assert result.output == snapshot(
        'Based on the documents you\'ve shared, both Document 1.pdf and Document 2.pdf contain the text "Dummy PDF file". These appear to be placeholder or sample PDF documents rather than files with substantial content.'
    )


async def test_bedrock_group_consecutive_tool_return_parts(bedrock_provider: BedrockProvider):
    """
    Test that consecutive ToolReturnPart objects are grouped into a single user message for Bedrock.
    """
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    now = datetime.datetime.now()
    # Create a ModelRequest with 3 consecutive ToolReturnParts
    req = [
        ModelRequest(parts=[UserPromptPart(content=['Hello'])]),
        ModelResponse(parts=[TextPart(content='Hi')]),
        ModelRequest(parts=[UserPromptPart(content=['How are you?'])]),
        ModelResponse(parts=[TextPart(content='Cloudy')]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='tool1', content='result1', tool_call_id='id1', timestamp=now),
                ToolReturnPart(tool_name='tool2', content='result2', tool_call_id='id2', timestamp=now),
                ToolReturnPart(tool_name='tool3', content='result3', tool_call_id='id3', timestamp=now),
            ]
        ),
    ]

    # Call the mapping function directly
    _, bedrock_messages = await model._map_messages(req)  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Hello'}]},
            {'role': 'assistant', 'content': [{'text': 'Hi'}]},
            {'role': 'user', 'content': [{'text': 'How are you?'}]},
            {'role': 'assistant', 'content': [{'text': 'Cloudy'}]},
            {
                'role': 'user',
                'content': [
                    {'toolResult': {'toolUseId': 'id1', 'content': [{'text': 'result1'}], 'status': 'success'}},
                    {'toolResult': {'toolUseId': 'id2', 'content': [{'text': 'result2'}], 'status': 'success'}},
                    {'toolResult': {'toolUseId': 'id3', 'content': [{'text': 'result3'}], 'status': 'success'}},
                ],
            },
        ]
    )


async def test_bedrock_mistral_tool_result_format(bedrock_provider: BedrockProvider):
    now = datetime.datetime.now()
    req = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='tool1', content={'foo': 'bar'}, tool_call_id='id1', timestamp=now),
            ]
        ),
    ]

    # Models other than Mistral support toolResult.content with text, not json
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    # Call the mapping function directly
    _, bedrock_messages = await model._map_messages(req)  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'toolResult': {'toolUseId': 'id1', 'content': [{'text': '{"foo":"bar"}'}], 'status': 'success'}},
                ],
            },
        ]
    )

    # Mistral requires toolResult.content to hold json, not text
    model = BedrockConverseModel('mistral.mistral-7b-instruct-v0:2', provider=bedrock_provider)
    # Call the mapping function directly
    _, bedrock_messages = await model._map_messages(req)  # type: ignore[reportPrivateUsage]

    assert bedrock_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'toolResult': {'toolUseId': 'id1', 'content': [{'json': {'foo': 'bar'}}], 'status': 'success'}},
                ],
            },
        ]
    )


async def test_bedrock_anthropic_no_tool_choice(bedrock_provider: BedrockProvider):
    my_tool = ToolDefinition(
        'my_tool',
        'This is my tool',
        {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
    )
    mrp = ModelRequestParameters(function_tools=[my_tool], allow_text_output=False, output_tools=[])

    # Models other than Anthropic support tool_choice
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    tool_config = model._map_tool_config(mrp)  # type: ignore[reportPrivateUsage]

    assert tool_config == snapshot(
        {
            'tools': [
                {
                    'toolSpec': {
                        'name': 'my_tool',
                        'description': 'This is my tool',
                        'inputSchema': {
                            'json': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}}
                        },
                    }
                }
            ],
            'toolChoice': {'any': {}},
        }
    )

    # Anthropic models don't support tool_choice
    model = BedrockConverseModel('us.anthropic.claude-3-7-sonnet-20250219-v1:0', provider=bedrock_provider)
    tool_config = model._map_tool_config(mrp)  # type: ignore[reportPrivateUsage]

    assert tool_config == snapshot(
        {
            'tools': [
                {
                    'toolSpec': {
                        'name': 'my_tool',
                        'description': 'This is my tool',
                        'inputSchema': {
                            'json': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}}
                        },
                    }
                }
            ]
        }
    )
