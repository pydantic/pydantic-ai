from __future__ import annotations as _annotations

import datetime
from typing import Any

import pytest
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
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage

from ..conftest import IsDatetime, IsInstance, IsStr, try_import

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


async def test_bedrock_model_structured_output(allow_model_requests: None, bedrock_provider: BedrockProvider):
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
                )
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


async def test_bedrock_model_thinking_part(allow_model_requests: None, bedrock_provider: BedrockProvider):
    deepseek_model = BedrockConverseModel('us.deepseek.r1-v1:0', provider=bedrock_provider)
    agent = Agent(deepseek_model)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content=IsStr()), ThinkingPart(content=IsStr())],
                usage=Usage(requests=1, request_tokens=12, response_tokens=882, total_tokens=894),
                model_name='us.deepseek.r1-v1:0',
                timestamp=IsDatetime(),
            ),
        ]
    )

    anthropic_model = BedrockConverseModel('us.anthropic.claude-3-7-sonnet-20250219-v1:0', provider=bedrock_provider)
    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=anthropic_model,
        model_settings=BedrockModelSettings(
            bedrock_additional_model_requests_fields={
                'thinking': {'type': 'enabled', 'budget_tokens': 1024},
            }
        ),
        message_history=result.all_messages(),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[IsInstance(TextPart), IsInstance(ThinkingPart)],
                usage=Usage(requests=1, request_tokens=12, response_tokens=882, total_tokens=894),
                model_name='us.deepseek.r1-v1:0',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature='ErcBCkgIAhABGAIiQMuiyDObz/Z/ryneAVaQDk4iH6JqSNKJmJTwpQ1RqPz07UFTEffhkJW76u0WVKZaYykZAHmZl/IbQOPDLGU0nhQSDDuHLg82YIApYmWyfhoMe8vxT1/WGTJwyCeOIjC5OfF0+c6JOAvXvv9ElFXHo3yS3am1V0KpTiFj4YCy/bqfxv1wFGBw0KOMsTgq7ugqHeuOpzNM91a/RgtYHUdrcAKm9iCRu24jIOCjr5+h',
                    ),
                    IsInstance(TextPart),
                ],
                usage=Usage(requests=1, request_tokens=636, response_tokens=690, total_tokens=1326),
                model_name='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
                timestamp=IsDatetime(),
            ),
        ]
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


async def test_bedrock_model_thinking_part_stream(allow_model_requests: None, bedrock_provider: BedrockProvider):
    m = BedrockConverseModel('us.deepseek.r1-v1:0', provider=bedrock_provider)
    agent = Agent(m)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='Okay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' asking how to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cross the street')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Let me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about how')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to approach')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' this. First')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', I need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to make sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I cover')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' all the basic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' consider different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' scenarios. Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' start with the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' obvious:')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' finding')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a cross')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='walk.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But wait,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not all')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' streets')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have cross')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='walks,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' especially in less')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' urban areas.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So I should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mention looking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a crosswalk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pedestrian crossing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signals')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 first.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Then, check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for traffic lights')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. If')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" there's")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a traffic light')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' walk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signal. But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' people might not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' know what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the symbols')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mean. Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' explain the "')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='walk"')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and "don')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\'t walk"')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signals. Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some places')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', there')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are count')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='down tim')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ers which')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can help.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" if there's")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' no traffic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' light?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Then they should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' look both ways')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for cars.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But how')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' many')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' times?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Usually')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' left')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-right')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-left,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but maybe clarify')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 that.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Wait, but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some countries,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traffic comes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from the opposite')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' direction')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the UK,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cars')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' drive on the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' left. So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe add')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a note')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about being')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' aware of the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' local traffic direction')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', distractions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' using a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' phone while crossing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Em')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='phasize the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' importance of staying')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' focused')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' being')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 distracted.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='What')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about children')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' people')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with disabilities?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe mention using')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pedestrian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bridges')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or tunnels')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if available.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also, making')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' eye contact with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' drivers to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ensure they see')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' you. But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' all drivers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' eye contact,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so maybe that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s not")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' always')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 reliable.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Oh')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' jaywalk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ing. Should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I mention that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" it's illegal")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in some places')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and safer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crosswalks?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Yes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s important for")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' legal')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' safety reasons.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also, even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if the walk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signal is on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for turning')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vehicles')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that might not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stop')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. B')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='icycles and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' motorcycles')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' quieter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' harder to hear')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', so remind')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' listen')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' as')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' well as look')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

Wait\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', what about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' at night')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='? W')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='earing reflective clothing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or carrying')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a light to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' visible. That')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s a good")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' point. Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in groups')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if possible,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' as more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' people are more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' visible.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But during')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' COVID')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' social distancing affects')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that? Not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure if that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s still")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' concern,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but maybe skip')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' part unless necessary')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

Let\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me outline')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the steps:')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 1.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Find a safe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' point. ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2. Observe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traffic signals.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 3.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Look both ways')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='4. Make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s safe.")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 5.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Stay visible')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. 6')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Walk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' straight')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' across.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 7')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Stay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alert. Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' add tips')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different situations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like un')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='marked')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crossings, intersections')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signals')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
, etc.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', include safety')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tips like avoiding')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' distractions,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' watching')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for turning')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vehicles, and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' being cautious')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' at night.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe mention pedestrian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rights')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' also the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cautious regardless')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I mention')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' using')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pedestrian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bridges')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or under')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='passes as')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alternatives?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Yes, that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s a good")),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 idea.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Wait, but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in some countries')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if you')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have the right')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of way,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' drivers might not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stop.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" So it's")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' better to always')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ensure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vehicle')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is stopping before')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crossing. Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' emphasize')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if you')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have the signal')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cars are stopping')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

Also\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', the importance')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assuming')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that drivers can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' see you.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' drivers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' distracted too')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' being')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' proactive in')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 ensuring safety.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me check if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I missed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' anything.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' basic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', but adding')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' extra tips makes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it comprehensive')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Okay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=', I think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" that's a")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' solid structure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Now,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' present it in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a clear,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' step-by-step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' manner with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bullet')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' points or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' numbered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' list')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Make sure the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' language is simple')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and easy to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' understand,')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' avoiding jargon.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Al')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='right, time')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to put it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' all together.\n')),
            PartStartEvent(
                index=1,
                part=TextPart(content='Crossing the'),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' street safely involves')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' careful')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' observation')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and awareness.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=" Here's")),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a step')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-by-step guide')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
:

###\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Basic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Steps**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **Find a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Safe Spot')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Use a **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='crosswalk**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or pedestrian signal')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' if available.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Avoid j')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aywalking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=', as it')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' many areas')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and less')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' safe.  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='   -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' If no')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crosswalk exists')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=', choose')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a well')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-lit area with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' clear visibility in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2. **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Check Traffic Signals')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' - Wait')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' "walk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='" signal')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' green light')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' at intersections')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='   - Watch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for count')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='down timers')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to ensure you')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' have enough time')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3. **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Look Both')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ways**:  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='   - **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Left-Right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-Left**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Glance left')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=', right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=', and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' left again (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='or right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-left-right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in countries')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' where traffic drives')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on the left')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=', like the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' UK).  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='   - Listen')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for approaching')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vehicles,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' especially bikes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or quiet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' electric')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4. **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Ensure All')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Traffic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Has')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Stopped**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' - Make')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' eye contact with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' drivers if possible')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wait')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for vehicles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to come')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' complete stop before')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stepping')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the road')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Watch for turning')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vehicles,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' even if you')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' have the right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5. **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Cross Prompt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ly and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Saf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ely**:  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='   - Walk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' straight across')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='—')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' run or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stop')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' midway')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' - Stay inside')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='walk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' lines')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' if they')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

6\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='. **Stay')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Alert Until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' You')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Reach the Other')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Side**:  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='   -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Keep scanning')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for traffic as')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='. Avoid')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' distractions like phones')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 or headphones.

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='### **Additional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Tips**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='At')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Night**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wear reflective')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' clothing or carry')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a flashlight')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to improve')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' visibility.  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='- **With')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Kids')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Disabilities**: Hold')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' hands with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' children,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' use assistive')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' devices')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (e.g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='., white')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' canes).')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Seek')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pedestrian bridges')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or under')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='passes if')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' available.  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='- **Un')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='marked Roads')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**: Cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' where')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can see on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='coming traffic clearly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=', and never')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' assume drivers see')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you.  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='- **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='International Travel')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**: Note')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' local traffic patterns')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (e.g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='., left')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-side driving)')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and pedestrian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' customs')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

### **\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Remember**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of Way ≠')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Inv')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='incibility**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Even with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' signal, double')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-check that')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cars')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' are stopping.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Distractions Kill')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**: Stay')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' focused—')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='no texting')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or scrolling while')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='By following these')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' steps,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ll minimize')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' risks and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' safely')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='!')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' 🚶')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='♂')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='️🚦')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='')),
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
        name='my_tool',
        description='This is my tool',
        parameters_json_schema={'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
    )
    mrp = ModelRequestParameters(output_mode='tool', function_tools=[my_tool], allow_text_output=False, output_tools=[])

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
