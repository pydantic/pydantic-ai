import json
import os
import re
from dataclasses import replace
from typing import Any, cast

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
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UnexpectedModelBehavior,
    UserPromptPart,
    capture_run_messages,
)
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool, MCPServerTool, WebSearchTool
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, RunUsage

from ..conftest import IsBytes, IsDatetime, IsStr, TestEnv, try_import
from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

with try_import() as imports_successful:
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary
    from openai.types.responses.response_usage import ResponseUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
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


def test_openai_responses_model(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    model = OpenAIResponsesModel('gpt-4o')
    assert model.model_name == 'gpt-4o'
    assert model.system == 'openai'


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
    assert result.output == snapshot({'name': 'Brazil', 'age': 2023})


async def test_openai_responses_reasoning_effort(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='low'))
    result = await agent.run(
        'Explain me how to cook uruguayan alfajor. Do not send whitespaces at the end of the lines.'
    )
    assert [line.strip() for line in result.output.splitlines()] == snapshot(
        [
            'Ingredients for the dough:',
            '• 300 g cornstarch',
            '• 200 g flour',
            '• 150 g powdered sugar',
            '• 200 g unsalted butter',
            '• 3 egg yolks',
            '• Zest of 1 lemon',
            '• 1 teaspoon vanilla extract',
            '• A pinch of salt',
            '',
            'Ingredients for the filling (dulce de leche):',
            '• 400 g dulce de leche',
            '',
            'Optional coating:',
            '• Powdered sugar for dusting',
            '• Grated coconut',
            '• Crushed peanuts or walnuts',
            '• Melted chocolate',
            '',
            'Steps:',
            '1. In a bowl, mix together the cornstarch, flour, powdered sugar, and salt.',
            '2. Add the unsalted butter cut into small pieces. Work it into the dry ingredients until the mixture resembles coarse breadcrumbs.',
            '3. Incorporate the egg yolks, lemon zest, and vanilla extract. Mix until you obtain a smooth and homogeneous dough.',
            '4. Wrap the dough in plastic wrap and let it rest in the refrigerator for at least one hour.',
            '5. Meanwhile, prepare a clean workspace by lightly dusting it with flour.',
            '6. Roll out the dough on the working surface until it is about 0.5 cm thick.',
            '7. Use a round cutter (approximately 3-4 cm in diameter) to cut out circles. Re-roll any scraps to maximize the number of cookies.',
            '8. Arrange the circles on a baking sheet lined with parchment paper.',
            '9. Preheat the oven to 180°C (350°F) and bake the cookies for about 10-12 minutes until they are lightly golden at the edges. They should remain soft.',
            '10. Remove the cookies from the oven and allow them to cool completely on a rack.',
            '11. Once the cookies are cool, spread dulce de leche on the flat side of one cookie and sandwich it with another.',
            '12. If desired, roll the edges of the alfajores in powdered sugar, grated coconut, crushed nuts, or dip them in melted chocolate.',
            '13. Allow any coatings to set before serving.',
            '',
            'Enjoy your homemade Uruguayan alfajores!',
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
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name":"Londos"}',
                        tool_call_id=IsStr(),
                        id='fc_67e547c540648191bc7505ac667e023f0ae6111e84dd5c08',
                    ),
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name":"London"}',
                        tool_call_id=IsStr(),
                        id='fc_67e547c55c3081919da7a3f7fe81a1030ae6111e84dd5c08',
                    ),
                ],
                usage=RequestUsage(details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_67e547c48c9481918c5c4394464ce0c60ae6111e84dd5c08',
                finish_reason='stop',
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
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
It seems "Londos" might be incorrect or unknown. If you meant something else, please clarify.

For **London**, it's located at approximately latitude 51° N and longitude 0° W.\
""",
                        id='msg_67e547c615ec81918d6671a184f82a1803a2086afed73b47',
                    )
                ],
                usage=RequestUsage(input_tokens=335, output_tokens=44, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_67e547c5a2f08191802a1f43620f348503a2086afed73b47',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
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
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_image',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_681134d47cf48191b3f62e4d28b6c3820fe7a5a4e2123dc3',
                    )
                ],
                usage=RequestUsage(input_tokens=40, output_tokens=11, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_681134d3aa3481919ca581a267db1e510fe7a5a4e2123dc3',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 1c8566',
                        tool_call_id='call_FLm3B1f8QAan0KpbUXhNY8bA',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file 1c8566:',
                            image_content,
                        ],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The fruit in the image is a kiwi.',
                        id='msg_681134d770d881919f3a3148badde27802cbfeaababb040c',
                    )
                ],
                usage=RequestUsage(input_tokens=1185, output_tokens=11, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_681134d53c48819198ce7b89db78dffd02cbfeaababb040c',
                finish_reason='stop',
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
    assert result.output == snapshot('The document contains the text "Dummy PDF file."')


async def test_openai_responses_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'The main content of this document is a simple text placeholder: "Dummy PDF file."'
    )


async def test_openai_responses_text_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot(
        'The main content of this document is an example of a TXT file type, with an explanation of the use of placeholder names like "John Doe" and "Jane Doe" in legal, medical, and other contexts. It discusses the practice in the U.S. and Canada, mentions equivalent practices in other English-speaking countries, and touches on cultural references. The document also notes that it\'s an example file created by an online conversion tool, with content sourced from Wikipedia under a Creative Commons license.'
    )


async def test_openai_responses_image_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot("Hello! I see you've shared an image of a potato. How can I assist you today?")


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
                                id='msg_67e554a28bec8191b56d3e2331eff88006c52f0e511c76ed',
                            )
                        ],
                        usage=RequestUsage(input_tokens=278, output_tokens=9, details={'reasoning_tokens': 0}),
                        model_name='gpt-4o-2024-08-06',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                        provider_details={'finish_reason': 'completed'},
                        provider_response_id='resp_67e554a21aa88191b65876ac5e5bbe0406c52f0e511c76ed',
                        finish_reason='stop',
                    )
                )

    assert output_text == snapshot(['The capital of France is Paris.'])


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
                ]
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
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0e3d55e9502941380068c4aa9a62f48195a373978ed720ac63',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
async def test_openai_responses_model_instructions(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

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
                        content='The capital of France is Paris.',
                        id='msg_67f3fdfe15b881918d7b865e6a5f4fb1003bc73febb56d77',
                    )
                ],
                usage=RequestUsage(input_tokens=24, output_tokens=8, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_67f3fdfd9fa08191a3d5825db81b8df6003bc73febb56d77',
                finish_reason='stop',
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
                instructions='You are a helpful assistant.',
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
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9299, cache_read_tokens=8448, output_tokens=577, details={'reasoning_tokens': 512}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_028829e50fbcad090068c9c82e1e0081958ddc581008b39428',
                finish_reason='stop',
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
                instructions='You are a helpful assistant.',
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
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9506, cache_read_tokens=8576, output_tokens=439, details={'reasoning_tokens': 384}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_028829e50fbcad090068c9c83b9fb88195b6b84a32e1fc83c0',
                finish_reason='stop',
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
                instructions='You are a helpful assistant.',
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
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9463, cache_read_tokens=8320, output_tokens=660, details={'reasoning_tokens': 512}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0b385a0fdc82fd920068c4aaf3ced88197a88711e356b032c4',
                finish_reason='stop',
            ),
        ]
    )
    assert result.output == snapshot(IsStr())


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
                instructions='You are a helpful assistant.',
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
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9939, cache_read_tokens=8320, output_tokens=1610, details={'reasoning_tokens': 1344}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0b4f29854724a3120068c4ab0b660081919707b95b47552782',
                finish_reason='stop',
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
                instructions='You are a helpful assistant.',
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
                            'sources': [{'type': 'api', 'url': None, 'name': 'oai-weather'}],
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
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_00a60507bf41223d0068c9d2fbf93481a0ba2a7796ae2cab4c',
                finish_reason='stop',
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
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    provider_name='openai',
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta={'query': 'weather: San Francisco, CA', 'type': 'search'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                ),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={'sources': [{'type': 'api', 'url': None, 'name': 'oai-weather'}], 'status': 'completed'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=3,
                part=ThinkingPart(
                    content='',
                    id='rs_00a60507bf41223d0068c9d300b23481a0b77a03d911213220',
                    signature='gAAAAABoydMLww_4DcIPiCy5zW1t-Dtx57JodsdP9YkyDCvY9r0nBfoD-bwcBl8FfcFxRuq5nK5ndf90J6Xdxmvl9kGVbCUYSFaOd-kWeE4tgwXM8BwwE3jVs6ZMG3BdiWyXS3alnUO5jcE6kuXeun1LufAdZ6rWtJl3zSmqPycnTh9zoQ4cBBxLDq_qcVS1fgU4WjsWgjCZw6ZWRPWwZk8hywmU7ykCLH7SXItM4oH1m_GCEounJAS8YR4ajUh5KAdN6a1wngfGnXhYdzov98yiNLP6Nrkgr--K4vMTTqWXLTcR6fbWgkijanIeKmfSErCjMT6k5TrAbkFx0rgblHbdQii7zj8seV1BWZse92_k4sltxfc5Ocpyho1YSHhgxyGL7g442xMUEibjPCv6kwPMcW9yGu9wPMWsfPYCXpBbG6kQibQPNFJ_bEubwBRaqdSDq93Aqr1YkTYBja7Tewn8UfzZ8YYaGe5y_K4ZD47lfvDp019dOdXmOuZGC1ECRrMqKzSFYVG1CFY1VhjGdPmzobDoMcpZcLn25s1pg6lnNqNQwOk_IA4MvUcCU5HHD5YjmFkEy5-i_iRoDVu5coK0zyEMvPJ_h10y_ByszcfzS9e0ht5CSilckkFdxTBkZ5epp0YIg1e-PrZ790P-I35Ucquam9OXyULV1Y5bn9ohZa93Tv0JZRxUeTDG72_28xRj8tkJaBAZjoCC7VICw39KVmz-ZkuVN6IIX1WdNzyC4d808-2Tz4UZaU42-wxEWDnSDMD7iZu1Bi9fKKwAYBJt_OcEsJwpW63ZaUSG2PVFfm7a3wRcSMxMTUTTJB7L1Keu1hmNepif5tavn3P35nSq28D_IJyAqAgX7ZyROk2bJqjzSE4A0MddqAoBFFqKBi68n49KH09vDtDXIoh8jVWuIgowgVGr8pN3kuhLI9cir4Pr_WES0tPD7yWHPTzrD7OIJCfQbr_4Y4dEza4ixNi0RTADWzMUZBfr7bvwIsgvg6ZNuQlx_d71Go5VDsT2KI8H8AldiRvNWoLyYTFGyK9Kot97YsS5sEmSYgNAH48NU7pgnM0jNDQU1G39nTNFEjL_ziDwjDT5g3jm4S_gbQfwx-XFT3Pv-JYR-E71AqR--Lg71OsASq49rrlULfl5OENfiT-NB6x8MqnfUI6NpcCsOWLp8XfRbgqmZFutLIi43pcnxEe3cXHLWGF77qJXP6dFb-G5Ide7n9tAOoEgfsVu7hCDPEQ_xrIYRdc2DzDPUMCtXBai24E0AnQF8kxsEtlDW_YmAgGNTl9Gx0tFSGdDuUCsNx__c7v-_LOMWycXUKmH3iEr_su83oGIMapNp2PnLccN4iOxspdZQq0C6WBaR6SrdnGzK-0KwRPRoyKDLNWS8zfluR5bIgKlqd3Sbv_7eL-WO4LQXMvdKP3KS-DBt1HbA-gmyFW03iX2smPQbtVmRLWi1vG329R_07-tHMJSO9OQy6_6aiyO8Rgpbl_CHa1Q9BEkI2csonayDJRPvEXBPuk9-NPUP4VLNPB7npWBLlAqes5ZmhagnC7srTL0fFiLGLJiAxWo1f0BBiIlXjwqHdlgBjTw0KryCnEU8Ic8ATzrqEXXhs-FTBCcWInf3Bt5bzUhy20g7cTtYP-VCbsku-lXQ6wceWrfQVFtjKKICD8I4g9QusAIAvgCUm7J2rR3TLkzwOKngdTFPGQrQ1TYzlkA7q_Ew1uZpaPRckMaEioZYC6Sv_B0rgW0nyBJ0GLrB3AUN60hDrOFntyFHp0FM-Zh1SY-GKGBwZwVetOzM0ZAJ-NreFg1XVgyLTYDNjUrYJjRhr_JARsZ5t0pU4_yI6dPqM5jKO5_k4UpZspfQon6d2-NlWX0EDmz6G4CMTx0TScehYHrQZtPzpVnivc8h_pmXV3jO5GLzNeLWoB70SDPTETo1Of4txiEUaC2komu5B7MN9aR4c7VBOTv1NIjoiZcrd1HFACzZ7r1qAE-G38j1f1YhfZ0_TiMmtfR1cqjAKcFkyRM7rZMyMvvnsH7NFq59gFgWZt0dy0aAdw03XWXFNT67lrw58OYC3NcVozH4SKlmleu7TfjHNWSnJVjJ66riLn9DZWVxPeTk4zuISZn0yyaoXcdW8OMn_mJ9vP-8L1wElMyxKbtBRz-0cW7MshmJ3YXmHWDKbnqETSbDMtqcN_QyRJovopwlptJ8VzL7biuURRFw-l63Kc9vKP72Z-QWOUIPLB4q4nX4yb-IV0mkWFxIUlfv5Cze2anf7zDFyGzeU9xG0onfhJE4HFKcoUT8MzfrHZ0dDZtnEYeL5Xem3GuHpwEVGCxRE_J1joTmJfeWxSVnr2Vey9gaPmXCyRrdKS75v9xSXJFfHvcOO8Qp35Dzk-yFqL3dSOJfOEwDZbEf6QnV7VU1EhJvW4XmRS-wsRLMLCYcLrOx96NHEwb2h2l6gNfbCVJoQrMhMg68qBPnoSYLhML2ho7hWkSNZFy61yX5I-oEJV5XdtjFcBkyurmUD6uYTkJSqXyxLexQiPbT-uv49Yp9cAfFBG23sC9lUQ=',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=4,
                part=TextPart(content='San Francisco', id='msg_00a60507bf41223d0068c9d30b055481a0b0ee28a021919c94'),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' weather')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' today')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='Tuesday')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' September')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='16')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='202')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='):')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' Mostly')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' sunny')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' pleasant')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' Current')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' conditions')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' around')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='71')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=';')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' expected')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' high')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' near')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='73')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' low')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' around')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='58')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' A light jacket')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' is useful')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' for the')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' cooler evening')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='. ')),
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
                    content={'sources': [{'type': 'api', 'url': None, 'name': 'oai-weather'}], 'status': 'completed'},
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
                instructions='You are a helpful assistant.',
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
                            'sources': [{'type': 'api', 'url': None, 'name': 'oai-weather'}],
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
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_00a60507bf41223d0068c9d31574d881a090c232646860a771',
                finish_reason='stop',
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


@pytest.mark.vcr()
async def test_reasoning_model_with_temperature(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, model_settings=OpenAIResponsesModelSettings(temperature=0.5))
    result = await agent.run('What is the capital of Mexico?')
    assert result.output == snapshot(
        'The capital of Mexico is Mexico City. It serves as the political, cultural, and economic heart of the country and is one of the largest metropolitan areas in the world.'
    )


@pytest.mark.vcr()
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
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_68477f0bb8e4819cba6d781e174d77f8001fd29e2d5573f7',
                    )
                ],
                usage=RequestUsage(input_tokens=62, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0b40a8819cb8d55594bc2c232a001fd29e2d5573f7',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_ZWkVhdUjupo528U9dqgFeRkH',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city":"Mexico City","country":"Mexico"}',
                        tool_call_id='call_iFBd0zULhSZRR908DfH73VwN',
                        id='fc_68477f0c91cc819e8024e7e633f0f09401dc81d4bc91f560',
                    )
                ],
                usage=RequestUsage(input_tokens=85, output_tokens=20, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0bfda8819ea65458cd7cc389b801dc81d4bc91f560',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_iFBd0zULhSZRR908DfH73VwN',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


@pytest.mark.vcr()
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
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id='call_aTJhYjzmixZaVGqwl5gn2Ncr',
                        id='fc_68477f0dff5c819ea17a1ffbaea621e00356a60c98816d6a',
                    )
                ],
                usage=RequestUsage(input_tokens=36, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0d9494819ea4f123bba707c9ee0356a60c98816d6a',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_aTJhYjzmixZaVGqwl5gn2Ncr',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The largest city in Mexico is Mexico City.',
                        id='msg_68477f0ebf54819d88a44fa87aadaff503434b607c02582d',
                    )
                ],
                usage=RequestUsage(input_tokens=59, output_tokens=11, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0e2b28819d9c828ef4ee526d6a03434b607c02582d',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
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
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_68477f0fa7c081a19a525f7c6f180f310b8591d9001d2329',
                    )
                ],
                usage=RequestUsage(input_tokens=66, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0f220081a1a621d6bcdc7f31a50b8591d9001d2329',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_tTAThu8l2S9hNky2krdwijGP',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"city":"Mexico City","country":"Mexico"}',
                        id='msg_68477f10846c81929f1e833b0785e6f3020197534e39cc1f',
                    )
                ],
                usage=RequestUsage(input_tokens=89, output_tokens=16, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0fde708192989000a62809c6e5020197534e39cc1f',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
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
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_68477f1168a081a3981e847cd94275080dd57d732903c563',
                    )
                ],
                usage=RequestUsage(input_tokens=153, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f10f2d081a39b3438f413b3bafc0dd57d732903c563',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_UaLahjOtaM2tTyYZLxTCbOaP',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result":{"kind":"CityLocation","data":{"city":"Mexico City","country":"Mexico"}}}',
                        id='msg_68477f1235b8819d898adc64709c7ebf061ad97e2eef7871',
                    )
                ],
                usage=RequestUsage(input_tokens=176, output_tokens=26, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f119830819da162aa6e10552035061ad97e2eef7871',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
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
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_68482f1b0ff081a1b37b9170ee740d1e02f8ef7f2fb42b50',
                    )
                ],
                usage=RequestUsage(input_tokens=107, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68482f12d63881a1830201ed101ecfbf02f8ef7f2fb42b50',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_FrlL4M0CbAy8Dhv4VqF1Shom',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"city":"Mexico City","country":"Mexico"}',
                        id='msg_68482f1c159081918a2405f458009a6a044fdb7d019d4115',
                    )
                ],
                usage=RequestUsage(input_tokens=130, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68482f1b556081918d64c9088a470bf0044fdb7d019d4115',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
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
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"type": "object", "properties": {"result": {"anyOf": [{"type": "object", "properties": {"kind": {"type": "string", "const": "CityLocation"}, "data": {"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CityLocation"}, {"type": "object", "properties": {"kind": {"type": "string", "const": "CountryLanguage"}, "data": {"properties": {"country": {"type": "string"}, "language": {"type": "string"}}, "required": ["country", "language"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CountryLanguage"}]}}, "required": ["result"], "additionalProperties": false}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id=IsStr(),
                        id='fc_68482f2889d481a199caa61de7ccb62c08e79646fe74d5ee',
                    )
                ],
                usage=RequestUsage(input_tokens=283, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68482f1d38e081a1ac828acda978aa6b08e79646fe74d5ee',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_my4OyoVXRT0m7bLWmsxcaCQI',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"type": "object", "properties": {"result": {"anyOf": [{"type": "object", "properties": {"kind": {"type": "string", "const": "CityLocation"}, "data": {"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CityLocation"}, {"type": "object", "properties": {"kind": {"type": "string", "const": "CountryLanguage"}, "data": {"properties": {"country": {"type": "string"}, "language": {"type": "string"}}, "required": ["country", "language"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CountryLanguage"}]}}, "required": ["result"], "additionalProperties": false}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result":{"kind":"CityLocation","data":{"city":"Mexico City","country":"Mexico"}}}',
                        id='msg_68482f296bfc81a18665547d4008ab2c06b4ab2d00d03024',
                    )
                ],
                usage=RequestUsage(input_tokens=306, output_tokens=22, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68482f28c1b081a1ae73cbbee012ee4906b4ab2d00d03024',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
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


@pytest.mark.vcr()
async def test_openai_previous_response_id(allow_model_requests: None, openai_api_key: str):
    """Test if previous responses are detected via previous_response_id in settings"""
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    result = await agent.run('The secret key is sesame')
    settings = OpenAIResponsesModelSettings(openai_previous_response_id=result.all_messages()[-1].provider_response_id)  # type: ignore
    result = await agent.run('What is the secret code?', model_settings=settings)
    assert result.output == snapshot('sesame')


@pytest.mark.vcr()
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
    result = await agent.run('what is the first secret key', message_history=history, model_settings=settings)
    assert result.output == snapshot('sesame')


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
            model_name='claude-3-5-sonnet-latest',
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
                model_name='claude-3-5-sonnet-latest',
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
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='4', id='123')],
                usage=RequestUsage(input_tokens=14, output_tokens=1, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_response_id='123',
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
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42cb1aaec819cb992bd92a8c7766007460311b0c8d3de',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=2199, details={'reasoning_tokens': 1920}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42c902794819cb9335264c342f65407460311b0c8d3de',
                finish_reason='stop',
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
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42cd36134819c800463490961f7df07460311b0c8d3de',
                    ),
                ],
                usage=RequestUsage(input_tokens=314, output_tokens=2737, details={'reasoning_tokens': 2112}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42cb3d520819c9d28b07036e9059507460311b0c8d3de',
                finish_reason='stop',
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
                ]
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
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_0114iHK2ditgTf1N8FWomc4E',
                finish_reason='stop',
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
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    TextPart(content=IsStr(), id='msg_68c42d0b5e5c819385352dde1f447d910ad492c7955fc6fc'),
                ],
                usage=RequestUsage(input_tokens=306, output_tokens=3134, details={'reasoning_tokens': 2496}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42ce277ac8193ba08881bcefabaf70ad492c7955fc6fc',
                finish_reason='stop',
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
                ]
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
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42d26866c819da8d5c606621c911608fbf9b1584184ff',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=1680, details={'reasoning_tokens': 1408}),
                model_name='o3-mini-2025-01-31',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42d0fb418819dbfa579f69406b49508fbf9b1584184ff',
                finish_reason='stop',
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
                instructions="You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually update it as you make progress against the user's prompt",
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ToolCallPart(
                        tool_name='update_plan',
                        args=IsStr(),
                        tool_call_id='call_gL7JE6GDeGGsFubqO2XGytyO',
                        id='fc_68c42d3e9e4881968b15fbb8253f58540e8bc41441c948f6',
                    ),
                ],
                usage=RequestUsage(input_tokens=124, output_tokens=1926, details={'reasoning_tokens': 1792}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42d28772c819684459966ee2201ed0e8bc41441c948f6',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_plan',
                        content='plan updated',
                        tool_call_id='call_gL7JE6GDeGGsFubqO2XGytyO',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually update it as you make progress against the user's prompt",
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr(), id='msg_68c42d408eec8196ae1c5883e07c093e0e8bc41441c948f6')],
                usage=RequestUsage(
                    input_tokens=2087, cache_read_tokens=2048, output_tokens=124, details={'reasoning_tokens': 0}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42d3fd6a08196bce23d6be960ff8a0e8bc41441c948f6',
                finish_reason='stop',
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
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', id='rs_123', signature='123', provider_name='openai'),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_response_id='123',
            ),
        ]
    )

    _, openai_messages = await model._map_messages(result.all_messages(), model_settings=model.settings or {})  # type: ignore[reportPrivateUsage]
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
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='1', id='rs_123', signature='123', provider_name='openai'),
                    ThinkingPart(content='2', id='rs_123'),
                    ThinkingPart(content='3', id='rs_123'),
                    ThinkingPart(content='4', id='rs_123'),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_response_id='123',
            ),
        ]
    )

    _, openai_messages = await model._map_messages(result.all_messages(), model_settings=model.settings or {})  # type: ignore[reportPrivateUsage]
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
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42de022c881948db7ed1cc2529f2e0202c9ad459e0d23',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(content=IsStr(), id='msg_68c42de31d348194a251b43ad913ef140202c9ad459e0d23'),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=248, details={'reasoning_tokens': 64}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42ddf9bbc8194aa7b97304dd909cb0202c9ad459e0d23',
                finish_reason='stop',
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
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42de4f63c819fb31b6019a4eaf67c051f82c608a83beb',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(content=IsStr(), id='msg_68c42de8a410819faf7a9cbebd2b4bc4051f82c608a83beb'),
                ],
                usage=RequestUsage(input_tokens=142, output_tokens=355, details={'reasoning_tokens': 128}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42de4afcc819f995a1c59fe87c9d5051f82c608a83beb',
                finish_reason='stop',
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
                ]
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
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1493, cache_read_tokens=1280, output_tokens=125, details={'reasoning_tokens': 64}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cdba511c7081a389e67b16621029c609b7445677780c8f',
                finish_reason='stop',
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
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68cdba6c100481a394047de63f3e175009b7445677780c8f',
                        signature='gAAAAABozbpuOXVfjIYw7Gw6uSeadpkyaqMU1Frav7mTaf9LP8p8YuC8CWR9fYa02yZ5oYr1mqmYraD8ViOE33zqO2HBCdiWpOkVdNX-s4SGuPPB7ewyM7bDD4XbaSzo-Q5I6MgZmvVGWDGodqa3MfSKKNcGyD4aEfryQRLi4ObvHE5yuOqRo8FzGXMqe_pFdnvJXXD7njyfUofhWNvQPsLVLQFA_g_e7WKXtJJf_2JY183oi7-jNQ6rD9wGhM81HWSv0sTSBIHMpcE44rvlVQMFuh_rOPVUHUhT7vED7fYtrMoaPl46yDBc148T3MfXTnS-zm163zBOa34Yy_VXjyXw04a8Ig32y72bJY7-PRpZdBaeqD3BLvXfMuY4C911Z7FSxVze36mUxVO62g0uqV4PRw9qFA9mG37KF2j0ZsRzfyAClK1tu5omrYpenVKuRlrOO6JFtgyyE9OtLJxqvRNRKgULe2-cOQlo5S74t9lSMgcSGQFqF4JKG0A4XbzlliIcvC3puEzObHz-jArn_2BVUL_OPqx9ohJ9ZxAkXYgf0IRNYiKF4fOwKufYa5scL1kx2VAmsmEv5Yp5YcWlriB9L9Mpg3IguNBmq9DeJPiEQBtlnuOpSNEaNMTZQl4jTHVLgA5eRoCSbDdqGtQWgQB5wa7eH085HktejdxFeG7g-Fc1neHocRoGARxwhwcTT0U-re2ooJp99c0ujZtym-LiflSQUICi59VMAO8dNBE3CqXhG6S_ZicUmAvguo1iGKaKElMBv1Tv5qWcs41eAQkhRPBXQXoBD6MtBLBK1M-7jhidVrco0uTFhHBUTqx3jTGzE15YUJAwR69WvIOuZOvJdcBNObYWF9k84j0bZjJfRRbJG0C7XbU=',
                        provider_name='openai',
                    ),
                    TextPart(content='256', id='msg_68cdba6e02c881a3802ed88715e0be4709b7445677780c8f'),
                ],
                usage=RequestUsage(input_tokens=793, output_tokens=7, details={'reasoning_tokens': 0}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cdba6a610481a3b4533f345bea8a7b09b7445677780c8f',
                finish_reason='stop',
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
                ]
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
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=3727, cache_read_tokens=3200, output_tokens=347, details={'reasoning_tokens': 128}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c35098e6fc819e80fb94b25b7d031b0f2d670b80edc507',
                finish_reason='stop',
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0, part=ThinkingPart(content='', id='rs_68c3509b2ee0819eba32735182d275ad0f2d670b80edc507')
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
                delta=ThinkingPartDelta(
                    signature_delta=IsStr(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                ),
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
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                ),
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
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=5,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                ),
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
            PartStartEvent(
                index=6,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=7, part=TextPart(content='123', id='msg_68c350a75ddc819ea5406470460be7850f2d670b80edc507')
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='456')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='^')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='123')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta=' equals')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='180')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='302')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='106')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='304')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='044')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='807')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='508')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='140')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='927')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='865')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='938')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='572')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='807')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='342')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='688')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='638')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='559')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='680')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='488')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='440')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='159')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='857')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='958')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='502')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='360')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='813')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='732')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='502')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='197')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='826')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='969')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='863')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='225')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='730')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='871')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='630')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='436')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='419')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='794')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='758')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='932')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='074')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='350')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='380')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='367')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='697')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='649')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='814')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='626')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='542')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='926')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='602')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='664')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='707')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='275')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='874')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='269')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='201')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='777')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='743')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='912')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='313')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='197')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='516')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='323')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='690')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='221')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='274')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='713')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='845')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='895')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='457')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='748')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='735')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='309')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='484')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='337')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='191')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='373')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='255')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='527')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='928')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='271')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='785')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='206')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='382')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='967')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='998')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='984')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='330')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='482')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='105')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='350')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='942')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='229')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='970')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='677')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='054')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='940')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='838')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='210')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='936')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='952')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='303')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='939')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='401')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='656')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='756')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='127')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='607')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='778')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='599')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='667')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='243')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='702')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='814')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='072')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='746')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='219')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='431')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='942')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='293')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='005')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='416')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='411')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='635')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='076')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='021')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='296')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='045')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='493')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='305')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='133')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='645')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='615')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='566')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='590')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='735')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='965')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='652')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='587')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='934')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='290')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='425')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='473')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='827')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='719')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='935')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='012')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='870')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='093')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='575')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='987')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='789')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='431')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='818')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='047')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='013')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='404')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='691')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='795')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='773')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='170')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='405')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='764')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='614')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='646')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='054')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='949')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='298')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='846')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='184')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='678')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='296')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='813')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='625')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='595')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='333')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='311')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='611')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='385')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='251')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='735')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='244')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='505')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='448')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='443')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='050')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='050')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='547')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='161')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='779')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='229')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='749')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='134')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='489')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='643')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='622')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='579')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='100')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='908')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='331')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='839')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='817')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='426')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='366')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='854')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='332')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='416')),
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
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_meaning_of_life',
                        args='{}',
                        tool_call_id='call_3WCunBU7lCG1HHaLmnnRJn8I',
                        id='fc_68cc4fa649ac8195b0c6c239cd2c14470548824120ffcf74',
                    )
                ],
                usage=RequestUsage(input_tokens=36, output_tokens=15, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cc4fa5603481958e2143685133fe530548824120ffcf74',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_meaning_of_life',
                        content=42,
                        tool_call_id='call_3WCunBU7lCG1HHaLmnnRJn8I',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
The meaning of life, according to popular culture and famously in Douglas Adams' "The Hitchhiker's Guide to the Galaxy," is 42!

If you're looking for a deeper or philosophical answer, let me know your perspective or context, and I can elaborate further.\
""",
                        id='msg_68cc4fa7693081a184ff6f32e5209ab00307c6d4d2ee5985',
                    )
                ],
                usage=RequestUsage(input_tokens=61, output_tokens=56, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cc4fa6a8a881a187b0fe1603057bff0307c6d4d2ee5985',
                finish_reason='stop',
            ),
        ]
    )

    _, openai_messages = await model._map_messages(messages, model_settings=model.settings or {})  # type: ignore[reportPrivateUsage]
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is the meaning of life?'},
            {
                'name': 'get_meaning_of_life',
                'arguments': '{}',
                'call_id': 'call_3WCunBU7lCG1HHaLmnnRJn8I',
                'type': 'function_call',
            },
            {'type': 'function_call_output', 'call_id': 'call_3WCunBU7lCG1HHaLmnnRJn8I', 'output': '42'},
            {
                'role': 'assistant',
                'content': """\
The meaning of life, according to popular culture and famously in Douglas Adams' "The Hitchhiker's Guide to the Galaxy," is 42!

If you're looking for a deeper or philosophical answer, let me know your perspective or context, and I can elaborate further.\
""",
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
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='653a61',
        )
    )
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Create a chart of y=x^2 for x=-5 to 5',
                        timestamp=IsDatetime(),
                    )
                ]
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
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='653a61',
                        ),
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
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2973, cache_read_tokens=1920, output_tokens=707, details={'reasoning_tokens': 512}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cdc382bc98819083a5b47ec92e077b0187028ba77f15f7',
                finish_reason='stop',
            ),
        ]
    )

    result = await agent.run('Style it more futuristically.', message_history=messages)
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='81863d',
        )
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Style it more futuristically.',
                        timestamp=IsDatetime(),
                    )
                ]
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
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='81863d',
                        ),
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
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=4614, cache_read_tokens=1792, output_tokens=1844, details={'reasoning_tokens': 1024}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cdc39da72481909e0512fef9d646240187028ba77f15f7',
                finish_reason='stop',
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
    assert agent_run.result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='df0d78',
        )
    )
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Create a chart of y=x^2 for x=-5 to 5',
                        timestamp=IsDatetime(),
                    )
                ]
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
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='df0d78',
                        ),
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
                    ),
                ],
                usage=RequestUsage(input_tokens=2772, output_tokens=1166, details={'reasoning_tokens': 896}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_06c1a26fd89d07f20068dd9367869c819788cb28e6f19eff9b',
                finish_reason='stop',
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
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    provider_name='openai',
                ),
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
            PartStartEvent(
                index=2,
                part=FilePart(
                    content=BinaryImage(
                        data=IsBytes(),
                        media_type='image/png',
                        identifier='df0d78',
                    ),
                    id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                ),
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
            ),
            PartStartEvent(
                index=4, part=TextPart(content='Here', id='msg_06c1a26fd89d07f20068dd937ecbd48197bd91dc501bd4a4d4')
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

    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='68b13f',
        )
    )
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ]
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
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='68b13f',
                        ),
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
                    TextPart(content='', id='msg_68cdc42eae2c81918eeacdbceb60d7fa08537600f5445fc6'),
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
                provider_details={'finish_reason': 'completed'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )

    result = await agent.run('Now give it a sombrero.', message_history=messages)
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='2b4fea',
        )
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Now give it a sombrero.',
                        timestamp=IsDatetime(),
                    )
                ]
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
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='2b4fea',
                        ),
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
                    TextPart(content='', id='msg_68cdc4c5951c8191ace8044f1e89571508537600f5445fc6'),
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
                provider_details={'finish_reason': 'completed'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_image_generation_stream(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, output_type=BinaryImage)

    async with agent.run_stream('Generate an image of an axolotl') as result:
        assert await result.get_output() == snapshot(
            BinaryImage(
                data=IsBytes(),
                media_type='image/png',
                identifier='be46a2',
            )
        )

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Generate an image of an axolotl.') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='69eaa4',
        )
    )
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ]
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
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='69eaa4',
                        ),
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
                provider_details={'finish_reason': 'completed'},
                provider_response_id=IsStr(),
                finish_reason='stop',
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
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='image_generation',
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=2,
                part=FilePart(
                    content=BinaryImage(
                        data=IsBytes(),
                        media_type='image/png',
                        identifier='69eaa4',
                    ),
                    id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                ),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartStartEvent(
                index=2,
                part=FilePart(
                    content=BinaryImage(
                        data=IsBytes(),
                        media_type='image/png',
                        identifier='69eaa4',
                    ),
                    id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                ),
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
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_68cdec207364819f94cc61029ed4e1d2079003437d26d0c0',
                        signature='gAAAAABozexg98F3au1443vbazt85gDycVpBeC63gC3bUrFg_jpx3BYCksfaKwRkdFirkKpRHBYG1go0Kbzl_1w7ZqwT-BcPYX1Q7a--X9S-5OvU43ikr2kPbJgGhjWv1W37aLPKfIFOvt0VspdWlN3Diq6u67ktR0R0BXOFuSSv7WQmRAIl2JxiDzquSCb12Uc6i61LxNLpzalYk9lyW4Nm5c4czAsbUNv22AeTsB-nwWzO9NJyPfe6cVybz5gdShl8lKIvBsjHMonyiZqqzuSZakXSlbteA8yvZmOYO9XT1ruXNhCoFnq4pNuMHiZuOyCSIvASPUebLO9Q9uMv-t2aKku_4Xg4VsNUuQQpJfZ_gqYJyZFmANBTAcbMGBVotJS0V0ZkO9kyM9W99udKPuS-IzAkPCiHuBsjMaQzhM9xA2-uNm169O4P6TfqD65iCYAMLvwZbXauC6etI17If_731Oo0k4qcR5G7vruSd_Om-vwAGmb0u3_FNcI73bNw4WkhfGdUucfquaGIBkSm2_H63SOfLHOiLsnLIZT1QkEm2k2LP6O57WHAE3H1Rd1nKkw7GQcY-oQlF9_AGi_okM_nCNf7RXzfEocRYXlkSwoJT67TGV_rp1aSzhxKgzOzntDlcOT4FdyVY_0MMtxnBANZL2mBM8nvmqogWOnrJui-7qmFYP6DJj-GkoqPhi_gXlb9gTHtus_lQtLvJSXFhDlQDT40dPbU6T716Bp1TRQSAtj5s4aIhEupcAyoza4Vzko9WF0lrwyTKeU0Yh2nkrieKNqL9XbhknW9bp8dRMRM5-cG4WZHWgD5nLV4EP-8xcZc0ApwBLebiUhkX3MZ7phpS_ZpwnHxO8GDnvuPNT6pXbvvrUL-Vitwe-qWoodziaUetUcopfpGlBYnZb6_cJOVFUoJDxPxgOkLiUW0WSSe-NsU0XllTG75sVx6cciU7LbDJjkPU3yLOrJcwpE81Khsefm-yyk4BOMzPbhJKg9Lrs9pZmuhXa0-GdMP6fHks4UKYeR5pX7zTU36t4BUYFssq-p8QqSuXah64S50W644yvaJR52PzkjDhNtl_lBbLRsVvlimUz2vGuJfFrATSnw-pkuGEDznc5BfdjqNDxncVWpwu0u2HnfAXbzBBtgNFVvWUegjRhBAwqrlWdjhvTcfzp2xUAoaXcQdRmZUh9Cw1Ib3pvejHk8l56i2Gf5X0VrDVxj2av3LugJ0sDgahri96q_EfYVD6TEF5D1lz-GBFTWJ3CZgdICN1HM2ny8yNg60U7EWFNqN3Rn9wx1LnntWcKxmwLRfTWFr3k3-TTHGam9Nw41T8hSkpGmpDYM2LsOcqdGa6ZYlFEBstzvf8JmPtBxIlD-nFIAVLo0Y4lFmt29x8hi56Ybu3_xdwY63SMj0rydOF0bX1K1QXcIJrSY4joCesFCJ05vB9F39HNByztFA3cBvyZkwzpmif9juYOpM2qfZVtG8oCnaBU90u7lkIvfgRd2H8lH0pnVTLjSq885mjhxRad1pnD-0rIoPY7ppz-quELsMOVqNTZWW8mkCy6uKiWna-3m0THLNz0CNFCnIz5ZTmchoj0x1WewlSamGL6n_SnB0H_lIXq9RUZlwrd1ASQtiOnI94pBRHPXfQ3a4PfNyeZv9GOKHr4pR0uHATkCd-rldmWQ5kG6MbNjNMJChaCMuSwrJiBondTeGIZswsMwOfV5eHgoZwfzkTxD3xo91CLGsE77IDjp05mNYc_KMbEqSmnDWpPJzsHWAX5vahfVXsRFx1Jjw-D4d5PIU_Mx4mwjPLjTrWkd5Hl2uKWvjw0Hb55lHT8D3cGlvIjRnTGM6ndLGndeKjXobs0NBcmgDQNLpHMccHAbb2UVIdvx8cF3zhkc6FdKJ17TZGJQbzWSqkkKwNUCYuRrf6CeHoEZh2ErArmcj57QVdXeDREoSzHhu1XRBfgc-ey4WF2RMzJ47Y0LNxFHpLxEk4Tpg4jcMvfRRKv-8tIu5kZeVSBqZTac90ly9qDeb32mv131YXy9G1A4RCLhjCoxM9M5r8a3AnDIvoMCusPxAM5mHpLEdv8A_5mliza9hztzmDKIQq7sQAow_XafhRJfX0dOeOft6aLkLTaq3KHN1HE0xy9Fupg5ut_AUd8tIvst7Av6vHPiK7gR-rxgWNx2FmKKXhvcWqMeeqFwSyq_18Y4w3ibPc-l-oYXD5gEMAaXGa6WgyaeNBSHNup7HDYlm_WLK3UdH-5IuIMxkhLFgDFED7g9lFl5V0I4bmW8Vqg0AqmNC2hkADsr7mNjuYr3NOzGCTsE2Ed992VG_e3k2PqtHVegZ41jxZTbfqYRM_HYg2rOdnzNN_NV3w0hpKmV-F6UIrb8Sfpscdwa2kz9Zz2huG_QGzI6AyjoepNgOYYZ_Phq4JMtrjge8HcOGRR3z8_hV21N61o5Z9yBfwx5kSnAvPXuAgeuh2MIfB_koOrwbEryE4HJTSA9b2IAs3BS9cvAu36VHVYP7Cj-U3EZxaxRIqusjNHb3WIzswa4AXFfjK65aaCCizHwxTSdn53nlSe_f7y9pMnOV3o467Vtt-oR9dqz92pauT1MxXYdBD8o0q7iOgHsqdcHBDvQmWOlpag9W1sF5uEyzA9bxDGlKLA9xtvfwnC2Y3nv9i-haFaoknx8eQGsM_bzWC54o6VlhJiNeeRUuTNEAsoTXv13usbK6MtSXoGvWPNDkj0CJ4sluSWmOidGG6DW0bEj58HYCedg71EKyNcrxcN7JJFklv1ju6IVY4k6pLMPu6kbYdKzXaxFWkOmB27Qz0dgBbK7hHhHagJWjOKrHWNiOphce9SHf9xQP2U0uJRNKqNnqmKRZLBQ-QcwRUxzaFrDByG7nRV0ybRHpJvrRfCjq-FVF0b8JXudNY3D2gI4p7WJeBqk84bt5a00DzmTZap1cEYi8VYs6zwFRa-LTemxgq8jOaeR33x8I8tDBbnqJdOV_o88Mek2E0mXkKfl_3ZzPzajcjyNHYdfhNHfcB0Oeyk6w_TXnVxOJVHuoYUsXuB4gDDMENk2yubSUFjOUzuZhXXZ4CdE4bymK54tzj9qATV0c7k-YsRlfUk5fX0JRQF6iMMWl0VtVgi3HolN7S5K4Tbs9tB0ByOS5H5N0D2BjlDl-kmd9Yrn6-B7Dxz2rxCtj_TQneCyMzfb6edt8X17c-iLsOV-fZOP3mFTCcnI6HyMxaLQDENgPF12gW3mDZyX2gAd7iRFXCHzmTnVkUEBhlCoIbq5Dc5s0niA3Jq9MHvhtgcXaUusGqCxOtQ22_CWixDntwS3hQx1KxEUzYZxThNSv-f5ji56iiWOB9uVzv6XLPdGYkN5CgdKzqPsim6VWfz52q2TTSZFRuaMGZsJuEpprdcWpHcHiBZsk3Qa-giYazywLHxj3Ph_Kf7kByYuzspVpGX3bct_unkM8jQzg8AWkn0LuFIY8ScyzXAVECJbFiOv4HipzyoyYcpDiPmadV-qAG8ghpw90xwcvgrGnHm5pzHQaqZVPKWXrHnkaJFgQ7OyEpmX-U52R5t4PZCeWJA90zAnvmL7s5MFEDLzSlzXPFjzUo3T8XWz6p6tUksLsGdUUqILT7hYdWwCdDnUJYZvjysEVIkNPNx3lAGW_e82KYUZUjQqiQqsbu1meJkdZdvMPmOlVEGx9oFvTW8cZrZZfJ1EKNOPR_1eyGdh5ZOfxGU2sogpOST3U6dxdeUwpEdhl1dTaZA_chepczFdHI7aMoKSISbpNC7nNhKNfQIpBcwPLtAhlgCOQKuBRMJUPK7VcOpl_nhz3FKexgJ0X_hFqN5qygPtKITZ_Et5ME9k5tzelHVAiosKhWOLngFJzKZkYFavSYUOHemVmUo7licjeUTPAb0d6kKbqqrJ44ZP22gL-Rw0YZa7t0VM7LrD4jD_taPhqmZ28CCchPAfP6Umo6txaiXDusvPG7v-zUGh4XWQlV_5SBSQwvPHUfQHlogHH69jxfJYRD3F-lDIfKR9ta8RuCuCdd2VsBUUAWFaRuulGcLiQ1BFWZ2Fx7W6rXgW8QkufqX9sp0zgTK9fCPN0rPLLU27BUwn0EB3EZe4bDBYrm9xAxYCgoOOE28GMpZsznXXIVaAXvUGVTBnXks9ycNghOw35fsAniH64H8u7Yp45JH8BMb7sk4zHcW_An1TzmUxjq3_JdOo9kkzNEu_qs0Y9btf4M6NajoVWlttXT2RD8XPxIrASjFMv8fxYJG0OsVHJoKTnGFlHqy13dIDMtfOkT658EuGe_pHwp8B1zI1jjpPmMW2MG3TAlumFgy9T0iyw0V3dp6qCZCjavMPIJ2HD9V9TJYueufIptuN1_hZF-xESc1ENH6z_NtMBJmW0hwCU9a6UqZ-K3_ZsJ7EK7q06V53KOSzzaJUY7SgUAFgwfbqFpvqi8emyHMxxIFylw_E1iauYGssBb_tzY1fTWwk9n4h4A9bdKesz_JBNRuKmKyOeYOS2LdJGhp9VOsbTvJtJ3MvwXuiH9rgFTXzXbDgIHxSUnUkMo4mJhnWXiqENnfqc_oOgx6zkicwwX05Bjn_k4nj800fT4RkJTY81dXy54TbvqBgPR--nTkq-WJD0e8_9hjOrtD7c7_p08FhVGWl_d499ylunVccNQ51zfeP0uGt8uYK-S9fNN8ED5PJ5iBycLRU0uvyFdsZFS0DJ4N96xSrdkdPAZ5CjfIn4bawnCs8MrHg47sCgONAqrZ1WZTgyL8j8J0kdfVJTZoLRud4oFmM4t8nQsL1sxH322EWVaXsy9aWAMVWGcc2fesk4jVODZ4mfGtnvm-Q8ifjV7wqRa7s5uXv_-JFUawxNvHqMw2AvG1M0RJTIEnRkCKB_8CXNb4K-GRw12XtZs1KRED6j1yPzLwabcWNl2BpR9JCBmogxFFuPPLdevD2JWptQ3YwTQSIMridFriwY8g31iFMFrNub9Z356Vc9zyMvZ0sRBB4oK6iSy8Y39nU3512YdGpDmMTH-T5Cxy3pi1dphlbYHcar1Yl-n5gSS0_dezGILZbxfvGbxOAIL-AGwSE4JJpUFn6N0xRclaNTPDk9VkKpVktgJeWqkWFI1xdBUp0K9BQdNs6XsJsfMCaIE-ed9SbQpJRdc1S7m6lFKFZFvHujK1zplJemcpOrYIpsuTD8wzWJAxzWwtUoo0DXi0YQOrnFdo9PbcTscUGyNArCcGjOYiL5_oMwhJh5kg5L_OPLH3UhiLI_aHs4gT7nX__HHm_3GVOov7-kK_Rs5lGsZtFQ-yoau74FnzVn8N3YWMqWzCVbwGRBhvAm_fdI53h77CZeJm9yWZkWTzrc0Ua2HqA7KB-eZgCNx9Ox9n0Ilh42hdGLFSwOEF1oEHw63x4ZTENyAP7j7mHWnyVbqRVSZOlP6w8JquYAtS9hUJQb8GZs2MFuFcwTO8SX8gMWWVM3tGtpnz7UwqhcFMZHrJ0ehues8kdtrlqR2BWPYAquFhOQebvqhlYxieuLggwAIxudJr3PFYJrAgAxivKZlhKRSoa1HCFSp_4k_kuVp5AvfBMViLJ39pi_xHX8PnokEmi3_GZiz31hGjj1GgNg8HZCkQtQJcOjMA3YPE3kQHy76YIdxzL59EdWsfuHj9IhJ4J_PoHTg5k683yXkvcXsz_sOYj9j6eMNn3Krp-ZFmWbtmqpWuvKS7vmE2Wtq9RmzvkGPC0jeY6J_ndbyCCljLpFGbvSYC50ejEulcMSUw1ypHhHzoDzO1jFo1xBDfta1H_hAsLkmYKOseTZce6hHIZR5xUwWStmu0y3OoOeVT3cZXEErr7qPLxEzB6nTlSgzLRX-ncRCnnUy0f_pXWwBE9dE6w1Nq4Y9BuS2Ip9oZZvj-gRCxTmmJLL8RYd-G9EeiBSlf09kezq4plUsJwvvPqVlj5CVLk6DkEsTvAM5r8DK3FYnLUgf7hQI_EZ8P9VN_rrPTPa25y19MR1-XIsVGLrVPmyXcj_w6XLBxEc1J8qQ37NZCw_DRfS0_ZUUqSDNlIy-y-DsdLIVtXE-BzpUkgqx5ADmbsiFFj87rS7wE5Yi2-btSpkWdTjaBbsWBgN3sOvcZgFt-AQrCK8BMLENwFrkFicExianGcwm55IlO2iUvCNjz84ogzU0IYhzDbHGCZBzYaLNX4JIVeonOoQuKA6QGueCffX1MKQ8fDKVZQc11Pr9rwjPNR3xxAs0G4ykW0c3809YE60dLbP-AZ8Qm6vxvAmTzd3XlMFaC-KbBxskXtMJRJqxUd5jYx-YQMQiOA8wLytP8INXSHi2_lgMI8QivRHoKx4oAl-qTx_qz0Ut4iw192OVWoJqTjhwIE_NB1Um9GawsbM9gb8VgI69RTPmQi9obao9kkQEUU-NrNtDkumrltR9Q9fXFyRbKn5Hs0PHzC2yiRFa_IqGEqJeqz2CSBkdC5GauukZj2XumrXKKCIYV2vJ-42D6MUqD138clb-prSiOe4dA2-wrKPjaaWdpaMeAbLXbLkHuH4K5h6HOpYIVMu_JWdnzzDnvye0o8A8vrxFZsOSDeiHPcoDyN6-6KYQwOlKpjiswFVdoD_e3Aw9ge6YHDOggsKeyERzv46Hc2GwW_GWrwaWgXSv-lePsj94L1b17WG_y4oEKbNMAmE4A4EBH1aJI6_wEeySONFKhbyOIwUBcRKCoKT40dbGie7kW23rrzSTQHpjdd4NbAbNAGEtbAJbVwZbFC2RMwMWQbz8mmnMSdVw0VN6UVaCdUatYzATx-gk-lg_8ikyOhMwA_lXFI9pfnHIIqGmKHDJdrE4pdDBK5wVLB1H_CYqzUneygueeTUc_rHiMj7QdekTGh4KnA2lurqXemNemePYl621G_FqimReuPlTPmn88h-DvnmcVhTJEMCT2O6huU2_6yOlyulJw-mq6KMS_b3PSB3bgYmJEH7EM7ljv7lu0ZIqZm9xbpVTntP2s9jDBPMb95q8AnllIwH6uZEYdyiIDU9NQj9RIlR4RZ8Oi5b6n2IuKojvP6aK47npwKMvC4IOAMmGaE3_S2X0vFnpT1djiFPAjvklOEA8o0k91YV_z-QbWwVTd5fdRgaqXcqb3dLAuvnFq0bOpJMchlL9lrC02JW07Xi9Lw4EIoDsPBRr5s5DWLG8kLfpEv-YNZatryAbjbwMtJX0O9aNxLzG1bzyXVFwBb_HUgkotdbmlT7cXMMX8D4_xIQcDyjwo925uAVU3va121ddHt2JasKJc1vMyuOKXuDg5F3dYByqEQQH8zywgBi_IRtUx8aXNoNXobUf3LGaLc-3HqVTZby9Xv_Axm-s5jSqcEPL4WJX60rY28qx_lmajcmcVLY7irKovGmfEIlVeHYdMxDxZue2nHHHc7NmF7dGORg870x5iGUcqtmjvOx-NIqVShVQlalGOdIF3_u6r8xlvOL8xWJ6WLXKNaAFVtaoDdObEoR3stpG8iaaaGNcrRqIH7J86ntBUmQ8chIgOOvPz81W0Op5jbSc4d2e_qAAuvt6j_qElh9qsGNMVPoY9DlxqqrlJ3ojG_6yqlX_jP6NGSx31vxc-IgggbyeEm2BVt1v3dJv_Z1Kmt8PGa3--atQsH3cemYMbWcwX2mAdMCHLY7xQJs_X9OgDaPd25bGPHFJQVNM4cypyPSWXEDiR-xIqkB7NzTrSEGf82wSgIPRAo7k-CBpbBVvsuLre4zof8K5u7QH2jYFQA0-9YtDa_Uyc_JcErLU7AroB5ZZr7s_s9qx5w_wJxMNW1NLZJdOKUVryP6DJPqTNmly3PT88GaPUDdrcQa7NplSF1FcqFmzOlZc1Se_fSkgEcFKrX9M7_65bTgLbAvj-8bCTIxZp0-PcmoQO4mrkNkW2Vr1rfSuOH5_vqJCT7s9A-H77RjLseMxJ27YF_dEXOHjsEV0DPM_zfno9_Lb6gL0rqffxhKdM_xRhSsLg47PJlCgoZkL2xVTJfahwydAzbr4ZWbZ84WwprOimGLJBdcQgjHEhxx0_cT9tFpUKkUbpAlUYqm1e23JGUXKrS9kBqM68RjcBAiaAgeKMx_5pNU4RzuJqCARPYgqhsy50LxkN6DgIMiDglLxho4wUYHVVkez3XXPHmO8FJcRuuk85Z-7qDeExR-Dp8vHkAvSGuyuUKHdyaQdhigMm18qFzu5-L_m3-C7xEblO7KJBDtK9mDdaviyLU9bQA4i7lLgUUdkCClPjm7yDt3SgjDZKXpBeD_HEb134vjhc6EVJN0IK0bf-lSswaAPJYcGc9ArU_mch_F1w0X1ISwvFHWKLYBM1bUSCr7vLyX-j-sbwroVSJ53bTedlaUpoVhEjKxcksjSgnndvAeeHAwHV-HCkOH_zqhxp5aTO1vNXeNESG7WoSRLE1_6vF7ZzTGmpG3GIbuBexi0BUe95UwrAJpdtzWMrAiRNQ3LlB8JkPIqX_ZXmHsu1bo84_x1u-VWQZFNTOitRXyLdA9GtDqg2q7RUJLj1Axq6CzjcewoW380K7r95Kimr6RDUUkvi85SjekZuV8NNd328heZWNSdov9lzmJlCokX6N5VDpg-NIPvjJyaQtf6YZfYtOk7pp_koGxatbsxZTSVh_DwXz7K8VfC2mEhr_xdDPe9nKAPrU4r_J9zdtAkkzdJssMWhSCMTj8z3l1bbIzMt05ivwEGLZYqZ5Sy2-duQVd1wA0NfKB2HjfgSyYhX5wN4aDW15copsEtPTrxCLidwc75rLdoi5Ch1Jt74v5UzKh8mxDaGqjdWeHkDrHbVy5hDVk00n1TZ5UAzWCq3lge3717y0ECZX992BrkqjffUYe0dZUUJr_3GWKSS0AZHg5uI-1Tka_DGqF7mWwdz7XpafzU4siolRkGa3QYmB8LWqdbAnpvte-uSwxAiQm2LiFF9h4dOO8U_2gNsniQViwLkD0KTHnuk1_N94P6QsypEL2bcPWCvCw1SgMmgKBaXYpP8FIN1trlpqyk_0abqnq9GoeCI2AjAOkHI0LnNsM8lTwWNqh1b8YVco3or8J4dOPeVQ=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_68cdec307db4819fbc6af5c42bc6f373079003437d26d0c0',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='c51b7b',
                        ),
                        id='ig_68cdec307db4819fbc6af5c42bc6f373079003437d26d0c0',
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
                        tool_call_id='ig_68cdec307db4819fbc6af5c42bc6f373079003437d26d0c0',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_68cdec605234819fab332bfc0ba35a5d079003437d26d0c0'),
                ],
                usage=RequestUsage(
                    input_tokens=2799, cache_read_tokens=2048, output_tokens=1390, details={'reasoning_tokens': 1216}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cdec1f3290819f99d9caba8703b251079003437d26d0c0',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text or call a tool.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_68cdec62725c819f92668a905ed1738d079003437d26d0c0',
                        signature='gAAAAABozey1vHMbs_Q2Ou1f-stI__3zJS-qTasRefyc_eyOxqogM8UGbPpL8D6PLFHcypshJpa9SQli-qZRIyG4ioUDLsKpBwFbfjdIhps667-8st03DRTRP0ms2izupS0ae6QqY9qrsPSchrSF2o2PlJOWKZAFJ609S0hGX8VDtrU8nESfp78NQ5HgpgXksXQTxk3_xRmXES2AThlUD0LYykoVKRX-xOyPQsOK7aDEs1CIk3lG1meiXdtJxP1Jm9JQGLWk6kePWUgwnAs818LMVvjcj8GWzFjxKUQlI3S855vYngivkMqYqh4gOcDRGRWej4NRzRmhOK-2yrATl26qnpRwNA1YXkFtn1ojxEkXD99P8RIXNItH4KW19ALs7ZizQmQlKzd96eyPT16OSLqEIfHAXWEKwoB2vTM2ExvHK4il76X9XmgRDy_CI3HAPI-7M3787MJBEY3z9cBe2sIS_GtSk12_GXRBUREhu8wcc4920FxkufYegHd3FzKxBRjyxGpR-jLyI24ahOZRKvoXi4-n1v4umoD5OSMjYpMtr0ykwIBQyyqldi9KqHBpJCzB0wA3JyAn-4JvQsXwIeeAtq3bNSJFaaf9aLJ5OwMO9I6IIWGxoQ1mzqmCs5cVwwjeJLzEc0T5g2qWJdXxdYmjesvMj3pJtgIq3iR2105LydhUiKE-0VLVAQGg-lnjkCtj-muEqlko2_FCHQ7b_hA0VkOUIKOYUDHRtwgtaeNnUpiWk8L7GnBNHtVQ7_kHEGj00UIVC4CKiJqESXS1om73Xt1K1-bglZLfSKfjrAd6E3W51cKQXM7KOfmpRwP-9DThdeOBgjlmMFveru6NYl2ntiu7GF8JJAvjebF3Q6SR4AtFSp8zrZVjlduW3hzQtKaROHLBVgH1KaMST-Nfnkn4AHCbhYNGSZxg4J8M3hh-BLba-lM7o9d0cHsHSORXeuAg40qioVCZNCtIooo6fAdWSAULw-uGdAbRbrJq5nE-w_Lilyeb2mWnMwMbKBHjQO7Kwe92UHvve46vMkiSSX-wZYwthfbO9BM6_ha5BJOtwNggKuBXxqMVizL8WKdvdTVwzP1guDvuVgVoCYKl340jB_EE2V-L-YzbSdaxHi09Gi6E10MgdaSGhNqUJMZWXrezkT6pyRYYRIWhaaImIuQf6JybMUH5hHH8DDEKvofLnQmgPVccU6womuYosIgLOLPOetK6OEFlMsQPBn95hb6jY5vETMyhiVYAVxaeXgk5WA-NdYQ-F2q4kQQS6Ku858AaO9rXMYpkaJ0nIubIbdgQMaXzq6ha5Z0BaxrJhxCDqmHXA5-INBBqWDw0AgJcAlMM_a5ShZA4zVWu4ydzikq8PlnATLbFzOkL8WBhJGRyveKaSyknPfPCBvXu_1l-hwAyvFv2dWWBloD_IIJ3ID_qtjgT7epwC88qmQd5ICPMMdx4DjEu72hU-rTIz30ks970qi_dKVbgpsLAJbHeCCWXBJzJ3UKrC6sc7Kj3rhtelPNEJqFyuB_EJOVX4o7R0AZvkk3wWs3IZ2nE9TeuYckw-hHg35YsC43QSrbIZUbfonaN57ZbwyrnEwnhH3oYXaMiPqqywH0CsZYrO0QLuAeGJTknlpWHgLEVmz39-e3UZFb6WIGIfSFaZHAFpmQBiPjae-qudcbvvfmAgBHqt-Aq4D_06ySnELOtDWlFusZcmHQwG9b0OCDXt6KRTR_-49uuoPCoXlv2nKb2eFhXp1gp6m6WsH21XNaeLU4RF4PR6oRUh-TCLzyBtwCscukF_3gBvcTdwJ4io0Fu6YVtEJux_Ec1vCaQHlUVGtZR7JDVyE5lu1y-aZx0u4s6HheF0bHLYaFgqgOmaNNWwK_jldqp99ZhU7Qat8GcG8YLLEJ04WDIp6_i_Ri7OUf5xgTEkAxS9gOxeJ1EMKR2oB0pf7YJ18XkNqeA7zxXmufJNIbEmXOC3XQKiDY9-2UzTyqjzdZ4V2naUggs7DjAAntcHhVFLGOZgGeQ5FLJ9jfzFlpE8mAg95ZtVvPzYBNFaPoTynqUlukCH4eje_62w_u2TruBMSU3cOV5IqVTLMHu2uwxHWdA1zrVi32LMv8FEYZ8nPyyk_BdapV-SRGQKn1yjGml__I5ksVlqNWVC3BX4hkIxH9K1bO6HjWP2-cdRtTYtRNcOOGOZZv5RtKfpvzjIK5o6d45KDK-jp39_cY9Veyawzc4XwT7jkyL8U0YNsRTEjafcPONK7yWasrOIzNuUppBFdMyER_R8Q1bTMQp1sE-NoAN-0MqupZe1jltzga6i6KLWuOXtMm1_DeHHH3OPNq-kfVs19gbQD13R9kMNjgu8FbAWdoVreG24tKUVSf2nWKReXwxc3WiiODaYTew2ynZ3BUchm3eKebybh4lKoYCw6lLoclnD7smFkP4-72RfaTylVe_npaWU_kWIhxItYosjWGc_ScJhLFdwAOUaNijGNX0TmeMb3uaESw23E3Y0M2pAC_wVkaHvJEYv_nYebwYc6yON5oMFAnOmqaJ14d-LfhuATtTrqD8fGXTPi73rpN5IpA3fVklkyUuu0GWqvsRujNnEcN2nL43LPRwFvwYSeL_tXJoTpTxdgkwjEg4dJr6hImPIwk6Yu_a159LyNKIeZOSZbHi0gao7OeDiSM2_1zb_srjUXgiVPJ26r3GCz07dhcwqUS8lU6nx5q1ncCt9mjTgUG0qmVPzDjfRmeqOU1gPZhh2XQeiXp2AWB-_9M9vu2EiSeOO2gfYKeaFjVhBLbjOeKs9r0xt5JXpdx5IDB1JOfK_MSlKwjl2kOlNo7p3e7RIVB38MQoih95hSz5E6EyOoH2O6KAXLM3qhgcmXe-XJwQKOkA8rPRclgvN-hLSopl6mDGqg7dNKXZB6SQspQPiB4mx91Wwt62aMqWa7Ve-thQ6-oCq9IQTLT34xpCOpHWz4b8kSfr-WDg9cTDdvYIGEDGj_XlE6CK5PuxzezWGYvYggwVJC_65zkGHFBURkBCRHc7otAyxMTKMCTKZnYWUdUEGpTCBonYFPOlh_ApDPnh1h3feF1x8AEIoVkq_ADRKhKbs28y44LQGp9pe49LdsgKN7YT8_azHknz5frvCUQJu8aiP_TaFCAiZhUc0JSATHPv3q6ZsNoUzF8ZfeSWuI3ZpQyoyMq4wxcyH1sFNqtIHd_iS2U9AkEACp6q6FAohK_O7GaDr_T7VfeO3JuRL9icg82WtPOtgJ1o7oqZxa1lfypGkWAgX_KF5aytblSoxlwn-Zk84Mf_YlbJEBO2mUa0Me5uLhM183akeG4y06FR-ANXrsYxrGmhpIWfl90WAKEG4ExdaexgzQOKXedTLOKnIwDW0ZSAu6O-Eiddn2w5GBO93OGEGQwkXDddA9PB9--mdyHgJM1OvXFYIWtziDLZ5qHAUrUFpYUth9B9mV4TXXLeLqx25TO7DL9czT6cGPeYNSOR3HpNdw93Pbro_kDIk-4zrmhlLd-I6w1tSisK7veJE0Y9Svvf6VLky34iV0RlIw-Z976oPP3rVJpSIujNbuFm84V2EvHFkG1oaHoViDxp4QesbToOdhl8GfnHHuwPaT78C1rwucflil_XkLAi0PBv0uGYLWaaAVlhm_lNZc4CYd7qHcdgma8dL27kHwVUJctIzphtV4yAhL-06SY-kwR9snNogfzDSuuNTVHPofu0_B7qA9vjwf3jeE8WJUbq0HxPSIGMSvJ57uh2YzRPWnMJ2PmARWgHAYoWUsZknHyr7DlKqzV1kxrLmr40xk98D9-Qkkq3enbEazhNWXRTIGMuWQbM1FQB3VL8GX3Xa1TjujWQS8_nC9LTId1XzNWsSCNOg8gHXqvt5hAeJVZ9GMTFJREPCBk5ts9odm3N3bjK5KgA3pNahnOS0u_c2auAc1t9C_5xqxEnIRxE6R1lVHz7hDTebYMcpldFI8IEuToExCjUUpoW6FmpeJrv16hwzTqbX69oToXJ2YA4WYCerDob0BAMHKWDC0nqPJTp9wjq2OnC5YKmMKfJ7lKhVprYG14RKnJfOJdChRpWQH2iK1Hc12sCJBJT2Lvm138cQeRApo2uUTs1dpJ-faoPvb6H_Wg-9ys5JsZtrzUXVpj4ttC84M7dZnbdyEh64iOoUINTSG4yryVDfN7PokiuN3DHuuXlWG8hkWAP4uyZg6sXdeSBGfVOjmG7Tc0zNqxZFkCMo9XEaLk_Zz6X7fQPM-MQVkSrb_0-S7tNwcufe3Y0nEnKpM1WohnyojInkRUzPWLUcBukzYpwuXg3MxxayDLwfjfybNb3hO_aNF_5mLz8s40IotW2Oxa1eKfZO5igty32jA5ipXwPjJjBj5yVgmTESDPmYrPkJCdFUtJJpGRTRX30BbLAybWkhJWBGilR-f5wp7yHp1pAcLl8YIKi-Di35kCCG0qtTimMeyb6E9hQww06pdDKAuVuDmC-RhgG9-x6MlmeJgKeNu2VeANq5lIiAKuDEXz4cs-TBrC_PC4j7dnaIbvchVdxx9Regcqfa2XTeXsKJB8uGwHDBQ_qu6zyKJ2ANyqk8x7Vmmz3YNREGNgiiRQK-zDjU2ZRJN0DJnHxdFMYcmywKz0QOab4orE01eqaG_Zy3vr7fDLOMJOl_puAjLnF7xiVmj1UWC2FwoWngvCrGO-2wsHB7fnvJapv-NpIoW146QuSVwhMROuZ-5SmzQz4MlF-ARtdkEvCAO45BZTM-AnTlOiMqMXspKrHGCiwE_S4FStVdHT4D7m6Ha4q_BRD9nBYp9r0vP4QdsfsK0pegYawDfZ3U9Vuk0dRZp3Z2QBXDLuIs0-0ol7_liDeC64KCdL2zrDSQCXeSQGtDpBdbSDf5xkzQ1N_IStAz6dI_FLbLqRPfE9tf3AfA9k7j3BwN_K6eeVfRueVnf0VOXukHURPwMTVE9C8IdPA8OnwLBD-k1YT2aV3Aow6eC4Dwl7cW2FtQ_BuHcpjMfWuJFiUuw9L8bLCWSlJ3rYdX_MYv-9mY--9d3r49p5FHAoK-bJS5yxHg2Q6Eg6DrSC4dORU5p_fKxrPklgOJKbZoQyb96_dp2qkCVgPjN1naoHfhgqHp49MxalqSx9n-vd86NTSXNw22lnboj4qAsLg54P4RSayO-U6jvj09ZuGtLGrCvKPAgHl_xP4dtpdbOC8OJ3nwHNfxAgZr2P5JEaFHN3RuIQGH5gkwNEA2otd40YptNJvD6r4gasyBPNY15jeYYL18yFQobK60gnXQweT_JXTlnGgbvgoGw6rJ2nifXw_o6Q_uBDQE9MOCJiN0FoivBdcwk-NHU-nNg93U30q4zbGbJ96O0E2X2qRyoOaEhBKAqkHrQM-NGjX7aj1YJFCxIFtQ00OQjT0vhGng6-3cB01l44mGdDFV47PZLzcFrGx7iDaR1Oy2Na5rOLphgT1_zvVF2mCw10HYxOYwSDnm6uqo_kADX_HQAGxQx1xDZ9iaZZGD00hdoZp55kRPGt6uZ9ZEEXPWUpAV8mPUlu0DcK6rr948_btFuMWo9KrVneCiF4qxkk_Q56aFoRa_HQwEEuh3sp9PSCT9wIfBYSR3661Ox7C-25rHNkrdv7XC8ArnQ7xT04y9JYOF_XLxtOLlAzscUryz56Qk7lexv7Qqe80fTTSRNFseDNFCxLjWm6wAo_kpaTOOvFiPUxlYwcmW4hy-hW8zH_JRVPaBCCKR_s68yEEAVve-8q0ePzqcElS4Bt8cFyU2kqodDmzzh2HAFcrA_ScDciNbwKWeJublK97rkWmLCuPZwJcvKNT_FfgYU5OxtF6G2TVVhUcbVAhPQFFegVn_8hYkZ3mJFkZa8FYYNMSGn8nssR8F9RInU9PFLfBiI2_sDJKwfhgLZ9Lb4x2Idx1-XNh9pAFkCv4eRKCoQNrfxZ4VgD7L2UeLTiTBOsH117rSSF3Aq5kaLLV9yqXc7bB8ZVdvsQ64ET6Z3sNM6xk78phFYy-Dnl9sM73dX2qpUefxdItLtql4E_jwm-D2qRdGtdkm3FpQhYhTKgfm--H3hj64AfDQP-7WVykl508pb7Ultl6j9ne28UqtsV6LHaXX3Raq78sZZl7pmkgp3dSZlZXuo0zwyh0eR8aDp201EdOqbG6a8Vs7cHo2pjy_31ZrIGA_rZhFRN5uVohUvtPWCwwV3D6qq9XQRFK6GsiIMgO8oH8v2AB_qopEXWkStyn9HgA-UntHqncuVnnVrprWFYfWVuTDkJhTC1-i9ZgdA1eUvZVxl-I4SfmDG540HUVVRw-Kt15vS7K2vc1FDiQA91Iya8eMjVKOs96Cr--cngQ_zZw062ZqKp7avO0EOTamvz8Zi2a0XzdIH5dH0_9_kXE_T4U43Ud_29QMElOxJfxt-9p9lBNBjfEkUvBayPX2XWlePtoQL89QLg-Xwe7RaGu0hvUiROaArk47B_703dSKDll2V5eUKc5f0H7icWyqp8u5rnjQsafBu6RCWHr7YJ1Pk5TBaw1qQCvNA2Z_FVZWN18No61fV16DHyjoiHBBjTROCS7m7_3snv4SDkoiFvIswpEt7pbYtbXLp5t0EQOQkWQOITojrGYIUi9c23kdiXr3EoPsvSMKlJcDyag25wzVNfA-bfiMCkdHPqaaTFBqYqlwA0SI6bwHKhqFKdGkrJ5YIxgaFLsLwwiNMSBR2wWXSFRXz-HCMewHkVV4LpGxkzeihWXJUO3gXeJuVY1EGxB8U15BUtQQcAAe2Om9KZtOsS2kRtO7vZq62E8jl7bUbTmw0XTZ4eh3xg-IRiK1ynur6XqtZH1kNQFq7k60X-6cFwDS7eDGdzrgl-Kbl6VSzxpwxu5Lz-KIjV5gGBUcOjnIpRuwY_s',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_68cdec701280819fab216c216ff58efe079003437d26d0c0',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='c9d559',
                        ),
                        id='ig_68cdec701280819fab216c216ff58efe079003437d26d0c0',
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
                        tool_call_id='ig_68cdec701280819fab216c216ff58efe079003437d26d0c0',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_68cdecb54530819f9e25118291f5d1fe079003437d26d0c0'),
                ],
                usage=RequestUsage(
                    input_tokens=2858, cache_read_tokens=1920, output_tokens=1071, details={'reasoning_tokens': 896}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cdec61d0a0819fac14ed057a9946a1079003437d26d0c0',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_image_or_text_output(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=str | BinaryImage)

    result = await agent.run('Tell me a two-sentence story about an axolotl.')
    assert result.output == snapshot(IsStr())

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='f77253',
        )
    )


async def test_openai_responses_image_and_text_output(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()])

    result = await agent.run('Tell me a two-sentence story about an axolotl with an illustration.')
    assert result.output == snapshot(IsStr())
    assert result.response.files == snapshot(
        [
            BinaryImage(
                data=IsBytes(),
                media_type='image/png',
                identifier='fbb409',
            )
        ]
    )


async def test_openai_responses_image_generation_with_tool_output(allow_model_requests: None, openai_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()], output_type=Animal)

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(Animal(species='Axolotl', name='Axie'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0360827931d9421b0068dd832972fc81a0a1d7b8703a3f8f9c',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0360827931d9421b0068dd833f660c81a09fc92cfc19fb9b13',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='918a98',
                        ),
                        id='ig_0360827931d9421b0068dd833f660c81a09fc92cfc19fb9b13',
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
                        tool_call_id='ig_0360827931d9421b0068dd833f660c81a09fc92cfc19fb9b13',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_0360827931d9421b0068dd836f4de881a0ae6d58054d203eb2'),
                ],
                usage=RequestUsage(input_tokens=2253, output_tokens=1755, details={'reasoning_tokens': 1600}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0360827931d9421b0068dd8328c08c81a0ba854f245883906f',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please include your response in a tool call.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0360827931d9421b0068dd8371573081a09265815c4896c60f',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"species":"Axolotl","name":"Axie"}',
                        tool_call_id='call_eE7MHM5WMJnMt5srV69NmBJk',
                        id='fc_0360827931d9421b0068dd83918a8c81a08a765e558fd5e071',
                    ),
                ],
                usage=RequestUsage(input_tokens=587, output_tokens=2587, details={'reasoning_tokens': 2560}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0360827931d9421b0068dd8370a70081a09d6de822ee43bbc4',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_eE7MHM5WMJnMt5srV69NmBJk',
                        timestamp=IsDatetime(),
                    )
                ]
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
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_09b7ce6df817433c0068dd840825f481a08746132be64b7dbc',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_09b7ce6df817433c0068dd8418e65881a09a80011c41848b07',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='4ed317',
                        ),
                        id='ig_09b7ce6df817433c0068dd8418e65881a09a80011c41848b07',
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
                        tool_call_id='ig_09b7ce6df817433c0068dd8418e65881a09a80011c41848b07',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='{"species":"Ambystoma mexicanum","name":"Axolotl"}',
                        id='msg_09b7ce6df817433c0068dd8455d66481a0a265a59089859b56',
                    ),
                ],
                usage=RequestUsage(input_tokens=1789, output_tokens=1312, details={'reasoning_tokens': 1152}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_09b7ce6df817433c0068dd8407c37881a0ad817ef3cc3a3600',
                finish_reason='stop',
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
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"species": {"type": "string"}, "name": {"type": "string"}}, "required": ["species", "name"], "title": "Animal", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0d14a5e3c26c21180068dd8721f7e08190964fcca3611acaa8',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0d14a5e3c26c21180068dd87309a608190ab2d8c7af59983ed',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='958792',
                        ),
                        id='ig_0d14a5e3c26c21180068dd87309a608190ab2d8c7af59983ed',
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
                        tool_call_id='ig_0d14a5e3c26c21180068dd87309a608190ab2d8c7af59983ed',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='{"species":"axolotl","name":"Axel"}',
                        id='msg_0d14a5e3c26c21180068dd8763b4508190bb7487109f73e1f4',
                    ),
                ],
                usage=RequestUsage(input_tokens=1812, output_tokens=1313, details={'reasoning_tokens': 1152}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0d14a5e3c26c21180068dd871d439081908dc36e63fab0cedf',
                finish_reason='stop',
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
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='160d47',
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of the animal returned by the get_animal tool.',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0481074da98340df0068dd88e41588819180570a0cf50d0e6e',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='get_animal',
                        args='{}',
                        tool_call_id='call_t76xO1K2zqrJkawkU3tur8vj',
                        id='fc_0481074da98340df0068dd88f000688191afaf54f799b1dfaf',
                    ),
                ],
                usage=RequestUsage(input_tokens=389, output_tokens=721, details={'reasoning_tokens': 704}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0481074da98340df0068dd88dceb1481918b1d167d99bc51cd',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_animal',
                        content='axolotl',
                        tool_call_id='call_t76xO1K2zqrJkawkU3tur8vj',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0481074da98340df0068dd88fb39c0819182d36f882ee0904f',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='160d47',
                        ),
                        id='ig_0481074da98340df0068dd88fb39c0819182d36f882ee0904f',
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
                        tool_call_id='ig_0481074da98340df0068dd88fb39c0819182d36f882ee0904f',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_0481074da98340df0068dd8934b3f48191920fd2feb9de2332'),
                ],
                usage=RequestUsage(input_tokens=1294, output_tokens=65, details={'reasoning_tokens': 0}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0481074da98340df0068dd88f0ba04819185a168065ef28040',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_multiple_images(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=BinaryImage)

    result = await agent.run('Generate two separate images of axolotls.')
    # The first image is used as output
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='2a8c51',
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate two separate images of axolotls.',
                        timestamp=IsDatetime(),
                    )
                ]
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
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='2a8c51',
                        ),
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
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='dd7c41',
                        ),
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
                    TextPart(content='', id='msg_0b6169df6e16e9690068dd8163a99c8191ae96a95eaa8e6365'),
                ],
                usage=RequestUsage(
                    input_tokens=2675,
                    output_tokens=2157,
                    details={'reasoning_tokens': 1984},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0b6169df6e16e9690068dd80d64aec81919c65f238307673bb',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_image_generation_jpeg(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool(output_format='jpeg')], output_type=BinaryImage)

    result = await agent.run('Generate an image of axolotl.')

    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/jpeg',
            identifier='df8cd2',
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of axolotl.',
                        timestamp=IsDatetime(),
                    )
                ]
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
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/jpeg',
                            identifier='df8cd2',
                        ),
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
                    TextPart(content='', id='msg_08acbdf1ae54befc0068dd9d468248819786f55b61db3a9a60'),
                ],
                usage=RequestUsage(input_tokens=1889, output_tokens=1434, details={'reasoning_tokens': 1280}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_08acbdf1ae54befc0068dd9ced226c8197a2e974b29c565407',
                finish_reason='stop',
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
                ]
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
                    ),
                ],
                usage=RequestUsage(input_tokens=103, output_tokens=409, details={'reasoning_tokens': 384}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_001fd29e2d5573f70068ece2e6dfbc819c96557f0de72802be',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_LIXPi261Xx3dGYzlDsOoyHGk',
                        timestamp=IsDatetime(),
                    )
                ]
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
                id='test_server',
                url='https://api.githubcopilot.com/mcp/',
                authorization_token=os.getenv('GITHUB_ACCESS_TOKEN', 'mock-access-token'),
                allowed_tools=['search_repositories'],
            ),
        ],
    )

    result = await agent.run('How many current public repos does user @Pydantic have?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How many current public repos does user @Pydantic have?', timestamp=IsDatetime()
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_list_tools',
                        args={
                            'id': 'mcpl_04507193d793fa550068ed13e10bec81969d1d58d6d4c7a218',
                            'server_label': 'test_server',
                            'tools': [
                                {
                                    'input_schema': {
                                        'properties': {
                                            'minimal_output': {
                                                'default': True,
                                                'description': 'Return minimal repository information (default: true). When false, returns full GitHub API repository objects.',
                                                'type': 'boolean',
                                            },
                                            'order': {
                                                'description': 'Sort order',
                                                'enum': ['asc', 'desc'],
                                                'type': 'string',
                                            },
                                            'page': {
                                                'description': 'Page number for pagination (min 1)',
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'perPage': {
                                                'description': 'Results per page for pagination (min 1, max 100)',
                                                'maximum': 100,
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'query': {
                                                'description': "Repository search query. Examples: 'machine learning in:name stars:>1000 language:python', 'topic:react', 'user:facebook'. Supports advanced search syntax for precise filtering.",
                                                'type': 'string',
                                            },
                                            'sort': {
                                                'description': 'Sort repositories by field, defaults to best match',
                                                'enum': ['stars', 'forks', 'help-wanted-issues', 'updated'],
                                                'type': 'string',
                                            },
                                        },
                                        'required': ['query'],
                                        'type': 'object',
                                    },
                                    'name': 'search_repositories',
                                    'annotations': {'read_only': True},
                                    'description': 'Find GitHub repositories by name, description, readme, topics, or other metadata. Perfect for discovering projects, finding examples, or locating specific repositories across GitHub.',
                                }
                            ],
                            'type': 'mcp_list_tools',
                            'error': None,
                        },
                        tool_call_id='mcpl_04507193d793fa550068ed13e10bec81969d1d58d6d4c7a218',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_list_tools',
                        content={
                            'id': 'mcpl_04507193d793fa550068ed13e10bec81969d1d58d6d4c7a218',
                            'server_label': 'test_server',
                            'tools': [
                                {
                                    'input_schema': {
                                        'properties': {
                                            'minimal_output': {
                                                'default': True,
                                                'description': 'Return minimal repository information (default: true). When false, returns full GitHub API repository objects.',
                                                'type': 'boolean',
                                            },
                                            'order': {
                                                'description': 'Sort order',
                                                'enum': ['asc', 'desc'],
                                                'type': 'string',
                                            },
                                            'page': {
                                                'description': 'Page number for pagination (min 1)',
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'perPage': {
                                                'description': 'Results per page for pagination (min 1, max 100)',
                                                'maximum': 100,
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'query': {
                                                'description': "Repository search query. Examples: 'machine learning in:name stars:>1000 language:python', 'topic:react', 'user:facebook'. Supports advanced search syntax for precise filtering.",
                                                'type': 'string',
                                            },
                                            'sort': {
                                                'description': 'Sort repositories by field, defaults to best match',
                                                'enum': ['stars', 'forks', 'help-wanted-issues', 'updated'],
                                                'type': 'string',
                                            },
                                        },
                                        'required': ['query'],
                                        'type': 'object',
                                    },
                                    'name': 'search_repositories',
                                    'annotations': {'read_only': True},
                                    'description': 'Find GitHub repositories by name, description, readme, topics, or other metadata. Perfect for discovering projects, finding examples, or locating specific repositories across GitHub.',
                                }
                            ],
                            'type': 'mcp_list_tools',
                            'error': None,
                        },
                        tool_call_id='mcpl_04507193d793fa550068ed13e10bec81969d1d58d6d4c7a218',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_04507193d793fa550068ed13e3445c81968bfac1f471006aa8',
                        signature='gAAAAABo7RPwBlGhrl9wLbNIVNY20JNHmCrVJFt4aMltys-m2T5tcNUsVVQVMFZlCQnxQJRNrQIFpeMiASvak7ETlmzUAGs5dtQ0esVI-PWUYoYi85daNwlW5DAwo9uxZzm19-wy6dSHfeQtM_F2nzmL2pIoO_qxbGYXQcZVHXPLOXAqzbedzVZRTdH-1j4DVKE_c9WPb62xFpp7DAoxRsfyak40VcdTivvmpVWfJ-x47FcEE_OMew6PNnBlspi1k4yEvjKZ8_OwvbyWeKeR4SuhZUeAbz6ZWOqptQ3lHoH9pprvTjoB8h248ZipfOoAzETJ8sAzLCV-KLgo4_IcsAvXpR5RjxbvaWzgSG9UBiAGlNDlh6dwVBdvnNvDNqkeMW1wYHiqTFDUdxNvHmFnYRIVmg4ZQ6ndjrXu76QY1N7cXoo0gjHHlezzsR-atyveiKW6QmZtlGV5IzRwzZpb7uZaX-ofWa4Fc562vlgJ71WWlMwn4qMIdksm3gaQ5-n6Z65Ci0LAD7xNVTPYzcKXAvO8YzMVXShgoCePpiJsB_Pc2_3lH2YzxNCBGHnwAAqvfU529IqXOClirz5wSMMW60oSmwB-Xj4Prv5778aTQgDFBM8X12vWqJie9CXuVwsjeeyWyGt0ZNnx5jX0VdXbRUNK-mjFX3f-gWNkPvzGnVgQ6XmPFhfO53Sy5KeTEVaVEvnTZdO2YuOQFOWYFIddc7Wq5SnMK7C_LMdURnguBRs-PpEX-C2-kwDH4whkABYO-V8F_PrFPdeGh2Sx3rwhtpoPAd5qM30DDCgf1A8zmVhvMf-7H5mQzXblUJu1l4KwKlm5jWkUX9X7bOk4TwVxubej7U7Q39A8ln2AcejtAcBo-8QGLdyf_q_WsNM45UvmS2HE76iAClfQBrEvHMPZj4Tysuz0_7adhiL-OkxL8fyrcyQZls1cbQot_qudXJ3ICnbs5ss6Z8VdIPYLIPuVRpWHK8K4J_DiyxMXZC6t1Um7FjrO4qw-3lWAtFCRGA5xIDkTshqsRQReFm6XmeKO5FrfwSDkLGMTUGwMWlKjoMP3GNLPMmEdq-hJ2snp1ZlROAfXDLB6BPkmmrTOe8XQv5XCwCCqvSgjhFhfnvsC7ulNSiZLAuDu96EY_YfjX0dNIPJN7G_I_L-WwCxRqEyH85jwM5pPRoRSFxyjeRrb9_JTeIilvgmF9RXiYIB4tnWl0XNuuZZsgZlzFXtINWJ3LmOURZ6dUwkqMire_UCcalQM5caZRG4uNzG-4erAShbttJwN8_-ieNqSKbJ2PX5McUVWXcQXZmn9DZHfdrwgp2dJQT9BS8w4Zj5iASXF_9OwIsZET0E9j2JEOHC2PS1BU2HLORZRpy4C2z0GYsDZpqGf2fOBfXGHJED_p51OXbdgbJ8hxm_ZPgk4f7TMScjFW4BvaGHelldskms4e1Znv85o5J37TSKT6A55CM4RbFLWsNDRdb4OI2Rx3NLj_twcznCCXX4XkB1Ex5Tyw-BWxMwzaZkGcBK6cG-r0-Fj8LIcbAg4JIBMQWT4pPG1fopdLephXcRosrjj9gapsCV0mgGLU5gMYwztmEgdDe7t4hO659PuH6Yk0vxLWGl-U6MyZzOdeSr8vikyJ7WmsLd7nFIJOEj3lAWliEslnQlDnhmouUp79sGN7ciylSxhuxm8F6Wi8TqJGU-yJdqReK2jIhL6AmImaW4_FusxOV3twYNy7yv0a8SCIs9KPOaaxUARUiuIfvN1tRIlkTcLJUypfuII3ZzFP_vrCNJ_FUphpojUAWhh8CQhW8YeME9otpJLQkLRVb1Mv0500VkCJgTMdYHfbxAbNyxLxdFABtKBE172QKh7RwQA2rJxHm2Ng57iWusHhb2yFHp8MaRF3hN0ro780uljZ6ymAlqYy3_auqTGZb0c538_OKf2ZUQmc_NI1M05S950cZTMUyy7ll5WRb7ZIQMv3g_Pk8Q-aqtOSCuB4KpxYA1yCdh6PDaRBUfdobj9kAN_9pSqwyGxo9Jpeky2viDXtKonOEk-dZnK-BlRM_4GRPLvpQYw0MKQxVKcJ7yKRd_e-zZDl641G1zE0Ycntte9U51JkwgERzHdbs8W967tdoSGRYsU72lcKdaFTxHBrYzrlzdWxFGmLdTZ-VrfjJbxU8ZSifRnHFXAknLaCyVsZeBoBrz68sEEteE8oTFZTGUKgtkXhP3iEwrtYvLDG44fO0lAd9tOUXh23doZHcncq9zUqVe99z5Qkc3g1DieeaHBj2r8AJKHwqwI_uUWhzV3dwnCK-o6vYGoff1UztGHiJT5Zgol-qwIrL_3o6CijNZzMEuhShH7zTKMZJVdc79juY3Q9LMSRVQIH1qpsV56ISOOhy9pVamYEDcxpv_oosPs8b77bu5RztTx-BwBhsnWxz5k6S80toGtthZAOQsPETUhB81m_1ZNjkmmuzDrqK2ITA3CY-C2EZVQf-5mD4K0jAwcbtmw57PkP65WI9fEumGTGVHKnUQ3UUyl1BDm8ZGiTLtk0BjK4gEdi2CGbre-nGARfwGA8igCDqyZqw2pEg0AFV2ELu7Qia25ZOSt1UyCSX5rReYjEIG6L5IOiM34p3sISsOFmi59O4P76Ls2oNfY32TTYulDWPX2R6z8mvfBlgAHbtNh17FkATnbv7BuRU3_3ZhTIrbK2I7hROn80vVM1c_A8aMmBCLoINfifHdhooNOreTnhGSvA2uillBBVEhbsU9eXf7SqsVcZl0jA5NqlJ8DsLKk1Qc54qudZzPdGOzB9aFJQfPAubJErXFaHr9gmxlJOmoDYouYzArpX939VL0IkCt-KbK997imQ2UFtyRYtDC-7dVya2Alg4ua8XyWHtezY7ADZBcMSdpwoXvZHWd0jCBwm5mJ16oMW4dao-M4xNM5Dp2NzK5QCHPt77LxqP3fNdFI7cjsZSM32tLwxqQfvvyrsBYemE55xISrj0VGj8fNS4qBC4T8DXiJ6LJgh7-wXrPeGihvIfhpATYjdQSjyf-6TXSB8HwzkLVdoWakokXBqpWfLMG8Qn2UNNuyr3heuOqOxh7kIKdr0ucJXAd1KlyVZoVGkMZIg74pqR9CSuB82889jSqYfsBKUzzN4mk8dE9QLZDu-11nnIrlD2JmZiJYaq49ZrMpYX8KZtvM1YjdAyXHrM6WPFBs4MXco7DZJUCTgadgmSDZ5FHG1sry4Wl3bvjSA1BwSk9naQFYeKRHmANEBJXfDV-a2HT7QbY=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_04507193d793fa550068ed13e71e5c8196b9588faf7b4f61de',
                            'arguments': '{"minimal_output":true,"order":"desc","page":1,"perPage":100,"query":"user:pydantic"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'approval_request_id': None,
                            'status': 'failed',
                            'error': {'code': 32600, 'message': 'Session terminated', 'type': 'mcp_protocol_error'},
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13e71e5c8196b9588faf7b4f61de',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_04507193d793fa550068ed13e71e5c8196b9588faf7b4f61de',
                            'arguments': '{"minimal_output":true,"order":"desc","page":1,"perPage":100,"query":"user:pydantic"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'approval_request_id': None,
                            'status': 'failed',
                            'error': {'code': 32600, 'message': 'Session terminated', 'type': 'mcp_protocol_error'},
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13e71e5c8196b9588faf7b4f61de',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_04507193d793fa550068ed13e7caa88196ad5ec0abaa0315a3',
                        signature='gAAAAABo7RPwhF0RC2doViW9PTmEGAAWqjmI3o6aRABDFpqWMNYpySD0RtxVltuNWYnocK5_IIoA35C4Y0k8gQLR6cHF2CzpC4nFO4GCZpm4awByYiWkk9q3AP5PICjIco5BdCszCZBsOoaAtbUVI_wjj0Qomj3D1axk0frsXvTx9mzeEnjeHQQxS6V6Z53UDcwnfQ6pxXCzUDcAb2L5VAp_pw2lKOf34npFRQ11tkuU8djhuapZjNGXI8cyaX7bZifVoIaIxELGsIA6p7pbDxaC9EaZPttde9n3vP66Jt-qJfsumebs-lz_X8aosVVwszVoKoK0XLjzfTvkRA-S_Rjl_pJBvfUSlOJqMDNc2N5BqwrHwzS8B1Dqkx_JzI7zoVIgPTNwcKpIMBWSDjUbe2vjGOAyjLJQYLO9ah4Q1curobn18dKrOay3mYhoyKq28cx78lp-zwQvABHYW4PUcpj29g9HCLTZZQz2UXb-OSqy_QHEsxzlSlXY52SneDAW5i1euMRKYGHKZ6plVlLKHHv8PWsqDktb68lMZHJLmEpNPF7dCn3Bw39QrcUhO7-yURNgad0GflXGZG7yzeQ1-68cYVECD1Kvri3I5F_aQO9Sbh6a1iV_X8_xLcITi-MSOHfKzL-aqdih7nEWJeYgsgk1bYeDvDJgrGOqckNQjjX5OEBN73X-vkEPaCtBLswJ-Vr-ENqaPGVlX5zNUyOFhzFEdZUcZ4lFGYgeTapWOsW8m0bowUouvgXP0-7neRQ25nLnt95gJjdZEYE5Sl_SuRMz016rAcNIoa7Rgw1-u7sqoKtDf9YZ9mrxXmguV8NvYQVIgRS8UvxpJTtWYQIaCKuxm2_1A4YV4uGdSpRnd5RBhVpO8tphWy3tJNUxaZIa3E_DersilTfQmJEZbVGlI8b0-OiyZvlkRAb5STX4z04-3aoiKx_D44k=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_04507193d793fa550068ed13e84cec819697e55f0f6d7baec7',
                            'arguments': '{"minimal_output":true,"order":"desc","page":1,"perPage":100,"query":"user:Pydantic"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'approval_request_id': None,
                            'status': 'failed',
                            'error': {'code': 32600, 'message': 'Session terminated', 'type': 'mcp_protocol_error'},
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13e84cec819697e55f0f6d7baec7',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_04507193d793fa550068ed13e84cec819697e55f0f6d7baec7',
                            'arguments': '{"minimal_output":true,"order":"desc","page":1,"perPage":100,"query":"user:Pydantic"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'approval_request_id': None,
                            'status': 'failed',
                            'error': {'code': 32600, 'message': 'Session terminated', 'type': 'mcp_protocol_error'},
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13e84cec819697e55f0f6d7baec7',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_04507193d793fa550068ed13e8fedc81968c9fe669111ef8b0',
                        signature='gAAAAABo7RPwDENXNHyUucg1eGSOTdXv5VI0iMY2uLpeev4ModlAMbFZKrQYiSbAgmXgRSK80Uwl1ykNsxDjWMdcvlVeSQ3t40RjcfBpPfkZS5exjeNiHHT4I8p1M6-d9JhXY9BjtdATrArGIhchWnWTs0s1RfQbimhKIUlFLDNqty35mg6qZbS6lcGCTmc_yFuqTAlgQIEcsYDhDo6GBwHIJOKAoK-U7ZgGhaXi3TXfscV7qyqnd9O4UBY3GAYX4VfBem6y4QgCOamr7UA3-KyouI3m0XQrKcIxCB1KD9kY8NbU9ZJtpGOBq_yccjZ8hLTpuSClgiIIPFCryRzFyGsmWLgXNpH1eK3W5na2BTxMyopZYHdEKg-pce-cKoVun590by4G7-RQC7BHIB_SOzs7s_dvjFIelh8KrI0nTVXPDnNj4aDWkxXzFw_eWHTlEc1--ex34MPdQ9EIxTmHfJl8prE3Ld3SQd1FXl-TqpwQnI42MuPMRisAXP6mwfmDR-tkn0xvcP2SjLLs2BFEuTjTaxzHZUpnKMCw_y67_MdNZW8nlr0Ameo-iybsW_tRNf7QmQKYfDeiq43kx-dE53FghotnomFH-bMnDk5mdbfsxj7Y6Dd9m5slOD96s_D9SE9JRFk6w3Kok6E94d5_8vx9pKmVGlq7MCoPHxdtMjuiPdYLdUIcn-Tb5bZXtU2ISlwD56ez8AIB5D4Xh0k_hN5CuI6Zp6Rx5W6bJiIhtt7s51RyKOE8vOJCl2tAx5u15bCSwcocyGRoNxv8xzUYhDCuEeN8QSyxY_qSPJUJk6d0XtCa8fUuWEvuxgE12tFaWZjXunZ8y7u6352lfzKgYukB-5tMwrNXfa0wDhtvr1ZePBwy47A7vbgk1jBt1NMljSD0fs_AmaFv1pA9VwElq2eTKRR2PoZKNKLS_GhVWjNKKlXKDPmNYEcg5Qf4xa3o8lhb_hl4bHfUfbs8eXmnHLIkeiCQy7ukdhgDCDVKt4ryzjccCZKC5TaIv8tUDfkuFrQ_M-bFxBzeeTFGaTqTpUeMhjR3eUtfnAo643IxlLYHR6B4JmqaXkWvA0qtCj9U40URpsAldcXMC60E00SzsiagZUGQrw3dpHRF12d9KAMKd56lTvRRTicph3UWYjjt82929kPCBufb1lWd83MCTlDOAcqr00LfO-NJXxkUQNcpT_Pb1qYV8u9tC_8I1uBejfhVmK6X7vRN4eCGGVHqwZug3I6s4Y3ClkTZFfCqjJV908wjSs9TJtd8XJd8nn6ZixmuvDZs8080z3EWaucj2WeaDxjSAkqtsM_G420OpzLqWFjNH_Eu_FT5ofRxb25OfJdy4GJ9-G6AFEMfwjS2DGFuIWLsNY20sR34f7RC_zfW8miN7RRoIXMVl9xVGafoKU4s9z_jVI3qmAQFoWP2fKV2naA-0LxR2-YwRfGb_Np7ThJlVkSE7sJNc2-Chy3IkUijHmoa65uX3EAk26UyLRzjkpT1yf3VBZM7tfYVMctxX1BMn7pWFB4Vx7QcndfXQXBdd9j9GvgluWnJF3j11Z9Ix86k1ru7OvuT8ZvnUgaa8qnhmPw7swqPZi1ucC0wGWa4AGAf8icjSKDBPH2du_N3sDWkp38BxKZN3xj0kyXzZTTwa09yNoPFGjtI1AyRV6boUlnx6Pi1hiTIl6RTtyx21rLPRI71GGFGkr92TIG_xKtQn7DvEIDxLkyl_Mt0MXVr1pibSydYcKMmpxM_k1Mg6_nzbTZb_6MMZtQVV0JrOt5e054I8iBvK6k1A5625UKdKHFhEjO4ygQHwfWqUBMFw8CZ13FhIHmIR04AJOwvgrGWgIyeyemLkVFa3tBPmSPiBUg0Zisg-8I9E_34rplqj-GiQhtMEAWCUYIemN56LPaI90M8w6kiJAw4mMn5cGLivlFOp_ny2t0ttwAahibffZy1wrbVDKC32jHEbkf8-RoTnXG4NKe7ZXODPmfqvAcZWAbQgdbjv6w_NR_rVYu5Exfr8egKauuiqJm3I2gumKlQ7RALjSQs1ZoMAftz9g8a2cBtB8pIyoqynp7sXEPTY9yVmPtIuUjK_n2conlxTgVs-Tf2coPyQpXI93QpZ7WFXXV9LBmqBWJSjZkJlYjVIrmf382ly-WvUeqPxXJp43kmxdR-YFVQH-DyBNfWFJAiDh8nXlfukn9PsCqHcZHlg2Ejeq5ZJkIrjxE1_Ql-9l9ndeaiSmo4dTjDXh7LSa26AzvF_Ra7itBza1LHx6E9OGYKf5CFeAAdocV7zThCHwVb6Q7VMIlm6yX6Yk5geE4qqiOYB24CsJYXV5oDOrhyfQ0v35-FukeTEPRdG_NuXg-_29rBz9WYjXekvsmaJ_I4kilBKEfBVCPHKTHN2R3UHWAWxRicrwKJG70W7gdbbZFdnwKcV6OQvSRbi6ffuwYcWAkZu9HnMwjVvNTwRqh4A0LX8rCTyYuzbIUAvkr3pHKOEHMLZa3fmveOauA_PQLpX4q8zDIhjxqKqYkR74pFym3Bsu4Vu72TCb0x3YHcM9SOEwaKiZplZmHtYqJlxJA_4KO5xaDVL0Rvp_QfDegBj15BzUemyZIet0n5fF8MkzuslWBEkar_NodUE6Qsq0Xgx7Ph7yiuZSdqQy8W-r1KzME_FhserB8plVYLfW-gU826FXU5MS9ulk1ROi6oI8fsmUEPcSU8Qt1ao6e6HUARrMusInIHjilwF3b9BPY1apfeh3nEfMsgvjATxroITsiTnVORNlcyr9DBIUfbj3A8Fby9ZYYLp8K-VBIc7w1xckiHJEotdGSsDBDXUYNiZiC7CXy0TKws2xpxY4GFXw7HBC-ImgHEDZqzw7GjEJFnNYbNnjoRR54C9lG2o_QGtl89VHUXXqUVCMesAs6E9BqdX3u6B1fQlTfJ6zPN5V7NDIz3rspEYFa2HUIF0SzsKhANovwwu9PYGkzwTC-rCmxN1LCn_nRI5BgfnbtTxPRDOHoGgLCrvCRCKXdSpzdmaZ0a-Ivp2HufIKsEJFPdOc0clPocrmhIrJyX3FVTSNin7D-hBxOONVCHupHE5zxQCLqvKM_p8-dEYwuusMEyxr8Jg-dBK2B29WM4-ee9XHqFc8UL0I5on1Z2ujNRCKXTqfW5KrfyVClnG4jkm5NeJmPqBYdjnFqJnr29ZbULWFm9ku1VcE_61z6eJn-4JEFS8-1fkJtRpR1807yZ9o6ZmxtE7pGL1c9wiTZFtOg6N5IXQDmqaWnvRZkluzhZbUQfQszlgrq_h12uvC5bmcmBwnKYLRO3TwL77QmAMHvfSU79xPdawXfdmYZ3Ux0NgEjzbLIrbngMxJa8Y0UJGE5g4rao6apAj5U6UoVZScuO5et7Prkujfxodn4QbfOZipYDLitm9Wl7-Oeg-y37iyjheNeEqGKURI-29cIgZ_MEsG82Hvn2AAVYay69Qa9wemwz0ICYEdscv0iXy3weEXfpt1T-IB1OrPHnR2triX2Nj2jaXlgaOKZC27wHcREP8Yjb7IxjFECA7toB0I9Sbiqr6QGytWtuAbCtlCbAnVhQZs8BRCXn3uyGdUtK_849RYFdRq18X8HRGz-x2pb7oaryKrl-iGtE29IMACtHbDXHfRnUn1R7uaCbkrHyAsp-QjMClXkWraSGlmtxFBtq-HDyJvscvSfFuZumwBDZLtRXMlIwBV0ueyTGHyBA-g39rYZyLmG3Y5gf2TsuAqmn5lqYGWmgUP3AUI_O2EMbmCR1QYlCSgbz_Xp9VBzZrNztMxP3W6-uFtengj4uSH4Gxr9QjFwDTBmJtTexfUbKJFrQUemZmnqzb1Y2jIJl7_g3gFWen10mn6s4Bgt0ICqd0sxuhwOyrdFWt2CSb77gnSdbflBXCQa5xdLRHfOAcgYAseq4a8Pjve2QtjUdCAI1yUn7A4D90JssIZ6uRJYrnE-Gi4Di2b4CUBhsEhpvvAZMj-Shp9EC5bj2VwqQUHIB8iflPKs0brzwKGmF_6LgEyMb32f4sG_pkkHzdWXvTnhwNI_g0WxhTHBlBXlmatIEnJL-5d9UX0duYCqv5Il2V8E4nLq5RVOXiwyhI74N7optrqR6vhUMU-ntuTshc01k3cqEo-XRso09X7BY5AHnoLI_QaQWLFfBOmnF8VM-ecuWHCWaZHACgDkZ9_RT0PMXd8YrWxf1Al9a8kkp0FVSq1gG_s_7DzgpQ8oi9Twp9fhs4SVtjpg2JZBaSjq3e7G8uvllXiuI_k5uZ6p4zdfH6n_GB2plSAXOiEJRE9cCyGfOYxf54d5MprslS8tLUCT_5hTz_6ciPgGUXI1JhypORYMZ1Ah3OrWI9YusIkO3LLZ2ECh1MSO_uELFMNHDAB49x6GGsZYsZv6xW0s4Zav2yLsyap1GJjNmw0uUXxMwRZf9_A2TCnxB1RS0Ds0tfRFlCeeXHDfT8OXT4aN1gXF33736vOUOrKkfuFYAcbvWl4naLu5siVfZUVpeIZsn028fhvuhOmsJUcGGcSWtsT1gbmcRDq9WjV-k9qcPOIgQjrhyOTrrukV7AWt0cnmTF3qYptviW9OAtUOvVk095YsAOBFiT4l6KiEnf3k6iW8HwLCNWHgEpLTiyfeix_Ax9ff92JrMkU-3uOT6jTHit9Cs8pAyHTUvuBEHsT8ca-lAdlqYiNAmTM2lF23ECg3z6WK8dOyua-ZZ1fotW1u3DsMWX5hBJI5Dc6P_VSsGQMj7KLgeEgeV_n_VhMxUb__yZ2ezfQsImFtZrSIztLBmkwWHaoB_KaNQVgWX9pbEbCfaRNRrY-ZDVLEEgd8my41v0tqnBQ-iN9n_32OJuLFWCxlV9wNB3vw5mqSskIxvxMJfcgL3Gy3jZF3R14xRjRgnLX58cmNJjKqRt6fQX7ClI6GfXUz23dPvt57GZios46E3dMZuVuZSpLQ1g3xneP6pBDc2l5fFHtRtDqG5CrUVyJzl1sDGgDreyfIgFZ7ixUtR8ptf9wEG-G_JP6YUoomSATnli-odJxpHySoLstvQmPfD8rWzqiSDKsBKejXZYIVnsv4Z0xGF5IwbID6N1xVrW1jE5psusQF73uCxHlOEXaLuS1Od3BMyp7zQFEhjjkVb9k0t7GY-_xys0kwInyeL34ov9TZ8hO5cii6HgXv4Gg866_MRghXse1P4z7Y5CR6ieHwls6ue996_MQY2gpZBnzvzaSUq096z5R5wp3bsGMIPLeaLptTtaJF7Cj7DylfmdLJOSVi86a53anitDCk8VYfmn7vi6HwYHAgVF6-84MlEKzp-5QeELmOZyVLjEv4IuyG5bvJglwQLhoTecSNS8haaWe88UTn_RfOTZ51aM9HJWeaxa8oONxdXq0UihgFPBMLd6UsPAoXkhutK7tzYOyTuzOGFaqSw0L4JtT9fn-hriBEtFWy6Cofuc4tpDsJuKT_IiaIX6JaU1vP1bflmcNuiOFd0K_WvgaZK0YvF2rMT7QmSiLGReiSGThb1GGS-IGU1wM4FJ5RoMh598hM8bLfqcQYZcIJlR7m9VHo2qyPZUiCtgFzUsqvmQoTYCYRBVn34CM9UMJGT3gEXpYBVvW-FVWTeNWR9a1QFQbEdlS2tVzyb34SwLaq9TZHns2f21PIOp3zl2Q4o3XQUeFPyYQVkBypUL4Fv9ZwoBrUWtxCoEd859dsgv93XfYozZtbL2WuZYXlzAW_5GHxwpUcrGScyfF8BF8U_vfGG2alDzRpGN1t9A1FTjjktmeOBNyVeuGzjilGq1_at-d-tiltXMyK2wL-I6Hp5xBk-MUj9mD9vVNALdWrFywfpJHJayRIlW0ML86vFbwG4YIjOM9gBOi1Pz3cQ7hQFN4fv9IZ4oooIo6V2jq-FL0kTGCSDv2CTU0xNj48ta8uSzhLO0kf4',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='As of right now, the Pydantic GitHub account has 17 public repositories.',
                        id='msg_04507193d793fa550068ed13f02d7c8196afba59eeeb18c096',
                    ),
                ],
                usage=RequestUsage(input_tokens=871, output_tokens=1392, details={'reasoning_tokens': 1280}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_04507193d793fa550068ed13e0cc488196a6f4c49747240ea9',
                finish_reason='stop',
            ),
        ]
    )
    messages = result.all_messages()
    result = await agent.run(
        'Can you tell me more about pydantic-ai repo? Keep your answer short', message_history=messages
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How many current public repos does user @Pydantic have?', timestamp=IsDatetime()
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_list_tools',
                        args={
                            'id': 'mcpl_04507193d793fa550068ed13e10bec81969d1d58d6d4c7a218',
                            'server_label': 'test_server',
                            'tools': [
                                {
                                    'input_schema': {
                                        'properties': {
                                            'minimal_output': {
                                                'default': True,
                                                'description': 'Return minimal repository information (default: true). When false, returns full GitHub API repository objects.',
                                                'type': 'boolean',
                                            },
                                            'order': {
                                                'description': 'Sort order',
                                                'enum': ['asc', 'desc'],
                                                'type': 'string',
                                            },
                                            'page': {
                                                'description': 'Page number for pagination (min 1)',
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'perPage': {
                                                'description': 'Results per page for pagination (min 1, max 100)',
                                                'maximum': 100,
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'query': {
                                                'description': "Repository search query. Examples: 'machine learning in:name stars:>1000 language:python', 'topic:react', 'user:facebook'. Supports advanced search syntax for precise filtering.",
                                                'type': 'string',
                                            },
                                            'sort': {
                                                'description': 'Sort repositories by field, defaults to best match',
                                                'enum': ['stars', 'forks', 'help-wanted-issues', 'updated'],
                                                'type': 'string',
                                            },
                                        },
                                        'required': ['query'],
                                        'type': 'object',
                                    },
                                    'name': 'search_repositories',
                                    'annotations': {'read_only': True},
                                    'description': 'Find GitHub repositories by name, description, readme, topics, or other metadata. Perfect for discovering projects, finding examples, or locating specific repositories across GitHub.',
                                }
                            ],
                            'type': 'mcp_list_tools',
                            'error': None,
                        },
                        tool_call_id='mcpl_04507193d793fa550068ed13e10bec81969d1d58d6d4c7a218',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_list_tools',
                        content={
                            'id': 'mcpl_04507193d793fa550068ed13e10bec81969d1d58d6d4c7a218',
                            'server_label': 'test_server',
                            'tools': [
                                {
                                    'input_schema': {
                                        'properties': {
                                            'minimal_output': {
                                                'default': True,
                                                'description': 'Return minimal repository information (default: true). When false, returns full GitHub API repository objects.',
                                                'type': 'boolean',
                                            },
                                            'order': {
                                                'description': 'Sort order',
                                                'enum': ['asc', 'desc'],
                                                'type': 'string',
                                            },
                                            'page': {
                                                'description': 'Page number for pagination (min 1)',
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'perPage': {
                                                'description': 'Results per page for pagination (min 1, max 100)',
                                                'maximum': 100,
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'query': {
                                                'description': "Repository search query. Examples: 'machine learning in:name stars:>1000 language:python', 'topic:react', 'user:facebook'. Supports advanced search syntax for precise filtering.",
                                                'type': 'string',
                                            },
                                            'sort': {
                                                'description': 'Sort repositories by field, defaults to best match',
                                                'enum': ['stars', 'forks', 'help-wanted-issues', 'updated'],
                                                'type': 'string',
                                            },
                                        },
                                        'required': ['query'],
                                        'type': 'object',
                                    },
                                    'name': 'search_repositories',
                                    'annotations': {'read_only': True},
                                    'description': 'Find GitHub repositories by name, description, readme, topics, or other metadata. Perfect for discovering projects, finding examples, or locating specific repositories across GitHub.',
                                }
                            ],
                            'type': 'mcp_list_tools',
                            'error': None,
                        },
                        tool_call_id='mcpl_04507193d793fa550068ed13e10bec81969d1d58d6d4c7a218',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_04507193d793fa550068ed13e3445c81968bfac1f471006aa8',
                        signature='gAAAAABo7RPwBlGhrl9wLbNIVNY20JNHmCrVJFt4aMltys-m2T5tcNUsVVQVMFZlCQnxQJRNrQIFpeMiASvak7ETlmzUAGs5dtQ0esVI-PWUYoYi85daNwlW5DAwo9uxZzm19-wy6dSHfeQtM_F2nzmL2pIoO_qxbGYXQcZVHXPLOXAqzbedzVZRTdH-1j4DVKE_c9WPb62xFpp7DAoxRsfyak40VcdTivvmpVWfJ-x47FcEE_OMew6PNnBlspi1k4yEvjKZ8_OwvbyWeKeR4SuhZUeAbz6ZWOqptQ3lHoH9pprvTjoB8h248ZipfOoAzETJ8sAzLCV-KLgo4_IcsAvXpR5RjxbvaWzgSG9UBiAGlNDlh6dwVBdvnNvDNqkeMW1wYHiqTFDUdxNvHmFnYRIVmg4ZQ6ndjrXu76QY1N7cXoo0gjHHlezzsR-atyveiKW6QmZtlGV5IzRwzZpb7uZaX-ofWa4Fc562vlgJ71WWlMwn4qMIdksm3gaQ5-n6Z65Ci0LAD7xNVTPYzcKXAvO8YzMVXShgoCePpiJsB_Pc2_3lH2YzxNCBGHnwAAqvfU529IqXOClirz5wSMMW60oSmwB-Xj4Prv5778aTQgDFBM8X12vWqJie9CXuVwsjeeyWyGt0ZNnx5jX0VdXbRUNK-mjFX3f-gWNkPvzGnVgQ6XmPFhfO53Sy5KeTEVaVEvnTZdO2YuOQFOWYFIddc7Wq5SnMK7C_LMdURnguBRs-PpEX-C2-kwDH4whkABYO-V8F_PrFPdeGh2Sx3rwhtpoPAd5qM30DDCgf1A8zmVhvMf-7H5mQzXblUJu1l4KwKlm5jWkUX9X7bOk4TwVxubej7U7Q39A8ln2AcejtAcBo-8QGLdyf_q_WsNM45UvmS2HE76iAClfQBrEvHMPZj4Tysuz0_7adhiL-OkxL8fyrcyQZls1cbQot_qudXJ3ICnbs5ss6Z8VdIPYLIPuVRpWHK8K4J_DiyxMXZC6t1Um7FjrO4qw-3lWAtFCRGA5xIDkTshqsRQReFm6XmeKO5FrfwSDkLGMTUGwMWlKjoMP3GNLPMmEdq-hJ2snp1ZlROAfXDLB6BPkmmrTOe8XQv5XCwCCqvSgjhFhfnvsC7ulNSiZLAuDu96EY_YfjX0dNIPJN7G_I_L-WwCxRqEyH85jwM5pPRoRSFxyjeRrb9_JTeIilvgmF9RXiYIB4tnWl0XNuuZZsgZlzFXtINWJ3LmOURZ6dUwkqMire_UCcalQM5caZRG4uNzG-4erAShbttJwN8_-ieNqSKbJ2PX5McUVWXcQXZmn9DZHfdrwgp2dJQT9BS8w4Zj5iASXF_9OwIsZET0E9j2JEOHC2PS1BU2HLORZRpy4C2z0GYsDZpqGf2fOBfXGHJED_p51OXbdgbJ8hxm_ZPgk4f7TMScjFW4BvaGHelldskms4e1Znv85o5J37TSKT6A55CM4RbFLWsNDRdb4OI2Rx3NLj_twcznCCXX4XkB1Ex5Tyw-BWxMwzaZkGcBK6cG-r0-Fj8LIcbAg4JIBMQWT4pPG1fopdLephXcRosrjj9gapsCV0mgGLU5gMYwztmEgdDe7t4hO659PuH6Yk0vxLWGl-U6MyZzOdeSr8vikyJ7WmsLd7nFIJOEj3lAWliEslnQlDnhmouUp79sGN7ciylSxhuxm8F6Wi8TqJGU-yJdqReK2jIhL6AmImaW4_FusxOV3twYNy7yv0a8SCIs9KPOaaxUARUiuIfvN1tRIlkTcLJUypfuII3ZzFP_vrCNJ_FUphpojUAWhh8CQhW8YeME9otpJLQkLRVb1Mv0500VkCJgTMdYHfbxAbNyxLxdFABtKBE172QKh7RwQA2rJxHm2Ng57iWusHhb2yFHp8MaRF3hN0ro780uljZ6ymAlqYy3_auqTGZb0c538_OKf2ZUQmc_NI1M05S950cZTMUyy7ll5WRb7ZIQMv3g_Pk8Q-aqtOSCuB4KpxYA1yCdh6PDaRBUfdobj9kAN_9pSqwyGxo9Jpeky2viDXtKonOEk-dZnK-BlRM_4GRPLvpQYw0MKQxVKcJ7yKRd_e-zZDl641G1zE0Ycntte9U51JkwgERzHdbs8W967tdoSGRYsU72lcKdaFTxHBrYzrlzdWxFGmLdTZ-VrfjJbxU8ZSifRnHFXAknLaCyVsZeBoBrz68sEEteE8oTFZTGUKgtkXhP3iEwrtYvLDG44fO0lAd9tOUXh23doZHcncq9zUqVe99z5Qkc3g1DieeaHBj2r8AJKHwqwI_uUWhzV3dwnCK-o6vYGoff1UztGHiJT5Zgol-qwIrL_3o6CijNZzMEuhShH7zTKMZJVdc79juY3Q9LMSRVQIH1qpsV56ISOOhy9pVamYEDcxpv_oosPs8b77bu5RztTx-BwBhsnWxz5k6S80toGtthZAOQsPETUhB81m_1ZNjkmmuzDrqK2ITA3CY-C2EZVQf-5mD4K0jAwcbtmw57PkP65WI9fEumGTGVHKnUQ3UUyl1BDm8ZGiTLtk0BjK4gEdi2CGbre-nGARfwGA8igCDqyZqw2pEg0AFV2ELu7Qia25ZOSt1UyCSX5rReYjEIG6L5IOiM34p3sISsOFmi59O4P76Ls2oNfY32TTYulDWPX2R6z8mvfBlgAHbtNh17FkATnbv7BuRU3_3ZhTIrbK2I7hROn80vVM1c_A8aMmBCLoINfifHdhooNOreTnhGSvA2uillBBVEhbsU9eXf7SqsVcZl0jA5NqlJ8DsLKk1Qc54qudZzPdGOzB9aFJQfPAubJErXFaHr9gmxlJOmoDYouYzArpX939VL0IkCt-KbK997imQ2UFtyRYtDC-7dVya2Alg4ua8XyWHtezY7ADZBcMSdpwoXvZHWd0jCBwm5mJ16oMW4dao-M4xNM5Dp2NzK5QCHPt77LxqP3fNdFI7cjsZSM32tLwxqQfvvyrsBYemE55xISrj0VGj8fNS4qBC4T8DXiJ6LJgh7-wXrPeGihvIfhpATYjdQSjyf-6TXSB8HwzkLVdoWakokXBqpWfLMG8Qn2UNNuyr3heuOqOxh7kIKdr0ucJXAd1KlyVZoVGkMZIg74pqR9CSuB82889jSqYfsBKUzzN4mk8dE9QLZDu-11nnIrlD2JmZiJYaq49ZrMpYX8KZtvM1YjdAyXHrM6WPFBs4MXco7DZJUCTgadgmSDZ5FHG1sry4Wl3bvjSA1BwSk9naQFYeKRHmANEBJXfDV-a2HT7QbY=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_04507193d793fa550068ed13e71e5c8196b9588faf7b4f61de',
                            'arguments': '{"minimal_output":true,"order":"desc","page":1,"perPage":100,"query":"user:pydantic"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'approval_request_id': None,
                            'status': 'failed',
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13e71e5c8196b9588faf7b4f61de',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_04507193d793fa550068ed13e71e5c8196b9588faf7b4f61de',
                            'arguments': '{"minimal_output":true,"order":"desc","page":1,"perPage":100,"query":"user:pydantic"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'approval_request_id': None,
                            'status': 'failed',
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13e71e5c8196b9588faf7b4f61de',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_04507193d793fa550068ed13e7caa88196ad5ec0abaa0315a3',
                        signature='gAAAAABo7RPwhF0RC2doViW9PTmEGAAWqjmI3o6aRABDFpqWMNYpySD0RtxVltuNWYnocK5_IIoA35C4Y0k8gQLR6cHF2CzpC4nFO4GCZpm4awByYiWkk9q3AP5PICjIco5BdCszCZBsOoaAtbUVI_wjj0Qomj3D1axk0frsXvTx9mzeEnjeHQQxS6V6Z53UDcwnfQ6pxXCzUDcAb2L5VAp_pw2lKOf34npFRQ11tkuU8djhuapZjNGXI8cyaX7bZifVoIaIxELGsIA6p7pbDxaC9EaZPttde9n3vP66Jt-qJfsumebs-lz_X8aosVVwszVoKoK0XLjzfTvkRA-S_Rjl_pJBvfUSlOJqMDNc2N5BqwrHwzS8B1Dqkx_JzI7zoVIgPTNwcKpIMBWSDjUbe2vjGOAyjLJQYLO9ah4Q1curobn18dKrOay3mYhoyKq28cx78lp-zwQvABHYW4PUcpj29g9HCLTZZQz2UXb-OSqy_QHEsxzlSlXY52SneDAW5i1euMRKYGHKZ6plVlLKHHv8PWsqDktb68lMZHJLmEpNPF7dCn3Bw39QrcUhO7-yURNgad0GflXGZG7yzeQ1-68cYVECD1Kvri3I5F_aQO9Sbh6a1iV_X8_xLcITi-MSOHfKzL-aqdih7nEWJeYgsgk1bYeDvDJgrGOqckNQjjX5OEBN73X-vkEPaCtBLswJ-Vr-ENqaPGVlX5zNUyOFhzFEdZUcZ4lFGYgeTapWOsW8m0bowUouvgXP0-7neRQ25nLnt95gJjdZEYE5Sl_SuRMz016rAcNIoa7Rgw1-u7sqoKtDf9YZ9mrxXmguV8NvYQVIgRS8UvxpJTtWYQIaCKuxm2_1A4YV4uGdSpRnd5RBhVpO8tphWy3tJNUxaZIa3E_DersilTfQmJEZbVGlI8b0-OiyZvlkRAb5STX4z04-3aoiKx_D44k=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_04507193d793fa550068ed13e84cec819697e55f0f6d7baec7',
                            'arguments': '{"minimal_output":true,"order":"desc","page":1,"perPage":100,"query":"user:Pydantic"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'approval_request_id': None,
                            'status': 'failed',
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13e84cec819697e55f0f6d7baec7',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_04507193d793fa550068ed13e84cec819697e55f0f6d7baec7',
                            'arguments': '{"minimal_output":true,"order":"desc","page":1,"perPage":100,"query":"user:Pydantic"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'approval_request_id': None,
                            'status': 'failed',
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13e84cec819697e55f0f6d7baec7',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_04507193d793fa550068ed13e8fedc81968c9fe669111ef8b0',
                        signature='gAAAAABo7RPwDENXNHyUucg1eGSOTdXv5VI0iMY2uLpeev4ModlAMbFZKrQYiSbAgmXgRSK80Uwl1ykNsxDjWMdcvlVeSQ3t40RjcfBpPfkZS5exjeNiHHT4I8p1M6-d9JhXY9BjtdATrArGIhchWnWTs0s1RfQbimhKIUlFLDNqty35mg6qZbS6lcGCTmc_yFuqTAlgQIEcsYDhDo6GBwHIJOKAoK-U7ZgGhaXi3TXfscV7qyqnd9O4UBY3GAYX4VfBem6y4QgCOamr7UA3-KyouI3m0XQrKcIxCB1KD9kY8NbU9ZJtpGOBq_yccjZ8hLTpuSClgiIIPFCryRzFyGsmWLgXNpH1eK3W5na2BTxMyopZYHdEKg-pce-cKoVun590by4G7-RQC7BHIB_SOzs7s_dvjFIelh8KrI0nTVXPDnNj4aDWkxXzFw_eWHTlEc1--ex34MPdQ9EIxTmHfJl8prE3Ld3SQd1FXl-TqpwQnI42MuPMRisAXP6mwfmDR-tkn0xvcP2SjLLs2BFEuTjTaxzHZUpnKMCw_y67_MdNZW8nlr0Ameo-iybsW_tRNf7QmQKYfDeiq43kx-dE53FghotnomFH-bMnDk5mdbfsxj7Y6Dd9m5slOD96s_D9SE9JRFk6w3Kok6E94d5_8vx9pKmVGlq7MCoPHxdtMjuiPdYLdUIcn-Tb5bZXtU2ISlwD56ez8AIB5D4Xh0k_hN5CuI6Zp6Rx5W6bJiIhtt7s51RyKOE8vOJCl2tAx5u15bCSwcocyGRoNxv8xzUYhDCuEeN8QSyxY_qSPJUJk6d0XtCa8fUuWEvuxgE12tFaWZjXunZ8y7u6352lfzKgYukB-5tMwrNXfa0wDhtvr1ZePBwy47A7vbgk1jBt1NMljSD0fs_AmaFv1pA9VwElq2eTKRR2PoZKNKLS_GhVWjNKKlXKDPmNYEcg5Qf4xa3o8lhb_hl4bHfUfbs8eXmnHLIkeiCQy7ukdhgDCDVKt4ryzjccCZKC5TaIv8tUDfkuFrQ_M-bFxBzeeTFGaTqTpUeMhjR3eUtfnAo643IxlLYHR6B4JmqaXkWvA0qtCj9U40URpsAldcXMC60E00SzsiagZUGQrw3dpHRF12d9KAMKd56lTvRRTicph3UWYjjt82929kPCBufb1lWd83MCTlDOAcqr00LfO-NJXxkUQNcpT_Pb1qYV8u9tC_8I1uBejfhVmK6X7vRN4eCGGVHqwZug3I6s4Y3ClkTZFfCqjJV908wjSs9TJtd8XJd8nn6ZixmuvDZs8080z3EWaucj2WeaDxjSAkqtsM_G420OpzLqWFjNH_Eu_FT5ofRxb25OfJdy4GJ9-G6AFEMfwjS2DGFuIWLsNY20sR34f7RC_zfW8miN7RRoIXMVl9xVGafoKU4s9z_jVI3qmAQFoWP2fKV2naA-0LxR2-YwRfGb_Np7ThJlVkSE7sJNc2-Chy3IkUijHmoa65uX3EAk26UyLRzjkpT1yf3VBZM7tfYVMctxX1BMn7pWFB4Vx7QcndfXQXBdd9j9GvgluWnJF3j11Z9Ix86k1ru7OvuT8ZvnUgaa8qnhmPw7swqPZi1ucC0wGWa4AGAf8icjSKDBPH2du_N3sDWkp38BxKZN3xj0kyXzZTTwa09yNoPFGjtI1AyRV6boUlnx6Pi1hiTIl6RTtyx21rLPRI71GGFGkr92TIG_xKtQn7DvEIDxLkyl_Mt0MXVr1pibSydYcKMmpxM_k1Mg6_nzbTZb_6MMZtQVV0JrOt5e054I8iBvK6k1A5625UKdKHFhEjO4ygQHwfWqUBMFw8CZ13FhIHmIR04AJOwvgrGWgIyeyemLkVFa3tBPmSPiBUg0Zisg-8I9E_34rplqj-GiQhtMEAWCUYIemN56LPaI90M8w6kiJAw4mMn5cGLivlFOp_ny2t0ttwAahibffZy1wrbVDKC32jHEbkf8-RoTnXG4NKe7ZXODPmfqvAcZWAbQgdbjv6w_NR_rVYu5Exfr8egKauuiqJm3I2gumKlQ7RALjSQs1ZoMAftz9g8a2cBtB8pIyoqynp7sXEPTY9yVmPtIuUjK_n2conlxTgVs-Tf2coPyQpXI93QpZ7WFXXV9LBmqBWJSjZkJlYjVIrmf382ly-WvUeqPxXJp43kmxdR-YFVQH-DyBNfWFJAiDh8nXlfukn9PsCqHcZHlg2Ejeq5ZJkIrjxE1_Ql-9l9ndeaiSmo4dTjDXh7LSa26AzvF_Ra7itBza1LHx6E9OGYKf5CFeAAdocV7zThCHwVb6Q7VMIlm6yX6Yk5geE4qqiOYB24CsJYXV5oDOrhyfQ0v35-FukeTEPRdG_NuXg-_29rBz9WYjXekvsmaJ_I4kilBKEfBVCPHKTHN2R3UHWAWxRicrwKJG70W7gdbbZFdnwKcV6OQvSRbi6ffuwYcWAkZu9HnMwjVvNTwRqh4A0LX8rCTyYuzbIUAvkr3pHKOEHMLZa3fmveOauA_PQLpX4q8zDIhjxqKqYkR74pFym3Bsu4Vu72TCb0x3YHcM9SOEwaKiZplZmHtYqJlxJA_4KO5xaDVL0Rvp_QfDegBj15BzUemyZIet0n5fF8MkzuslWBEkar_NodUE6Qsq0Xgx7Ph7yiuZSdqQy8W-r1KzME_FhserB8plVYLfW-gU826FXU5MS9ulk1ROi6oI8fsmUEPcSU8Qt1ao6e6HUARrMusInIHjilwF3b9BPY1apfeh3nEfMsgvjATxroITsiTnVORNlcyr9DBIUfbj3A8Fby9ZYYLp8K-VBIc7w1xckiHJEotdGSsDBDXUYNiZiC7CXy0TKws2xpxY4GFXw7HBC-ImgHEDZqzw7GjEJFnNYbNnjoRR54C9lG2o_QGtl89VHUXXqUVCMesAs6E9BqdX3u6B1fQlTfJ6zPN5V7NDIz3rspEYFa2HUIF0SzsKhANovwwu9PYGkzwTC-rCmxN1LCn_nRI5BgfnbtTxPRDOHoGgLCrvCRCKXdSpzdmaZ0a-Ivp2HufIKsEJFPdOc0clPocrmhIrJyX3FVTSNin7D-hBxOONVCHupHE5zxQCLqvKM_p8-dEYwuusMEyxr8Jg-dBK2B29WM4-ee9XHqFc8UL0I5on1Z2ujNRCKXTqfW5KrfyVClnG4jkm5NeJmPqBYdjnFqJnr29ZbULWFm9ku1VcE_61z6eJn-4JEFS8-1fkJtRpR1807yZ9o6ZmxtE7pGL1c9wiTZFtOg6N5IXQDmqaWnvRZkluzhZbUQfQszlgrq_h12uvC5bmcmBwnKYLRO3TwL77QmAMHvfSU79xPdawXfdmYZ3Ux0NgEjzbLIrbngMxJa8Y0UJGE5g4rao6apAj5U6UoVZScuO5et7Prkujfxodn4QbfOZipYDLitm9Wl7-Oeg-y37iyjheNeEqGKURI-29cIgZ_MEsG82Hvn2AAVYay69Qa9wemwz0ICYEdscv0iXy3weEXfpt1T-IB1OrPHnR2triX2Nj2jaXlgaOKZC27wHcREP8Yjb7IxjFECA7toB0I9Sbiqr6QGytWtuAbCtlCbAnVhQZs8BRCXn3uyGdUtK_849RYFdRq18X8HRGz-x2pb7oaryKrl-iGtE29IMACtHbDXHfRnUn1R7uaCbkrHyAsp-QjMClXkWraSGlmtxFBtq-HDyJvscvSfFuZumwBDZLtRXMlIwBV0ueyTGHyBA-g39rYZyLmG3Y5gf2TsuAqmn5lqYGWmgUP3AUI_O2EMbmCR1QYlCSgbz_Xp9VBzZrNztMxP3W6-uFtengj4uSH4Gxr9QjFwDTBmJtTexfUbKJFrQUemZmnqzb1Y2jIJl7_g3gFWen10mn6s4Bgt0ICqd0sxuhwOyrdFWt2CSb77gnSdbflBXCQa5xdLRHfOAcgYAseq4a8Pjve2QtjUdCAI1yUn7A4D90JssIZ6uRJYrnE-Gi4Di2b4CUBhsEhpvvAZMj-Shp9EC5bj2VwqQUHIB8iflPKs0brzwKGmF_6LgEyMb32f4sG_pkkHzdWXvTnhwNI_g0WxhTHBlBXlmatIEnJL-5d9UX0duYCqv5Il2V8E4nLq5RVOXiwyhI74N7optrqR6vhUMU-ntuTshc01k3cqEo-XRso09X7BY5AHnoLI_QaQWLFfBOmnF8VM-ecuWHCWaZHACgDkZ9_RT0PMXd8YrWxf1Al9a8kkp0FVSq1gG_s_7DzgpQ8oi9Twp9fhs4SVtjpg2JZBaSjq3e7G8uvllXiuI_k5uZ6p4zdfH6n_GB2plSAXOiEJRE9cCyGfOYxf54d5MprslS8tLUCT_5hTz_6ciPgGUXI1JhypORYMZ1Ah3OrWI9YusIkO3LLZ2ECh1MSO_uELFMNHDAB49x6GGsZYsZv6xW0s4Zav2yLsyap1GJjNmw0uUXxMwRZf9_A2TCnxB1RS0Ds0tfRFlCeeXHDfT8OXT4aN1gXF33736vOUOrKkfuFYAcbvWl4naLu5siVfZUVpeIZsn028fhvuhOmsJUcGGcSWtsT1gbmcRDq9WjV-k9qcPOIgQjrhyOTrrukV7AWt0cnmTF3qYptviW9OAtUOvVk095YsAOBFiT4l6KiEnf3k6iW8HwLCNWHgEpLTiyfeix_Ax9ff92JrMkU-3uOT6jTHit9Cs8pAyHTUvuBEHsT8ca-lAdlqYiNAmTM2lF23ECg3z6WK8dOyua-ZZ1fotW1u3DsMWX5hBJI5Dc6P_VSsGQMj7KLgeEgeV_n_VhMxUb__yZ2ezfQsImFtZrSIztLBmkwWHaoB_KaNQVgWX9pbEbCfaRNRrY-ZDVLEEgd8my41v0tqnBQ-iN9n_32OJuLFWCxlV9wNB3vw5mqSskIxvxMJfcgL3Gy3jZF3R14xRjRgnLX58cmNJjKqRt6fQX7ClI6GfXUz23dPvt57GZios46E3dMZuVuZSpLQ1g3xneP6pBDc2l5fFHtRtDqG5CrUVyJzl1sDGgDreyfIgFZ7ixUtR8ptf9wEG-G_JP6YUoomSATnli-odJxpHySoLstvQmPfD8rWzqiSDKsBKejXZYIVnsv4Z0xGF5IwbID6N1xVrW1jE5psusQF73uCxHlOEXaLuS1Od3BMyp7zQFEhjjkVb9k0t7GY-_xys0kwInyeL34ov9TZ8hO5cii6HgXv4Gg866_MRghXse1P4z7Y5CR6ieHwls6ue996_MQY2gpZBnzvzaSUq096z5R5wp3bsGMIPLeaLptTtaJF7Cj7DylfmdLJOSVi86a53anitDCk8VYfmn7vi6HwYHAgVF6-84MlEKzp-5QeELmOZyVLjEv4IuyG5bvJglwQLhoTecSNS8haaWe88UTn_RfOTZ51aM9HJWeaxa8oONxdXq0UihgFPBMLd6UsPAoXkhutK7tzYOyTuzOGFaqSw0L4JtT9fn-hriBEtFWy6Cofuc4tpDsJuKT_IiaIX6JaU1vP1bflmcNuiOFd0K_WvgaZK0YvF2rMT7QmSiLGReiSGThb1GGS-IGU1wM4FJ5RoMh598hM8bLfqcQYZcIJlR7m9VHo2qyPZUiCtgFzUsqvmQoTYCYRBVn34CM9UMJGT3gEXpYBVvW-FVWTeNWR9a1QFQbEdlS2tVzyb34SwLaq9TZHns2f21PIOp3zl2Q4o3XQUeFPyYQVkBypUL4Fv9ZwoBrUWtxCoEd859dsgv93XfYozZtbL2WuZYXlzAW_5GHxwpUcrGScyfF8BF8U_vfGG2alDzRpGN1t9A1FTjjktmeOBNyVeuGzjilGq1_at-d-tiltXMyK2wL-I6Hp5xBk-MUj9mD9vVNALdWrFywfpJHJayRIlW0ML86vFbwG4YIjOM9gBOi1Pz3cQ7hQFN4fv9IZ4oooIo6V2jq-FL0kTGCSDv2CTU0xNj48ta8uSzhLO0kf4',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='As of right now, the Pydantic GitHub account has 17 public repositories.',
                        id='msg_04507193d793fa550068ed13f02d7c8196afba59eeeb18c096',
                    ),
                ],
                usage=RequestUsage(input_tokens=871, output_tokens=1392, details={'reasoning_tokens': 1280}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_04507193d793fa550068ed13e0cc488196a6f4c49747240ea9',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_04507193d793fa550068ed13f269d481969f71e83863052982',
                        signature='gAAAAABo7RP5r0iJ3BdiQhxeJ1LDClOf-qpGHbMTzLVRTRn5QQAIeCk-6-DotZQ53w6BU4pStJO1RTufuDwwtiHu1NVXaqcB3sT8LUJa0KzmxHqu8p2idEKktPEz1LwgpIVmgRliGz7oj2UBA6JoSxT38TXti8moNQBno9UDd95dqOcEkKsORyo_b_q4rbL0i9viQKiLvsLqwNow-fsgnWeOdqoFz5RShxZBzHBEDfqQcACelFbqb3N9gpoILi3lPq15AdiO1yqo8vCDke6SZuwn0kcITbWGDCOx0174LD6_p-aRDxQJWtiXM52HuyRAiiavc95jBQ6Dg85QjO-vliUqOSfYthbrNM2N-Op2SceZw9zfGjqDcNJGu5Y5qL6ZLDHI6x2ponatUwjUN3HQ5qFtKH0JBcqajMKKcB9PUqrTd4J5GdJ9TC82kxT0Pms_N_C_mg80Mm3ByhW2L8zo34e2zha6Jqy8RV0kSprSYuhSSwEQ4XcWwIplOB7RrGKSwhDSmMAJFd7496cbttdzXr4jfpetcKg0F0pRlousKnv_026nbzS2gCLbeFAGpGlSr0M7iQL7FB9gLO08SxHNUBPylHaiCmJWiZQ7xFH0IwIzzdtjkYCMrVLf7nNneXO16s8nR4ltBxfsfRdUzVNZkEiYr-gqFSUIgoN-6F455aw8LwVklfszu9trLsZCoMgIFTKzUJDJAMelHqgbEhJFCxvMgL6wAiFpf6IIK0Z2OuL1mYK8nm8j0MO7HabG6adaQs_V2Bu5Y4YhshTZOGWoiT6jZLDMYru33afcv05bJECSJqVXO5NOAPyA7G0zUvqkFKo9tzINuIH80_vqA7ATpclgecHo-0u0YDo-4gKAIN_gt6sFHOwjGtAK_FpjZURkB5pDiYwLbu0gInFf9viQvNFIJ9vWjCX6NLgkDVAKWWbPXSLyCmZJfZyyhw6WPTHWZX0Oz3jl031T5cVipYCt-1WlR8qce_Dpb4Wzek79uGGpID6rw9urKzhuz8GK_meChgijzptnipGdmCUFi5hilbszyLfmOzudBW-E4qIBts16E4aHZDaFqfxJFBmblkR_fzyvuYKhlfm4b_1NmZ7MWFLNc-zviHyU2khWrMhf_IJqvD18u26A47JPBCWHzgSEo6mqbfMbw-lzOhEI53oH0FZRmd_sC50rpL8qguJYWIa3GBvmIgtYhFSaj4W1bRJbttjBxoA5W0Ny90ZFHsKXcGiDzSsQ5k9PJDNnbV3rqcR8RKZY5t7MN7KYfLAZD0Gc4GEU2kN4OeU6A9PSlgaqd_NSUEE3fI7rRtMywkOrCVBESRDk2bj21KYZhyeMOuZxk-SUc0urjDlJpCc2jDW9tFxng5mTORRllB4gMw_fykr5fA-KxJARwDHN4LX3zXsmJdyhn7RF3_il1_nZsuJE1SSrxWD8ACS-8Hclg3TZ8kpHohthE6tAeK_99nLL1D2px8nKOvPMljAYCNYpMy_g_2_NeYV5r3xX1leaeAG0u7d2ssBeElyzAwnpSMPtI0WStN4lcLfbs2DwAJtl7l2e-3pZtv5TODelMAYrKCyG6M4tb_vnlCpQAzw20NinJE6z8N9WT1-NO8pF4zoHzui9a9DNu4ks_j3bsM6MsYf6r7iwR0NSbpnR2RY2KngQq-RopylUTu4V4I8YgVvjufEKdkm4AMBUnVjqO_5urU6vluktGUl6x5b24vvsG6t4hUBH2TPlnkKE-cz8QwEsScIe03qMh241p8ZEMjCVSteeqpjHJCuQUAYAdCYzNidRYmpMV3HBAU8vB4x6cLX-n3Y680a44lYXw9Uefb-YPk_TXYCC4eWb5kahYZWvvmIdbA3ey-8A4nVliaJHvwedSqkMB9cGMxAtO1YbmdJdwuHG-Vmuhm2HJWjBBBFeLDdJkVaIvQN4dvb8EBlKeijTsoGjRE_ozWrgJ1LEuemnurp3O8pdhU9mLlrTHfqC89-AWRsEAUcye9pK21aWqOgXUDCLGR5WEFvuLo4-PxsdH67CjJTAakhaOom4WHwCARoXGSkRzGLNwOD48QZYHxKThnsXsWLUlFZkUoY8qxIYuODTsYMTUZnybBbM4lggf-ckL26LyQxFXHlk5KfEq4sQ7ebrLtg4CwjHYoeTYIZFUFSJqWy4_gJxTEfRTC_c2bSxUeisXLGXfdlGmDJP8uZxmvrxmyhSurZabfUFlTqYjCCYL9jwZlwmJTPGFy3EZgxHnL3oyTVE6WDswvDsllNPjTqX3e4juXViWikzY-6iqnAv419E2O85ZUgTPWk=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_04507193d793fa550068ed13f4e9188196ab8090591caf215d',
                            'arguments': '{"minimal_output":false,"order":"desc","page":1,"perPage":100,"query":"user:pydantic pydantic-ai"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': '{"total_count": 4, "incomplete_results": false, "items": [{"id": 818331198, "node_id": "R_kgDOMMa-Pg", "owner": {"login": "pydantic", "id": 110818415, "node_id": "O_kgDOBpr0bw", "avatar_url": "https://avatars.githubusercontent.com/u/110818415?v=4", "html_url": "https://github.com/pydantic", "gravatar_id": "", "type": "Organization", "site_admin": false, "url": "https://api.github.com/users/pydantic", "events_url": "https://api.github.com/users/pydantic/events{/privacy}", "following_url": "https://api.github.com/users/pydantic/following{/other_user}", "followers_url": "https://api.github.com/users/pydantic/followers", "gists_url": "https://api.github.com/users/pydantic/gists{/gist_id}", "organizations_url": "https://api.github.com/users/pydantic/orgs", "received_events_url": "https://api.github.com/users/pydantic/received_events", "repos_url": "https://api.github.com/users/pydantic/repos", "starred_url": "https://api.github.com/users/pydantic/starred{/owner}{/repo}", "subscriptions_url": "https://api.github.com/users/pydantic/subscriptions"}, "name": "pydantic-ai", "full_name": "pydantic/pydantic-ai", "description": "GenAI Agent Framework, the Pydantic way", "homepage": "https://ai.pydantic.dev", "default_branch": "main", "created_at": "2024-06-21T15:55:04Z", "pushed_at": "2025-10-13T13:43:12Z", "updated_at": "2025-10-13T13:43:17Z", "html_url": "https://github.com/pydantic/pydantic-ai", "clone_url": "https://github.com/pydantic/pydantic-ai.git", "git_url": "git://github.com/pydantic/pydantic-ai.git", "ssh_url": "git@github.com:pydantic/pydantic-ai.git", "svn_url": "https://github.com/pydantic/pydantic-ai", "language": "Python", "fork": false, "forks_count": 1298, "open_issues_count": 326, "open_issues": 326, "stargazers_count": 12888, "watchers_count": 12888, "watchers": 12888, "size": 89097, "permissions": {"admin": false, "maintain": false, "pull": true, "push": false, "triage": false}, "allow_forking": true, "web_commit_signoff_required": false, "topics": ["agent-framework", "genai", "llm", "pydantic", "python"], "archived": false, "disabled": false, "license": {"key": "mit", "name": "MIT License", "url": "https://api.github.com/licenses/mit", "spdx_id": "MIT"}, "private": false, "has_issues": true, "has_wiki": false, "has_pages": false, "has_projects": true, "has_downloads": true, "has_discussions": false, "is_template": false, "url": "https://api.github.com/repos/pydantic/pydantic-ai", "archive_url": "https://api.github.com/repos/pydantic/pydantic-ai/{archive_format}{/ref}", "assignees_url": "https://api.github.com/repos/pydantic/pydantic-ai/assignees{/user}", "blobs_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/blobs{/sha}", "branches_url": "https://api.github.com/repos/pydantic/pydantic-ai/branches{/branch}", "collaborators_url": "https://api.github.com/repos/pydantic/pydantic-ai/collaborators{/collaborator}", "comments_url": "https://api.github.com/repos/pydantic/pydantic-ai/comments{/number}", "commits_url": "https://api.github.com/repos/pydantic/pydantic-ai/commits{/sha}", "compare_url": "https://api.github.com/repos/pydantic/pydantic-ai/compare/{base}...{head}", "contents_url": "https://api.github.com/repos/pydantic/pydantic-ai/contents/{+path}", "contributors_url": "https://api.github.com/repos/pydantic/pydantic-ai/contributors", "deployments_url": "https://api.github.com/repos/pydantic/pydantic-ai/deployments", "downloads_url": "https://api.github.com/repos/pydantic/pydantic-ai/downloads", "events_url": "https://api.github.com/repos/pydantic/pydantic-ai/events", "forks_url": "https://api.github.com/repos/pydantic/pydantic-ai/forks", "git_commits_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/commits{/sha}", "git_refs_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/refs{/sha}", "git_tags_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/tags{/sha}", "hooks_url": "https://api.github.com/repos/pydantic/pydantic-ai/hooks", "issue_comment_url": "https://api.github.com/repos/pydantic/pydantic-ai/issues/comments{/number}", "issue_events_url": "https://api.github.com/repos/pydantic/pydantic-ai/issues/events{/number}", "issues_url": "https://api.github.com/repos/pydantic/pydantic-ai/issues{/number}", "keys_url": "https://api.github.com/repos/pydantic/pydantic-ai/keys{/key_id}", "labels_url": "https://api.github.com/repos/pydantic/pydantic-ai/labels{/name}", "languages_url": "https://api.github.com/repos/pydantic/pydantic-ai/languages", "merges_url": "https://api.github.com/repos/pydantic/pydantic-ai/merges", "milestones_url": "https://api.github.com/repos/pydantic/pydantic-ai/milestones{/number}", "notifications_url": "https://api.github.com/repos/pydantic/pydantic-ai/notifications{?since,all,participating}", "pulls_url": "https://api.github.com/repos/pydantic/pydantic-ai/pulls{/number}", "releases_url": "https://api.github.com/repos/pydantic/pydantic-ai/releases{/id}", "stargazers_url": "https://api.github.com/repos/pydantic/pydantic-ai/stargazers", "statuses_url": "https://api.github.com/repos/pydantic/pydantic-ai/statuses/{sha}", "subscribers_url": "https://api.github.com/repos/pydantic/pydantic-ai/subscribers", "subscription_url": "https://api.github.com/repos/pydantic/pydantic-ai/subscription", "tags_url": "https://api.github.com/repos/pydantic/pydantic-ai/tags", "trees_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/trees{/sha}", "teams_url": "https://api.github.com/repos/pydantic/pydantic-ai/teams", "visibility": "public"}, {"id": 1052844101, "node_id": "R_kgDOPsEgRQ", "owner": {"login": "pydantic", "id": 110818415, "node_id": "O_kgDOBpr0bw", "avatar_url": "https://avatars.githubusercontent.com/u/110818415?v=4", "html_url": "https://github.com/pydantic", "gravatar_id": "", "type": "Organization", "site_admin": false, "url": "https://api.github.com/users/pydantic", "events_url": "https://api.github.com/users/pydantic/events{/privacy}", "following_url": "https://api.github.com/users/pydantic/following{/other_user}", "followers_url": "https://api.github.com/users/pydantic/followers", "gists_url": "https://api.github.com/users/pydantic/gists{/gist_id}", "organizations_url": "https://api.github.com/users/pydantic/orgs", "received_events_url": "https://api.github.com/users/pydantic/received_events", "repos_url": "https://api.github.com/users/pydantic/repos", "starred_url": "https://api.github.com/users/pydantic/starred{/owner}{/repo}", "subscriptions_url": "https://api.github.com/users/pydantic/subscriptions"}, "name": "pydantic-ai-chat", "full_name": "pydantic/pydantic-ai-chat", "default_branch": "main", "created_at": "2025-09-08T16:16:54Z", "pushed_at": "2025-09-25T15:30:52Z", "updated_at": "2025-10-08T15:17:58Z", "html_url": "https://github.com/pydantic/pydantic-ai-chat", "clone_url": "https://github.com/pydantic/pydantic-ai-chat.git", "git_url": "git://github.com/pydantic/pydantic-ai-chat.git", "ssh_url": "git@github.com:pydantic/pydantic-ai-chat.git", "svn_url": "https://github.com/pydantic/pydantic-ai-chat", "language": "TypeScript", "fork": false, "forks_count": 3, "open_issues_count": 0, "open_issues": 0, "stargazers_count": 16, "watchers_count": 16, "watchers": 16, "size": 487, "permissions": {"admin": false, "maintain": false, "pull": true, "push": false, "triage": false}, "allow_forking": true, "web_commit_signoff_required": false, "archived": false, "disabled": false, "private": false, "has_issues": true, "has_wiki": false, "has_pages": false, "has_projects": false, "has_downloads": true, "has_discussions": false, "is_template": false, "url": "https://api.github.com/repos/pydantic/pydantic-ai-chat", "archive_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/{archive_format}{/ref}", "assignees_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/assignees{/user}", "blobs_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/blobs{/sha}", "branches_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/branches{/branch}", "collaborators_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/collaborators{/collaborator}", "comments_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/comments{/number}", "commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/commits{/sha}", "compare_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/compare/{base}...{head}", "contents_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/contents/{+path}", "contributors_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/contributors", "deployments_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/deployments", "downloads_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/downloads", "events_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/events", "forks_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/forks", "git_commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/commits{/sha}", "git_refs_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/refs{/sha}", "git_tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/tags{/sha}", "hooks_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/hooks", "issue_comment_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/issues/comments{/number}", "issue_events_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/issues/events{/number}", "issues_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/issues{/number}", "keys_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/keys{/key_id}", "labels_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/labels{/name}", "languages_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/languages", "merges_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/merges", "milestones_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/milestones{/number}", "notifications_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/notifications{?since,all,participating}", "pulls_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/pulls{/number}", "releases_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/releases{/id}", "stargazers_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/stargazers", "statuses_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/statuses/{sha}", "subscribers_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/subscribers", "subscription_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/subscription", "tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/tags", "trees_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/trees{/sha}", "teams_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/teams", "visibility": "public"}, {"id": 1040842163, "node_id": "R_kgDOPgn9sw", "owner": {"login": "pydantic", "id": 110818415, "node_id": "O_kgDOBpr0bw", "avatar_url": "https://avatars.githubusercontent.com/u/110818415?v=4", "html_url": "https://github.com/pydantic", "gravatar_id": "", "type": "Organization", "site_admin": false, "url": "https://api.github.com/users/pydantic", "events_url": "https://api.github.com/users/pydantic/events{/privacy}", "following_url": "https://api.github.com/users/pydantic/following{/other_user}", "followers_url": "https://api.github.com/users/pydantic/followers", "gists_url": "https://api.github.com/users/pydantic/gists{/gist_id}", "organizations_url": "https://api.github.com/users/pydantic/orgs", "received_events_url": "https://api.github.com/users/pydantic/received_events", "repos_url": "https://api.github.com/users/pydantic/repos", "starred_url": "https://api.github.com/users/pydantic/starred{/owner}{/repo}", "subscriptions_url": "https://api.github.com/users/pydantic/subscriptions"}, "name": "pydantic-ai-gateway", "full_name": "pydantic/pydantic-ai-gateway", "default_branch": "main", "created_at": "2025-08-19T15:29:12Z", "pushed_at": "2025-10-13T12:46:43Z", "updated_at": "2025-10-13T12:46:47Z", "html_url": "https://github.com/pydantic/pydantic-ai-gateway", "clone_url": "https://github.com/pydantic/pydantic-ai-gateway.git", "git_url": "git://github.com/pydantic/pydantic-ai-gateway.git", "ssh_url": "git@github.com:pydantic/pydantic-ai-gateway.git", "svn_url": "https://github.com/pydantic/pydantic-ai-gateway", "language": "TypeScript", "fork": false, "forks_count": 1, "open_issues_count": 6, "open_issues": 6, "stargazers_count": 52, "watchers_count": 52, "watchers": 52, "size": 883, "permissions": {"admin": false, "maintain": false, "pull": true, "push": false, "triage": false}, "allow_forking": true, "web_commit_signoff_required": false, "archived": false, "disabled": false, "license": {"key": "agpl-3.0", "name": "GNU Affero General Public License v3.0", "url": "https://api.github.com/licenses/agpl-3.0", "spdx_id": "AGPL-3.0"}, "private": false, "has_issues": true, "has_wiki": true, "has_pages": false, "has_projects": true, "has_downloads": true, "has_discussions": false, "is_template": false, "url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway", "archive_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/{archive_format}{/ref}", "assignees_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/assignees{/user}", "blobs_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/blobs{/sha}", "branches_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/branches{/branch}", "collaborators_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/collaborators{/collaborator}", "comments_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/comments{/number}", "commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/commits{/sha}", "compare_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/compare/{base}...{head}", "contents_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/contents/{+path}", "contributors_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/contributors", "deployments_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/deployments", "downloads_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/downloads", "events_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/events", "forks_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/forks", "git_commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/commits{/sha}", "git_refs_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/refs{/sha}", "git_tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/tags{/sha}", "hooks_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/hooks", "issue_comment_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/issues/comments{/number}", "issue_events_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/issues/events{/number}", "issues_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/issues{/number}", "keys_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/keys{/key_id}", "labels_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/labels{/name}", "languages_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/languages", "merges_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/merges", "milestones_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/milestones{/number}", "notifications_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/notifications{?since,all,participating}", "pulls_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/pulls{/number}", "releases_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/releases{/id}", "stargazers_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/stargazers", "statuses_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/statuses/{sha}", "subscribers_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/subscribers", "subscription_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/subscription", "tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/tags", "trees_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/trees{/sha}", "teams_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/teams", "visibility": "public"}, {"id": 1053149228, "node_id": "R_kgDOPsXILA", "owner": {"login": "pydantic", "id": 110818415, "node_id": "O_kgDOBpr0bw", "avatar_url": "https://avatars.githubusercontent.com/u/110818415?v=4", "html_url": "https://github.com/pydantic", "gravatar_id": "", "type": "Organization", "site_admin": false, "url": "https://api.github.com/users/pydantic", "events_url": "https://api.github.com/users/pydantic/events{/privacy}", "following_url": "https://api.github.com/users/pydantic/following{/other_user}", "followers_url": "https://api.github.com/users/pydantic/followers", "gists_url": "https://api.github.com/users/pydantic/gists{/gist_id}", "organizations_url": "https://api.github.com/users/pydantic/orgs", "received_events_url": "https://api.github.com/users/pydantic/received_events", "repos_url": "https://api.github.com/users/pydantic/repos", "starred_url": "https://api.github.com/users/pydantic/starred{/owner}{/repo}", "subscriptions_url": "https://api.github.com/users/pydantic/subscriptions"}, "name": "pydantic-ai-temporal-example", "full_name": "pydantic/pydantic-ai-temporal-example", "default_branch": "main", "created_at": "2025-09-09T04:05:06Z", "pushed_at": "2025-10-01T05:29:58Z", "updated_at": "2025-10-06T10:56:14Z", "html_url": "https://github.com/pydantic/pydantic-ai-temporal-example", "clone_url": "https://github.com/pydantic/pydantic-ai-temporal-example.git", "git_url": "git://github.com/pydantic/pydantic-ai-temporal-example.git", "ssh_url": "git@github.com:pydantic/pydantic-ai-temporal-example.git", "svn_url": "https://github.com/pydantic/pydantic-ai-temporal-example", "language": "Python", "fork": false, "forks_count": 1, "open_issues_count": 0, "open_issues": 0, "stargazers_count": 4, "watchers_count": 4, "watchers": 4, "size": 104, "permissions": {"admin": false, "maintain": false, "pull": true, "push": false, "triage": false}, "allow_forking": true, "web_commit_signoff_required": false, "archived": false, "disabled": false, "private": false, "has_issues": true, "has_wiki": true, "has_pages": false, "has_projects": true, "has_downloads": true, "has_discussions": false, "is_template": false, "url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example", "archive_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/{archive_format}{/ref}", "assignees_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/assignees{/user}", "blobs_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/blobs{/sha}", "branches_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/branches{/branch}", "collaborators_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/collaborators{/collaborator}", "comments_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/comments{/number}", "commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/commits{/sha}", "compare_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/compare/{base}...{head}", "contents_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/contents/{+path}", "contributors_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/contributors", "deployments_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/deployments", "downloads_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/downloads", "events_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/events", "forks_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/forks", "git_commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/commits{/sha}", "git_refs_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/refs{/sha}", "git_tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/tags{/sha}", "hooks_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/hooks", "issue_comment_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/issues/comments{/number}", "issue_events_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/issues/events{/number}", "issues_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/issues{/number}", "keys_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/keys{/key_id}", "labels_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/labels{/name}", "languages_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/languages", "merges_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/merges", "milestones_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/milestones{/number}", "notifications_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/notifications{?since,all,participating}", "pulls_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/pulls{/number}", "releases_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/releases{/id}", "stargazers_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/stargazers", "statuses_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/statuses/{sha}", "subscribers_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/subscribers", "subscription_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/subscription", "tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/tags", "trees_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/trees{/sha}", "teams_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/teams", "visibility": "public"}]}',
                            'approval_request_id': None,
                            'status': 'completed',
                            'error': None,
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13f4e9188196ab8090591caf215d',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_04507193d793fa550068ed13f4e9188196ab8090591caf215d',
                            'arguments': '{"minimal_output":false,"order":"desc","page":1,"perPage":100,"query":"user:pydantic pydantic-ai"}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': '{"total_count": 4, "incomplete_results": false, "items": [{"id": 818331198, "node_id": "R_kgDOMMa-Pg", "owner": {"login": "pydantic", "id": 110818415, "node_id": "O_kgDOBpr0bw", "avatar_url": "https://avatars.githubusercontent.com/u/110818415?v=4", "html_url": "https://github.com/pydantic", "gravatar_id": "", "type": "Organization", "site_admin": false, "url": "https://api.github.com/users/pydantic", "events_url": "https://api.github.com/users/pydantic/events{/privacy}", "following_url": "https://api.github.com/users/pydantic/following{/other_user}", "followers_url": "https://api.github.com/users/pydantic/followers", "gists_url": "https://api.github.com/users/pydantic/gists{/gist_id}", "organizations_url": "https://api.github.com/users/pydantic/orgs", "received_events_url": "https://api.github.com/users/pydantic/received_events", "repos_url": "https://api.github.com/users/pydantic/repos", "starred_url": "https://api.github.com/users/pydantic/starred{/owner}{/repo}", "subscriptions_url": "https://api.github.com/users/pydantic/subscriptions"}, "name": "pydantic-ai", "full_name": "pydantic/pydantic-ai", "description": "GenAI Agent Framework, the Pydantic way", "homepage": "https://ai.pydantic.dev", "default_branch": "main", "created_at": "2024-06-21T15:55:04Z", "pushed_at": "2025-10-13T13:43:12Z", "updated_at": "2025-10-13T13:43:17Z", "html_url": "https://github.com/pydantic/pydantic-ai", "clone_url": "https://github.com/pydantic/pydantic-ai.git", "git_url": "git://github.com/pydantic/pydantic-ai.git", "ssh_url": "git@github.com:pydantic/pydantic-ai.git", "svn_url": "https://github.com/pydantic/pydantic-ai", "language": "Python", "fork": false, "forks_count": 1298, "open_issues_count": 326, "open_issues": 326, "stargazers_count": 12888, "watchers_count": 12888, "watchers": 12888, "size": 89097, "permissions": {"admin": false, "maintain": false, "pull": true, "push": false, "triage": false}, "allow_forking": true, "web_commit_signoff_required": false, "topics": ["agent-framework", "genai", "llm", "pydantic", "python"], "archived": false, "disabled": false, "license": {"key": "mit", "name": "MIT License", "url": "https://api.github.com/licenses/mit", "spdx_id": "MIT"}, "private": false, "has_issues": true, "has_wiki": false, "has_pages": false, "has_projects": true, "has_downloads": true, "has_discussions": false, "is_template": false, "url": "https://api.github.com/repos/pydantic/pydantic-ai", "archive_url": "https://api.github.com/repos/pydantic/pydantic-ai/{archive_format}{/ref}", "assignees_url": "https://api.github.com/repos/pydantic/pydantic-ai/assignees{/user}", "blobs_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/blobs{/sha}", "branches_url": "https://api.github.com/repos/pydantic/pydantic-ai/branches{/branch}", "collaborators_url": "https://api.github.com/repos/pydantic/pydantic-ai/collaborators{/collaborator}", "comments_url": "https://api.github.com/repos/pydantic/pydantic-ai/comments{/number}", "commits_url": "https://api.github.com/repos/pydantic/pydantic-ai/commits{/sha}", "compare_url": "https://api.github.com/repos/pydantic/pydantic-ai/compare/{base}...{head}", "contents_url": "https://api.github.com/repos/pydantic/pydantic-ai/contents/{+path}", "contributors_url": "https://api.github.com/repos/pydantic/pydantic-ai/contributors", "deployments_url": "https://api.github.com/repos/pydantic/pydantic-ai/deployments", "downloads_url": "https://api.github.com/repos/pydantic/pydantic-ai/downloads", "events_url": "https://api.github.com/repos/pydantic/pydantic-ai/events", "forks_url": "https://api.github.com/repos/pydantic/pydantic-ai/forks", "git_commits_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/commits{/sha}", "git_refs_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/refs{/sha}", "git_tags_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/tags{/sha}", "hooks_url": "https://api.github.com/repos/pydantic/pydantic-ai/hooks", "issue_comment_url": "https://api.github.com/repos/pydantic/pydantic-ai/issues/comments{/number}", "issue_events_url": "https://api.github.com/repos/pydantic/pydantic-ai/issues/events{/number}", "issues_url": "https://api.github.com/repos/pydantic/pydantic-ai/issues{/number}", "keys_url": "https://api.github.com/repos/pydantic/pydantic-ai/keys{/key_id}", "labels_url": "https://api.github.com/repos/pydantic/pydantic-ai/labels{/name}", "languages_url": "https://api.github.com/repos/pydantic/pydantic-ai/languages", "merges_url": "https://api.github.com/repos/pydantic/pydantic-ai/merges", "milestones_url": "https://api.github.com/repos/pydantic/pydantic-ai/milestones{/number}", "notifications_url": "https://api.github.com/repos/pydantic/pydantic-ai/notifications{?since,all,participating}", "pulls_url": "https://api.github.com/repos/pydantic/pydantic-ai/pulls{/number}", "releases_url": "https://api.github.com/repos/pydantic/pydantic-ai/releases{/id}", "stargazers_url": "https://api.github.com/repos/pydantic/pydantic-ai/stargazers", "statuses_url": "https://api.github.com/repos/pydantic/pydantic-ai/statuses/{sha}", "subscribers_url": "https://api.github.com/repos/pydantic/pydantic-ai/subscribers", "subscription_url": "https://api.github.com/repos/pydantic/pydantic-ai/subscription", "tags_url": "https://api.github.com/repos/pydantic/pydantic-ai/tags", "trees_url": "https://api.github.com/repos/pydantic/pydantic-ai/git/trees{/sha}", "teams_url": "https://api.github.com/repos/pydantic/pydantic-ai/teams", "visibility": "public"}, {"id": 1052844101, "node_id": "R_kgDOPsEgRQ", "owner": {"login": "pydantic", "id": 110818415, "node_id": "O_kgDOBpr0bw", "avatar_url": "https://avatars.githubusercontent.com/u/110818415?v=4", "html_url": "https://github.com/pydantic", "gravatar_id": "", "type": "Organization", "site_admin": false, "url": "https://api.github.com/users/pydantic", "events_url": "https://api.github.com/users/pydantic/events{/privacy}", "following_url": "https://api.github.com/users/pydantic/following{/other_user}", "followers_url": "https://api.github.com/users/pydantic/followers", "gists_url": "https://api.github.com/users/pydantic/gists{/gist_id}", "organizations_url": "https://api.github.com/users/pydantic/orgs", "received_events_url": "https://api.github.com/users/pydantic/received_events", "repos_url": "https://api.github.com/users/pydantic/repos", "starred_url": "https://api.github.com/users/pydantic/starred{/owner}{/repo}", "subscriptions_url": "https://api.github.com/users/pydantic/subscriptions"}, "name": "pydantic-ai-chat", "full_name": "pydantic/pydantic-ai-chat", "default_branch": "main", "created_at": "2025-09-08T16:16:54Z", "pushed_at": "2025-09-25T15:30:52Z", "updated_at": "2025-10-08T15:17:58Z", "html_url": "https://github.com/pydantic/pydantic-ai-chat", "clone_url": "https://github.com/pydantic/pydantic-ai-chat.git", "git_url": "git://github.com/pydantic/pydantic-ai-chat.git", "ssh_url": "git@github.com:pydantic/pydantic-ai-chat.git", "svn_url": "https://github.com/pydantic/pydantic-ai-chat", "language": "TypeScript", "fork": false, "forks_count": 3, "open_issues_count": 0, "open_issues": 0, "stargazers_count": 16, "watchers_count": 16, "watchers": 16, "size": 487, "permissions": {"admin": false, "maintain": false, "pull": true, "push": false, "triage": false}, "allow_forking": true, "web_commit_signoff_required": false, "archived": false, "disabled": false, "private": false, "has_issues": true, "has_wiki": false, "has_pages": false, "has_projects": false, "has_downloads": true, "has_discussions": false, "is_template": false, "url": "https://api.github.com/repos/pydantic/pydantic-ai-chat", "archive_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/{archive_format}{/ref}", "assignees_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/assignees{/user}", "blobs_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/blobs{/sha}", "branches_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/branches{/branch}", "collaborators_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/collaborators{/collaborator}", "comments_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/comments{/number}", "commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/commits{/sha}", "compare_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/compare/{base}...{head}", "contents_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/contents/{+path}", "contributors_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/contributors", "deployments_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/deployments", "downloads_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/downloads", "events_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/events", "forks_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/forks", "git_commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/commits{/sha}", "git_refs_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/refs{/sha}", "git_tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/tags{/sha}", "hooks_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/hooks", "issue_comment_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/issues/comments{/number}", "issue_events_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/issues/events{/number}", "issues_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/issues{/number}", "keys_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/keys{/key_id}", "labels_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/labels{/name}", "languages_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/languages", "merges_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/merges", "milestones_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/milestones{/number}", "notifications_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/notifications{?since,all,participating}", "pulls_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/pulls{/number}", "releases_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/releases{/id}", "stargazers_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/stargazers", "statuses_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/statuses/{sha}", "subscribers_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/subscribers", "subscription_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/subscription", "tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/tags", "trees_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/git/trees{/sha}", "teams_url": "https://api.github.com/repos/pydantic/pydantic-ai-chat/teams", "visibility": "public"}, {"id": 1040842163, "node_id": "R_kgDOPgn9sw", "owner": {"login": "pydantic", "id": 110818415, "node_id": "O_kgDOBpr0bw", "avatar_url": "https://avatars.githubusercontent.com/u/110818415?v=4", "html_url": "https://github.com/pydantic", "gravatar_id": "", "type": "Organization", "site_admin": false, "url": "https://api.github.com/users/pydantic", "events_url": "https://api.github.com/users/pydantic/events{/privacy}", "following_url": "https://api.github.com/users/pydantic/following{/other_user}", "followers_url": "https://api.github.com/users/pydantic/followers", "gists_url": "https://api.github.com/users/pydantic/gists{/gist_id}", "organizations_url": "https://api.github.com/users/pydantic/orgs", "received_events_url": "https://api.github.com/users/pydantic/received_events", "repos_url": "https://api.github.com/users/pydantic/repos", "starred_url": "https://api.github.com/users/pydantic/starred{/owner}{/repo}", "subscriptions_url": "https://api.github.com/users/pydantic/subscriptions"}, "name": "pydantic-ai-gateway", "full_name": "pydantic/pydantic-ai-gateway", "default_branch": "main", "created_at": "2025-08-19T15:29:12Z", "pushed_at": "2025-10-13T12:46:43Z", "updated_at": "2025-10-13T12:46:47Z", "html_url": "https://github.com/pydantic/pydantic-ai-gateway", "clone_url": "https://github.com/pydantic/pydantic-ai-gateway.git", "git_url": "git://github.com/pydantic/pydantic-ai-gateway.git", "ssh_url": "git@github.com:pydantic/pydantic-ai-gateway.git", "svn_url": "https://github.com/pydantic/pydantic-ai-gateway", "language": "TypeScript", "fork": false, "forks_count": 1, "open_issues_count": 6, "open_issues": 6, "stargazers_count": 52, "watchers_count": 52, "watchers": 52, "size": 883, "permissions": {"admin": false, "maintain": false, "pull": true, "push": false, "triage": false}, "allow_forking": true, "web_commit_signoff_required": false, "archived": false, "disabled": false, "license": {"key": "agpl-3.0", "name": "GNU Affero General Public License v3.0", "url": "https://api.github.com/licenses/agpl-3.0", "spdx_id": "AGPL-3.0"}, "private": false, "has_issues": true, "has_wiki": true, "has_pages": false, "has_projects": true, "has_downloads": true, "has_discussions": false, "is_template": false, "url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway", "archive_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/{archive_format}{/ref}", "assignees_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/assignees{/user}", "blobs_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/blobs{/sha}", "branches_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/branches{/branch}", "collaborators_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/collaborators{/collaborator}", "comments_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/comments{/number}", "commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/commits{/sha}", "compare_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/compare/{base}...{head}", "contents_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/contents/{+path}", "contributors_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/contributors", "deployments_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/deployments", "downloads_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/downloads", "events_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/events", "forks_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/forks", "git_commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/commits{/sha}", "git_refs_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/refs{/sha}", "git_tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/tags{/sha}", "hooks_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/hooks", "issue_comment_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/issues/comments{/number}", "issue_events_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/issues/events{/number}", "issues_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/issues{/number}", "keys_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/keys{/key_id}", "labels_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/labels{/name}", "languages_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/languages", "merges_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/merges", "milestones_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/milestones{/number}", "notifications_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/notifications{?since,all,participating}", "pulls_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/pulls{/number}", "releases_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/releases{/id}", "stargazers_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/stargazers", "statuses_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/statuses/{sha}", "subscribers_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/subscribers", "subscription_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/subscription", "tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/tags", "trees_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/git/trees{/sha}", "teams_url": "https://api.github.com/repos/pydantic/pydantic-ai-gateway/teams", "visibility": "public"}, {"id": 1053149228, "node_id": "R_kgDOPsXILA", "owner": {"login": "pydantic", "id": 110818415, "node_id": "O_kgDOBpr0bw", "avatar_url": "https://avatars.githubusercontent.com/u/110818415?v=4", "html_url": "https://github.com/pydantic", "gravatar_id": "", "type": "Organization", "site_admin": false, "url": "https://api.github.com/users/pydantic", "events_url": "https://api.github.com/users/pydantic/events{/privacy}", "following_url": "https://api.github.com/users/pydantic/following{/other_user}", "followers_url": "https://api.github.com/users/pydantic/followers", "gists_url": "https://api.github.com/users/pydantic/gists{/gist_id}", "organizations_url": "https://api.github.com/users/pydantic/orgs", "received_events_url": "https://api.github.com/users/pydantic/received_events", "repos_url": "https://api.github.com/users/pydantic/repos", "starred_url": "https://api.github.com/users/pydantic/starred{/owner}{/repo}", "subscriptions_url": "https://api.github.com/users/pydantic/subscriptions"}, "name": "pydantic-ai-temporal-example", "full_name": "pydantic/pydantic-ai-temporal-example", "default_branch": "main", "created_at": "2025-09-09T04:05:06Z", "pushed_at": "2025-10-01T05:29:58Z", "updated_at": "2025-10-06T10:56:14Z", "html_url": "https://github.com/pydantic/pydantic-ai-temporal-example", "clone_url": "https://github.com/pydantic/pydantic-ai-temporal-example.git", "git_url": "git://github.com/pydantic/pydantic-ai-temporal-example.git", "ssh_url": "git@github.com:pydantic/pydantic-ai-temporal-example.git", "svn_url": "https://github.com/pydantic/pydantic-ai-temporal-example", "language": "Python", "fork": false, "forks_count": 1, "open_issues_count": 0, "open_issues": 0, "stargazers_count": 4, "watchers_count": 4, "watchers": 4, "size": 104, "permissions": {"admin": false, "maintain": false, "pull": true, "push": false, "triage": false}, "allow_forking": true, "web_commit_signoff_required": false, "archived": false, "disabled": false, "private": false, "has_issues": true, "has_wiki": true, "has_pages": false, "has_projects": true, "has_downloads": true, "has_discussions": false, "is_template": false, "url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example", "archive_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/{archive_format}{/ref}", "assignees_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/assignees{/user}", "blobs_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/blobs{/sha}", "branches_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/branches{/branch}", "collaborators_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/collaborators{/collaborator}", "comments_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/comments{/number}", "commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/commits{/sha}", "compare_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/compare/{base}...{head}", "contents_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/contents/{+path}", "contributors_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/contributors", "deployments_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/deployments", "downloads_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/downloads", "events_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/events", "forks_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/forks", "git_commits_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/commits{/sha}", "git_refs_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/refs{/sha}", "git_tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/tags{/sha}", "hooks_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/hooks", "issue_comment_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/issues/comments{/number}", "issue_events_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/issues/events{/number}", "issues_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/issues{/number}", "keys_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/keys{/key_id}", "labels_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/labels{/name}", "languages_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/languages", "merges_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/merges", "milestones_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/milestones{/number}", "notifications_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/notifications{?since,all,participating}", "pulls_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/pulls{/number}", "releases_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/releases{/id}", "stargazers_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/stargazers", "statuses_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/statuses/{sha}", "subscribers_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/subscribers", "subscription_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/subscription", "tags_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/tags", "trees_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/git/trees{/sha}", "teams_url": "https://api.github.com/repos/pydantic/pydantic-ai-temporal-example/teams", "visibility": "public"}]}',
                            'approval_request_id': None,
                            'status': 'completed',
                            'error': None,
                        },
                        tool_call_id='mcp_04507193d793fa550068ed13f4e9188196ab8090591caf215d',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_04507193d793fa550068ed13f6449881968c5c319e16192cda',
                        signature='gAAAAABo7RP5gjcL2-vaNfHjHEQTB9hn4YlILP4D3ORA_ftdOzSO1j6kdWEhHjETQLsNAOMFrpotZVLV8gkSKoRZi2-ro9UOO0uK59UNr0bfJ0by6CSdq51oYRyIxj-V9JIwwugY5Uv8m9yQytkWuZC-78EZCYOJFlw3msPOwaiIG1Aflxtqt0PerQxB-ESilTBA1ta3lzmOcOELngKwFsTMhwcQ99N5G2l2KMbG3nSxUGXFooIWev_dMKwDNYne31xrZlfGuAfpaz5bXAhNhdHrc21m-KxjF9Vwz3GwlUCKLUZ305qZWEdwr1GNnES_jnoU3HKyIfqfKqfmrk4Sa6fSDagEOhADX52A4Ghg6WLOxungr10fFj8S48b43CA_ei36t8ZNOnwljkHO6AxaPDYbaB8JZTjE2EmQ-rV2fCUespp-k45LIDlRqLxJOzzhweo9k37AEHd8pdowD9MpRYQS4KhggJaZ1tuUPxBwWfbjKeQ19AF8gyfk772-T1geH2D3SI53TiQqvJofTI4CVxF-M1dJIFGcgGimXmxJOjKEnvwRtsdTprX0t-0p5G_PpoMGomrRVwXvjh4Y8i8H02gkcW6--lHMXxt-MmjA-OQys0vEvuKs5JgklEL4m2eDNnH5Pief-8PpLveZn05ZTcButXYSpNbSCi-otcC18MljwuMALMTxh3GPCE-yULJNeaeYmub3mWU6tdXBkwx81k71pTMsTK880Ci7aGeMZkqvdoHf4pNHCWEavnu1ocFto3RN3PCKWPQhJHMzhEU903fkr_8CE7pqcOMz4cfj1dFSrBl_oa1NncrJX3xAqiF1fVPQxgJA-LKZyGp247imr-oqSOkdMBnF5PlxfaVgfWM3UbfX0xBRSS00jlSuqY45Tfk2554Dr9c0-b-IynItnBTIXJ-VwckozHmh8jKoO3MaBsFtZkWOypNlpldnDSZ7LWgvptpbkyz5mN71YivY9yb9jfxV-RGKSFIo8P8pP91HBsDFFpBHeSdTbmQpF_NZQEFkRCVgazNvGQgiYvWVmtvijtsD2XhoZvjkVRUhFgKsKeAcscqEzXUf_dHinoO0b2bVZumN8IUCeZxUYOw8JG9BZPC2BPkKql3XkFCgWhp-SJmaDzkWLX_qRVYtYP5sWatJRCUI6Rb0MpE5cF-Qhn2I7ZAE0w15vKYX1PJt1ffonSbnXtGHWwZv3hc2zPA-2tfTUUw8wrEhb8RVdX-qwrxW8cgAkb2UbSrCloGdNMyy9NFtej1fETM8e-QkfrxFY_-51H9tQIWr5bT7tTfWt54PkT49F7TtGPU_fXtBK5XrLi25daBX3MC4X_FDXbKb_97-A7-U4CPl8EQx9ElQo6Evoyhhdv1vn-FncKVO74tiM2xfglHPb64zTFUn6TD0h-Tty2PCzm_z3Ph7jkCW_tWenVRt_x51FqkMfyVC8llzxsvwg6hvHDFb342CyrEjenv2Umw4i1HJ49kitgO-ZmFcAgmRrJWDm8GTwWTHbN-AUpIJdfnoacuoblfROKw1FcoZPnr4lkmAVhrzdvqPqXxR1ZRhrXsOuUsFPb9FCIt2GuB-Sd4lRjOpeeD-QY7TG4UAVrUE42YQbVuKedOOy2IBPYMvGZiy_42-gxfj3fNdubhk8_ybfONpXpV302sNjH105lyI6QtZ4Q705Kd3q8KSrvo7YyJrvZK0VLc7Hk2ApQbMFogGbCSp9_pyTg_cD4hycyWZGG91FBq9Hl79FHGQDlXQcyfyY71NUf2hJO8BE7QISjqcWSKZSIlcnhxKGgzMbsKjvo0Zadue7J3ZrqSDX0-qpLRigrsULPjzxHD3VjE77Z70T3LjMl774q3QWuAt87pF-l9sEkWYFLNcFkTHVznxeHLMCyvZLBOQDsxySNdJxG5FDCyFYKkcnop5h6mIIV5s_MpQ',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='pydantic-ai is Pydantic GenAI agent framework (Python), with ~12.9 K ⭐, ~1.3 K forks, 326 open issues; homepage: https://ai.pydantic.dev.',
                        id='msg_04507193d793fa550068ed13f84688819697b035b4d76faaa5',
                    ),
                ],
                usage=RequestUsage(input_tokens=7256, output_tokens=490, details={'reasoning_tokens': 384}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_04507193d793fa550068ed13f1127c819686aa079952ab515c',
                finish_reason='stop',
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
                id='test_server',
                url='https://api.githubcopilot.com/mcp/',
                authorization_token=os.getenv('GITHUB_ACCESS_TOKEN', 'mock-access-token'),
                allowed_tools=['search_repositories', 'list_commits'],
            ),
        ],
    )

    event_parts: list[Any] = []

    async with agent.iter(
        user_prompt='Can you tell me more about pydantic-ai repo? Keep your answer short'
    ) as agent_run:
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
                        content='Can you tell me more about pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_list_tools',
                        args={
                            'id': 'mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                            'server_label': 'test_server',
                            'tools': [],
                            'type': 'mcp_list_tools',
                            'error': None,
                        },
                        tool_call_id='mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_list_tools',
                        content={
                            'id': 'mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                            'server_label': 'test_server',
                            'tools': [
                                {
                                    'input_schema': {
                                        'properties': {
                                            'author': {
                                                'description': 'Author username or email address to filter commits by',
                                                'type': 'string',
                                            },
                                            'owner': {'description': 'Repository owner', 'type': 'string'},
                                            'page': {
                                                'description': 'Page number for pagination (min 1)',
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'perPage': {
                                                'description': 'Results per page for pagination (min 1, max 100)',
                                                'maximum': 100,
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'repo': {'description': 'Repository name', 'type': 'string'},
                                            'sha': {
                                                'description': 'Commit SHA, branch or tag name to list commits of. If not provided, uses the default branch of the repository. If a commit SHA is provided, will list commits up to that SHA.',
                                                'type': 'string',
                                            },
                                        },
                                        'required': ['owner', 'repo'],
                                        'type': 'object',
                                    },
                                    'name': 'list_commits',
                                    'annotations': {'read_only': True},
                                    'description': 'Get list of commits of a branch in a GitHub repository. Returns at least 30 results per page by default, but can return more if specified using the perPage parameter (up to 100).',
                                },
                                {
                                    'input_schema': {
                                        'properties': {
                                            'minimal_output': {
                                                'default': True,
                                                'description': 'Return minimal repository information (default: true). When false, returns full GitHub API repository objects.',
                                                'type': 'boolean',
                                            },
                                            'order': {
                                                'description': 'Sort order',
                                                'enum': ['asc', 'desc'],
                                                'type': 'string',
                                            },
                                            'page': {
                                                'description': 'Page number for pagination (min 1)',
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'perPage': {
                                                'description': 'Results per page for pagination (min 1, max 100)',
                                                'maximum': 100,
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'query': {
                                                'description': "Repository search query. Examples: 'machine learning in:name stars:>1000 language:python', 'topic:react', 'user:facebook'. Supports advanced search syntax for precise filtering.",
                                                'type': 'string',
                                            },
                                            'sort': {
                                                'description': 'Sort repositories by field, defaults to best match',
                                                'enum': ['stars', 'forks', 'help-wanted-issues', 'updated'],
                                                'type': 'string',
                                            },
                                        },
                                        'required': ['query'],
                                        'type': 'object',
                                    },
                                    'name': 'search_repositories',
                                    'annotations': {'read_only': True},
                                    'description': 'Find GitHub repositories by name, description, readme, topics, or other metadata. Perfect for discovering projects, finding examples, or locating specific repositories across GitHub.',
                                },
                            ],
                            'type': 'mcp_list_tools',
                            'error': None,
                        },
                        tool_call_id='mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20a271948191a97ecce27848f461',
                        signature='gAAAAABo7SCjE_wICveOw0z6kFvwtVJSAi8SGeS7LooWILhZ1wQTZR3QZHVylNYvEkcAWc9VpOKvCjdyffXyz45penIN9cSr5WCk7vP0DULso01ZudhhFPVypQ68yogEx7xMXwmVJbOqoyOZb3M39dnKkFtjQeMWd4AzGfPC8ufjRt9SEbTlRjq06bRoNBmVYP8rQCftHh9BWIvGKU0Fe5rwB7S0BtFmcowfwFs1lMyKN89BTITrVXrTCmai6mw_JvNj9PCuDiOMfuYbtbmZKb0OZOjiUyErSSFhpdyEqQZbeBF1frZVtPjPJCEIusEIHK5cT65odpRKn2uakRcUcCKGcSB6Sv0OTiSF7nJW79p-eJIfeFXxwJwiCQswp3mUt5U2a0kVsxDv2XUhP3ONOE_-vhNS2EGwRD8IglEw1F0zYqXItUUSjb2vig0SA1sNUVK4oK_EwaxYUYgxtgaC42e7qzo3Hs_18S6FXxaMSygoyk4jVMhIVJb7xbQc7-hWF1ohdXJRTpLNSvZ-z_FTePo_3Bye3HVFpSWOpUAGd1peSVrEiXXaN5kMTJpeZfHbXVTX2FN7ugabl_IgdcVOH-seE1cD5kNbodVbjxnDdxzMx8q10xtzlI7Ze9CPFJb6BZFWKEZri-sJUJAbSRBfH8NIHJQj1UjDx5vg4YRI3TvUR8RIhedC54_25cb_6HD3_0sghACWhL_9uyzPgMw4tjzIQQNtHDduB2BWpKapkSgUkJNAaujU_YOuq-uttzV2nOBXr1H6APbbG_kgNbHs1a4Z_Fvb9Jzg2oKZ1dJOAt81_VK6s2gU50EqTWCQi_k8Hnj4MnUnQIpIyy1jjHDxheKIkOo_7eTSkYsgEapY57S0NGZ6T6Ru-XE8Hql_xa7VhW7VtRyF3LADgK5DD5rjHhzp-dIbhc6Y-4KgTLvaHU-Dohj1t8uZEP8niv3I3JKKgWDWVJ9vPbioeGqOINooUzZcIaJg4BztTcP8CBMsDX5w6MPO3IoBwnhS50YLIFrkDplwqLsnIEwpaxROm0csRrTcUpl4ZiY18R9PXyBp5J8tH0Y7BqaJ6NTAdFT2OLL6TntZ6UtTCaR7U4p_H45DUl0ZKsnjxBjNQWS67DatY2U44S69Qh86ken4gh44cXA4r5NnpRbCuZmo7BhH2wH5yDRhhZc2AQ5WUwpP0eyDnjF8xunNftbwyISfNV7WuHcdaYwiu29WOx6O3c0FLJO-KhV0LvK-VZZWXYTWY55pqv436FtVoq0XS4AiKIN4zwXHx_zuy3Zw_lO41VJUZbBNL4zcAv7GDukWyiFkocpWMBhLhISuuh0T65B3R5W8_HWYJbi-zlORXGZaw3OtJqTFPKBqhRH-chnBoSy8D15C3gahfV9FKjdIuYruJ6laevbUecvydkHG8GrdfKDizmQsRYLPAxApJX4GI8fHMG5vqBgUfdDpZaWhkEzoGwywWparXBx6LVrGVKAa2GUHAhyevUooKzLuPcIeauniU0cTSgHBpGtHLvJJf-XZFQfHMeIwYxDs9exI4EwRTKRst4wq-foobKcYhRGjqv9O7BrdFIiar8iO0yiBfzdA-gCV63WajMhhWfwh7c9svtge-ZdvCvdSCmYAJRHlgBxUZ8bN2sENy3ZuW5lVk_RQBKtcjTi0uAHm6TT66bZ0JIYTp8Mns9PDTQov-p3BDTtxp4_iNFs0RnOHXvJ7nNNvUHJ2lrdO6NzquhE5iodvXsuWa0wikvsJrxVN1J9Azg==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                            'arguments': '',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'in_progress',
                            'approval_request_id': None,
                            'error': None,
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                            'arguments': '{"query":"pydantic-ai","sort":"best match","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'failed',
                            'approval_request_id': None,
                            'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20a490108191823c8351b9c7b1eb',
                        signature='gAAAAABo7SCllu3A0J5TLhrToCHtBqlJXTNnUiSjyawmoj11hxQvRsn-kRItP0vUjdkSzapSsaiTTCoHAnYP6Uh52eogYu4emHs5I3e40WoGQjQ8Aog8imp25ZxF7OHWfqynluKfum-8kCYE6GHi-1k9zTIrBPZpfdwdJPKng_AjAxgknHgaNx31Xkbyjtrwl13AkGvLNlbQXhLiD21EAwcc2gBEe46BwtRXfLlms1uFIkuUWeNUupnOaHkjOrchPrqWECJA9A8gAlg29JOaDW6THXksjb6PIBvRh0UOaY_WhKEsbqdVLU-hKnACCGY3gk4EksSwT65CHJd1ByKhvi03-n1GD3w0_tlLdAGQhHeVO25kDR_16Y5geFllZHJ26XN6EsQiIE6UCsACa07iXgrrAPH3HsnKz-FNM-fgWaxihbEJShbmg0BjmTcMIGxcmLf9xbt9sDc6Hy0iZQTgVWyDj23jN1hTpIIu382KlbgMDd-c_A8GFQp8ySlh8CmuNaTjq6e8oOknbsDDladBx19FrHGcWVmpZLkaa1WoJ-_JpAo6CzqYQJMIBdm2R4w9O9km7jNgtEVJSWe4zxdYVfNqocishUfZoeD1Ncr9bkQ4AOlw0oTmrb3kEa3mRaG1gy0KHC_EuBfPMmr89XeivxzTak8QZg9meG8n1tKSqjHf8EJEIn0VBnAReR7lFpd0Sr-vuVz7Y0kVJenvSNljm2oYmwaMXctycGh5GkHAh8FfsvvGoiRVomZ4hCIy70X2p4_0nMkwNORPGkBIfEhHuL_VcIulMqhkGdtf-Q7a7D9Q6jmuSplXvKhxZH31ay3zzUc0J-1guIboH0R3GJXmBHRLi_PXao_2GS1Nhjx38PBFejmIvtliITF6YhP7DFa29eMacvRqKEiKdGZKAX1NdluBq7YSYJhOPVnV9LVosBps-ZLIfkK6d4ZRidWPcVf1j0tRBKPTHUITctPdNrk6kzyTeXUy6wA9yhq6jkcpCqJQr0wF_tc-4OkyQ1IJ05WYhrg4fAMgmu2DIWQ7NyQ5H9gbsIHEvu_QxXfNq_wlBOcgyosyxfitEgUrxCQKUbro-7rKZaVjRRPOjfHRr8j9gry-lCh4ixC8KkDdJaJ108B3szuBKMZIuhr3b_mys6XX0UGQi7RNTxwnNgt1sWnyh-F1W0r4DN1dnf7IxnAmRVhIjcG67IHhZDgsyG71tTN-0DqxmSb2Gg7p0cW__HYDFw3uYn_BeFCByY6oJJdwIGC0hqXDxd2p3qmJTxHiLPoICPRQOM2oRC1fW_XFb2VH27swHyvS-8tdkCw4OtAIchwKpALSkpg0AUdWemc0EseGxtIo49M9c7Ed5WzFXzawWKKUxJDhWQHeV7J6xOtQFs4XmwftyzbUBYVcPDcz0nz3BvSHRq5tjdKu5H5X0t_i8L9UAZAqiF58nw==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                            'arguments': '',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'in_progress',
                            'approval_request_id': None,
                            'error': None,
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                            'arguments': '{"query":"pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'failed',
                            'approval_request_id': None,
                            'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20a62f648191b719c874cfe47a3d',
                        signature='gAAAAABo7SCorCLOFD5VcIU8sG7u4WVtbDowsj_gwMf8GNhOmqvSk8cxQUpKRgJV3bOa8xuz4f65f_TvG9LXdkHbZZV658XLxJliS-0ugk2jOTNLBTyEJfxsmAO3GRApwRWvb-BihcREuGHlEopM2Pw0wUJtWoVRHDqfAVECPkZrZLbW1gujztl7zCpx8l45WngYy37iUWYepVmNBm-1NiDDyPxM2gbGMkCoszu57RgWzB3dAo8KbPtpfTb5aTvLdqfi1vusRb1l8RTtTFUSYW8fuJmFccA83rej30oSN6YydiSCvO5p3_cIH7AJTQiMyMYOIrZX070frtQmezvaeRZWpIV7f2chaxdilerEmaYstO6MWg8zuolvr3oLThUp8xD89IJf8KF3HUhkXRNrvvsorl_XjGwmbuOZLaADmn5GqhyFm90VTDkgx2G1ms3X8C8OA6kz_iuo6u1TrZA0VzkFGFxZoUog9-w5Bp0oemZMuS5PtcJJK3b7BQ3f0Hz7Ol3Sp0Arm6bfbt18ZOVmy6FU3S5C7ah5uHGQf8H8e9dwOcJhsjwtEmSaYqq5akC-gX0D39QGgO-yQ7jL2glutE1pFtybVQNcREbEyt1E0K6IqW72ksR9ymN8H0T-MIzVC72peLXpVzrZ_DUZevNLLK3hIhpivngZe4ecP1ilbKs1mD4p0j7F3_lgsGObcZTOn3ku-WkLItmFLIUUrVdG5Em7lPM8lgiOWXoCIs7Xa39s8j5Rlbd-odUbWmAuUHfv7MC2c-moo3V_0BJ9eTVRi5_i6r0LbTxaWITZOpTQBCD93PczuQkEgekwJJXVI4BKCfF4xa7x4iAe4G9NRKiwjnrDxC-jFr-lxsEZ2pLR9BPQgG3FQwhS4v1HfD5teTr9YpAsS65UZfHDAdb1KKIJDtnWzrdPlx9ffHPS0hgaZSbwyZdNs7ifWBxOfpNX_Mk2V2lKUEyBKN2o6MTbN4bmODmsjJAdVVhRiIFlHmM37U0lMqZf-e13i4dDYoTZ3bwjACKUIeir-BYQphWtOCl9GT5BvEg6mI_aPoZB4VzIRAXil064UwPoA1GoTNDw0sXptNTmuqoJJVjx0nyRhoZLKvWFddHYIE_F1wHp4GV7pt6vUNeRiahG7NV6Abyoa7wEC9fYb2pHWaH01ER8iwyU_ggY_wwyD0Uf8-UjLDf_GkwXPX8ipA65ssTWSzD8bYPGkAoIGotTMYWxPEvzXXGb1WuDrE8czYTyMKM1Dj6XlBQQf-C-ujcl7BjmJqPbQkgvcOda7PHBQXM2yCYA6cLQH0wvDtuhzRFmrQlDebXYoS0i1yIT1Fl8YxmGVOEDbG5evTPFt14G90AMCLU1FH-N9c8Zo-jUDI8k9RiWKYlys3aXggJyzeZkEeyhetmuxNlJhl9kZ4gut4GCqveajz_0t6YEO_YHqpqPhmXRXFzZOUFXjPdPjMiw4CDzNqTLJBGT5eb2ekNCrROavtAG69a9z0wS4fyV1Bfzy0D1LZA22Bz3YVoIcaXixcyrfFZn2lQD2v8Jitv_OvG5qhc4x7daIWwq0-mJRZc5Ot-q2kQhhi1gnjNzCR-NAH4kqokHtCU30_HBjJdB0Yo2NVoEUu7i5ZdXZ5JsblEnKHutERGzxhf6ow9lC6QsE0iZGBmD88SOT9yth_aL48JFJpunjuzzjnZp7hfTbt2BIcbywEHiWYEO6K57_gIUsfHa_0irOFrzBisqSpBbgacZj8NFIGEw_E9sRgUaZ5I3X-PONHyj36raxv_kd6J9Y2uvNBgeVxnuLpNPRHNP8TR-t1Yks0A7975rNOdOzFYL-uCEmC-sbDB3DtBimJ2bsPkmUpTpgA-ds49DlE0zr_fREMrgQMLNBagtvOvN3OH0oUsWVZcCJYr3ADu4CZfwkDOuWQSjja4u3loG6PTjbIBe1IVL-wGa22nir5Dvi4GGd-G--9xYJVrSKYsxE4NK5M8TJu2gVPPYuGTqOel2RVbNEHR3E2pVXHLwMiwXBLjLFGA7y6ou4cgmSoB95G582Ycq2v_qMuWitdE2DITdtLgCv3bKTsRZiKqXTn6j0g4qpicowle2ob4uBulOgK8thTb3PCCbRzi6atNP1mXVgP6vCtyS72D4BCaT198fl3BtkohoqPVxL2be2fYRg1WFnudFM0jg9iYzeW6N3aE5khoEUZ4Ek5SZFT8Pggmfh1_HAg==',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='pydantic-ai is an open-source Python library (MIT) that uses large language models to generate Pydantic model classes (and JSON-Schema) from example data. It provides a CLI and Python API, supports multiple LLM backends (OpenAI, Azure, Hugging Face, etc.), and lets you customize prompts. Repo: https://github.com/birbland/pydantic-ai',
                        id='msg_01264dba19113f900068ed20a8441481918c02d9dfa25a3146',
                    ),
                ],
                usage=RequestUsage(input_tokens=876, output_tokens=571, details={'reasoning_tokens': 384}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_01264dba19113f900068ed20a0538c8191966b895a0e63290c',
                finish_reason='stop',
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='mcp_list_tools',
                    args={
                        'id': 'mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                        'server_label': 'test_server',
                        'tools': [],
                        'type': 'mcp_list_tools',
                        'error': None,
                    },
                    tool_call_id='mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_list_tools',
                    content={
                        'id': 'mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                        'server_label': 'test_server',
                        'tools': [
                            {
                                'input_schema': {
                                    'properties': {
                                        'author': {
                                            'description': 'Author username or email address to filter commits by',
                                            'type': 'string',
                                        },
                                        'owner': {'description': 'Repository owner', 'type': 'string'},
                                        'page': {
                                            'description': 'Page number for pagination (min 1)',
                                            'minimum': 1,
                                            'type': 'number',
                                        },
                                        'perPage': {
                                            'description': 'Results per page for pagination (min 1, max 100)',
                                            'maximum': 100,
                                            'minimum': 1,
                                            'type': 'number',
                                        },
                                        'repo': {'description': 'Repository name', 'type': 'string'},
                                        'sha': {
                                            'description': 'Commit SHA, branch or tag name to list commits of. If not provided, uses the default branch of the repository. If a commit SHA is provided, will list commits up to that SHA.',
                                            'type': 'string',
                                        },
                                    },
                                    'required': ['owner', 'repo'],
                                    'type': 'object',
                                },
                                'name': 'list_commits',
                                'annotations': {'read_only': True},
                                'description': 'Get list of commits of a branch in a GitHub repository. Returns at least 30 results per page by default, but can return more if specified using the perPage parameter (up to 100).',
                            },
                            {
                                'input_schema': {
                                    'properties': {
                                        'minimal_output': {
                                            'default': True,
                                            'description': 'Return minimal repository information (default: true). When false, returns full GitHub API repository objects.',
                                            'type': 'boolean',
                                        },
                                        'order': {
                                            'description': 'Sort order',
                                            'enum': ['asc', 'desc'],
                                            'type': 'string',
                                        },
                                        'page': {
                                            'description': 'Page number for pagination (min 1)',
                                            'minimum': 1,
                                            'type': 'number',
                                        },
                                        'perPage': {
                                            'description': 'Results per page for pagination (min 1, max 100)',
                                            'maximum': 100,
                                            'minimum': 1,
                                            'type': 'number',
                                        },
                                        'query': {
                                            'description': "Repository search query. Examples: 'machine learning in:name stars:>1000 language:python', 'topic:react', 'user:facebook'. Supports advanced search syntax for precise filtering.",
                                            'type': 'string',
                                        },
                                        'sort': {
                                            'description': 'Sort repositories by field, defaults to best match',
                                            'enum': ['stars', 'forks', 'help-wanted-issues', 'updated'],
                                            'type': 'string',
                                        },
                                    },
                                    'required': ['query'],
                                    'type': 'object',
                                },
                                'name': 'search_repositories',
                                'annotations': {'read_only': True},
                                'description': 'Find GitHub repositories by name, description, readme, topics, or other metadata. Perfect for discovering projects, finding examples, or locating specific repositories across GitHub.',
                            },
                        ],
                        'type': 'mcp_list_tools',
                        'error': None,
                    },
                    tool_call_id='mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=2,
                part=ThinkingPart(
                    content='',
                    id='rs_01264dba19113f900068ed20a271948191a97ecce27848f461',
                    signature='gAAAAABo7SCjE_wICveOw0z6kFvwtVJSAi8SGeS7LooWILhZ1wQTZR3QZHVylNYvEkcAWc9VpOKvCjdyffXyz45penIN9cSr5WCk7vP0DULso01ZudhhFPVypQ68yogEx7xMXwmVJbOqoyOZb3M39dnKkFtjQeMWd4AzGfPC8ufjRt9SEbTlRjq06bRoNBmVYP8rQCftHh9BWIvGKU0Fe5rwB7S0BtFmcowfwFs1lMyKN89BTITrVXrTCmai6mw_JvNj9PCuDiOMfuYbtbmZKb0OZOjiUyErSSFhpdyEqQZbeBF1frZVtPjPJCEIusEIHK5cT65odpRKn2uakRcUcCKGcSB6Sv0OTiSF7nJW79p-eJIfeFXxwJwiCQswp3mUt5U2a0kVsxDv2XUhP3ONOE_-vhNS2EGwRD8IglEw1F0zYqXItUUSjb2vig0SA1sNUVK4oK_EwaxYUYgxtgaC42e7qzo3Hs_18S6FXxaMSygoyk4jVMhIVJb7xbQc7-hWF1ohdXJRTpLNSvZ-z_FTePo_3Bye3HVFpSWOpUAGd1peSVrEiXXaN5kMTJpeZfHbXVTX2FN7ugabl_IgdcVOH-seE1cD5kNbodVbjxnDdxzMx8q10xtzlI7Ze9CPFJb6BZFWKEZri-sJUJAbSRBfH8NIHJQj1UjDx5vg4YRI3TvUR8RIhedC54_25cb_6HD3_0sghACWhL_9uyzPgMw4tjzIQQNtHDduB2BWpKapkSgUkJNAaujU_YOuq-uttzV2nOBXr1H6APbbG_kgNbHs1a4Z_Fvb9Jzg2oKZ1dJOAt81_VK6s2gU50EqTWCQi_k8Hnj4MnUnQIpIyy1jjHDxheKIkOo_7eTSkYsgEapY57S0NGZ6T6Ru-XE8Hql_xa7VhW7VtRyF3LADgK5DD5rjHhzp-dIbhc6Y-4KgTLvaHU-Dohj1t8uZEP8niv3I3JKKgWDWVJ9vPbioeGqOINooUzZcIaJg4BztTcP8CBMsDX5w6MPO3IoBwnhS50YLIFrkDplwqLsnIEwpaxROm0csRrTcUpl4ZiY18R9PXyBp5J8tH0Y7BqaJ6NTAdFT2OLL6TntZ6UtTCaR7U4p_H45DUl0ZKsnjxBjNQWS67DatY2U44S69Qh86ken4gh44cXA4r5NnpRbCuZmo7BhH2wH5yDRhhZc2AQ5WUwpP0eyDnjF8xunNftbwyISfNV7WuHcdaYwiu29WOx6O3c0FLJO-KhV0LvK-VZZWXYTWY55pqv436FtVoq0XS4AiKIN4zwXHx_zuy3Zw_lO41VJUZbBNL4zcAv7GDukWyiFkocpWMBhLhISuuh0T65B3R5W8_HWYJbi-zlORXGZaw3OtJqTFPKBqhRH-chnBoSy8D15C3gahfV9FKjdIuYruJ6laevbUecvydkHG8GrdfKDizmQsRYLPAxApJX4GI8fHMG5vqBgUfdDpZaWhkEzoGwywWparXBx6LVrGVKAa2GUHAhyevUooKzLuPcIeauniU0cTSgHBpGtHLvJJf-XZFQfHMeIwYxDs9exI4EwRTKRst4wq-foobKcYhRGjqv9O7BrdFIiar8iO0yiBfzdA-gCV63WajMhhWfwh7c9svtge-ZdvCvdSCmYAJRHlgBxUZ8bN2sENy3ZuW5lVk_RQBKtcjTi0uAHm6TT66bZ0JIYTp8Mns9PDTQov-p3BDTtxp4_iNFs0RnOHXvJ7nNNvUHJ2lrdO6NzquhE5iodvXsuWa0wikvsJrxVN1J9Azg==',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                        'arguments': '',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                        'arguments': '{"query":"pydantic-ai","sort":"best match","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=5,
                part=ThinkingPart(
                    content='',
                    id='rs_01264dba19113f900068ed20a490108191823c8351b9c7b1eb',
                    signature='gAAAAABo7SCllu3A0J5TLhrToCHtBqlJXTNnUiSjyawmoj11hxQvRsn-kRItP0vUjdkSzapSsaiTTCoHAnYP6Uh52eogYu4emHs5I3e40WoGQjQ8Aog8imp25ZxF7OHWfqynluKfum-8kCYE6GHi-1k9zTIrBPZpfdwdJPKng_AjAxgknHgaNx31Xkbyjtrwl13AkGvLNlbQXhLiD21EAwcc2gBEe46BwtRXfLlms1uFIkuUWeNUupnOaHkjOrchPrqWECJA9A8gAlg29JOaDW6THXksjb6PIBvRh0UOaY_WhKEsbqdVLU-hKnACCGY3gk4EksSwT65CHJd1ByKhvi03-n1GD3w0_tlLdAGQhHeVO25kDR_16Y5geFllZHJ26XN6EsQiIE6UCsACa07iXgrrAPH3HsnKz-FNM-fgWaxihbEJShbmg0BjmTcMIGxcmLf9xbt9sDc6Hy0iZQTgVWyDj23jN1hTpIIu382KlbgMDd-c_A8GFQp8ySlh8CmuNaTjq6e8oOknbsDDladBx19FrHGcWVmpZLkaa1WoJ-_JpAo6CzqYQJMIBdm2R4w9O9km7jNgtEVJSWe4zxdYVfNqocishUfZoeD1Ncr9bkQ4AOlw0oTmrb3kEa3mRaG1gy0KHC_EuBfPMmr89XeivxzTak8QZg9meG8n1tKSqjHf8EJEIn0VBnAReR7lFpd0Sr-vuVz7Y0kVJenvSNljm2oYmwaMXctycGh5GkHAh8FfsvvGoiRVomZ4hCIy70X2p4_0nMkwNORPGkBIfEhHuL_VcIulMqhkGdtf-Q7a7D9Q6jmuSplXvKhxZH31ay3zzUc0J-1guIboH0R3GJXmBHRLi_PXao_2GS1Nhjx38PBFejmIvtliITF6YhP7DFa29eMacvRqKEiKdGZKAX1NdluBq7YSYJhOPVnV9LVosBps-ZLIfkK6d4ZRidWPcVf1j0tRBKPTHUITctPdNrk6kzyTeXUy6wA9yhq6jkcpCqJQr0wF_tc-4OkyQ1IJ05WYhrg4fAMgmu2DIWQ7NyQ5H9gbsIHEvu_QxXfNq_wlBOcgyosyxfitEgUrxCQKUbro-7rKZaVjRRPOjfHRr8j9gry-lCh4ixC8KkDdJaJ108B3szuBKMZIuhr3b_mys6XX0UGQi7RNTxwnNgt1sWnyh-F1W0r4DN1dnf7IxnAmRVhIjcG67IHhZDgsyG71tTN-0DqxmSb2Gg7p0cW__HYDFw3uYn_BeFCByY6oJJdwIGC0hqXDxd2p3qmJTxHiLPoICPRQOM2oRC1fW_XFb2VH27swHyvS-8tdkCw4OtAIchwKpALSkpg0AUdWemc0EseGxtIo49M9c7Ed5WzFXzawWKKUxJDhWQHeV7J6xOtQFs4XmwftyzbUBYVcPDcz0nz3BvSHRq5tjdKu5H5X0t_i8L9UAZAqiF58nw==',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=6,
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                        'arguments': '',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=7,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                        'arguments': '{"query":"pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=8,
                part=ThinkingPart(
                    content='',
                    id='rs_01264dba19113f900068ed20a62f648191b719c874cfe47a3d',
                    signature='gAAAAABo7SCorCLOFD5VcIU8sG7u4WVtbDowsj_gwMf8GNhOmqvSk8cxQUpKRgJV3bOa8xuz4f65f_TvG9LXdkHbZZV658XLxJliS-0ugk2jOTNLBTyEJfxsmAO3GRApwRWvb-BihcREuGHlEopM2Pw0wUJtWoVRHDqfAVECPkZrZLbW1gujztl7zCpx8l45WngYy37iUWYepVmNBm-1NiDDyPxM2gbGMkCoszu57RgWzB3dAo8KbPtpfTb5aTvLdqfi1vusRb1l8RTtTFUSYW8fuJmFccA83rej30oSN6YydiSCvO5p3_cIH7AJTQiMyMYOIrZX070frtQmezvaeRZWpIV7f2chaxdilerEmaYstO6MWg8zuolvr3oLThUp8xD89IJf8KF3HUhkXRNrvvsorl_XjGwmbuOZLaADmn5GqhyFm90VTDkgx2G1ms3X8C8OA6kz_iuo6u1TrZA0VzkFGFxZoUog9-w5Bp0oemZMuS5PtcJJK3b7BQ3f0Hz7Ol3Sp0Arm6bfbt18ZOVmy6FU3S5C7ah5uHGQf8H8e9dwOcJhsjwtEmSaYqq5akC-gX0D39QGgO-yQ7jL2glutE1pFtybVQNcREbEyt1E0K6IqW72ksR9ymN8H0T-MIzVC72peLXpVzrZ_DUZevNLLK3hIhpivngZe4ecP1ilbKs1mD4p0j7F3_lgsGObcZTOn3ku-WkLItmFLIUUrVdG5Em7lPM8lgiOWXoCIs7Xa39s8j5Rlbd-odUbWmAuUHfv7MC2c-moo3V_0BJ9eTVRi5_i6r0LbTxaWITZOpTQBCD93PczuQkEgekwJJXVI4BKCfF4xa7x4iAe4G9NRKiwjnrDxC-jFr-lxsEZ2pLR9BPQgG3FQwhS4v1HfD5teTr9YpAsS65UZfHDAdb1KKIJDtnWzrdPlx9ffHPS0hgaZSbwyZdNs7ifWBxOfpNX_Mk2V2lKUEyBKN2o6MTbN4bmODmsjJAdVVhRiIFlHmM37U0lMqZf-e13i4dDYoTZ3bwjACKUIeir-BYQphWtOCl9GT5BvEg6mI_aPoZB4VzIRAXil064UwPoA1GoTNDw0sXptNTmuqoJJVjx0nyRhoZLKvWFddHYIE_F1wHp4GV7pt6vUNeRiahG7NV6Abyoa7wEC9fYb2pHWaH01ER8iwyU_ggY_wwyD0Uf8-UjLDf_GkwXPX8ipA65ssTWSzD8bYPGkAoIGotTMYWxPEvzXXGb1WuDrE8czYTyMKM1Dj6XlBQQf-C-ujcl7BjmJqPbQkgvcOda7PHBQXM2yCYA6cLQH0wvDtuhzRFmrQlDebXYoS0i1yIT1Fl8YxmGVOEDbG5evTPFt14G90AMCLU1FH-N9c8Zo-jUDI8k9RiWKYlys3aXggJyzeZkEeyhetmuxNlJhl9kZ4gut4GCqveajz_0t6YEO_YHqpqPhmXRXFzZOUFXjPdPjMiw4CDzNqTLJBGT5eb2ekNCrROavtAG69a9z0wS4fyV1Bfzy0D1LZA22Bz3YVoIcaXixcyrfFZn2lQD2v8Jitv_OvG5qhc4x7daIWwq0-mJRZc5Ot-q2kQhhi1gnjNzCR-NAH4kqokHtCU30_HBjJdB0Yo2NVoEUu7i5ZdXZ5JsblEnKHutERGzxhf6ow9lC6QsE0iZGBmD88SOT9yth_aL48JFJpunjuzzjnZp7hfTbt2BIcbywEHiWYEO6K57_gIUsfHa_0irOFrzBisqSpBbgacZj8NFIGEw_E9sRgUaZ5I3X-PONHyj36raxv_kd6J9Y2uvNBgeVxnuLpNPRHNP8TR-t1Yks0A7975rNOdOzFYL-uCEmC-sbDB3DtBimJ2bsPkmUpTpgA-ds49DlE0zr_fREMrgQMLNBagtvOvN3OH0oUsWVZcCJYr3ADu4CZfwkDOuWQSjja4u3loG6PTjbIBe1IVL-wGa22nir5Dvi4GGd-G--9xYJVrSKYsxE4NK5M8TJu2gVPPYuGTqOel2RVbNEHR3E2pVXHLwMiwXBLjLFGA7y6ou4cgmSoB95G582Ycq2v_qMuWitdE2DITdtLgCv3bKTsRZiKqXTn6j0g4qpicowle2ob4uBulOgK8thTb3PCCbRzi6atNP1mXVgP6vCtyS72D4BCaT198fl3BtkohoqPVxL2be2fYRg1WFnudFM0jg9iYzeW6N3aE5khoEUZ4Ek5SZFT8Pggmfh1_HAg==',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=9, part=TextPart(content='p', id='msg_01264dba19113f900068ed20a8441481918c02d9dfa25a3146')
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='yd')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='antic')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='ai')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' open')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='source')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' Python')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' library')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='MIT')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' that')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' uses')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' large')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' language')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' models')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' generate')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' P')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='yd')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='antic')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' model')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' classes')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='and')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' JSON')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='Schema')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' from')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' example')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' data')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' It')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' provides')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' CLI')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' Python')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' API')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' supports')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' multiple')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' L')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='LM')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' back')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='ends')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='Open')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='AI')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' Azure')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' Hug')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='ging')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' Face')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' etc')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='.),')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' lets')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' you')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' customize')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' prompts')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' Repo')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta=' https')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='://')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='github')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='.com')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='/b')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='ir')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='b')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='land')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='/p')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='yd')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='antic')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=9, delta=TextPartDelta(content_delta='ai')),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='mcp_list_tools',
                    args={
                        'id': 'mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                        'server_label': 'test_server',
                        'tools': [],
                        'type': 'mcp_list_tools',
                        'error': None,
                    },
                    tool_call_id='mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_list_tools',
                    content={
                        'id': 'mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                        'server_label': 'test_server',
                        'tools': [
                            {
                                'input_schema': {
                                    'properties': {
                                        'author': {
                                            'description': 'Author username or email address to filter commits by',
                                            'type': 'string',
                                        },
                                        'owner': {'description': 'Repository owner', 'type': 'string'},
                                        'page': {
                                            'description': 'Page number for pagination (min 1)',
                                            'minimum': 1,
                                            'type': 'number',
                                        },
                                        'perPage': {
                                            'description': 'Results per page for pagination (min 1, max 100)',
                                            'maximum': 100,
                                            'minimum': 1,
                                            'type': 'number',
                                        },
                                        'repo': {'description': 'Repository name', 'type': 'string'},
                                        'sha': {
                                            'description': 'Commit SHA, branch or tag name to list commits of. If not provided, uses the default branch of the repository. If a commit SHA is provided, will list commits up to that SHA.',
                                            'type': 'string',
                                        },
                                    },
                                    'required': ['owner', 'repo'],
                                    'type': 'object',
                                },
                                'name': 'list_commits',
                                'annotations': {'read_only': True},
                                'description': 'Get list of commits of a branch in a GitHub repository. Returns at least 30 results per page by default, but can return more if specified using the perPage parameter (up to 100).',
                            },
                            {
                                'input_schema': {
                                    'properties': {
                                        'minimal_output': {
                                            'default': True,
                                            'description': 'Return minimal repository information (default: true). When false, returns full GitHub API repository objects.',
                                            'type': 'boolean',
                                        },
                                        'order': {
                                            'description': 'Sort order',
                                            'enum': ['asc', 'desc'],
                                            'type': 'string',
                                        },
                                        'page': {
                                            'description': 'Page number for pagination (min 1)',
                                            'minimum': 1,
                                            'type': 'number',
                                        },
                                        'perPage': {
                                            'description': 'Results per page for pagination (min 1, max 100)',
                                            'maximum': 100,
                                            'minimum': 1,
                                            'type': 'number',
                                        },
                                        'query': {
                                            'description': "Repository search query. Examples: 'machine learning in:name stars:>1000 language:python', 'topic:react', 'user:facebook'. Supports advanced search syntax for precise filtering.",
                                            'type': 'string',
                                        },
                                        'sort': {
                                            'description': 'Sort repositories by field, defaults to best match',
                                            'enum': ['stars', 'forks', 'help-wanted-issues', 'updated'],
                                            'type': 'string',
                                        },
                                    },
                                    'required': ['query'],
                                    'type': 'object',
                                },
                                'name': 'search_repositories',
                                'annotations': {'read_only': True},
                                'description': 'Find GitHub repositories by name, description, readme, topics, or other metadata. Perfect for discovering projects, finding examples, or locating specific repositories across GitHub.',
                            },
                        ],
                        'type': 'mcp_list_tools',
                        'error': None,
                    },
                    tool_call_id='mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                        'arguments': '',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                        'arguments': '{"query":"pydantic-ai","sort":"best match","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                        'arguments': '',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                        'arguments': '{"query":"pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )

    event_parts: list[Any] = []

    async with agent.iter(
        user_prompt='What is the latest commit hash in that repo?', message_history=messages
    ) as agent_run:
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
                        content='Can you tell me more about pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_list_tools',
                        args={
                            'id': 'mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                            'server_label': 'test_server',
                            'tools': [],
                            'type': 'mcp_list_tools',
                            'error': None,
                        },
                        tool_call_id='mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_list_tools',
                        content={
                            'id': 'mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                            'server_label': 'test_server',
                            'tools': [
                                {
                                    'input_schema': {
                                        'properties': {
                                            'author': {
                                                'description': 'Author username or email address to filter commits by',
                                                'type': 'string',
                                            },
                                            'owner': {'description': 'Repository owner', 'type': 'string'},
                                            'page': {
                                                'description': 'Page number for pagination (min 1)',
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'perPage': {
                                                'description': 'Results per page for pagination (min 1, max 100)',
                                                'maximum': 100,
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'repo': {'description': 'Repository name', 'type': 'string'},
                                            'sha': {
                                                'description': 'Commit SHA, branch or tag name to list commits of. If not provided, uses the default branch of the repository. If a commit SHA is provided, will list commits up to that SHA.',
                                                'type': 'string',
                                            },
                                        },
                                        'required': ['owner', 'repo'],
                                        'type': 'object',
                                    },
                                    'name': 'list_commits',
                                    'annotations': {'read_only': True},
                                    'description': 'Get list of commits of a branch in a GitHub repository. Returns at least 30 results per page by default, but can return more if specified using the perPage parameter (up to 100).',
                                },
                                {
                                    'input_schema': {
                                        'properties': {
                                            'minimal_output': {
                                                'default': True,
                                                'description': 'Return minimal repository information (default: true). When false, returns full GitHub API repository objects.',
                                                'type': 'boolean',
                                            },
                                            'order': {
                                                'description': 'Sort order',
                                                'enum': ['asc', 'desc'],
                                                'type': 'string',
                                            },
                                            'page': {
                                                'description': 'Page number for pagination (min 1)',
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'perPage': {
                                                'description': 'Results per page for pagination (min 1, max 100)',
                                                'maximum': 100,
                                                'minimum': 1,
                                                'type': 'number',
                                            },
                                            'query': {
                                                'description': "Repository search query. Examples: 'machine learning in:name stars:>1000 language:python', 'topic:react', 'user:facebook'. Supports advanced search syntax for precise filtering.",
                                                'type': 'string',
                                            },
                                            'sort': {
                                                'description': 'Sort repositories by field, defaults to best match',
                                                'enum': ['stars', 'forks', 'help-wanted-issues', 'updated'],
                                                'type': 'string',
                                            },
                                        },
                                        'required': ['query'],
                                        'type': 'object',
                                    },
                                    'name': 'search_repositories',
                                    'annotations': {'read_only': True},
                                    'description': 'Find GitHub repositories by name, description, readme, topics, or other metadata. Perfect for discovering projects, finding examples, or locating specific repositories across GitHub.',
                                },
                            ],
                            'type': 'mcp_list_tools',
                            'error': None,
                        },
                        tool_call_id='mcpl_01264dba19113f900068ed20a0664881919b18a9a3989974cf',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20a271948191a97ecce27848f461',
                        signature='gAAAAABo7SCjE_wICveOw0z6kFvwtVJSAi8SGeS7LooWILhZ1wQTZR3QZHVylNYvEkcAWc9VpOKvCjdyffXyz45penIN9cSr5WCk7vP0DULso01ZudhhFPVypQ68yogEx7xMXwmVJbOqoyOZb3M39dnKkFtjQeMWd4AzGfPC8ufjRt9SEbTlRjq06bRoNBmVYP8rQCftHh9BWIvGKU0Fe5rwB7S0BtFmcowfwFs1lMyKN89BTITrVXrTCmai6mw_JvNj9PCuDiOMfuYbtbmZKb0OZOjiUyErSSFhpdyEqQZbeBF1frZVtPjPJCEIusEIHK5cT65odpRKn2uakRcUcCKGcSB6Sv0OTiSF7nJW79p-eJIfeFXxwJwiCQswp3mUt5U2a0kVsxDv2XUhP3ONOE_-vhNS2EGwRD8IglEw1F0zYqXItUUSjb2vig0SA1sNUVK4oK_EwaxYUYgxtgaC42e7qzo3Hs_18S6FXxaMSygoyk4jVMhIVJb7xbQc7-hWF1ohdXJRTpLNSvZ-z_FTePo_3Bye3HVFpSWOpUAGd1peSVrEiXXaN5kMTJpeZfHbXVTX2FN7ugabl_IgdcVOH-seE1cD5kNbodVbjxnDdxzMx8q10xtzlI7Ze9CPFJb6BZFWKEZri-sJUJAbSRBfH8NIHJQj1UjDx5vg4YRI3TvUR8RIhedC54_25cb_6HD3_0sghACWhL_9uyzPgMw4tjzIQQNtHDduB2BWpKapkSgUkJNAaujU_YOuq-uttzV2nOBXr1H6APbbG_kgNbHs1a4Z_Fvb9Jzg2oKZ1dJOAt81_VK6s2gU50EqTWCQi_k8Hnj4MnUnQIpIyy1jjHDxheKIkOo_7eTSkYsgEapY57S0NGZ6T6Ru-XE8Hql_xa7VhW7VtRyF3LADgK5DD5rjHhzp-dIbhc6Y-4KgTLvaHU-Dohj1t8uZEP8niv3I3JKKgWDWVJ9vPbioeGqOINooUzZcIaJg4BztTcP8CBMsDX5w6MPO3IoBwnhS50YLIFrkDplwqLsnIEwpaxROm0csRrTcUpl4ZiY18R9PXyBp5J8tH0Y7BqaJ6NTAdFT2OLL6TntZ6UtTCaR7U4p_H45DUl0ZKsnjxBjNQWS67DatY2U44S69Qh86ken4gh44cXA4r5NnpRbCuZmo7BhH2wH5yDRhhZc2AQ5WUwpP0eyDnjF8xunNftbwyISfNV7WuHcdaYwiu29WOx6O3c0FLJO-KhV0LvK-VZZWXYTWY55pqv436FtVoq0XS4AiKIN4zwXHx_zuy3Zw_lO41VJUZbBNL4zcAv7GDukWyiFkocpWMBhLhISuuh0T65B3R5W8_HWYJbi-zlORXGZaw3OtJqTFPKBqhRH-chnBoSy8D15C3gahfV9FKjdIuYruJ6laevbUecvydkHG8GrdfKDizmQsRYLPAxApJX4GI8fHMG5vqBgUfdDpZaWhkEzoGwywWparXBx6LVrGVKAa2GUHAhyevUooKzLuPcIeauniU0cTSgHBpGtHLvJJf-XZFQfHMeIwYxDs9exI4EwRTKRst4wq-foobKcYhRGjqv9O7BrdFIiar8iO0yiBfzdA-gCV63WajMhhWfwh7c9svtge-ZdvCvdSCmYAJRHlgBxUZ8bN2sENy3ZuW5lVk_RQBKtcjTi0uAHm6TT66bZ0JIYTp8Mns9PDTQov-p3BDTtxp4_iNFs0RnOHXvJ7nNNvUHJ2lrdO6NzquhE5iodvXsuWa0wikvsJrxVN1J9Azg==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                            'arguments': '',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'in_progress',
                            'approval_request_id': None,
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                            'arguments': '{"query":"pydantic-ai","sort":"best match","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'failed',
                            'approval_request_id': None,
                            'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20a3ca908191bdb2dc331e4e8e2c',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20a490108191823c8351b9c7b1eb',
                        signature='gAAAAABo7SCllu3A0J5TLhrToCHtBqlJXTNnUiSjyawmoj11hxQvRsn-kRItP0vUjdkSzapSsaiTTCoHAnYP6Uh52eogYu4emHs5I3e40WoGQjQ8Aog8imp25ZxF7OHWfqynluKfum-8kCYE6GHi-1k9zTIrBPZpfdwdJPKng_AjAxgknHgaNx31Xkbyjtrwl13AkGvLNlbQXhLiD21EAwcc2gBEe46BwtRXfLlms1uFIkuUWeNUupnOaHkjOrchPrqWECJA9A8gAlg29JOaDW6THXksjb6PIBvRh0UOaY_WhKEsbqdVLU-hKnACCGY3gk4EksSwT65CHJd1ByKhvi03-n1GD3w0_tlLdAGQhHeVO25kDR_16Y5geFllZHJ26XN6EsQiIE6UCsACa07iXgrrAPH3HsnKz-FNM-fgWaxihbEJShbmg0BjmTcMIGxcmLf9xbt9sDc6Hy0iZQTgVWyDj23jN1hTpIIu382KlbgMDd-c_A8GFQp8ySlh8CmuNaTjq6e8oOknbsDDladBx19FrHGcWVmpZLkaa1WoJ-_JpAo6CzqYQJMIBdm2R4w9O9km7jNgtEVJSWe4zxdYVfNqocishUfZoeD1Ncr9bkQ4AOlw0oTmrb3kEa3mRaG1gy0KHC_EuBfPMmr89XeivxzTak8QZg9meG8n1tKSqjHf8EJEIn0VBnAReR7lFpd0Sr-vuVz7Y0kVJenvSNljm2oYmwaMXctycGh5GkHAh8FfsvvGoiRVomZ4hCIy70X2p4_0nMkwNORPGkBIfEhHuL_VcIulMqhkGdtf-Q7a7D9Q6jmuSplXvKhxZH31ay3zzUc0J-1guIboH0R3GJXmBHRLi_PXao_2GS1Nhjx38PBFejmIvtliITF6YhP7DFa29eMacvRqKEiKdGZKAX1NdluBq7YSYJhOPVnV9LVosBps-ZLIfkK6d4ZRidWPcVf1j0tRBKPTHUITctPdNrk6kzyTeXUy6wA9yhq6jkcpCqJQr0wF_tc-4OkyQ1IJ05WYhrg4fAMgmu2DIWQ7NyQ5H9gbsIHEvu_QxXfNq_wlBOcgyosyxfitEgUrxCQKUbro-7rKZaVjRRPOjfHRr8j9gry-lCh4ixC8KkDdJaJ108B3szuBKMZIuhr3b_mys6XX0UGQi7RNTxwnNgt1sWnyh-F1W0r4DN1dnf7IxnAmRVhIjcG67IHhZDgsyG71tTN-0DqxmSb2Gg7p0cW__HYDFw3uYn_BeFCByY6oJJdwIGC0hqXDxd2p3qmJTxHiLPoICPRQOM2oRC1fW_XFb2VH27swHyvS-8tdkCw4OtAIchwKpALSkpg0AUdWemc0EseGxtIo49M9c7Ed5WzFXzawWKKUxJDhWQHeV7J6xOtQFs4XmwftyzbUBYVcPDcz0nz3BvSHRq5tjdKu5H5X0t_i8L9UAZAqiF58nw==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                            'arguments': '',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'in_progress',
                            'approval_request_id': None,
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                            'arguments': '{"query":"pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'failed',
                            'approval_request_id': None,
                            'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20a5933c8191a56fe5fa4b8dcd53',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20a62f648191b719c874cfe47a3d',
                        signature='gAAAAABo7SCorCLOFD5VcIU8sG7u4WVtbDowsj_gwMf8GNhOmqvSk8cxQUpKRgJV3bOa8xuz4f65f_TvG9LXdkHbZZV658XLxJliS-0ugk2jOTNLBTyEJfxsmAO3GRApwRWvb-BihcREuGHlEopM2Pw0wUJtWoVRHDqfAVECPkZrZLbW1gujztl7zCpx8l45WngYy37iUWYepVmNBm-1NiDDyPxM2gbGMkCoszu57RgWzB3dAo8KbPtpfTb5aTvLdqfi1vusRb1l8RTtTFUSYW8fuJmFccA83rej30oSN6YydiSCvO5p3_cIH7AJTQiMyMYOIrZX070frtQmezvaeRZWpIV7f2chaxdilerEmaYstO6MWg8zuolvr3oLThUp8xD89IJf8KF3HUhkXRNrvvsorl_XjGwmbuOZLaADmn5GqhyFm90VTDkgx2G1ms3X8C8OA6kz_iuo6u1TrZA0VzkFGFxZoUog9-w5Bp0oemZMuS5PtcJJK3b7BQ3f0Hz7Ol3Sp0Arm6bfbt18ZOVmy6FU3S5C7ah5uHGQf8H8e9dwOcJhsjwtEmSaYqq5akC-gX0D39QGgO-yQ7jL2glutE1pFtybVQNcREbEyt1E0K6IqW72ksR9ymN8H0T-MIzVC72peLXpVzrZ_DUZevNLLK3hIhpivngZe4ecP1ilbKs1mD4p0j7F3_lgsGObcZTOn3ku-WkLItmFLIUUrVdG5Em7lPM8lgiOWXoCIs7Xa39s8j5Rlbd-odUbWmAuUHfv7MC2c-moo3V_0BJ9eTVRi5_i6r0LbTxaWITZOpTQBCD93PczuQkEgekwJJXVI4BKCfF4xa7x4iAe4G9NRKiwjnrDxC-jFr-lxsEZ2pLR9BPQgG3FQwhS4v1HfD5teTr9YpAsS65UZfHDAdb1KKIJDtnWzrdPlx9ffHPS0hgaZSbwyZdNs7ifWBxOfpNX_Mk2V2lKUEyBKN2o6MTbN4bmODmsjJAdVVhRiIFlHmM37U0lMqZf-e13i4dDYoTZ3bwjACKUIeir-BYQphWtOCl9GT5BvEg6mI_aPoZB4VzIRAXil064UwPoA1GoTNDw0sXptNTmuqoJJVjx0nyRhoZLKvWFddHYIE_F1wHp4GV7pt6vUNeRiahG7NV6Abyoa7wEC9fYb2pHWaH01ER8iwyU_ggY_wwyD0Uf8-UjLDf_GkwXPX8ipA65ssTWSzD8bYPGkAoIGotTMYWxPEvzXXGb1WuDrE8czYTyMKM1Dj6XlBQQf-C-ujcl7BjmJqPbQkgvcOda7PHBQXM2yCYA6cLQH0wvDtuhzRFmrQlDebXYoS0i1yIT1Fl8YxmGVOEDbG5evTPFt14G90AMCLU1FH-N9c8Zo-jUDI8k9RiWKYlys3aXggJyzeZkEeyhetmuxNlJhl9kZ4gut4GCqveajz_0t6YEO_YHqpqPhmXRXFzZOUFXjPdPjMiw4CDzNqTLJBGT5eb2ekNCrROavtAG69a9z0wS4fyV1Bfzy0D1LZA22Bz3YVoIcaXixcyrfFZn2lQD2v8Jitv_OvG5qhc4x7daIWwq0-mJRZc5Ot-q2kQhhi1gnjNzCR-NAH4kqokHtCU30_HBjJdB0Yo2NVoEUu7i5ZdXZ5JsblEnKHutERGzxhf6ow9lC6QsE0iZGBmD88SOT9yth_aL48JFJpunjuzzjnZp7hfTbt2BIcbywEHiWYEO6K57_gIUsfHa_0irOFrzBisqSpBbgacZj8NFIGEw_E9sRgUaZ5I3X-PONHyj36raxv_kd6J9Y2uvNBgeVxnuLpNPRHNP8TR-t1Yks0A7975rNOdOzFYL-uCEmC-sbDB3DtBimJ2bsPkmUpTpgA-ds49DlE0zr_fREMrgQMLNBagtvOvN3OH0oUsWVZcCJYr3ADu4CZfwkDOuWQSjja4u3loG6PTjbIBe1IVL-wGa22nir5Dvi4GGd-G--9xYJVrSKYsxE4NK5M8TJu2gVPPYuGTqOel2RVbNEHR3E2pVXHLwMiwXBLjLFGA7y6ou4cgmSoB95G582Ycq2v_qMuWitdE2DITdtLgCv3bKTsRZiKqXTn6j0g4qpicowle2ob4uBulOgK8thTb3PCCbRzi6atNP1mXVgP6vCtyS72D4BCaT198fl3BtkohoqPVxL2be2fYRg1WFnudFM0jg9iYzeW6N3aE5khoEUZ4Ek5SZFT8Pggmfh1_HAg==',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='pydantic-ai is an open-source Python library (MIT) that uses large language models to generate Pydantic model classes (and JSON-Schema) from example data. It provides a CLI and Python API, supports multiple LLM backends (OpenAI, Azure, Hugging Face, etc.), and lets you customize prompts. Repo: https://github.com/birbland/pydantic-ai',
                        id='msg_01264dba19113f900068ed20a8441481918c02d9dfa25a3146',
                    ),
                ],
                usage=RequestUsage(input_tokens=876, output_tokens=571, details={'reasoning_tokens': 384}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_01264dba19113f900068ed20a0538c8191966b895a0e63290c',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the latest commit hash in that repo?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20aa125c8191a162cf51e4f7abb3',
                        signature='gAAAAABo7SCtck8m9kX4XRUPQ8v6VHc-AjqlsN8f1X4ztYz_bg4UXXr8H2Ns984NfGj7c5DHDN6obl-kF0rGX_tvCO3ga2RUmcZkqu5j677MV1LkY7rB6hvsSC4CB8Mb_yCmZeq1iwP7NJkhqHGqcZueoGiJlWvFx_2QJTODOrDLbDm4zLte41ZR1WUdWtaqq9ZBFIO_EzJxLiuKvmGKuwpNG2-qAGz86IQjhXot2pnRQTszEeM4GRJEiwW4WdfLYChV1nCvhEyeRppMOULpQyrxE2rDSiky3g0sRZ3_AR7t9vl8VDwbmopXix9K5PtddpGBw6jNMfWSFsdmI4Ivleewopb3O5_NuNJjJP2sfOzQ4BmrhLi9izfkFELnRHxhTl6zjl050TGirouasJpcwVHr9KjB-IcsaNMGnpI-M2EKAV2j1Dk8QLAwiJQYxopECGcBXOTlhQ5OxOFxNJuHvZGsMIruHNS0uAgal-YunJ6wCUpIxt8X4UbILki27Tb4-FoR-7V-RQJdSQ_t_kKWts0mltzNLO8gM3178qPRMb-XhseFvclI9KxSZvqiibvagk39gkWpPATm9z0qKgbUSNr7jhipcrm77qMiXhaL5CYkml0HbYpLukVGBgtYYvNXJmDl86gBHAUOKlfbWxHYmU_BLKQh1am5R_sqqVZ61iIbuqJagvxCxgyCRJSf6YeugU3TmbqHy4GMu134tISGXsSgD5GKj1dDfHuBBDY_a7wHg7Rh7h_TS0YmdAUXrAuzuvRAd7s2WB99BYaVVi-PHpC8QOdm41a4N7wIMdBuMAiTrhcIPfLVTqApnWMvRNnpSGtScFcd1jGJpHDmCNT2Zy0eIxOgcnguDgfWIb8jjAk_l833WileKpCEnwaNQvhqd1cbGKTv-o2JZVpTIXQJTpXrUHE7lEbs6sV9A6PAnphz3Yjs-WPBwpkgYEhz1dDeF_wDiBmSnM8Bk3V8d--qGvjT2ls8Sigleu2f9QTPwYqiZNV6_LoXMwK7vBGMR3TB_NQJq2B08Gjjv8GNnlP9z83PcWSLMyeJKBNJaAGiI-pOwI9f1GO36prPmJrVlxXFxRAHmZaDJqfXm_FxSsN_TZZZsAlPENgja_YZkYrcD10oo0W0mZY_lwXUrMXNOFmyGcDUtWkc-YK10vl-L5kAGb7GzPuApE5VaBjh7_E9Fj8likVB5cqBGFNDhhOe431WpX7dgt5whhii4hfL6HAv9G985aFuu92p_Rj_7fyFofRkamPYCIzTfm3GJM0tVb-OfLfKY36qncG2DADLJIOAbHyTU3OUxdKbV842YeLiyn-dlWfotN_0z1iNl4iJBgpv7aTjBqaemZRyACTS6DOBS9HYkiYc9GPwEQmc7U_C229xTN2_KLej0n-b3AkVrflOh0R1BvppVH5Fuvv8iEmcNIPk-I8w0ozVNCl9sntU7KRoPgD0vN0ul_pC9HHyQqAs__ZgvJCLmpP2WsDmM9BnCK9mx15fnumj8pjay1EtjGBbYXix908P_0TeGAGpjGyeaWuLtWJ2Ee5-ChHbJ4fhrl36kErGL8QTl5Y8D_HAZ7TuVz_NpppkIGvxCOZ7mVDB7W20_FzxTzTgrD1nW_h04m2Zvga1Lj6UNKlnTP8WxtXmBPYtJ73oY3AJBcKgqUeUXr4SjrfubDTgUe_j70DviFd7iBnBPyVfqRYZVTTSOIb4QFUsNlUwRWWlwsNfojVthzdmxZ2IKPtaoWDyN81C4syC8Q32ujNxVw3rh3rhvRTLpUfoXHQ5yoNuZOBES9nsPxmfEPiNCR2XPWNfdp-WdQlxdpAnBc4RUV4NgwZpTJ2JnXakldJp2YemewKtbU__Bh5Q9hsDRADdm85oe1evICe8nWK7z9rPEMqK-lrIQ7g7fFc6NA3XSwquEPumcIjRbBQyyd6hamT-8mXHnVQBsA4CJDrvAxvmDxaCiG7J4BK4KjmM05iSt8RlpSoArH6kEYjBWWfdMzUJapaRQRej4VstvNbInK7bq3IGkJpfZeNCQ8laFS4SkcfkInlDn8Pk5OXgwQlSdsKTTBrRmllOgLeU2A8Lyu8tfLFpVeNTZJJug5pj1rrFwQcyLWyr6HuTBIqBa-63Ied8rMU5l-KD6OqAqOLZyQWUcwhteCLW14yBKkJJe44Ml6Ox44FJ7nPj5N-Q3r7B_VAkJSZLayG8LcjusQA1LLFZfsT2xtXbw-pke5LWxfbfhH8P-Kw4nBYZCof3BKVhr3kfgAsghO03KFtKDsBuyN6nuhUJlfnKo3RSZl0pJDRLOW698JsGtaLW42cS0ZcbNlurJJAsWf2QVfaXsKr3-aRSr399ugKkVpAlMwVLeB_wocSlsF10SM4eNF2UkZnWpSP_Hgarh6aTl0mmnxbvEhcNR1nT5kF3bLozqh9ET-4MdUvq221XPX3-GDqjAIzktzBXlfKNtKTYYzLL97YnLVBXML0rSCmv7ajkBT6rkHw42bszEed_DWvXuqXxOXUbLQ_rH6WYRZGWUFnU-084_kIL1j3f5EHsAufb5TycdF7d_ypVOJ2MMFtemdOxDiwt18spnWwMjyefVzMsHc9u8ia_2g53xuMF0XiB2THv2iWK6M0rGYAEu6XS5nlOeWt0mXINsKeNGtQDCUxL_ihmnEFds2dSymYfdccw834G882bStP40V8P3RKlMgwMB6mrt4P0LhLPmAbdFW5EbPjJEr8fGCjyg1L8tEzq_VNhtK-CK2u22mvU0lIwyp0HKFouGGf2fr49ON5-bjajNuh1H3U5VJXqThAXVmc9BYjCYGgXJrY=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                            'arguments': '',
                            'name': 'list_commits',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'in_progress',
                            'approval_request_id': None,
                            'error': None,
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                            'arguments': '{"owner":"birbland","repo":"pydantic-ai","author":"","page":1,"perPage":1,"sha":""}',
                            'name': 'list_commits',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'failed',
                            'approval_request_id': None,
                            'error': {
                                'type': 'mcp_tool_execution_error',
                                'content': [
                                    {
                                        'type': 'text',
                                        'text': 'failed to list commits: : GET https://api.github.com/repos/birbland/pydantic-ai/commits?page=1&per_page=1: 404 Not Found []',
                                        'annotations': None,
                                        'meta': None,
                                    }
                                ],
                            },
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20aedd88819191ce657ba3901f3b',
                        signature='gAAAAABo7SCvJu7D1daOEGJtJo9OqWIpUvDM8SBE622ikyy5fTFiTmgeinJJuROJ4VABk-A1ieBbPw0TY8DRAWcBZIYLdS47qRJxEg3_cgbtiJK9fsAAcKIi4loo5CPHxbd04C0uHhMOKASrThksMiH4Fa__M2O01-1JpgoO0OZ3Ll5ZXRXocxkVsEmHIxXezk0uz7A7kkoxUfo8GKjcr_BQojtku0fSQj5iQIVy7M_eGo9e7deZMFb61OPZ4xnsLjwn3ljuXi1IN8R0awjUPm27U52hKFhCFyLWKdpsmsg6IkqlKxVyhl1pfrZ81DJJUpRjCkrh-L_2ccl13fCM25EVi8gB6uIm0ts9Hk9oCliLLjwTAe45T_6SMX5ZPEK7AE0zjyFoGJ1mtlj61_1a0Eo5S0xBpHg7dJDiNMzPQ84N6sDoF_KUnv49CxZnAHZ8JNjsyUOrXcUSE9xTAnGsQ0dg_Fxmi7ZBUMuq0Ngg9gJh8ovdZ8i0Ij_Ln4iCulrFA0gB9CuXhTEn-9He-pSjoAQjoMlr8_NlRC4Jq7Y8ZMQ-xbBZVHXUps5PFW_Hqo3cospgPJiP97ybsSNO2U3430BPAQztF4C5DU_ak62FQMh5Xb27ZrNflVEIiYcaY7pluBBNNfo9r4hf3NIECd6sv5NVZnUQVSBqpWREKy1RlRTw_dMJxwPlLpsuVRWfYTUOc3WImHMy5Of1fQO4lGMZeX3iTWa0RJ-B1pNG8xSUNMGPhAi6Z5xtriRqq8KM-H8wSutQ_4I57mnZU9G2x4zNjxUk6CFJAgSTcV-wHlK_971UN1uQ_iuegWCtVdUAblJIcmWGtLx-WbR4S3OVahNExJacXRPn68DEfO2gbqyKZL6obqEZP68nhSAzbf940UqIOAQbzRiM0tpJc3KiSObI6TKNw8xHae2I0IJToXt-F0m_5fRdYWZ0wCDB5M_y71rMNQWT8gLhxi9VmFX1MCU5yjCkfA8hterEsoBHx-HtpFsKMdb4erJknsa9YCw2rqmzFqit9zLeLp_NZCpTLCuB3I0U8HBEfNBr-WVtoccJfHbX2fnt-7O99mUgeDIUd5LyHGnAf4-UZO3aVFcN2hlHIYvehz4fxhVRKtl0lyhe6nI4glXvixkv1GiAvp0nuTeLbRxKgsXV1eukObeau9Xrxy_Qu7aEb3FpJZtqlRR7r6CORSA94OCwcm0=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                            'arguments': '',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'in_progress',
                            'approval_request_id': None,
                            'error': None,
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                            'arguments': '{"query":"birbland/pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': '{"total_count": 0, "incomplete_results": false, "items": []}',
                            'status': 'completed',
                            'approval_request_id': None,
                            'error': None,
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20b24d448191aca08726d96a1b9b',
                        signature='gAAAAABo7SC0SAPup3riyM8IB9BUJlA-gEqj0SdQB-vhZJiegBCPxBWfP7RYhsDhk4ZRmfX9Ea-OGfwIDP6b02QshqVwNBlV_9y5kdAWwU44csAB2i1r6tXAU95K35KbdWPF5rIlxUVunc8b8G5mEIpjrBm4o1Ja_0VPfLddFHQ3jHUGk57qp-U93EG7Owq7Mr5_0AKsa1yuiy_SnCRRujwP7KBIyzoquZd2hGgSCO5vFI31zFTo8Uq8zDbRvTjdCVvf4wXdk8NhWj9bUHtp9Udng6Up5Yk0y_4jA4Qp_sWa_auRXBYYGMg6pWHtjqfnrjzPmtJfbZI9uWW5K7-KZ7rAjj5EO2CYgAAghrtSLth80bNRvo9X3aL0Nr_33krq8ebPwMu6rYwEY0AbSJilgSqZ5RGaOO5AzGJDDzjo0SEwKxITsJnLSIGOguYf5Xq8U2DTFTuSbkUvdTBtNQErPjHouZr-jCjXska6T2M2Y_C7igyD6mlf1-y92kRmkr5HdMaqoD99soOzJIg0gvTAvxlt9HD4focgtkNGwt1qnHXCwch82lpiurr5G5uOGaql-m6GM40Fa0clmnc8sfSYemjVhfLZ8O_rzUY_-bzoUj9BbrPRd8WW01_1AJ38f5976eA7xUJ6vMgHABvsojmhHTMaRGnXNBeFz1gQDxr_ZJJ8i_O9Tursq4CfaMzSloI66W8V39OIVlM4mDwh4P8LSq2nTr9XgIo2AEenFH-m8030QyN8r6HEfbCUR-lINvXMZlfDsPYvmD9Os4g34kGIKD-LP4cnka4NWMLlO8E0k02cEzCzagOXFWjF81At11rAAECCilAOBvFFUcH8h8Vfu9vdY2zLcvrF4Vo16eA3yiykdduV2yqamhh8JqdPEcDKyCO0hgwwrBbGm6U8sXDkCb7WguIQkW9C1kKvTBjt8R6halFwwOPaTJZw1GFay5q1HwCWVYPGvBHZ8ftaOf4PvDfMPsM5DmkKWkx9JnF231k7KuaRrnBnUAfQr_OjIX6Qzx_qQlDr6d7pM91iENR4tNqECPsmLysyezf2NWFnRPGrHJCYptIIAtPryCe26yGs06h6xMH16ilOUmOqMcaR0sMWgvh7cnVzTn9Ld_GxDo7eSpuUE5fWfULn8b60ATxPxfaOWPXNah-FXBVstHjoPnjcr1UwTxZaHniIOM6x65GdV-hGq3I0GE05NoHyDmxqrrrH0e5BMDMfLSkxxioHVJH76fMr1sWM0RatfA8bp5m5SdVkqZqwNc4ISDUepOdHXMQ-L6AeOD0fTjZLgLQTiVpFiOdd1xTbOzEKoVyUC3B8BLzOgCkIXOe0Y9LWbPLBiVDQ_fTw4zA1Dg5sbTlMEo18P7gRuHCguwnwWdV1BapFKlGB4siS3_gzzANfhIrBd0DAEi58huQ7BQYyiM687bB-fdbVXDDkONkEHAHp-9NYI1uNtE2Y2dFWBrCcI_Lx3rTL160o9wPk4yWgXBV0OG7MzUi_PBvQufC8H_ZSQRr56sm3O1TvqKrxQRb-3W3SjjXTUppCJcC4GrwYt3EQGDoP4BTibpjxNOApCztBEF7Gp0cQx8-EqowPnG-Um1JPvQ_muOU3CtfH',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                            'arguments': '',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'in_progress',
                            'approval_request_id': None,
                            'error': None,
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                            'arguments': '{"query":"pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":false}',
                            'name': 'search_repositories',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'failed',
                            'approval_request_id': None,
                            'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20b553248191aac5726e644bee06',
                        signature='gAAAAABo7SC5jcGJ6jMXEXn_eUzCBYYoiNwyUku-6dq3nXHcddbsNLrIzQZ4th-nONpHrIsACXKgm0017UgYttCex_bBeeLi10biUtJBtg7nFPrCTqAxX8u5nRFb5RQy-CXZGkUFk2o0C4Pjwn_KcsTwnI2LtX_rLu8jN2B1JjHDJpfzRi9wOez8E_F3MTwXdAFWQ_ccIcNegA7v_7GcuwnsNdhTFgDsZwPrMloQ98bOWQQhmrk_gHEM8tzQKVJBNn6yzLUn_3b8nkhX6W0cN465J-xCtxWP7_KT5EurorkH2fsHNa9r1MUDEr0HH8Ls4wEAe4mVEAC9H_Bsb6BFTrIt0sM-2gM1w4KSpdSQwoe01bR71G3foU6bRqtIoG1Rdpv5XXMdkS0AR1rNUehSltCnXuHevoelk3EWEDRO9u6yzVa5oZAuvAyWv5xt_sZ-7OZCacnOYuIXyWZ5iGxSYcZ4UYFmIkaguixXRrVnZ-HhyUN_wwFB-6hqVG197Bjx0Jt9mL_7-FkOiahCYXVzQ4nMbBre0gXwA0kbxa_sl_qL6a8lIM90XRoJzTA2KiUZikYZJy8vS4AxVt3wC64qqWqDwFbveM4t2P_ymbOrUN07o_AvU1pbApdwPVQkxl_RCZ_bINVn8c2gXzXFmBSPWxBtKmAQuM6EjeqUenaHmBqNyiEs8Yg5tghy1_ORynFDdaXn4I5iKLCt4gatlnbJMyUKmUVw54zcNr4hQaetIkGnjl_pkQJ6T9G1IkhcLOljN1dfBEYjjTw1DEGLEHbJl4B1-ovhMd2igGHkW10R9AaKPwCW1ZsiGQ6R7jRedm5yIcmSZSHf_KNGafAcls73P9hPJSn6g7CDfmQyjfTC-Xv-IxTxNOGdh0Vh8aIbPhMNTrRqaGrCMoPhad1lmZ8y-OyJ68awMbt8gl3n8TRP-im2g26ZwyTIQTJG9hvEO3O4_DB9lKrZB79POCi3BGA9aKyeX2sOb3PXjurDm4agb0rvpllcVCGILWE2yal4eHW4gb3V0z54YNe4EHyfo6sO3gcUQ-fKSnfzAnD8rEpFzEzQnCZuOSiELn14pQSHllyX_3C0imhkycZI65HxsBRGqxwHdoRH361M4Cv0T5jhAL0eE3lnh3TUGQM35kwI6HKcqfl3SdPBjKcNE4-Il5uuX7jayYoQc4pDbTD07qhPUlc2alHpnxs4hCW_vtjvDeQIK-N6icdR0_hIZtXvHHcTHH7vzYX1d9WYv1vpanKVZI_bZ8wja5r_iupH2UCI6kN0GaeDpX4y2pD3znGXE9zGN0dQzZibBWuGZAYzR8KcWUmEwcDfkrtIREyFsoC4cwxJFGqP5__Ix8hXubDVT_4sAjsxhRyi6TlT8ovfcS7z2-4vt0BE3Y6snLa73eXXnOpAzoZyPn5-XHlxS1cAwE_-_H_HVPQ724LzpE5f45z4YIl7Qs4R3UdG_Qqq_7CS2uYn91YMVFk5GA5iU06DJ0TULOKFDBzOq968U6IxuzEKAVVVUfWqtrLjxGIYsS99gvaVO-UvCCIS05m3IbikU1Opq35V23jIjmj-ABf4Hb8n8xXTEbcjboJMnWNDlVa5EzlNTbHyNkTybcxn9Ms4tqH33CJo8ckkwPJjvJ-FH7yzqd_-07K3tNjiKjKC8QfuoDkDhLao8oUCvOsO17tP97Re2L7-eiayI65zbVJx6VWbLOo3uOj3m-uiSNFz0D9_PobhxtxzCO8uLdkFNgewmlvARca9b1X1cKjSyU4EQlGI70gSBMRVL8ivCmFMWqNshU6dg8JglA663OGBqn06Z6evKJ42i6Wfg_qStbKN6Q_UEb7ui3v50TYtAl85-ClCwSivnQaFp1Gmg3ZzovETulR5B7wKfEO1PkFk9qZu0YBbXw_yYgo9cDsSEg5SQWeOrvxBBevbwyTC3EOSKx-vqdMRDeKs0szu7XtTVWFwl3r4fOrVVFU_m9o7JO0=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_call',
                        args={
                            'id': 'mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                            'arguments': '',
                            'name': 'list_commits',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'in_progress',
                            'approval_request_id': None,
                            'error': None,
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_call',
                        content={
                            'id': 'mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                            'arguments': '{"owner":"birbland","repo":"pydantic-ai","author":"","page":1,"perPage":1,"sha":"main"}',
                            'name': 'list_commits',
                            'server_label': 'test_server',
                            'type': 'mcp_call',
                            'output': None,
                            'status': 'failed',
                            'approval_request_id': None,
                            'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                        },
                        tool_call_id='mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_01264dba19113f900068ed20ba6fb4819187a758a692c32ef9',
                        signature='gAAAAABo7SDATu0aE5uxw8bwMrwhIUB1-fWUgek4WHxbR1pj9nvaUA7AvsFOw6pabaCzNqQGptzS1z53ls3i3FNrt9EYD8HYG8r7TmA71ydccAQrxdihl6hV8-jtAkK1btiU6Ur3X2aNR8iBf3hBugsqfevuEpKsGgqw2ASPWY_6d21ojaqdIGVxwDXc1W466Y6i_r66LsJWqeNmm55uoPG7IQqo2JAp5sSNRxAP_QE4x4-9XqN34fWXfJURvYuMDyZYTe2x7KCcKrGLc7X2Wou6DNXQeKWhO-zFQLPdaVbeT2pazb90DBb2jhTkLBmYEJmy88fDjmaHe5c4LFS4mBDaRPTnETPqRBxSdpGwIVUZ3DiskGHEgqK_GxfOE0UBzQonHIbTQYoho-52GVIpmaiPIPDaV9efC_OGnitYZrtVPndyA7XWyvtjftqReTGL5Xx331iswkKVJtwCYBj-nN-BA-Au_xSy_iy5YjD7314WhSA0bR9ZtxSjF50NmEX4CO6lOeXS83ODAYkOdi0MKbshs-AjeXWQmFZucEyNxucIRyWyoRnvxOqtgXQ37o-HpgrD35AJpDcAWXnrh4l-BTC63i8ZwWTnIiDkeXkOmUxjqiaIBLUoC852q9AbLzS_Ly9FEM1IlHG-12KhO3D1oU0S1tx-0nt6lj40qVN00IyrXQKc1UnGSttg1xIwuhUuG8vE83WZrs7DBJzVpZEobzqxBcOWfLDciLPLgxDHfZ7KhOKpoAa8OX76CuphrXrk_kd_7QHmka7H3wJ3vq6XrUAhKguJtmXZp4xJqxcqDrDS9c98dcp-UsnScnlIEOA3jf-j-VnCVk_lp2PH3iLpEIK6Ujr5m5K2b-5frZ0p-Vqc7s_5jRLsgZWEyI-jre-BQLvPt1_GBTqUrkQ9LVPsNjcp-mhT07Rr0FYuN-ix32gGpoTAvZtbLVLdqveSspVtrikuEKDv4tCSRT0WKawDujMcLpjepUKuSXkJfwGI9gXBuMY037L4a_9ucXRjqjj068kwF_TdlLe7utXSdqGJwui6XkJUgI1BWxGI84pWzieXbo0eRhhe_zK_UaB4-Nq1XZIuJZ3Nz2qTfEMhX-9-WDlHl8VxCi6_UcqP26TmVulBlyvz38P3zgHDM-5kJ6DBia_UDi-7asHKLG3GJ-VKQHiJGzOgkkvNny9UY-H0pJpsLuxIVunsrGL5fbqsDfsdQd7lYUZ3nz291JBnd7gmjzBGRe3rwCk_IbTThl11aP4373F0Q5XHAoJcCAaYeEYGoehT6CdRknek3A9b3BwKGdIqIG6wOglNPg8-znat4ynbF13asaW1oWMnje9O6qYLNg8Fl6Bztz9Ufy1Kh01UFcnm_pS8sW9mP6rRQYffhyGlAXPighlr0xRG0QAbdBUC_wLyIL7EWg5Nb5LQ8W_hvzAy76OboCB5rO4Gg6HOIt7KucpJDSP9UecUxLZN-5ir0PU6V7AS57awYUdAbxXkLJmWXcHL3y2Svsf56lHIzrwoPhmMMM59OaTGgCOYazMSWg3zmRdFlUWGdzzx-hrwNMb3q6LYdplA_648UErNWdOO3A90AiSdNCKo9Q4sRXkqPC15fBNaO5d9iLuGHysj0BrMzx7Q0CMtDRcXQBC7CcihKeTMXwsuVqi1OLrzk7d6JjiYu8-s0JBRaDJfyD6MEyKZo49TkfOrLx2OX_ArJRD4xtbjFabRCZVKXKLhSlRP67xPIuhLbNasNsKL0BjZlnFXgcSSrmMABpZY1wuyCPFl84uUwHR1dxnN3IJJsAzWBAicDhvcwhpvK8uGPKPtIRM1G5hFpFCT9dCfCkFPJ1y1h9np9N6-Hf4Q5BZ0oT3i95mIIrsbFcQrPorrJA93oOiziN3elBen36XwEiKpTv3RU2Co-za0PPq5FdMaBf5_wkcheEiP6UsRV8dLtMFYS71l2nLFPAMIyOu9U0YRKcPQgIDIO3kbc8FwDOTX4yxdX8gwSkAsmD6zSADYL_l8IJWMtIOB3QAsE9WwAlwqvqkiuGEHonQDEZTkcH_5PhVmpavFxVkLaAMTbuqabIYFGzf6w-ucfHIrwLRHL9a-WHVMuPosaN9e9gmplDgdouL3Eqte-Br6sgT7NXntO5QZhR1USjGymuwtk4R-g9ueK4ThV7epWrolWfKycq9MqLV_4-ZGpCqqIorz5Bz17GqkAVFII8J_TVeMHJq-XekQFRxuNC61w3FKbT-v8QMHhQceZFALvpTKlyzrTG17X_RNO5y5NmAlRFmH0vmh7duqYHeBjvggPdWAa_XaCDnZmMSKApPVzNojg_63y7rpKLSTqzHIcJExh3NNjmtEV4YdL3qS8OUYuDiv1gxWjV23rsqc102Q6R2jxzwOz0pmuNd2Bmk3JkEbn6bspn38OLcvPzDM7iW9oSOlXFd4zgfbGd8pHJN0u1gi1S0FKFgOeqIAXQyfHDZvk7dwlsYfU8GuxHwD4g9xNxrSSuPlDRsMUWaUtf78AwiZOQi87lalgI0Nq35RUgnvUHveLItLPXfU54qmYezaDHxYbHvtbJ4wgmxzuVTew2pHX-2mKZZ2iFAe4Qoi3dcQSMpwNpEYP9n_Ve_tGw9jwqiQ0-kMr7PdNYo9Hcf42GvbAPgsMU2tXJNPy3oeH_03v962BT892d8fZjr2xEmOkpwkxxQ2vtAZpN_xpX_4_9XMXYiE4mcCLIXGzt1oFEkSGyP1TfOo7SMObDHzxeCUlNWfLrip2eRx_V3JTxaqnX4NvLfUNDazoubnJICvyqnqBevH1_9Wls_h9NrAd5an83e32jCUkY7CWlb5rmxs3KzcrbKgLd1LODuS_VTqxHSchM5Aga7qi2zCVtEdHdmRBqoze5NRYjUm_btgOs7hQ2bNSU4QJf8AwrMNr4ib96e8JePKUo5jmiOxOEHGXmb1JsEU2940XAXFVrgAcKdzDCTOPZoErotW8GQ52aQJPR3TeUUiZ-z0st-ISIpSx8Zu4cjPmJnhbC_TxtJ1FlAk-97Xc5IncABu3wP3SJM28wNZO44qvzZZNxrzX2PGvP6cWjurJZB5o9dgU9DBDFDpBZf2V9KZy2hV4sz2QTmoJQam5J5I3SBmciHvlJLRK0r115W62bhxFG5YJfZ11vPTBc_pKqwnoam0xlV9kRNbfW5LFs8AMCUtbjAZck0ca1o8GW2mcStfqyyxsvmOLcZG6NiLrcu_aAnQ9hIToJ5J9jqvxzpyF9THyMS9qbMYvw2lvFX3lPkH3RzETV7OkCthyFMDicaDTB6kocsdfhKepXxOviDU8LFlUueGA6kLWVLI8S6YrYNT9EU4xFqRdyRv5T8ARZEjiCbu-zoHovo2ysdjuvOKH3XO6kHL58FIGfOq_z95vwtcqfGtC_BFcWRYaSSNFugWeNa59zsqfFxgyTbDeJhldtd2AYPHApJi_eHfvX4HcA6_IQRsTFEgBhnd-Pjyoq17bh7fYj8gQsFJl3htEwZ3ecIoe20cEsPVqi26HJJSkWeYWz25-0fMAh1R_SY_kCjvxiUHx2LsmR7CF11sN1YfCys3AIxIi4cjLoUii_ygvPG9dcm2hXHdNoKmlUSQHZLK-f0z6za1igsKmZxrCBMrvR682el2hk7L9lEyp0JjSqSpA2Do5ErXrDi4VlNdBgyOW5rMrWwkfaSN0P2FJRDP5GEvMMEcUYVtaDQkwEx36L8fOqNaNcgfp3Fnt-OCMxK6C-B6rYdzSSbT_QosaHmf2HJ5dOjjvJroXwJB0XR6q5ShA7eepBWFLmhM6fXZRI-3gMkHt_O5OwuOENTHjzQb9EyZ3ADcFTs4vcl__fFlqWFlJenmj3f2g8ect1HAXzD2-rJD9SITWzkFCuJK9yRLvqIH4rBT08JUZm85a5BPN1kXJDVPdfGQjOmBsfIQz6m-hR8Z_cGMNi5MkZNYkBL1zHRA4s5KO0UoS-kR03_nTwwWzk8N_DymgyR49Pgf_gmWDSSVN9E2GmcAUaNjKASecJzJ0UJmz75WzmK4eTR02BRviebcr7-ykLLpuf-7EDzx',
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
I was not able to fetch it automatically (GitHub returned a 404), but you can see the latest commit hash here:  \n\
https://github.com/birbland/pydantic-ai/commits/main\
""",
                        id='msg_01264dba19113f900068ed20c0edd4819191da764a7494bab3',
                    ),
                ],
                usage=RequestUsage(input_tokens=1822, output_tokens=1459, details={'reasoning_tokens': 1216}),
                model_name='o4-mini-2025-04-16',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_01264dba19113f900068ed20a99ddc81918b9dc04ed1537d30',
                finish_reason='stop',
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_01264dba19113f900068ed20aa125c8191a162cf51e4f7abb3',
                    signature='gAAAAABo7SCtck8m9kX4XRUPQ8v6VHc-AjqlsN8f1X4ztYz_bg4UXXr8H2Ns984NfGj7c5DHDN6obl-kF0rGX_tvCO3ga2RUmcZkqu5j677MV1LkY7rB6hvsSC4CB8Mb_yCmZeq1iwP7NJkhqHGqcZueoGiJlWvFx_2QJTODOrDLbDm4zLte41ZR1WUdWtaqq9ZBFIO_EzJxLiuKvmGKuwpNG2-qAGz86IQjhXot2pnRQTszEeM4GRJEiwW4WdfLYChV1nCvhEyeRppMOULpQyrxE2rDSiky3g0sRZ3_AR7t9vl8VDwbmopXix9K5PtddpGBw6jNMfWSFsdmI4Ivleewopb3O5_NuNJjJP2sfOzQ4BmrhLi9izfkFELnRHxhTl6zjl050TGirouasJpcwVHr9KjB-IcsaNMGnpI-M2EKAV2j1Dk8QLAwiJQYxopECGcBXOTlhQ5OxOFxNJuHvZGsMIruHNS0uAgal-YunJ6wCUpIxt8X4UbILki27Tb4-FoR-7V-RQJdSQ_t_kKWts0mltzNLO8gM3178qPRMb-XhseFvclI9KxSZvqiibvagk39gkWpPATm9z0qKgbUSNr7jhipcrm77qMiXhaL5CYkml0HbYpLukVGBgtYYvNXJmDl86gBHAUOKlfbWxHYmU_BLKQh1am5R_sqqVZ61iIbuqJagvxCxgyCRJSf6YeugU3TmbqHy4GMu134tISGXsSgD5GKj1dDfHuBBDY_a7wHg7Rh7h_TS0YmdAUXrAuzuvRAd7s2WB99BYaVVi-PHpC8QOdm41a4N7wIMdBuMAiTrhcIPfLVTqApnWMvRNnpSGtScFcd1jGJpHDmCNT2Zy0eIxOgcnguDgfWIb8jjAk_l833WileKpCEnwaNQvhqd1cbGKTv-o2JZVpTIXQJTpXrUHE7lEbs6sV9A6PAnphz3Yjs-WPBwpkgYEhz1dDeF_wDiBmSnM8Bk3V8d--qGvjT2ls8Sigleu2f9QTPwYqiZNV6_LoXMwK7vBGMR3TB_NQJq2B08Gjjv8GNnlP9z83PcWSLMyeJKBNJaAGiI-pOwI9f1GO36prPmJrVlxXFxRAHmZaDJqfXm_FxSsN_TZZZsAlPENgja_YZkYrcD10oo0W0mZY_lwXUrMXNOFmyGcDUtWkc-YK10vl-L5kAGb7GzPuApE5VaBjh7_E9Fj8likVB5cqBGFNDhhOe431WpX7dgt5whhii4hfL6HAv9G985aFuu92p_Rj_7fyFofRkamPYCIzTfm3GJM0tVb-OfLfKY36qncG2DADLJIOAbHyTU3OUxdKbV842YeLiyn-dlWfotN_0z1iNl4iJBgpv7aTjBqaemZRyACTS6DOBS9HYkiYc9GPwEQmc7U_C229xTN2_KLej0n-b3AkVrflOh0R1BvppVH5Fuvv8iEmcNIPk-I8w0ozVNCl9sntU7KRoPgD0vN0ul_pC9HHyQqAs__ZgvJCLmpP2WsDmM9BnCK9mx15fnumj8pjay1EtjGBbYXix908P_0TeGAGpjGyeaWuLtWJ2Ee5-ChHbJ4fhrl36kErGL8QTl5Y8D_HAZ7TuVz_NpppkIGvxCOZ7mVDB7W20_FzxTzTgrD1nW_h04m2Zvga1Lj6UNKlnTP8WxtXmBPYtJ73oY3AJBcKgqUeUXr4SjrfubDTgUe_j70DviFd7iBnBPyVfqRYZVTTSOIb4QFUsNlUwRWWlwsNfojVthzdmxZ2IKPtaoWDyN81C4syC8Q32ujNxVw3rh3rhvRTLpUfoXHQ5yoNuZOBES9nsPxmfEPiNCR2XPWNfdp-WdQlxdpAnBc4RUV4NgwZpTJ2JnXakldJp2YemewKtbU__Bh5Q9hsDRADdm85oe1evICe8nWK7z9rPEMqK-lrIQ7g7fFc6NA3XSwquEPumcIjRbBQyyd6hamT-8mXHnVQBsA4CJDrvAxvmDxaCiG7J4BK4KjmM05iSt8RlpSoArH6kEYjBWWfdMzUJapaRQRej4VstvNbInK7bq3IGkJpfZeNCQ8laFS4SkcfkInlDn8Pk5OXgwQlSdsKTTBrRmllOgLeU2A8Lyu8tfLFpVeNTZJJug5pj1rrFwQcyLWyr6HuTBIqBa-63Ied8rMU5l-KD6OqAqOLZyQWUcwhteCLW14yBKkJJe44Ml6Ox44FJ7nPj5N-Q3r7B_VAkJSZLayG8LcjusQA1LLFZfsT2xtXbw-pke5LWxfbfhH8P-Kw4nBYZCof3BKVhr3kfgAsghO03KFtKDsBuyN6nuhUJlfnKo3RSZl0pJDRLOW698JsGtaLW42cS0ZcbNlurJJAsWf2QVfaXsKr3-aRSr399ugKkVpAlMwVLeB_wocSlsF10SM4eNF2UkZnWpSP_Hgarh6aTl0mmnxbvEhcNR1nT5kF3bLozqh9ET-4MdUvq221XPX3-GDqjAIzktzBXlfKNtKTYYzLL97YnLVBXML0rSCmv7ajkBT6rkHw42bszEed_DWvXuqXxOXUbLQ_rH6WYRZGWUFnU-084_kIL1j3f5EHsAufb5TycdF7d_ypVOJ2MMFtemdOxDiwt18spnWwMjyefVzMsHc9u8ia_2g53xuMF0XiB2THv2iWK6M0rGYAEu6XS5nlOeWt0mXINsKeNGtQDCUxL_ihmnEFds2dSymYfdccw834G882bStP40V8P3RKlMgwMB6mrt4P0LhLPmAbdFW5EbPjJEr8fGCjyg1L8tEzq_VNhtK-CK2u22mvU0lIwyp0HKFouGGf2fr49ON5-bjajNuh1H3U5VJXqThAXVmc9BYjCYGgXJrY=',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                        'arguments': '',
                        'name': 'list_commits',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                        'arguments': '{"owner":"birbland","repo":"pydantic-ai","author":"","page":1,"perPage":1,"sha":""}',
                        'name': 'list_commits',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {
                            'type': 'mcp_tool_execution_error',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': 'failed to list commits: : GET https://api.github.com/repos/birbland/pydantic-ai/commits?page=1&per_page=1: 404 Not Found []',
                                    'annotations': None,
                                    'meta': None,
                                }
                            ],
                        },
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=3,
                part=ThinkingPart(
                    content='',
                    id='rs_01264dba19113f900068ed20aedd88819191ce657ba3901f3b',
                    signature='gAAAAABo7SCvJu7D1daOEGJtJo9OqWIpUvDM8SBE622ikyy5fTFiTmgeinJJuROJ4VABk-A1ieBbPw0TY8DRAWcBZIYLdS47qRJxEg3_cgbtiJK9fsAAcKIi4loo5CPHxbd04C0uHhMOKASrThksMiH4Fa__M2O01-1JpgoO0OZ3Ll5ZXRXocxkVsEmHIxXezk0uz7A7kkoxUfo8GKjcr_BQojtku0fSQj5iQIVy7M_eGo9e7deZMFb61OPZ4xnsLjwn3ljuXi1IN8R0awjUPm27U52hKFhCFyLWKdpsmsg6IkqlKxVyhl1pfrZ81DJJUpRjCkrh-L_2ccl13fCM25EVi8gB6uIm0ts9Hk9oCliLLjwTAe45T_6SMX5ZPEK7AE0zjyFoGJ1mtlj61_1a0Eo5S0xBpHg7dJDiNMzPQ84N6sDoF_KUnv49CxZnAHZ8JNjsyUOrXcUSE9xTAnGsQ0dg_Fxmi7ZBUMuq0Ngg9gJh8ovdZ8i0Ij_Ln4iCulrFA0gB9CuXhTEn-9He-pSjoAQjoMlr8_NlRC4Jq7Y8ZMQ-xbBZVHXUps5PFW_Hqo3cospgPJiP97ybsSNO2U3430BPAQztF4C5DU_ak62FQMh5Xb27ZrNflVEIiYcaY7pluBBNNfo9r4hf3NIECd6sv5NVZnUQVSBqpWREKy1RlRTw_dMJxwPlLpsuVRWfYTUOc3WImHMy5Of1fQO4lGMZeX3iTWa0RJ-B1pNG8xSUNMGPhAi6Z5xtriRqq8KM-H8wSutQ_4I57mnZU9G2x4zNjxUk6CFJAgSTcV-wHlK_971UN1uQ_iuegWCtVdUAblJIcmWGtLx-WbR4S3OVahNExJacXRPn68DEfO2gbqyKZL6obqEZP68nhSAzbf940UqIOAQbzRiM0tpJc3KiSObI6TKNw8xHae2I0IJToXt-F0m_5fRdYWZ0wCDB5M_y71rMNQWT8gLhxi9VmFX1MCU5yjCkfA8hterEsoBHx-HtpFsKMdb4erJknsa9YCw2rqmzFqit9zLeLp_NZCpTLCuB3I0U8HBEfNBr-WVtoccJfHbX2fnt-7O99mUgeDIUd5LyHGnAf4-UZO3aVFcN2hlHIYvehz4fxhVRKtl0lyhe6nI4glXvixkv1GiAvp0nuTeLbRxKgsXV1eukObeau9Xrxy_Qu7aEb3FpJZtqlRR7r6CORSA94OCwcm0=',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                        'arguments': '',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=5,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                        'arguments': '{"query":"birbland/pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': '{"total_count": 0, "incomplete_results": false, "items": []}',
                        'status': 'completed',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=6,
                part=ThinkingPart(
                    content='',
                    id='rs_01264dba19113f900068ed20b24d448191aca08726d96a1b9b',
                    signature='gAAAAABo7SC0SAPup3riyM8IB9BUJlA-gEqj0SdQB-vhZJiegBCPxBWfP7RYhsDhk4ZRmfX9Ea-OGfwIDP6b02QshqVwNBlV_9y5kdAWwU44csAB2i1r6tXAU95K35KbdWPF5rIlxUVunc8b8G5mEIpjrBm4o1Ja_0VPfLddFHQ3jHUGk57qp-U93EG7Owq7Mr5_0AKsa1yuiy_SnCRRujwP7KBIyzoquZd2hGgSCO5vFI31zFTo8Uq8zDbRvTjdCVvf4wXdk8NhWj9bUHtp9Udng6Up5Yk0y_4jA4Qp_sWa_auRXBYYGMg6pWHtjqfnrjzPmtJfbZI9uWW5K7-KZ7rAjj5EO2CYgAAghrtSLth80bNRvo9X3aL0Nr_33krq8ebPwMu6rYwEY0AbSJilgSqZ5RGaOO5AzGJDDzjo0SEwKxITsJnLSIGOguYf5Xq8U2DTFTuSbkUvdTBtNQErPjHouZr-jCjXska6T2M2Y_C7igyD6mlf1-y92kRmkr5HdMaqoD99soOzJIg0gvTAvxlt9HD4focgtkNGwt1qnHXCwch82lpiurr5G5uOGaql-m6GM40Fa0clmnc8sfSYemjVhfLZ8O_rzUY_-bzoUj9BbrPRd8WW01_1AJ38f5976eA7xUJ6vMgHABvsojmhHTMaRGnXNBeFz1gQDxr_ZJJ8i_O9Tursq4CfaMzSloI66W8V39OIVlM4mDwh4P8LSq2nTr9XgIo2AEenFH-m8030QyN8r6HEfbCUR-lINvXMZlfDsPYvmD9Os4g34kGIKD-LP4cnka4NWMLlO8E0k02cEzCzagOXFWjF81At11rAAECCilAOBvFFUcH8h8Vfu9vdY2zLcvrF4Vo16eA3yiykdduV2yqamhh8JqdPEcDKyCO0hgwwrBbGm6U8sXDkCb7WguIQkW9C1kKvTBjt8R6halFwwOPaTJZw1GFay5q1HwCWVYPGvBHZ8ftaOf4PvDfMPsM5DmkKWkx9JnF231k7KuaRrnBnUAfQr_OjIX6Qzx_qQlDr6d7pM91iENR4tNqECPsmLysyezf2NWFnRPGrHJCYptIIAtPryCe26yGs06h6xMH16ilOUmOqMcaR0sMWgvh7cnVzTn9Ld_GxDo7eSpuUE5fWfULn8b60ATxPxfaOWPXNah-FXBVstHjoPnjcr1UwTxZaHniIOM6x65GdV-hGq3I0GE05NoHyDmxqrrrH0e5BMDMfLSkxxioHVJH76fMr1sWM0RatfA8bp5m5SdVkqZqwNc4ISDUepOdHXMQ-L6AeOD0fTjZLgLQTiVpFiOdd1xTbOzEKoVyUC3B8BLzOgCkIXOe0Y9LWbPLBiVDQ_fTw4zA1Dg5sbTlMEo18P7gRuHCguwnwWdV1BapFKlGB4siS3_gzzANfhIrBd0DAEi58huQ7BQYyiM687bB-fdbVXDDkONkEHAHp-9NYI1uNtE2Y2dFWBrCcI_Lx3rTL160o9wPk4yWgXBV0OG7MzUi_PBvQufC8H_ZSQRr56sm3O1TvqKrxQRb-3W3SjjXTUppCJcC4GrwYt3EQGDoP4BTibpjxNOApCztBEF7Gp0cQx8-EqowPnG-Um1JPvQ_muOU3CtfH',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=7,
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                        'arguments': '',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=8,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                        'arguments': '{"query":"pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":false}',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=9,
                part=ThinkingPart(
                    content='',
                    id='rs_01264dba19113f900068ed20b553248191aac5726e644bee06',
                    signature='gAAAAABo7SC5jcGJ6jMXEXn_eUzCBYYoiNwyUku-6dq3nXHcddbsNLrIzQZ4th-nONpHrIsACXKgm0017UgYttCex_bBeeLi10biUtJBtg7nFPrCTqAxX8u5nRFb5RQy-CXZGkUFk2o0C4Pjwn_KcsTwnI2LtX_rLu8jN2B1JjHDJpfzRi9wOez8E_F3MTwXdAFWQ_ccIcNegA7v_7GcuwnsNdhTFgDsZwPrMloQ98bOWQQhmrk_gHEM8tzQKVJBNn6yzLUn_3b8nkhX6W0cN465J-xCtxWP7_KT5EurorkH2fsHNa9r1MUDEr0HH8Ls4wEAe4mVEAC9H_Bsb6BFTrIt0sM-2gM1w4KSpdSQwoe01bR71G3foU6bRqtIoG1Rdpv5XXMdkS0AR1rNUehSltCnXuHevoelk3EWEDRO9u6yzVa5oZAuvAyWv5xt_sZ-7OZCacnOYuIXyWZ5iGxSYcZ4UYFmIkaguixXRrVnZ-HhyUN_wwFB-6hqVG197Bjx0Jt9mL_7-FkOiahCYXVzQ4nMbBre0gXwA0kbxa_sl_qL6a8lIM90XRoJzTA2KiUZikYZJy8vS4AxVt3wC64qqWqDwFbveM4t2P_ymbOrUN07o_AvU1pbApdwPVQkxl_RCZ_bINVn8c2gXzXFmBSPWxBtKmAQuM6EjeqUenaHmBqNyiEs8Yg5tghy1_ORynFDdaXn4I5iKLCt4gatlnbJMyUKmUVw54zcNr4hQaetIkGnjl_pkQJ6T9G1IkhcLOljN1dfBEYjjTw1DEGLEHbJl4B1-ovhMd2igGHkW10R9AaKPwCW1ZsiGQ6R7jRedm5yIcmSZSHf_KNGafAcls73P9hPJSn6g7CDfmQyjfTC-Xv-IxTxNOGdh0Vh8aIbPhMNTrRqaGrCMoPhad1lmZ8y-OyJ68awMbt8gl3n8TRP-im2g26ZwyTIQTJG9hvEO3O4_DB9lKrZB79POCi3BGA9aKyeX2sOb3PXjurDm4agb0rvpllcVCGILWE2yal4eHW4gb3V0z54YNe4EHyfo6sO3gcUQ-fKSnfzAnD8rEpFzEzQnCZuOSiELn14pQSHllyX_3C0imhkycZI65HxsBRGqxwHdoRH361M4Cv0T5jhAL0eE3lnh3TUGQM35kwI6HKcqfl3SdPBjKcNE4-Il5uuX7jayYoQc4pDbTD07qhPUlc2alHpnxs4hCW_vtjvDeQIK-N6icdR0_hIZtXvHHcTHH7vzYX1d9WYv1vpanKVZI_bZ8wja5r_iupH2UCI6kN0GaeDpX4y2pD3znGXE9zGN0dQzZibBWuGZAYzR8KcWUmEwcDfkrtIREyFsoC4cwxJFGqP5__Ix8hXubDVT_4sAjsxhRyi6TlT8ovfcS7z2-4vt0BE3Y6snLa73eXXnOpAzoZyPn5-XHlxS1cAwE_-_H_HVPQ724LzpE5f45z4YIl7Qs4R3UdG_Qqq_7CS2uYn91YMVFk5GA5iU06DJ0TULOKFDBzOq968U6IxuzEKAVVVUfWqtrLjxGIYsS99gvaVO-UvCCIS05m3IbikU1Opq35V23jIjmj-ABf4Hb8n8xXTEbcjboJMnWNDlVa5EzlNTbHyNkTybcxn9Ms4tqH33CJo8ckkwPJjvJ-FH7yzqd_-07K3tNjiKjKC8QfuoDkDhLao8oUCvOsO17tP97Re2L7-eiayI65zbVJx6VWbLOo3uOj3m-uiSNFz0D9_PobhxtxzCO8uLdkFNgewmlvARca9b1X1cKjSyU4EQlGI70gSBMRVL8ivCmFMWqNshU6dg8JglA663OGBqn06Z6evKJ42i6Wfg_qStbKN6Q_UEb7ui3v50TYtAl85-ClCwSivnQaFp1Gmg3ZzovETulR5B7wKfEO1PkFk9qZu0YBbXw_yYgo9cDsSEg5SQWeOrvxBBevbwyTC3EOSKx-vqdMRDeKs0szu7XtTVWFwl3r4fOrVVFU_m9o7JO0=',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=10,
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                        'arguments': '',
                        'name': 'list_commits',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=11,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                        'arguments': '{"owner":"birbland","repo":"pydantic-ai","author":"","page":1,"perPage":1,"sha":"main"}',
                        'name': 'list_commits',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=12,
                part=ThinkingPart(
                    content='',
                    id='rs_01264dba19113f900068ed20ba6fb4819187a758a692c32ef9',
                    signature='gAAAAABo7SDATu0aE5uxw8bwMrwhIUB1-fWUgek4WHxbR1pj9nvaUA7AvsFOw6pabaCzNqQGptzS1z53ls3i3FNrt9EYD8HYG8r7TmA71ydccAQrxdihl6hV8-jtAkK1btiU6Ur3X2aNR8iBf3hBugsqfevuEpKsGgqw2ASPWY_6d21ojaqdIGVxwDXc1W466Y6i_r66LsJWqeNmm55uoPG7IQqo2JAp5sSNRxAP_QE4x4-9XqN34fWXfJURvYuMDyZYTe2x7KCcKrGLc7X2Wou6DNXQeKWhO-zFQLPdaVbeT2pazb90DBb2jhTkLBmYEJmy88fDjmaHe5c4LFS4mBDaRPTnETPqRBxSdpGwIVUZ3DiskGHEgqK_GxfOE0UBzQonHIbTQYoho-52GVIpmaiPIPDaV9efC_OGnitYZrtVPndyA7XWyvtjftqReTGL5Xx331iswkKVJtwCYBj-nN-BA-Au_xSy_iy5YjD7314WhSA0bR9ZtxSjF50NmEX4CO6lOeXS83ODAYkOdi0MKbshs-AjeXWQmFZucEyNxucIRyWyoRnvxOqtgXQ37o-HpgrD35AJpDcAWXnrh4l-BTC63i8ZwWTnIiDkeXkOmUxjqiaIBLUoC852q9AbLzS_Ly9FEM1IlHG-12KhO3D1oU0S1tx-0nt6lj40qVN00IyrXQKc1UnGSttg1xIwuhUuG8vE83WZrs7DBJzVpZEobzqxBcOWfLDciLPLgxDHfZ7KhOKpoAa8OX76CuphrXrk_kd_7QHmka7H3wJ3vq6XrUAhKguJtmXZp4xJqxcqDrDS9c98dcp-UsnScnlIEOA3jf-j-VnCVk_lp2PH3iLpEIK6Ujr5m5K2b-5frZ0p-Vqc7s_5jRLsgZWEyI-jre-BQLvPt1_GBTqUrkQ9LVPsNjcp-mhT07Rr0FYuN-ix32gGpoTAvZtbLVLdqveSspVtrikuEKDv4tCSRT0WKawDujMcLpjepUKuSXkJfwGI9gXBuMY037L4a_9ucXRjqjj068kwF_TdlLe7utXSdqGJwui6XkJUgI1BWxGI84pWzieXbo0eRhhe_zK_UaB4-Nq1XZIuJZ3Nz2qTfEMhX-9-WDlHl8VxCi6_UcqP26TmVulBlyvz38P3zgHDM-5kJ6DBia_UDi-7asHKLG3GJ-VKQHiJGzOgkkvNny9UY-H0pJpsLuxIVunsrGL5fbqsDfsdQd7lYUZ3nz291JBnd7gmjzBGRe3rwCk_IbTThl11aP4373F0Q5XHAoJcCAaYeEYGoehT6CdRknek3A9b3BwKGdIqIG6wOglNPg8-znat4ynbF13asaW1oWMnje9O6qYLNg8Fl6Bztz9Ufy1Kh01UFcnm_pS8sW9mP6rRQYffhyGlAXPighlr0xRG0QAbdBUC_wLyIL7EWg5Nb5LQ8W_hvzAy76OboCB5rO4Gg6HOIt7KucpJDSP9UecUxLZN-5ir0PU6V7AS57awYUdAbxXkLJmWXcHL3y2Svsf56lHIzrwoPhmMMM59OaTGgCOYazMSWg3zmRdFlUWGdzzx-hrwNMb3q6LYdplA_648UErNWdOO3A90AiSdNCKo9Q4sRXkqPC15fBNaO5d9iLuGHysj0BrMzx7Q0CMtDRcXQBC7CcihKeTMXwsuVqi1OLrzk7d6JjiYu8-s0JBRaDJfyD6MEyKZo49TkfOrLx2OX_ArJRD4xtbjFabRCZVKXKLhSlRP67xPIuhLbNasNsKL0BjZlnFXgcSSrmMABpZY1wuyCPFl84uUwHR1dxnN3IJJsAzWBAicDhvcwhpvK8uGPKPtIRM1G5hFpFCT9dCfCkFPJ1y1h9np9N6-Hf4Q5BZ0oT3i95mIIrsbFcQrPorrJA93oOiziN3elBen36XwEiKpTv3RU2Co-za0PPq5FdMaBf5_wkcheEiP6UsRV8dLtMFYS71l2nLFPAMIyOu9U0YRKcPQgIDIO3kbc8FwDOTX4yxdX8gwSkAsmD6zSADYL_l8IJWMtIOB3QAsE9WwAlwqvqkiuGEHonQDEZTkcH_5PhVmpavFxVkLaAMTbuqabIYFGzf6w-ucfHIrwLRHL9a-WHVMuPosaN9e9gmplDgdouL3Eqte-Br6sgT7NXntO5QZhR1USjGymuwtk4R-g9ueK4ThV7epWrolWfKycq9MqLV_4-ZGpCqqIorz5Bz17GqkAVFII8J_TVeMHJq-XekQFRxuNC61w3FKbT-v8QMHhQceZFALvpTKlyzrTG17X_RNO5y5NmAlRFmH0vmh7duqYHeBjvggPdWAa_XaCDnZmMSKApPVzNojg_63y7rpKLSTqzHIcJExh3NNjmtEV4YdL3qS8OUYuDiv1gxWjV23rsqc102Q6R2jxzwOz0pmuNd2Bmk3JkEbn6bspn38OLcvPzDM7iW9oSOlXFd4zgfbGd8pHJN0u1gi1S0FKFgOeqIAXQyfHDZvk7dwlsYfU8GuxHwD4g9xNxrSSuPlDRsMUWaUtf78AwiZOQi87lalgI0Nq35RUgnvUHveLItLPXfU54qmYezaDHxYbHvtbJ4wgmxzuVTew2pHX-2mKZZ2iFAe4Qoi3dcQSMpwNpEYP9n_Ve_tGw9jwqiQ0-kMr7PdNYo9Hcf42GvbAPgsMU2tXJNPy3oeH_03v962BT892d8fZjr2xEmOkpwkxxQ2vtAZpN_xpX_4_9XMXYiE4mcCLIXGzt1oFEkSGyP1TfOo7SMObDHzxeCUlNWfLrip2eRx_V3JTxaqnX4NvLfUNDazoubnJICvyqnqBevH1_9Wls_h9NrAd5an83e32jCUkY7CWlb5rmxs3KzcrbKgLd1LODuS_VTqxHSchM5Aga7qi2zCVtEdHdmRBqoze5NRYjUm_btgOs7hQ2bNSU4QJf8AwrMNr4ib96e8JePKUo5jmiOxOEHGXmb1JsEU2940XAXFVrgAcKdzDCTOPZoErotW8GQ52aQJPR3TeUUiZ-z0st-ISIpSx8Zu4cjPmJnhbC_TxtJ1FlAk-97Xc5IncABu3wP3SJM28wNZO44qvzZZNxrzX2PGvP6cWjurJZB5o9dgU9DBDFDpBZf2V9KZy2hV4sz2QTmoJQam5J5I3SBmciHvlJLRK0r115W62bhxFG5YJfZ11vPTBc_pKqwnoam0xlV9kRNbfW5LFs8AMCUtbjAZck0ca1o8GW2mcStfqyyxsvmOLcZG6NiLrcu_aAnQ9hIToJ5J9jqvxzpyF9THyMS9qbMYvw2lvFX3lPkH3RzETV7OkCthyFMDicaDTB6kocsdfhKepXxOviDU8LFlUueGA6kLWVLI8S6YrYNT9EU4xFqRdyRv5T8ARZEjiCbu-zoHovo2ysdjuvOKH3XO6kHL58FIGfOq_z95vwtcqfGtC_BFcWRYaSSNFugWeNa59zsqfFxgyTbDeJhldtd2AYPHApJi_eHfvX4HcA6_IQRsTFEgBhnd-Pjyoq17bh7fYj8gQsFJl3htEwZ3ecIoe20cEsPVqi26HJJSkWeYWz25-0fMAh1R_SY_kCjvxiUHx2LsmR7CF11sN1YfCys3AIxIi4cjLoUii_ygvPG9dcm2hXHdNoKmlUSQHZLK-f0z6za1igsKmZxrCBMrvR682el2hk7L9lEyp0JjSqSpA2Do5ErXrDi4VlNdBgyOW5rMrWwkfaSN0P2FJRDP5GEvMMEcUYVtaDQkwEx36L8fOqNaNcgfp3Fnt-OCMxK6C-B6rYdzSSbT_QosaHmf2HJ5dOjjvJroXwJB0XR6q5ShA7eepBWFLmhM6fXZRI-3gMkHt_O5OwuOENTHjzQb9EyZ3ADcFTs4vcl__fFlqWFlJenmj3f2g8ect1HAXzD2-rJD9SITWzkFCuJK9yRLvqIH4rBT08JUZm85a5BPN1kXJDVPdfGQjOmBsfIQz6m-hR8Z_cGMNi5MkZNYkBL1zHRA4s5KO0UoS-kR03_nTwwWzk8N_DymgyR49Pgf_gmWDSSVN9E2GmcAUaNjKASecJzJ0UJmz75WzmK4eTR02BRviebcr7-ykLLpuf-7EDzx',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=13, part=TextPart(content='I', id='msg_01264dba19113f900068ed20c0edd4819191da764a7494bab3')
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' was')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' not')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' able')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' fetch')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' it')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' automatically')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='Git')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='Hub')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' returned')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='404')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='),')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' but')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' you')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' see')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' latest')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' commit')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' hash')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' here')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='https')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='://')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='github')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='.com')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='/b')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='ir')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='bl')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='and')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='/p')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='yd')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='antic')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='ai')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='comm')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='its')),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta='/main')),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                        'arguments': '',
                        'name': 'list_commits',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                        'arguments': '{"owner":"birbland","repo":"pydantic-ai","author":"","page":1,"perPage":1,"sha":""}',
                        'name': 'list_commits',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {
                            'type': 'mcp_tool_execution_error',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': 'failed to list commits: : GET https://api.github.com/repos/birbland/pydantic-ai/commits?page=1&per_page=1: 404 Not Found []',
                                    'annotations': None,
                                    'meta': None,
                                }
                            ],
                        },
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20adcffc81918442f79ec52db795',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                        'arguments': '',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                        'arguments': '{"query":"birbland/pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":true}',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': '{"total_count": 0, "incomplete_results": false, "items": []}',
                        'status': 'completed',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20afc8c08191869efa4466533a77',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                        'arguments': '',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                        'arguments': '{"query":"pydantic-ai","sort":"stars","order":"desc","page":1,"perPage":10,"minimal_output":false}',
                        'name': 'search_repositories',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20b48aa48191b1e033546ad0a24f',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='mcp_call',
                    args={
                        'id': 'mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                        'arguments': '',
                        'name': 'list_commits',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'in_progress',
                        'approval_request_id': None,
                        'error': None,
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_call',
                    content={
                        'id': 'mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                        'arguments': '{"owner":"birbland","repo":"pydantic-ai","author":"","page":1,"perPage":1,"sha":"main"}',
                        'name': 'list_commits',
                        'server_label': 'test_server',
                        'type': 'mcp_call',
                        'output': None,
                        'status': 'failed',
                        'approval_request_id': None,
                        'error': {'type': 'mcp_protocol_error', 'code': 32600, 'message': 'Session terminated'},
                    },
                    tool_call_id='mcp_01264dba19113f900068ed20b913888191a4cb41e5a5870789',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )
