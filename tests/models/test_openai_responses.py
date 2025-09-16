import json
from dataclasses import replace
from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool, WebSearchTool
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry
from pydantic_ai.messages import (
    BinaryContent,
    BuiltinToolCallEvent,
    BuiltinToolCallPart,
    BuiltinToolResultEvent,
    BuiltinToolReturnPart,
    DocumentUrl,
    FinalResultEvent,
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
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, RunUsage

from ..conftest import IsDatetime, IsStr, TestEnv, try_import
from .mock_openai import MockOpenAIResponses, response_message

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
                    ),
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name":"London"}',
                        tool_call_id=IsStr(),
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
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id=IsStr())],
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
                        tool_call_id='call_FLm3B1f8QAan0KpbUXhNY8bA|fc_681134d47cf48191b3f62e4d28b6c3820fe7a5a4e2123dc3',
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
                        tool_name='web_search_call',
                        args={'query': 'top world news September 12, 2025 Reuters', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aaab56508195a1effa9583720d20',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_0e3d55e9502941380068c4aaab56508195a1effa9583720d20',
                        metadata={'status': 'completed'},
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
                        tool_name='web_search_call',
                        args={'query': 'Nepal protests September 12 2025 Reuters', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aab0c534819593df0190332e7aa3',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_0e3d55e9502941380068c4aab0c534819593df0190332e7aa3',
                        metadata={'status': 'completed'},
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
                        tool_name='web_search_call',
                        args={
                            'query': 'UN Security Council condemns attack in Doha September 12 2025 Reuters',
                            'type': 'search',
                        },
                        tool_call_id='ws_0e3d55e9502941380068c4aab597f48195ac7021b00e057308',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_0e3d55e9502941380068c4aab597f48195ac7021b00e057308',
                        metadata={'status': 'completed'},
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
                        tool_name='web_search_call',
                        args={
                            'query': 'Israel airstrikes Yemen Sanaa September 10 2025 Reuters death toll',
                            'type': 'search',
                        },
                        tool_call_id='ws_0e3d55e9502941380068c4aabe83a88195b7a5ec62ec10a26e',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_0e3d55e9502941380068c4aabe83a88195b7a5ec62ec10a26e',
                        metadata={'status': 'completed'},
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
                        tool_name='web_search_call',
                        args={'query': 'typhoon September 12 2025', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac378308195aca61a302c5ebae6',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_0e3d55e9502941380068c4aac378308195aca61a302c5ebae6',
                        metadata={'status': 'completed'},
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
                        tool_name='web_search_call',
                        args={'query': 'Nepal protests September 12 2025 BBC', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac9b92081958054d2ec8fabe63f',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_0e3d55e9502941380068c4aac9b92081958054d2ec8fabe63f',
                        metadata={'status': 'completed'},
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
                        id='rs_046f1027167eded50068c4aae5ae68819799b37d0e9c90ff1b',
                        signature='gAAAAABoxKrsyPp96SF6Q6K5D2oOYPgsu3MrivUnFDtvuyVCFckLaeAPVhndpbwr4mWRVCX_3yRWlbEHpeSb6eOjleDyJGyDj6AdRo_XBWCXVfuuI91fLrTQ9U1A_upFzbwVC0uYWgErr6-LoM1kAR305dlAHEtgN4FnYgIafhE82FFuMY7D5-h1kzXoIoUWayo1WtEVoWhDObeGqNZBYdypNt2zqQZCYLLBtKkJrJtxA3txXYovDxMrgJ8obg3JIBi3m_YOlJoQHehkpDjo8hR2C2bhsJJ1-ZRKMQp3f6leTG4eNc_-tPCe6IKNw_uAZVvMdbLkEHY4o6wGuRtSc9OJet8Pfamnalb4DKGT_f6Uw4dk2-wqBM8WK5EbU-tPYCo6i5S8ou44o7QDPWECbCqRchnIxr1eE6jSk0xRYVMON4dlEPQ8N6nMuH2776T91zgEP_kEdDW_KMnuB52K_SaJ7qtvrnvf5UUd3MzpTgiIFRF3pmGCGku-_qXFwtHLH81zIjT80jm8JK8hCmBpG-Cwn2mVt5MgJt3HUoATUhEaaAy6fCFPn9n_xC9yODl5NgzFi1e4wbxbG-mh6FhjmBXckBzJAaJAHCGqZJSlKHKozsO4wJWLXtVWqJD12Tp1AcagPIZqPjzBNjtLUN_aJBBb5A-ZPcnZVa0ij7-HlS-ngJd33cR965lDhAMHoen-QBuYmVIPXU31XYtMo-dfu1s7B0f1XFi7m232l10MKk0yhp8Ou7hYwbgkBAtZwREFMelQvQQpfwSPdjin91v6UNSjJt4hdBfQpR6VzWOcKCwzxlYws9rvtCydvwqOfynUO1Q_qkyY5HExXRjlzFPdiQiRFcvo7uWyRYRdoV8NBTg-nTnjYOcHWTJ0ol6Ibv7q8tIYVQFHwQ00kJ9B2JZmA6lYO-1CYawGOLUmSaW5WCTSeW8C5WBdijtIqUKRt6BNEZDjsoACBZln7ZkDDSXRpnmsoba3Tyq4zf2P_8ldzZzeztWEA02xkJIPbz6Jd4AdL_Kf2njsDcpfrJR9jMPtiZulIgTSwE_jpn4ufENQcmqNC7MknESbEHg8v6L8FtcBOLRldztNQgOsbNmv21a86OAz-NiQ6O9Yxtq2ZkfYz0VEKzajM5RJqxwByTCTY0WLsh_0jF1reGaTuGXAo_05z4r2ush-_t8_CB70CU5UpCk2BWBtcgKT1XeYpHJVPrN1pryd9sOYvxUu70R9DwRSkifk2J9uhuBAi-g-9Igtlerro3fOVwnP0Gzd0NVve48ikBu2g8mO-p1mUzoTzmZ8cM-XgAtx1JSRBU2D75sGHFgYH-lpYYqg2jcmWe7EE5YT-H6LoZEcvDxVfai_KGht9vZIzZMBKmRKwIPJdh0qSoW_FbhsdbZBKOtYRQYru3KqRugInpq3U08Qf97Rk67l0W4bDoYX-HkpQm0uWduoShLWU__1UzTN1ekiO41oAJhkL4C-XygsIfs-izdrDJb7YmpMofx5vkLmGMK9LC2xa9_FEVowOOyv6gtQflEx8fbcq2GnN8fh-NBU6wNbpmuNmuXm8ceqt8V-Vl01Xh73IBxioN8khdKBOfTKRlc_G7DtE-3UNZLIk6SAV3AdptAnil_iS7RD13U4l-BSs95d0bNJvh6x1ZN0ipVsK5_1hrdNIkbHrPoJzyQwUsVZX02WUKUC3oH6uuSBlEvYuSkQG34KO6EARaFJap_V_LYkfIlsU1fJQ6wvBnMGICWoNrFQmPeAoIcKpsncPKNOKKJ4grzHRU5Y5yI9jCsJ4DzZH1-RNlgjfrmeVFv-u7-jEYwZTbRtEsv2hnZcCGA93jhAg2ZXorU32WB2ScYUhQ9-NMZmSYMcZZB5FYbcpRPBoW__OG7pzvUBUp2xWw_4N9cg9JrNIY2sElYXmxdWf5jaNMNXVKH3gS9_8rJk7Faa8nCncevs1scRrE-o4y6Cb3vt836yTW5ZJSY8xOkqmDqQ7z698nOm9XgTdnMygXkuKZubCxTH1Ols_Vg3MgCh3Hitpqp7KsJomcplMZco6jvtRdKXfM6YllDLJDc68n7PpuIVmXS2vJoRSJYz2RPe6OGN_n64vDZQduQhoqSjSg5WOM92EDy-r1o-kjxZcFUxapdV8Sq271O5NvRXFQ==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'weather: San Francisco, CA', 'type': 'search'},
                        tool_call_id='ws_046f1027167eded50068c4aae7e5a4819793b2c41e11f3e363',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_046f1027167eded50068c4aae7e5a4819793b2c41e11f3e363',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_046f1027167eded50068c4aae853c48197beea48d1b360e686',
                        signature='gAAAAABoxKrsWnQiXGJpgHQ0WEPJM9xPEkY4wvYGcNlGvE2A694inFAaEQPbLSHbf_6LZLbkGTYp5ow-GTjqWGgyZUpFp0WPbFNjAXZZseHel5tFPpPWR_UaxwqoxZv5zqoZXLACOarrBf7tqtbB7rRMtIhxUR4as6M1YF0RjWO6AYLtnAjdyYSBJeEo40NlHXaWNLucuRZYUDX6HFNwYR9HUIYd83sJMiCFV2_ECY8o_VSnX83Wjnhn3VRuQesZwXOIx1zjwlbY3HlmLu2bYru2I0_-UVmIyCFb9mEaB23vTBKj1rtvKe7PDgY4_9zvWKV9W2GMGO2yfoe0Zs8oVdUNtNyGHm-M3ikBkffLnlZ0OIVI8Xo5ibx5z5Bt8zB_Ho20yvvAMOZJ-abREy7s_XdeJfWqJxsdYlCRkrsILD3GuYpch03zL1ryWjLEDdZYWP80fM6bb3VYrSY5kJan5yDXllZGsog-vE36vGg21XfrGskoC8ROto2-4hbvM-pqdD_k-XAEX2hudnZpaCAHRmi6WGyCqvha22sd-Z6asXtYZfWw7bibmkr187xKGCTc62mq-Bz8i1NLiF10v00h69bHs-nH9oGJ6orP8CvpFLNVL_heNZdfF1qCeP5AKvbURBMUM_texbZ_YH5WkyL0Y6LnZCvPJmGaKfcpbDXf0G1t08SXjSDlrgcAxGRSLw0apdnpbXuwD21tRngcMgFGUZ4frDb8fK9tTcEd6LP4-tPDgdhlMM2wNoYDISFOVxw_ycO0I9Uf6JG98c3vosqjW0B9e_VmS_qBBA9tbhE2F6RygZ3W8fJ9UXb_Z6eEE6vefNSRa-Aum1VPRqbq3kMmiap7AdaKwwih5NcBJVAGk4_z7Ady8qFQFls1YL2feNtZEwY7cd_mOyn8c5ahcvMYhP7PLlA_51boVFhJfDotJk9-f0sqd90Pryq6mP6wkRm4kmuuO7dO25B6RT3D90O9MvnIW-o6twzQNIEsrwX9wLauk4WGxzCWmtea44u0DWJ2fkIL1bBB2PttqOdwv3IyDQWhnl5OLfIAMlTXtlg_RYkarQHJPePpG3TGrJNRnE8thoaYsxiqfOnXSGz7GnfVTxDpKodl14Po-MS5hQ8STcmDe4XDRAG9pZI9sjRLHyk45RxerlPCxJrg7BUyNt9v6Cme_EKSXVRqos-Sl07p_VRTEn-Uuf_Rjle-iSqaPrfkgtXOo4jIblIz86Yg3TxSPZSX4KvKuQA-NeF1Ja9GGMSaRzUfTHwtthEUlAau7Zk2gJWPKFTlFGjaTxRxloUdZthCuCdObcKSOC-VfiLqZE3lsAFFxY3kLOEpl_OFSUgo4G9o4qkZ8Gv9scZwaei9FF8dwm3bvoan9glRlBcZbgy6Xpfn2OffRYQgCc4ycx1BbliZY71SKRm3psBp67LULBLIeaBExSEwFpJ0A_AXGd-lNntekOOS1QaJ4svpEDbIQPUZo03quVM-vpTmKzQXZBz7quykxnKOXMA8Kze5ot8InGkxkie5KuhkviDs1jvr00JELzd3vi7BVgRzWARZmVUhgTEVQMGLJKnvUpayN4R0SjZ7NSu9Kig_8zB7XLvDbADV4jRFUzRa1pEGToh8A8SWFudOo4M2lL0z-Vu_GMwScXsbBPrNoDSroaYRpZ-et-3KXoSgE1Pk0-71EFF7_YuINcWWlfoWyTmdXMBoTGKa0T8oxn8dtLuJiFB18oN5T8vgaKqOVnx0SWXweoqnpQXVop061uy0U-TmkomEgwbnbkmBew5OedbkSS1ngHktZvUx8F6nIEbdAmdhuL6DoqkyqanMnIM32og4TgECmgOBogmxulebfg60imk5pX5S8q04WkBm-pnHGF90L2yKnZJ5MFrR_w9K3-aWnwatGH1w_0-gPl1ZUdBXxahNMyeN5eiTulKgR7NYo9SYbFkGfYhyk5iSWXXGIjaOtw2ByCi3D8gp_NSQd991n3ZaxHqokQmURCTsCqwR09o2l8jIOS7n9DEiC-hVrZ0MISEUKC2wYLngiOHS2DLgyGtARH-sMmkiyJ7KPjmGWZuDLlB5009rTiPv_wEdcCu8sE50oKq0M_x9Y82zFzFILqY3oGMR6JbvFT141Zp9jo1GyCmoT_8FybPh1ofBq4wrTHfe3jBG8XT4ttuATaQ2fT3-LdkLVmuUlHYdYpK2FyklISfSsN3zDAUlr7XLuMDx1sqHR927nch_QUK6EsHX8048Ay7VK7FsVAddyuTN5rh6yQQt5Pi2iMggPPlNxckggh8C_xxZxM-c7SLb_nXUtN2dqJaO8khApUhjZZCK7_J1i0Lg51nBSn2vY8yRqGQ_q33WObENL5NEqi6ASMQtYI6ABU_PYh7j3pm_OS2L1HEJN0gQfBp67yKrGurwESyuLpWiYCUkI4LaHOoI-7HYrJG4qkGHlKr3D_QFF1SB',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='San Francisco weather today (Friday, September 12, 2025): Mostly sunny. Currently around 68°F (20°C); expect a high near 68°F and a low around 59°F, dry conditions. ',
                        id='msg_046f1027167eded50068c4aaeb219481979d742f413b2501c1',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9396, cache_read_tokens=8448, output_tokens=513, details={'reasoning_tokens': 448}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_046f1027167eded50068c4aae53ec48197b3bce501067a0364',
                finish_reason='stop',
            ),
        ]
    )
    assert result.output == snapshot(
        'San Francisco weather today (Friday, September 12, 2025): Mostly sunny. Currently around 68°F (20°C); expect a high near 68°F and a low around 59°F, dry conditions. '
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
                        id='rs_046f1027167eded50068c4aaed09488197b2ed0f034f6af738',
                        signature='gAAAAABoxKrz75j-uTcKB4mIzyIDC8-SSbgOWJCb5PB6g0I9M64BtMXUgu5jTEU0iccQNSkNXajVOVzg_oAURVLWqLSqLX-OTxtIVZUH0A5sdLMdlHtCWT16JBuwj97PFeA9gybIbqY7E5ySR14Ov-SYkUCbjuDM7yAZUu_2fBNBlokNR7lHk1WkZoGVnoJ7qEI5c577KH1x72ysamQq-FqLA5MGXP1ea3wBGHy7RWXzDBmAfIQU6oRDHudkk3EK-oKjpSFZ802phEBojGS0p_88E8KZwur5PCsNTN2rQFTtectX9grIqFhN_FVffSaBQ58K_5WJXS4A-tZ0dmT3aEL-oBSeFqpOGLvhtmueip8h-eeBxAUfTorq64QUlEwUa8wEMzrwNqgVEg_s7FsMRzcYN9VS1UGK9AHSQbeW1XNn-plEidmXOkYc3yIGpWvTP32jq6sSJlUo5MAnICC1VlCcMmX7mqhsjt1RqyyjcvEjPGHhUZOhQJOMWieAaNifIU5INVf7x1Vb0ir1lJGnS_m4oz3YpbcCsyMtUu_dKmYKUGLnZ29oKpbHHehA4QIQDh6IBTixmapOFBZkHGl7IMaMNhQyfzArxOhPYoKIrrqYWWGPq0UouiRcBMw5cFwSCetwpqNKZnl7HdGBU-TEB5b7eFVaVQzQDe1tDT3XiqS3enNFARFcgKqpQGUdmG8PbcwjP0f2DE8Ul5zMpX60aca__5ngdOD6U6T5dQh1BNgO_CMVLQP2ZrWjKixBqYjW_8W4osQXJ93is4GJhpmue44jVmImWgNPdqwClzCVzcRKh17-6SP7UCfe09Ct_Sz_5J62gow91xN9-iGGmFEI4iAPh3rGbDJHTxCPlNHJa8LA6pMvlrccHotGvzkaW31oty-je6VwWI_OYiV0jRmp8ZyAde7E6kskdSwgf1JYduDyYkWavhMybu3_VmZXrqtd7O1_7UdkjPqalJfAdlMK9TYSgIs5MpyiczKTccNfVCWkDo04Dvbprff3-Ff2YISqs-1nTquk0rLwss_-FrkjkCK6JVpwURgzJ2BZRpeHJKdeh42eHU7kUcbryE4ThnJl3Sk4db2yWZGKCYND7GfiNzG-Tz6av6M9BLGra2jJ2qbshyfO7QBQe71PeUQAh0JPiCFxU-aWsMs-Lke7jOGrW88mZYZN2ud79i6ya2hlc1B63ZlLeEB4GF_IaNU9AWAUCJQAN9mBob9TdzkAkojHOuD0T7z6AxM9-8Vn4LV3t16pCNgh7QCfm2EfdZKiXaZMEmJQUVeW897tznumBZViBc1XIR4VfDXDYGIOIhJHneSDvn1-E2H9iiRWrVcoMyWDJQVI14Vu8ho-J_jFrweb3eRFityUEwbAw8hTVQh5dsUYeSGdpLAno_oTlSwKjwwzwL2D2rcXb3XWZYS354A0fLntyLOW0Eb8hjSyq31YLOABaRB8ueE3kRmIqiDli-MWH0XkiBm6XtoYWJ3Lb_XQqL5e4wBzJ4BNDEULSXxsAm02_ujYuESyiyBcCe-Bfhb117F1ECHqC6DDyONhNkxzx9YQazBfonH2b8qkp3Va6jxfvrYH37cyGNNX7OEDS-gdft7sC8nBCq7TWSQpUgeDjDbrV-6mr7b8PPMMUHTRWTzPrhB9_1o4SZOkAohgxAxUjRjlnzK929-_50CWvdeih-vbpQa2AHABwlW9auEduthbBOyB7VdDXQvWmpfXSTj56a305NNK1alawlF7Gs8c72bTmqNVNIlCeYX8OcmWfhGUx2zfiHuL0I12gnFFwyR8P6SWnbF_yjCB2hv5rIXv47K4HZ3FysHuQShmfO4GPIFRXzXsjUSN9ZlB0JIMZjO-tA7Ax6dE5aTyaEHfqK9SyunNV9u0esNiaiKKE6f-zXnpYAhMtOTLLKFI4ITTuoCWOOlRJIs6-DUZ3d0fBpaiQAtI8McDEqTOcPmfjQF00MV-6AAQKIvnSr4-oJmDRUVz_X8IIZexc22vXxKRGQfNrpMiErx2ttvvbCyHot1WlP76SysAEntIc4KMt03kQ9ioZo-0_JYuSUA3GRQP0zDTUaVnwMThpqEYaF0hkxM02A62F6clj1VCZNNwAAPpzj3CIK8Rq8md3wGKH_20BoUVH1opg0RRYRKAuW5pmYlQ0BGdtUTAEAu2wOFTOFZ6RYfGJL97WtDLyYnuxKS05-hwjWruJqXckVlMS8OxiX9hauonLRAFntQOTSYiIFAXyo3mFLu5RfEtegwjAVEtHddKIcf-L_mWWpfbRuD3bsGtuJsho3JyRBYDcVJxGNx0-qoL30xoTcf-R9DjzSr7a7COBXofgkrIo4aI7_018Q9kL0ug3gvq4daXoZkjk0itREzf1VruHJ8cFMnZu0ohbeo4iOlBZ7ex-dGIz991XVxMLX9T_xVWLU4dCkDvP4NL8iLbUCXH29A_4RFTLU28gU-QgLOOSVWOmHrUOKFobZlbV8Ws1k2grjXJRUzFvkz-TsYLg2u1liuHHP3Sm2NXk5bFsmGpDXjwjdmkykWP3wb7rt5okgeSKhvmym70CoP99K3AyB1hhBvSlTJ1QYJKXWbVxEub6QOeXZJFH7JJxG8NM0ewXg7p9aSc6IOfS_3VnxPYcQtnBrS_DuIZtX5OlOL343_38wfXqej-fu0PnJK8RAruXPK20UIO8x5cf89XZg00ObLsrcX-nacQ6rGoxy3n7ZvkoNoiTWTAlF8fSsTAHOUMaYnTrY1vi8K9q5FzOXqHqd5gJQXQCT8RI8VazWSTuwP72wEoSnLnJLsH76FqZnkcpBLCw1d7etZaDvFSFq3_8mW-NtxlOgw8hZiVPddUpHNvEZjl_xxWXifRqWEQVm4TVeBhm75ouF9M4tQ1uYjUk8Tp8YHTR2bWIgf5nJ5iSS8fHIKCx_jADNwmpRND33FEQOcBENTF0db8boc_mgqcJFDnBmzrDQWx',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'weather: Mexico City, Mexico', 'type': 'search'},
                        tool_call_id='ws_046f1027167eded50068c4aaeff1b881979d374e51fdcd3dd8',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_046f1027167eded50068c4aaeff1b881979d374e51fdcd3dd8',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_046f1027167eded50068c4aaf0a2f88197bc3beb4256fa439b',
                        signature='gAAAAABoxKrz7EpugbbTBtlfNWmGteArdEei0DCIUk0cb65ryJw72GynQwAQsU-NlW02lVkY1qeB0cOmRguCUkgkGwF3GsDuv_jR7SCNZqXtwaScbsKsqzQuOEo2Q0AcuIoYOsYnZm51NE9fbZHq20bj1NFcUq4WT6R85YnG0oD9frFSTQwLs2BjlUjrQu-0Sfq8FyQKfQ6SwoC8ww7qEkAbN2Ajh-ljH1UY_hvIzwzmx_iwUJNnJrHOsark75vyxyu9Ncr_h-HBvQXw_D-6c941UW-5T4eAx6W_tnJRVQrZ8JSHGT7MsbMUUdWIdzJrGGOqj5yfu-2mRQTC4zzWW-lwxVhS-rW7p790jCk9GiZKw6N7OfLfbVbNBhe0griOqJTUJvsCs7gP9WtSW7uo-6E0lUQ_ry5zbs4ZRyrWjMmAyXSlTvL-BvFsYaTdAcxLct2L9e3G10d78MpOWkAjzN5knEQP_FTa5EXyxq65UchrnfnFio8SrjZQ7axTQrra-0wYGLuy4PvD-V8jVW_vxN9VUuVzA5x7IPaQTie79Vb7mbhxpkgSe6m6oVHK64QtxG28M36B8r4gLF3YG8I7bw2V7oHsYr1ygoY_CcfQWsgFjMwppUuDS2XN6sXMuZzGBGdyf74Wsl8aLSnMiHb8sfK4cp0IEteIDHPmjun21H_933rPFDHPy4YvKXiEOfpkv61pkMss4TUu1_wcZh4QMY8hNp0_H79muXDPCOB7a7PdUb-bGeEDOGlo-fbkSr3pVVuERV3UUfWbiPjGVycTyts6Rz5rBNkcLoc2qSLg95gYV9ET8EKVFtWi51L1gfFOLnPpeUdzMeTKmFPKG6JaBCGWwZw2XELKxmPqs9lPFuJPYlLnpaWWIBTU8h9zC37xup-nksbUB819Z5C8yE68bygLaX4gbeNO-7HOxxRCfaRuusUuOvN0VZNb03RgFf-BVExuSeuhaH9ZOLQUwypok04Dvugtn2rm1R-6HnDxd3pAeWqTx8FOMfRGLzEXfJBoEHpKAhtGY5qkDQfUKKQiGfOnU7uWl0q-ocuK3kVctMAAD6BZwIT5039KszrMiLuK_uHMbbnACmx4fz5ALtiyvI3kqDbfQYaCw_b3KFJjSJ0vozFAhIj5OIh9EBoVqZRRrJ6cUcDtR_95mqzDnhavL58_WkZWgJ0vzmwiaDq0xPV7rb1xlPhoX9lRrhlX8UlATwC5nkw4AydTO7GG62Gag-o_CCIVN4NaMGPS-cMNanSJstfu8e4eeMSW050UREFuZYoGVsRo21zHiF5GgXkiaVwJb0OOKODp1_pepJ0xW-YkIxGcwdaXK4J61PWbtaPTpXl5DqfU8ewo-i1NIocLbu8EyoXxMiFXKdcPeoW-9mj7SoqU-lrEllMv5VQPgpoS1pPuhmVDzTYStVhGj_o64O9ukOrtJ-ffKgrPbB0650ebnAuW1LTRXvNUodyNqwUOnVvkIpwgtLcsZaJzMuDJVQF2CuWsTodcZvvnSUftodqNpQWtMJfP7AgW495OzXRzRT1yPlf7Y74CrT_SpbhrehjvrMvS6PQqYAckIhH2ihWVlzqhyKjishZQZDVKpAk9HRgA13_gfUTl_SjTlHUtDwtjQHPnCrP1-ZKCHD3Hqt4uGxq5ha0F2nYjKlKebEgM_wyx_ijcCYY79Jvr_W5wp1X0W-D4moCIwi2QmD4x9M2MaAXDtnqnJLw51XR8vP9qXyMMciKkzgJZe7xXXqWh4OWbdxxm-i9fIP8UhN-mT70dygtkMIKFN06VKN5uVXvrTx8cD1mps3tr-s1jGHl6j7yNTe7HLznTbsDfDPDmc9dsgzx1FPcId5o=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='Mexico City weather today (Friday, September 12, 2025): Periods of rain, mostly cloudy. Currently about 65°F (18°C); expect a high near 71°F (22°C) and a low around 57°F (14°C). ',
                        id='msg_046f1027167eded50068c4aaf287608197925592588b70b714',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9726, cache_read_tokens=8448, output_tokens=522, details={'reasoning_tokens': 448}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_046f1027167eded50068c4aaeca98c8197ba1dea5869971473',
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
                        tool_name='web_search_call',
                        args={'query': 'weather: Utrecht, Netherlands', 'type': 'search'},
                        tool_call_id='ws_0b385a0fdc82fd920068c4aaf7037081978e951ac15bf07978',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_0b385a0fdc82fd920068c4aaf7037081978e951ac15bf07978',
                        metadata={'status': 'completed'},
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
                        tool_name='web_search_call',
                        args={'query': 'weather: Brazil, Bahia, Salvador', 'type': 'search'},
                        tool_call_id='ws_0b4f29854724a3120068c4ab0f070c8191b903ff534320cb64',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=None,
                        tool_call_id='ws_0b4f29854724a3120068c4ab0f070c8191b903ff534320cb64',
                        metadata={'status': 'completed'},
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
    async with agent.iter(user_prompt='Give me the top 3 news in the world today.') as agent_run:
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
                        content='Give me the top 3 news in the world today.',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab274e0c81959ad36da9d42a937c',
                        signature='gAAAAABoxKspT3bObZIX1c4Rj5n8fotSQUhl8qQHqqs9aoy3JEUbbOKD3QVYemAzZ0uNqxUy6bncn7pX_VOYoqT7dosumGmkYMcmZcy7baqB9eNVpoqvnzJcY-FzvhYm_UJCcFnAyYELQD1trdlAUc_bFnB40zCWiGTIVliQLjimjx2ch_QhwExcFQlNklWPfZA509vbMQjrUifGWQqm5YW6hN00gGV6wbro3lLMMAT0vG8Mf86Wcu1ykL_DvIYCdv5gnCjHdparV4kNxaC1feFmQ0dVNyymD2eOLbCzYdeEFvZGPM3vUqkgwggTwDqJoBy-6Md8sQGfzEI2-7OMQiR3qaYiqMVKp_zL1TbIgM2f3yEEe7KYIFvaKwqX-uR0NmKoZZAPFi5xlc4dbqZyFfFWtThy4kqtfitEa-OkM343Qr4j3cvsmmNNOFlt2nYZHgZ7Mtn0VWeCvorYpJev3JB98caMA-_MLlgeqbbUCSiitBFdyfoVx2XAIJmr2hecTqiiEn8ITCXuguwgxQFaeUokwzJ4nOGLs2xVCqSTwvSj0hzIHPNrOwdFem3-wWleA_-VOuR3-GL3hIiHBmGxqO7vqL-WxrcGSfHzfqvUv0zTeCsidIxqgT2IRezrB-BVWDnuTbZ9CKa0Lz-Uj7xF0jExBvBqqrvu39zwtGL3yTbAbzwcw0ta-EXI7_scg2ey-t9L67qEqHbjzmbGNahreBexo_EI2axTz71aDiPoFXWlAdKG8SuHJW0w1w1yWHpkTaFSQsxySsfJCNJ05tiEG7DYnS4JkthOeXfkye2NDdKqa9heVPU-NenvmZOv8neiM4Cila8xhuTLjBYNnNsR3pYJqxsa7Wuqxf5yl7meiwU5_qi5kGmuKxU8WEiq3xTjlQ11HW_1wWuaTPxj9uj6vstMWKRFYXq-GAduAL4N55xHuD1pcKLdYkbSpnvx5wa8GcC1SvPHQd7dOlxdMaWDw_AkeTvcyfHpamTS_fPvMiO_PZB9v2NE4ubHrbUmhK-G4BqM1D0uo-8zwC6wJiUX5xun6h1D6ZuPN6TBGbus5DoxAbpNp5q7w_b-7O3QEtOXUpfW4qCKyegbNWddPCpOpRKqjc_ua1lnNyccYM-l22rkydPsnkOhIItPGySZEU5XoSmBmJ0bZ70sTUUfSatU36LukZNvv_eVOSqrfKleKEYCwEqBNJPPK9XGMIi8RHbVnnBPyanCFvQKi3b978BE06TNyefA9nWZ2GKKjn7zVlUHTUffCpch7RGOZI9q4BTq5S02aoYA3h1y6nDBjU23W7S_xiaRhS5F7YdB1EZKh7xzcow0cMsrQvWbre1FqqbRdZ5HvmgO9l-AV_vAYWq42mMyJRzxf8sYLOLMM9WMlsCDFEghMo4f_82EjrxJ0VrwDV-BOmruYAXmEAEcOmIjuuzT2oxaNiX74z9BA4DaO-kZB0njB21Q2YwhdLtpwKQAXMIITVsZrXPPKinhE8cINiIYjD_1BAVnNwsYvsbq7hViNaB-ns35i-Or-jwS0MqqX1SgEX8BTAM3Wz_h-pHJr7o0f9xv4ZjwGuzUwQEUyzvrOp08kIcSnByp0gP-t1IGHrVDN5iv1cuFqmhp4AaAyuWt1pxcX7PGZ4m_CjkXxwKR8lNef1AtSRBXFg7OK1Q1VvJHlrsWPW2BzwKkcMWXN-xCywGl7Nn8Jobff40wQCqUkp6HUp_U_CP1ND2Qm1TWcK1zb4R6oRJmAW5qHOHhGd-S3aAw4OmIOD_zoUTxVkVkVVlra23IdZNkl_Yx__stOsBj1wGmQ4V2',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'top world news September 12, 2025', 'type': 'search'},
                        tool_call_id='ws_08313d91a201f39c0068c4ab29923c81959ce13065d52636be',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://www.reuters.com/world/china/global-markets-wrapup-2pix-2025-09-12/',
                            'https://www.reuters.com/sports/records-races-rivalries-2025-09-12/',
                            'https://www.investopedia.com/5-things-to-know-before-the-stock-market-opens-september-12-2025-11808712',
                            'https://en.wikipedia.org/wiki/Worlds_Collide%3A_Las_Vegas',
                            'https://en.wikipedia.org/wiki/2025_storming_of_the_Nepalese_Parliament',
                            'https://en.wikipedia.org/wiki/2025_in_Palestine',
                            'https://havanatimes.org/news/international-news-briefs-for-friday-september-12-2025/',
                            'https://en.wikipedia.org/wiki/2025_Nepalese_Gen_Z_protests',
                            'https://en.wikipedia.org/wiki/2025_World_Wrestling_Championships_%E2%80%93_Men%27s_freestyle_125_kg',
                            'https://en.wikipedia.org/wiki/2025_in_Japan',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab29923c81959ce13065d52636be',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab2c23c08195ab957ef38a2d96bd',
                        signature='gAAAAABoxKstZbcUiAU78M7zEi0_gp_EpbxebSkIgILXimN3z4WjGC9OSxdsumD39NdGZtLBKr486UgvfVqRgC6xgx8xCkuv1LItk0b1_1tiDw0-7U2dx-FvPb8AnQyuYleq8uQI_IrcrmsD1GBmmCNC6ypBAGTW9Ipfpv7XCOuKRj5w9kArdqwk2wLF8CiliOSf9ulLB1qj4eJC7NeR4iFbENaJgbjbe-H39EPyQgthn3UiXtt_Q_44m_i1C0KTq0phb3JCwxHFf9g_tNAVuCW52dEZ6mRlOlk6bYrzQ5EuF06Je37vNIYOsdb66mgrcNop3Kez-Yv0c8BdjElRarNTcOxa11AmGa4UyHB4NmZK9c4c3Nv_Mlh-TnQvikoSj0WHD3vz3sOv-2QVMTfazGvgvQHFCxnIj7zzf-QIlZBmr1BwRV5ZOTd4cjU3a9N4B2HZ9jeyy0CftUKAOzn4m3s5u4e9LqYTyzss_i7zkRQ0PXnVuqplkrV8s-C4MvDi69MocioWaVUz7ptG2R1zePXpkMsfXMCNN5F4m9zmUJSHvFJON601TVjipV9VL-5i783vQv1NmCs9D9HtqphK68AT-HpXqA2AfmLlCWenz2jFsuOCSG7rzhDwoPOF5rl4IqqeG3CcVIam2wPT1bA9_R6s-QhFLot4Q8RExR485-pUzmTJiHypkrVO0wQbUL_bh7ld_Es4cSZVYmkIkpAqmCKbiaKjJagthph6XsneBJwiQ53YGflSO5-SXu_EAikDkTcDvEfY8o5KGQalc8_XyU4HJmXm9WhI0euD2wnVipr4CgMFhbCVeHp0RG-aVOHYlbCXZuUPN1rtb1Z7gVOdmU824wiJKjnEPXPR5Fe9gwSN2dh3QKdDtI6luDOk8_kaCDpMIUoAWWGdilGLDQ96KIKeY6GFO31f_ko4whUDxjZTteTaz3S7R3u_QwEhoQ0a8ZTdU07drfIqlAjPcU7plZmMarf7kj7DPgbq1y8XhIPZINmF2e3hiIPkdFDM_RUQurpBcG3yLG6S2Uem-I4-jK0JZoZEI1wPmDYFpqfRQSXLBGU0pbq34wgIB2yjqmg_QmmaAs0FJtZUrK0uV4OBF3ATkN2wuxt_0xW5yqCiHXzLUgZrQQREPeghEZGsx5k4Tv4RTH2sQ5w0dUqkbQVehsHv2Rs_-bcw7s6TN8FT8kidQhCHEiszM5_Sx5egh25dxcC1s2Oj0HuQI8fb6xKY75vE4VxMnMHCNDRn5LPZN5n9Fi-pSDwwsnpv2S9OHXmzsVahNV9i0umg9RGgSjZE4UMjl45iQIcz7A==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'Reuters world news September 12, 2025', 'type': 'search'},
                        tool_call_id='ws_08313d91a201f39c0068c4ab2d62b88195942f89b36436a6ad',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://www.reuters.com/world/china/global-markets-wrapup-2pix-2025-09-12/',
                            IsStr(),
                            'https://www.bbc.co.uk/schedules/p00fzl9k/2025/03/12',
                            'https://en.wikipedia.org/wiki/2025_Arab%E2%80%93Islamic_extraordinary_summit',
                            'https://www.reuters.com/business/aerospace-defense/global-markets-september-risk-graphic-pix-2025-09-01/',
                            'https://www.sbs.com.au/ondemand/news-series/bbc-news-at-six/season-0/bbc-news-at-six-s0-ep7/2446206531714',
                            'https://www.sbs.com.au/ondemand/news-series/bbc-news-at-ten/2023/bbc-news-at-ten-120925/2446206531715',
                            'https://en.wikipedia.org/wiki/Israeli_airstrike_on_Hamas_leadership_in_Qatar',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab2d62b88195942f89b36436a6ad',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab2efd748195bfc563f1f2542da3',
                        signature='gAAAAABoxKsvpbBd0CZh9aEkvM-5RUjAWiLv7rtZn7xprqhnP2ZZGO1OCL8nVF79nAUw1ZDvX2CgjDmZIzu1x-aQpbl7M_j0JAZRczokBZ-Yy6DPbWGITLzu9VEsWkq6d3yTFARwLc4PJ5UlKyv7F5obsnBxUw4T2b9FW64WeFB4ffZ9NaVkbxoFnvmxZ_fCQ5GA3aqS7ybD05lJB187_GS6ICdzz6IhM7LV4sSbUl24EInB7iq_EbNj2nhu9C93yU5R5BdY811tVd03K38OCcjuD5sUryu0vswJ3sMBVFEisEnhMWBobDnCjreUJeOPTem8amX4rsp-xMcDpARBZdfJpuxgNOgLIkaqd-meZ6NWd3cVrxWs2ydEKhy-qmjIgsp6P-ne4vd9ashIPAZtjs9HOEQEO_ohI1sMru3FwwMD7CdG0kfR1uScD3GSVbm3Q9smWozpbHmF_3O5K_9YzbpZSWCD7aX5GHTEUC49wBu4yVI9mjzso6oXECim1byT0LCOXVaczmyAFi6K_gYH38t6TLgSbJdxMOZBwqHX2_FQCK8Xe8AOVl_jUhBaLP7QOCOYfgidwe8O4wpm77ffKsXrNF7_EYLCf8YijVfkdUidjmJKmP6B9s-gxiwb5X7p1FbvI0l5z5hoZUpfcdhyLGNj0Qr2Vr4uOVANaYDtWGqzq8YQRfD2NU_qr08qd5jqAJljrVxMeCEyCQLg0pSK2hkULF9qB-zuysGFQvelksFyfeTV3CcfMo7leh4i2tbAllSfmJGjdVlJhnKttnlBhHseP6NakClQErtWtSpyFj-uPXp3PIr6mrzKmaWLc8mx8ns4hIp5P3y5QcXG5e0RcruA4fiVzZ1E5gVQk6Q45LVVyhYx98Qmy5oEV1gui7GzyUxyEozDEvyxDYACC91APrIkfWDsaFc2FQtcthkfGzzI5CMV-o5EwlgGHH_zc3k4cuVYkfRm80c5aYdCFkIcT39Of6EuORSbk8B4e0JUaHndvIKTmdWtTCkdk8ZoZQK7vF0LtIOGBBuiMX5t4bkzh6hcf5sRefh86MVAX4TisTXRNYsXWRp26B80f6ihAIOrGkGkCXiHSXv-bjKhF69IBT5R2SxDmnzTfXtXXCPsX78u-ZcpOVM9vJvdlBL7NsBtX0GyaUSdYqa_mfVfAuP7n0ZjYnTCH0yxfg==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={
                            'query': 'Reuters September 12, 2025 Nepal protests death toll 51 Reuters',
                            'type': 'search',
                        },
                        tool_call_id='ws_08313d91a201f39c0068c4ab2fd4308195845a2b59c42a3e30',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://www.reuters.com/business/media-telecom/death-toll-nepals-anti-graft-protests-jumps-51-2025-09-12/',
                            'https://www.reuters.com/world/asia-pacific/nepals-former-chief-justice-sushila-karki-chosen-interim-leader-presidents-2025-09-12/',
                            'https://apnews.com/article/aad7af5ddfb56b7484fa9db591297c9e',
                            'https://www.theguardian.com/world/2025/sep/12/nepal-appoints-its-first-female-pm-after-historic-week-of-deadly-protests',
                            'https://www.jagonews24.com/en/international/news/85536',
                            'https://www.aljazeera.com/news/2025/9/9/nepal-lifts-social-media-ban-after-19-killed-in-protests-report',
                            'https://www.aljazeera.com/news/2025/9/12/nepal-protest-death-toll-reaches-51-as-12500-prisoners-remain-on-the-run',
                            'https://www.washingtonpost.com/business/2025/09/08/nepal-protest-social-media-ban/57e5190a-8c8f-11f0-8260-0712daa5c125_story.html',
                            'https://www.moneycontrol.com/world/nepal-protest-death-toll-rises-to-51-article-13542486.html',
                            'https://en.wikipedia.org/wiki/2025_Nepalese_Gen_Z_protests',
                            'https://amp.dw.com/en/nepal-death-toll-in-anti-corruption-protests-rises/a-73972986',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab2fd4308195845a2b59c42a3e30',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab31d8308195b6934cc50cd1c283',
                        signature='gAAAAABoxKszDd7P5gHv5ofcME_U5PomZY5Mc_dOURSCy2x5WvVW6Z6UkrvsbWdA7BiTAOGXwGLqsduuGQaGAIbkMHDN-7Ykn6ys9K8xEAgTtZ9uWGWDFz5tN-F8NSaJ2aJNuTit-ctAfvs5HVqrwHId66B7f-4XAg8sAb72YVzG40qy4jdAcEe01nPjUbWTOomi2NLSVvBfrsXYn6BF80DUnCTdKbz0z8xQzYtocKb-iiptCFvS2LpfQrYgXmj3g-GpjlgF5vKG_FkkQmkMrk-SWOUdOqIiTF2CTqUKIYFDG99JZ_q19BVqclXu-cJEWlovFtXG19AXGN-EPaNFaMLuQ_SeCKH5q8qyfVyptU0cf2J2G7_c3jCYqdqayK3NDeFVJUAaFaYawKuYnuaYktL5dh45ptU3aXstJLBvwg6eo_Ka7nCXDSwVjkaXAdxVux8zV3fLTO9oyjAbGSuL7iAt4-Nfre48731WzZXeIYEzsBY-fSDfRVuGfN0ExUtKPsXhZkUSmCeiPe6Qr7g8L_w8P_45qYkcEpp3lJ3TkhSlHu_WQZL_VbqCoDhJNQESoIVXoTWMDsMxgO25Av8F3yxQFL-BTJ8T_YyfuuWs6Sr3Lc_GMdUV-X1jvjpfmLO00N4hMWjCdpVa_hu9Fk2nofQrxZ3i3xntQn7WDUGiELR9OSsKaBB1a5n9OseNXL-V3FXJjb8K7vL1CcxjDP7m7vbnFeII9IQ9uBzLLh59k_7VUetC_pXdIjalKKSZuLWCKz2vgNk4ecFVjevPUokQQPqn_fsGwSEH2DreTe97nkGouJhDNTUcaIX35PMKOuR38LOzYSjspg-qg_zJTPSatEdub896qES_w-XoBT0DXJQ1HSv5IvOPgEehNXGAAhSAVDkaUJyJJ7PDZvctEfRFv85Vx4NB0qvPWGqMp37zt8r-WONyq9oLBUqx4plxZebLEKkWS9_pQ2VRl88Em3HNw0Vts5v9IWExMkrtm1TlriztsucnSEkkNEa--vE4UKBox5O3QrTbBJ8_HFPdhJ3Ocof01Ypp5QsR2LdMlaoBBi81OtqzhlmlIfdTaXpyY47bZP5GDIy_oaJD5y2WdqUdBtvSAtspemdzv_qKxHygmwPQ8qxR6z5mY_igyNhD7pHxphAvsePL2w-7fXwdoxntf0aLF6HcDhB8l0yheS827k38bUUN-4han5vSLDM4WgkOEyawiYL2-Yy28EnchHFT9zGhL7GG3NrWnNIMatZLyT6yfup_8D9oqMs1iLzRd9ivG0gYGXpTAYtqpr9TNmjr5qYLsuQyrUhhxdVW_wtJTIA2xmRVyAdzB3g=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={
                            'query': 'Reuters September 12, 2025 Qatar Israeli strike Doha Hamas leaders update Reuters',
                            'type': 'search',
                        },
                        tool_call_id='ws_08313d91a201f39c0068c4ab3324008195a9d746750af5bdce',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://www.reuters.com/world/middle-east/trump-meet-qatari-prime-minister-after-israeli-attack-doha-2025-09-12/',
                            'https://apnews.com/article/380034326d3158e8f40462c87abc36ec',
                            'https://www.theguardian.com/world/2025/sep/12/un-general-assembly-to-back-hamas-free-government-for-palestine',
                            'https://www.aljazeera.com/news/2025/9/11/un-security-council-members-slam-israel-over-strike-on-qatar',
                            'https://www.aljazeera.com/news/2025/9/9/israeli-military-says-it-has-attacked-hamas-leadership-in-qatars-doha',
                            'https://www.cnbc.com/2025/09/09/israel-hamas-qatar-doha-attack.html',
                            'https://dppa.un.org/en/mtg-sc-9992-usg-dicarlo-11-sep-2025',
                            'https://www.cbsnews.com/news/israel-attack-doha-qatar-hamas-leadership/',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab3324008195a9d746750af5bdce',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab35c90081959da9b09d0e5a48a1',
                        signature='gAAAAABoxKs3A3pHeqZeRxRfLGuFGnDBXic3mPtkqRkC65cYqGZG4Ktxw7uSm1PS-3IcJop-pAlEhQImOLGb-X0yJEXJi9H_CjR9_q-8Q8JBf3IuLViwd_SSm72-A9PFe7kUkeIsIhmYfSpBqXAmPELqFpPzRqwpkWZpDCkDLiz2jDOLybXLipsXILNfYhyKqGT9Qz_4nRaYsJ9hSjG3vFFwVPHVqU7giSATDV_FxLnciOVnNyFCujPf7X_duRCYq_-lWpEWXXpatRO_qwcbcp_5DxWY4nHYUh1Zn5kNVPiW0rCYmHXFhyEwgm_6yBQAmTFv_3hVMievubKdQrTePXJdxgD7lvJXvjkpTws_KzZwmYcHVPYDmlcJQql4x_x4HLeEQOt2ZOjr90I9IC-wYGTG30ZduyuowcAjsvInjJHqOLEQQDcho80t3krAspPX62jtYIQwGEmHggE52JE33QcC9gDaY-Gfz7ToV3yxwarMXVGREk2aa2n7GqTG7R9c8GOb__xoc-2JUu95gkHLzX-NE8iy6P45PrqJ1s6WxDmiS407wHP_GlRw3CuHofcpwmaapnh6-RBWKjC22v7Gl6wkMgcRO5fVkG5vdNIDwjs6he-9k_VsdEWal8Af8Xht_EQYKSfy3UK1DpqQjOvVahtdZTopctS-Q3tRmi7paV2G8I8AwfkW6AQG9VZnj9DmMMAS8RR7vvXHRSrvAKJmTWI93PTPK02QscVJhBjohY0nEpGum5nhFE8MOEBIdw7ZsuLWs62b1ze8XVaWhXpXd3bXv3ZrLlfAvzxMBXmpjXNB3gUcClETpbKpqkAMxYtpJiHTlFrldQ1kPVJJDZCS3N15ePXIErVdEXjUuqTf8RD0gv9wistzChCA3YIwGpOYFfn4e-RZrRuNFWO4JlNc-GgQkD8iiEwHS1EuDEOqdlxKwVLw_5vSmJIvGCZO-R7OQX_V9rfh-cApMCsvn_AX6mEao_7Z_pj7LaEHL3_lJzaQlbz6toEk1R2YmCG4KcEt2uKKigXAI0s6yiYylWx9v_g0JBZHY628pYCNQVEVeSpAZiXnuTxhbiI6J1dUaUBM469J68u6ASnfOoEjZkz4UEkIPXXPw_nhXGOJKI45Wztf3MKnwftovY8X9SCtJkuP9Cqr5NfKB3Vnzlw5N7FFoq8_3YvrH4snkbJCcMahoDxVRO765md4SaFiUwlkCq2tQOH3_jNFAVj3OaRpBjbBUS2hIeQiT4OmQrKMwI7RN3J5jZhc4rmTAvSiyj9bmwGmZp5n66BEJQe-SXIfhKzqhBN-NlWFof3h2v_jaZKcCPzpLpMDZ4UGOzAiU-ThshCp6V37X85OS5xHwLtsrazCqhzjSnJabkZTDduJzteS9GIi7vKjccu8hTyINI8khhTiVMOyDtdtMBeoHRjF9HaPlHKzQGryH9QNMLQlEWU-9z3k809EUHzSgqzCW-4pvofT_JMILiFIq0H9C2Bl94ISnkskTL8Rw2qLZLAvaHgBGjfWkdrrGW83eC4=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={
                            'query': 'Reuters September 12, 2025 UN General Assembly votes Hamas disarmament resolution 142-10',
                            'type': 'search',
                        },
                        tool_call_id='ws_08313d91a201f39c0068c4ab378af8819598e6fdbafc72d7c7',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://apnews.com/article/6afa02f5fb4fc8823ac0a6b74d69e2d4',
                            'https://www.reuters.com/world/middle-east/un-overwhelmingly-endorses-two-state-solution-declaration-that-condemns-hamas-2025-09-12/',
                            'https://www.theguardian.com/world/2025/sep/12/un-general-assembly-to-back-hamas-free-government-for-palestine',
                            'https://www.wsls.com/news/world/2025/09/12/un-assembly-votes-overwhelmingly-to-back-two-state-solution-to-israel-palestinian-conflict/',
                            'https://www.newstimes.com/news/world/article/un-assembly-votes-overwhelmingly-to-back-21045225.php',
                            'https://www.thehour.com/news/world/article/un-assembly-votes-overwhelmingly-to-back-21045225.php',
                            'https://www.thenationalnews.com/news/us/2025/09/12/un-general-assembly-votes-in-favour-of-a-future-palestinian-state-but-without-hamas/',
                            'https://www.ksat.com/news/world/2025/09/12/un-assembly-votes-overwhelmingly-to-back-two-state-solution-to-israel-palestinian-conflict/',
                            'https://www.investing.com/news/world-news/un-overwhelmingly-endorses-declaration-on-twostate-solution-for-israel-palestinians-4237326',
                            'https://www.timesunion.com/news/world/article/un-assembly-votes-overwhelmingly-to-back-21045225.php',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab378af8819598e6fdbafc72d7c7',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab39eb0881958876945512303a0b',
                        signature='gAAAAABoxKs9pCnCXetuP17cr4n_rnyui-nIi3Zg1rlysWhXDDi7Bfv7fH0HVPshe6ESao5XNuqLSRBXS-bqwAB19H7DXFGH308yY7wXZC3LiAT6QOCHoEF-kSs_oM5ctRnbTxg9McfjSS18e1umdQg2LwnquOkUjsjbk_HKydkjNjbDUeitzy4c1rJJwn14ObvIWGvMyyl-ARGbxMI7lFzE6XMo9amfpGMnHw_6WscZhF_MenyBJvcJ5-E1qJp0uOGXemvmq-e5KlmV8eNmV3lJmJb3tsEYG_jnOv0iBCK1oduxTlpkfCv4kAmrdbj__u_NB4mjIMKjYJmoIzwPrX_QPcaDQ1FaJElo9wH1_WXPlHHwKVfxkceUEgs6ZNUj6Gi5v69sIens0r1KgBRQwUSzJTrjDyjzwzaVAsFfIK0_-mgPPEmoLdGNGuYMMW003I0HbnNoR8rLHf_cJcsxeZ8M37wHm8IkiI4XVUuiy25UPna4LWUifA-C_3Vw6Gpw6Kv0sJQYnd5xcvEsYTLQceHFuTPGkzPvmg3vImk7WYVOM88XhzJHaMX-Uk-Xi-2VX52Cr-6mE0RIO6GAIP8r_iEpoy0k6mRvkFmqFIIFMJ5uk8VdOSbdTRKZ_6ySfCpvbExpe9FeJ8PFDX9zxA5BbjDYzsx53gdLRhsxzEYuwZUjfqsdaKRjssYegOm29S-XnI1w7qzZjArOyBc9L44M3TKvbNmgIiBVGTXMHl6nO7GWU3RTKF_lx5wZTS_eud19UVUK-JpmJvyEWcBorg0fSzSruFoqDjT_jQbjb8N7XO9ppphafpLV29eU-7EnW-GKJ1-M3ejov7fFFwrjwTErZCvyhqBqvQEv9Q7f_WPa7rReCbMQ1sojEC1g1b9FPiRFNiN5AOWFm0GKxLTHZxUZttNq41Uz7KNpDioG7bJyEqhZQl2wOxm45Lmc-WBmA02S7NK6oSYucgbNJof60uWHVA2AaVqwqD6k_8E76lNllD2hCGUhwmeNa3l08UUSIE03TUkHsXCWQ7ujsjfJqueP7rcgHYWk-ZnhfWdLXcgvpj-CEus68FcR-Dq--D0JFZ9-uqEg83hF9-xPiYJ3w8C0ltR_cp3iJlRAlM5YNLV3x_pyWdS5tCJdy7RGgXrj6GVl87T8MGKCllHq7RkJ0jMP03jN5hxVi6V49XWyEP5Wo6MpMHGr5DSATCEvgnPToe5M2oTwakyLoHy6D2Lw1ir_i1YWqA-k4Tfg4RL7Ngb-Vzj7QbrXwjlobKzgWY6vzsgNvBAAFNyKHUV43ZVHzfSeNqznZaGgXzcGPjj_otS6qj9jwYHhhZU8S9DditOAPt2_J5G9kczB4xdXhrefGLn0BQV-cQ3f7wkVnnwD3mfu1HalHmAy9YnE5k78Cx-U9YTX0ZV6f8o0PUZuw_akwKKESJZT8SoI9RqpO7_Ios2QJvoiG35kLXmkvvUJCowkfiwiN5J59sUnigUCvqFSCs7gH7bCE-pDkfrGd9sam_KXNZMQ7WiyD-u8sbzfRlm9DLR_3VaYnXhAeQpJQVrxYrdY1INmOWqA1rSWcNJCvtVo3zuYn7F4P1mq4N-DcAy2CfchRLtU4miREI1h9CKi4OoAXtTUfVipdmPb-p_UrHndvblGirSJxI_U05Pbz5G13Cg61rPGiPOzphFFcBFF9fctpCwQRG9xMdLP7MOM3Yl1yM-fU2iA6H_7poQzLKgw46lPUQC95kH_466592T3cjHlSQzLPzpmkX8IaSwiLUSLC-s-_G77GHjVfkkGwHiLmZu1HfDy5yAsvnGvXwNqOcSMWC1lZwKTfIb0CgMpMjG4K939pnhMoPDiJOl_LIBT0Of4gLIREdNc4hbKOX6MHXc7it-IF8cvKpMI5xWtmNpxwWsBh-BwK1m8NiW3pla6dtWFhknAToKDnZUz6b37-m9vzOttKVCdTtMFvHDqVw8NYJQxk7kt84sWLjvjiBEUdfM6IginxDNy4jt_gcidyHOy8_BF2SLip5L4xAwlcjwj6aV0dNKZLugxogAxgADJIo3fj3rmoxiPZjXDKPxCSUCG83kkKKNiWl-5_4PX20ffXhMvLBHzwarcRB3ui1dFqu2dlm8GZhdVniGLIP-fIN8zhuoj42vLDFm6OaE_4wEuHl0gMuIcXrTjudK21bv7zIXdOFI_lOf4RArvfrPc_oGwoUgPRIQtTEsjnqYMIvK9mXQhcuACRsWygSbwHglRy7cyhAJYrsLE2gb1v96dogG16gDK0DC816wzaRzXgnfIQIzI19zAPRtrdeVJxjqadTd7RrvHOPDwoB824-WGajvct3YCf9n_uE5wz5YvH7rbEGdkf-C3wXtybFYzn592cbcXEaumhAwyQIewYFpBHBDWjWeIK_Ms84DXFglc3A3Sendy6hZU1RwHzLsonPXdsSEAO2Bema0aWU9bzSuXJrrgtBW-KxWjodhv3g==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'September 12, 2025 major earthquake', 'type': 'search'},
                        tool_call_id='ws_08313d91a201f39c0068c4ab3dd990819593e473934de5c75f',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://nypost.com/2025/09/12/world-news/ukraine-strikes-key-russian-oil-terminal-in-overnight-attack/',
                            'https://www.reuters.com/world/europe/russia-belarus-start-zapad-2025-joint-military-exercise-2025-09-12/',
                            'https://www.reuters.com/world/europe/kremlin-says-russia-ukraine-talks-are-paused-accuses-europe-hindering-them-2025-09-12/',
                            'https://en.wikipedia.org/wiki/Typhoon_Tapah_%282025%29',
                            'https://www.aljazeera.com/news/2025/9/12/russia-ukraine-war-list-of-key-events-day-1296',
                            'https://en.wikipedia.org/wiki/Tropical_Storm_Peipah_%282025%29',
                            'https://en.wikipedia.org/wiki/2025_Myanmar_earthquake',
                            'https://en.wikipedia.org/wiki/2025_Afghanistan_earthquake',
                            'https://en.wikipedia.org/wiki/2025_Kamchatka_earthquake',
                            'https://en.wikipedia.org/wiki/2025_Drake_Passage_earthquake',
                            'https://www.thehansindia.com/news/international/deadly-earthquake-in-afghanistan-tremors-in-argentina-alaska-september-2025-1003393',
                            'https://public.wmo.int/media/news/hurricane-fiona-hits-caribbean-and-canada-typhoon-nanmadol-hits-japan',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab3dd990819593e473934de5c75f',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab3fb0b8819581587a21cba58542',
                        signature='gAAAAABoxKtDkEeBkvBFiIw4clb4bv6W9I2QEU68uc2ZnblfIjBLq8sumx0B6Zw_f0V2DrQ0tfwJPUOZnj-lQ_QhxnfDRP3-UQqfcDmeSeCTyvwB7czT7NusncgiF9XNWXMr8lc1KCmZJKEBLJIuAKdlBWR3aK67Q9_IQfEpvtkAI9Iu3nitNDFjjdSpGcZ5W2EaNkqkcDQZ1jnlal-tGBryXRmU66knOlWJ7Afk4Qmw2vpLnSLI24HNXiJhOQdaGoousI9A7VK_XvHvOCYOpwquz_keU59hEPxcO1eDrLuoWABm8HvECwOhMQWCViCOeQ-E3iIP1kNhxIbJBXdEMzC6YFwkjI0C7lQmS92YPmNRveBkxBw0lVZrTeHMCV5ksgkif1E1EdQ5rTuLTWvLiw4OZzCnkN1fbVnEJ1IEWtk2flyjyea041SnWSAADpV5fEuTYtD4WX7NEfN2dDvmXgv-vj-cqo_-xkoV0-SC9t0HqUv_R1UmDE93arrWJjayK-OrSvEQkXu1Zz7dDeSx7EIBKhe8G1AM_tHiFREXNau1fkrq9e4O1-G60pZdFJ8opu_Ybf8Odb-ko237muU5VKJg_CgdVxhGlAFeYWN2WpyYm_VWLpAW7bB9egHR3S8JMwTrfKGSvDSIfG8i-_cRYls5tOi8FNpKXJJIyMX44DmqP600jmj7kFYrutf1RcLFQZtT0SgnBPOYIlvLg13M3V16iYZ1hh7AR95jUvjI-GnWysW02mUHTC_Ae_VqIxn7Zz6njbd0NK4LJdWE-jQj_xhG_w3QGoIaI7ALK6HqzXqR4kTKEiBR2QBj6mYJ6rM6N4Y-UI1L-ep_mWJTvvhZxou4GOBu68MmFOLrNQ6cHW3MS1bVEX4S7yiwRH9mV--TFjhdIdIcP9rDrWjKrtMMSm3Zzl_GmxCO7kaiA-_XzUC7jk-dEx3SaU2uiAw5KAkSqOHTFUWGjGOOHESDxm8JP3rZjup1NHYBszlxGnt2Ux2cnKBm1k-z3Px6_VGqxpcs-CcZwsuHncu_ZxD1-Z34ivmgeB4dTJRi85d2DJJ8Rk6PQxNyh_hBvVHKlMH34_2JPOiQ7gUQzlLH8c68bNsbwRnU6ebOfvU903cQ0dgpMZrAuN-R6RDWIul3kjAJsEnnCzeP6fVdA1b5i8QHGOaB4VG0fJxJuN8NiDplS2awyupwMfQbqatai24C6acczmz4rkufHw5jORPmmRIjlTXmwLGla9xz39HyfntBoAm0tKy8uahYWOGJCftmDr2nKD7k3-JufRnZaPZwKFGd2P6zEyWN7_cZFnYERKlPKf3k0IBCix0_QVu1Gk6_98aANFVkZO73e0YjBPdpNMgQ5Q6KhC3rk6ZEc1GzcTX6OJNVNujPt6cybO9CsS-BBGRf7QNPAuXvVlCrTmUDMSJclPZE-IXM64oOolU3cEOZUevPKMztgN7Bsx6kPnJTZW6ANL2rvreQO5xVX2U5ob9rg22USQO4YmEQn_JAgx-PjLg9BTbqE-vBHIuSlhON54LeAr--7cQipOGp2yisvodl6ZnYlA3tKaTiUcWlkdE4HMXn5zM57dqHUPrNCci9FdXsay2E-buXKnfq3Fllk_6SPB8cXvz60Pg4gufkw855DB4aoF3HCQSR5UFaQQ_4Zb3WiI4Ic4kQl3HVIGSDel9KmReyvJTWg3eiaZp25pXOXqR7OADu7LBb57CW6a2xkF-Xd8UWhCWR-HbZuPO6o3OaKHQ4jzC0Q1xSDrdXGHgnDOL6SCiUfSVd-bjkaAuiCLuwDz-Ac_CRm1X5rhJjbTN7-eHaCEhCqOUg9_WtMM2B8nXyx-pjNbxF5R43CmIHiuVxyI7hEbwe1U-zT7Pzph2dfPNpPl4agolA9e9Ucqtl7A8pkGPN0NQAzHkDlzNGS7soVnzKBB_60t9ZTNWa-ndxa0TFXAAQjNrFDLCezb6S67pL9Z3B39aSrroL8oHuHEn9PJARoBZpW9PDbjsm-4KEZf2YE-rMDPuEnCLKNqnAM0cQTGgEXm8E4O4cK58IbYlN44LWB02EiuidX-A1VxH53Ty_DtXAeIRpPhcNkOp-mcCniiidYzpkca-lZGXWVsOyS3U3fUgZBmyqyFGPX66u1R0k3W_vn6KMldvEsCrZ_bqo2phg_8i4kimHZLJf3ZGu1AKfl8T5xCDa-I54O-MBQP4JxsKxZhNtknUofzLoAeZ8RkB_MUjIprTiVvs=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'DW September 12, 2025 top stories world', 'type': 'search'},
                        tool_call_id='ws_08313d91a201f39c0068c4ab4308288195bff2b3316b62cb4f',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://en.wikipedia.org/wiki/September_2025_Israeli_attacks_in_Yemen',
                            'https://fmep.org/resource/top-news-analysis-on-israel-palestine-september-6-12-2025/',
                            'https://www.sbs.com.au/ondemand/news-series/france-24-international-news/season-0/france-24-international-news-s0-ep7/2446206531749',
                            'https://www.sbs.com.au/ondemand/news-series/france-24-french-news/season-0/france-24-french-news-s0-ep7/2446206531754',
                            'https://en.wikipedia.org/wiki/Israeli_airstrike_on_Hamas_leadership_in_Qatar',
                            'https://en.wikipedia.org/wiki/2025_in_Palestine',
                            'https://en.wikipedia.org/wiki/Assassination_of_Anas_Al-Sharif',
                            'https://www.sbs.com.au/ondemand/news-series/spanish-news-france-24/spanish-news-france-24-2025/spanish-news-france-24-120925/2446206531904',
                            'https://tv.apple.com/au/episode/friday-september-12-2025/umc.cmc.7hlnpg91k6gwgl8hnmn4vayn4',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab4308288195bff2b3316b62cb4f',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab452f3c8195bf1084b78906cb0d',
                        signature='gAAAAABoxKtORYIZjmQ1Ro_cqQ2taAbm20J__IL49064r4j5FCBUj11JhBsq5FtQcIBwl0HrT-MYYtsAQ9j2nxZzDHwNLMBIAtXQtKAMWViqcB3B3oaHdhfJYRk6eK88TjhSZPitNtxnWsDE9ZbjCdybVqsWLwPiAdASb7IxALYkF_L9WSKIrMO_0I0pjBmkX-rKhL2gjEV3yS3jg34LQ9INdcIjtNHeL2fnhe51i8f5RywMMcyehFTJu9jX10rBIBnLRjPBM1t1UQ4_XLrU5LVv_KxaBs14WJimLFgTuLI3rVCktXk9aMegWi-zIXxnZ3x1nVpyx_ZmNVvmZudd2q2iIDSSkNafEApVuhXwvUSsQD4hPMUK-1r8HXvXie1jZjpE3cH8o5ocHHDenKKDXsxZnj0iaNaKvsbYqak19kyEjrXtfNF-1I-Dv1ke3dt073GL7UrC-u1y2-DxI1SRIkIN4HtlARshnqSxdrvdJoPdBlEZGLs9_qKzCCFCqlkxX09bgRrZsCPoQelJ0VyuhLDNyutxJE4gwj64vNizLmjch4K3jOW2YvgQyMSddMZAtF87XNNmp5MVYyCQUn_EpZj91aZSKFdGgx9CmfKHKBwUQawyeNSZvMU4IY7nMOcGVnjAoLkhFhJVq5AxSpfpcOadH8Twy9ISKM_9BhjyGgrSnbfU3na_8ixeHlDoEes9Dwp0NbGYPBLnvTyvM99ONXDSvsaBzR8mMTNMJ9tp77aA5x2McrVXYVX-cqflONp-2mZ6xGXUm4uxKbt9hOuTRrCCCR8WlyYkZnkVFeNE-dEpeMJCgbNEE4ik7BtiQoyq9nXD5v_FxPFgZyx5wnRa2__FcbZ_Hl-fD3IHTeHRZItXaclFCBJhBU1K2fXe9HJM-Iwo9AI2Cz1T5Yql2VuWQwJ1CidiZWuCjd-uCe16v-gq0R7hUjO0LyVygOZt9JnFPLW3nzItbIgJiIN92wcOGKD-pM2ROCCX2lu1xG5NmlOfhjG1EVhZu4GAAgWgHPBSEndzgtrzZ3SVgBjjtbqMiq2hNmtX1k5BW_52OBqjQ4hdYYpj9Vb2swnhKomPI9de80nIFXGcwztt_xcNu7KmyE-9-rrMGPxpE-_fs3py-ZhtgyrJbEnsQ0aVLhrOW66CS8VnRCs9Nn8C_5pU3Ihqh352eG477TirjmfehVak-sO-w9mqAfec0eItoW23hcSjYRTMIJv8hUaU32HPSSEPZt6-H3H2Mstyp6xynGTPRoX_gVsYSDrYhJ5ziVmwpAhsx_KM2pZkbo-MK9JdLIkQzJQv2z6TBWRCGUkqKb9FIIZfwlF9o9ww90fMW9S955Rvs3hiWIvvAiCtod9r2QPz9okjn7qbB35geEBVfufyVRJR1d31YYF3j_VqbYb_Izs48waeukyw9Ps9jE-34ye8Oas0ai3ShpU6QxTXZLmVTO5lH2-qSoo379GWhFeSThyjMVcpSLDvTPu6NYBXq47oYa0v5lf5mKgDXuefqN_bBa5frNUlU5krIhxj09mSwG7pfqG8b9hvICnJMAiQSGKBKZlbSgXTL_bACXN4jmcwd_3RUBK2kR4kZD2Q-pG4NxeXdmA4CvJHQsH1hTiRnmwvEA8jAUUx1Nf-cmNIzYscgnb9T4guZxZForCNIH3FnTDaaiPPMRSdK7D5yvtgcXckZ4GX_WKo1RjWQjVRIrAFeKPl4qKAZUk7Nbni2S_10xuMPqtuhar9UTOz0tg33C0yEoKU0pm-0wlJR486MXUOlkcL19paxTxzGbxsXgO6SdOI1NiMZ6Y6o-flmYUzAF_8qODuaPVAKspNiZcns0mxUaBtW1LArOfChkB-vV38d915pQXhk4qF7Tfi0td528vlBra-dO1eY98bs3yRsG5P_CoJNGNorSSPgCGnYG_c2NVO_JRjLZJ2FXRQ1pFyzBD9yPU1HosaHNF5Ib9FPSNhMTVtErIRaIhpjEUdkJryjkvPvMkqMIkqpHomXMns8jyQZKCphTz3QTMw9QcvigzwYTIVb4WSrW9QowlHjdljbr8vD3i9V20zDKxi1F7wwf9-dtWHF-AxdN_kSqSI2EbWeP3lGp8GkDHDzi8B0-PL1Ty9EAPcj_78DIvWCqLxQSY_KWhlJNaumYhGgGsUOWGlKPktyGoEHNawax6gk0pYI77ZZ9xxIOOTmwD5M2O7AsDCzKUroKFx3dIVFRHJXFVMMZ4lYbjFG7IkhxY7IaJdH7u1fhJjgBPdnr1K9UVnadu79-B6-FJO2U1dmBimPHOeJ71GPwthpEEgGd8PKVFH7VTh27KgUyDE_fyiHkvUzZGOXZsmSb0M4Icu_C6Ix7aqvlUy4fjSpxsj54SKrivh1C-G5LwA9F35x2bVbRXAw6Ut5VnOsbzotRBVJ3zzj8lQWs2tmxmSVLWOeIWyxuv_OMUmRXPbS4G8hjiMXt4d3PZM5w0PYsves9HxiyL7QL8VNKty_jyCl1Pp0VgDaYOiMccIoLgogcWlGJS8IeCTYzKsIyAyP64MgRXEBPnZBPRp8QxzhOVqPy30CdEKf9rBsFIInpb5Y0sOK6nGEiqvJDI7GuH7sfMwIHKE5UV5gaftd1ZaENi8CtGGz1H6zgTefCppmRLtDVqicuKOHre2Qs0wGkIi-nQfwcDkRuz-jD57cEjtMSXGkMLwjXSqaSozEg59_6IC2VqdIo4S01tILQ8-k964VlTmED0xXrV0sHa65h-Y_sqfCjU7IFAmSbjay2x98m4ulOKi-4ulWV8DnZuF1c2UrSPURh_z49tlScxtUIgL2IcKyBNK3Qb8rKLhRQWNsmAwNqRcEQ5k9Q0xcIA6jjKln-ma_LIuOQaQ4NE8aiRhF-kqxV_45FTRm7RgY9ccpSMFM_ZgM6b6Hsw4VE4HXbCESEUCnckl_54cRaeQ9767zIo5GaRHDqiMMS14b3hwMquwx76QcypnhjQtt1W7rPnJ_5Vl1ASZIJuwtUaFJFxvH2qTHft64lqpvvf94labXJkM2YBiUT7zxMYXve502cD6QwqV5ltP6RxYMcS-ez9E50553zmseWM5o9ZvzqMTDepCwE4o8Nlpku1wUkfIizaOtzRe__6bVRF_JTV-i1UnE1mw9UBC5TW1stKpkaW-bxNf8vN8jS-ND_fWfkjO7-MO4uya7sTv4zRL0YBnkwtRT581NYeAD64df8n56A5YtGRafHqkTCVqdi_GERdI8UyCK9SwTRXbwgOUd-IpvG7uTb0KPr_t_Uk6JWzYlydNsjOGDSV7tXPkVhINQJRBtaYRpIFcUafEnY-Jn43oWbatLO8XCLbSh4emcCdAEZUOw4-gq6tiHyNe8b5OxffINQy0YLTbYyw4rxNCdczb1kV8BHhdJ-BOGlBHskQvz8OgluYAa1p5lJzyazVobxaq-cJhI2CbkKOxL_jAAgcHk1T6jkzoVO3BqUAQsQGe6hDatLyHfVqBiCU0_4OQBHvTOl9Pw7EmGyjaNf1Vv5r_Jh1mk04gA0cysvCMYw0ObvxULyXzohnqCDSM9iJ36CUhxeBJyNRu86GhsqVsjxmLeIIG2VkjOQBR0pSv0Uo-co7G1-lZ8xhlntnlnLBEYGS-ffnT-GpQDckF9DLB8_gKjH5XPjhNdf5fwbOk_1aY9EH__e4ghgt6YqG1QRAOO37YPkN20S4nVD5oGGaR7nLnIoC3b4moFW2vgbGZjW8ElSBDd6kQWSGFnKG5zu3zXZNAUjYhyucqX4Y50BnVpO2kTYTmqhXc2HBKZHdMJBZ9QM7ejdXUchLpmZhpclgjPWdFkmuZoTF59A1CdCp9eUFOd9234-T-lYRQ6KkervCWdMyfMect7qQJV-iCSMvzIQCwmxGlFdUoYzX8ANZkco61EsiA2T8HyvOsuYsuSVkaO_KHrqMwAsCSq-RrG06TMfLavIqBkbZlAmzzzjLTmD9Xf6XTMmwT6DpriJdvvZOfZXBuJNlDVQtsM14GuwWBYpc7aw0YktDPcYDBVVQvyaNCIcf7zUQMNxiMATDCVrVwwzBAR9u2Q3aTdxprfcCG9b7lFLzLrNpXLzI7GXZ7UUKCr-BAeF4BmlmDdCyVCqT5RWqpCG13swVP6sH7sfVxw72ClVLJzfSlG80C5pp_hS-Jpp9k84atI9cqljBD5CXVoKv2UKZRLnNGJn4HdkmONXQ480o2sEIa7ZXnT5xcJTL8zzsS5zZPjE7pJ9qxVzRRvuCrQBqLiLxsXVOuPXiz0LWCAgVMojZgavvoey2JoN6g1CuinF6SzNM9tdSIw90gq5iydeLpV2qbhDegg2e761FLwBHx4kfy1qIjOQr01aGf0v8r1pCR0Q6_RqPq5J5U3Tnc4rowdkjKY9NrfpqY2gHMPeBdPsrcAZFYLiY4YmZozUcHvLYo9XtxEOUtLBDdBkB-PAT2Tt5x_D8rgGlYnPGhrTIaBFQVwmLj1iGaznVMWD-HXP1omDZhTNts8P4W2M56NMPn0Pea6XDbq7CHrD4fQmDtdKhlMOP4PQK-7wh2L1DJEo27ZWKR4KwYheKonuZZPOopTjqPjNrtvkw4Jq9YTWcByIMzYMxYqF5MLNMl24wkKuu-xBB7ssV_-gC93bFvsPqL0uTsRs9Z6Pto4AKkWiQqvc-EqpPIbQ==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={
                            'query': 'Reuters Qatar to host emergency Arab Islamic summit after Israeli strike in Doha September 2025',
                            'type': 'search',
                        },
                        tool_call_id='ws_08313d91a201f39c0068c4ab4e23a88195a39e83d44d538ad9',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://www.reuters.com/world/middle-east/qatar-pm-meet-top-us-officials-friday-after-israeli-attack-doha-2025-09-11/',
                            'https://apnews.com/article/380034326d3158e8f40462c87abc36ec',
                            'https://www.theguardian.com/world/2025/sep/12/un-general-assembly-to-back-hamas-free-government-for-palestine',
                            'https://www.thenationalnews.com/news/mena/2025/09/11/benjamin-netanyahu-reckless-qatar-says-after-israeli-leader-accuses-doha-of-harbouring-hamas/',
                            'https://en.wikipedia.org/wiki/2025_Arab%E2%80%93Islamic_extraordinary_summit',
                            'https://www.euronews.com/2025/09/11/qatar-says-israeli-air-strike-on-hamas-members-in-doha-killed-any-hope-for-hostage-release',
                            'https://www.aljazeera.com/news/2025/9/11/qatar-holds-funeral-for-victims-of-israeli-attack-amid-regional-solidarity',
                            'https://www.aljazeera.com/news/2025/9/10/gulf-region-at-risk-qatar-seeks-collective-response-to-israeli-attack',
                            'https://saudigazette.com.sa/article/654886/World/Qatar-to-host-emergency-Arab-Islamic-summit-after-Israeli-attack-on-Doha',
                            'https://dailytimes.com.pk/1367098/doha-to-host-urgent-arab-islamic-summit-after-israeli-strike/',
                            'https://www.hurriyetdailynews.com/qatar-signals-collective-response-from-region-to-israels-strike-213488',
                            'https://english.aawsat.com/arab-world/5185339-doha-host-emergency-arab-islamic-summit-after-israeli-strike',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab4e23a88195a39e83d44d538ad9',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab506d588195b94c9de5f5e30f4d',
                        signature='gAAAAABoxKtVzuQn3NjqOMhVpXMk3LeJWf-aL8_Sd7_TMGz8JTmM0be8STmragAQDlzCooRITaRV-aOfKWWgeEouDr9ryYSKd65A-71Zc4Nv_uuD7Nnger-4-V7ODgpQonSMsv67mxlA1NICRZaKsGnlDeuJkKQwqLQXxHZqfTLTExp56a-xN-vEIjb2ubRQdMfaP6mdD6Xpfebr81wDnmb9HcZ92oZz7mNaH6VIbLb2Q96j0GdAmTXgSPBZq75YkLkLruQ2s-SWiNarKH6Dih9ZiJPBGGupDcLoSXyKUkw1oKSMH3u4MxMZJRiD9rw_kPvMqWHqdcbWxtML_Kg3OTbQ50kPdmZPdpMxG3H8wvId5y7LibxNH5TNwfQVAIyMmGT_C0ukpjGiF81mT8a8e56h5ez8oJKSXVaKpU2D1pAVFYrGehWCqKljEhg4l4jwHjlpDjiIGqO-Ur1gJjmm4dQJLkbMGFgCO6AsDOuGy8YAXENUt-7J9izTNwShMpxhzF7ijDItEPUalInvl7EhubBsSs9gW0fm9J2xrExjncVeseqiiqIG4Z5ZURVyQCfdEfZ9hcR_b7aAX8GQ9vURjZeWFOpBpHCtXCRS_tXu3yH2FL6qaY8uSathgR0eriwQ4sKsj8FA5mvYZiXqB2W2KtoVzbNtuWM57dALPeUnGPJUzKKOQuh0PmwRqzf98IRg4_lDm6j3qH5HhwwGG_iUJdIT5Lhr2tLFVhNY8Cc8NBsJgNSS5a3pgRo-rLj_OIhyB5uUpto7Yc8Zw-VUZCbTTltOXQ2smO9vPiB2PMG1Ngx3oPaYTGCexWNxF0Yd-dW_JsK8a5cpcvija9mPsj5pkpXpFdoXViKHbGE5RiR1o1T1drhjp-DnpNUpYb8wlUvClPZTNmrAi1RUcGXQTdWeYHA0ZPM-0JmUhRSB4lIrT1DfEIBQqHl9A8PIXfhGCN1B-li3gEaXO0XrL0P2enFbwzFFCxQAjsvIs3DRhmE4im4l06u0-GrO3P8iZt-7bKyhvIBMGslVcUdtwI7NZU8GRXodQLdsVDciHYERI1YN39VwWVRX7Dkkotuo8hqf_Nv2dpNyn_17PKrULftrpGdGnbaRc1iw97jZ0yyXytpHwexuolI4KMrNE64RrMolOhR1COjKSMus8Yw9dORwn9RjP7sHMdIckAKi6rAEEdSuCNcwyCOlsoF8u2Zi8wfdkmXWxq5NMoRGLA2Fg97AttpwzEL1Q5UfyPkg8UkSE6OD5dQMIjqPPOrkzVTpjp1mi0zdOsgCrj1B35M4WhKMJq9x-ntvaIyZhm-oRlXor-QNoT5px-sPQKjbM1Li4sNol1zaJDNxp-ALKcR46zz3ERcEN9NLcCSnldXT8qgLccqiLuIcKmuagWpERUVjexfs4KnhmTtQlWgaiwWF7mHaXu22vNWxL-hfuau_Qeu0RVVjaVPAnlaLtJDfVi8NxSVDG5AIvi5h6jRvsH7FbiuTxNXJ8Wwj_OrgISFiEal7vVShQERXVKNT7J9DtfmSu_P70RrHSsxUKmZkhJrs5IoH4yp8V36tnQgG0w2PuOl65C1e-r5IGCQ_Lv4mjBWjjlr4UuupJ861PrBexpSd5tDQaS-9AnLF1YXE3Zc68cjYruafHvFN1koxY2cet-TjAROrcuY2kgjs6nFunhiVQVfCjx0lGONrXmBOBVaYKWteRa-bUzwgs-RSvpUmnTLQTEWHQbp-TGgM_lPPT3XmLY-g5yRMcUgbK0bZrbVGCE_jmnjaCbw1bKSbA8WbzJhSl_YGWz-dVFVMWlocVod51_aGEzKN8S68bvid73O1NmdT08Fp9w7bZk-Ui32fDMIvxQkZm5eYRkce_cXkhW8AuyUCrh5euuHv8KdnWk5CAjCY4iI_ZwI3GInWYHXiCbcUua1sSYHbxPDRPRLGvRs-Syh1BdSGudRWqraNQrvAvbM6F840_pDnFkf5ZWBz_yP7b4v56lZEOvwbkEDTTuWJacWiw4b-60X7H9PsPoBMGlB4GHh1T0kOTb7EmUJSG51k3QpBeNqEmZZVcqm9Eo4SS0DsrlNuYzv75dfrHwvrTNF9XXns1P3fH-1vsNifLzYibxQkZccC2bxfbDnA5JIa2GXnUPDtFhY6QhxyrVXYbNaBwzHZfix7IuFoKtCgLE5O_pNfUS84yWZ_SpzLqi5odYMYIUbUjgPXPvP65ERpn4MbUxGwT9k_5qXcmUE8iJ7reafjCCCfwNuKh9dYKCgxtVIY04ry-cwZreocFMB3wQDrRaMIEBdVYRc0rAZpvC3MdDPmV85g1cR73UJLmiACsegnyQmSTaYRB4rV1LGwY6xLhID_ikcZ2veAT-LXenMUUXzBCFU9SDCT2Ss3BZ0hFc7EvxwOnqaA5uVYnW3J-iirL5GMgfhrVDFVEOPpFCChHlJAUoPgaDYn8SWhV7Ekx1Uo7aM7y9xclRDm4dciaN8MwZ6-GRf1OIvGIH0f4AUoDaMvVCENAXTAvWl7oJRWl2OPyox0NBt-DTqZy0iXqH3DMnIinlQIAlZGlFR1FZMCbqvneo4P-r1BzrKWVyr9loUR-loFBE-Wot_jhiyPlcHpk5P935dO1gsPQ9h0aFfLzqrU5vLbmnGGrCcPabN9rCfOe1pjWwBtabRtFB5MIXf9iP4H6PNu1G18OhTBE6LcJp-b9daMJubtumQywt3DPym1w0L3MPblv3LyyO7REUTuNT0EN3PgN0VM3WtB4j626i43Fcz1e6XHU0Ull8_M',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_08313d91a201f39c0068c4ab5555f08195b138210c2a22c5ae',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=188609,
                    cache_read_tokens=156672,
                    output_tokens=2478,
                    details={'reasoning_tokens': 2176},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_08313d91a201f39c0068c4ab26d8408195aac8f967c6938d48',
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
                    id='rs_08313d91a201f39c0068c4ab274e0c81959ad36da9d42a937c',
                    signature='gAAAAABoxKspT3bObZIX1c4Rj5n8fotSQUhl8qQHqqs9aoy3JEUbbOKD3QVYemAzZ0uNqxUy6bncn7pX_VOYoqT7dosumGmkYMcmZcy7baqB9eNVpoqvnzJcY-FzvhYm_UJCcFnAyYELQD1trdlAUc_bFnB40zCWiGTIVliQLjimjx2ch_QhwExcFQlNklWPfZA509vbMQjrUifGWQqm5YW6hN00gGV6wbro3lLMMAT0vG8Mf86Wcu1ykL_DvIYCdv5gnCjHdparV4kNxaC1feFmQ0dVNyymD2eOLbCzYdeEFvZGPM3vUqkgwggTwDqJoBy-6Md8sQGfzEI2-7OMQiR3qaYiqMVKp_zL1TbIgM2f3yEEe7KYIFvaKwqX-uR0NmKoZZAPFi5xlc4dbqZyFfFWtThy4kqtfitEa-OkM343Qr4j3cvsmmNNOFlt2nYZHgZ7Mtn0VWeCvorYpJev3JB98caMA-_MLlgeqbbUCSiitBFdyfoVx2XAIJmr2hecTqiiEn8ITCXuguwgxQFaeUokwzJ4nOGLs2xVCqSTwvSj0hzIHPNrOwdFem3-wWleA_-VOuR3-GL3hIiHBmGxqO7vqL-WxrcGSfHzfqvUv0zTeCsidIxqgT2IRezrB-BVWDnuTbZ9CKa0Lz-Uj7xF0jExBvBqqrvu39zwtGL3yTbAbzwcw0ta-EXI7_scg2ey-t9L67qEqHbjzmbGNahreBexo_EI2axTz71aDiPoFXWlAdKG8SuHJW0w1w1yWHpkTaFSQsxySsfJCNJ05tiEG7DYnS4JkthOeXfkye2NDdKqa9heVPU-NenvmZOv8neiM4Cila8xhuTLjBYNnNsR3pYJqxsa7Wuqxf5yl7meiwU5_qi5kGmuKxU8WEiq3xTjlQ11HW_1wWuaTPxj9uj6vstMWKRFYXq-GAduAL4N55xHuD1pcKLdYkbSpnvx5wa8GcC1SvPHQd7dOlxdMaWDw_AkeTvcyfHpamTS_fPvMiO_PZB9v2NE4ubHrbUmhK-G4BqM1D0uo-8zwC6wJiUX5xun6h1D6ZuPN6TBGbus5DoxAbpNp5q7w_b-7O3QEtOXUpfW4qCKyegbNWddPCpOpRKqjc_ua1lnNyccYM-l22rkydPsnkOhIItPGySZEU5XoSmBmJ0bZ70sTUUfSatU36LukZNvv_eVOSqrfKleKEYCwEqBNJPPK9XGMIi8RHbVnnBPyanCFvQKi3b978BE06TNyefA9nWZ2GKKjn7zVlUHTUffCpch7RGOZI9q4BTq5S02aoYA3h1y6nDBjU23W7S_xiaRhS5F7YdB1EZKh7xzcow0cMsrQvWbre1FqqbRdZ5HvmgO9l-AV_vAYWq42mMyJRzxf8sYLOLMM9WMlsCDFEghMo4f_82EjrxJ0VrwDV-BOmruYAXmEAEcOmIjuuzT2oxaNiX74z9BA4DaO-kZB0njB21Q2YwhdLtpwKQAXMIITVsZrXPPKinhE8cINiIYjD_1BAVnNwsYvsbq7hViNaB-ns35i-Or-jwS0MqqX1SgEX8BTAM3Wz_h-pHJr7o0f9xv4ZjwGuzUwQEUyzvrOp08kIcSnByp0gP-t1IGHrVDN5iv1cuFqmhp4AaAyuWt1pxcX7PGZ4m_CjkXxwKR8lNef1AtSRBXFg7OK1Q1VvJHlrsWPW2BzwKkcMWXN-xCywGl7Nn8Jobff40wQCqUkp6HUp_U_CP1ND2Qm1TWcK1zb4R6oRJmAW5qHOHhGd-S3aAw4OmIOD_zoUTxVkVkVVlra23IdZNkl_Yx__stOsBj1wGmQ4V2',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'top world news September 12, 2025', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab29923c81959ce13065d52636be',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/world/china/global-markets-wrapup-2pix-2025-09-12/',
                        'https://www.reuters.com/sports/records-races-rivalries-2025-09-12/',
                        'https://www.investopedia.com/5-things-to-know-before-the-stock-market-opens-september-12-2025-11808712',
                        'https://en.wikipedia.org/wiki/Worlds_Collide%3A_Las_Vegas',
                        'https://en.wikipedia.org/wiki/2025_storming_of_the_Nepalese_Parliament',
                        'https://en.wikipedia.org/wiki/2025_in_Palestine',
                        'https://havanatimes.org/news/international-news-briefs-for-friday-september-12-2025/',
                        'https://en.wikipedia.org/wiki/2025_Nepalese_Gen_Z_protests',
                        'https://en.wikipedia.org/wiki/2025_World_Wrestling_Championships_%E2%80%93_Men%27s_freestyle_125_kg',
                        'https://en.wikipedia.org/wiki/2025_in_Japan',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab29923c81959ce13065d52636be',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=3,
                part=ThinkingPart(
                    content='',
                    id='rs_08313d91a201f39c0068c4ab2c23c08195ab957ef38a2d96bd',
                    signature='gAAAAABoxKstZbcUiAU78M7zEi0_gp_EpbxebSkIgILXimN3z4WjGC9OSxdsumD39NdGZtLBKr486UgvfVqRgC6xgx8xCkuv1LItk0b1_1tiDw0-7U2dx-FvPb8AnQyuYleq8uQI_IrcrmsD1GBmmCNC6ypBAGTW9Ipfpv7XCOuKRj5w9kArdqwk2wLF8CiliOSf9ulLB1qj4eJC7NeR4iFbENaJgbjbe-H39EPyQgthn3UiXtt_Q_44m_i1C0KTq0phb3JCwxHFf9g_tNAVuCW52dEZ6mRlOlk6bYrzQ5EuF06Je37vNIYOsdb66mgrcNop3Kez-Yv0c8BdjElRarNTcOxa11AmGa4UyHB4NmZK9c4c3Nv_Mlh-TnQvikoSj0WHD3vz3sOv-2QVMTfazGvgvQHFCxnIj7zzf-QIlZBmr1BwRV5ZOTd4cjU3a9N4B2HZ9jeyy0CftUKAOzn4m3s5u4e9LqYTyzss_i7zkRQ0PXnVuqplkrV8s-C4MvDi69MocioWaVUz7ptG2R1zePXpkMsfXMCNN5F4m9zmUJSHvFJON601TVjipV9VL-5i783vQv1NmCs9D9HtqphK68AT-HpXqA2AfmLlCWenz2jFsuOCSG7rzhDwoPOF5rl4IqqeG3CcVIam2wPT1bA9_R6s-QhFLot4Q8RExR485-pUzmTJiHypkrVO0wQbUL_bh7ld_Es4cSZVYmkIkpAqmCKbiaKjJagthph6XsneBJwiQ53YGflSO5-SXu_EAikDkTcDvEfY8o5KGQalc8_XyU4HJmXm9WhI0euD2wnVipr4CgMFhbCVeHp0RG-aVOHYlbCXZuUPN1rtb1Z7gVOdmU824wiJKjnEPXPR5Fe9gwSN2dh3QKdDtI6luDOk8_kaCDpMIUoAWWGdilGLDQ96KIKeY6GFO31f_ko4whUDxjZTteTaz3S7R3u_QwEhoQ0a8ZTdU07drfIqlAjPcU7plZmMarf7kj7DPgbq1y8XhIPZINmF2e3hiIPkdFDM_RUQurpBcG3yLG6S2Uem-I4-jK0JZoZEI1wPmDYFpqfRQSXLBGU0pbq34wgIB2yjqmg_QmmaAs0FJtZUrK0uV4OBF3ATkN2wuxt_0xW5yqCiHXzLUgZrQQREPeghEZGsx5k4Tv4RTH2sQ5w0dUqkbQVehsHv2Rs_-bcw7s6TN8FT8kidQhCHEiszM5_Sx5egh25dxcC1s2Oj0HuQI8fb6xKY75vE4VxMnMHCNDRn5LPZN5n9Fi-pSDwwsnpv2S9OHXmzsVahNV9i0umg9RGgSjZE4UMjl45iQIcz7A==',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'Reuters world news September 12, 2025', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab2d62b88195942f89b36436a6ad',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=5,
                part=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/world/china/global-markets-wrapup-2pix-2025-09-12/',
                        IsStr(),
                        'https://www.bbc.co.uk/schedules/p00fzl9k/2025/03/12',
                        'https://en.wikipedia.org/wiki/2025_Arab%E2%80%93Islamic_extraordinary_summit',
                        'https://www.reuters.com/business/aerospace-defense/global-markets-september-risk-graphic-pix-2025-09-01/',
                        'https://www.sbs.com.au/ondemand/news-series/bbc-news-at-six/season-0/bbc-news-at-six-s0-ep7/2446206531714',
                        'https://www.sbs.com.au/ondemand/news-series/bbc-news-at-ten/2023/bbc-news-at-ten-120925/2446206531715',
                        'https://en.wikipedia.org/wiki/Israeli_airstrike_on_Hamas_leadership_in_Qatar',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab2d62b88195942f89b36436a6ad',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=6,
                part=ThinkingPart(
                    content='',
                    id='rs_08313d91a201f39c0068c4ab2efd748195bfc563f1f2542da3',
                    signature='gAAAAABoxKsvpbBd0CZh9aEkvM-5RUjAWiLv7rtZn7xprqhnP2ZZGO1OCL8nVF79nAUw1ZDvX2CgjDmZIzu1x-aQpbl7M_j0JAZRczokBZ-Yy6DPbWGITLzu9VEsWkq6d3yTFARwLc4PJ5UlKyv7F5obsnBxUw4T2b9FW64WeFB4ffZ9NaVkbxoFnvmxZ_fCQ5GA3aqS7ybD05lJB187_GS6ICdzz6IhM7LV4sSbUl24EInB7iq_EbNj2nhu9C93yU5R5BdY811tVd03K38OCcjuD5sUryu0vswJ3sMBVFEisEnhMWBobDnCjreUJeOPTem8amX4rsp-xMcDpARBZdfJpuxgNOgLIkaqd-meZ6NWd3cVrxWs2ydEKhy-qmjIgsp6P-ne4vd9ashIPAZtjs9HOEQEO_ohI1sMru3FwwMD7CdG0kfR1uScD3GSVbm3Q9smWozpbHmF_3O5K_9YzbpZSWCD7aX5GHTEUC49wBu4yVI9mjzso6oXECim1byT0LCOXVaczmyAFi6K_gYH38t6TLgSbJdxMOZBwqHX2_FQCK8Xe8AOVl_jUhBaLP7QOCOYfgidwe8O4wpm77ffKsXrNF7_EYLCf8YijVfkdUidjmJKmP6B9s-gxiwb5X7p1FbvI0l5z5hoZUpfcdhyLGNj0Qr2Vr4uOVANaYDtWGqzq8YQRfD2NU_qr08qd5jqAJljrVxMeCEyCQLg0pSK2hkULF9qB-zuysGFQvelksFyfeTV3CcfMo7leh4i2tbAllSfmJGjdVlJhnKttnlBhHseP6NakClQErtWtSpyFj-uPXp3PIr6mrzKmaWLc8mx8ns4hIp5P3y5QcXG5e0RcruA4fiVzZ1E5gVQk6Q45LVVyhYx98Qmy5oEV1gui7GzyUxyEozDEvyxDYACC91APrIkfWDsaFc2FQtcthkfGzzI5CMV-o5EwlgGHH_zc3k4cuVYkfRm80c5aYdCFkIcT39Of6EuORSbk8B4e0JUaHndvIKTmdWtTCkdk8ZoZQK7vF0LtIOGBBuiMX5t4bkzh6hcf5sRefh86MVAX4TisTXRNYsXWRp26B80f6ihAIOrGkGkCXiHSXv-bjKhF69IBT5R2SxDmnzTfXtXXCPsX78u-ZcpOVM9vJvdlBL7NsBtX0GyaUSdYqa_mfVfAuP7n0ZjYnTCH0yxfg==',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=7,
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'Reuters September 12, 2025 Nepal protests death toll 51 Reuters', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab2fd4308195845a2b59c42a3e30',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=8,
                part=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/business/media-telecom/death-toll-nepals-anti-graft-protests-jumps-51-2025-09-12/',
                        'https://www.reuters.com/world/asia-pacific/nepals-former-chief-justice-sushila-karki-chosen-interim-leader-presidents-2025-09-12/',
                        'https://apnews.com/article/aad7af5ddfb56b7484fa9db591297c9e',
                        'https://www.theguardian.com/world/2025/sep/12/nepal-appoints-its-first-female-pm-after-historic-week-of-deadly-protests',
                        'https://www.jagonews24.com/en/international/news/85536',
                        'https://www.aljazeera.com/news/2025/9/9/nepal-lifts-social-media-ban-after-19-killed-in-protests-report',
                        'https://www.aljazeera.com/news/2025/9/12/nepal-protest-death-toll-reaches-51-as-12500-prisoners-remain-on-the-run',
                        'https://www.washingtonpost.com/business/2025/09/08/nepal-protest-social-media-ban/57e5190a-8c8f-11f0-8260-0712daa5c125_story.html',
                        'https://www.moneycontrol.com/world/nepal-protest-death-toll-rises-to-51-article-13542486.html',
                        'https://en.wikipedia.org/wiki/2025_Nepalese_Gen_Z_protests',
                        'https://amp.dw.com/en/nepal-death-toll-in-anti-corruption-protests-rises/a-73972986',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab2fd4308195845a2b59c42a3e30',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=9,
                part=ThinkingPart(
                    content='',
                    id='rs_08313d91a201f39c0068c4ab31d8308195b6934cc50cd1c283',
                    signature='gAAAAABoxKszDd7P5gHv5ofcME_U5PomZY5Mc_dOURSCy2x5WvVW6Z6UkrvsbWdA7BiTAOGXwGLqsduuGQaGAIbkMHDN-7Ykn6ys9K8xEAgTtZ9uWGWDFz5tN-F8NSaJ2aJNuTit-ctAfvs5HVqrwHId66B7f-4XAg8sAb72YVzG40qy4jdAcEe01nPjUbWTOomi2NLSVvBfrsXYn6BF80DUnCTdKbz0z8xQzYtocKb-iiptCFvS2LpfQrYgXmj3g-GpjlgF5vKG_FkkQmkMrk-SWOUdOqIiTF2CTqUKIYFDG99JZ_q19BVqclXu-cJEWlovFtXG19AXGN-EPaNFaMLuQ_SeCKH5q8qyfVyptU0cf2J2G7_c3jCYqdqayK3NDeFVJUAaFaYawKuYnuaYktL5dh45ptU3aXstJLBvwg6eo_Ka7nCXDSwVjkaXAdxVux8zV3fLTO9oyjAbGSuL7iAt4-Nfre48731WzZXeIYEzsBY-fSDfRVuGfN0ExUtKPsXhZkUSmCeiPe6Qr7g8L_w8P_45qYkcEpp3lJ3TkhSlHu_WQZL_VbqCoDhJNQESoIVXoTWMDsMxgO25Av8F3yxQFL-BTJ8T_YyfuuWs6Sr3Lc_GMdUV-X1jvjpfmLO00N4hMWjCdpVa_hu9Fk2nofQrxZ3i3xntQn7WDUGiELR9OSsKaBB1a5n9OseNXL-V3FXJjb8K7vL1CcxjDP7m7vbnFeII9IQ9uBzLLh59k_7VUetC_pXdIjalKKSZuLWCKz2vgNk4ecFVjevPUokQQPqn_fsGwSEH2DreTe97nkGouJhDNTUcaIX35PMKOuR38LOzYSjspg-qg_zJTPSatEdub896qES_w-XoBT0DXJQ1HSv5IvOPgEehNXGAAhSAVDkaUJyJJ7PDZvctEfRFv85Vx4NB0qvPWGqMp37zt8r-WONyq9oLBUqx4plxZebLEKkWS9_pQ2VRl88Em3HNw0Vts5v9IWExMkrtm1TlriztsucnSEkkNEa--vE4UKBox5O3QrTbBJ8_HFPdhJ3Ocof01Ypp5QsR2LdMlaoBBi81OtqzhlmlIfdTaXpyY47bZP5GDIy_oaJD5y2WdqUdBtvSAtspemdzv_qKxHygmwPQ8qxR6z5mY_igyNhD7pHxphAvsePL2w-7fXwdoxntf0aLF6HcDhB8l0yheS827k38bUUN-4han5vSLDM4WgkOEyawiYL2-Yy28EnchHFT9zGhL7GG3NrWnNIMatZLyT6yfup_8D9oqMs1iLzRd9ivG0gYGXpTAYtqpr9TNmjr5qYLsuQyrUhhxdVW_wtJTIA2xmRVyAdzB3g=',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=10,
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={
                        'query': 'Reuters September 12, 2025 Qatar Israeli strike Doha Hamas leaders update Reuters',
                        'type': 'search',
                    },
                    tool_call_id='ws_08313d91a201f39c0068c4ab3324008195a9d746750af5bdce',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=11,
                part=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/world/middle-east/trump-meet-qatari-prime-minister-after-israeli-attack-doha-2025-09-12/',
                        'https://apnews.com/article/380034326d3158e8f40462c87abc36ec',
                        'https://www.theguardian.com/world/2025/sep/12/un-general-assembly-to-back-hamas-free-government-for-palestine',
                        'https://www.aljazeera.com/news/2025/9/11/un-security-council-members-slam-israel-over-strike-on-qatar',
                        'https://www.aljazeera.com/news/2025/9/9/israeli-military-says-it-has-attacked-hamas-leadership-in-qatars-doha',
                        'https://www.cnbc.com/2025/09/09/israel-hamas-qatar-doha-attack.html',
                        'https://dppa.un.org/en/mtg-sc-9992-usg-dicarlo-11-sep-2025',
                        'https://www.cbsnews.com/news/israel-attack-doha-qatar-hamas-leadership/',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab3324008195a9d746750af5bdce',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=12,
                part=ThinkingPart(
                    content='',
                    id='rs_08313d91a201f39c0068c4ab35c90081959da9b09d0e5a48a1',
                    signature='gAAAAABoxKs3A3pHeqZeRxRfLGuFGnDBXic3mPtkqRkC65cYqGZG4Ktxw7uSm1PS-3IcJop-pAlEhQImOLGb-X0yJEXJi9H_CjR9_q-8Q8JBf3IuLViwd_SSm72-A9PFe7kUkeIsIhmYfSpBqXAmPELqFpPzRqwpkWZpDCkDLiz2jDOLybXLipsXILNfYhyKqGT9Qz_4nRaYsJ9hSjG3vFFwVPHVqU7giSATDV_FxLnciOVnNyFCujPf7X_duRCYq_-lWpEWXXpatRO_qwcbcp_5DxWY4nHYUh1Zn5kNVPiW0rCYmHXFhyEwgm_6yBQAmTFv_3hVMievubKdQrTePXJdxgD7lvJXvjkpTws_KzZwmYcHVPYDmlcJQql4x_x4HLeEQOt2ZOjr90I9IC-wYGTG30ZduyuowcAjsvInjJHqOLEQQDcho80t3krAspPX62jtYIQwGEmHggE52JE33QcC9gDaY-Gfz7ToV3yxwarMXVGREk2aa2n7GqTG7R9c8GOb__xoc-2JUu95gkHLzX-NE8iy6P45PrqJ1s6WxDmiS407wHP_GlRw3CuHofcpwmaapnh6-RBWKjC22v7Gl6wkMgcRO5fVkG5vdNIDwjs6he-9k_VsdEWal8Af8Xht_EQYKSfy3UK1DpqQjOvVahtdZTopctS-Q3tRmi7paV2G8I8AwfkW6AQG9VZnj9DmMMAS8RR7vvXHRSrvAKJmTWI93PTPK02QscVJhBjohY0nEpGum5nhFE8MOEBIdw7ZsuLWs62b1ze8XVaWhXpXd3bXv3ZrLlfAvzxMBXmpjXNB3gUcClETpbKpqkAMxYtpJiHTlFrldQ1kPVJJDZCS3N15ePXIErVdEXjUuqTf8RD0gv9wistzChCA3YIwGpOYFfn4e-RZrRuNFWO4JlNc-GgQkD8iiEwHS1EuDEOqdlxKwVLw_5vSmJIvGCZO-R7OQX_V9rfh-cApMCsvn_AX6mEao_7Z_pj7LaEHL3_lJzaQlbz6toEk1R2YmCG4KcEt2uKKigXAI0s6yiYylWx9v_g0JBZHY628pYCNQVEVeSpAZiXnuTxhbiI6J1dUaUBM469J68u6ASnfOoEjZkz4UEkIPXXPw_nhXGOJKI45Wztf3MKnwftovY8X9SCtJkuP9Cqr5NfKB3Vnzlw5N7FFoq8_3YvrH4snkbJCcMahoDxVRO765md4SaFiUwlkCq2tQOH3_jNFAVj3OaRpBjbBUS2hIeQiT4OmQrKMwI7RN3J5jZhc4rmTAvSiyj9bmwGmZp5n66BEJQe-SXIfhKzqhBN-NlWFof3h2v_jaZKcCPzpLpMDZ4UGOzAiU-ThshCp6V37X85OS5xHwLtsrazCqhzjSnJabkZTDduJzteS9GIi7vKjccu8hTyINI8khhTiVMOyDtdtMBeoHRjF9HaPlHKzQGryH9QNMLQlEWU-9z3k809EUHzSgqzCW-4pvofT_JMILiFIq0H9C2Bl94ISnkskTL8Rw2qLZLAvaHgBGjfWkdrrGW83eC4=',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=13,
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={
                        'query': 'Reuters September 12, 2025 UN General Assembly votes Hamas disarmament resolution 142-10',
                        'type': 'search',
                    },
                    tool_call_id='ws_08313d91a201f39c0068c4ab378af8819598e6fdbafc72d7c7',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=14,
                part=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://apnews.com/article/6afa02f5fb4fc8823ac0a6b74d69e2d4',
                        'https://www.reuters.com/world/middle-east/un-overwhelmingly-endorses-two-state-solution-declaration-that-condemns-hamas-2025-09-12/',
                        'https://www.theguardian.com/world/2025/sep/12/un-general-assembly-to-back-hamas-free-government-for-palestine',
                        'https://www.wsls.com/news/world/2025/09/12/un-assembly-votes-overwhelmingly-to-back-two-state-solution-to-israel-palestinian-conflict/',
                        'https://www.newstimes.com/news/world/article/un-assembly-votes-overwhelmingly-to-back-21045225.php',
                        'https://www.thehour.com/news/world/article/un-assembly-votes-overwhelmingly-to-back-21045225.php',
                        'https://www.thenationalnews.com/news/us/2025/09/12/un-general-assembly-votes-in-favour-of-a-future-palestinian-state-but-without-hamas/',
                        'https://www.ksat.com/news/world/2025/09/12/un-assembly-votes-overwhelmingly-to-back-two-state-solution-to-israel-palestinian-conflict/',
                        'https://www.investing.com/news/world-news/un-overwhelmingly-endorses-declaration-on-twostate-solution-for-israel-palestinians-4237326',
                        'https://www.timesunion.com/news/world/article/un-assembly-votes-overwhelmingly-to-back-21045225.php',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab378af8819598e6fdbafc72d7c7',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=15,
                part=ThinkingPart(
                    content='',
                    id='rs_08313d91a201f39c0068c4ab39eb0881958876945512303a0b',
                    signature='gAAAAABoxKs9pCnCXetuP17cr4n_rnyui-nIi3Zg1rlysWhXDDi7Bfv7fH0HVPshe6ESao5XNuqLSRBXS-bqwAB19H7DXFGH308yY7wXZC3LiAT6QOCHoEF-kSs_oM5ctRnbTxg9McfjSS18e1umdQg2LwnquOkUjsjbk_HKydkjNjbDUeitzy4c1rJJwn14ObvIWGvMyyl-ARGbxMI7lFzE6XMo9amfpGMnHw_6WscZhF_MenyBJvcJ5-E1qJp0uOGXemvmq-e5KlmV8eNmV3lJmJb3tsEYG_jnOv0iBCK1oduxTlpkfCv4kAmrdbj__u_NB4mjIMKjYJmoIzwPrX_QPcaDQ1FaJElo9wH1_WXPlHHwKVfxkceUEgs6ZNUj6Gi5v69sIens0r1KgBRQwUSzJTrjDyjzwzaVAsFfIK0_-mgPPEmoLdGNGuYMMW003I0HbnNoR8rLHf_cJcsxeZ8M37wHm8IkiI4XVUuiy25UPna4LWUifA-C_3Vw6Gpw6Kv0sJQYnd5xcvEsYTLQceHFuTPGkzPvmg3vImk7WYVOM88XhzJHaMX-Uk-Xi-2VX52Cr-6mE0RIO6GAIP8r_iEpoy0k6mRvkFmqFIIFMJ5uk8VdOSbdTRKZ_6ySfCpvbExpe9FeJ8PFDX9zxA5BbjDYzsx53gdLRhsxzEYuwZUjfqsdaKRjssYegOm29S-XnI1w7qzZjArOyBc9L44M3TKvbNmgIiBVGTXMHl6nO7GWU3RTKF_lx5wZTS_eud19UVUK-JpmJvyEWcBorg0fSzSruFoqDjT_jQbjb8N7XO9ppphafpLV29eU-7EnW-GKJ1-M3ejov7fFFwrjwTErZCvyhqBqvQEv9Q7f_WPa7rReCbMQ1sojEC1g1b9FPiRFNiN5AOWFm0GKxLTHZxUZttNq41Uz7KNpDioG7bJyEqhZQl2wOxm45Lmc-WBmA02S7NK6oSYucgbNJof60uWHVA2AaVqwqD6k_8E76lNllD2hCGUhwmeNa3l08UUSIE03TUkHsXCWQ7ujsjfJqueP7rcgHYWk-ZnhfWdLXcgvpj-CEus68FcR-Dq--D0JFZ9-uqEg83hF9-xPiYJ3w8C0ltR_cp3iJlRAlM5YNLV3x_pyWdS5tCJdy7RGgXrj6GVl87T8MGKCllHq7RkJ0jMP03jN5hxVi6V49XWyEP5Wo6MpMHGr5DSATCEvgnPToe5M2oTwakyLoHy6D2Lw1ir_i1YWqA-k4Tfg4RL7Ngb-Vzj7QbrXwjlobKzgWY6vzsgNvBAAFNyKHUV43ZVHzfSeNqznZaGgXzcGPjj_otS6qj9jwYHhhZU8S9DditOAPt2_J5G9kczB4xdXhrefGLn0BQV-cQ3f7wkVnnwD3mfu1HalHmAy9YnE5k78Cx-U9YTX0ZV6f8o0PUZuw_akwKKESJZT8SoI9RqpO7_Ios2QJvoiG35kLXmkvvUJCowkfiwiN5J59sUnigUCvqFSCs7gH7bCE-pDkfrGd9sam_KXNZMQ7WiyD-u8sbzfRlm9DLR_3VaYnXhAeQpJQVrxYrdY1INmOWqA1rSWcNJCvtVo3zuYn7F4P1mq4N-DcAy2CfchRLtU4miREI1h9CKi4OoAXtTUfVipdmPb-p_UrHndvblGirSJxI_U05Pbz5G13Cg61rPGiPOzphFFcBFF9fctpCwQRG9xMdLP7MOM3Yl1yM-fU2iA6H_7poQzLKgw46lPUQC95kH_466592T3cjHlSQzLPzpmkX8IaSwiLUSLC-s-_G77GHjVfkkGwHiLmZu1HfDy5yAsvnGvXwNqOcSMWC1lZwKTfIb0CgMpMjG4K939pnhMoPDiJOl_LIBT0Of4gLIREdNc4hbKOX6MHXc7it-IF8cvKpMI5xWtmNpxwWsBh-BwK1m8NiW3pla6dtWFhknAToKDnZUz6b37-m9vzOttKVCdTtMFvHDqVw8NYJQxk7kt84sWLjvjiBEUdfM6IginxDNy4jt_gcidyHOy8_BF2SLip5L4xAwlcjwj6aV0dNKZLugxogAxgADJIo3fj3rmoxiPZjXDKPxCSUCG83kkKKNiWl-5_4PX20ffXhMvLBHzwarcRB3ui1dFqu2dlm8GZhdVniGLIP-fIN8zhuoj42vLDFm6OaE_4wEuHl0gMuIcXrTjudK21bv7zIXdOFI_lOf4RArvfrPc_oGwoUgPRIQtTEsjnqYMIvK9mXQhcuACRsWygSbwHglRy7cyhAJYrsLE2gb1v96dogG16gDK0DC816wzaRzXgnfIQIzI19zAPRtrdeVJxjqadTd7RrvHOPDwoB824-WGajvct3YCf9n_uE5wz5YvH7rbEGdkf-C3wXtybFYzn592cbcXEaumhAwyQIewYFpBHBDWjWeIK_Ms84DXFglc3A3Sendy6hZU1RwHzLsonPXdsSEAO2Bema0aWU9bzSuXJrrgtBW-KxWjodhv3g==',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=16,
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'September 12, 2025 major earthquake', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab3dd990819593e473934de5c75f',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=17,
                part=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://nypost.com/2025/09/12/world-news/ukraine-strikes-key-russian-oil-terminal-in-overnight-attack/',
                        'https://www.reuters.com/world/europe/russia-belarus-start-zapad-2025-joint-military-exercise-2025-09-12/',
                        'https://www.reuters.com/world/europe/kremlin-says-russia-ukraine-talks-are-paused-accuses-europe-hindering-them-2025-09-12/',
                        'https://en.wikipedia.org/wiki/Typhoon_Tapah_%282025%29',
                        'https://www.aljazeera.com/news/2025/9/12/russia-ukraine-war-list-of-key-events-day-1296',
                        'https://en.wikipedia.org/wiki/Tropical_Storm_Peipah_%282025%29',
                        'https://en.wikipedia.org/wiki/2025_Myanmar_earthquake',
                        'https://en.wikipedia.org/wiki/2025_Afghanistan_earthquake',
                        'https://en.wikipedia.org/wiki/2025_Kamchatka_earthquake',
                        'https://en.wikipedia.org/wiki/2025_Drake_Passage_earthquake',
                        'https://www.thehansindia.com/news/international/deadly-earthquake-in-afghanistan-tremors-in-argentina-alaska-september-2025-1003393',
                        'https://public.wmo.int/media/news/hurricane-fiona-hits-caribbean-and-canada-typhoon-nanmadol-hits-japan',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab3dd990819593e473934de5c75f',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=18,
                part=ThinkingPart(
                    content='',
                    id='rs_08313d91a201f39c0068c4ab3fb0b8819581587a21cba58542',
                    signature='gAAAAABoxKtDkEeBkvBFiIw4clb4bv6W9I2QEU68uc2ZnblfIjBLq8sumx0B6Zw_f0V2DrQ0tfwJPUOZnj-lQ_QhxnfDRP3-UQqfcDmeSeCTyvwB7czT7NusncgiF9XNWXMr8lc1KCmZJKEBLJIuAKdlBWR3aK67Q9_IQfEpvtkAI9Iu3nitNDFjjdSpGcZ5W2EaNkqkcDQZ1jnlal-tGBryXRmU66knOlWJ7Afk4Qmw2vpLnSLI24HNXiJhOQdaGoousI9A7VK_XvHvOCYOpwquz_keU59hEPxcO1eDrLuoWABm8HvECwOhMQWCViCOeQ-E3iIP1kNhxIbJBXdEMzC6YFwkjI0C7lQmS92YPmNRveBkxBw0lVZrTeHMCV5ksgkif1E1EdQ5rTuLTWvLiw4OZzCnkN1fbVnEJ1IEWtk2flyjyea041SnWSAADpV5fEuTYtD4WX7NEfN2dDvmXgv-vj-cqo_-xkoV0-SC9t0HqUv_R1UmDE93arrWJjayK-OrSvEQkXu1Zz7dDeSx7EIBKhe8G1AM_tHiFREXNau1fkrq9e4O1-G60pZdFJ8opu_Ybf8Odb-ko237muU5VKJg_CgdVxhGlAFeYWN2WpyYm_VWLpAW7bB9egHR3S8JMwTrfKGSvDSIfG8i-_cRYls5tOi8FNpKXJJIyMX44DmqP600jmj7kFYrutf1RcLFQZtT0SgnBPOYIlvLg13M3V16iYZ1hh7AR95jUvjI-GnWysW02mUHTC_Ae_VqIxn7Zz6njbd0NK4LJdWE-jQj_xhG_w3QGoIaI7ALK6HqzXqR4kTKEiBR2QBj6mYJ6rM6N4Y-UI1L-ep_mWJTvvhZxou4GOBu68MmFOLrNQ6cHW3MS1bVEX4S7yiwRH9mV--TFjhdIdIcP9rDrWjKrtMMSm3Zzl_GmxCO7kaiA-_XzUC7jk-dEx3SaU2uiAw5KAkSqOHTFUWGjGOOHESDxm8JP3rZjup1NHYBszlxGnt2Ux2cnKBm1k-z3Px6_VGqxpcs-CcZwsuHncu_ZxD1-Z34ivmgeB4dTJRi85d2DJJ8Rk6PQxNyh_hBvVHKlMH34_2JPOiQ7gUQzlLH8c68bNsbwRnU6ebOfvU903cQ0dgpMZrAuN-R6RDWIul3kjAJsEnnCzeP6fVdA1b5i8QHGOaB4VG0fJxJuN8NiDplS2awyupwMfQbqatai24C6acczmz4rkufHw5jORPmmRIjlTXmwLGla9xz39HyfntBoAm0tKy8uahYWOGJCftmDr2nKD7k3-JufRnZaPZwKFGd2P6zEyWN7_cZFnYERKlPKf3k0IBCix0_QVu1Gk6_98aANFVkZO73e0YjBPdpNMgQ5Q6KhC3rk6ZEc1GzcTX6OJNVNujPt6cybO9CsS-BBGRf7QNPAuXvVlCrTmUDMSJclPZE-IXM64oOolU3cEOZUevPKMztgN7Bsx6kPnJTZW6ANL2rvreQO5xVX2U5ob9rg22USQO4YmEQn_JAgx-PjLg9BTbqE-vBHIuSlhON54LeAr--7cQipOGp2yisvodl6ZnYlA3tKaTiUcWlkdE4HMXn5zM57dqHUPrNCci9FdXsay2E-buXKnfq3Fllk_6SPB8cXvz60Pg4gufkw855DB4aoF3HCQSR5UFaQQ_4Zb3WiI4Ic4kQl3HVIGSDel9KmReyvJTWg3eiaZp25pXOXqR7OADu7LBb57CW6a2xkF-Xd8UWhCWR-HbZuPO6o3OaKHQ4jzC0Q1xSDrdXGHgnDOL6SCiUfSVd-bjkaAuiCLuwDz-Ac_CRm1X5rhJjbTN7-eHaCEhCqOUg9_WtMM2B8nXyx-pjNbxF5R43CmIHiuVxyI7hEbwe1U-zT7Pzph2dfPNpPl4agolA9e9Ucqtl7A8pkGPN0NQAzHkDlzNGS7soVnzKBB_60t9ZTNWa-ndxa0TFXAAQjNrFDLCezb6S67pL9Z3B39aSrroL8oHuHEn9PJARoBZpW9PDbjsm-4KEZf2YE-rMDPuEnCLKNqnAM0cQTGgEXm8E4O4cK58IbYlN44LWB02EiuidX-A1VxH53Ty_DtXAeIRpPhcNkOp-mcCniiidYzpkca-lZGXWVsOyS3U3fUgZBmyqyFGPX66u1R0k3W_vn6KMldvEsCrZ_bqo2phg_8i4kimHZLJf3ZGu1AKfl8T5xCDa-I54O-MBQP4JxsKxZhNtknUofzLoAeZ8RkB_MUjIprTiVvs=',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=19,
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'DW September 12, 2025 top stories world', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab4308288195bff2b3316b62cb4f',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=20,
                part=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://en.wikipedia.org/wiki/September_2025_Israeli_attacks_in_Yemen',
                        'https://fmep.org/resource/top-news-analysis-on-israel-palestine-september-6-12-2025/',
                        'https://www.sbs.com.au/ondemand/news-series/france-24-international-news/season-0/france-24-international-news-s0-ep7/2446206531749',
                        'https://www.sbs.com.au/ondemand/news-series/france-24-french-news/season-0/france-24-french-news-s0-ep7/2446206531754',
                        'https://en.wikipedia.org/wiki/Israeli_airstrike_on_Hamas_leadership_in_Qatar',
                        'https://en.wikipedia.org/wiki/2025_in_Palestine',
                        'https://en.wikipedia.org/wiki/Assassination_of_Anas_Al-Sharif',
                        'https://www.sbs.com.au/ondemand/news-series/spanish-news-france-24/spanish-news-france-24-2025/spanish-news-france-24-120925/2446206531904',
                        'https://tv.apple.com/au/episode/friday-september-12-2025/umc.cmc.7hlnpg91k6gwgl8hnmn4vayn4',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab4308288195bff2b3316b62cb4f',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=21,
                part=ThinkingPart(
                    content='',
                    id='rs_08313d91a201f39c0068c4ab452f3c8195bf1084b78906cb0d',
                    signature='gAAAAABoxKtORYIZjmQ1Ro_cqQ2taAbm20J__IL49064r4j5FCBUj11JhBsq5FtQcIBwl0HrT-MYYtsAQ9j2nxZzDHwNLMBIAtXQtKAMWViqcB3B3oaHdhfJYRk6eK88TjhSZPitNtxnWsDE9ZbjCdybVqsWLwPiAdASb7IxALYkF_L9WSKIrMO_0I0pjBmkX-rKhL2gjEV3yS3jg34LQ9INdcIjtNHeL2fnhe51i8f5RywMMcyehFTJu9jX10rBIBnLRjPBM1t1UQ4_XLrU5LVv_KxaBs14WJimLFgTuLI3rVCktXk9aMegWi-zIXxnZ3x1nVpyx_ZmNVvmZudd2q2iIDSSkNafEApVuhXwvUSsQD4hPMUK-1r8HXvXie1jZjpE3cH8o5ocHHDenKKDXsxZnj0iaNaKvsbYqak19kyEjrXtfNF-1I-Dv1ke3dt073GL7UrC-u1y2-DxI1SRIkIN4HtlARshnqSxdrvdJoPdBlEZGLs9_qKzCCFCqlkxX09bgRrZsCPoQelJ0VyuhLDNyutxJE4gwj64vNizLmjch4K3jOW2YvgQyMSddMZAtF87XNNmp5MVYyCQUn_EpZj91aZSKFdGgx9CmfKHKBwUQawyeNSZvMU4IY7nMOcGVnjAoLkhFhJVq5AxSpfpcOadH8Twy9ISKM_9BhjyGgrSnbfU3na_8ixeHlDoEes9Dwp0NbGYPBLnvTyvM99ONXDSvsaBzR8mMTNMJ9tp77aA5x2McrVXYVX-cqflONp-2mZ6xGXUm4uxKbt9hOuTRrCCCR8WlyYkZnkVFeNE-dEpeMJCgbNEE4ik7BtiQoyq9nXD5v_FxPFgZyx5wnRa2__FcbZ_Hl-fD3IHTeHRZItXaclFCBJhBU1K2fXe9HJM-Iwo9AI2Cz1T5Yql2VuWQwJ1CidiZWuCjd-uCe16v-gq0R7hUjO0LyVygOZt9JnFPLW3nzItbIgJiIN92wcOGKD-pM2ROCCX2lu1xG5NmlOfhjG1EVhZu4GAAgWgHPBSEndzgtrzZ3SVgBjjtbqMiq2hNmtX1k5BW_52OBqjQ4hdYYpj9Vb2swnhKomPI9de80nIFXGcwztt_xcNu7KmyE-9-rrMGPxpE-_fs3py-ZhtgyrJbEnsQ0aVLhrOW66CS8VnRCs9Nn8C_5pU3Ihqh352eG477TirjmfehVak-sO-w9mqAfec0eItoW23hcSjYRTMIJv8hUaU32HPSSEPZt6-H3H2Mstyp6xynGTPRoX_gVsYSDrYhJ5ziVmwpAhsx_KM2pZkbo-MK9JdLIkQzJQv2z6TBWRCGUkqKb9FIIZfwlF9o9ww90fMW9S955Rvs3hiWIvvAiCtod9r2QPz9okjn7qbB35geEBVfufyVRJR1d31YYF3j_VqbYb_Izs48waeukyw9Ps9jE-34ye8Oas0ai3ShpU6QxTXZLmVTO5lH2-qSoo379GWhFeSThyjMVcpSLDvTPu6NYBXq47oYa0v5lf5mKgDXuefqN_bBa5frNUlU5krIhxj09mSwG7pfqG8b9hvICnJMAiQSGKBKZlbSgXTL_bACXN4jmcwd_3RUBK2kR4kZD2Q-pG4NxeXdmA4CvJHQsH1hTiRnmwvEA8jAUUx1Nf-cmNIzYscgnb9T4guZxZForCNIH3FnTDaaiPPMRSdK7D5yvtgcXckZ4GX_WKo1RjWQjVRIrAFeKPl4qKAZUk7Nbni2S_10xuMPqtuhar9UTOz0tg33C0yEoKU0pm-0wlJR486MXUOlkcL19paxTxzGbxsXgO6SdOI1NiMZ6Y6o-flmYUzAF_8qODuaPVAKspNiZcns0mxUaBtW1LArOfChkB-vV38d915pQXhk4qF7Tfi0td528vlBra-dO1eY98bs3yRsG5P_CoJNGNorSSPgCGnYG_c2NVO_JRjLZJ2FXRQ1pFyzBD9yPU1HosaHNF5Ib9FPSNhMTVtErIRaIhpjEUdkJryjkvPvMkqMIkqpHomXMns8jyQZKCphTz3QTMw9QcvigzwYTIVb4WSrW9QowlHjdljbr8vD3i9V20zDKxi1F7wwf9-dtWHF-AxdN_kSqSI2EbWeP3lGp8GkDHDzi8B0-PL1Ty9EAPcj_78DIvWCqLxQSY_KWhlJNaumYhGgGsUOWGlKPktyGoEHNawax6gk0pYI77ZZ9xxIOOTmwD5M2O7AsDCzKUroKFx3dIVFRHJXFVMMZ4lYbjFG7IkhxY7IaJdH7u1fhJjgBPdnr1K9UVnadu79-B6-FJO2U1dmBimPHOeJ71GPwthpEEgGd8PKVFH7VTh27KgUyDE_fyiHkvUzZGOXZsmSb0M4Icu_C6Ix7aqvlUy4fjSpxsj54SKrivh1C-G5LwA9F35x2bVbRXAw6Ut5VnOsbzotRBVJ3zzj8lQWs2tmxmSVLWOeIWyxuv_OMUmRXPbS4G8hjiMXt4d3PZM5w0PYsves9HxiyL7QL8VNKty_jyCl1Pp0VgDaYOiMccIoLgogcWlGJS8IeCTYzKsIyAyP64MgRXEBPnZBPRp8QxzhOVqPy30CdEKf9rBsFIInpb5Y0sOK6nGEiqvJDI7GuH7sfMwIHKE5UV5gaftd1ZaENi8CtGGz1H6zgTefCppmRLtDVqicuKOHre2Qs0wGkIi-nQfwcDkRuz-jD57cEjtMSXGkMLwjXSqaSozEg59_6IC2VqdIo4S01tILQ8-k964VlTmED0xXrV0sHa65h-Y_sqfCjU7IFAmSbjay2x98m4ulOKi-4ulWV8DnZuF1c2UrSPURh_z49tlScxtUIgL2IcKyBNK3Qb8rKLhRQWNsmAwNqRcEQ5k9Q0xcIA6jjKln-ma_LIuOQaQ4NE8aiRhF-kqxV_45FTRm7RgY9ccpSMFM_ZgM6b6Hsw4VE4HXbCESEUCnckl_54cRaeQ9767zIo5GaRHDqiMMS14b3hwMquwx76QcypnhjQtt1W7rPnJ_5Vl1ASZIJuwtUaFJFxvH2qTHft64lqpvvf94labXJkM2YBiUT7zxMYXve502cD6QwqV5ltP6RxYMcS-ez9E50553zmseWM5o9ZvzqMTDepCwE4o8Nlpku1wUkfIizaOtzRe__6bVRF_JTV-i1UnE1mw9UBC5TW1stKpkaW-bxNf8vN8jS-ND_fWfkjO7-MO4uya7sTv4zRL0YBnkwtRT581NYeAD64df8n56A5YtGRafHqkTCVqdi_GERdI8UyCK9SwTRXbwgOUd-IpvG7uTb0KPr_t_Uk6JWzYlydNsjOGDSV7tXPkVhINQJRBtaYRpIFcUafEnY-Jn43oWbatLO8XCLbSh4emcCdAEZUOw4-gq6tiHyNe8b5OxffINQy0YLTbYyw4rxNCdczb1kV8BHhdJ-BOGlBHskQvz8OgluYAa1p5lJzyazVobxaq-cJhI2CbkKOxL_jAAgcHk1T6jkzoVO3BqUAQsQGe6hDatLyHfVqBiCU0_4OQBHvTOl9Pw7EmGyjaNf1Vv5r_Jh1mk04gA0cysvCMYw0ObvxULyXzohnqCDSM9iJ36CUhxeBJyNRu86GhsqVsjxmLeIIG2VkjOQBR0pSv0Uo-co7G1-lZ8xhlntnlnLBEYGS-ffnT-GpQDckF9DLB8_gKjH5XPjhNdf5fwbOk_1aY9EH__e4ghgt6YqG1QRAOO37YPkN20S4nVD5oGGaR7nLnIoC3b4moFW2vgbGZjW8ElSBDd6kQWSGFnKG5zu3zXZNAUjYhyucqX4Y50BnVpO2kTYTmqhXc2HBKZHdMJBZ9QM7ejdXUchLpmZhpclgjPWdFkmuZoTF59A1CdCp9eUFOd9234-T-lYRQ6KkervCWdMyfMect7qQJV-iCSMvzIQCwmxGlFdUoYzX8ANZkco61EsiA2T8HyvOsuYsuSVkaO_KHrqMwAsCSq-RrG06TMfLavIqBkbZlAmzzzjLTmD9Xf6XTMmwT6DpriJdvvZOfZXBuJNlDVQtsM14GuwWBYpc7aw0YktDPcYDBVVQvyaNCIcf7zUQMNxiMATDCVrVwwzBAR9u2Q3aTdxprfcCG9b7lFLzLrNpXLzI7GXZ7UUKCr-BAeF4BmlmDdCyVCqT5RWqpCG13swVP6sH7sfVxw72ClVLJzfSlG80C5pp_hS-Jpp9k84atI9cqljBD5CXVoKv2UKZRLnNGJn4HdkmONXQ480o2sEIa7ZXnT5xcJTL8zzsS5zZPjE7pJ9qxVzRRvuCrQBqLiLxsXVOuPXiz0LWCAgVMojZgavvoey2JoN6g1CuinF6SzNM9tdSIw90gq5iydeLpV2qbhDegg2e761FLwBHx4kfy1qIjOQr01aGf0v8r1pCR0Q6_RqPq5J5U3Tnc4rowdkjKY9NrfpqY2gHMPeBdPsrcAZFYLiY4YmZozUcHvLYo9XtxEOUtLBDdBkB-PAT2Tt5x_D8rgGlYnPGhrTIaBFQVwmLj1iGaznVMWD-HXP1omDZhTNts8P4W2M56NMPn0Pea6XDbq7CHrD4fQmDtdKhlMOP4PQK-7wh2L1DJEo27ZWKR4KwYheKonuZZPOopTjqPjNrtvkw4Jq9YTWcByIMzYMxYqF5MLNMl24wkKuu-xBB7ssV_-gC93bFvsPqL0uTsRs9Z6Pto4AKkWiQqvc-EqpPIbQ==',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=22,
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={
                        'query': 'Reuters Qatar to host emergency Arab Islamic summit after Israeli strike in Doha September 2025',
                        'type': 'search',
                    },
                    tool_call_id='ws_08313d91a201f39c0068c4ab4e23a88195a39e83d44d538ad9',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=23,
                part=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/world/middle-east/qatar-pm-meet-top-us-officials-friday-after-israeli-attack-doha-2025-09-11/',
                        'https://apnews.com/article/380034326d3158e8f40462c87abc36ec',
                        'https://www.theguardian.com/world/2025/sep/12/un-general-assembly-to-back-hamas-free-government-for-palestine',
                        'https://www.thenationalnews.com/news/mena/2025/09/11/benjamin-netanyahu-reckless-qatar-says-after-israeli-leader-accuses-doha-of-harbouring-hamas/',
                        'https://en.wikipedia.org/wiki/2025_Arab%E2%80%93Islamic_extraordinary_summit',
                        'https://www.euronews.com/2025/09/11/qatar-says-israeli-air-strike-on-hamas-members-in-doha-killed-any-hope-for-hostage-release',
                        'https://www.aljazeera.com/news/2025/9/11/qatar-holds-funeral-for-victims-of-israeli-attack-amid-regional-solidarity',
                        'https://www.aljazeera.com/news/2025/9/10/gulf-region-at-risk-qatar-seeks-collective-response-to-israeli-attack',
                        'https://saudigazette.com.sa/article/654886/World/Qatar-to-host-emergency-Arab-Islamic-summit-after-Israeli-attack-on-Doha',
                        'https://dailytimes.com.pk/1367098/doha-to-host-urgent-arab-islamic-summit-after-israeli-strike/',
                        'https://www.hurriyetdailynews.com/qatar-signals-collective-response-from-region-to-israels-strike-213488',
                        'https://english.aawsat.com/arab-world/5185339-doha-host-emergency-arab-islamic-summit-after-israeli-strike',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab4e23a88195a39e83d44d538ad9',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=24,
                part=ThinkingPart(
                    content='',
                    id='rs_08313d91a201f39c0068c4ab506d588195b94c9de5f5e30f4d',
                    signature='gAAAAABoxKtVzuQn3NjqOMhVpXMk3LeJWf-aL8_Sd7_TMGz8JTmM0be8STmragAQDlzCooRITaRV-aOfKWWgeEouDr9ryYSKd65A-71Zc4Nv_uuD7Nnger-4-V7ODgpQonSMsv67mxlA1NICRZaKsGnlDeuJkKQwqLQXxHZqfTLTExp56a-xN-vEIjb2ubRQdMfaP6mdD6Xpfebr81wDnmb9HcZ92oZz7mNaH6VIbLb2Q96j0GdAmTXgSPBZq75YkLkLruQ2s-SWiNarKH6Dih9ZiJPBGGupDcLoSXyKUkw1oKSMH3u4MxMZJRiD9rw_kPvMqWHqdcbWxtML_Kg3OTbQ50kPdmZPdpMxG3H8wvId5y7LibxNH5TNwfQVAIyMmGT_C0ukpjGiF81mT8a8e56h5ez8oJKSXVaKpU2D1pAVFYrGehWCqKljEhg4l4jwHjlpDjiIGqO-Ur1gJjmm4dQJLkbMGFgCO6AsDOuGy8YAXENUt-7J9izTNwShMpxhzF7ijDItEPUalInvl7EhubBsSs9gW0fm9J2xrExjncVeseqiiqIG4Z5ZURVyQCfdEfZ9hcR_b7aAX8GQ9vURjZeWFOpBpHCtXCRS_tXu3yH2FL6qaY8uSathgR0eriwQ4sKsj8FA5mvYZiXqB2W2KtoVzbNtuWM57dALPeUnGPJUzKKOQuh0PmwRqzf98IRg4_lDm6j3qH5HhwwGG_iUJdIT5Lhr2tLFVhNY8Cc8NBsJgNSS5a3pgRo-rLj_OIhyB5uUpto7Yc8Zw-VUZCbTTltOXQ2smO9vPiB2PMG1Ngx3oPaYTGCexWNxF0Yd-dW_JsK8a5cpcvija9mPsj5pkpXpFdoXViKHbGE5RiR1o1T1drhjp-DnpNUpYb8wlUvClPZTNmrAi1RUcGXQTdWeYHA0ZPM-0JmUhRSB4lIrT1DfEIBQqHl9A8PIXfhGCN1B-li3gEaXO0XrL0P2enFbwzFFCxQAjsvIs3DRhmE4im4l06u0-GrO3P8iZt-7bKyhvIBMGslVcUdtwI7NZU8GRXodQLdsVDciHYERI1YN39VwWVRX7Dkkotuo8hqf_Nv2dpNyn_17PKrULftrpGdGnbaRc1iw97jZ0yyXytpHwexuolI4KMrNE64RrMolOhR1COjKSMus8Yw9dORwn9RjP7sHMdIckAKi6rAEEdSuCNcwyCOlsoF8u2Zi8wfdkmXWxq5NMoRGLA2Fg97AttpwzEL1Q5UfyPkg8UkSE6OD5dQMIjqPPOrkzVTpjp1mi0zdOsgCrj1B35M4WhKMJq9x-ntvaIyZhm-oRlXor-QNoT5px-sPQKjbM1Li4sNol1zaJDNxp-ALKcR46zz3ERcEN9NLcCSnldXT8qgLccqiLuIcKmuagWpERUVjexfs4KnhmTtQlWgaiwWF7mHaXu22vNWxL-hfuau_Qeu0RVVjaVPAnlaLtJDfVi8NxSVDG5AIvi5h6jRvsH7FbiuTxNXJ8Wwj_OrgISFiEal7vVShQERXVKNT7J9DtfmSu_P70RrHSsxUKmZkhJrs5IoH4yp8V36tnQgG0w2PuOl65C1e-r5IGCQ_Lv4mjBWjjlr4UuupJ861PrBexpSd5tDQaS-9AnLF1YXE3Zc68cjYruafHvFN1koxY2cet-TjAROrcuY2kgjs6nFunhiVQVfCjx0lGONrXmBOBVaYKWteRa-bUzwgs-RSvpUmnTLQTEWHQbp-TGgM_lPPT3XmLY-g5yRMcUgbK0bZrbVGCE_jmnjaCbw1bKSbA8WbzJhSl_YGWz-dVFVMWlocVod51_aGEzKN8S68bvid73O1NmdT08Fp9w7bZk-Ui32fDMIvxQkZm5eYRkce_cXkhW8AuyUCrh5euuHv8KdnWk5CAjCY4iI_ZwI3GInWYHXiCbcUua1sSYHbxPDRPRLGvRs-Syh1BdSGudRWqraNQrvAvbM6F840_pDnFkf5ZWBz_yP7b4v56lZEOvwbkEDTTuWJacWiw4b-60X7H9PsPoBMGlB4GHh1T0kOTb7EmUJSG51k3QpBeNqEmZZVcqm9Eo4SS0DsrlNuYzv75dfrHwvrTNF9XXns1P3fH-1vsNifLzYibxQkZccC2bxfbDnA5JIa2GXnUPDtFhY6QhxyrVXYbNaBwzHZfix7IuFoKtCgLE5O_pNfUS84yWZ_SpzLqi5odYMYIUbUjgPXPvP65ERpn4MbUxGwT9k_5qXcmUE8iJ7reafjCCCfwNuKh9dYKCgxtVIY04ry-cwZreocFMB3wQDrRaMIEBdVYRc0rAZpvC3MdDPmV85g1cR73UJLmiACsegnyQmSTaYRB4rV1LGwY6xLhID_ikcZ2veAT-LXenMUUXzBCFU9SDCT2Ss3BZ0hFc7EvxwOnqaA5uVYnW3J-iirL5GMgfhrVDFVEOPpFCChHlJAUoPgaDYn8SWhV7Ekx1Uo7aM7y9xclRDm4dciaN8MwZ6-GRf1OIvGIH0f4AUoDaMvVCENAXTAvWl7oJRWl2OPyox0NBt-DTqZy0iXqH3DMnIinlQIAlZGlFR1FZMCbqvneo4P-r1BzrKWVyr9loUR-loFBE-Wot_jhiyPlcHpk5P935dO1gsPQ9h0aFfLzqrU5vLbmnGGrCcPabN9rCfOe1pjWwBtabRtFB5MIXf9iP4H6PNu1G18OhTBE6LcJp-b9daMJubtumQywt3DPym1w0L3MPblv3LyyO7REUTuNT0EN3PgN0VM3WtB4j626i43Fcz1e6XHU0Ull8_M',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=25,
                part=TextPart(
                    content='Here are three of the', id='msg_08313d91a201f39c0068c4ab5555f08195b138210c2a22c5ae'
                ),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' biggest global stories as')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' of Friday, September')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' 12, ')),
            PartDeltaEvent(
                index=25,
                delta=TextPartDelta(
                    content_delta="""\
2025:

-\
"""
                ),
            ),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' Nepal crisis: After')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' days of deadly anti')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta='-corruption protests,')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' Nepal appointed former chief')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' justice Sushila')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' Karki as interim')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' prime minister. Police say')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' the death toll has')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' risen to 51')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=', with more than')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' 1,300')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' injured. Curf')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta='ews remain in place')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' and thousands of escaped')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' prisoners are still at')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' large. ')),
            PartDeltaEvent(
                index=25,
                delta=TextPartDelta(
                    content_delta='([reuters.com](https://www.reuters.com/world/asia-pacific/nepals-former-chief-justice-sushila-karki-chosen-interim-leader-presidents-2025-09-12/?utm_source=openai)'
                ),
            ),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(
                index=25,
                delta=TextPartDelta(
                    content_delta="""\


- UN General Assembly\
"""
                ),
            ),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' vote: The UNGA')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' nonbinding resolution laying out')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' steps toward a two‑')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta='state solution and calling for')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' a Hamas‑free')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' Palestinian government, passing ')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta='142–10 with ')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta='12 abstentions. ')),
            PartDeltaEvent(
                index=25,
                delta=TextPartDelta(
                    content_delta='([reuters.com](https://www.reuters.com/world/middle-east/un-overwhelmingly-endorses-two-state-solution-declaration-that-condemns-hamas-2025-09-12/?utm_source=openai)'
                ),
            ),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(
                index=25,
                delta=TextPartDelta(
                    content_delta="""\


- Fallout from Israeli\
"""
                ),
            ),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' strike in Qatar:')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' Diplomatic shockwaves continue after Israel')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' strike in')),
            PartDeltaEvent(
                index=25,
                delta=TextPartDelta(content_delta=IsStr()),
            ),
            PartDeltaEvent(
                index=25, delta=TextPartDelta(content_delta=' will host an emergency Arab‑Islamic summit Sept. 14–15. ')
            ),
            PartDeltaEvent(
                index=25,
                delta=TextPartDelta(
                    content_delta='([reuters.com](https://www.reuters.com/world/middle-east/trump-meet-qatari-prime-minister-after-israeli-attack-doha-2025-09-12/?utm_source=openai))'
                ),
            ),
            PartDeltaEvent(
                index=25,
                delta=TextPartDelta(
                    content_delta="""\


Want a version focused\
"""
                ),
            ),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' on business/markets')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' or U.S.')),
            PartDeltaEvent(index=25, delta=TextPartDelta(content_delta=' angles?')),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'top world news September 12, 2025', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab29923c81959ce13065d52636be',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/world/china/global-markets-wrapup-2pix-2025-09-12/',
                        'https://www.reuters.com/sports/records-races-rivalries-2025-09-12/',
                        'https://www.investopedia.com/5-things-to-know-before-the-stock-market-opens-september-12-2025-11808712',
                        'https://en.wikipedia.org/wiki/Worlds_Collide%3A_Las_Vegas',
                        'https://en.wikipedia.org/wiki/2025_storming_of_the_Nepalese_Parliament',
                        'https://en.wikipedia.org/wiki/2025_in_Palestine',
                        'https://havanatimes.org/news/international-news-briefs-for-friday-september-12-2025/',
                        'https://en.wikipedia.org/wiki/2025_Nepalese_Gen_Z_protests',
                        'https://en.wikipedia.org/wiki/2025_World_Wrestling_Championships_%E2%80%93_Men%27s_freestyle_125_kg',
                        'https://en.wikipedia.org/wiki/2025_in_Japan',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab29923c81959ce13065d52636be',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'Reuters world news September 12, 2025', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab2d62b88195942f89b36436a6ad',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/world/china/global-markets-wrapup-2pix-2025-09-12/',
                        IsStr(),
                        'https://www.bbc.co.uk/schedules/p00fzl9k/2025/03/12',
                        'https://en.wikipedia.org/wiki/2025_Arab%E2%80%93Islamic_extraordinary_summit',
                        'https://www.reuters.com/business/aerospace-defense/global-markets-september-risk-graphic-pix-2025-09-01/',
                        'https://www.sbs.com.au/ondemand/news-series/bbc-news-at-six/season-0/bbc-news-at-six-s0-ep7/2446206531714',
                        'https://www.sbs.com.au/ondemand/news-series/bbc-news-at-ten/2023/bbc-news-at-ten-120925/2446206531715',
                        'https://en.wikipedia.org/wiki/Israeli_airstrike_on_Hamas_leadership_in_Qatar',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab2d62b88195942f89b36436a6ad',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'Reuters September 12, 2025 Nepal protests death toll 51 Reuters', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab2fd4308195845a2b59c42a3e30',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/business/media-telecom/death-toll-nepals-anti-graft-protests-jumps-51-2025-09-12/',
                        'https://www.reuters.com/world/asia-pacific/nepals-former-chief-justice-sushila-karki-chosen-interim-leader-presidents-2025-09-12/',
                        'https://apnews.com/article/aad7af5ddfb56b7484fa9db591297c9e',
                        'https://www.theguardian.com/world/2025/sep/12/nepal-appoints-its-first-female-pm-after-historic-week-of-deadly-protests',
                        'https://www.jagonews24.com/en/international/news/85536',
                        'https://www.aljazeera.com/news/2025/9/9/nepal-lifts-social-media-ban-after-19-killed-in-protests-report',
                        'https://www.aljazeera.com/news/2025/9/12/nepal-protest-death-toll-reaches-51-as-12500-prisoners-remain-on-the-run',
                        'https://www.washingtonpost.com/business/2025/09/08/nepal-protest-social-media-ban/57e5190a-8c8f-11f0-8260-0712daa5c125_story.html',
                        'https://www.moneycontrol.com/world/nepal-protest-death-toll-rises-to-51-article-13542486.html',
                        'https://en.wikipedia.org/wiki/2025_Nepalese_Gen_Z_protests',
                        'https://amp.dw.com/en/nepal-death-toll-in-anti-corruption-protests-rises/a-73972986',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab2fd4308195845a2b59c42a3e30',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={
                        'query': 'Reuters September 12, 2025 Qatar Israeli strike Doha Hamas leaders update Reuters',
                        'type': 'search',
                    },
                    tool_call_id='ws_08313d91a201f39c0068c4ab3324008195a9d746750af5bdce',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/world/middle-east/trump-meet-qatari-prime-minister-after-israeli-attack-doha-2025-09-12/',
                        'https://apnews.com/article/380034326d3158e8f40462c87abc36ec',
                        'https://www.theguardian.com/world/2025/sep/12/un-general-assembly-to-back-hamas-free-government-for-palestine',
                        'https://www.aljazeera.com/news/2025/9/11/un-security-council-members-slam-israel-over-strike-on-qatar',
                        'https://www.aljazeera.com/news/2025/9/9/israeli-military-says-it-has-attacked-hamas-leadership-in-qatars-doha',
                        'https://www.cnbc.com/2025/09/09/israel-hamas-qatar-doha-attack.html',
                        'https://dppa.un.org/en/mtg-sc-9992-usg-dicarlo-11-sep-2025',
                        'https://www.cbsnews.com/news/israel-attack-doha-qatar-hamas-leadership/',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab3324008195a9d746750af5bdce',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={
                        'query': 'Reuters September 12, 2025 UN General Assembly votes Hamas disarmament resolution 142-10',
                        'type': 'search',
                    },
                    tool_call_id='ws_08313d91a201f39c0068c4ab378af8819598e6fdbafc72d7c7',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://apnews.com/article/6afa02f5fb4fc8823ac0a6b74d69e2d4',
                        'https://www.reuters.com/world/middle-east/un-overwhelmingly-endorses-two-state-solution-declaration-that-condemns-hamas-2025-09-12/',
                        'https://www.theguardian.com/world/2025/sep/12/un-general-assembly-to-back-hamas-free-government-for-palestine',
                        'https://www.wsls.com/news/world/2025/09/12/un-assembly-votes-overwhelmingly-to-back-two-state-solution-to-israel-palestinian-conflict/',
                        'https://www.newstimes.com/news/world/article/un-assembly-votes-overwhelmingly-to-back-21045225.php',
                        'https://www.thehour.com/news/world/article/un-assembly-votes-overwhelmingly-to-back-21045225.php',
                        'https://www.thenationalnews.com/news/us/2025/09/12/un-general-assembly-votes-in-favour-of-a-future-palestinian-state-but-without-hamas/',
                        'https://www.ksat.com/news/world/2025/09/12/un-assembly-votes-overwhelmingly-to-back-two-state-solution-to-israel-palestinian-conflict/',
                        'https://www.investing.com/news/world-news/un-overwhelmingly-endorses-declaration-on-twostate-solution-for-israel-palestinians-4237326',
                        'https://www.timesunion.com/news/world/article/un-assembly-votes-overwhelmingly-to-back-21045225.php',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab378af8819598e6fdbafc72d7c7',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'September 12, 2025 major earthquake', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab3dd990819593e473934de5c75f',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://nypost.com/2025/09/12/world-news/ukraine-strikes-key-russian-oil-terminal-in-overnight-attack/',
                        'https://www.reuters.com/world/europe/russia-belarus-start-zapad-2025-joint-military-exercise-2025-09-12/',
                        'https://www.reuters.com/world/europe/kremlin-says-russia-ukraine-talks-are-paused-accuses-europe-hindering-them-2025-09-12/',
                        'https://en.wikipedia.org/wiki/Typhoon_Tapah_%282025%29',
                        'https://www.aljazeera.com/news/2025/9/12/russia-ukraine-war-list-of-key-events-day-1296',
                        'https://en.wikipedia.org/wiki/Tropical_Storm_Peipah_%282025%29',
                        'https://en.wikipedia.org/wiki/2025_Myanmar_earthquake',
                        'https://en.wikipedia.org/wiki/2025_Afghanistan_earthquake',
                        'https://en.wikipedia.org/wiki/2025_Kamchatka_earthquake',
                        'https://en.wikipedia.org/wiki/2025_Drake_Passage_earthquake',
                        'https://www.thehansindia.com/news/international/deadly-earthquake-in-afghanistan-tremors-in-argentina-alaska-september-2025-1003393',
                        'https://public.wmo.int/media/news/hurricane-fiona-hits-caribbean-and-canada-typhoon-nanmadol-hits-japan',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab3dd990819593e473934de5c75f',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={'query': 'DW September 12, 2025 top stories world', 'type': 'search'},
                    tool_call_id='ws_08313d91a201f39c0068c4ab4308288195bff2b3316b62cb4f',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://en.wikipedia.org/wiki/September_2025_Israeli_attacks_in_Yemen',
                        'https://fmep.org/resource/top-news-analysis-on-israel-palestine-september-6-12-2025/',
                        'https://www.sbs.com.au/ondemand/news-series/france-24-international-news/season-0/france-24-international-news-s0-ep7/2446206531749',
                        'https://www.sbs.com.au/ondemand/news-series/france-24-french-news/season-0/france-24-french-news-s0-ep7/2446206531754',
                        'https://en.wikipedia.org/wiki/Israeli_airstrike_on_Hamas_leadership_in_Qatar',
                        'https://en.wikipedia.org/wiki/2025_in_Palestine',
                        'https://en.wikipedia.org/wiki/Assassination_of_Anas_Al-Sharif',
                        'https://www.sbs.com.au/ondemand/news-series/spanish-news-france-24/spanish-news-france-24-2025/spanish-news-france-24-120925/2446206531904',
                        'https://tv.apple.com/au/episode/friday-september-12-2025/umc.cmc.7hlnpg91k6gwgl8hnmn4vayn4',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab4308288195bff2b3316b62cb4f',
                    metadata={'status': 'completed'},
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search_call',
                    args={
                        'query': 'Reuters Qatar to host emergency Arab Islamic summit after Israeli strike in Doha September 2025',
                        'type': 'search',
                    },
                    tool_call_id='ws_08313d91a201f39c0068c4ab4e23a88195a39e83d44d538ad9',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search_call',
                    content=[
                        'https://www.reuters.com/world/middle-east/qatar-pm-meet-top-us-officials-friday-after-israeli-attack-doha-2025-09-11/',
                        'https://apnews.com/article/380034326d3158e8f40462c87abc36ec',
                        'https://www.theguardian.com/world/2025/sep/12/un-general-assembly-to-back-hamas-free-government-for-palestine',
                        'https://www.thenationalnews.com/news/mena/2025/09/11/benjamin-netanyahu-reckless-qatar-says-after-israeli-leader-accuses-doha-of-harbouring-hamas/',
                        'https://en.wikipedia.org/wiki/2025_Arab%E2%80%93Islamic_extraordinary_summit',
                        'https://www.euronews.com/2025/09/11/qatar-says-israeli-air-strike-on-hamas-members-in-doha-killed-any-hope-for-hostage-release',
                        'https://www.aljazeera.com/news/2025/9/11/qatar-holds-funeral-for-victims-of-israeli-attack-amid-regional-solidarity',
                        'https://www.aljazeera.com/news/2025/9/10/gulf-region-at-risk-qatar-seeks-collective-response-to-israeli-attack',
                        'https://saudigazette.com.sa/article/654886/World/Qatar-to-host-emergency-Arab-Islamic-summit-after-Israeli-attack-on-Doha',
                        'https://dailytimes.com.pk/1367098/doha-to-host-urgent-arab-islamic-summit-after-israeli-strike/',
                        'https://www.hurriyetdailynews.com/qatar-signals-collective-response-from-region-to-israels-strike-213488',
                        'https://english.aawsat.com/arab-world/5185339-doha-host-emergency-arab-islamic-summit-after-israeli-strike',
                    ],
                    tool_call_id='ws_08313d91a201f39c0068c4ab4e23a88195a39e83d44d538ad9',
                    metadata={'status': 'completed'},
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
                        id='rs_08313d91a201f39c0068c4ab5b97588195a9191e0b947ea50b',
                        signature='gAAAAABoxKub1Bqz7YiOUZfis2Adw7-4H2Mh38ch2b1u8XTubz34jghLHyRdGMFTbD-5Ofdi8zOvJGNDF_m6f9yUFcW2uhk4lOYn4UUd7awr730f2UC6Fmxq7wCOQMwElB6Jq7TUKi1TkdSt3IcZbQK2a3G77aldRdJA4fr7slWRlkk-4pLuolWF9XMlxfJinGUfOMFARJuNDkL1rLyxef6QaQhPvqaG-sefDXpJyz557uXKfRHtOvlDsxcvphd980dN-gXYwgBwfNpPEYL78Dw5WGfUel2H0erw3dFdkz1OzbffIVEvwEfntQh3bYstVsNP3iOShg9Ka121s1f4PQPzwkyKwc1mDO0lwYCJZ2LjGpQOOA-IKhMrB6dp9RBgugPOAXmHG_QDTdjluJ_QlQMLtrxuDXUqFVA2jYHtjEqplXCl4A25p4R-YvHgEM4Ks1MJjx-rLyBsH7lLYgUoUMxSba5MaqFVtRj7e72pKdZXJ4iaEylFzsGHSsTWxrW3jkoumTzoRlww_NOWwb0pdNYTcBNb3kRFpbcomT2iPxnrsPdwrX9NZaOfB-v0BuhKGNXSA8oZ_VHFHT_MoWaxKls1f5gfcWuqSht_PbzEIuayg4nj_bCCYEWuN1elM2ff8WMYn4YYEnQZ_iTkTH1PR1fN8Lq2kB96PibDIxCQgfdsmzsx2-ZncruREvdLOxXQBw1PN0Z34Sim9PiyhUyYqNvccqokwtk0VRwB-Hn3aIH3lcYGrOszpBk3hQGz64DeSJLGy4-Byci8qv-JZ8pvfM6MtF4piZP3EqgY5z38kKeBg3IdbVqI7qg4I3fDURzRemADaY3z7RhK70ztWcJ9W_v_wqATM2hpCj8KE7FAryEaq_A9cgLGXac98XYYeO4eAs_RNFoZv7keNNyZ0q0qjk5qJIAb4h4CKt_6A7bLpDXHy4DX65CVZZ77UDpPvRta5tYJqTXt7e6JemTqtU9eCjmtIqvjFq_V9KsE9g7TRxNTx1Ver5YgSpcSHmK1SYwNgv1rl-LgnfKl4YwnD0D2c7Q1JVc7IXJQJEyfhq7XP0sZXcmihwOk43gRehc8b0wBZANe44sNPnxRFQ7Aqj18M8cojrYYRZ4wX61GdgTn22aLYAWz378DK09b0htpAibQuV9tY1xxncObiE8xGW3ZBUmBzPA-BKXE4G4HIjzc0BC0O3lM5UoLcTqTQF0Bxxt9ufQ7ZMPHJgAxXPGPvoXAG7J9OwJndW20mylce_zUbFk5Nsxg8cVkdNRrUuAOxXWUVtJphR9Y3VBapOAOTubkK_WOyfYkkorNZlcqHGmdcAiJAqZ2W2cCo7NXPyWJrQLA0PTW1EUPJHkZ2iFZloKXREsrYMJuaE2HRuyH6VmakhroA_LYOFUDQUsFbW0Yu3WSXDNQhxCeod5BnZG239GhHTaUkbwIKNdOgVYmQtNPxVuvKXwSe254PaeoLSjwaGIrUp92-aRau7Ew2mRBPNHpBQPpFTErIygPYbXKrOwyETpRSwNpuUXbGcUCizbjKa5xEN4MjNytNpwQmRl_bLSvsY15jcphGKLsCWxlkd7bMJXzyeDfOa_rQT8pVuEU5X3K5d_dZa4DnSK_oCpRTZ6JV9HxI2paVmrp8qeNuy93phQl8a7lICWibx-K6jQewrBlX6Z1R2sdWlDRB2f7J1EBdVJZxI612khJzS4wI2j13uae2nGZZdLRTjjB_MKt55o58sbkFy5_8z4RGKbGBAW5rBN7n5gJVQiVJMd3AUzNuNV68VBTDGXvEV9Cn94cE70pAm2LFvlAIzjUiEbu5QbrPUtvUbc4IcDp_LUj3VS0gQX--54veZIh2FydLdoheEQcY-g0zhv7U1_eBhORQkqVd_A8viTblTWIsKSxpr4-RsCh8ub3rSFQsBldKmjWwyAg-hyUIUFCEt8EUmVezMiof9APcerYsP76YhwiZ_eNgLMHogPqefAjK7IKkTPhm25qmMvaQjN77ksgq7vn47bROu7icnRztELzve7ggF8jsLZp2_aJirxT0WxozooDwaacgMsznmNCok1o0VU6E2zTGGqubgqmeWSNWx9EBAiZSiGWeJ2v_Tc6Gk5tc_q3v__eBSYwpa427nAjGtKANSqQGg1mADT6xveNZqtcaXNXvjncZp6x6V-IqUG_lfbKh_FN5ljKe0kHRHMZsX67K3E1UvS3s0VrNQ3zokz8F7xkCsrL8lwgR_Va1Y2xqpky_rgXfZgCJpd45C0CF0PPPFKZ0b5lqvNgxaqQ4sxwqoWGhbqm_Zq4r9-eXdnGzPepFPEfSHxQTA8VHHex0nEwvMu6vmJwG3iwWrjVtEsLdHT5PPmj7LdUumSAhIdzF6skhkMjHyYRnoms0QQtJMUctiq7tWV65fUddNJ4a88nnhoyD9cBDz8IZ_aDJ4P2Y6urXMSVxsMCsfYp_YPGe3KmJ1GOTnv08sHSQxoools_e4DeB9Pt9HdaBX6ku6VkpjvwPGJePmhb9M_99r59aO4Fgitj4ONfaNH1w-Ir9ptaMZcRKgwqZP-WaykE4DjYEwpVcbm22ULayWwylPoOVJJV-yfQO43CBPIYtpct5KjIaqXEdtTHU3YsmM-oXQnLHAkcWft4dcjADnmQ19hQPsWSI7dGKZkG7BSKN06ie4g0hqMmR5Gxd0CvAlnb3nO5ipa71krHJUHmQZrl8kU1l4BXTq-I3_MNM5qwGJWAfW7igf8C4Mkel3AM6eOy9WUu67wbJVyfvT70g2PCNDWoioS1ZtlkFBLa0PU23yIO-G24LP6uQSGjHu4Lx5WSxGzwlzl1DdkfasABCetkfYLQ',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'Mexico City news September 12, 2025', 'type': 'search'},
                        tool_call_id='ws_08313d91a201f39c0068c4ab5f2b208195ac9457000f5b4c6e',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://apnews.com/article/8230844a53bf03772cfefd1b84336933',
                            'https://www.theguardian.com/world/2025/sep/12/mexico-acting-under-coercion-to-constrain-china-with-50-tariff-on-cars-says-beijing',
                            'https://www.reuters.com/world/americas/mexican-officials-speak-with-china-tariffs-next-week-2025-09-12/',
                            'https://www.reuters.com/data/tropical-storm-mario-tracker-2025-09-12/',
                            'https://mexiconewsdaily.com/lifestyle/whats-on-this-september-in-mexico-city/',
                            'https://www.aztecabajio.com/noticias/hoy-no-circula-12-septiembre-2025-que-autos-y-placas-descansan-en-cdmx-y-edomex-hoy',
                            'https://www.aljazeera.com/news/2025/8/12/mexico-city-airport-flights-hit-for-second-day-due-to-torrential-rains',
                            'https://en.wikipedia.org/wiki/Worlds_Collide%3A_Las_Vegas',
                            'https://mexiconewsdaily.com/mexico-city-plus/whats-on-in-september-in-mexico-city/',
                            'https://en.wikipedia.org/wiki/Protests_against_gentrification_in_Mexico_City',
                            'https://mexiconewsdaily.com/news/intense-rain-floods-mexico-city-zocalo-airport-closure/',
                            'https://en.wikipedia.org/wiki/Estadio_Azteca',
                            'https://en.wikipedia.org/wiki/Guadalajara_125_Open',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab5f2b208195ac9457000f5b4c6e',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab60617c8195b2a51bcab5708595',
                        signature='gAAAAABoxKubBc300RGdq4-JRxx1LB_3nRY1N-hFDCsE7cJQFucj5vJT0XQwVC23u7wZIeAdjwML1D_UgSHJ7bIDPpJX1ifIApLfrj3SYuIvYvLB7B8-ryTjNLyDLlM03Q8xAqBwBtUtgEFNfG3wp3Tr5_g4oavVrQIs_IcTqAC-VHENEfuF8MY7ioVqVHMYVbqbT2F3pGN1ST1t5MoN4iLZNMkINkDPmCvTl1dYLCi7V4Q5gsp0chfKSKW22kiz3y6io7E28xGhlfxHfxXptTHIMy__VZblqKCmUx8zKzt0kr_sIBwl2kOtDM9DgvERW1q42ANe2j5UdgeMlS1qdLmx2Ek0mniQryKH_lYFSsGegoSNUPKnziK34CeamitVP9aMVH8QvsQQyvYVJY4R_2DrGhPgjyjTQOru8sFEtd_pRx38xYhQirgoWEapBT0uRLY6XrxIxSPx_ZoStPXWkA75RFlQwn9S00W9auLv42o-ZkGCWNvXQMqJbYqWbR_VmVH5o800-MEQhBVwlpwJABRigurlc5G03Nkun-5Wc65vsi2eJFiZV6fS3vpvwU80Nk3YZuEOvseoYOzr3A9-iO14KL60a7S5pYUY-xooeDoJbc-KUk7FUC2Lql9xHQadKpWaA4YEqQe6RZAl3hY2o-XOtiPxLlG-VXT6XyThVazWFS5zc9r2llcpwE4X62FC4vvsEPNGoWBfQJfrlWo9tUbIzhPj06ECK4wu9Yrb1GLsP7N07MtIBArLdhGkd4vfSVHVHXGz-TL8APiCwk5NatP88jznAWzd3k9cVpFTHYoKD7NhghVjQd4UonXFUWGTieGYWGWttQth-j58OshqxpSDbHqJ4wYmEVtvl1vWlo2mT4vyy6XfWzBcllZKPWGQFMyjGWu57ahHTqPzz8g5Qih3ULAVYMxa8CKHz7liXpZguoT_DgsFS85D-PdonWNAs114EPpY4MPg-eAJsrT2lKVQ5XEYIC8yKKypdyi8QxsuKo8KL8FichPLjh-N4FaM2mlmxJQDqZ_SdVTxmhYU0I0ZLcxjrjLBa2fSISBOGKDiQROo6XxvoYu6vTBDlmQahfCzQmuchF66ljbTckKWxR4j-5z9pTNiX7w4-RKJ8zGRn3mM9ksLzxkkc9rQPRIUh1ss0KywzfFlcAHKMU4MeUCnZ5eIPG7CiT9zHbOpHSS5JQ_wuuWzBfxKteO0Fsi7v8iL4Uo51f6so0t2PzvhRf_s58VssHyPDCGJr7swp-gFIZ9SEzks0GfruFaOb4zPih5TueIo85YlCN2HcTd4fFz0p335PPxpfFc1IAm5yp_OHquFTaJPuIAL6L9JTlkUNR6fLG2foLYkPSg0M_Ix9TbKkSqPX4bD9G3IKQ-tA4Ol6QrTKfk40OY3wtRtEAXOCi9EaWIuTpMundrpKpwLsmn-lWje6PF0ecDiuZIJ3usNioSPmc58eHSQeJgFk2nN5My9gQ3rszkgYzSZ0JHK9bD2WDthx8dxUxBbgbaDQK-zroV5dblVlg5adkRKEtr0WzeN-rtgIM9p1BjjBZOMyUf5CpGjAgvjoWO5eRJjpZLyXvG610opfDQ-aAd8QiBIxu_EuqRh-Su6Okt955fcuuAF-JPTJOEsOw==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'CDMX noticias hoy 12 de septiembre de 2025', 'type': 'search'},
                        tool_call_id='ws_08313d91a201f39c0068c4ab62d2e481959ec30d522913cd30',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://www.reuters.com/world/Y3CVQ63C2ZOMBN4RG57TYSDO2Q-2025-09-12/',
                            'https://elpais.com/mexico/2025-09-12/sheinbaum-rebaja-la-tension-con-china-por-la-imposicion-de-aranceles-no-son-medidas-coercitivas-queremos-una-buena-relacion.html',
                            'https://www.reuters.com/world/americas/mexican-officials-speak-with-china-tariffs-next-week-2025-09-12/',
                            'https://mvsnoticias.com/nacional/cdmx/2025/9/12/hoy-no-circula-viernes-12-de-septiembre-2025-que-autos-descansan-en-cdmx-edomex-710924.html',
                            'https://www.telediario.mx/comunidad/calidad-del-aire-cdmx-hoy-12-de-septiembre-2025',
                            'https://www.milenio.com/comunidad/hoy-no-circula-12-septiembre-2025-cdmx-edomex-que-autos-descansan-atm',
                            'https://mvsnoticias.com/nacional/cdmx/2025/9/12/marchas-que-habra-hoy-12-de-septiembre-en-cdmx-afectaciones-rutas-alternas-710981.html',
                            'https://www.aztecabajio.com/noticias/hoy-no-circula-12-septiembre-2025-que-autos-y-placas-descansan-en-cdmx-y-edomex-hoy',
                            'https://www.ambito.com/mexico/informacion-general/estado-del-metro-cdmx-y-metrobus-hoy-12-septiembre-2025-estas-son-las-lineas-afectadas-y-estaciones-cerradas-n6189569',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab62d2e481959ec30d522913cd30',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab6546bc81959a507db99c0f89dd',
                        signature='gAAAAABoxKub0OT2rm1OPEO17YYFzdysqb0EY64EQ03G7jb3IQbPbsni0RDUAwbFULDy_WFQ_2cFncFappLDeDf7JGBQBRoSywkpmKyp3-H_SmIM4z54pFCdA-7ZeD4CWR5SMTKmep0OzFMYWoG0BTEum7j2fzO92C8JGUZ_nJuIovMfAjw1KGDAPBDRKj2PUMUs5QDtSLA3yCfs3BfV8oN2nPgGxNwGIfKIXtOrJH94sS1x_yN4IX7UKoiwfT47Vgn0fYnEZcs5BPjJ13Ru1cosIo0Edeu-Rkve7efYpseLa4Efr1DTw4xh6s9Nh_p8usxtPcFwquKPhyewzx9C8hOahVB-8IqfUmdUIRsgZvynd6sgpbrVZb5d33UH10BSv-xLrQr0O7iArgj6eBAjPvkEP8v2ysc5gGJ7R_rifze0_mL5cZskt5EO6tHXPOK0x-eGfECGlOzzBM9kn46Jmb7Utynt6M4titEEQdIWAcy80IoBucavpjG7w0JoczkB9zObo9FC-ogbkmh9_gE7nhv4AmtDYdc9bu_IYwl0mHcnDTwv4f0_3pLSqOymHnbIGN9IZ4F2OOBzPXjojcRRVu_t58aSK0ZcJ5f9kKPtRe9T7EV7C6I9WsMv5MKkA7sV7aOIXT2VKAT2JkrkhcL_DSx4qQWZX1N4rRTl318HdlgyBdwOwxiMHBcC0-U0bI0C0SUKKYe5-hM7-RXRed3oYr9u0TlAUjia8LN4U6vcolKZsgwiTBc95TN7kIqGpvMIleeQgXrMtqlr3rQlh_0SlMD6lZFxGgxzl4ZXvwWH2kZ3WhjgDl5mNb9EoDlHYbg8w-SDV_lOextrJQ8n6IxaJNHum6RD4fRUjXqx5qOSDnzFXy8gBy6Ry0_dra6dttHS2auKpBttuWgy85tGqZgdeMGba594PEk2kN05tkzwgb7E8HtIVkTWb-cUR2EJUvKWfvC8GmGNoZICpnHWbfE3UuzAJx-xJii249h2N2G2MgEIkHDzWCkgquOGZQ7KRQD0DlNaiEp8Taou5wkLx1iyrT3PkXViEkr1KNsQT1EzzWLZXbh8KJvhcKOMeXc9XAS_7TRzsgm0bv6HOr_9gpO8D1OpGBkLNXkTZAYyfq5fIyPpaXjPxp_WLkat7EuH_DriPXiyxOMjXkrApgYqKugekMzfBt8XxyKQF4y7TTmnlcqEcb_OOPv_MNTB8sux9K8mKGwpFBiHaVfbZ4zUpEBT6L6SqULz7J7tu-GPhHAFZWMchwjB3kp2EM4lTx9w6keuysKLQGyF7Eb7a_UtufynSKtCGZGmyezLllxsqZcfkTBFwYv2ad8e-Vtj1MTOeaXfyOpXieRo9ugmTe5ryArBQRuB4kw3KT6YRzGhXNx49CFqFOqgu5tPY0IXx_wEoq4hPOEeXaMtqcZpZ_IP_1ik0G8M5QMHuaBagi9fZRi5XhOo4lrtWm5Lv0ibEudaoi9nBYHxgHrTeTZNiyclICtnI-DzNGyjjzh0e1z0B0f1N2_TtNaqfApXumA7vJHPIMAJBz9uKDm6G_-I0m8zD1hmKHzUHOxFwYhMx-ZHjiATbhPXUAifpJxyZAJ9nucJPFi5ID6TPKzJGbLzotOSF4biWdd5FpZaLZZus9sMhvF8RH_wEYuA8n-8juLy6A6GkANY2b6M-6_d1csWmpeM3ARlsWZIObDKaamR_mtRh9EbkAQ1kP5AjXSyPIp2Jf-PrPsiopUCuBTwcpuUUqsMLvox6b5pMWSVJUafCXWSc5xFmWV5GjQLkvyIZWKyNNm57muRXJJWQ0x3kUj0RRHudMyEc2phD5mFyI7-o4pDtuTbit6ELErML7hKR4sQY4GdXUKJZCItL9e4gRhEb0h5YgwlW3gEAuttLHeupSymAJDhDMMV_joYiclF0SIRaoc53hfinsRbAavTyL41ZXg4psvBlE23StB8i5fCr9rP-gz14nuOHb6-na4VsqbE0U_qlK6Bo_u3vaR_B1oTRJEM7ia9vvGzCmTLqpPecdExKVDTZiZD62UsozdQp2te14D03O8NZUL97bqAEKO7E4n7__hBYCZBKkS3KN5TFZnz0At2eg89dugUCbiLdHYotdPRfaCLUxLlgP5Eb9obCUrQt1dJ9BFeI9e9i-X40I1cxYJsIAbO0IDB5zHieie-mVg1J-x16KQ4REgBCGEr',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'weather: Mexico, Mexico City', 'type': 'search'},
                        tool_call_id='ws_08313d91a201f39c0068c4ab69d5bc81958a568c6c8fa3b010',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[None],
                        tool_call_id='ws_08313d91a201f39c0068c4ab69d5bc81958a568c6c8fa3b010',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab6ad7788195bb938271bba3f4d2',
                        signature='gAAAAABoxKubSfwu9qF0dXFk44AW4nlSPtlO3T-tgqA_wA2hTdPB99nObzhisu-YAyO_683Azi0VDdp1nRDMPbtJmSXZQBNS_G8kUJvWnHouPy9vkRAyqpPerpCsQ_HgDmxC-UP2abQflg3yXGt-p0xZWeTtpSV12BQxvyDstAkUQDl_o5nHTcX3E91b_42NXA4lb1ijrJWPknoxAcJFwcygvcSaOPOQt6mMC6k2wTOlgJr3Cb3OGCMLlfXmsCK6NKWE_vk50ufUWb4fgvdRL1_44SUxZLvIJWYyJKC8ITL3CGdpArzUP5s_RGa5Re13P2L29VaD-wXSDf4F7zEuQGidZVhOlnj0HoSK9LkPdwVQeofYGN7noi7iQ4bMR7cz2gbrqjhcgDECkPfI5lFvBfHgp1CctM-X_YYwv4g1BSsS3_K5n5UAAlR01MqylfN40zDcjts8J5mof-vE-RM04T1EMT6KKsT5aso2PYZNqrUj4XTJb-9JcNR-IXkORsy9Q4B2_SzQ_RRtN3XUMrSF0KioT4C-nJutBsRZkTxreuTB9OGiLfiaNi-7KYsAbZon5SdihT6e7gxINuIJ-TNvXvDj9veg2liJVijJxL8I4qSZWf-bvDpkGXTwESeb2z6vEoFzKIMNKzTqLTv8CFOWEYH-_npGXS1pMjHAQdEa7RZoxsNCGIPDpB3nT9dZJfqbKaZx7osDjjs2tIGz5stXFaXM9GWCEYarKFDytooVikHamm1m702D2xT47f-9KFFrH3-4OJlshdrr50r2q6ZXKsiRevTWX1QmV-xr-p-EF0ZjrKtEirX7EqgPkArjCJalXZ-D1hEKmfDl2jiMNGMOb9fYsTViSrbfm6PRDy8r1ejNxr6P3Tb-jrMycsD4bsvNuARVIlKLoNGNIUkwAWOazcEBozkq_BEIXYAFRJ0Q6io92daIpJxu8XDaynWfBuLGxLnE5S6R7NSyZ3jw3orbrpPgJRdXk43jGGJmGKpkJ8PmISpZbKH8gbo9hKXaVugM3iMg4ZCcdsWWW5av2ysq2H7UytKCHfLmOcIhs58z0v5_rCf4mWZw4T9T9OHVacW2EJrpLXjnUQatgdeW_lBioGWJ0rAiHouU4sY2ZYUs4jpVvzncdK8eSlZjAef0jLISep8xKUOu0xZKyvxcbL0Ek_KTpO6wx1l1txqJnMbXYwek6n8qSXvwR9do8TBa-o7ehjHE8McLjqg-t_1ha0WDeFwIA7djuDgfAR7TPJ1ruWzYcDvkTPmPewEKS5EM4KbJV8sVwmfCLNilEqkXrcjYIrqS9BgCtwjtrc_0_QYJGDOBE8UrrlQv6sHkE-Q884-NI5PtlOQGl086psWpNZ2sJyWKEqmPdfhXpi_yQPxs1qlb8HcUOfb_8qsKmOpYTooQPRMiJvowssWcjMOCurUKKC0jIvvltMvV8c_XqOftxHInIgYEfEbOH48YOfRXj5_fBTvMg_pL9yjSSZ0u_53J0zrfd6obdGphGwL8OPIOTssAqNqppRFXqGCbtuyurYLjxbo2JZ_xAtu2BTqdRmQP9AmAxjfXhjRZuuAMtBfjQiWVrXCMlo0SNrw0H7EOTR8daXT7JcHj9zxcRB9kFLe9dmg3Uv7tXOBh2RosfDY5mIM889KMGwLkRBoBvlzmYSsi5Anku9xsqoAXzDkvh8BIJyFPUWf3TXvWP2JZZQPP6c39aBZv_EIUeeruBrlGix7NN4SgNslHPxDjJ3E70cGcEumUs5y27zw-eNRG3Xb2_wWEnKhnsdBFwWLOTNMcGmYolf9_WZHiVngZv-BkYHlv8Zo3Tmjt0jkznrYqzGWa2rToP_XSdrzfgif_Wj_XI1x3uyV7aY6TkI9tTodyBiVy8JtFILASI5ZHp73QIGe_oB6u9A9EESot0tASe_AVsxc0q0QfcQyFc7mMjugPsB-sH5VwtJXgExiN-M2u2d7YpIW3uuChDcwOfNMI5Lzby38rhUz3vtfONBW76WXuVr9DeATs724p_lorOrt0VVooMnne9TdxWLIiGcY4g0Q4Im8NMJE7QuRPKefLDGn9UDKH7FlYs1_kFctkMoStReT3BAXgaNYBOzkRHfEOcl-hZrCkuvDfAYVAOxjFs8oeN-3cE3K-R2H5PeRCN8E2S7Yci1biIu_Ys3haOYD5ycNqI-7ECQTQAzlpaMN-FQ459gfSumMD-8NKEAzoeNC02clAv6oIpHcbbXrb9xC_WrfUnLVBCjRcMdP0FosYUrObCRqK3zEiRzm98zQde-yw7aN12cfW5BUuEsEe6dMtm-ZZh3GoQSHd8pNAHJV6FmDJqa2u854oZPd7JV387294qvLB5BgI3yhJx92pXzboZNQVzwaFhsD9jUQb00ieQkRhnaHOByvza1g3_CXFxELOIrRCzgfo_KOY_loO4TNm3b8xhalxZdBSvWKW0W02f3MAGqpFCqEXqgozsF-8uGYVsUftMf09V-V2sUWxZybLHWSL4QdLscx_acNHc2eiR_2dK9r6ogKzsKQJ3IYk_mzuCd7x4ow1D-l9GNh9jJWTSfVwCs_iqUnBu_WR5SbEPbq8xV5ak38k6xUMxiP3a0VpX359LN1pt7rkaGWyPDAOabWjpY0e8pugBBLOpmyWiLpyMgIMu0ncTEojKABPnlXrnfWJNxs-GoBvUeudBSQSJI0fN0OvOh9qM5U3_ZL6',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'CDMX 12 septiembre 2025 noticias manifestaciones', 'type': 'search'},
                        tool_call_id='ws_08313d91a201f39c0068c4ab6ef8b08195a21908304d53c18c',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://elpais.com/mexico/2025-09-12/angustia-y-solidaridad-entre-las-familias-afectadas-por-la-explosion-en-iztapalapa-no-tiene-seguro-ni-trabajo-fijo-pido-que-lo-apoyen.html',
                            'https://www.reuters.com/world/us/tanker-truck-explodes-mexican-highway-killing-3-injuring-70-2025-09-11/',
                            'https://apnews.com/article/5a33dbd858958bb8e376e29d5a3601bc',
                            'https://www.ambito.com/mexico/informacion-general/lluvias-se-estancan-cdmx-y-edomex-el-pronostico-del-tiempo-este-viernes-12-septiembre-2025-n6188740',
                            'https://serpientesyescaleras.mx/lluvias-historicas-paralizan-el-aicm-y-provocan-caos-en-la-cdmx/',
                            'https://www.infobae.com/mexico/2025/08/11/aicm-suspende-momentaneamente-operaciones-de-aterrizaje-y-despegue-por-fuertes-lluvias-en-cdmx/',
                            'https://www.unotv.com/estados/ciudad-de-mexico/lluvias-cdmx-martes-12-de-agosto-2025/',
                            'https://mvsnoticias.com/nacional/cdmx/2025/9/12/marchas-que-habra-hoy-12-de-septiembre-en-cdmx-afectaciones-rutas-alternas-710981.html',
                            'https://mvsnoticias.com/nacional/cdmx/2025/8/12/aicm-vuelos-demorados-cancelados-tiempo-en-el-aeropuerto-de-cdmx-tras-suspension-por-lluvias-706202.html',
                            'https://www.elimparcial.com/mexico/2025/08/12/el-inundado-aicm-cancela-vuelos-ante-las-fuertes-lluvias-y-tormentas-electricas-que-azotan-a-la-cdmx//',
                            'https://heraldodemexico.com.mx/nacional/2025/9/2/aicm-emite-comunicado-urgente-por-las-lluvias-de-hoy-de-septiembre-727084.html',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab6ef8b08195a21908304d53c18c',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab705024819586850395449f2109',
                        signature='gAAAAABoxKubGN_bPok1NiyJWK8rldCfIzHKSssz4BcCW3vAeEoBneXL6vSu6fsNeMZwg8EVCx6GYy21jK5Q0_4-IM6MYUlpkAOoVguv01JLg8C8vJD6inryy1p9fOmXPGLR6DyHIu5P0Nq6lp8UXZvXY3wILdhLLv2WGWvnwZ6bLVwH8vPs0lQTdnygUKo7DEPRN4vtTyMDnpai-WHwkcmO6TPIEJXGbruNGWxWMSN6KidOExdnuHXbaKvlHhJZnho6q9GN1WuGOcB3BFuKRpS6ff7rI3Xf8sz4_TzyI1xGxk_a5ufRT10as_0BVr7p0X4cpGgXW0w1UJbSi_YHDNEmH90BhDQkNNRf3gVPEjbIifnXxj2frdo_GNc0VWrxlpm2BS7LzBbtl3W9m_gfqMjrB-h0nErZvUToapUSS56L6gE4gFjk7FoXFMp9A79dmiH8c0IhPrhT_7-56EHCHD-zR2kZ4J23lX_Bu796vziJbrcKb1S3y7D-r5RiLaFrPfOrezsKjJYf1b7K-dr_dNN7U7xtt6pq47hScY3qolykdNTYFFFSjV0-drigJgElZnLUm9fmdJOa9sAJiuxsxKTOKyezHw_oRwdq_D8dlJeXTCU8WQWR1RRzHPFWAKEBEAyaykKybxi4CX80WY5-Ba3L-O8qauEuIlEJqYPDKxn_BAR4U_mMwDIZufRxlnhaZbS4mjIzDsL1sz9EKDacQCGTPrEoBmihTT8LGCD3x0JRe3ud__d3goulkQOrnDVfo2SorNfnoeqhVDbjFUOnwz_DlfPo3D-F2aevtAm9RvsVNfyIvqqgkGs8o9L9ExNBDZg0TGKmpeDHnzidoQaXsuId9FywesUdONvlEO85toloBKkG5eZY2Fu1Dg15IJSc-82tP0GWx6w5i02dda3mAk-R0IWbVrgUuf7EbBZafk1mPuVWc_xdUkcBjdCLRIgGd0vuFs4KY31rM6ucC-tXnSVw2xdJoaq23wyF5_OU461Tw91csvn8ZysQlKFsj5dBT2MCF88QesaMcBgNI5dBxZz4tL9ocS01_ZI00yD5Z0IguOfSU_OZh24dW7D3vwEL2bh62GtVphJTq5HaVWy-8dtW-nwtlog6qEOjNWLgMzLIg6Q4l7S-kfITlXC5hKCncYVffljtwvw4c6A73C0QBwLNH26hCxE13pMgYVGjOCBuaQtEKW7w4v-i1ZFzLsMLs2dVTnD2cWuBBECidQozRSecQnX-R5gOZkLx2H4wSXKyfbEyIHT_onqb8pr22l_rkwl7JdT_p2jt8NWV83-RMstXhLz4s92w9CjKL-rQnQQZyCuvZEpnuuyb_mbFTiEpmUZ56SDCzc3wqztVBOYd4VsctYivk0CAJha99dxZ84fbwMGNFUh_5Ic_NI4QSXwquwwbh1ox3gLDyOO57M3McaK3N-s-LXZYrrHZXENV4zEIslVOv49FPJbpN5a9QiA71AHvWVOkR9qmXFHmadBkgFqfPVsa0zUR6fXHZheS5koMkrwNtEecbeF7f9aueZ0hKIJ3XfMor6jSRqgCF4f_Dgybe_or20133wHcPe3zpR6khxDSAUHEVW-1iGwDgItP-EYVhAt4Iynxbi2KMlVAcclVaJAgrit_B06J1aDaV89u0kcKdC6p-cnIRv180pJYs_fMfjtxiidMiX4oOeAHAJZRRU9QHJ9tWuqYT_B43z4nFDnN-IJfCx6fAaKiw_uJ5gzp1gbSrT_cPd3p-L5LEwDbhiICq44xIU6i4iQJsiXJ3WzwyovtJIwp0onUUj2h6-6ljU8jYJbDsqWp632sR0L7sBit08wa-lWOU0gf_cqeS-yWLOvszedMlB8XpxG7GppLZ5fb0JbjK4rTeSB2y7F7HwHVR_hHAWqvfEB_HaOLBCDAcnt6VmHHwUu7pLBVhRx17UFAg_o4P7lti66ncdoh9Ck40FtHp0TptPN_5mscGzf5H1mV6doMfQ4cfTAjBfpJy1Cyr6QyYzAXFoJ1g5IF1b7YbYWe3E4KfSRXTJdB2eveQPmgYarlWlmtk_dgD4ZisOIr3zLZ2SbFzhtpV5Cli9hRRuyVlRs_WBekruvNM-8cbl2wul9JRc6iXN4nT4iAbCaeAlmsSLnX69Mm6MONOF3f2ORjueJHaJOiU5h_sANGFTLHOzMpjWoXMg31fyfS5pF9jFrzPdZ8N4AF8_04TK1SLPqLIuntM9tJcZVJh8QxoKITiU5ZkjzUqdfSf_uYIdQ008FWJRsLNGu6X-NFYfTTKns6g8mGNru_jbVLauuyAl92rq7gxlyKTTnIw8t3x-INjvMY8LwdCtraTQUefbRZn3aRjfxGsSDzoINKFvS5ZsqjxqZe-BXgREVtxgZT8Q3Bl-CPp187rcei-wv9PZ0pevwnxNGJ-Jydzp1lbMQBG4z4aZ7tehtBmOWQhmOv4HAQ9X4bwFZeM1utpDTiBRLwfkIKxlovIAnq8XmwHu2S9loTOyl4sGBZWQDQ9pIwrxPl5cN47On2dUKissp7mUFqDP2awO33lgq1QOwAbg-b5pZ_LuVZ8SkB9uQE8xzzYsyffAOHhDOyaJCO_RGrWvhoq1f5PZbQ3M6B3IhxrmUhnmyuuLwfgw-J6KEkb4n8kDKG1Grj4bHGn9xH3PIXKda6m1TiA3PylUysgl_cxAwULG9SToqKQdF024iLrNTmYEBkwRp_nlJurJfMcaKJ337c_nKgbT8mIBfw8x6hEP3D90tO7oHQgS13-li3Ss-OhYZqPLXu2nZdcqvAe-RmqS8ZiwVEdESyng-QdJcy44HeCQ_I6Ro=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={'query': 'Cutzamala 12 de septiembre de 2025 CDMX agua corte', 'type': 'search'},
                        tool_call_id='ws_08313d91a201f39c0068c4ab76cc088195bd4e023113eb0e4f',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://www.tropicanafm.com/2025/estos-son-los-barrios-de-bogota-que-tendran-cortes-de-agua-este-viernes-12-de-septiembre-449708.html',
                            'https://www.tropicanafm.com/2025/cortes-de-agua-en-bogota-estos-son-los-barrios-afectados-este-martes-9-de-septiembre-449480.html',
                            'https://www.tropicanafm.com/2025/cortes-de-agua-en-bogota-programacion-del-8-al-12-de-septiembre-del-2025-449410.html',
                            'https://www.foxsports.com.mx/2025/03/12/corte-de-agua-en-cdmx-2025-anuncian-tandeo-masivo-en-marzo-en-estas-284-colonias-afectadas/',
                            'https://www.record.com.mx/contra/cdmx-estas-son-las-zonas-que-se-quedaran-sin-agua-en-septiembre',
                            'https://heraldodemexico.com.mx/nacional/2025/9/1/cortes-de-agua-en-cdmx-estas-son-las-300-colonias-que-se-veran-afectadas-en-septiembre-726765.html',
                            'https://www.milenio.com/comunidad/corte-de-agua-en-cdmx-cual-alcaldia-afectara-cuando-sera',
                            'https://www.elfinanciero.com.mx/cdmx/2023/11/13/corte-de-agua-sistema-cutzamala-alcaldias-cdmx-que-no-seran-afectadas-noviembre-2023/?outputType=amp',
                            'https://mexico.as.com/actualidad/corte-de-agua-en-cdmx-2025-alcaldias-afectadas-como-pedir-una-pipa-gratuita-y-hasta-cuando-habra-reduccion-n/',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab76cc088195bd4e023113eb0e4f',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab7887c881958dbedf1d7c4b39cf',
                        signature='gAAAAABoxKubop23Acf-11ep33KsX37BHNy6lm8b_SteBCvcxjDW1HePUzECIh4a_5qbHlw7a26_cvXtmY3thouxDh8G3ic4yYCBcmD7nAUrSFEPB5jbfTnIKMG9-9RZddKklo1jb47OtiUwSKXVmS2Yj_fAg0SiZ_pa2NRCP23MfgK3d6a5Fm6yA3DSBgZQ3O4yEuhSzMoLTvp9rfxqTFq0l0CtldZapAHMTMzHRqLQahdY4-6pEeQZ7zbZ23MTX7M_f228A43hAMcoprnHVEUJsfPq8JtCVnVvyUjMKjfP8X8jLaKiXhLhiwNleAXsbxoHUa8THD3xR6cUwYq5DVQ_-sYbcyG21fvut5gvXUxb-xCvpqODI9KfV1ESePsmE5bEYgOtNOLJqDvM009ir-hUwBUYrFzNKOKleS6awRHwfy8sClYCelB-v1jiYl5eKptMvU5anbpUB7TXWZFh-mB1BStouS2ss2kM95SLMmi4OkWRcEIrrsCcgHJxe2Uhvgg0po6kTZmGBjVgoihI5FS0cmp97ihcam-3hK90WhCJ_nKR5Bf0XI_SB8lFTLjqKkpUucz2yw-yvEYUZshHCjEyeTWKnCAeL1ENCLzXeVIKdJSecBoZYZMxrV8oHR9-MqWNcrOJgVKeMaFRBKJVItGGGkIhzfY8GQENLpa8BaWLrO_-HlyqMbxU18orKqn6bXOQ0MXw5DroyvKrqRLIhIZwu0qRCW_VyEM8JFFO8ga1yxBWsq-qGIKOd5xwKYAp1r--0oWo1Okf4ODm-C-zvMRs7bQzx2wWUd4olMRZ-c8O1G4EGnbWiB3UFTI_UlWM4lg2FaXEb0raRH3YN9YdSFrdPKztNA_4tuAD1wkzKoKZEXbRHs_lGPoUCESnFHjJvDqfZIXej2OyugdaxKMnauJnTLgoDwOU0XAFFBtLyT1TzO2VjE5M9y74eshrDVk9trex1hSBrTmDLM5YRlZ0eiv8iDY89W2WGY_0OKhUdewZgduhR2uaWA82demXgpUaNzgVNI8TX-CZgk7tquklxV1gUiWaEBwknCjl_qd1MzBeRuN8yYfs4lqD_ns98y9yzH8JW2Ibp1WmuxYy8zWpfqCQV2XMYRNq40peOFw9HSmVkcmFM7vu_VwwZ2_qnowBJkrDBoz-ls_Q0IiissU4JJb3wwqCFwNFnGWqKmVJYRzEFZbR9HDfbCsfAPtcRbVZ2sXpbIOUjhMAJk9FPy65B5ek-suigC-NyUNq74pfZ7RizvxQJ0PHTmyCW-eXr_ODF0FEufmMHV7znFr-DZmOOWKxrlrI1FGkGikSc1_SDHsljFzwLDwzgVOlmSOuCNzpLvxtMoNlOfm8qmFvVJYjv2aRmxL9j97a8J8Qj2rAxXf2CMYKfiHnQeHAjCqvG_iYwx2I8Ox_AQSTTy_AdyVd36dubHmPUQIHpovLgFbJ4G4Z6bh6vdIgyqzYuL2Qv1cgxa3Ms6atkt93--s5JQUwITJKDaLXyzyi_pwM0NKtbqatpOPatCTo8m4twW478IPJrg4MwAQeWCMSzIDJshlz0a2KUQyHs7xbPLmK-6JgpDqhGlVyoBQmqXQIcQk1n8OA7It-1_yuqAIYywWXZkJ3_9uY6p2jR-SkbPl9cHnPtNzHXxBpJV2FRHMGLEaVDCIfbxiW_y7x8fHefKObiqS8G2_QbbK9FFgbQhx2Zk52mgldlgCOH6c2fW4K6gqaGA6rcEmelrGLyBUnGa2Bvhg9Sis7EKkz1GtmMLGx7viPpbJiEfC_2lOwjjiwEYrOVL_H98xMWzTlcoYjHi7Fgso7hgI2HAIyRCMEG9p8NQAvGK42z3PHvxovzhpYbQguSWbn9K7pNhLFJVwli4Ut0kbaP3mBePZOC8XXA9RyfyS9xopKOPFTDzSwje8uc8-bQgPI9sn8qDmQ05sPOmv0On1PpVK1fTELQdHOEsZi_vIyRw8ElxwmNuxPs2fIY5hKU8pgkyXgJTqn1nEgIRbglnop7OA2DHq2PeROQ-irgS3v78uIQZiKmBt58a_NP_WXVTBeFbjrA8OoCesDgdc2yTjDFwekRGet9HkxcuIub76oPwEw1X_CcH7_JYYa3W7TQgwJOK6qJVNBJkghkeVaCZB5rAco54nMqKczK1ovhnGMDAbrJH9dsWG_crKM2FH2hC7Dhv4ErylyFec7xELKu6VJ51gTWyTR23RDgCtmeD5Q_p0fZZV5adiw59Qt8NqS703mzn8pN3ak6LHOLbcTxdTRg9JUlv3KXce7qmb68_O83V4BNDNw0DcsZG5It4hDJJZMd_pM121alcz7-IIl9FhgwzP3WU-FnboBJ8U2krY6hsdyzriOEkACeIBzFmYzA1bQ9VIg3bC-djrk2878SxKqQ3WK6aJClOnQ-Ed_qrfbw03XXWCwWKRPLR_M02BpKzm_4RfnzzHAXiKI4ZJqVD4Oj7G9bRsNtcxrDLoN-enn3gDf-ShuqMb6J0pofd3PVBJoOSpkz5u-jnRGzbtylrOzrjr7o70kBHZp8xiBLuZGQpzpBELc_OpEefyVTj6MOeJA6qo27eBh3R4w9yCfYtIH6XYfwvFm2PIKH0MG-xwRID2EwRIcrt-mU_byzzpNCGKyxpVYmc9Wr3-zQ1WjugAswztckt2HIRhscaxL0V2v7Dnm_FHMR4a2IzUV05uEH3LyjHJRHjWfSU9xl1c-twZ9Ru2q_tK55lQ9JRpvheXVsWMUH863UoTrJYW52EoeFrZWDtNx3-TyFErxGh6ZhVkE_QXu-6qUcP58MI6ZOkKfYnSjSD5N_zY1i4PJZPfrNHGPS_ky_rirkvSWrVsCG5UeJk4fzIU7n21NkxUuYgAU92Psa2ErbPJ4MH9hXf8-8k7MEYFF-Sk-JxaRPd2OoyU6DHucdK1FeRRoRdyYT6IZiHq6vkFngjXz1GfN_sDYFE24zByYu23MSEt3lcdD4yHlKyjBZuwnEvWuT0_L_wUaTPAm9XTD1QbLLURHkzUsrkjMinn9jL5L59qDyaLjrMmNQN_AWyFyseA5bVxOL9DNFWLyL1yd6vZF0SU64DhFABQbybNOnnjLnDBLGoRtb3TzTeq0NMBk0g9EygytD7x4gyrh_5aFi9ljt8jyrjW3WVMOx5bOFcORyFWF9Or6PUGgPAm8FBf494ugX4J4vJqFGJt5Bb06UVOQ23BoLbhNJUz5NzNdNwhfYhJ_gSl40EZK6MYWoJnnR_g4z3bVe9pJaXcEQF0YghWE1V7vGijMNcDKXiFMWF5Go4_vRUwQ-KkcSxNvj19bYGjdzYYNJ-i6kyCOUfT7_Bu5FT9N8B82NA6Id110r49F6QEHjQxNswCMolSBLJNNKQzIAoDbgmeJZbfVoL7lSjKBfe9APCMHqktaUHglCy_m5-wjDjlgqtwYnfGiMg3QFWXKmoXnk9diyah17U-k_PDOE5tmbHYFvowLZruklWMdAsTcozqd0juTLjHHpTq6igBVV2JYoZqg4FyD3dwjYPZK_5j7Fb5S9hoqQezL2LHqw4rkQ9Hs2SAhhyLKo_Gx1HWAO4_Df6u5_dvvSwoJ3tsukMf941ZaDuKTarkvFioJkN1dIcqdLr1fpjVQgh2TPHVyKHEJfNKhWIquXyepBSZWRV9OFKBRSfIB8oo4qVeyY4MJty9rthPOmcuf7JLz3Jgn5NgkC8gtTOpsxE_zOUjVdsork0vB9RChKFNxf7d4Xi3fUwb1ITHlKAbW0ulAm9_qKQb2yJD0M0sswryIvEkKoUlFOkx2vRjQSqW7Z4W_QMH1dytmg_RS5fgn0SJ0-grLxNmLuhUDkvZpeKIy8XzGJDdjc77Oo3M_H9k7p9pPA_dRZP6bxn188FU8oXQ2mGyhfwWS7ONqY_I89wELOL9zUGOid2lFXhAbhTQH9AVVIF_6YlFgCtnONQMHLQtpEm7rBOwGgK7eSE6lo91VUgpLNhWMjfvKxLnpVsLv3mOYqupPd5v4XOf_zJtsGvwK9H9Z8fpp8KYj1eBSZ_7wrSsSltAVnHFJyaw6GTDYvLEF-ERMjOYhuIsCn3IkBu_m7PKOZnRltcMSV1l8aA3QugDb-_LjY4nBI-w7W5Y23FO0cDgfo26-bA9i4fL009sMfIORSzvYmz1YoYrfYCGrQvsfQj7W3DrnkXPdRFZswmlxMLCb2SWhHLbHJwz4piXwu-wQiIbXIGgeXr1PFckxnJQu_iC7Y0nCpS60Q_vEjrxLkoWwhvgT70z5oZyDZOY-E1TXwVLu0ORpBqHUPUCZIHjy1-pM-X_w55el7yG_aEYCFENM_RZx16FWTfCEFyyW42PDl3nEshUkhSSNEvhM43jhtSFGMjFIni7T6eTvTA5x9nnSoVETql_PTVvO6SIIUljODteEOfRB1BzYQi0bdzhzjiuuoPNqva9qEx9lp8QGIusOHJZTfMj18s8bGyqhFFam-8saKjwkC5Y-wy3QMkGCR44OZVhgWA--2WddS0C629WVAXDhVccoNKRwgAS6rf0-iref4YJkotPHP7oi42QWhqun9sYxp6gu7hGEd1I_1BrIsr_i_zSjJt7fNa0ihl_BLdpfc2Ib9ZfBua3kbXJAh0lhaMH2k1UJNRIHQo_LFiTCbNJzSOYXax0WKyjq1VNMSaAEDZBhye4a49FXpOysvMdqHlbk9ASDXDXzg01uzR98WE1p-ekgKp2PwxfcjRo9d5n2hJfjyaFzlQy_oedsmWjGCMgG8bTrysnz4EnWteKzU6cMrrZj-JlZgZEAk_giWOjHxAphx77JbVb512cKgDbKzyY5zeJT9CwD16z22SWb12ios_K3idRCb4-N7y73aX7WSrrV2KJ1SoBs5ckUjdjoBZReWxYl8TOMSZvMjAvxsQWY8x0KFjAn6HZmHl2PwwIQCQb4RthiQ4tzMzk3NN7qiMHMR5WxbI-OpQuVi_NCb20a1IBRQgs6QizWLoOzO90NbezQdy0XZF0mgEK2kvXUbTjZamkL8xoDK5XvLn75QGYFS2JRFp7dE5Ybg67sBFWpcJ9Ud2LKYqVaoAsdml4jsCXz8fXl53XM418CkD6xBJf1u_kUFOPOjLTWE2s1AuZx_6rHjNgRlPROClVAAYj9AfGCcIwk0sS8B0Zn4APVPdaDntmlEYSOaKjvBorkBbghCa-OBiFjJIneu-bTA4ajM9m6wKTdPRGJ5sNT6VAXvVBYU_1sn03A7SCmvgs1pw==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search_call',
                        args={
                            'query': 'Grito de Independencia 2025 Zócalo CDMX cartelera 15 de septiembre 2025 concierto',
                            'type': 'search',
                        },
                        tool_call_id='ws_08313d91a201f39c0068c4ab8797748195b72ca464faba286b',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_call',
                        content=[
                            'https://elpais.com/us/migracion/2025-09-12/los-eventos-del-dia-de-la-independencia-mexicana-cancelados-en-todo-estados-unidos.html',
                            'https://elpais.com/us/migracion/2025-09-12/el-miedo-a-las-deportaciones-silencia-la-fiesta-de-la-independencia-mexicana-en-estados-unidos.html',
                            'https://mexico.as.com/tikitakas/la-arrolladora-banda-el-limon-dara-concierto-gratis-el-15-de-septiembre-en-el-zocalo-cdmx-a-que-hora-sera-n/',
                            'https://www.nmas.com.mx/ciudad-de-mexico/conciertos-gratis-cdmx-septiembre-2025-que-artistas-estaran-este-mes-lista/',
                            'https://www.nmas.com.mx/ciudad-de-mexico/la-arrolladora-banda-el-limon-estara-zocalo-cdmx-15-de-septiembre-2025-grito-independencia/',
                            'https://www.nmas.com.mx/ciudad-de-mexico/a-que-hora-sera-el-concierto-gratis-de-la-arrolladora-en-zocalo-cdmx-el-15-de-septiembre-2025/',
                            'https://www.record.com.mx/contra/la-arrolladora-moenia-y-julio-preciado-encabezan-la-cartelera-del-15-de-septiembre-en-cdmx',
                            'https://www.nmas.com.mx/nacional/sheinbaum-anuncia-evento-grito-de-independencia-2025-zocalo-cdmx-estara-intocable/',
                            'https://elpais.com/mexico/2024-08-23/banda-ms-en-el-zocalo-asi-sera-el-concierto-del-15-de-septiembre.html',
                        ],
                        tool_call_id='ws_08313d91a201f39c0068c4ab8797748195b72ca464faba286b',
                        metadata={'status': 'completed'},
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_08313d91a201f39c0068c4ab8a454c81959bc71db0f026334c',
                        signature='gAAAAABoxKubQfxe28iIHU-w4DfE65EXnvcwW7j8Ex0wRyeOuN_ieyWoAcJOAgP8ltAi4yCYY6QgAa64n6UrmGDoRLGLexdrKGr-h2yXznQrihn7Hvc4J3tHdyLIZ6cGlszi26plPp8ixEPLQsUpF30IfVSdn6oObwZqig4ZQtG7Qotqe4_7hKDHDWqMvIP8HOCqbLUT-dvaxhwx4UgX0VkpXErP7PlCeaDFc41YYQRVnEJr5G50FcL7KnoT1JqbnMxjMwYvrHHW6ZB1k-3EufKdDVw-r5D6YKu3s-u_cb6UTCpGX72gRc_b44xFfyWsH9hbbuyabycXkPzJtoEqnDkBraj4tB-px-6BOE7shXWQOpAkXGl6y1UblwgJjdMs2OhVHLGKzShlS04_UDKihis2bUtQuD4yXvXygtuys81Eu0RzFFrU2B01dWEW1gOX5eDRCRdV06ZUdLSpWMRLKNIJNO7DHTtMa9kYwjYctOBcQM4xyMw3zk2sZJ4FCLX4T2pHyJppCaF39c8AGxrxySRqAH4KoPVf_5mEhBJq_oA8TfeHaIEY2Z2-pt1GTDdf5Yb_8eAzlQ85yjegjzYHKX67tebP0XJ0z7oQScUuOvz4zl01AZbQhrBlBk1jDk1ajuDqQuNWaSH_LSowA4-xQJ75M6p5aTipTCUTvXiTV7YuXbCLs4wUxfCL5aE1d7GDvFAtX_j3oh4Zaw64S9VU9_GysUQZ8sVMvCj3ej6AS3dPl8q-Gl3AvRxB2ymugrZrM_AbKZe3f3fKRDSPV2NkNCNtHe1cCzJ6daIupFQFE-soIpUNqCEi-dUf2mL8-YBG6G0Nyhm9KQWTODyKuI3y0BXC-THAVigpSQ03nZG8BYwy4LNRJNd_zHAfkW8GhBo9uNG6nrGD82V-jf2feT1mZ_bauHwTp0PNSJKzNlXfaGGDpEoWr6NbDEnQPW7iooSCLS5T2gPLWnAAzWXkQjwF7ht7yXWhm_S_N58U_cZVTBvQzYD_FDU8xWzXqqe4E4Tt_x4HN2yxutMflQuvHEhixNK7ac1A9Cl0Grf83v9YAqppN0leDkLwZvcEPVlZX-4QGdGdFdO6C5Wq50hNJR6p8dD3irhlYu8T0BiV3UJia0eS7JxZ4Yiv2yIvIBu9k8iZNZHgRHpDgGg-YuLeV0BWKoR3ipBL2irB0dppaKEnx0W7ywy4AvbTVLyXLGcIDoLvzsSnHi10gidRraGKWiQiK6dESspyvyngGn2fchA4fKBE-dKwfJt4Zfvihx_4hE3SsQf-3wEBFjLKsba8rvV4HLpczA7zTzzkX5hfh11qe0my8EZBAGxZgeO0Z0kLeaWvebdTsRDfDRPn-2h_90C33vNNhFWbzpLUPc_T_ZDDyEpgmDlq2nGojGYB0AGT8bVnd9EPxNYXjoFiLAcPgeFVd2gV2QJL-nh2Abv2HUN0wUMFDbd7TrVpGx8b0dI1LFwgqQY13L-VwR5Nr7QktGuqqfPeEoos99IXK_tfIpACVPnz0pOW-n8Br98Bj4pSb2mAEqoEEf5kQnZC5exNYkaCKgfxcNNMhqyMLAW9FbaE-kBKHe4piBpP2rbmUwgg1vjTChlyGd5OEHjpE1Wz4V4027QNpxNRY26JeEy9aMTCAqBin3jDnG6vEqRRu4GCRDiXSLo2F9aurpNlnevmZFK0TJReXQ1c7g9Ex8Y2ZlCQojJZL8uDRGDLuad_kzkHbLfEm45XioQASefsSoe9AfC8J9PKxvAMnW2M6mCaD3nqSaPU-JV-TcLfVR32v_pSMaJ900AGu_RBwl7wGhCGlGM9gt-jOlXWrsEBFvGToPU9JjSrfrGTx2-9zbDdLzTcG2ZY5zBImTRNmO7lpmDf_Ios1bYLMDQVgnScvFTEiRiKHQpbxgmK57KQH_LiDXTvVBWkKO6UL6G7aY5llkasxJOrE1RH-DfBMYb6MBySwjHWoW9H8DbWn6omNPNKAhet7A8_gO_wiJ40o1w320ZdS-idgpZNtAtN753HD0AROOTkRTYom8KdE05Dt2ngJ8IFIXqx93uj7TbapQN6INqca40E9qlh8Hw1_kddWp7TbEu7x3BOnw5I5IWXubXhtq1Kyv6ZLbVUllXK-SslN2oU7Mlc2T9I5rkOtMJLkIEf2wkcK5zp6zzfmbJDm2LFG7jIYAnpQYk23XGkFnA4gCuUdbIlLKcrrbrwx8x1jHr0CMTfJuQgjy672d7Xc3P8VUWPnp_GAjAtGbYXrBf3xLg6i98yqqCVgeFSjCjT9tejwSaIoKODApz8m_-ngMtyiTAtcvChDsleYyQz6xem_e5mkzzU9SIUItgtpsyrKBWatNUn7Wi9L7TuCa2xAh_o8nBcIkFYZ_Pvp8BaJiD0RSYky9-QrraraJDftHJwghDT7fG61K8QZN4hFhLGBDegSBwj-OGV1ORljJ32mfY3fU2gtOJVfZvlLOm5NkGkNGLJwKneWTh4xArs3Z0fL3Phrri7A2EyUviNLAm9Ve-VMWgMVj2PBbI8j32rjZRl_JwssD7mBTNfaVWvxr_lN_ctkWIKrpRkT3g4vUEczu21oVAtNVZzuAOempZ4bCZ2xmPOP6OjZHPoHJUq5kyQ5iYHx8XvcY-8-FyVzANyg7nNynrBzNd2sKVHknxvl6xMyeUqRUJWlOmPCKyPIOv49LNx1ZCFir9GzORGEwU7uixiy2umZZF416C6aBp0wIsRZiFCOKtcQZkSJmcG50Z8GztRHGfSISTVcPSgs_OY3mY7jONO05S5Bo0vFCSkBc2xiEqZytGu9GS3YMjml2hBvzx-Q0nli_vLWiAhzgTBN7js-sr872K34Vimk6AJ5q6lyT4EfXSLY-o4W8o3kbFhW77-RakIh3dvG1pOnSS_PzCdn0wo9E-VGvrrIjNdt0iXEiVk1eyfeELlCg9R_sRTo9aKsR7FeXDoxHfyKMlhpa5xtTf5M5EF7m0bNel33A0SgTw6uh-kQuaa84poBXo3yeBPHD7Lxt-V6UH3k5sz3MnxXcZkJlvwio2e8TS84QbsPmvNbPrVDEFPq_eHN9VQ5-eu4V2oRkwXXpW_17RnOgsuYfEJTY2pMog5c3h-QiyK2VNm7vRGOgWqmYxXeR7F-Bxbr36ZIQJYkgoQtLNmxGfY_Qdzigic9baNHNHArkvyNmsot21TnCYxm2b1WryNdbkIMVLAcmodio8CRUljKJh73g85gmC2FwA0dmn_bFOmb4EUWgPrQzsXKpZrsrwlrPbmK8FrhJ8dijNn9ft3BPqh5UjQdNQIx7RVBbb0P1IUqZwWhErn9weovGXcZyNAtLx_VEuMJBBUE8pHkqMEKEct',
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
Here are three key Mexico City updates for Friday, September 12, 2025:

- Iztapalapa tanker explosion aftermath: Authorities report at least 3 dead and about 70 injured after a gas tanker overturned and exploded on a highway near Iztapalapa; investigations and victim support continue. Some outlets report higher tolls as hospitals update cases. ([reuters.com](https://www.reuters.com/world/us/tanker-truck-explodes-mexican-highway-killing-3-injuring-70-2025-09-11/?utm_source=openai))

- Mobility alerts today: Multiple marches and gatherings are planned around midday and afternoon in Centro Histórico (Zócalo/Plaza de la Constitución), Reforma–Insurgentes, and several boroughs (GAM, Tlalpan, Coyoacán, Xochimilco, Venustiano Carranza), with suggested alternate routes via Circuito Interior, Viaducto, Periférico and Calzada de Tlalpan. Also, Hoy No Circula applies 5:00–22:00 for blue sticker vehicles with plate endings 9 or 0 (hologram 1 and 2). ([mvsnoticias.com](https://mvsnoticias.com/nacional/cdmx/2025/9/12/marchas-que-habra-hoy-12-de-septiembre-en-cdmx-afectaciones-rutas-alternas-710981.html?utm_source=openai))

- Independence Day in the Zócalo: La Arrolladora Banda El Limón will play a free concert on Monday, September 15, starting at 8:00 pm, ahead of the Grito de Independencia ceremony. ([nmas.com.mx](https://www.nmas.com.mx/ciudad-de-mexico/la-arrolladora-banda-el-limon-estara-zocalo-cdmx-15-de-septiembre-2025-grito-independencia/?utm_source=openai))

If you want, I can add a quick weather and air-quality heads-up for today or flag any Metro/Metrobús issues.\
""",
                        id='msg_08313d91a201f39c0068c4ab9302448195927733883581e15e',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=113049, cache_read_tokens=90240, output_tokens=2904, details={'reasoning_tokens': 2560}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_08313d91a201f39c0068c4ab5b3afc8195b063acedfc62b3ba',
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
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
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
                        tool_call_id='call_ZWkVhdUjupo528U9dqgFeRkH|fc_68477f0bb8e4819cba6d781e174d77f8001fd29e2d5573f7',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city":"Mexico City","country":"Mexico"}',
                        tool_call_id='call_iFBd0zULhSZRR908DfH73VwN|fc_68477f0c91cc819e8024e7e633f0f09401dc81d4bc91f560',
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
                        tool_call_id='call_iFBd0zULhSZRR908DfH73VwN|fc_68477f0c91cc819e8024e7e633f0f09401dc81d4bc91f560',
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
                        tool_call_id='call_aTJhYjzmixZaVGqwl5gn2Ncr|fc_68477f0dff5c819ea17a1ffbaea621e00356a60c98816d6a',
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
                        tool_call_id='call_aTJhYjzmixZaVGqwl5gn2Ncr|fc_68477f0dff5c819ea17a1ffbaea621e00356a60c98816d6a',
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
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
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
                        tool_call_id='call_tTAThu8l2S9hNky2krdwijGP|fc_68477f0fa7c081a19a525f7c6f180f310b8591d9001d2329',
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
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
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
                        tool_call_id='call_UaLahjOtaM2tTyYZLxTCbOaP|fc_68477f1168a081a3981e847cd94275080dd57d732903c563',
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
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
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
                        tool_call_id='call_FrlL4M0CbAy8Dhv4VqF1Shom|fc_68482f1b0ff081a1b37b9170ee740d1e02f8ef7f2fb42b50',
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
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
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
                        tool_call_id='call_my4OyoVXRT0m7bLWmsxcaCQI|fc_68482f2889d481a199caa61de7ccb62c08e79646fe74d5ee',
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
                        tool_call_id='call_gL7JE6GDeGGsFubqO2XGytyO|fc_68c42d3e9e4881968b15fbb8253f58540e8bc41441c948f6',
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
                        tool_call_id='call_gL7JE6GDeGGsFubqO2XGytyO|fc_68c42d3e9e4881968b15fbb8253f58540e8bc41441c948f6',
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

    result = await agent.run(user_prompt="what's 123456 to the power of 123?")
    assert result.all_messages() == snapshot(
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
                        id='rs_0cccbe0ca95f935e0068c4986ab048819f9c0f1478f3a295c2',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_interpreter_call',
                        args={
                            'id': 'ci_0cccbe0ca95f935e0068c4986ed690819f9dbd272c4c70a011',
                            'code': """\
n = pow(123456, 123)
len_str = len(str(n))
len_str, str(n)[:50], str(n)[-50:]\
""",
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                            'outputs': [
                                {
                                    'logs': """\
(627,
 '18030210630404480750814092786593857280734268863855',
 '29749134489643622579100908331839817426366854332416')\
""",
                                    'type': 'logs',
                                }
                            ],
                            'status': 'completed',
                            'type': 'code_interpreter_call',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c4986ed690819f9dbd272c4c70a011',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_interpreter_call',
                        content=[
                            """\
(627,
 '18030210630404480750814092786593857280734268863855',
 '29749134489643622579100908331839817426366854332416')\
"""
                        ],
                        tool_call_id='ci_0cccbe0ca95f935e0068c4986ed690819f9dbd272c4c70a011',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0cccbe0ca95f935e0068c498746e68819fbcaef179058df3cc',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_interpreter_call',
                        args={
                            'id': 'ci_0cccbe0ca95f935e0068c49877936c819fbcd8ce718f5bc31a',
                            'code': """\
str_n = str(n)
str_n[:200]\
""",
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                            'outputs': [
                                {
                                    'logs': "'18030210630404480750814092786593857280734268863855968048844015985795850236081373250219782696986322573087163043641979475893207435038036769764981462654292660266470727587426920177774391231319751632369022'",
                                    'type': 'logs',
                                }
                            ],
                            'status': 'completed',
                            'type': 'code_interpreter_call',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c49877936c819fbcd8ce718f5bc31a',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_interpreter_call',
                        content=[
                            "'18030210630404480750814092786593857280734268863855968048844015985795850236081373250219782696986322573087163043641979475893207435038036769764981462654292660266470727587426920177774391231319751632369022'"
                        ],
                        tool_call_id='ci_0cccbe0ca95f935e0068c49877936c819fbcd8ce718f5bc31a',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0cccbe0ca95f935e0068c49877a260819faa558e81b74d00a3',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
123456^123 equals:
1803021063040448075081409278659385728073426886385596804884401598579585023608137325021978269698632257308716304364197947589320743503803676976498146265429266026647072758742692017777439123131975163236902290188202654590011462134235078832526827852273018210815142998256983234516628795109978467862737585124291404312560193679040132194219142159564780429384029784135632838235232349153620928650701305446902198201185265537637166663065255873102180259349606640396746581577358565927727487182715643033427374054356948524185042601095118624154879402247254855530736695404690558487305849085262939429771481280865270916688165704128104324472168440830911119701969936\
""",
                        id='msg_0cccbe0ca95f935e0068c49877dd4c819fb0949326ca53a0cb',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2629, cache_read_tokens=2304, output_tokens=354, details={'reasoning_tokens': 128}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0cccbe0ca95f935e0068c498673f08819f9ca3a5ffea0f8f34',
                finish_reason='stop',
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run(user_prompt='how about to the power of 124?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about to the power of 124?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0cccbe0ca95f935e0068c4987dfd64819fbd3984a1d0b6ae8a',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_interpreter_call',
                        args={
                            'id': 'ci_0cccbe0ca95f935e0068c498806d50819f81b64eebd0a9afb5',
                            'code': """\
n = pow(123456, 124)
str_n = str(n)
len(str_n)\
""",
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                            'outputs': [{'logs': '632', 'type': 'logs'}],
                            'status': 'completed',
                            'type': 'code_interpreter_call',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c498806d50819f81b64eebd0a9afb5',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_interpreter_call',
                        content=['632'],
                        tool_call_id='ci_0cccbe0ca95f935e0068c498806d50819f81b64eebd0a9afb5',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_interpreter_call',
                        args={
                            'id': 'ci_0cccbe0ca95f935e0068c49880a0fc819f9b7df3346d75008a',
                            'code': 'str_n[:200]',
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                            'outputs': [
                                {
                                    'logs': "'22259376835872155755725046390617312444503298968562023914380868375424124867456620159791334926391434395830488007158642181758718171000558674481055514534483546658574101450333778574673152438538112575257499'",
                                    'type': 'logs',
                                }
                            ],
                            'status': 'completed',
                            'type': 'code_interpreter_call',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c49880a0fc819f9b7df3346d75008a',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_interpreter_call',
                        content=[
                            "'22259376835872155755725046390617312444503298968562023914380868375424124867456620159791334926391434395830488007158642181758718171000558674481055514534483546658574101450333778574673152438538112575257499'"
                        ],
                        tool_call_id='ci_0cccbe0ca95f935e0068c49880a0fc819f9b7df3346d75008a',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_interpreter_call',
                        args={
                            'id': 'ci_0cccbe0ca95f935e0068c498829764819fa5d652f285f51fb4',
                            'code': 'str_n[-200:]',
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                            'outputs': [
                                {
                                    'logs': "'67296874992296948890272517148532094528812086889630338972525614076265743359820638354575643811422961497469318295177638226345664642985187040350404616587909147553443069125481739015616500189546368462749696'",
                                    'type': 'logs',
                                }
                            ],
                            'status': 'completed',
                            'type': 'code_interpreter_call',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c498829764819fa5d652f285f51fb4',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_interpreter_call',
                        content=[
                            "'67296874992296948890272517148532094528812086889630338972525614076265743359820638354575643811422961497469318295177638226345664642985187040350404616587909147553443069125481739015616500189546368462749696'"
                        ],
                        tool_call_id='ci_0cccbe0ca95f935e0068c498829764819fa5d652f285f51fb4',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_interpreter_call',
                        args={
                            'id': 'ci_0cccbe0ca95f935e0068c498850e30819f925c0f879f69541b',
                            'code': 'str_n',
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                            'outputs': [
                                {
                                    'logs': "'22259376835872155755725046390617312444503298968562023914380868375424124867456620159791334926391434395830488007158642181758718171000558674481055514534483546658574101450333778574673152438538112575257499957691072558869631827866367698332298176634455912721514439215697282609503998798205923943259906494776122169432383635142770936484489945914800519238574614164957266189867725276688514819945285124592421478579353115389025899375605896015758767296874992296948890272517148532094528812086889630338972525614076265743359820638354575643811422961497469318295177638226345664642985187040350404616587909147553443069125481739015616500189546368462749696'",
                                    'type': 'logs',
                                }
                            ],
                            'status': 'completed',
                            'type': 'code_interpreter_call',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c498850e30819f925c0f879f69541b',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_interpreter_call',
                        content=[
                            "'22259376835872155755725046390617312444503298968562023914380868375424124867456620159791334926391434395830488007158642181758718171000558674481055514534483546658574101450333778574673152438538112575257499957691072558869631827866367698332298176634455912721514439215697282609503998798205923943259906494776122169432383635142770936484489945914800519238574614164957266189867725276688514819945285124592421478579353115389025899375605896015758767296874992296948890272517148532094528812086889630338972525614076265743359820638354575643811422961497469318295177638226345664642985187040350404616587909147553443069125481739015616500189546368462749696'"
                        ],
                        tool_call_id='ci_0cccbe0ca95f935e0068c498850e30819f925c0f879f69541b',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
123456^124 equals:
22259376835872155755725046390617312444503298968562023914380868375424124867456620159791334926391434395830488007158642181758718171000558674481055514534483546658574101450333778574673152438538112575257499957691072558869631827866367698332298176634455912721514439215697282609503998798205923943259906494776122169432383635142770936484489945914800519238574614164957266189867725276688514819945285124592421478579353115389025899375605896015758767296874992296948890272517148532094528812086889630338972525614076265743359820638354575643811422961497469318295177638226345664642985187040350404616587909147553443069125481739015616500189546368462749696\
""",
                        id='msg_0cccbe0ca95f935e0068c49886b928819f942185e7cf43c579',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=5973, cache_read_tokens=5120, output_tokens=221, details={'reasoning_tokens': 0}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0cccbe0ca95f935e0068c4987bea4c819fbd0d975202557b61',
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
                        tool_name='code_interpreter_call',
                        args={
                            'id': 'ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                            'code': """\
n = pow(123456, 123)
len(str(n))\
""",
                            'container_id': 'cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e',
                            'outputs': None,
                            'status': 'completed',
                            'type': 'code_interpreter_call',
                        },
                        tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_interpreter_call',
                        content=None,
                        tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_interpreter_call',
                        args={
                            'id': 'ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                            'code': 'str(n)[:100], str(n)[-100:]',
                            'container_id': 'cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e',
                            'outputs': None,
                            'status': 'completed',
                            'type': 'code_interpreter_call',
                        },
                        tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_interpreter_call',
                        content=None,
                        tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_interpreter_call',
                        args={
                            'id': 'ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                            'code': 'n',
                            'container_id': 'cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e',
                            'outputs': None,
                            'status': 'completed',
                            'type': 'code_interpreter_call',
                        },
                        tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_interpreter_call',
                        content=None,
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
                    tool_name='code_interpreter_call',
                    args={
                        'id': 'ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        'code': """\
n = pow(123456, 123)
len(str(n))\
""",
                        'container_id': 'cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e',
                        'outputs': None,
                        'status': 'completed',
                        'type': 'code_interpreter_call',
                    },
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='code_interpreter_call',
                    content=None,
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='code_interpreter_call',
                    args={
                        'id': 'ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        'code': 'str(n)[:100], str(n)[-100:]',
                        'container_id': 'cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e',
                        'outputs': None,
                        'status': 'completed',
                        'type': 'code_interpreter_call',
                    },
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='code_interpreter_call',
                    content=None,
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=5,
                part=BuiltinToolCallPart(
                    tool_name='code_interpreter_call',
                    args={
                        'id': 'ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                        'code': 'n',
                        'container_id': 'cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e',
                        'outputs': None,
                        'status': 'completed',
                        'type': 'code_interpreter_call',
                    },
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=6,
                part=BuiltinToolReturnPart(
                    tool_name='code_interpreter_call',
                    content=None,
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
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='code_interpreter_call',
                    args={
                        'id': 'ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        'code': """\
n = pow(123456, 123)
len(str(n))\
""",
                        'container_id': 'cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e',
                        'outputs': None,
                        'status': 'completed',
                        'type': 'code_interpreter_call',
                    },
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='code_interpreter_call',
                    content=None,
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='code_interpreter_call',
                    args={
                        'id': 'ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        'code': 'str(n)[:100], str(n)[-100:]',
                        'container_id': 'cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e',
                        'outputs': None,
                        'status': 'completed',
                        'type': 'code_interpreter_call',
                    },
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='code_interpreter_call',
                    content=None,
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='code_interpreter_call',
                    args={
                        'id': 'ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                        'code': 'n',
                        'container_id': 'cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e',
                        'outputs': None,
                        'status': 'completed',
                        'type': 'code_interpreter_call',
                    },
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='code_interpreter_call',
                    content=None,
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
