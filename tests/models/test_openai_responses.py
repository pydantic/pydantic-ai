import json

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..conftest import IsDatetime, IsStr, TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIModelSettings, OpenAIResponsesModel
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
    assert result.data == snapshot('The capital of France is Paris.')


async def test_openai_responses_model_simple_response_with_tool_call(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    agent = Agent(model=model)

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        return 'Potato City'

    result = await agent.run('What is the capital of PotatoLand?')
    assert result.data == snapshot('The capital of PotatoLand is Potato City.')


async def test_openai_responses_result_type(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class MyResult(TypedDict):
        name: str
        age: int

    agent = Agent(model=model, result_type=MyResult)
    result = await agent.run('Give me the name and age of Brazil, Argentina, and Chile.')
    assert result.data == snapshot({'name': 'Brazil', 'age': 2023})  # pragma: no cover


async def test_openai_responses_reasoning_effort(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, model_settings=OpenAIModelSettings(openai_reasoning_effort='low'))
    result = await agent.run(
        'Explain me how to cook uruguayan alfajor. Do not send whitespaces at the end of the lines.'
    )
    assert [line.strip() for line in result.data.splitlines()] == snapshot(
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


async def test_openai_responses_system_prompt(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, system_prompt='You are a helpful assistant.')
    result = await agent.run('What is the capital of France?')
    assert result.data == snapshot('The capital of France is Paris.')


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
                    TextPart(content=''),
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
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
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
"""
                    )
                ],
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.data == snapshot('The fruit in the image is a kiwi.')


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
    assert result.data == snapshot('The document contains the text "Dummy PDF file."')


async def test_openai_responses_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.data == snapshot('The main content of this document is a simple text placeholder: "Dummy PDF file."')


async def test_openai_responses_text_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.data == snapshot(
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
    assert result.data == snapshot("Hello! I see you've shared an image of a potato. How can I assist you today?")


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

    assert output_text == snapshot(['The capital of France is Paris.'])


async def test_openai_responses_model_http_error(allow_model_requests: None, openai_api_key: str):
    """Set temperature to -1 to trigger an error, given only values between 0 and 1 are allowed."""
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, model_settings=OpenAIModelSettings(temperature=-1))

    with pytest.raises(ModelHTTPError):
        async with agent.run_stream('What is the capital of France?'):
            ...
