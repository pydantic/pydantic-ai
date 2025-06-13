from __future__ import annotations as _annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any, Union

import pytest
from httpx import Request
from inline_snapshot import Is, snapshot
from pydantic import BaseModel
from pytest_mock import MockerFixture
from typing_extensions import TypedDict

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    AudioUrl,
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
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.result import StructuredTextOutput, TextOutput, ToolOutput, Usage

from ..conftest import IsDatetime, IsInstance, IsStr, try_import

with try_import() as imports_successful:
    from google.genai import _api_client
    from google.genai.types import HarmBlockThreshold, HarmCategory

    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture(scope='module')
def google_provider(gemini_api_key: str) -> GoogleProvider:
    return GoogleProvider(api_key=gemini_api_key)


async def test_google_model(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    assert model.base_url == 'https://generativelanguage.googleapis.com/'
    assert model.system == 'google-gla'
    agent = Agent(model=model, system_prompt='You are a chatbot.')

    result = await agent.run('Hello!')
    assert result.output == snapshot('Hello there! How can I help you today?\n')
    assert result.usage() == snapshot(
        Usage(
            requests=1,
            request_tokens=7,
            response_tokens=11,
            total_tokens=18,
            details={'text_prompt_tokens': 7, 'text_candidates_tokens': 11},
        )
    )
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
                parts=[TextPart(content='Hello there! How can I help you today?\n')],
                usage=Usage(
                    requests=1,
                    request_tokens=7,
                    response_tokens=11,
                    total_tokens=18,
                    details={'text_prompt_tokens': 7, 'text_candidates_tokens': 11},
                ),
                model_name='gemini-1.5-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
        ]
    )


async def test_google_model_structured_response(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
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
    assert result.usage() == snapshot(
        Usage(
            requests=2,
            request_tokens=224,
            response_tokens=35,
            total_tokens=259,
            details={'text_prompt_tokens': 224, 'text_candidates_tokens': 35},
        )
    )
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
                    ToolCallPart(
                        tool_name='temperature', args={'date': '2022-01-01', 'city': 'London'}, tool_call_id=IsStr()
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=101,
                    response_tokens=14,
                    total_tokens=115,
                    details={'text_prompt_tokens': 101, 'text_candidates_tokens': 14},
                ),
                model_name='gemini-1.5-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='temperature', content='30°C', tool_call_id=IsStr(), timestamp=IsDatetime()
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'temperature': '30°C', 'date': '2022-01-01', 'city': 'London'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=123,
                    response_tokens=21,
                    total_tokens=144,
                    details={'text_prompt_tokens': 123, 'text_candidates_tokens': 21},
                ),
                model_name='gemini-1.5-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_google_model_stream(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash-exp', provider=google_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_output()
    assert data == snapshot('The capital of France is Paris.\n')


async def test_google_model_retry(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-pro-preview-03-25', provider=google_provider)
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
                    SystemPromptPart(content='You are a helpful chatbot.', timestamp=IsDatetime()),
                    UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_capital', args={'country': 'France'}, tool_call_id=IsStr())],
                usage=Usage(
                    requests=1,
                    request_tokens=57,
                    response_tokens=15,
                    total_tokens=173,
                    details={'thoughts_tokens': 101, 'text_prompt_tokens': 57},
                ),
                model_name='models/gemini-2.5-pro-preview-05-06',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='The country is not supported.',
                        tool_name='get_capital',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='I am sorry, I cannot fulfill this request. The country you provided is not supported.'
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=104,
                    response_tokens=18,
                    total_tokens=122,
                    details={'text_prompt_tokens': 104},
                ),
                model_name='models/gemini-2.5-pro-preview-05-06',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
        ]
    )


async def test_google_model_max_tokens(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'max_tokens': 5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is')


async def test_google_model_top_p(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'top_p': 0.5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')


async def test_google_model_thinking_config(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-pro-preview-03-25', provider=google_provider)
    settings = GoogleModelSettings(google_thinking_config={'include_thoughts': False})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


async def test_google_model_gla_labels_raises_value_error(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)
    settings = GoogleModelSettings(google_labels={'environment': 'test', 'team': 'analytics'})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=settings)

    # Raises before any request is made.
    with pytest.raises(ValueError, match='labels parameter is not supported in Gemini API.'):
        await agent.run('What is the capital of France?')


@pytest.fixture(autouse=True)
def vertex_provider_auth(mocker: MockerFixture) -> None:  # pragma: lax no cover
    # Locally, we authenticate via `gcloud` CLI, so we don't need to patch anything.
    if not os.getenv('CI'):
        return  # pragma: lax no cover

    @dataclass
    class NoOpCredentials:
        token = 'my-token'
        quota_project_id = 'pydantic-ai'

        def refresh(self, request: Request): ...

        def expired(self) -> bool:
            return False

    return_value = (NoOpCredentials(), 'pydantic-ai')
    mocker.patch.object(_api_client, '_load_auth', return_value=return_value)


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
async def test_google_model_vertex_provider(allow_model_requests: None):
    provider = GoogleProvider(location='global')
    model = GoogleModel('gemini-2.0-flash', provider=provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.')
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
async def test_google_model_vertex_labels(allow_model_requests: None):  # pragma: lax no cover
    provider = GoogleProvider(location='global', project='pydantic-ai')
    model = GoogleModel('gemini-2.0-flash', provider=provider)
    settings = GoogleModelSettings(google_labels={'environment': 'test', 'team': 'analytics'})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')


async def test_google_model_iter_stream(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.')

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        return 'Paris'  # pragma: lax no cover

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
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_capital', args={'country': 'France'}, tool_call_id=IsStr()),
            ),
            IsInstance(FunctionToolCallEvent),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_capital', content='Paris', tool_call_id=IsStr(), timestamp=IsDatetime()
                ),
                tool_call_id=IsStr(),
            ),
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_temperature', args={'city': 'Paris'}, tool_call_id=IsStr()),
            ),
            IsInstance(FunctionToolCallEvent),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_temperature', content='30°C', tool_call_id=IsStr(), timestamp=IsDatetime()
                ),
                tool_call_id=IsStr(),
            ),
            PartStartEvent(index=0, part=TextPart(content='The temperature in')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Paris is 30°C.\n')),
        ]
    )


async def test_google_model_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot('The fruit in the image is a kiwi.')


async def test_google_model_video_as_binary_content_input(
    allow_model_requests: None, video_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot("""\
Okay, I can describe what's visible in the image you sent.

The image shows a camera setup in an outdoor environment. Here's a breakdown of what's in the frame:

*   **Camera Monitor:** A small, external monitor is mounted on top of the camera. The monitor is displaying a preview of the scene the camera is capturing. The preview shows a scene of a dirt road, rocks, and a distant mountain range.
*   **Camera:** The camera itself is attached to a tripod. It's a compact camera, and the lens is visible.
*   **Tripod:** The camera is mounted on a tripod to keep it stable.
*   **Background:** The background is blurred, but it appears to be a mountainous or desert landscape with rocks and vegetation. The colors suggest that it could be late afternoon or early morning, with warm light illuminating the scene.

In summary, the image depicts a camera setup being used to film or photograph a desert landscape.\
""")


async def test_google_model_image_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot("That's a potato!\n")


async def test_google_model_video_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video',
            VideoUrl(url='https://github.com/pydantic/pydantic-ai/raw/refs/heads/main/tests/assets/small_video.mp4'),
        ]
    )
    assert result.output == snapshot("""\
Certainly! Based on the image, here's what I can tell you:

**Visual Description:**

The image shows a small video monitor (likely a field monitor used in filmmaking or photography) mounted on top of a camera tripod. The monitor screen displays a scenic shot of a mountainous landscape, possibly a canyon or ravine. The background is a blurred, out-of-focus view of a similar landscape, suggesting that the monitor is set up in the field.
""")


async def test_google_model_document_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot('The main content of the document is the phrase "Dummy PDF file".\n')


async def test_google_model_text_document_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot(
        'The main content of this TXT document is an explanation of the placeholder names "John Doe" (and variations) used when the true identity of a person is unknown or withheld, primarily in legal contexts in the United States and Canada. It also discusses alternative names used in other countries, different uses of the name, and the origin of the practice.\n'
    )


async def test_google_model_text_as_binary_content_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    text_content = BinaryContent(data=b'This is a test document.', media_type='text/plain')

    result = await agent.run(['What is the main content on this document?', text_content])
    assert result.output == snapshot('The main content of the document is that it is a test document.\n')


async def test_google_model_instructions(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

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
                parts=[TextPart(content='The capital of France is Paris.\n')],
                usage=Usage(
                    requests=1,
                    request_tokens=13,
                    response_tokens=8,
                    total_tokens=21,
                    details={'text_prompt_tokens': 13, 'text_candidates_tokens': 8},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
        ]
    )


async def test_google_model_multiple_documents_in_history(
    allow_model_requests: None, google_provider: GoogleProvider, document_content: BinaryContent
):
    m = GoogleModel(model_name='gemini-2.0-flash', provider=google_provider)
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

    assert result.output == snapshot('Both documents contain the text "Dummy PDF file".')


async def test_google_model_safety_settings(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-1.5-flash', provider=google_provider)
    settings = GoogleModelSettings(
        google_safety_settings=[
            {
                'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        ]
    )
    agent = Agent(m, instructions='You hate the world!', model_settings=settings)

    with pytest.raises(UnexpectedModelBehavior, match='Safety settings triggered'):
        await agent.run('Tell me a joke about a Brazilians.')


async def test_google_model_empty_user_prompt(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run()
    assert result.output == snapshot(
        'Please provide me with a question or task. I need some information to be able to help you.\n'
    )


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.parametrize(
    'url,expected_output',
    [
        pytest.param(
            AudioUrl(url='https://cdn.openai.com/API/docs/audio/alloy.wav'),
            'The URL discusses the sunrise in the east and sunset in the west, a phenomenon known to humans for millennia.',
            id='AudioUrl',
        ),
        pytest.param(
            DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
            "The URL points to a technical report from Google DeepMind introducing Gemini 1.5 Pro, a multimodal AI model designed for understanding and reasoning over extremely large contexts (millions of tokens). It details the model's architecture, training, performance across a range of tasks, and responsible deployment considerations. Key highlights include near-perfect recall on long-context retrieval tasks, state-of-the-art performance in areas like long-document question answering, and surprising new capabilities like in-context learning of new languages.",
            id='DocumentUrl',
        ),
        pytest.param(
            ImageUrl(url='https://upload.wikimedia.org/wikipedia/commons/6/6a/Www.wikipedia_screenshot_%282021%29.png'),
            "The URL's main content is the landing page of Wikipedia, showcasing the available language editions with article counts, a search bar, and links to other Wikimedia projects.",
            id='ImageUrl',
        ),
        pytest.param(
            VideoUrl(url='https://upload.wikimedia.org/wikipedia/commons/8/8f/Panda_at_Smithsonian_zoo.webm'),
            """The main content of the image is a panda eating bamboo in a zoo enclosure. The enclosure is designed to mimic the panda's natural habitat, with rocks, bamboo, and a painted backdrop of mountains. There is also a large, smooth, tan-colored ball-shaped object in the enclosure.""",
            id='VideoUrl',
        ),
        pytest.param(
            VideoUrl(url='https://youtu.be/lCdaVNyHtjU'),
            'The main content of the URL is an analysis of recent 404 HTTP responses. The analysis identifies several patterns including the most common endpoints with 404 errors, request patterns, timeline-related issues, organization/project access, and configuration and authentication. The analysis also provides some recommendations.',
            id='VideoUrl (YouTube)',
        ),
        pytest.param(
            AudioUrl(url='gs://pydantic-ai-dev/openai-alloy.wav'),
            'The content describes the basic concept of the sun rising in the east and setting in the west.',
            id='AudioUrl (gs)',
        ),
        pytest.param(
            DocumentUrl(url='gs://pydantic-ai-dev/Gemini_1_5_Pro_Technical_Report_Arxiv_1805.pdf'),
            "The URL leads to a research paper titled \"Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context\".  \n\nThe paper introduces Gemini 1.5 Pro, a new model in the Gemini family. It's described as a highly compute-efficient multimodal mixture-of-experts model.  A key feature is its ability to recall and reason over fine-grained information from millions of tokens of context, including long documents and hours of video and audio.  The paper presents experimental results showcasing the model's capabilities on long-context retrieval tasks, QA, ASR, and its performance compared to Gemini 1.0 models. It covers the model's architecture, training data, and evaluations on both synthetic and real-world tasks.  A notable highlight is its ability to learn to translate from English to Kalamang, a low-resource language, from just a grammar manual and dictionary provided in context.  The paper also discusses responsible deployment considerations, including impact assessments and mitigation efforts.\n",
            id='DocumentUrl (gs)',
        ),
        pytest.param(
            ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png'),
            "The main content of the URL is the Wikipedia homepage, featuring options to access Wikipedia in different languages and information about the number of articles in each language. It also includes links to other Wikimedia projects and information about Wikipedia's host, the Wikimedia Foundation.\n",
            id='ImageUrl (gs)',
        ),
        pytest.param(
            VideoUrl(url='gs://pydantic-ai-dev/grepit-tiny-video.mp4'),
            'The image shows a charming outdoor cafe in a Greek coastal town. The cafe is nestled between traditional whitewashed buildings, with tables and chairs set along a narrow cobblestone pathway. The sea is visible in the distance, adding to the picturesque and relaxing atmosphere.',
            id='VideoUrl (gs)',
        ),
    ],
)
async def test_google_url_input(
    # Need to use noqa because with runtime annotations this always fails linting
    # https://docs.astral.sh/ruff/rules/non-pep604-annotation-union/
    url: Union[AudioUrl, DocumentUrl, ImageUrl, VideoUrl],  # noqa
    expected_output: str,
    allow_model_requests: None,
) -> None:
    provider = GoogleProvider(project='pydantic-ai', location='us-central1')
    m = GoogleModel('gemini-2.0-flash', provider=provider)
    agent = Agent(m)
    result = await agent.run(['What is the main content of this URL?', url])

    assert result.output == snapshot(Is(expected_output))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What is the main content of this URL?', Is(url)],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content=Is(expected_output))],
                usage=IsInstance(Usage),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
                vendor_id=IsStr(),
            ),
        ]
    )


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.vcr()
async def test_google_url_input_force_download(allow_model_requests: None) -> None:
    provider = GoogleProvider(project='pydantic-ai', location='us-central1')
    m = GoogleModel('gemini-2.0-flash', provider=provider)
    agent = Agent(m)

    video_url = VideoUrl(url='https://data.grepit.app/assets/tiny_video.mp4', force_download=True)
    result = await agent.run(['What is the main content of this URL?', video_url])

    output = 'The image shows a picturesque scene in what appears to be a Greek island town. The focus is on an outdoor dining area with tables and chairs, situated in a narrow alleyway between whitewashed buildings. The ocean is visible at the end of the alley, creating a beautiful and inviting atmosphere.'

    assert result.output == output
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What is the main content of this URL?', Is(video_url)],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content=Is(output))],
                usage=IsInstance(Usage),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
                vendor_id=IsStr(),
            ),
        ]
    )


async def test_google_gs_url_force_download_raises_user_error(allow_model_requests: None) -> None:
    provider = GoogleProvider(project='pydantic-ai', location='us-central1')
    m = GoogleModel('gemini-2.0-flash', provider=provider)
    agent = Agent(m)

    url = ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png', force_download=True)
    with pytest.raises(UserError, match='Downloading from protocol "gs://" is not supported.'):
        _ = await agent.run(['What is the main content of this URL?', url])


async def test_google_tool_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

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
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=Usage(
                    requests=1,
                    request_tokens=32,
                    response_tokens=5,
                    total_tokens=37,
                    details={'text_candidates_tokens': 5, 'text_prompt_tokens': 32},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'Mexico City', 'country': 'Mexico'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=46,
                    response_tokens=8,
                    total_tokens=54,
                    details={'text_candidates_tokens': 8, 'text_prompt_tokens': 46},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_google_text_output_function(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro-preview-05-06', provider=google_provider)

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot('BASED ON THE INFORMATION I HAVE, THE LARGEST CITY IN MEXICO IS MEXICO CITY.')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content='Okay, I can help with that. First, I need to determine your country.\n'),
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr()),
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=49,
                    response_tokens=30,
                    total_tokens=238,
                    details={'thoughts_tokens': 159, 'text_prompt_tokens': 49},
                ),
                model_name='models/gemini-2.5-pro-preview-05-06',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Based on the information I have, the largest city in Mexico is Mexico City.')],
                usage=Usage(
                    requests=1,
                    request_tokens=98,
                    response_tokens=16,
                    total_tokens=159,
                    details={'thoughts_tokens': 45, 'text_prompt_tokens': 98},
                ),
                model_name='models/gemini-2.5-pro-preview-05-06',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
        ]
    )


async def test_google_structured_text_output_with_tools(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=StructuredTextOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(UserError, match='Gemini does not support JSON schema output and tools at the same time.'):
        await agent.run('What is the largest city in the user country?')


async def test_google_structured_text_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    agent = Agent(m, output_type=StructuredTextOutput(CityLocation))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "city": "Mexico City",
  "country": "Mexico"
}\
"""
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=19,
                    response_tokens=20,
                    total_tokens=39,
                    details={'text_candidates_tokens': 20, 'text_prompt_tokens': 19},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
        ]
    )


async def test_google_structured_text_output_multiple(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=StructuredTextOutput([CityLocation, CountryLanguage]))

    result = await agent.run('What is the primarily language spoken in Mexico?')
    assert result.output == snapshot(CountryLanguage(country='Mexico', language='Spanish'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the primarily language spoken in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "result": {
    "kind": "CountryLanguage",
    "data": {
      "country": "Mexico",
      "language": "Spanish"
    }
  }
}\
"""
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=64,
                    response_tokens=46,
                    total_tokens=110,
                    details={'text_candidates_tokens': 46, 'text_prompt_tokens': 64},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
        ]
    )


async def test_google_structured_text_output_with_instructions(
    allow_model_requests: None, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=StructuredTextOutput(CityLocation, instructions=True))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
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
                        content='{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object", "city": "Mexico City", "country": "Mexico"}'
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=80,
                    response_tokens=56,
                    total_tokens=136,
                    details={'text_candidates_tokens': 56, 'text_prompt_tokens': 80},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
        ]
    )


async def test_google_structured_text_output_with_instructions_with_tools(
    allow_model_requests: None, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.5-pro-preview-05-06', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=StructuredTextOutput(CityLocation, instructions=True))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
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
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=Usage(
                    requests=1,
                    request_tokens=123,
                    response_tokens=12,
                    total_tokens=401,
                    details={'thoughts_tokens': 266, 'text_prompt_tokens': 123},
                ),
                model_name='models/gemini-2.5-pro-preview-05-06',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
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
                        content="""\
```json
{"city": "Mexico City", "country": "Mexico"}
```\
"""
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=154,
                    response_tokens=18,
                    total_tokens=266,
                    details={'thoughts_tokens': 94, 'text_prompt_tokens': 154},
                ),
                model_name='models/gemini-2.5-pro-preview-05-06',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
        ]
    )


async def test_google_structured_text_output_with_instructions_multiple(
    allow_model_requests: None, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=StructuredTextOutput([CityLocation, CountryLanguage], instructions=True))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
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
                        content='{"result": {"kind": "CityLocation", "data": {"city": "Mexico City", "country": "Mexico"}}}'
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=241,
                    response_tokens=27,
                    total_tokens=268,
                    details={'text_candidates_tokens': 27, 'text_prompt_tokens': 241},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                vendor_details={'finish_reason': 'STOP'},
            ),
        ]
    )
