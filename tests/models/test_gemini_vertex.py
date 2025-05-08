import os

import pytest
from inline_snapshot import Is, snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models.gemini import GeminiModel

from ..conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-auth not installed'),
    pytest.mark.anyio,
]


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.parametrize(
    'url,expected_output',
    [
        pytest.param(
            AudioUrl(url='https://cdn.openai.com/API/docs/audio/alloy.wav'),
            'The URL discusses the phenomenon of the sun rising in the east and setting in the west, a basic observation known to humans for millennia.',
            id='AudioUrl',
        ),
        # pytest.param(
        #     DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
        #     (
        #         'The URL points to a report titled "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context". The report introduces the latest model in the Gemini family, Gemini 1.5 Pro, which is a highly'
        #         ' compute-efficient multimodal mixture-of-experts model capable of recalling and reasoning over fine-grained information'
        #         ' from millions of tokens of context, including multiple long documents and hours of video and audio. The report'
        #         " compares Gemini 1.5 Pro's performance with Gemini 1.0 Pro and Ultra across several benchmarks. The model's"
        #         ' responsible deployment, training, and evaluation processes are detailed.'
        #     ),
        #     id='DocumentUrl',
        # ),
        # pytest.param(
        #     ImageUrl(url='https://upload.wikimedia.org/wikipedia/commons/6/6a/Www.wikipedia_screenshot_%282021%29.png'),
        #     'The main content of the URL is the Wikipedia multilingual portal, showcasing the availability of Wikipedia in various languages, their respective article counts, and links to download the Wikipedia app.',
        #     id='ImageUrl',
        # ),
        # pytest.param(
        #     VideoUrl(url='https://data.grepit.app/assets/tiny_video.mp4'),
        #     'The image shows a picturesque alley in a Greek island town, likely in the Cyclades. The alley is lined with traditional white-washed buildings, and has a restaurant or cafe setup with tables and chairs. The alley leads to a view of the blue Aegean Sea. The scene evokes a relaxing, vacation-like atmosphere.',
        #     id='VideoUrl',
        # ),
        # pytest.param(
        #     VideoUrl(url='https://youtu.be/lCdaVNyHtjU'),
        #     (
        #         'The main content of the URL is a discussion about analyzing recent 404 HTTP responses using Logfire.'
        #         ' The analysis identifies patterns and trends in the 404 errors, including the most common endpoints returning 404s,'
        #         ' request patterns, timeline-related issues, organization/project access problems, and configuration/authentication'
        #         ' issues. The analysis also provides recommendations for addressing the 404 errors.'
        #     ),
        #     id='VideoUrl (YouTube)',
        # ),
        # pytest.param(
        #     AudioUrl(url='gs://pydantic-ai-dev/openai-alloy.wav'),
        #     'The main content of the URL discusses the observation of the sun rising in the east and setting in the west, a phenomenon known for thousands of years.',
        #     id='AudioUrl (gs)',
        # ),
        # pytest.param(
        #     DocumentUrl(url='gs://pydantic-ai-dev/Gemini_1_5_Pro_Technical_Report_Arxiv_1805.pdf'),
        #     """The main content of the URL is a research report titled "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context". The report introduces Gemini 1.5 Pro, a highly compute-efficient multimodal model capable of recalling and reasoning over fine-grained information from millions of tokens of context. It details the model's architecture, performance on various benchmarks, and evaluations of its long-context capabilities across modalities like text, video, and audio. The report also discusses responsible deployment aspects, such as impact assessment and safety evaluations.\n""",
        #     id='DocumentUrl (gs)',
        # ),
        # pytest.param(
        #     ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png'),
        #     'The main content of the URL is the Wikipedia homepage, listing different languages and the number of articles available in each language. It also provides links to download the Wikipedia app, and other related projects.',
        #     id='ImageUrl (gs)',
        # ),
        # pytest.param(
        #     VideoUrl(url='gs://pydantic-ai-dev/grepit-tiny-video.mp4'),
        #     'The image shows a picturesque outdoor dining area in a Greek island town. The tables and chairs are set up in a narrow, whitewashed alleyway leading towards the blue sea. The white buildings are typical of Cycladic architecture.',
        #     id='VideoUrl (gs)',
        # ),
    ],
)
@pytest.mark.vcr()
async def test_url_input(
    url: AudioUrl | DocumentUrl | ImageUrl | VideoUrl, expected_output: str, allow_model_requests: None
) -> None:
    provider = GoogleVertexProvider(project_id='pydantic-ai', region='us-central1')
    agent = Agent(model=GeminiModel('gemini-2.0-flash', provider=provider))
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
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
            ),
        ]
    )
