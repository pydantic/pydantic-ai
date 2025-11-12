# pyright: reportDeprecated=false
import os
from dataclasses import dataclass

import pytest
from inline_snapshot import Is, snapshot
from pytest_mock import MockerFixture

from pydantic_ai import (
    Agent,
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsInstance, IsStr, try_import

with try_import() as imports_successful:
    from google.auth.transport.requests import Request

    from pydantic_ai.providers.google_vertex import GoogleVertexProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-auth not installed'),
    pytest.mark.anyio,
    pytest.mark.filterwarnings('ignore:Use `GoogleModel` instead.:DeprecationWarning'),
    pytest.mark.filterwarnings('ignore:`GoogleVertexProvider` is deprecated.:DeprecationWarning'),
]


@pytest.fixture(autouse=True)
def vertex_provider_auth(mocker: MockerFixture) -> None:  # pragma: lax no cover
    # Locally, we authenticate via `gcloud` CLI, so we don't need to patch anything.
    if not os.getenv('CI'):
        return

    @dataclass
    class NoOpCredentials:
        token = 'my-token'

        def refresh(self, request: Request): ...

    return_value = (NoOpCredentials(), 'pydantic-ai')
    mocker.patch('pydantic_ai.providers.google_vertex.google.auth.default', return_value=return_value)


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.vcr()
async def test_labels(allow_model_requests: None) -> None:  # pragma: lax no cover
    provider = GoogleVertexProvider(project_id='pydantic-ai', region='us-central1')
    m = GeminiModel('gemini-2.0-flash', provider=provider)
    agent = Agent(m)

    result = await agent.run(
        'What is the capital of France?',
        model_settings=GeminiModelSettings(gemini_labels={'environment': 'test', 'team': 'analytics'}),
    )
    assert result.output == snapshot('The capital of France is **Paris**.\n')


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.parametrize(
    'url,expected_output',
    [
        pytest.param(
            AudioUrl(url='https://cdn.openai.com/API/docs/audio/alloy.wav'),
            'The content of the URL discusses the observation that the sun rises in the east and sets in the west, a phenomenon that has been noted by humans for millennia.',
            id='AudioUrl',
        ),
        pytest.param(
            DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
            "The URL provided links to a technical report about Google DeepMind's Gemini 1.5 Pro model. Here's a summary of its main content:\n\n*   **Introduction of Gemini 1.5 Pro:** The report introduces Gemini 1.5 Pro, a new model in the Gemini family. It's described as a compute-efficient multimodal mixture-of-experts model.\n*   **Key Capabilities:** Gemini 1.5 Pro can handle millions of tokens of context, including long documents, and hours of video and audio. It excels in long-context retrieval tasks, and performs on par with Gemini 1.0 Ultra in various benchmarks.\n*   **Long-Context Ability:** The model demonstrates near-perfect retrieval up to 10M tokens, a big jump from existing models. It also learns to translate between English and Kalamang.\n*   **Model Architecture:** Gemini 1.5 Pro is based on a sparse mixture-of-expert (MoE) Transformer architecture.\n*   **Training and Evaluation:** The report outlines the training infrastructure and dataset, including multimodal and multilingual data. It details long-context evaluations across text, vision, and audio modalities. Evaluation tasks include needle-in-a-haystack, long-document QA, ASR, and translation.\n*   **Responsible Deployment:** Discusses Google's approach to responsible deployment of Gemini 1.5 Pro, including impact assessment and mitigation efforts.",
            id='DocumentUrl',
        ),
        pytest.param(
            ImageUrl(url='https://upload.wikimedia.org/wikipedia/commons/6/6a/Www.wikipedia_screenshot_%282021%29.png'),
            'The main content of the URL is the Wikipedia homepage, which features options to select different languages and search for articles. It also highlights various Wikimedia projects like Commons, Wikibooks, and Wiktionary.',
            id='ImageUrl',
        ),
        pytest.param(
            VideoUrl(url='https://upload.wikimedia.org/wikipedia/commons/8/8f/Panda_at_Smithsonian_zoo.webm'),
            'The main content of the URL is a video of a panda bear sitting amongst bamboo shoots in an enclosure. The enclosure has a rocky background, an artificial mountain scene painted on the back wall, a window, and a large beige ball. The panda is eating bamboo.',
            id='VideoUrl',
        ),
        pytest.param(
            VideoUrl(url='https://youtu.be/lCdaVNyHtjU'),
            'The main content of the URL is a code editor (likely VS Code) displaying a TypeScript file named "browserRouter.tsx". The code seems to define routing logic for a web application, likely using React, potentially including features like lazy loading of components.  A chat window is open where a user has asked an AI assistant to analyze recent 404 HTTP responses using logfile data. The AI provides a detailed analysis of the 404 errors, breaking them down by common endpoints, request patterns, timeline issues, project access problems, and configuration/authentication concerns. The analysis also contains some recommendations for the user.',
            id='VideoUrl (YouTube)',
        ),
        pytest.param(
            AudioUrl(url='gs://pydantic-ai-dev/openai-alloy.wav'),
            "The URL's main content is about the observation that the sun rises in the east and sets in the west.",
            id='AudioUrl (gs)',
        ),
        pytest.param(
            DocumentUrl(url='gs://pydantic-ai-dev/Gemini_1_5_Pro_Technical_Report_Arxiv_1805.pdf'),
            "The main content of this URL is a technical report about Gemini 1.5 Pro, a new model from Google DeepMind's Gemini family, that is designed to unlock multimodal understanding across millions of tokens of context. It describes the model architecture, performance benchmarks, and new capabilities, such as near-perfect recall in long-context retrieval tasks and the ability to translate between English and Kalamang, a language with very few speakers, using only a provided grammar manual. The report also covers the responsible deployment of the model, including impact assessments and safety evaluations.",
            id='DocumentUrl (gs)',
        ),
        pytest.param(
            ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png'),
            'The main content of the URL appears to be the multilingual portal page for Wikipedia, "The Free Encyclopedia." It showcases the numerous language editions available, highlighting the number of articles in each. It also provides links to download the app and to other Wikimedia projects.\n',
            id='ImageUrl (gs)',
        ),
        pytest.param(
            VideoUrl(url='gs://pydantic-ai-dev/grepit-tiny-video.mp4'),
            'The image shows a narrow alleyway between white buildings, leading to the Aegean Sea. There are tables and chairs set up as if for a cafe or restaurant, giving the impression of a picturesque, seaside dining spot. The water is a vibrant blue with choppy waves. The overall scene is reminiscent of the Greek islands, particularly Mykonos or Santorini.',
            id='VideoUrl (gs)',
        ),
    ],
)
@pytest.mark.vcr()
async def test_url_input(
    url: AudioUrl | DocumentUrl | ImageUrl | VideoUrl, expected_output: str, allow_model_requests: None
) -> None:  # pragma: lax no cover
    provider = GoogleVertexProvider(project_id='pydantic-ai', region='us-central1')
    m = GeminiModel('gemini-2.0-flash', provider=provider)
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
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=Is(expected_output))],
                usage=IsInstance(RequestUsage),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.vcr()
async def test_url_input_force_download(allow_model_requests: None) -> None:  # pragma: lax no cover
    provider = GoogleVertexProvider(project_id='pydantic-ai', region='us-central1')
    m = GeminiModel('gemini-2.0-flash', provider=provider)
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
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=Is(output))],
                usage=IsInstance(RequestUsage),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_gs_url_force_download_raises_user_error(allow_model_requests: None) -> None:
    provider = GoogleVertexProvider(project_id='pydantic-ai', region='us-central1')
    m = GeminiModel('gemini-2.0-flash', provider=provider)
    agent = Agent(m)

    url = ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png', force_download=True)
    with pytest.raises(UserError, match='Downloading from protocol "gs://" is not supported.'):
        _ = await agent.run(['What is the main content of this URL?', url])
