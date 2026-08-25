from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from urllib.parse import urlparse

import pytest
from typing_extensions import assert_never

from pydantic_ai import (
    Agent,
    BinaryContent,
    Citation,
    ContentCitationAnchor,
    DocumentCitationSource,
    MarkerCitationAnchor,
    ModelMessage,
    ModelResponse,
    TextPart,
    WebCitationSource,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import try_import
from .citation_utils import citations_from_messages

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleCloudLocation, GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as openrouter_available:
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as xai_available:
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.native_tools import XSearchTool
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


@pytest.fixture()
async def vertex_provider(
    request: pytest.FixtureRequest, vertex_provider_auth: None
) -> GoogleCloudProvider | None:  # pragma: lax no cover
    """Only construct the optional Vertex provider for Vertex matrix cases."""
    if 'google-vertex' not in request.node.name:  # pyright: ignore[reportUnknownMemberType]
        return None
    if not google_available():
        pytest.skip('google dependencies not installed')

    record_mode = cast(Any, request.config).getoption('record_mode')
    if not os.getenv('CI', False) and record_mode not in {'all', 'new_episodes', 'rewrite'}:
        pytest.skip('Requires properly configured local google vertex config to pass')

    project = os.getenv('GOOGLE_PROJECT', 'pydantic-ai')
    location = cast(GoogleCloudLocation, os.getenv('GOOGLE_LOCATION', 'global'))
    return GoogleCloudProvider(project=project, location=location)


@dataclass(frozen=True)
class ExpectedWebCitation:
    source_labels: list[str]
    excerpt_counts: list[int]
    anchor: ContentCitationAnchor | MarkerCitationAnchor | None
    anchor_text: str | None = None


@dataclass(frozen=True)
class WebCitationCase:
    id: str
    provider: Literal['anthropic', 'google-gemini', 'google-vertex', 'openai', 'openrouter', 'xai']
    stream: bool = False
    expected: list[ExpectedWebCitation] = field(default_factory=list[ExpectedWebCitation])


WEB_CASES = [
    WebCitationCase(
        'anthropic',
        'anthropic',
        expected=snapshot(
            [
                ExpectedWebCitation(['pypi.org'], [1], None),
                ExpectedWebCitation(['github.com'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
            ]
        ),
    ),
    WebCitationCase(
        'anthropic-stream',
        'anthropic',
        stream=True,
        expected=snapshot(
            [
                ExpectedWebCitation(['pypi.org'], [1], None),
                ExpectedWebCitation(['github.com'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
            ]
        ),
    ),
    WebCitationCase(
        'google-gemini',
        'google-gemini',
        expected=snapshot(
            [
                ExpectedWebCitation(
                    [
                        'pydantic.dev',
                        'pydantic.dev',
                        'pydantic.dev',
                        'github.com',
                    ],
                    [0, 0, 0, 0],
                    ContentCitationAnchor(start=0, end=79),
                    'The official documentation for Pydantic AI can be found on the Pydantic website',
                ),
                ExpectedWebCitation(
                    ['pydantic.dev', 'github.com'],
                    [0, 0],
                    ContentCitationAnchor(start=81, end=215),
                    'It provides comprehensive information on Pydantic AI, which is described as the Python AI SDK, '
                    'offering a typed, extensible agent loop',
                ),
                ExpectedWebCitation(
                    ['pydantic.dev', 'github.com'],
                    [0, 0],
                    ContentCitationAnchor(start=217, end=331),
                    'The documentation covers various aspects, including agents, realtime voice, image generation, '
                    'embeddings, and more',
                ),
                ExpectedWebCitation(
                    ['together.ai'],
                    [0],
                    ContentCitationAnchor(start=333, end=477),
                    'Pydantic AI aims to simplify building production-grade generative AI applications, bringing a '
                    'type-safe approach to working with language models',
                ),
            ]
        ),
    ),
    WebCitationCase(
        'google-vertex',
        'google-vertex',
        expected=snapshot(
            [
                ExpectedWebCitation(
                    ['pydantic.dev'],
                    [0],
                    ContentCitationAnchor(start=0, end=84),
                    'The official documentation for Pydantic AI can be found on the Pydantic Docs website',
                ),
                ExpectedWebCitation(['github.com'], [0], ContentCitationAnchor(start=142, end=146), 'dev`'),
                ExpectedWebCitation(
                    ['pydantic.dev'],
                    [0],
                    ContentCitationAnchor(start=148, end=316),
                    'Pydantic AI is described as the Python AI SDK, offering a typed, extensible agent loop that '
                    'supports various applications like web frontends, terminals, and voice calls',
                ),
                ExpectedWebCitation(
                    ['pydantic.dev'],
                    [0],
                    ContentCitationAnchor(start=318, end=400),
                    'It includes features for agents, real-time voice, image generation, and embeddings',
                ),
            ]
        ),
    ),
    WebCitationCase(
        'google-vertex-stream',
        'google-vertex',
        stream=True,
        expected=snapshot(
            [
                ExpectedWebCitation(['github.com'], [0], ContentCitationAnchor(start=96, end=100), 'dev`'),
                ExpectedWebCitation(
                    [
                        'pydantic.dev',
                        'together.ai',
                        'github.com',
                    ],
                    [0, 0, 0],
                    ContentCitationAnchor(start=102, end=248),
                    'Pydantic AI is described as the Python AI SDK, offering a typed and extensible agent loop for '
                    'building production-grade generative AI applications',
                ),
                ExpectedWebCitation(
                    ['pydantic.dev', 'github.com'],
                    [0, 0],
                    ContentCitationAnchor(start=251, end=391),
                    'The documentation provides an overview of Pydantic AI, covering aspects like agents, real-time '
                    'voice, image generation, embeddings, and more',
                ),
                ExpectedWebCitation(
                    ['pydantic.dev', 'github.com'],
                    [0, 0],
                    ContentCitationAnchor(start=393, end=586),
                    'It details how Pydantic AI enables the creation of complex, long-running multi-agent '
                    'collaborations and supports various capabilities like web search, memory, sub-agents, and '
                    'context management',
                ),
                ExpectedWebCitation(
                    ['pydantic.dev', 'pydantic.dev'],
                    [0, 0],
                    ContentCitationAnchor(start=588, end=734),
                    'The documentation also highlights features such as structured output data using Pydantic '
                    'models, enabling type-safe data extraction and validation',
                ),
                ExpectedWebCitation(
                    ['pydantic.dev', 'pydantic.dev'],
                    [0, 0],
                    ContentCitationAnchor(start=736, end=817),
                    'Furthermore, it discusses instrumentation with Pydantic Logfire for observability',
                ),
            ]
        ),
    ),
    WebCitationCase(
        'openai',
        'openai',
        expected=snapshot(
            [
                ExpectedWebCitation(
                    ['github.com'],
                    [0],
                    MarkerCitationAnchor(start=70, end=143),
                    '([github.com](https://github.com/pydantic/pydantic-ai?utm_source=openai))',
                )
            ]
        ),
    ),
    WebCitationCase(
        'openai-stream',
        'openai',
        stream=True,
        expected=snapshot(
            [
                ExpectedWebCitation(
                    ['github.com'],
                    [0],
                    MarkerCitationAnchor(start=71, end=144),
                    '([github.com](https://github.com/pydantic/pydantic-ai?utm_source=openai))',
                )
            ]
        ),
    ),
    WebCitationCase(
        'openrouter',
        'openrouter',
        expected=snapshot(
            [
                ExpectedWebCitation(['github.com'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['github.com'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['github.com'], [1], None),
            ]
        ),
    ),
    WebCitationCase(
        'openrouter-stream',
        'openrouter',
        stream=True,
        expected=snapshot(
            [
                ExpectedWebCitation(['github.com'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['github.com'], [1], None),
                ExpectedWebCitation(['pydantic.dev'], [1], None),
                ExpectedWebCitation(['github.com'], [1], None),
            ]
        ),
    ),
    WebCitationCase(
        'xai',
        'xai',
        expected=snapshot(
            [
                ExpectedWebCitation(
                    ['x.com'],
                    [0],
                    MarkerCitationAnchor(start=227, end=283),
                    '[[1]](https://x.com/pydantic/status/1863538947059544218)',
                )
            ]
        ),
    ),
]


WEB_PROVIDER_AVAILABLE = {
    'anthropic': anthropic_available,
    'google-gemini': google_available,
    'google-vertex': google_available,
    'openai': openai_available,
    'openrouter': openrouter_available,
    'xai': xai_available,
}


def _web_citation_agent(
    case: WebCitationCase,
    *,
    anthropic_api_key: str,
    gemini_api_key: str,
    openai_api_key: str,
    openrouter_api_key: str,
    vertex_provider: GoogleCloudProvider | None,
    xai_provider: XaiProvider | None,
) -> tuple[Agent[None, str], str]:
    prompt = "Use web search to find Pydantic AI's documentation and cite it."
    settings = None
    if case.provider == 'anthropic':
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
        tool = WebSearchTool(max_uses=1)
    elif case.provider == 'google-gemini':
        model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=gemini_api_key))
        tool = WebSearchTool()
    elif case.provider == 'google-vertex':
        assert vertex_provider is not None
        model = GoogleModel('gemini-2.5-flash', provider=vertex_provider)
        tool = WebSearchTool()
    elif case.provider == 'openai':
        model = OpenAIResponsesModel('gpt-5.4-mini', provider=OpenAIProvider(api_key=openai_api_key))
        tool = WebSearchTool(max_uses=1)
        prompt = "Use web search to find Pydantic AI's GitHub repository and cite it."
    elif case.provider == 'openrouter':
        model = OpenRouterModel('deepseek/deepseek-chat', provider=OpenRouterProvider(api_key=openrouter_api_key))
        tool = WebSearchTool(max_uses=1)
        prompt = "Use web search to find Pydantic AI's GitHub repository and answer with its URL only."
    elif case.provider == 'xai':
        assert xai_provider is not None
        model = XaiModel('grok-4-fast-non-reasoning', provider=xai_provider)
        tool = XSearchTool(allowed_x_handles=['pydantic'], include_output=True)
        settings = ModelSettings(include_citations=True)
        prompt = 'Use X search to find a post by @pydantic about Pydantic AI. Summarize it and cite the post URL.'
    else:  # pragma: no cover
        assert_never(case.provider)

    return Agent(model, capabilities=[NativeTool(tool)], model_settings=settings), prompt


def _cited_text_parts(messages: list[ModelMessage]) -> list[TextPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, TextPart) and part.citations
    ]


def _web_citation_summary(
    cited_parts: list[TextPart],
) -> list[ExpectedWebCitation]:
    def source_label(source: WebCitationSource) -> str:
        domain = urlparse(source.url).netloc
        # Google grounding URLs are opaque redirects; their titles contain the useful source domain.
        return source.title if domain == 'vertexaisearch.cloud.google.com' and source.title else domain

    return [
        ExpectedWebCitation(
            source_labels=[
                source_label(source) for source in citation.sources if isinstance(source, WebCitationSource)
            ],
            excerpt_counts=[
                len(source.excerpts) for source in citation.sources if isinstance(source, WebCitationSource)
            ],
            anchor=citation.anchor,
            anchor_text=(
                part.content[citation.anchor.start : citation.anchor.end] if citation.anchor is not None else None
            ),
        )
        for part in cited_parts
        for citation in part.citations or []
    ]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in WEB_CASES])
async def test_web_citations(
    case: WebCitationCase,
    allow_model_requests: None,
    anthropic_api_key: str,
    gemini_api_key: str,
    openai_api_key: str,
    openrouter_api_key: str,
    vertex_provider: GoogleCloudProvider | None,
    xai_provider: XaiProvider | None,
) -> None:
    if not WEB_PROVIDER_AVAILABLE[case.provider]():
        pytest.skip(f'{case.provider} dependencies not installed')

    agent, prompt = _web_citation_agent(
        case,
        anthropic_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        vertex_provider=vertex_provider,
        xai_provider=xai_provider,
    )

    if case.stream:
        async with agent.run_stream(prompt) as result:
            await result.get_output()
    else:
        result = await agent.run(prompt)

    cited_parts = _cited_text_parts(result.all_messages())
    citations = [citation for part in cited_parts for citation in part.citations or []]
    assert all(isinstance(source, WebCitationSource) for citation in citations for source in citation.sources)
    assert _web_citation_summary(cited_parts) == case.expected


@dataclass(frozen=True)
class DocumentCitationCase:
    id: str
    provider: Literal['anthropic', 'bedrock'] = 'anthropic'
    stream: bool = False
    pdf: bool = False
    expected: list[Citation] = field(default_factory=list[Citation])


DOCUMENT_CASES = [
    DocumentCitationCase(
        id='anthropic-document',
        expected=snapshot(
            [
                Citation(
                    sources=[
                        DocumentCitationSource(
                            excerpts=['The return window is thirty days from purchase.'],
                            provider_details={
                                'document_index': 0,
                                'end_char_index': 47,
                                'start_char_index': 0,
                                'type': 'char_location',
                            },
                        )
                    ]
                )
            ]
        ),
    ),
    DocumentCitationCase(
        id='anthropic-document-stream',
        stream=True,
        expected=snapshot(
            [
                Citation(
                    sources=[
                        DocumentCitationSource(
                            excerpts=['The return window is thirty days from purchase.'],
                            provider_details={
                                'document_index': 0,
                                'end_char_index': 47,
                                'start_char_index': 0,
                                'type': 'char_location',
                            },
                        )
                    ]
                )
            ]
        ),
    ),
    DocumentCitationCase(
        id='anthropic-pdf',
        pdf=True,
        expected=snapshot(
            [
                Citation(
                    sources=[
                        DocumentCitationSource(
                            excerpts=['Dummy PDF file'],
                            provider_details={
                                'document_index': 0,
                                'end_page_number': 2,
                                'start_page_number': 1,
                                'type': 'page_location',
                            },
                        )
                    ]
                )
            ]
        ),
    ),
    DocumentCitationCase(
        id='bedrock-document',
        provider='bedrock',
        expected=snapshot(
            [
                Citation(
                    sources=[
                        DocumentCitationSource(
                            title='Document 1',
                            excerpts=['The return window is thirty days from purchase.'],
                            provider_details={
                                'location': {'documentChar': {'documentIndex': 0, 'start': 0, 'end': 47}}
                            },
                        )
                    ],
                    anchor=ContentCitationAnchor(start=0, end=47),
                )
            ]
        ),
    ),
    DocumentCitationCase(
        id='bedrock-document-stream',
        provider='bedrock',
        stream=True,
        expected=snapshot(
            [
                Citation(
                    sources=[
                        DocumentCitationSource(
                            title='Document 1',
                            excerpts=['The return window is thirty days from purchase.'],
                            provider_details={
                                'location': {'documentChar': {'documentIndex': 0, 'start': 0, 'end': 47}}
                            },
                        )
                    ],
                    anchor=ContentCitationAnchor(start=0, end=47),
                )
            ]
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in DOCUMENT_CASES])
async def test_document_citations(
    case: DocumentCitationCase,
    request: pytest.FixtureRequest,
    allow_model_requests: None,
    anthropic_api_key: str,
    document_content: BinaryContent,
) -> None:
    available = anthropic_available if case.provider == 'anthropic' else bedrock_available
    if not available():
        pytest.skip(f'{case.provider} dependencies not installed')

    if case.provider == 'anthropic':
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    else:
        model = BedrockConverseModel(
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=request.getfixturevalue('bedrock_provider')
        )
    agent = Agent(model, model_settings=ModelSettings(include_citations=True))
    prompt: str | list[str | BinaryContent]
    if case.pdf:
        prompt = ['What text appears in this PDF? Answer in one sentence and cite the document.', document_content]
    else:
        prompt = [
            'According to the document, what is the return window? Answer in one sentence and cite the document.',
            BinaryContent(data=b'The return window is thirty days from purchase.', media_type='text/plain'),
        ]

    if case.stream:
        async with agent.run_stream(prompt) as result:
            await result.get_output()
    else:
        result = await agent.run(prompt)

    assert citations_from_messages(result.all_messages()) == case.expected
