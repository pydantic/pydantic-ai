from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, assert_never
from urllib.parse import urlparse

import pytest

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
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import try_import
from .citation_utils import citations_from_messages

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

# Matching on the body makes a request that drops a citation setting fail instead of replaying the recording.
pytestmark = pytest.mark.vcr(additional_matchers=['body'])

# The Anthropic recordings sent this explicitly; without it, requests are streamed behind the scenes.
ANTHROPIC_SETTINGS = ModelSettings(max_tokens=4096)


@dataclass(frozen=True)
class ExpectedWebCitation:
    source_labels: list[str]
    excerpt_counts: list[int]
    anchor: ContentCitationAnchor | MarkerCitationAnchor | None
    anchor_text: str | None = None


@dataclass(frozen=True)
class WebCitationCase:
    id: str
    provider: Literal['anthropic', 'anthropic-web-fetch', 'google-gemini', 'google-vertex', 'openai', 'openai-chat']
    stream: bool = False
    expected: list[ExpectedWebCitation] = field(default_factory=list[ExpectedWebCitation])


WEB_CASES = [
    WebCitationCase(
        'anthropic-web-fetch',
        'anthropic-web-fetch',
        expected=snapshot(
            [
                ExpectedWebCitation(
                    source_labels=['raw.githubusercontent.com'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=33),
                    anchor_text='Pydantic AI uses the MIT License.',
                )
            ]
        ),
    ),
    WebCitationCase(
        'anthropic-web-fetch-stream',
        'anthropic-web-fetch',
        stream=True,
        expected=snapshot(
            [
                ExpectedWebCitation(
                    source_labels=['raw.githubusercontent.com'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=32),
                    anchor_text='Pydantic AI uses the MIT License',
                )
            ]
        ),
    ),
    WebCitationCase(
        'anthropic',
        'anthropic',
        expected=snapshot(
            [
                ExpectedWebCitation(
                    source_labels=['pypi.org', 'github.com'],
                    excerpt_counts=[1, 1],
                    anchor=ContentCitationAnchor(start=0, end=58),
                    anchor_text='The complete documentation is available at ai.pydantic.dev',
                ),
                ExpectedWebCitation(
                    source_labels=['pydantic.dev'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=100),
                    anchor_text='Pydantic AI is the Python AI SDK: a typed, extensible agent loop with every model a string swap away',
                ),
                ExpectedWebCitation(
                    source_labels=['pydantic.dev'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=170),
                    anchor_text='The same agent runs everywhere you need it: behind a web frontend, in the terminal, on a voice call, on a durable background queue, or as a plain object you call run() on',
                ),
                ExpectedWebCitation(
                    source_labels=['pydantic.dev'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=59),
                    anchor_text='structured outputs, typed dependency injection, typed tools',
                ),
                ExpectedWebCitation(
                    source_labels=['pydantic.dev'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=146),
                    anchor_text='support for virtually every model and provider (OpenAI, Anthropic, Google, Bedrock, Azure AI Foundry, Groq, Mistral, xAI, Ollama, and dozens more)',
                ),
            ]
        ),
    ),
    WebCitationCase(
        'anthropic-stream',
        'anthropic',
        stream=True,
        expected=snapshot(
            [
                ExpectedWebCitation(
                    source_labels=['pypi.org', 'github.com'],
                    excerpt_counts=[1, 1],
                    anchor=ContentCitationAnchor(start=0, end=58),
                    anchor_text='The complete documentation is available at ai.pydantic.dev',
                ),
                ExpectedWebCitation(
                    source_labels=['pydantic.dev'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=100),
                    anchor_text='Pydantic AI is the Python AI SDK: a typed, extensible agent loop with every model a string swap away',
                ),
                ExpectedWebCitation(
                    source_labels=['pydantic.dev'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=142),
                    anchor_text='From simple typed data extraction to complex, long-running multi-agent collaboration, Pydantic AI and Pydantic AI Harness have got you covered',
                ),
                ExpectedWebCitation(
                    source_labels=['pydantic.dev'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=171),
                    anchor_text='Support for virtually every model and provider (OpenAI, Anthropic, Google, Bedrock, Azure AI Foundry, Groq, Mistral, xAI, Ollama, and dozens more), swappable with a string',
                ),
                ExpectedWebCitation(
                    source_labels=['pydantic.dev'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=59),
                    anchor_text='Structured outputs, typed dependency injection, typed tools',
                ),
                ExpectedWebCitation(
                    source_labels=['pydantic.dev'],
                    excerpt_counts=[1],
                    anchor=ContentCitationAnchor(start=0, end=31),
                    anchor_text='Image generation and embeddings',
                ),
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
        'openai-chat',
        'openai-chat',
        expected=snapshot(
            [
                ExpectedWebCitation(
                    ['github.com'],
                    [0],
                    MarkerCitationAnchor(start=119, end=205),
                    '([github.com](https://github.com/pydantic/pydantic-ai?ref=peerlist&utm_source=openai))',
                )
            ]
        ),
    ),
    WebCitationCase(
        'openai-chat-stream',
        'openai-chat',
        stream=True,
        expected=snapshot(
            [
                ExpectedWebCitation(
                    ['github.com'],
                    [0],
                    MarkerCitationAnchor(start=249, end=335),
                    '([github.com](https://github.com/pydantic/pydantic-ai?ref=peerlist&utm_source=openai))',
                )
            ]
        ),
    ),
]


WEB_PROVIDER_AVAILABLE = {
    'anthropic': anthropic_available,
    'anthropic-web-fetch': anthropic_available,
    'google-gemini': google_available,
    'google-vertex': google_available,
    'openai': openai_available,
    'openai-chat': openai_available,
}


def _web_citation_agent(
    case: WebCitationCase,
    *,
    anthropic_api_key: str,
    gemini_api_key: str,
    openai_api_key: str,
    request: pytest.FixtureRequest,
) -> tuple[Agent[None, str], str]:
    prompt = "Use web search to find Pydantic AI's documentation and cite it."
    if case.provider == 'anthropic':
        model = AnthropicModel(
            'claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key), settings=ANTHROPIC_SETTINGS
        )
        tool = WebSearchTool(max_uses=1)
    elif case.provider == 'anthropic-web-fetch':
        # Web Fetch only returns citations when asked.
        model = AnthropicModel(
            'claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key), settings=ANTHROPIC_SETTINGS
        )
        tool = WebFetchTool(max_uses=1, enable_citations=True)
        # A plain-text page with no trailing whitespace or smart characters, which recording would rewrite and so move
        # the cited ranges that tie a citation to the fetched page.
        prompt = (
            'Fetch https://raw.githubusercontent.com/pydantic/pydantic-ai/main/LICENSE and say in one sentence which '
            'license Pydantic AI uses, citing it.'
        )
    elif case.provider == 'google-gemini':
        model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=gemini_api_key))
        tool = WebSearchTool()
    elif case.provider == 'google-vertex':
        model = GoogleModel('gemini-2.5-flash', provider=request.getfixturevalue('vertex_provider'))
        tool = WebSearchTool()
        prompt = "Use Google Search to find Pydantic AI's documentation and cite it."
    elif case.provider == 'openai':
        model = OpenAIResponsesModel('gpt-5.4-mini', provider=OpenAIProvider(api_key=openai_api_key))
        tool = WebSearchTool()
        prompt = "Use web search to find Pydantic AI's GitHub repository and cite it."
    elif case.provider == 'openai-chat':
        # Chat Completions search models always search, so no tool is needed.
        model = OpenAIChatModel('gpt-5-search-api', provider=OpenAIProvider(api_key=openai_api_key))
        tool = None
        prompt = "Find Pydantic AI's GitHub repository and cite it in one sentence."
    else:  # pragma: no cover
        assert_never(case.provider)

    return Agent(model, capabilities=[NativeTool(tool)] if tool else []), prompt


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
    request: pytest.FixtureRequest,
) -> None:
    if not WEB_PROVIDER_AVAILABLE[case.provider]():
        pytest.skip(f'{case.provider} dependencies not installed')

    agent, prompt = _web_citation_agent(
        case,
        anthropic_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        request=request,
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
                        )
                    ],
                    anchor=ContentCitationAnchor(start=0, end=47),
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
                        )
                    ],
                    anchor=ContentCitationAnchor(start=0, end=47),
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
                        )
                    ],
                    anchor=ContentCitationAnchor(start=0, end=42),
                )
            ]
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in DOCUMENT_CASES])
async def test_document_citations(
    case: DocumentCitationCase,
    allow_model_requests: None,
    anthropic_api_key: str,
    document_content: BinaryContent,
) -> None:
    if not anthropic_available():
        pytest.skip('anthropic dependencies not installed')

    model = AnthropicModel(
        'claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key), settings=ANTHROPIC_SETTINGS
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


def read_shipping_policy() -> BinaryContent:
    """Read the shipping policy document."""
    return BinaryContent(data=b'Orders ship within two business days.', media_type='text/plain')


TOOL_DOCUMENT_PROMPT: list[str | BinaryContent] = [
    'What are the return window and the shipping time? Read the shipping policy first. '
    'Answer in two sentences and cite the documents.',
    BinaryContent(data=b'The return window is thirty days from purchase.', media_type='text/plain'),
]


async def test_anthropic_tool_return_document_citations(allow_model_requests: None, anthropic_api_key: str) -> None:
    """Anthropic accepts citations enabled on documents in both the user prompt and a tool result."""
    if not anthropic_available():
        pytest.skip('anthropic dependencies not installed')

    model = AnthropicModel(
        'claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key), settings=ANTHROPIC_SETTINGS
    )
    agent = Agent(model, tools=[read_shipping_policy], model_settings=ModelSettings(include_citations=True))

    result = await agent.run(TOOL_DOCUMENT_PROMPT)

    assert citations_from_messages(result.all_messages()) == snapshot(
        [
            Citation(
                sources=[
                    DocumentCitationSource(
                        excerpts=['The return window is thirty days from purchase.'],
                    )
                ],
                anchor=ContentCitationAnchor(start=0, end=47),
            ),
            Citation(
                sources=[
                    DocumentCitationSource(
                        excerpts=['Orders ship within two business days.'],
                    )
                ],
                anchor=ContentCitationAnchor(start=0, end=37),
            ),
        ]
    )
