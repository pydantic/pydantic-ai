from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, assert_never
from urllib.parse import urlparse

import pytest
from pydantic import JsonValue

from pydantic_ai import (
    Agent,
    BinaryContent,
    Citation,
    CompactionPart,
    ContentCitationAnchor,
    DocumentCitationSource,
    MarkerCitationAnchor,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    WebCitationSource,
)
from pydantic_ai.capabilities import Citations, NativeTool
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import
from .citation_utils import citations_from_messages
from .conftest import content_blocks

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicCompaction, AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as openrouter_available:
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as xai_available:
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.native_tools import XSearchTool
    from pydantic_ai.providers.xai import XaiProvider

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
    provider: Literal[
        'anthropic',
        'anthropic-web-fetch',
        'google-gemini',
        'google-vertex',
        'openai',
        'openai-chat',
        'openrouter',
        'openrouter-perplexity',
        'xai',
    ]
    stream: bool = False
    expected: list[ExpectedWebCitation] = field(default_factory=list[ExpectedWebCitation])
    non_ascii: bool = False
    """Start the answer with non-ASCII text, so offsets in bytes or UTF-16 units would select the wrong text."""


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
        'openai-non-ascii',
        'openai',
        non_ascii=True,
        expected=snapshot(
            [
                ExpectedWebCitation(
                    source_labels=['github.com'],
                    excerpt_counts=[0],
                    anchor=MarkerCitationAnchor(start=69, end=142),
                    anchor_text='([github.com](https://github.com/pydantic/pydantic-ai?utm_source=openai))',
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
                    source_labels=['x.com'],
                    excerpt_counts=[0],
                    anchor=MarkerCitationAnchor(start=460, end=516),
                    anchor_text='[[1]](https://x.com/pydantic/status/2105281513579249831)',
                )
            ]
        ),
    ),
    WebCitationCase(
        'xai-non-ascii',
        'xai',
        non_ascii=True,
        expected=snapshot(
            [
                ExpectedWebCitation(
                    source_labels=['x.com'],
                    excerpt_counts=[0],
                    anchor=MarkerCitationAnchor(start=267, end=323),
                    anchor_text='[[1]](https://x.com/pydantic/status/2105281513579249831)',
                )
            ]
        ),
    ),
    WebCitationCase(
        'xai-stream',
        'xai',
        stream=True,
        expected=snapshot(
            [
                ExpectedWebCitation(
                    source_labels=['x.com'],
                    excerpt_counts=[0],
                    anchor=MarkerCitationAnchor(start=398, end=454),
                    anchor_text='[[1]](https://x.com/pydantic/status/2105281513579249831)',
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
                    source_labels=['github.com'],
                    excerpt_counts=[0],
                    anchor=MarkerCitationAnchor(start=119, end=205),
                    anchor_text='([github.com](https://github.com/pydantic/pydantic-ai?ref=peerlist&utm_source=openai))',
                )
            ]
        ),
    ),
    WebCitationCase(
        'openai-chat-non-ascii',
        'openai-chat',
        non_ascii=True,
        expected=snapshot(
            [
                ExpectedWebCitation(
                    source_labels=['github.com'],
                    excerpt_counts=[0],
                    anchor=MarkerCitationAnchor(start=114, end=187),
                    anchor_text='([github.com](https://github.com/pydantic/pydantic-ai?utm_source=openai))',
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
                    source_labels=['github.com'],
                    excerpt_counts=[0],
                    anchor=MarkerCitationAnchor(start=249, end=335),
                    anchor_text='([github.com](https://github.com/pydantic/pydantic-ai?ref=peerlist&utm_source=openai))',
                )
            ]
        ),
    ),
    WebCitationCase(
        'openrouter-perplexity',
        'openrouter-perplexity',
        expected=snapshot(
            [
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pypi.org'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['realpython.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pydantic.dev'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['www.aibase.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['machinelearningmastery.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pydantic.dev'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['www.star-history.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pydantic.dev'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pydantic.dev'], excerpt_counts=[0], anchor=None),
            ]
        ),
    ),
    WebCitationCase(
        'openrouter-perplexity-stream',
        'openrouter-perplexity',
        stream=True,
        expected=snapshot(
            [
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['api.github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pydantic.dev'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pydantic.dev'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pypi.org'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pydantic.dev'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pypi.org'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['pydantic.dev'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['github.com'], excerpt_counts=[0], anchor=None),
                ExpectedWebCitation(source_labels=['gist.github.com'], excerpt_counts=[0], anchor=None),
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
    'openrouter': openrouter_available,
    'openrouter-perplexity': openrouter_available,
    'xai': xai_available,
}


def _web_citation_agent(
    case: WebCitationCase,
    *,
    anthropic_api_key: str,
    gemini_api_key: str,
    openai_api_key: str,
    openrouter_api_key: str,
    xai_provider: XaiProvider | None,
    request: pytest.FixtureRequest,
) -> tuple[Agent[None, str], str]:
    prompt = "Use web search to find Pydantic AI's documentation and cite it."
    settings = None
    if case.provider == 'anthropic':
        model = AnthropicModel(
            'claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key), settings=ANTHROPIC_SETTINGS
        )
        tool = WebSearchTool(max_uses=1)
    elif case.provider == 'anthropic-web-fetch':
        # Web Fetch only returns citations when asked.
        model = AnthropicModel(
            'claude-sonnet-4-5',
            provider=AnthropicProvider(api_key=anthropic_api_key),
            settings=ModelSettings(max_tokens=4096, citations=True),
        )
        tool = WebFetchTool(max_uses=1)
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
    elif case.provider == 'openrouter':
        model = OpenRouterModel('deepseek/deepseek-chat', provider=OpenRouterProvider(api_key=openrouter_api_key))
        tool = WebSearchTool(max_uses=1)
        prompt = "Use web search to find Pydantic AI's GitHub repository and answer with its URL only."
    elif case.provider == 'openrouter-perplexity':
        # Perplexity's Sonar models always search, so no tool is needed.
        model = OpenRouterModel('perplexity/sonar', provider=OpenRouterProvider(api_key=openrouter_api_key))
        tool = None
        prompt = "Find Pydantic AI's GitHub repository and cite it in one sentence."
    elif case.provider == 'xai':
        assert xai_provider is not None
        model = XaiModel('grok-4-fast-non-reasoning', provider=xai_provider)
        tool = XSearchTool(allowed_x_handles=['pydantic'], include_output=True)
        settings = ModelSettings(citations=True)
        # A single search keeps the streamed parts in the same order as the complete response's.
        prompt = 'Run one X search for a post by @pydantic about Pydantic AI. Summarize it and cite the post URL.'
    else:  # pragma: no cover
        assert_never(case.provider)

    if case.non_ascii:
        # The cassette hooks rewrite smart quotes and dashes, which would shift offsets on replay.
        prompt += (
            " Start your answer with exactly '🐍 Pydantic AI (café): ', cite inline, and use only ASCII punctuation."
        )

    return Agent(model, capabilities=[NativeTool(tool)] if tool else [], model_settings=settings), prompt


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
    xai_provider: XaiProvider | None,
    request: pytest.FixtureRequest,
) -> None:
    if not WEB_PROVIDER_AVAILABLE[case.provider]():
        pytest.skip(f'{case.provider} dependencies not installed')

    agent, prompt = _web_citation_agent(
        case,
        anthropic_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        xai_provider=xai_provider,
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
    for part in cited_parts:
        for citation in part.citations or []:
            if isinstance(citation.anchor, MarkerCitationAnchor):
                marker = part.content[citation.anchor.start : citation.anchor.end]
                assert any(
                    isinstance(source, WebCitationSource) and urlparse(source.url).netloc in marker
                    for source in citation.sources
                ), marker
    if case.non_ascii:
        assert any(
            not part.content[: citation.anchor.start].isascii()
            for part in cited_parts
            for citation in part.citations or []
            if citation.anchor
        )


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
                        )
                    ],
                    anchor=ContentCitationAnchor(start=0, end=47),
                )
            ]
        ),
    ),
    DocumentCitationCase(
        id='bedrock-pdf',
        provider='bedrock',
        pdf=True,
        expected=snapshot(
            [
                Citation(
                    sources=[
                        DocumentCitationSource(
                            title='Document 1',
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
    request: pytest.FixtureRequest,
    allow_model_requests: None,
    anthropic_api_key: str,
    document_content: BinaryContent,
) -> None:
    available = anthropic_available if case.provider == 'anthropic' else bedrock_available
    if not available():
        pytest.skip(f'{case.provider} dependencies not installed')

    if case.provider == 'anthropic':
        model = AnthropicModel(
            'claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key), settings=ANTHROPIC_SETTINGS
        )
    else:
        model = BedrockConverseModel(
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=request.getfixturevalue('bedrock_provider')
        )
    agent = Agent(model, capabilities=[Citations()])
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


NativeCitationReplayProvider = Literal[
    'anthropic-messages', 'anthropic-document', 'anthropic-tool-document', 'bedrock-converse', 'openai-responses'
]


@pytest.mark.vcr(additional_matchers=['body'])
@pytest.mark.parametrize(
    'provider',
    [
        pytest.param('anthropic-messages', id='anthropic-messages'),
        pytest.param('anthropic-document', id='anthropic-document'),
        pytest.param('anthropic-tool-document', id='anthropic-tool-document'),
        pytest.param('bedrock-converse', id='bedrock-converse'),
        pytest.param('openai-responses', id='openai-responses'),
    ],
)
async def test_native_citation_replay_after_persisted_history(
    provider: NativeCitationReplayProvider,
    request: pytest.FixtureRequest,
    allow_model_requests: None,
    anthropic_api_key: str,
    openai_api_key: str,
) -> None:
    """Each provider accepts its own persisted native citation history on the next request."""
    available = {
        'anthropic-messages': anthropic_available,
        'anthropic-document': anthropic_available,
        'anthropic-tool-document': anthropic_available,
        'bedrock-converse': bedrock_available,
        'openai-responses': openai_available,
    }[provider]
    if not available():
        pytest.skip(f'{provider} dependencies not installed')
    agent: Agent[None, str]
    first_prompt: str | list[str | BinaryContent]
    if provider in ('anthropic-messages', 'anthropic-document', 'anthropic-tool-document'):
        model = AnthropicModel(
            'claude-sonnet-4-5',
            provider=AnthropicProvider(api_key=anthropic_api_key),
            settings=ANTHROPIC_SETTINGS,
        )
        if provider == 'anthropic-messages':
            agent = Agent(model, capabilities=[NativeTool(WebSearchTool(max_uses=1))])
            first_prompt = "Use web search to find Pydantic AI's documentation and cite it."
        elif provider == 'anthropic-tool-document':
            agent = Agent(model, tools=[read_shipping_policy], model_settings=ModelSettings(citations=True))
            first_prompt = TOOL_DOCUMENT_PROMPT
        else:
            agent = Agent(model, model_settings=ModelSettings(citations=True))
            first_prompt = [
                'According to the document, what is the return window? Answer in one sentence and cite the document.',
                BinaryContent(data=b'The return window is thirty days from purchase.', media_type='text/plain'),
            ]
    elif provider == 'bedrock-converse':
        agent = Agent(
            BedrockConverseModel(
                'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=request.getfixturevalue('bedrock_provider')
            ),
            model_settings=ModelSettings(citations=True),
        )
        first_prompt = [
            'According to the document, what is the return window? Answer in one sentence and cite the document.',
            BinaryContent(data=b'The return window is thirty days from purchase.', media_type='text/plain'),
        ]
    else:
        agent = Agent(
            OpenAIResponsesModel('gpt-5.4-mini', provider=OpenAIProvider(api_key=openai_api_key)),
            capabilities=[NativeTool(WebSearchTool())],
            model_settings=OpenAIResponsesModelSettings(openai_send_reasoning_ids=True),
        )
        first_prompt = "Use web search to find Pydantic AI's GitHub repository and cite it."

    first_result = await agent.run(first_prompt)
    assert citations_from_messages(first_result.all_messages())

    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first_result.all_messages()))
    second_result = await agent.run('Continue.', message_history=history)
    assert second_result.output


RETURNS_POLICY = BinaryContent(data=b'The return window is thirty days from purchase.', media_type='text/plain')
SHIPPING_POLICY = BinaryContent(data=b'Orders ship within two business days.', media_type='text/plain')
WARRANTY_POLICY = BinaryContent(data=b'Every product has a two-year warranty.', media_type='text/plain')


def _replayed_document_indices(body: dict[str, JsonValue]) -> list[list[JsonValue]]:
    """The `document_index` of each replayed document citation, per text block that has citations."""
    return [
        [citation.get('document_index') for citation in citations if isinstance(citation, dict)]
        for block in content_blocks(body, 'text')
        if isinstance(citations := block.get('citations'), list)
    ]


@pytest.mark.moves_cache_prefix(reason='compaction replaces the history before it')
async def test_anthropic_document_citation_replay_after_compacted_history(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    """Document citations replay correctly once compaction has left the earlier history out.

    The first answer cites the returns policy. The second prompt is large enough to be compacted, and Anthropic
    answers it from the compaction summary, without citations. Later requests start at the compaction block, so the
    returns answer and its document are no longer sent. The warranty policy sent after compaction is cited, and its
    citation is replayed with the index it has in the request that is actually sent: 0.
    """
    if not anthropic_available():
        pytest.skip('anthropic dependencies not installed')
    model = AnthropicModel(
        'claude-sonnet-4-6',
        provider=AnthropicProvider(api_key=anthropic_api_key, http_client=request_capture.client),
        settings=ANTHROPIC_SETTINGS,
    )
    agent = Agent(
        model,
        model_settings=ModelSettings(citations=True),
        capabilities=[AnthropicCompaction(token_threshold=50_000)],
    )
    # About 55k tokens, over the threshold, with no line ending in a space, which pre-commit would strip.
    padding = ' '.join(['The quick brown fox jumps over the lazy dog.'] * 5000)

    returns = await agent.run(['What is the return window? Cite the document.', RETURNS_POLICY])
    shipping = await agent.run(
        [f'Background notes: {padding}\n\nHow long does shipping take? Cite the document.', SHIPPING_POLICY],
        message_history=returns.all_messages(),
    )
    assert any(isinstance(part, CompactionPart) for part in shipping.new_messages()[-1].parts)
    warranty = await agent.run(
        ['How long is the warranty? Cite the document.', WARRANTY_POLICY], message_history=shipping.all_messages()
    )
    assert citations_from_messages(warranty.new_messages())
    await agent.run('Thanks. Repeat the warranty in one sentence.', message_history=warranty.all_messages())

    # Only the warranty answer is sent with a citation.
    assert _replayed_document_indices(request_capture.bodies('/v1/messages')[-1]) == snapshot([[0]])


@pytest.mark.moves_cache_prefix(reason='the last request leaves the first exchange out of the history')
async def test_anthropic_document_citation_replay_after_history_trimmed(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    """A document citation follows its document to its new position when earlier history is left out.

    The shipping policy is the second document when it's cited (index 1). Leaving the first exchange out makes it the
    only document, so the citation is replayed with index 0. Index 1 would make Anthropic reject the request.
    """
    if not anthropic_available():
        pytest.skip('anthropic dependencies not installed')
    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key=anthropic_api_key, http_client=request_capture.client),
        settings=ANTHROPIC_SETTINGS,
    )
    agent = Agent(model, model_settings=ModelSettings(citations=True))

    returns = await agent.run(['What is the return window? Cite the document.', RETURNS_POLICY])
    shipping = await agent.run(
        ['How long does shipping take? Cite the document.', SHIPPING_POLICY], message_history=returns.all_messages()
    )
    [shipping_answer] = [
        part for part in shipping.new_messages()[-1].parts if isinstance(part, TextPart) and part.citations
    ]
    assert [citation['document_index'] for citation in (shipping_answer.provider_details or {})['citations']] == [1]

    # What a history processor that keeps only the latest exchange would send.
    result = await agent.run(
        'Thanks. Repeat the shipping time in one sentence.', message_history=shipping.all_messages()[2:]
    )
    assert result.output

    assert _replayed_document_indices(request_capture.bodies('/v1/messages')[-1]) == snapshot([[0]])


# Body matching would sort this conversation's turns out of order; the requests are pinned by `request_capture`.
@pytest.mark.vcr()
async def test_anthropic_document_citation_replay_counts_fetched_pages(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    """A page fetched by Web Fetch is numbered like any other document, so later citations are replayed past it.

    The returns policy, the fetched license page and the shipping policy are documents 0, 1 and 2 when Anthropic cites
    them. Replaying all three citations with those indices shows our numbering of the request matches Anthropic's.
    """
    if not anthropic_available():
        pytest.skip('anthropic dependencies not installed')
    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key=anthropic_api_key, http_client=request_capture.client),
        settings=ModelSettings(max_tokens=4096, citations=True),
    )
    agent = Agent(model, capabilities=[NativeTool(WebFetchTool(max_uses=1))])

    # The license is a plain-text page with no trailing whitespace or smart characters, which recording would rewrite.
    license_and_returns = await agent.run(
        [
            'What is the return window? Then fetch https://raw.githubusercontent.com/pydantic/pydantic-ai/main/LICENSE '
            'and say which license Pydantic AI uses. Answer in two sentences, citing the document and the page.',
            RETURNS_POLICY,
        ]
    )
    shipping = await agent.run(
        ['How long does shipping take? Cite the document.', SHIPPING_POLICY],
        message_history=license_and_returns.all_messages(),
    )
    await agent.run('Thanks. Reply only OK.', message_history=shipping.all_messages())

    cited = [
        [citation['document_index'] for citation in (part.provider_details or {})['citations']]
        for part in _cited_text_parts(shipping.all_messages())
    ]
    assert cited == snapshot([[0], [1], [2]])
    assert _replayed_document_indices(request_capture.bodies('/v1/messages')[-1]) == cited


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
    agent = Agent(model, tools=[read_shipping_policy], model_settings=ModelSettings(citations=True))

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


# Cassetter writes body-matched interactions sorted by body, which puts this conversation's two turns out of order;
# `test_bedrock_tool_return_document_citations_request` in `test_bedrock.py` pins the request instead.
@pytest.mark.vcr()
async def test_bedrock_tool_return_document_citations(
    allow_model_requests: None, request: pytest.FixtureRequest
) -> None:
    """Bedrock cites documents in both the user prompt and a tool result."""
    if not bedrock_available():
        pytest.skip('bedrock dependencies not installed')

    model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=request.getfixturevalue('bedrock_provider')
    )
    agent = Agent(model, tools=[read_shipping_policy], model_settings=ModelSettings(citations=True))

    result = await agent.run(TOOL_DOCUMENT_PROMPT)

    assert citations_from_messages(result.all_messages()) == snapshot(
        [
            Citation(
                sources=[
                    DocumentCitationSource(
                        title='Document 1', excerpts=['The return window is thirty days from purchase.']
                    )
                ],
                anchor=ContentCitationAnchor(start=0, end=47),
            ),
            Citation(
                sources=[
                    DocumentCitationSource(title='Document 2', excerpts=['Orders ship within two business days.'])
                ],
                anchor=ContentCitationAnchor(start=0, end=37),
            ),
        ]
    )


def read_warranty_policy() -> BinaryContent:
    """Read the warranty policy document."""
    return WARRANTY_POLICY


@contextmanager
def _converse_request_bodies(model: BedrockConverseModel) -> Generator[list[dict[str, Any]]]:
    """The Converse request bodies as sent, read off boto's `before-send` event, which fires on cassette replay too."""
    bodies: list[dict[str, Any]] = []

    def capture(request: Any, **_: Any) -> None:
        bodies.append(json.loads(request.body))

    event = 'before-send.bedrock-runtime.Converse'
    model.client.meta.events.register_last(event, capture)
    try:
        yield bodies
    finally:
        model.client.meta.events.unregister(event, capture)


def _bedrock_document_indices(messages: list[ModelMessage]) -> list[list[Any]]:
    """The `documentIndex` of each document citation Bedrock returned, per cited text part."""
    return [
        [citation['location']['documentChar']['documentIndex'] for citation in part.provider_details['citations']]
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, TextPart) and part.provider_details and 'citations' in part.provider_details
    ]


def _replayed_bedrock_document_indices(body: dict[str, Any]) -> list[list[Any]]:
    """The `documentIndex` of each replayed document citation, per `citationsContent` block."""
    return [
        [citation['location']['documentChar']['documentIndex'] for citation in block['citationsContent']['citations']]
        for message in body['messages']
        for block in message['content']
        if 'citationsContent' in block
    ]


# Body matching would sort this conversation's turns out of order, as for the test above; the requests are pinned
# from the `before-send` event instead.
@pytest.mark.vcr()
@pytest.mark.moves_cache_prefix(reason='the last request leaves the first exchange out of the history')
async def test_bedrock_document_citation_replay_after_history_trimmed(
    allow_model_requests: None, request: pytest.FixtureRequest
) -> None:
    """Document citations follow their documents, including one a tool returned, when earlier history is left out.

    A support conversation cites the shipping policy from the prompt, the warranty policy a tool returned, and the
    returns policy from the prompt. Leaving the first exchange out takes the shipping policy with it, so the warranty
    and returns policies move from 1 and 2 to 0 and 1. Bedrock's citations in the answer to that request show its own
    numbering, which the replayed citations have to match.
    """
    if not bedrock_available():
        pytest.skip('bedrock dependencies not installed')
    model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=request.getfixturevalue('bedrock_provider')
    )
    agent = Agent(model, tools=[read_warranty_policy], model_settings=ModelSettings(citations=True))

    shipping = await agent.run(['How long does shipping take? Cite the document.', SHIPPING_POLICY])
    warranty = await agent.run(
        'How long is the warranty? Read the warranty policy and cite it.', message_history=shipping.all_messages()
    )
    returns = await agent.run(
        ['What is the return window? Cite the document.', RETURNS_POLICY], message_history=warranty.all_messages()
    )
    # Bedrock numbers the documents shipping 0, warranty 1 (in the tool result) and returns 2.
    assert _bedrock_document_indices(returns.all_messages()) == snapshot([[0], [1], [2]])

    # What a history processor that leaves out the first exchange would send.
    with _converse_request_bodies(model) as bodies:
        summary = await agent.run(
            'Without reading any policy again, summarise the warranty and the return window in two sentences, '
            'citing the documents.',
            message_history=returns.all_messages()[2:],
        )

    assert _replayed_bedrock_document_indices(bodies[-1]) == snapshot([[0], [1]])
    assert _bedrock_document_indices(summary.new_messages()) == snapshot([[0], [1]])
