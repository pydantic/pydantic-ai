from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx2
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
    ModelMessagesTypeAdapter,
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
from .conftest import RequestCapture

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
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
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
                            provider_details={'location': {'documentPage': {'documentIndex': 0, 'start': 1, 'end': 2}}},
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


NativeCitationReplayProvider = Literal[
    'anthropic-messages', 'anthropic-document', 'bedrock-converse', 'openai-responses'
]


@pytest.mark.vcr(match_on=['method', 'scheme', 'host', 'port', 'path', 'query', 'body'])
@pytest.mark.parametrize(
    'provider',
    [
        pytest.param('anthropic-messages', id='anthropic-messages'),
        pytest.param('anthropic-document', id='anthropic-document'),
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
    agent: Agent[None, str]
    first_prompt: str | list[str | BinaryContent]
    if provider in ('anthropic-messages', 'anthropic-document'):
        if not anthropic_available():
            pytest.skip('anthropic dependencies not installed')
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
        if provider == 'anthropic-messages':
            agent = Agent(model, capabilities=[NativeTool(WebSearchTool(max_uses=1))])
            first_prompt = "Use web search to find Pydantic AI's documentation and cite it."
        else:
            agent = Agent(model, model_settings=ModelSettings(include_citations=True))
            first_prompt = [
                'According to the document, what is the return window? Answer in one sentence and cite the document.',
                BinaryContent(data=b'The return window is thirty days from purchase.', media_type='text/plain'),
            ]
    elif provider == 'bedrock-converse':
        if not bedrock_available():
            pytest.skip('bedrock dependencies not installed')
        agent = Agent(
            BedrockConverseModel(
                'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=request.getfixturevalue('bedrock_provider')
            ),
            model_settings=ModelSettings(include_citations=True),
        )
        first_prompt = [
            'According to the document, what is the return window? Answer in one sentence and cite the document.',
            BinaryContent(data=b'The return window is thirty days from purchase.', media_type='text/plain'),
        ]
    else:
        if not openai_available():  # pragma: no cover
            pytest.skip('openai dependencies not installed')
        agent = Agent(
            OpenAIResponsesModel('gpt-5.4-mini', provider=OpenAIProvider(api_key=openai_api_key)),
            capabilities=[NativeTool(WebSearchTool(max_uses=1))],
            model_settings=OpenAIResponsesModelSettings(openai_send_reasoning_ids=True),
        )
        first_prompt = "Use web search to find Pydantic AI's GitHub repository and cite it."

    first_result = await agent.run(first_prompt)
    assert citations_from_messages(first_result.all_messages())

    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first_result.all_messages()))
    second_result = await agent.run('Continue.', message_history=history)
    assert second_result.output


@pytest.mark.vcr(match_on=['method', 'scheme', 'host', 'port', 'path', 'query', 'body'])
async def test_cross_provider_citation_replay_google_to_bedrock(
    allow_model_requests: None,
    gemini_api_key: str,
    request: pytest.FixtureRequest,
    request_capture: RequestCapture,
) -> None:
    """A persisted Gemini grounding citation falls back to unchanged text in Bedrock history."""
    if not google_available() or not bedrock_available():  # pragma: no cover
        pytest.skip('google and bedrock dependencies are required')

    request_capture.client.timeout = httpx2.Timeout(30)
    google_model = GoogleModel(
        'gemini-2.5-flash',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.client),
    )
    first_result = await Agent(google_model, capabilities=[NativeTool(WebSearchTool(max_uses=1))]).run(
        "Use web search to find Pydantic AI's documentation and cite it."
    )
    citations = citations_from_messages(first_result.all_messages())
    assert citations
    assert all(isinstance(citation.anchor, ContentCitationAnchor) for citation in citations)

    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first_result.all_messages()))
    bedrock_model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=request.getfixturevalue('bedrock_provider')
    )
    second_result = await Agent(bedrock_model).run(
        'Without searching again, briefly describe the sources attached to the previous answer.',
        message_history=history,
    )
    assert second_result.output


@pytest.mark.vcr(match_on=['method', 'scheme', 'host', 'port', 'path', 'query', 'body'])
async def test_cross_provider_document_citation_replay_bedrock_to_anthropic(
    allow_model_requests: None,
    anthropic_api_key: str,
    request: pytest.FixtureRequest,
    request_capture: RequestCapture,
) -> None:
    """A Bedrock text-document citation remains bound to the same document when replayed to Anthropic."""
    if not bedrock_available() or not anthropic_available():  # pragma: no cover
        pytest.skip('bedrock and anthropic dependencies are required')

    document = b'The return window is thirty days from purchase.'
    bedrock_model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=request.getfixturevalue('bedrock_provider')
    )
    first_result = await Agent(bedrock_model, model_settings=ModelSettings(include_citations=True)).run(
        [
            'According to the document, what is the return window? Answer in one sentence and cite the document.',
            BinaryContent(data=document, media_type='text/plain'),
        ]
    )
    [citation] = citations_from_messages(first_result.all_messages())
    [source] = citation.sources
    assert isinstance(source, DocumentCitationSource)
    assert source.excerpts == [document.decode()]

    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first_result.all_messages()))
    anthropic_model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key=anthropic_api_key, http_client=request_capture.client),
    )
    second_result = await Agent(anthropic_model).run('Continue.', message_history=history)
    assert second_result.output

    second_request = request_capture.bodies()[-1]
    [prior_assistant] = [message for message in second_request['messages'] if message['role'] == 'assistant']
    [prior_text] = prior_assistant['content']
    [replayed_citation] = prior_text['citations']
    assert replayed_citation == {
        'type': 'char_location',
        'cited_text': document.decode(),
        'document_index': 0,
        'document_title': 'Document 1',
        'start_char_index': 0,
        'end_char_index': len(document),
    }


@pytest.mark.vcr(match_on=['method', 'scheme', 'host', 'port', 'path', 'query', 'body'])
async def test_cross_provider_document_citation_replay_anthropic_to_bedrock(
    allow_model_requests: None,
    anthropic_api_key: str,
    request: pytest.FixtureRequest,
    request_capture: RequestCapture,
) -> None:
    """An Anthropic text-document citation remains bound to the same document when replayed to Bedrock."""
    if not anthropic_available() or not bedrock_available():  # pragma: no cover
        pytest.skip('anthropic and bedrock dependencies are required')

    document = b'The return window is thirty days from purchase.'
    anthropic_model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key=anthropic_api_key, http_client=request_capture.client),
    )
    first_result = await Agent(anthropic_model, model_settings=ModelSettings(include_citations=True)).run(
        [
            'According to the document, what is the return window? Answer in one sentence and cite the document.',
            BinaryContent(data=document, media_type='text/plain'),
        ]
    )
    [citation] = citations_from_messages(first_result.all_messages())
    [source] = citation.sources
    assert isinstance(source, DocumentCitationSource)
    assert source.excerpts == [document.decode()]

    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first_result.all_messages()))
    bedrock_model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=request.getfixturevalue('bedrock_provider')
    )
    second_result = await Agent(bedrock_model, model_settings=ModelSettings(include_citations=True)).run(
        'Continue.', message_history=history
    )
    assert second_result.output


@pytest.mark.vcr(match_on=['method', 'scheme', 'host', 'port', 'path', 'query', 'body'])
async def test_openai_responses_citation_replay_with_synthetic_item_id(
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
) -> None:
    """A foreign rendered-marker citation reaches Responses through a generated output-item ID."""
    if not openai_available():  # pragma: no cover
        pytest.skip('openai dependency is required')

    history = [
        ModelResponse(
            parts=[
                TextPart(
                    'Answer ([example.com](https://example.com))',
                    provider_name='openrouter',
                    citations=[
                        Citation(
                            sources=[WebCitationSource(url='https://example.com', title='Example')],
                            anchor=MarkerCitationAnchor(start=7, end=43),
                        )
                    ],
                )
            ],
            provider_name='openrouter',
        )
    ]
    model = OpenAIResponsesModel(
        'gpt-5.4-mini', provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )
    result = await Agent(model).run('Reply exactly DONE.', message_history=history)

    assert result.output == snapshot('DONE')
    assert request_capture.bodies()[0]['input'][0] == {
        'id': 'msg_pydantic_ai_0_0',
        'role': 'assistant',
        'status': 'completed',
        'type': 'message',
        'content': [
            {
                'type': 'output_text',
                'text': 'Answer ([example.com](https://example.com))',
                'annotations': [
                    {
                        'type': 'url_citation',
                        'url': 'https://example.com',
                        'title': 'Example',
                        'start_index': 7,
                        'end_index': 43,
                    }
                ],
            }
        ],
    }


_OPENROUTER_CITATION_CONTEXT_PROBE_PROMPT = """\
Use web search to verify what Pydantic AI is, then reply with exactly:
Pydantic AI is a Python agent framework.
Do not include URLs, source titles, citation markers, or source excerpts in the answer text.\
"""
_CITATION_CONTEXT_FOLLOW_UP = """\
Do not search again. Based only on structured citation metadata attached to the previous assistant message, return the
first source URL exactly. If the message you received contains no structured citation metadata, reply exactly
UNAVAILABLE. Do not infer or guess the URL from your own knowledge.\
"""
_GOOGLE_CITATION_CONTEXT_CASSETTE = (
    Path(__file__).parent
    / 'cassettes'
    / 'test_citations'
    / 'test_citation_context_without_native_replay[google-gemini].yaml'
)
_OPENROUTER_CITATION_CONTEXT_CASSETTE = (
    Path(__file__).parent
    / 'cassettes'
    / 'test_citations'
    / 'test_citation_context_without_native_replay[openrouter].yaml'
)
CitationContextProvider = Literal['google-gemini', 'openrouter']


@pytest.mark.parametrize(
    'provider',
    [
        pytest.param(
            'google-gemini',
            id='google-gemini',
            marks=pytest.mark.skipif(
                not os.getenv('GEMINI_API_KEY') and not _GOOGLE_CITATION_CONTEXT_CASSETTE.is_file(),
                reason='requires GEMINI_API_KEY to record the citation-context probe',
            ),
        ),
        pytest.param(
            'openrouter',
            id='openrouter',
            marks=pytest.mark.skipif(
                not os.getenv('OPENROUTER_API_KEY') and not _OPENROUTER_CITATION_CONTEXT_CASSETTE.is_file(),
                reason='requires OPENROUTER_API_KEY to record the citation-context probe',
            ),
        ),
    ],
)
async def test_citation_context_without_native_replay(
    provider: CitationContextProvider,
    allow_model_requests: None,
    gemini_api_key: str,
    openrouter_api_key: str,
    request_capture: RequestCapture,
) -> None:
    """Response citations remain local while the follow-up request receives plain assistant text."""
    if provider == 'google-gemini':
        if not google_available():  # pragma: no cover
            pytest.skip('google dependencies not installed')
        # Google derives its SDK deadline from the injected client's read timeout and rejects
        # httpx's five-second default because the Gemini API minimum is ten seconds.
        request_capture.client.timeout = httpx2.Timeout(30)
        model = GoogleModel(
            'gemini-2.5-flash',
            provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.client),
        )
        # Gemini only creates citations when groundingMetadata includes groundingChunks and groundingSupports.
        # Constraining it to a citation-free sentence produced only searchEntryPoint, which is not a citation.
        first_prompt = "Use web search to find Pydantic AI's documentation and cite it."
        first_agent = Agent(model, capabilities=[NativeTool(WebSearchTool(max_uses=1))])
    else:
        if not openrouter_available():  # pragma: no cover
            pytest.skip('openrouter dependencies not installed')
        model = OpenRouterModel(
            'deepseek/deepseek-chat',
            provider=OpenRouterProvider(api_key=openrouter_api_key, http_client=request_capture.client),
        )
        first_prompt = _OPENROUTER_CITATION_CONTEXT_PROBE_PROMPT
        first_agent = Agent(model, capabilities=[NativeTool(WebSearchTool(max_uses=1))])

    first_result = await first_agent.run(first_prompt)
    assert citations_from_messages(first_result.all_messages())

    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first_result.all_messages()))
    second_result = await Agent(model).run(_CITATION_CONTEXT_FOLLOW_UP, message_history=history)

    if provider == 'google-gemini':
        assert second_result.output == snapshot('UNAVAILABLE')
        second_request = request_capture.bodies()[-1]
        [prior_assistant] = [content for content in second_request['contents'] if content['role'] == 'model']
        assert prior_assistant == {'role': 'model', 'parts': [{'text': first_result.output}]}
    else:
        assert second_result.output == snapshot('UNAVAILABLE')
        second_request = request_capture.bodies()[-1]
        [prior_assistant] = [message for message in second_request['messages'] if message['role'] == 'assistant']
        assert prior_assistant == {'role': 'assistant', 'content': first_result.output}
