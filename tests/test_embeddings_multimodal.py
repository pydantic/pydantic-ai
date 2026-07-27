"""Multimodal embedding inputs: modality gating, content fusion, and provider mapping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Embedder
from pydantic_ai.embeddings import (
    EmbeddingContent,
    EmbeddingContentPart,
    EmbeddingInput,
    EmbeddingModality,
    TestEmbeddingModel,
    WrapperEmbeddingModel,
    embedding_modality,
    embedding_parts,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    ImageUrl,
    TextContent,
    VideoUrl,
)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import try_import

pytestmark = [
    pytest.mark.anyio,
]

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

with try_import() as google_imports_successful:
    from google.genai.types import Content, Part

    from pydantic_ai.embeddings.google import GoogleEmbeddingModel, GoogleEmbeddingSettings
    from pydantic_ai.providers.google import GoogleProvider

KIWI_IMAGE_URL = 'https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'


@pytest.mark.parametrize(
    'part,expected',
    [
        pytest.param('hello', 'text', id='str'),
        pytest.param(TextContent(content='hello'), 'text', id='text-content'),
        pytest.param(ImageUrl(url='https://example.com/img.png'), 'image', id='image-url'),
        pytest.param(AudioUrl(url='https://example.com/audio.mp3'), 'audio', id='audio-url'),
        pytest.param(VideoUrl(url='https://example.com/video.mp4'), 'video', id='video-url'),
        pytest.param(DocumentUrl(url='https://example.com/doc.pdf'), 'document', id='document-url'),
        pytest.param(BinaryImage(data=b'\x00', media_type='image/png'), 'image', id='binary-image'),
        pytest.param(BinaryContent(data=b'\x00', media_type='audio/mpeg'), 'audio', id='binary-audio'),
        pytest.param(BinaryContent(data=b'\x00', media_type='video/mp4'), 'video', id='binary-video'),
        pytest.param(BinaryContent(data=b'\x00', media_type='application/pdf'), 'document', id='binary-document'),
    ],
)
def test_embedding_modality(part: EmbeddingContentPart, expected: EmbeddingModality):
    assert embedding_modality(part) == expected


def test_embedding_parts():
    image = BinaryImage(data=b'\x00', media_type='image/png')
    assert embedding_parts('hello') == ['hello']
    assert embedding_parts(EmbeddingContent(['hello', image])) == ['hello', image]


async def test_test_model_embeds_every_modality(tiny_image: BinaryImage):
    """`TestEmbeddingModel` accepts every modality so multimodal apps can be tested against it."""
    model = TestEmbeddingModel(dimensions=4)
    result = await Embedder(model).embed_documents(['a kiwi fruit', tiny_image])

    assert len(result.embeddings) == 2
    assert result.inputs == ['a kiwi fruit', tiny_image]
    # Only the text is counted; the fake model has no way to tokenize an image.
    assert result.usage == snapshot(RequestUsage(input_tokens=3))


async def test_combined_content_yields_one_embedding(tiny_image: BinaryImage):
    model = TestEmbeddingModel(dimensions=4)
    content = EmbeddingContent(['a kiwi fruit', tiny_image])
    result = await Embedder(model).embed_documents(content)

    assert len(result.embeddings) == 1
    assert result.inputs == [content]


def test_wrapper_delegates_supported_modalities():
    assert WrapperEmbeddingModel(TestEmbeddingModel()).supported_modalities == TestEmbeddingModel().supported_modalities


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
async def test_unsupported_modality_raises(gemini_api_key: str, tiny_image: BinaryImage):
    """A model that can't embed the input fails locally, rather than as a provider error."""
    model = GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider(api_key=gemini_api_key))

    with pytest.raises(UserError, match=r'does not support image inputs\. Supported modalities: text\.'):
        await Embedder(model).embed_documents(['a kiwi fruit', tiny_image])


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
async def test_combined_content_on_text_only_model_raises(gemini_api_key: str):
    model = GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider(api_key=gemini_api_key))

    with pytest.raises(UserError, match=r'only supports plain text inputs, got `EmbeddingContent`\.'):
        await Embedder(model).embed_documents(EmbeddingContent(['a kiwi', 'fruit']))


def test_prepare_text_embed_unwraps_text_content():
    """Text-only implementations get plain strings, while the result keeps the original inputs."""
    tagged = TextContent(content='world', metadata={'chunk': 7})
    items, texts, _ = TestEmbeddingModel().prepare_text_embed(['hello', tagged])

    assert texts == ['hello', 'world']
    # The originals go to `EmbeddingResult.inputs`, so a `TextContent`'s metadata survives.
    assert items == ['hello', tagged]


@dataclass
class _GoogleMultimodalCase:
    """A multimodal request to `gemini-embedding-2`, asserted at the wire and at the result."""

    id: str
    inputs: EmbeddingInput | Sequence[EmbeddingInput]
    expected_parts: list[list[str]]
    """Per input sent to the API, a description of each part: the verbatim text, or `<media_type>`."""

    expected_embeddings: int
    settings: GoogleEmbeddingSettings | None = None
    expected_warning: str | None = None


_KIWI_IMAGE = BinaryImage(data=(Path(__file__).parent / 'assets' / 'kiwi.jpg').read_bytes(), media_type='image/jpeg')

_GOOGLE_MULTIMODAL_CASES = (
    [
        _GoogleMultimodalCase(
            id='image-only',
            inputs=_KIWI_IMAGE,
            expected_parts=[['<image/jpeg>']],
            expected_embeddings=1,
        ),
        _GoogleMultimodalCase(
            id='text-and-image-combined',
            inputs=EmbeddingContent(['a kiwi fruit', TextContent(content='green and fuzzy'), _KIWI_IMAGE]),
            expected_parts=[['a kiwi fruit', 'green and fuzzy', '<image/jpeg>']],
            expected_embeddings=1,
        ),
        _GoogleMultimodalCase(
            id='batch-text-and-image-url',
            inputs=['a kiwi fruit', ImageUrl(url=KIWI_IMAGE_URL)],
            # The task prefix conditions text; the downloaded image is sent as-is.
            expected_parts=[['title: none | text: a kiwi fruit'], ['<image/jpeg>']],
            expected_embeddings=2,
            settings=GoogleEmbeddingSettings(google_task='search result'),
            expected_warning='`google_task` only conditions inputs that are a single text part',
        ),
    ]
    if google_imports_successful()
    else []
)


def _describe_part(part: Part) -> str:
    """The text a part carries, or its media type, so a request's shape reads at a glance."""
    if part.text is not None:
        return part.text
    assert part.inline_data is not None, 'the embeddings API only takes text and inline data'
    return f'<{part.inline_data.mime_type}>'


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
@pytest.mark.vcr
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in _GOOGLE_MULTIMODAL_CASES])
async def test_google_multimodal(
    case: _GoogleMultimodalCase,
    gemini_api_key: str,
    disable_ssrf_protection_for_vcr: None,
    monkeypatch: pytest.MonkeyPatch,
):
    """`gemini-embedding-2` embeds images, and combines parts into a single vector."""
    provider = GoogleProvider(api_key=gemini_api_key)
    model = GoogleEmbeddingModel('gemini-embedding-2', provider=provider)

    captured: dict[str, list[Content]] = {}
    real_embed_content = provider.client.aio.models.embed_content

    async def spy(**kwargs: Any) -> Any:
        captured['contents'] = kwargs['contents']
        return await real_embed_content(**kwargs)

    monkeypatch.setattr(provider.client.aio.models, 'embed_content', spy)

    if case.expected_warning is not None:
        with pytest.warns(UserWarning, match=case.expected_warning):
            result = await Embedder(model).embed_documents(case.inputs, settings=case.settings)
    else:
        result = await Embedder(model).embed_documents(case.inputs, settings=case.settings)

    sent = [[_describe_part(part) for part in (content.parts or [])] for content in captured['contents']]
    assert sent == case.expected_parts
    assert len(result.embeddings) == case.expected_embeddings
    assert result.inputs == ([case.inputs] if not isinstance(case.inputs, list) else case.inputs)


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_describes_non_text_inputs(capfire: CaptureLogfire, tiny_image: BinaryImage):
    """Files are described by media type instead of being dumped into the span as bytes."""
    model = TestEmbeddingModel(dimensions=2)
    embedder = Embedder(model, instrument=InstrumentationSettings(include_content=True, include_binary_content=False))

    await embedder.embed_documents(
        [
            'a kiwi fruit',
            tiny_image,
            ImageUrl(url='https://example.com/img.png'),
            # No extension to infer a media type from; the span omits it rather than failing the request.
            ImageUrl(url='https://example.com/redirects/to/an/image'),
            EmbeddingContent([TextContent(content='a kiwi fruit'), tiny_image]),
        ]
    )

    [span] = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert span['attributes']['inputs'] == snapshot(
        [
            'a kiwi fruit',
            {'type': 'blob', 'modality': 'image', 'mime_type': 'image/jpeg'},
            {'type': 'uri', 'modality': 'image', 'uri': 'https://example.com/img.png', 'mime_type': 'image/png'},
            {'type': 'uri', 'modality': 'image', 'uri': 'https://example.com/redirects/to/an/image'},
            ['a kiwi fruit', {'type': 'blob', 'modality': 'image', 'mime_type': 'image/jpeg'}],
        ]
    )


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_includes_binary_content(capfire: CaptureLogfire, tiny_image: BinaryImage):
    model = TestEmbeddingModel(dimensions=2)
    embedder = Embedder(model, instrument=InstrumentationSettings(include_content=True))

    await embedder.embed_documents(tiny_image)

    [span] = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert span['attributes']['inputs'] == snapshot(
        [{'type': 'blob', 'modality': 'image', 'mime_type': 'image/jpeg', 'content': 'AAEC'}]
    )
