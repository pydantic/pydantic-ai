"""Multimodal embedding inputs: modality gating, content fusion, and provider mapping."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin

import anyio
import pytest
from vcr.cassette import Cassette

from pydantic_ai import Embedder
from pydantic_ai.embeddings import (
    EmbeddingContent,
    EmbeddingContentPart,
    EmbeddingInput,
    EmbeddingModality,
    EmbeddingModel,
    EmbeddingResult,
    EmbeddingSettings,
    TestEmbeddingModel,
    WrapperEmbeddingModel,
    embedding_parts,
)
from pydantic_ai.embeddings._modality import embedding_modality
from pydantic_ai.embeddings.result import EmbedInputType
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    TextContent,
    UploadedFile,
    UserContent,
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

with try_import() as openai_imports_successful:
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.embeddings.cohere import CohereEmbeddingModel
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as voyageai_imports_successful:
    from pydantic_ai.embeddings.voyageai import VoyageAIEmbeddingModel
    from pydantic_ai.providers.voyageai import VoyageAIProvider

with try_import() as bedrock_imports_successful:
    from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel

with try_import() as google_imports_successful:
    from google.genai.types import Content, Part

    from pydantic_ai.embeddings import google as google_embeddings
    from pydantic_ai.embeddings.google import GoogleEmbeddingModel, GoogleEmbeddingSettings
    from pydantic_ai.providers.google import GoogleProvider

if TYPE_CHECKING:
    from google.genai import Client

    from pydantic_ai.providers import Provider

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
    """Unit rather than VCR: `AudioUrl` and `VideoUrl` reach no other test in this file, since the
    audio and video cases send `BinaryContent`, so covering this dispatch through `Embedder` would
    cost two more URL-download recordings for no extra assertion.
    """
    assert embedding_modality(part) == expected


def test_embedding_parts():
    """Unit rather than VCR: this is the fan-out every provider builds its request from, and pinning
    it directly says `EmbeddingContent` unwraps in order while a bare part doesn't, which a provider
    test can only show indirectly.
    """
    image = BinaryImage(data=b'\x00', media_type='image/png')
    assert embedding_parts('hello') == ['hello']
    assert embedding_parts(EmbeddingContent(['hello', image])) == ['hello', image]


def test_embedding_content_part_tracks_the_messages_union():
    """`EmbeddingContentPart` re-spells `UserContent` minus two members, so it can't drift silently.

    Unit rather than VCR: the failure this guards is a new `messages` content type nobody decided
    about for embeddings, which no provider request can surface. Mirrors
    `tests/test_messages.py::test_multi_modal_content_types_matches_union`.
    """

    def members(union: Any) -> set[Any]:
        # `UserContent` nests `MultiModalContent`, itself an `Annotated` union, so flatten recursively:
        # comparing one level deep leaves that whole union as a single opaque member.
        flattened: set[Any] = set()
        for member in get_args(union):
            if get_origin(member) is Annotated:
                member = get_args(member)[0]
            if get_args(member):
                flattened |= members(member)
            else:
                flattened.add(member)
        return flattened

    assert members(EmbeddingContentPart) == members(UserContent) - {CachePoint, UploadedFile}


def test_embedding_content_rejects_bare_str():
    """A `str` is itself a sequence of parts, so this type-checks and would embed five characters."""
    with pytest.raises(UserError, match=r'`EmbeddingContent` takes a sequence of parts, not a single string\.'):
        EmbeddingContent('a kiwi')


def test_embedding_content_rejects_an_empty_sequence():
    """Every input yields one embedding, and an empty one has nothing to make a vector from.

    The modality gate loops over the parts, so an empty `EmbeddingContent` clears it and reaches the
    provider as an empty `Content`, which fails there instead of at the call site.
    """
    with pytest.raises(UserError, match=r'`EmbeddingContent` needs at least one part to embed\.'):
        EmbeddingContent([])


async def test_test_model_embeds_every_modality(tiny_image: BinaryImage):
    """`TestEmbeddingModel` accepts every modality so multimodal apps can be tested against it."""
    model = TestEmbeddingModel(dimensions=4)
    result = await Embedder(model).embed_documents(['a kiwi fruit', tiny_image])

    assert len(result.embeddings) == 2
    assert result.inputs == ['a kiwi fruit', tiny_image]
    # Only the text is counted; the fake model has no way to tokenize an image.
    assert result.usage == snapshot(RequestUsage(input_tokens=3))


async def test_batch_can_be_any_iterable():
    """A batch is normalized by testing for a *single* input, so any iterable of inputs works.

    Corpora are commonly held in generators, sets and array types rather than in a `list`; the
    annotation says `Sequence`, but narrowing the runtime to one would break those callers.
    """
    embedder = Embedder(TestEmbeddingModel(dimensions=2))

    from_generator = await embedder.embed_documents(text for text in ['a kiwi fruit', 'a mango'])  # pyright: ignore[reportArgumentType]
    assert from_generator.inputs == ['a kiwi fruit', 'a mango']

    from_set = await embedder.embed_documents({'a kiwi fruit'})  # pyright: ignore[reportArgumentType]
    assert from_set.inputs == ['a kiwi fruit']


async def test_combined_content_yields_one_embedding(tiny_image: BinaryImage):
    """One sequence element yields one vector however many parts it holds, and keeps the container.

    Unit rather than VCR: the Google cases pin the fusion on the wire, and what is left is the
    accounting every model shares — a result entry per input, holding the `EmbeddingContent` the
    caller passed rather than its parts.
    """
    model = TestEmbeddingModel(dimensions=4)
    content = EmbeddingContent(['a kiwi fruit', tiny_image])
    result = await Embedder(model).embed_documents(content)

    assert len(result.embeddings) == 1
    assert result.inputs == [content]


def test_wrapper_delegates_supported_modalities():
    """Unit rather than VCR: delegation is provider-independent, and the wrapper reaching the wrong
    model's modalities would gate a request that should have been allowed — a negative no cassette shows.
    """
    assert WrapperEmbeddingModel(TestEmbeddingModel()).supported_modalities == TestEmbeddingModel().supported_modalities


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
async def test_unsupported_modality_raises(gemini_api_key: str, tiny_image: BinaryImage):
    """A model that can't embed the input fails locally, rather than as a provider error."""
    model = GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider(api_key=gemini_api_key))

    with pytest.raises(
        UserError,
        match=r'Pydantic AI does not support image inputs for `gemini-embedding-001`\. Supported modalities: text\.',
    ):
        await Embedder(model).embed_documents(['a kiwi fruit', tiny_image])


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
async def test_combined_content_on_text_only_model_raises(gemini_api_key: str):
    """A model that can't fuse parts fails locally too, naming the fusion rather than the modality.

    All-text parts clear the modality gate, so this is the second thing that stops a request before
    it is made — and nothing reaches the provider for a cassette to witness.
    """
    model = GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider(api_key=gemini_api_key))

    with pytest.raises(
        UserError, match=r'can only embed a single text part per input; pass the parts as separate inputs'
    ):
        await Embedder(model).embed_documents(EmbeddingContent(['a kiwi', 'fruit']))


class _OverreachingEmbeddingModel(EmbeddingModel):
    """Advertises images but sends text, which is what `prepare_text_embed()` refuses.

    The modality gate passes an image through, so this is the one implementation mistake the "only
    supports plain text inputs" error exists for: a shipped model that advertises a modality its
    request mapping can't build. No provider we ship is wrong in this way, hence the local subclass.
    """

    @property
    def model_name(self) -> str:
        return 'overreaching-model'

    @property
    def system(self) -> str:
        return 'test'

    @property
    def supported_modalities(self) -> frozenset[EmbeddingModality]:
        return frozenset({'text', 'image'})

    async def embed(
        self,
        inputs: EmbeddingInput | Sequence[EmbeddingInput],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        items, texts, _ = self.prepare_text_embed(inputs, settings)
        return EmbeddingResult(
            embeddings=[[0.0] for _ in texts],
            inputs=items,
            input_type=input_type,
            model_name=self.model_name,
            provider_name=self.system,
        )


async def test_prepare_text_embed_rejects_a_file_and_not_the_text_beside_it():
    """Only the input the mapping can't build is refused; the text the model does handle goes through."""
    embedder = Embedder(_OverreachingEmbeddingModel())

    result = await embedder.embed_documents('a kiwi fruit')
    assert result.inputs == ['a kiwi fruit']

    with pytest.raises(UserError, match=r'`overreaching-model` only supports plain text inputs, got `ImageUrl`\.'):
        await embedder.embed_documents(ImageUrl(url='https://example.com/img.png'))


async def test_prepare_text_embed_rejects_a_single_part_that_is_not_text():
    """A one-part `EmbeddingContent` unwraps only when the part is text; a lone file is still refused.

    The modality gate lets the image through, so this needs a model that advertises images and sends
    text — the same mismatch `_OverreachingEmbeddingModel` exists for.
    """
    embedder = Embedder(_OverreachingEmbeddingModel())

    with pytest.raises(UserError, match=r'`overreaching-model` can only embed a single text part per input'):
        await embedder.embed_documents(EmbeddingContent([ImageUrl(url='https://example.com/img.png')]))


def test_prepare_text_embed_unwraps_text_content():
    """Text-only implementations get plain strings, while the result keeps the original inputs.

    Unit rather than VCR: the two halves are indistinguishable on the wire — every text-only provider
    sends `texts` and reports `items`, and a cassette only witnesses the first.
    """
    tagged = TextContent(content='world', metadata={'chunk': 7})
    items, texts, _ = TestEmbeddingModel().prepare_text_embed(['hello', tagged])

    assert texts == ['hello', 'world']
    # The originals go to `EmbeddingResult.inputs`, so a `TextContent`'s metadata survives.
    assert items == ['hello', tagged]


async def test_lookup_by_text_finds_wrapped_text():
    """`result[text]` looks through a `TextContent`, so how the text was passed doesn't change lookup."""
    result = await Embedder(TestEmbeddingModel(dimensions=2)).embed_documents(
        ['a kiwi fruit', TextContent(content='a mango')]
    )

    assert result['a kiwi fruit'] == result.embeddings[0]
    assert result['a mango'] == result.embeddings[1]


@dataclass(frozen=True)
class _TextOnlyProviderCase:
    """One `prepare_text_embed()` call site, and the recording its request replays against."""

    id: str
    model: Callable[[pytest.FixtureRequest], EmbeddingModel]
    """Built from a fixture at call time, so a case costs nothing when its extra isn't installed."""

    cassette: str
    """An existing recording of the same `embed_query('Hello, world!')` against this model.

    Reused rather than re-recorded because the request is byte-identical: `prepare_text_embed()`
    unwraps the `TextContent` before the provider builds its body, and the default VCR matcher
    ignores the body regardless.
    """

    marks: tuple[pytest.MarkDecorator, ...]


_TEXT_ONLY_PROVIDER_CASES = [
    _TextOnlyProviderCase(
        id='openai',
        model=lambda request: OpenAIEmbeddingModel(
            'text-embedding-3-small', provider=OpenAIProvider(api_key=request.getfixturevalue('openai_api_key'))
        ),
        cassette='../test_embeddings/TestOpenAI.test_query.yaml',
        marks=(pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed'),),
    ),
    _TextOnlyProviderCase(
        id='cohere',
        model=lambda request: CohereEmbeddingModel(
            'embed-v4.0', provider=CohereProvider(api_key=request.getfixturevalue('co_api_key'))
        ),
        cassette='../test_embeddings/TestCohere.test_query.yaml',
        marks=(pytest.mark.skipif(not cohere_imports_successful(), reason='Cohere not installed'),),
    ),
    _TextOnlyProviderCase(
        id='voyageai',
        model=lambda request: VoyageAIEmbeddingModel(
            'voyage-3.5', provider=VoyageAIProvider(api_key=request.getfixturevalue('voyage_api_key'))
        ),
        cassette='../test_embeddings/TestVoyageAI.test_query.yaml',
        marks=(pytest.mark.skipif(not voyageai_imports_successful(), reason='VoyageAI not installed'),),
    ),
    _TextOnlyProviderCase(
        id='bedrock-batched',
        model=lambda request: BedrockEmbeddingModel(
            'cohere.embed-v4:0', provider=request.getfixturevalue('bedrock_provider')
        ),
        cassette='../test_embeddings/TestBedrock.test_cohere_v4_minimal.yaml',
        marks=(pytest.mark.skipif(not bedrock_imports_successful(), reason='Bedrock not installed'),),
    ),
    _TextOnlyProviderCase(
        id='bedrock-per-input',
        # Titan takes one input per request, so Bedrock reports the inputs from a second place.
        model=lambda request: BedrockEmbeddingModel(
            'amazon.titan-embed-text-v2:0', provider=request.getfixturevalue('bedrock_provider')
        ),
        cassette='../test_embeddings/TestBedrock.test_titan_v2_minimal.yaml',
        marks=(pytest.mark.skipif(not bedrock_imports_successful(), reason='Bedrock not installed'),),
    ),
]


@pytest.mark.parametrize(
    'case',
    [pytest.param(c, id=c.id, marks=(*c.marks, pytest.mark.vcr(c.cassette))) for c in _TEXT_ONLY_PROVIDER_CASES],
)
async def test_text_only_provider_reports_original_inputs(case: _TextOnlyProviderCase, request: pytest.FixtureRequest):
    """A text-only provider sends the unwrapped text but reports the input the caller passed.

    `prepare_text_embed()` returns the inputs and the text to send separately, and every text-only
    provider passes the former to `EmbeddingResult.inputs`. A case per call site pins that fork end to
    end — Bedrock twice, since batching models and per-input models build the result separately — so a
    provider that regressed to reporting its own request payload fails here.
    `SentenceTransformerEmbeddingModel` is the one call site left unpinned: reaching it needs the
    Hugging Face Hub outage handling that lives with its own tests.
    """
    tagged = TextContent(content='Hello, world!', metadata={'chunk': 7})

    result = await Embedder(case.model(request)).embed_query(tagged)

    assert result.inputs == [tagged]
    assert result['Hello, world!'] == result.embeddings[0]


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.vcr('../test_embeddings/TestOpenAI.test_query.yaml')
@pytest.mark.parametrize(
    'part',
    [pytest.param('Hello, world!', id='str'), pytest.param(TextContent(content='Hello, world!'), id='text-content')],
)
async def test_single_text_part_content_embeds_on_a_text_only_provider(part: EmbeddingContentPart, openai_api_key: str):
    """A text-only provider takes a one-part `EmbeddingContent`: one part is nothing to fuse.

    Refusing it would make `EmbeddingContent` unusable for a caller that builds inputs uniformly and
    only sometimes has more than one part. Replays the same `embed_query('Hello, world!')` recording
    the cases above reuse — the unwrapping happens before the provider builds its body, so the
    request is byte-identical to embedding the bare part.
    """
    model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
    content = EmbeddingContent([part])

    result = await Embedder(model).embed_query(content)

    assert len(result.embeddings) == 1
    # The container is what the caller passed, so it is what the result reports.
    assert result.inputs == [content]


@dataclass(frozen=True)
class _Assets:
    """The binary fixtures a case builds its inputs from, so nothing is read from disk at import time."""

    image: BinaryImage
    audio: BinaryContent
    video: BinaryContent
    document: BinaryContent


@pytest.fixture
def assets(
    image_content: BinaryImage,
    audio_content: BinaryContent,
    video_content: BinaryContent,
    document_content: BinaryContent,
) -> _Assets:
    return _Assets(image=image_content, audio=audio_content, video=video_content, document=document_content)


@dataclass
class _GoogleMultimodalCase:
    """A multimodal request to `gemini-embedding-2`, asserted at the wire and at the result."""

    id: str
    inputs: Callable[[_Assets], EmbeddingInput | Sequence[EmbeddingInput]]
    expected_parts: list[list[str]]
    """Per input sent to the API, a description of each part: the verbatim text, or `<media_type>`."""

    expected_embeddings: int
    settings: GoogleEmbeddingSettings | None = None
    expected_warning: str | None = None


# The GoogleEmbeddingSettings(...) calls are only evaluated when the google extra is installed.
_GOOGLE_MULTIMODAL_CASES: list[_GoogleMultimodalCase] = (
    [
        _GoogleMultimodalCase(
            id='image-only',
            inputs=lambda assets: assets.image,
            expected_parts=[['<image/jpeg>']],
            expected_embeddings=1,
        ),
        _GoogleMultimodalCase(
            id='audio-only',
            inputs=lambda assets: assets.audio,
            expected_parts=[['<audio/mpeg>']],
            expected_embeddings=1,
        ),
        _GoogleMultimodalCase(
            id='video-only',
            inputs=lambda assets: assets.video,
            expected_parts=[['<video/mp4>']],
            expected_embeddings=1,
        ),
        _GoogleMultimodalCase(
            id='document-only',
            inputs=lambda assets: assets.document,
            expected_parts=[['<application/pdf>']],
            expected_embeddings=1,
        ),
        _GoogleMultimodalCase(
            id='text-and-image-combined',
            inputs=lambda assets: EmbeddingContent(
                ['a kiwi fruit', TextContent(content='green and fuzzy'), assets.image]
            ),
            expected_parts=[['a kiwi fruit', 'green and fuzzy', '<image/jpeg>']],
            expected_embeddings=1,
        ),
        _GoogleMultimodalCase(
            id='text-parts-combined',
            inputs=lambda _: EmbeddingContent(['a kiwi fruit', 'green and fuzzy']),
            # Fusing text parts costs the task prefix, which is why this warns on the default task too.
            expected_parts=[['a kiwi fruit', 'green and fuzzy']],
            expected_embeddings=1,
            expected_warning='an `EmbeddingContent` of several text parts is embedded unconditioned',
        ),
        _GoogleMultimodalCase(
            id='batch-text-and-image-url',
            inputs=lambda _: ['a kiwi fruit', ImageUrl(url=KIWI_IMAGE_URL)],
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
    assets: _Assets,
    gemini_api_key: str,
    disable_ssrf_protection_for_vcr: None,
    google_embed_content_spy: Callable[[Provider[Client]], dict[str, Any]],
):
    """`gemini-embedding-2` embeds every modality, and combines parts into a single vector."""
    provider = GoogleProvider(api_key=gemini_api_key)
    model = GoogleEmbeddingModel('gemini-embedding-2', provider=provider)
    captured = google_embed_content_spy(provider)

    inputs = case.inputs(assets)
    if case.expected_warning is not None:
        with pytest.warns(UserWarning, match=case.expected_warning):
            result = await Embedder(model).embed_documents(inputs, settings=case.settings)
    else:
        result = await Embedder(model).embed_documents(inputs, settings=case.settings)

    contents: list[Content] = captured['contents']
    sent = [[_describe_part(part) for part in (content.parts or [])] for content in contents]
    assert sent == case.expected_parts
    assert len(result.embeddings) == case.expected_embeddings
    assert result.inputs == (inputs if isinstance(inputs, list) else [inputs])


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
@pytest.mark.vcr
async def test_google_preview_embeds_a_file_and_still_sends_a_task_type(
    assets: _Assets,
    gemini_api_key: str,
    google_embed_content_spy: Callable[[Provider[Client]], dict[str, Any]],
):
    """`gemini-embedding-2-preview` embeds a file while conditioning on `task_type`, unlike `gemini-embedding-2`.

    Multimodality and task conditioning are separate capabilities, and this model has the first without
    the second: its request carries the image as `inline_data` *and* a `task_type`, where
    `gemini-embedding-2` conditions with a text prefix and sends `task_type=None`. Google documents the
    modalities for `gemini-embedding-2` only, so this recording is the evidence the preview accepts them.
    """
    provider = GoogleProvider(api_key=gemini_api_key)
    model = GoogleEmbeddingModel('gemini-embedding-2-preview', provider=provider)
    captured = google_embed_content_spy(provider)

    result = await Embedder(model).embed_documents(assets.image, settings=GoogleEmbeddingSettings(dimensions=128))

    contents: list[Content] = captured['contents']
    assert [[_describe_part(part) for part in (content.parts or [])] for content in contents] == [['<image/jpeg>']]
    assert captured['config'].task_type == 'RETRIEVAL_DOCUMENT'
    assert len(result.embeddings) == 1


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
def test_google_preview_supports_every_modality(gemini_api_key: str):
    """The preview is gated like `gemini-embedding-2`, while `gemini-embedding-001` stays text-only.

    Unit rather than VCR: the recording above witnesses one modality, and recording audio, video and a
    document against the preview too would pay three more requests to restate one frozenset. The
    text-only half is a rejection that never reaches the provider, pinned end to end by
    `test_unsupported_modality_raises`.
    """
    provider = GoogleProvider(api_key=gemini_api_key)

    assert GoogleEmbeddingModel('gemini-embedding-2-preview', provider=provider).supported_modalities == snapshot(
        frozenset({'text', 'image', 'audio', 'video', 'document'})
    )
    assert GoogleEmbeddingModel('gemini-embedding-001', provider=provider).supported_modalities == snapshot(
        frozenset({'text'})
    )


def _solid_png(red: int) -> BinaryImage:
    """A solid-colour PNG, distinct per `red` value.

    Built rather than read from `tests/assets/`: seven copies of `kiwi.jpg` would put close to a
    megabyte of base64 in the cassette, where seven of these take a few hundred bytes.
    """

    size = 64

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

    header = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)
    row = b'\x00' + bytes([red, 0x80, 0x40]) * size
    return BinaryImage(
        data=b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', header)
        + chunk(b'IDAT', zlib.compress(row * size))
        + chunk(b'IEND', b''),
        media_type='image/png',
    )


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
@pytest.mark.vcr
async def test_google_batches_more_images_than_the_documented_limit(
    gemini_api_key: str,
    google_embed_content_spy: Callable[[Provider[Client]], dict[str, Any]],
    vcr: Cassette,
):
    """Seven single-image inputs go out as one `batchEmbedContents` call and come back as seven vectors.

    `embed()` never chunks a batch, so Google's documented limit of 6 images for multimodal embedding
    would be a real defect if it counted images per HTTP request. This recording pins that it doesn't:
    the limit constrains the parts of one input, not a batch of one-image inputs.
    """
    provider = GoogleProvider(api_key=gemini_api_key)
    model = GoogleEmbeddingModel('gemini-embedding-2', provider=provider)
    captured = google_embed_content_spy(provider)

    images = [_solid_png(red) for red in range(0, 7 * 32, 32)]
    result = await Embedder(model).embed_documents(images, settings=GoogleEmbeddingSettings(dimensions=128))

    # The claim above is about HTTP, and the SDK spy alone would still pass if `google-genai` fanned
    # the seven contents out into a call each. The recording is what says it didn't, and against which
    # model id the batch endpoint was reached.
    assert [request.uri for request in vcr.requests] == snapshot(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        ['https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents']
    )

    contents: list[Content] = captured['contents']
    assert [len(content.parts or []) for content in contents] == [1] * 7
    assert len(result.embeddings) == 7
    assert result.inputs == images


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
@pytest.mark.vcr
async def test_google_rejects_more_images_than_one_input_may_hold(gemini_api_key: str):
    """The same seven images fused into one input is where the limit bites, and Google says where.

    The counterpart to the batch above: Google spells the limit out as "per input instance", so the
    ceiling belongs to `EmbeddingContent`, not to the batch, and chunking a batch would not help.
    """
    model = GoogleEmbeddingModel('gemini-embedding-2', provider=GoogleProvider(api_key=gemini_api_key))
    content = EmbeddingContent([_solid_png(red) for red in range(0, 7 * 32, 32)])

    with pytest.raises(ModelHTTPError, match=r'at most 6 image parts per input instance, but 7 were provided'):
        await Embedder(model).embed_documents(content, settings=GoogleEmbeddingSettings(dimensions=128))


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
async def test_concurrent_downloads_keep_part_order(monkeypatch: pytest.MonkeyPatch):
    """Parts download concurrently, so the order they finish in must not be the order they are sent.

    Unit rather than VCR, and deliberately so: under cassette playback a download returns without
    real latency, so completion order collapses back to start order and no recording can witness a
    reordering. Getting this wrong would silently pair each vector with the wrong input.
    """
    media_types = {'https://example.com/slow.bin': 'image/png', 'https://example.com/quick.bin': 'image/jpeg'}

    async def staggered_download(item: Any, **kwargs: Any) -> Any:
        # The first part finishes last, so collecting on completion would swap the two.
        await anyio.sleep(0.05 if item.url.endswith('slow.bin') else 0)
        return {'data': b'\x00', 'data_type': media_types[item.url]}

    monkeypatch.setattr(google_embeddings, 'download_item', staggered_download)

    contents = await google_embeddings._map_contents(  # pyright: ignore[reportPrivateUsage]
        [[ImageUrl(url=url) for url in media_types]]
    )

    content = contents[0]
    assert isinstance(content, Content)
    assert [_describe_part(part) for part in content.parts or []] == ['<image/png>', '<image/jpeg>']


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_describes_non_text_inputs(capfire: CaptureLogfire, tiny_image: BinaryImage):
    """Files are described by media type instead of being dumped into the span as bytes.

    The OTel GenAI spec only names image, audio and video, so a document carries no `modality` key —
    the same vocabulary an agent span uses, rather than one invented for embeddings.
    """
    model = TestEmbeddingModel(dimensions=2)
    embedder = Embedder(model, instrument=InstrumentationSettings(include_content=True, include_binary_content=False))

    await embedder.embed_documents(
        [
            'a kiwi fruit',
            tiny_image,
            BinaryContent(data=b'\x00\x01\x02', media_type='application/pdf'),
            ImageUrl(url='https://example.com/img.png'),
            DocumentUrl(url='https://example.com/doc.pdf'),
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
            {'type': 'blob', 'mime_type': 'application/pdf'},
            {'type': 'uri', 'modality': 'image', 'uri': 'https://example.com/img.png', 'mime_type': 'image/png'},
            {'type': 'uri', 'uri': 'https://example.com/doc.pdf', 'mime_type': 'application/pdf'},
            {'type': 'uri', 'modality': 'image', 'uri': 'https://example.com/redirects/to/an/image'},
            ['a kiwi fruit', {'type': 'blob', 'modality': 'image', 'mime_type': 'image/jpeg'}],
        ]
    )


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_includes_binary_content(capfire: CaptureLogfire, tiny_image: BinaryImage):
    """With `include_binary_content` the bytes ride along, base64-encoded beside the media type.

    The positive half of `test_instrumentation_describes_non_text_inputs`. Unit rather than VCR: the
    span is built from the inputs before any request, so a recording would only add a provider to an
    assertion that has nothing provider-specific in it.
    """
    model = TestEmbeddingModel(dimensions=2)
    embedder = Embedder(model, instrument=InstrumentationSettings(include_content=True))

    await embedder.embed_documents(tiny_image)

    [span] = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert span['attributes']['inputs'] == snapshot(
        [{'type': 'blob', 'modality': 'image', 'mime_type': 'image/jpeg', 'content': 'AAEC'}]
    )
