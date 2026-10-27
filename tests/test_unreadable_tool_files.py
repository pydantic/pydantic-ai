"""A file a tool returns that the model can't read is sent as a note naming it, rather than ending the run.

What the model can read comes from the profile's file-input flags, which each model adapter narrows to what its
message mapping can send. Most tests here use `FunctionModel` with an explicit profile, so the claim is the
replacement itself; `test_adapter_file_input_flags` pins each adapter's flags, and the per-provider behavior is
covered by `test_multimodal_tool_return_matrix` in `tests/models/test_multimodal_tool_returns.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.capabilities import FileUnderstanding
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as groq_available:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as mistral_available:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as cohere_available:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as huggingface_available:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

with try_import() as xai_available:
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider


MEMO = BinaryContent(b'RIFF', media_type='audio/wav', identifier='memo')
MEMO_NOTE = 'File memo (audio/wav, 4 bytes) was not sent: this model cannot read audio files.'
NO_AUDIO = ModelProfile(supports_audio_file_input=False)


def tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def calls_get_file_then(sent: list[list[ModelMessage]]) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    """A model function that calls `get_file`, then keeps the request carrying its result and answers."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not tool_returns(messages):
            return ModelResponse(parts=[ToolCallPart('get_file', {})])
        sent.append(messages)
        return ModelResponse(parts=[TextPart('done')])

    return respond


def agent_with_tool_file(model: Model, content: Any, **kwargs: Any) -> Agent[None, str]:
    agent = Agent(model, **kwargs)

    @agent.tool_plain
    def get_file() -> Any:
        return content

    return agent


async def test_tool_file_the_model_cannot_read_is_sent_as_a_note():
    """The note takes the file's place in the tool result; the history keeps the file."""
    sent: list[list[ModelMessage]] = []
    image = ImageUrl('https://example.com/chart.png')
    data = BinaryContent(b'{}', media_type='application/json', identifier='data')
    agent = agent_with_tool_file(
        FunctionModel(calls_get_file_then(sent), profile=NO_AUDIO), ['Recorded:', MEMO, image, data]
    )

    result = await agent.run('Get the file.')

    [request] = sent
    # An image is read, and JSON is no kind of file the flags judge, so both are left to the model adapter.
    assert tool_returns(request)[0].content == ['Recorded:', MEMO_NOTE, image, data]
    assert tool_returns(result.all_messages())[0].content == ['Recorded:', MEMO, image, data]


async def test_tool_file_the_model_can_read_is_sent_as_it_is():
    sent: list[list[ModelMessage]] = []
    agent = agent_with_tool_file(FunctionModel(calls_get_file_then(sent)), MEMO)

    await agent.run('Get the file.')

    [request] = sent
    assert tool_returns(request)[0].content == MEMO


@pytest.mark.parametrize(
    ('profile', 'content', 'note'),
    [
        pytest.param(
            ModelProfile(supports_image_input=False),
            BinaryContent(b'\x89PNG', media_type='image/png', identifier='chart'),
            'File chart (image/png, 4 bytes) was not sent: this model cannot read images.',
            id='image',
        ),
        pytest.param(
            ModelProfile(supports_document_input=False),
            DocumentUrl('https://example.com/report.pdf', identifier='report'),
            'File report (application/pdf) was not sent: this model cannot read documents.',
            id='document',
        ),
        pytest.param(
            ModelProfile(supported_document_media_types=frozenset({'application/pdf'})),
            BinaryContent(b'a,b', media_type='text/csv', identifier='table'),
            'File table (text/csv, 3 bytes) was not sent: this model cannot read text/csv documents.',
            id='document-media-type',
        ),
        pytest.param(
            ModelProfile(supports_video_input=False),
            VideoUrl('https://example.com/stream', media_type='video/mp4', identifier='clip'),
            'File clip (video/mp4) was not sent: this model cannot read video.',
            id='video',
        ),
        pytest.param(
            ModelProfile(supports_video_input=False),
            VideoUrl('https://example.com/stream', identifier='clip'),
            'File clip was not sent: this model cannot read video.',
            id='video-of-unknown-media-type',
        ),
    ],
)
async def test_note_names_the_file_and_what_the_model_cannot_read(profile: ModelProfile, content: Any, note: str):
    sent: list[list[ModelMessage]] = []
    agent = agent_with_tool_file(FunctionModel(calls_get_file_then(sent), profile=profile), content)

    await agent.run('Get the file.')

    [request] = sent
    assert tool_returns(request)[0].content == note


async def test_document_of_unknown_media_type_is_left_to_the_adapter():
    """Narrowing the document media types only judges the known ones; another is sent as it is."""
    sent: list[list[ModelMessage]] = []
    document = DocumentUrl('https://example.com/data.json')
    profile = ModelProfile(supported_document_media_types=frozenset({'application/pdf'}))
    agent = agent_with_tool_file(FunctionModel(calls_get_file_then(sent), profile=profile), document)

    await agent.run('Get the file.')

    [request] = sent
    assert tool_returns(request)[0].content == document


async def test_file_in_a_user_prompt_is_sent_as_it_is():
    """The user chose to send it, so it is the model adapter's to accept or refuse."""
    sent: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(messages)
        return ModelResponse(parts=[TextPart('done')])

    await Agent(FunctionModel(respond, profile=NO_AUDIO)).run(['Listen:', MEMO])

    [request] = sent
    prompt = request[-1].parts[-1]
    assert isinstance(prompt, UserPromptPart)
    assert prompt.content == ['Listen:', MEMO]


async def test_tool_return_content_is_sent_as_it_is():
    """`ToolReturn.content` is the developer adding user content on purpose, so it is not replaced either."""
    sent: list[list[ModelMessage]] = []
    agent = agent_with_tool_file(
        FunctionModel(calls_get_file_then(sent), profile=NO_AUDIO),
        ToolReturn(return_value='Attached.', content=[MEMO]),
    )

    await agent.run('Get the file.')

    [request] = sent
    user_prompts = [
        part
        for message in request
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert any(not isinstance(part.content, str) and MEMO in part.content for part in user_prompts)


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_unreadable_file_in_a_user_prompt_still_raises(allow_model_requests: None):
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test'))

    with pytest.raises(NotImplementedError, match='Unsupported binary content type in Anthropic user prompts'):
        await Agent(model).run(['Listen:', MEMO])


async def test_each_fallback_candidate_applies_its_own_profile():
    """The note is decided per candidate, so a fallback that reads the file is sent it."""
    first_sent: list[list[ModelMessage]] = []
    second_sent: list[list[ModelMessage]] = []

    def first(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not tool_returns(messages):
            return ModelResponse(parts=[ToolCallPart('get_file', {})])
        first_sent.append(messages)
        raise ModelHTTPError(status_code=503, model_name='first')

    model = FallbackModel(
        FunctionModel(first, profile=NO_AUDIO),
        FunctionModel(calls_get_file_then(second_sent)),
    )
    agent = agent_with_tool_file(model, MEMO)

    await agent.run('Get the file.')

    assert tool_returns(first_sent[0])[0].content == MEMO_NOTE
    assert tool_returns(second_sent[0])[0].content == MEMO


async def test_file_understanding_describes_the_file_before_a_note_is_needed():
    """`FileUnderstanding` runs before the request is prepared, so the model gets a description, not a note."""
    sent: list[list[ModelMessage]] = []

    def describe(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Someone reading a shopping list.')])

    agent = agent_with_tool_file(
        FunctionModel(calls_get_file_then(sent), profile=NO_AUDIO),
        MEMO,
        capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describe))],
    )

    await agent.run('Get the file.')

    [request] = sent
    assert tool_returns(request)[0].content == snapshot("""\
-----BEGIN FILE id="memo" type="text/plain"-----
Someone reading a shopping list.
-----END FILE id="memo"-----\
""")


def file_input_flags(model: Model) -> dict[str, object]:
    profile = model.profile
    return {
        'image': profile.get('supports_image_input'),
        'document': profile.get('supports_document_input'),
        'document_media_types': sorted(profile.get('supported_document_media_types', ())),
        'audio': profile.get('supports_audio_file_input'),
        'video': profile.get('supports_video_input'),
    }


ALL_DOCUMENTS = [
    'application/msword',
    'application/pdf',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/csv',
    'text/html',
    'text/markdown',
    'text/plain',
]


def _anthropic() -> Model:
    return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test'))


def _openai_chat() -> Model:
    return OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key='test'))


def _openai_chat_without_documents() -> Model:
    return OpenAIChatModel(
        'gpt-5',
        provider=OpenAIProvider(api_key='test'),
        profile=OpenAIModelProfile(openai_chat_supports_document_input=False),
    )


def _openai_responses() -> Model:
    return OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test'))


def _openrouter() -> Model:
    return OpenRouterModel('google/gemini-2.5-flash', provider=OpenRouterProvider(api_key='test'))


def _groq() -> Model:
    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key='test'))


def _mistral() -> Model:
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test'))


def _cohere() -> Model:
    return CohereModel('command-r-plus', provider=CohereProvider(api_key='test'))


def _huggingface() -> Model:
    return HuggingFaceModel('Qwen/Qwen2.5-VL-72B-Instruct', provider=HuggingFaceProvider(api_key='test'))


def _xai() -> Model:
    return XaiModel('grok-4-1-fast-non-reasoning', provider=XaiProvider(api_key='test'))


def _bedrock() -> Model:
    provider = BedrockProvider(region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test')
    return BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=provider)


def _google() -> Model:
    return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test'))


@pytest.mark.parametrize(
    ('make_model', 'expected'),
    [
        pytest.param(
            _anthropic,
            (
                {
                    'image': True,
                    'document': True,
                    'document_media_types': ['application/pdf', 'text/plain'],
                    'audio': False,
                    'video': False,
                }
            ),
            marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),
            id='anthropic',
        ),
        pytest.param(
            _openai_chat,
            ({'image': True, 'document': True, 'document_media_types': ALL_DOCUMENTS, 'audio': True, 'video': False}),
            marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
            id='openai_chat',
        ),
        pytest.param(
            _openai_chat_without_documents,
            (
                {
                    'image': True,
                    'document': True,
                    'document_media_types': ['text/csv', 'text/html', 'text/markdown', 'text/plain'],
                    'audio': True,
                    'video': False,
                }
            ),
            marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
            id='openai_chat_without_documents',
        ),
        pytest.param(
            _openai_responses,
            ({'image': True, 'document': True, 'document_media_types': ALL_DOCUMENTS, 'audio': False, 'video': False}),
            marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
            id='openai_responses',
        ),
        pytest.param(
            _openrouter,
            ({'image': True, 'document': True, 'document_media_types': ALL_DOCUMENTS, 'audio': True, 'video': True}),
            marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
            id='openrouter',
        ),
        pytest.param(
            _groq,
            ({'image': True, 'document': False, 'document_media_types': ALL_DOCUMENTS, 'audio': False, 'video': False}),
            marks=pytest.mark.skipif(not groq_available(), reason='groq not installed'),
            id='groq',
        ),
        pytest.param(
            _mistral,
            (
                {
                    'image': True,
                    'document': True,
                    'document_media_types': ['application/pdf', 'text/csv', 'text/html', 'text/markdown', 'text/plain'],
                    'audio': False,
                    'video': False,
                }
            ),
            marks=pytest.mark.skipif(not mistral_available(), reason='mistral not installed'),
            id='mistral',
        ),
        pytest.param(
            _cohere,
            (
                {
                    'image': False,
                    'document': False,
                    'document_media_types': ALL_DOCUMENTS,
                    'audio': False,
                    'video': False,
                }
            ),
            marks=pytest.mark.skipif(not cohere_available(), reason='cohere not installed'),
            id='cohere',
        ),
        pytest.param(
            _huggingface,
            ({'image': True, 'document': False, 'document_media_types': ALL_DOCUMENTS, 'audio': False, 'video': False}),
            marks=pytest.mark.skipif(not huggingface_available(), reason='huggingface not installed'),
            id='huggingface',
        ),
        pytest.param(
            _xai,
            ({'image': True, 'document': True, 'document_media_types': ALL_DOCUMENTS, 'audio': False, 'video': False}),
            marks=pytest.mark.skipif(not xai_available(), reason='xai not installed'),
            id='xai',
        ),
        pytest.param(
            _bedrock,
            ({'image': True, 'document': True, 'document_media_types': ALL_DOCUMENTS, 'audio': False, 'video': True}),
            marks=pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed'),
            id='bedrock',
        ),
        pytest.param(
            _google,
            ({'image': True, 'document': True, 'document_media_types': ALL_DOCUMENTS, 'audio': True, 'video': True}),
            marks=pytest.mark.skipif(not google_available(), reason='google not installed'),
            id='google',
        ),
    ],
)
def test_adapter_file_input_flags(make_model: Callable[[], Model], expected: dict[str, object]):
    """Each adapter turns off the kinds of file its message mapping refuses, so a tool's file of that kind is noted."""
    assert file_input_flags(make_model()) == expected


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
def test_adapter_narrowing_applies_over_the_user_profile():
    """A profile can narrow what the adapter sends further, but can't claim a file the mapping would refuse."""
    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key='test'),
        profile=ModelProfile(
            supports_audio_file_input=True,
            supported_document_media_types=frozenset({'application/pdf', 'text/csv'}),
        ),
    )
    assert file_input_flags(model) == snapshot(
        {
            'image': True,
            'document': True,
            'document_media_types': ['application/pdf'],
            'audio': False,
            'video': False,
        }
    )
    pdf = BinaryContent(b'%PDF', media_type='application/pdf', identifier='report')
    text = BinaryContent(b'notes', media_type='text/plain', identifier='notes')
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Get the files.')]),
        ModelResponse(parts=[ToolCallPart('get_file', {}, tool_call_id='call_1')]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    'get_file',
                    [pdf, text, AudioUrl('https://example.com/a.mp3', identifier='song')],
                    tool_call_id='call_1',
                )
            ]
        ),
    ]

    [*_, request] = model.prepare_messages(messages)

    assert tool_returns([request])[0].content == snapshot(
        [
            pdf,
            'File notes (text/plain, 5 bytes) was not sent: this model cannot read text/plain documents.',
            'File song (audio/mpeg) was not sent: this model cannot read audio files.',
        ]
    )
