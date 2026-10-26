"""Tests for `FileUnderstanding`.

All but the last test are unit tests on `FunctionModel` and `TestModel` rather than VCR recordings: they count
describer calls, set model profiles by hand and control concurrency, none of which a recording can do.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import anyio
import pytest

from pydantic_ai import (
    Agent,
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.capabilities import FileUnderstanding
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import sanitize_messages
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.google import GoogleProvider

READINESS_WAIT_TIMEOUT = 10


class Describer:
    """A fallback model that describes whatever file it is given, and counts how often it is asked."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        part = messages[-1].parts[-1]
        assert isinstance(part, UserPromptPart)
        [item] = part.content
        assert isinstance(item, ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent)
        return ModelResponse(parts=[TextPart(f'A description of {item.identifier}.')])


def last_prompt(seen: list[list[ModelMessage]]) -> list[object]:
    part = seen[-1][-1].parts[-1]
    assert isinstance(part, UserPromptPart)
    return [part.content] if isinstance(part.content, str) else list(part.content)


async def test_unsupported_files_are_described():
    """Files the model's profile rejects reach it as descriptions; the rest go through untouched."""
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    text_only = FunctionModel(capture, profile=ModelProfile(supports_image_input=False, supports_document_input=False))
    describer = Describer()
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])
    await agent.run(
        [
            'Compare these:',
            DocumentUrl('https://example.com/report.pdf'),
            BinaryContent(b'\x89PNG', media_type='image/png', identifier='chart'),
            # The same document again: one file, one description.
            DocumentUrl('https://example.com/report.pdf'),
        ]
    )

    # A description is prose, so each block says `text/plain` rather than the media type of the file it replaces.
    assert last_prompt(seen) == snapshot(
        [
            'Compare these:',
            """\
-----BEGIN FILE id="a5f6ba" type="text/plain"-----
A description of a5f6ba.
-----END FILE id="a5f6ba"-----\
""",
            """\
-----BEGIN FILE id="chart" type="text/plain"-----
A description of chart.
-----END FILE id="chart"-----\
""",
            """\
-----BEGIN FILE id="a5f6ba" type="text/plain"-----
A description of a5f6ba.
-----END FILE id="a5f6ba"-----\
""",
        ]
    )
    assert describer.calls == 2

    # A run without files asks nothing.
    await agent.run('just text')
    assert describer.calls == 2
    assert last_prompt(seen) == ['just text']


async def test_supported_files_are_sent_as_they_are():
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    describer = Describer()
    image = ImageUrl('https://example.com/cat.png')
    document = DocumentUrl('https://example.com/cat.pdf')
    video = VideoUrl('https://example.com/cat.mp4')
    audio = AudioUrl('https://example.com/cat.mp3')
    # Every kind of file is accepted by default: only a model adapter or profile that says otherwise turns one off.
    agent = Agent(
        FunctionModel(capture), capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))]
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(['Two', 'texts'])]),
        ModelResponse(parts=[TextPart('ok')]),
    ]
    await agent.run([image, document, video, audio], message_history=history)

    assert last_prompt(seen) == [image, document, video, audio]
    assert describer.calls == 0


async def test_video_and_audio_are_described_when_the_profile_rejects_them():
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    describer = Describer()
    model = FunctionModel(capture, profile=ModelProfile(supports_video_input=False, supports_audio_file_input=False))
    agent = Agent(model, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])
    await agent.run(
        [
            VideoUrl('https://example.com/cat.mp4', identifier='clip'),
            AudioUrl('https://example.com/cat.mp3', identifier='purr'),
        ]
    )

    assert last_prompt(seen) == snapshot(
        [
            """\
-----BEGIN FILE id="clip" type="text/plain"-----
A description of clip.
-----END FILE id="clip"-----\
""",
            """\
-----BEGIN FILE id="purr" type="text/plain"-----
A description of purr.
-----END FILE id="purr"-----\
""",
        ]
    )
    assert describer.calls == 2


async def test_urls_without_an_inferable_media_type_are_described():
    """A URL with no extension has no media type to read, so it counts as not text and the describer fetches it."""
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    text_only = FunctionModel(capture, profile=ModelProfile(supports_image_input=False, supports_document_input=False))
    describer = Describer()
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])
    await agent.run(
        [
            DocumentUrl('https://example.com/download?id=1', identifier='doc'),
            ImageUrl('https://example.com/render?id=2', identifier='img'),
        ],
        usage_limits=UsageLimits(request_limit=10),
    )

    assert describer.calls == 2
    assert last_prompt(seen) == snapshot(
        [
            """\
-----BEGIN FILE id="doc" type="text/plain"-----
A description of doc.
-----END FILE id="doc"-----\
""",
            """\
-----BEGIN FILE id="img" type="text/plain"-----
A description of img.
-----END FILE id="img"-----\
""",
        ]
    )


async def test_custom_instructions_reach_the_describer():
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('A cat.')])

    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(
        text_only,
        capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(capture), instructions='One sentence.')],
    )
    await agent.run([ImageUrl('https://example.com/cat.png')])
    request = seen[0][0]
    assert isinstance(request, ModelRequest)
    assert request.instructions == 'One sentence.'


async def test_fallback_model_is_left_alone():
    """A `FallbackModel` has no profile to read, so nothing is described until a model answers."""
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    describer = Describer()
    model = FallbackModel(FunctionModel(capture, profile=ModelProfile(supports_image_input=False)))
    agent = Agent(model, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])
    await agent.run([ImageUrl('https://example.com/cat.png')])
    assert describer.calls == 0
    assert isinstance(last_prompt(seen)[0], ImageUrl)


async def test_text_like_documents_are_inlined_not_described():
    """A document the model could read as text is inlined as it is; the describer is not asked."""
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    text_only = FunctionModel(capture, profile=ModelProfile(supports_document_input=False))
    describer = Describer()
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])
    with patch('pydantic_ai.capabilities.file_understanding.download_item', new_callable=AsyncMock) as download:
        download.return_value = {'data': 'Buy milk.', 'data_type': 'text/plain'}
        await agent.run(
            [
                DocumentUrl('https://example.com/notes.txt'),
                BinaryContent(b'a,b\n1,2', media_type='text/csv', identifier='table'),
            ]
        )

    assert last_prompt(seen) == snapshot(
        [
            """\
-----BEGIN FILE id="0f059c" type="text/plain"-----
Buy milk.
-----END FILE id="0f059c"-----\
""",
            """\
-----BEGIN FILE id="table" type="text/csv"-----
a,b
1,2
-----END FILE id="table"-----\
""",
        ]
    )
    assert describer.calls == 0


def _png(name: str) -> BinaryContent:
    return BinaryContent(f'\x89PNG {name}'.encode(), media_type='image/png', identifier=name)


async def test_separate_conversations_describe_the_same_file_again():
    """Descriptions live in each conversation's history, not in the process, so runs don't share them."""
    describer = Describer()
    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])

    await agent.run([ImageUrl('https://example.com/receipt.png?signature=abc')])
    await agent.run([ImageUrl('https://example.com/receipt.png?signature=abc')])
    await agent.run([_png('chart')])
    await agent.run([_png('chart')])
    assert describer.calls == 4


async def test_the_same_bytes_under_two_media_types_are_two_files():
    """The media type decides whether bytes are described or decoded, so it is part of their identity."""
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    text_only = FunctionModel(capture, profile=ModelProfile(supports_image_input=False, supports_document_input=False))
    describer = Describer()
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])

    data = b'a,b\n1,2'
    await agent.run([BinaryContent(data, media_type='image/png', identifier='as-image')])
    await agent.run([BinaryContent(data, media_type='text/csv', identifier='as-text')])

    assert describer.calls == 1, 'the image was described; the CSV is inlined, not described'
    # The CSV reached the model as its own content, not as the description written for the image.
    assert last_prompt(seen) == snapshot(
        [
            """\
-----BEGIN FILE id="as-text" type="text/csv"-----
a,b
1,2
-----END FILE id="as-text"-----\
""",
        ]
    )


async def test_a_url_is_not_described_twice_within_a_run():
    """Within one run the description is reused, so a history carrying the same file costs one description."""
    describer = Describer()
    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])

    url = ImageUrl('https://example.com/cat.png')
    await agent.run([url, 'and again:', url])
    assert describer.calls == 1


async def test_a_file_is_described_every_time_without_a_history_to_record_it_in():
    """With no request in the run's history, a description is reused only within the request it was made for.

    An agent run always has its request in the history, so this calls the hook directly with a bare `RunContext`.
    """
    describer = Describer()
    capability: FileUnderstanding[None] = FileUnderstanding(fallback_subagent_model=FunctionModel(describer))
    item = ImageUrl('https://example.com/cat.png')
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    request_context = ModelRequestContext(
        model=TestModel(profile=ModelProfile(supports_image_input=False)),
        messages=[ModelRequest(parts=[UserPromptPart([item, 'and again:', item])])],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    for _ in range(2):
        await capability.before_model_request(ctx, request_context)
    assert describer.calls == 2, 'the same URL twice in one request is described once, and again in the next'
    assert ctx.usage.requests == 2


async def test_a_file_only_in_the_request_is_recorded_on_the_latest_request():
    """A file a history processor added to the request isn't in the history, so the latest request records it.

    Application data already under the capability's key that isn't a description map is replaced, not trusted.
    """
    describer = Describer()
    capability: FileUnderstanding[None] = FileUnderstanding(fallback_subagent_model=FunctionModel(describer))
    item = ImageUrl('https://example.com/cat.png')
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart('hi')],
            metadata={
                '__pydantic_ai__': {'file_understanding': {'file_understanding': 'not descriptions'}, 'other': 1},
                'app': 1,
            },
        )
    ]
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), messages=history)
    request_context = ModelRequestContext(
        model=TestModel(profile=ModelProfile(supports_image_input=False)),
        messages=[ModelRequest(parts=[UserPromptPart([item])])],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    await capability.before_model_request(ctx, request_context)
    await capability.before_model_request(ctx, request_context)
    assert describer.calls == 1
    [request] = history
    assert isinstance(request, ModelRequest)
    assert request.metadata == snapshot(
        {
            '__pydantic_ai__': {
                'file_understanding': {
                    'file_understanding': {'url:https://example.com/cat.png': 'A description of 750ff4.'}
                },
                'other': 1,
            },
            'app': 1,
        }
    )


async def test_describing_counts_toward_the_runs_usage_and_limits():
    """The describing requests are the agent run's own spend: they show in its usage and its limits bound them."""
    describer = Describer()
    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])

    result = await agent.run([_png('first'), _png('second')], usage_limits=UsageLimits(request_limit=3))
    assert describer.calls == 2
    # Two describing requests and the agent's own.
    assert result.usage.requests == 3

    # Two files would fit under the limit, but not with the agent's own request after them, so neither is sent.
    with pytest.raises(UsageLimitExceeded, match='request_limit of 2'):
        await agent.run([_png('third'), _png('fourth')], usage_limits=UsageLimits(request_limit=2))
    assert describer.calls == 2


async def test_distinct_files_are_described_concurrently():
    """Each distinct file is described once, and all of them at the same time rather than one after another."""
    started = {'first': anyio.Event(), 'second': anyio.Event()}
    described: list[str] = []

    async def describe(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        part = messages[-1].parts[-1]
        assert isinstance(part, UserPromptPart)
        [item] = part.content
        assert isinstance(item, BinaryContent)
        described.append(item.identifier)
        started[item.identifier].set()
        # Each call waits for the other one to start, which it can only do if both are in flight at once.
        other = 'second' if item.identifier == 'first' else 'first'
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await started[other].wait()
        return ModelResponse(parts=[TextPart(f'A description of {item.identifier}.')])

    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describe))])
    await agent.run([_png('first'), _png('second'), 'and the first again:', _png('first')])

    assert sorted(described) == ['first', 'second']


async def test_a_url_is_described_once_across_the_turns_of_a_conversation():
    """The description is recorded in the history, so a later turn served by another worker reuses its text."""
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    def agent(describer: Describer) -> Agent[None, str]:
        text_only = FunctionModel(capture, profile=ModelProfile(supports_document_input=False))
        return Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])

    first_worker, second_worker = Describer(), Describer()
    first = await agent(first_worker).run([DocumentUrl('https://example.com/report.pdf'), 'What is this?'])
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first.all_messages()))
    await agent(second_worker).run('And who wrote it?', message_history=history)

    assert (first_worker.calls, second_worker.calls) == (1, 0)
    sent_first, sent_second = seen[0][0], seen[1][0]
    assert isinstance(sent_first, ModelRequest) and isinstance(sent_second, ModelRequest)
    assert sent_second.parts == sent_first.parts, 'the earlier request is sent with the same description'
    # What is recorded is application data on the request, which the model never receives.
    request = first.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.metadata == snapshot(
        {
            '__pydantic_ai__': {
                'file_understanding': {
                    'file_understanding': {'url:https://example.com/report.pdf': 'A description of a5f6ba.'}
                }
            }
        }
    )


async def test_descriptions_in_untrusted_history_are_not_used():
    """History a client sent back goes through `sanitize_messages`, which drops the recorded descriptions.

    So a file is described again, here one a tool returns, rather than replaced with text from the client's history.
    """
    seen: list[list[ModelMessage]] = []

    def call_tool_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('get_chart', {})])

    describer = Describer()
    text_only = FunctionModel(call_tool_then_answer, profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])

    @agent.tool_plain
    def get_chart() -> BinaryContent:
        return _png('chart')

    first = await agent.run('Show me the chart.')
    assert describer.calls == 1
    # The client sends the history back with different text recorded for the chart.
    [recorded] = [m for m in first.all_messages() if isinstance(m, ModelRequest) and m.metadata]
    assert recorded.metadata is not None
    records = recorded.metadata['__pydantic_ai__']['file_understanding']['file_understanding']
    recorded.metadata['__pydantic_ai__']['file_understanding']['file_understanding'] = dict.fromkeys(
        records, 'Not the chart.'
    )
    client_json = ModelMessagesTypeAdapter.dump_json(first.all_messages())

    with pytest.warns(UserWarning, match='`__pydantic_ai__` metadata was dropped from 1 client-submitted message'):
        history = sanitize_messages(ModelMessagesTypeAdapter.validate_json(client_json))
    await agent.run('Show me the chart again.', message_history=history)

    assert describer.calls == 2
    sent = ModelMessagesTypeAdapter.dump_json(seen[-1]).decode()
    assert 'Not the chart.' not in sent
    assert 'A description of chart.' in sent


async def test_a_file_is_described_once_across_the_requests_of_a_run():
    """A tool call makes a second request in the same run, which reads the description from the history."""
    describer = Describer()
    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(describer))])

    @agent.tool_plain
    def look_again() -> str:
        return 'looked'

    result = await agent.run([_png('chart'), 'Look again, then answer.'])
    assert sum(isinstance(message, ModelRequest) for message in result.all_messages()) == 2
    assert describer.calls == 1


async def test_a_text_document_by_url_is_downloaded_once_per_conversation():
    """The downloaded text is recorded like a description, so a later turn doesn't download it again."""
    text_only = TestModel(profile=ModelProfile(supports_document_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_subagent_model=FunctionModel(Describer()))])
    with patch('pydantic_ai.capabilities.file_understanding.download_item', new_callable=AsyncMock) as download:
        download.return_value = {'data': 'x' * 100_001, 'data_type': 'text/plain'}
        first = await agent.run([DocumentUrl('https://example.com/big.txt'), 'Summarize.'])
        await agent.run('Again.', message_history=first.all_messages())
        assert download.call_count == 1


@pytest.mark.skipif(not imports_successful(), reason='anthropic or google not installed')
@pytest.mark.vcr()
async def test_a_real_adapter_that_cannot_read_a_video_gets_a_real_description(
    allow_model_requests: None, anthropic_api_key: str, gemini_api_key: str, video_content: BinaryContent
):
    # Claude takes no video, so without the capability this request raises in the Anthropic adapter.
    claude = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    gemini = GoogleModel('gemini-3.8-flash', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(claude, capabilities=[FileUnderstanding(fallback_subagent_model=gemini)])

    result = await agent.run(['What happens in this video? Answer in one short sentence.', video_content])

    assert result.output == snapshot(
        'This file describes a still photograph, not a video, showing a camera on a tripod in a red rock canyon, with its screen displaying a dirt road leading toward distant mountains.'
    )
    assert result.usage.requests == 2
