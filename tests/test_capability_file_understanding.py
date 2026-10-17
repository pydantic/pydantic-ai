from __future__ import annotations

from unittest.mock import AsyncMock, patch

import anyio
import pytest

from pydantic_ai import (
    Agent,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.capabilities import FileUnderstanding, file_understanding
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

from ._inline_snapshot import snapshot

pytestmark = pytest.mark.anyio

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
        assert isinstance(item, ImageUrl | DocumentUrl | VideoUrl | BinaryContent)
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
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])
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
    # Images and documents are accepted by default, video is not.
    agent = Agent(FunctionModel(capture), capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(['Two', 'texts'])]),
        ModelResponse(parts=[TextPart('ok')]),
    ]
    await agent.run([image, document, VideoUrl('https://example.com/cat.mp4')], message_history=history)

    prompt = last_prompt(seen)
    assert prompt[0] is image
    assert prompt[1] is document
    assert isinstance(prompt[2], str) and 'type="text/plain"' in prompt[2]
    assert describer.calls == 1


async def test_video_is_sent_when_the_profile_accepts_it():
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    describer = Describer()
    video = VideoUrl('https://example.com/cat.mp4')
    model = FunctionModel(capture, profile=ModelProfile(supports_video_input=True))
    agent = Agent(model, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])
    await agent.run([video])

    assert last_prompt(seen) == [video]
    assert describer.calls == 0


async def test_custom_instructions_reach_the_describer():
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('A cat.')])

    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(
        text_only,
        capabilities=[FileUnderstanding(fallback_model=FunctionModel(capture), instructions='One sentence.')],
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
    agent = Agent(model, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])
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
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])
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


async def test_descriptions_are_bounded(monkeypatch: pytest.MonkeyPatch):
    """The oldest description goes when the cache is full, so a long-lived agent does not grow without limit."""
    monkeypatch.setattr(file_understanding, '_MAX_DESCRIPTIONS', 2)
    describer = Describer()
    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])

    for name in ['first', 'second', 'third']:
        await agent.run([_png(name)])
    assert describer.calls == 3

    await agent.run([_png('third')])
    assert describer.calls == 3, 'the newest is still cached'
    await agent.run([_png('first')])
    assert describer.calls == 4, 'the oldest was evicted'


async def test_a_url_description_is_not_reused_by_another_run():
    """A URL is not its content, so one run's description of it is never handed to the next.

    The same URL can serve different bytes to different callers and stop resolving once a signature expires,
    so reusing a description across runs would hand one caller a description of what another caller fetched.
    """
    describer = Describer()
    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])

    # `media_type` is explicit because inferring it from a URL with a query string works on 3.11+ but not
    # on 3.10; what this test is about is the signature, not the inference.
    signed = ImageUrl('https://example.com/receipt.png?signature=abc', media_type='image/png')
    await agent.run([signed])
    await agent.run([signed])
    assert describer.calls == 2

    # Bytes are their own identity, so those are reused across runs.
    await agent.run([_png('chart')])
    await agent.run([_png('chart')])
    assert describer.calls == 3


async def test_the_same_bytes_under_two_media_types_are_two_files():
    """The media type decides whether bytes are described or decoded, so it is part of their identity."""
    seen: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    text_only = FunctionModel(capture, profile=ModelProfile(supports_image_input=False, supports_document_input=False))
    describer = Describer()
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])

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
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])

    url = ImageUrl('https://example.com/cat.png')
    await agent.run([url, 'and again:', url])
    assert describer.calls == 1


async def test_a_url_is_described_every_time_without_a_run_id():
    """With no run to scope it to, a URL's description is not kept across requests, only shared within one."""
    describer = Describer()
    capability: FileUnderstanding[None] = FileUnderstanding(fallback_model=FunctionModel(describer))
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


async def test_describing_counts_toward_the_runs_usage_and_limits():
    """The describing requests are the agent run's own spend: they show in its usage and its limits bound them."""
    describer = Describer()
    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])

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
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describe))])
    await agent.run([_png('first'), _png('second'), 'and the first again:', _png('first')])

    assert sorted(described) == ['first', 'second']
