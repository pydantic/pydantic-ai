from __future__ import annotations

from unittest.mock import AsyncMock, patch

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
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile

from ._inline_snapshot import snapshot

pytestmark = pytest.mark.anyio


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
        ]
    )

    assert last_prompt(seen) == snapshot(
        [
            'Compare these:',
            """\
-----BEGIN FILE id="a5f6ba" type="application/pdf"-----
A description of a5f6ba.
-----END FILE id="a5f6ba"-----\
""",
            """\
-----BEGIN FILE id="chart" type="image/png"-----
A description of chart.
-----END FILE id="chart"-----""",
        ]
    )
    assert describer.calls == 2

    # The same file is not described twice, and a run without files asks nothing.
    await agent.run([DocumentUrl('https://example.com/report.pdf')])
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
    assert isinstance(prompt[2], str) and 'type="video/mp4"' in prompt[2]
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


async def test_descriptions_are_bounded(monkeypatch: pytest.MonkeyPatch):
    """The oldest description goes when the cache is full, so a long-lived agent does not grow without limit."""
    monkeypatch.setattr(file_understanding, '_MAX_DESCRIPTIONS', 2)
    describer = Describer()
    text_only = TestModel(profile=ModelProfile(supports_image_input=False))
    agent = Agent(text_only, capabilities=[FileUnderstanding(fallback_model=FunctionModel(describer))])

    for name in ['first', 'second', 'third']:
        await agent.run([ImageUrl(f'https://example.com/{name}.png')])
    assert describer.calls == 3

    await agent.run([ImageUrl('https://example.com/third.png')])
    assert describer.calls == 3, 'the newest is still cached'
    await agent.run([ImageUrl('https://example.com/first.png')])
    assert describer.calls == 4, 'the oldest was evicted'
