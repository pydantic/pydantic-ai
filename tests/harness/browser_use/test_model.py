"""Tests for PydanticAIChatModel (browser-use chat model backed by a Pydantic AI model)."""

from __future__ import annotations

import base64

import pytest
from browser_use.llm.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.browser_use import PydanticAIChatModel, resolve_chat_model


class _Facts(BaseModel):
    x: int


class _Other(BaseModel):
    y: str


_PNG = base64.b64encode(b'not-a-real-png').decode()


class TestMessageMapping:
    async def test_conversation_structure(self) -> None:
        seen: list[list[ModelMessage]] = []

        def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            return ModelResponse(parts=[TextPart('ok')])

        model = PydanticAIChatModel(FunctionModel(capture))
        result = await model.ainvoke(
            [
                SystemMessage(content='sys'),
                UserMessage(content='hello'),
                AssistantMessage(content='hi there'),
                UserMessage(content='next'),
            ]
        )

        assert result.completion == 'ok'
        [messages] = seen
        first, second, third = messages
        assert isinstance(first, ModelRequest)
        assert [type(part) for part in first.parts] == [SystemPromptPart, UserPromptPart]
        assert isinstance(second, ModelResponse)
        assert isinstance(second.parts[0], TextPart)
        assert second.parts[0].content == 'hi there'
        assert isinstance(third, ModelRequest)
        assert isinstance(third.parts[0], UserPromptPart)
        assert third.parts[0].content == 'next'

    async def test_multimodal_and_list_content(self) -> None:
        seen: list[list[ModelMessage]] = []

        def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            return ModelResponse(parts=[TextPart('ok')])

        model = PydanticAIChatModel(FunctionModel(capture))
        await model.ainvoke(
            [
                SystemMessage(content=[ContentPartTextParam(text='sys a'), ContentPartTextParam(text='sys b')]),
                AssistantMessage(content=None),
                AssistantMessage(content=[ContentPartTextParam(text='said'), ContentPartRefusalParam(refusal='no')]),
                AssistantMessage(content='and more'),
                UserMessage(
                    content=[
                        ContentPartTextParam(text='look at this'),
                        ContentPartImageParam(image_url=ImageURL(url=f'data:image/png;base64,{_PNG}')),
                        ContentPartImageParam(image_url=ImageURL(url=f'data:;base64,{_PNG}', media_type='image/jpeg')),
                        ContentPartImageParam(image_url=ImageURL(url='https://example.com/shot.png')),
                    ]
                ),
            ]
        )

        [messages] = seen
        # The empty assistant message maps to nothing: three messages remain.
        system_request, refusal_response, user_request = messages
        assert isinstance(system_request, ModelRequest)
        assert isinstance(system_request.parts[0], SystemPromptPart)
        assert system_request.parts[0].content == 'sys a\nsys b'
        assert isinstance(refusal_response, ModelResponse)
        assert isinstance(refusal_response.parts[0], TextPart)
        assert refusal_response.parts[0].content == 'said\nno'
        assert isinstance(user_request, ModelRequest)
        assert isinstance(user_request.parts[0], UserPromptPart)
        text, png, jpeg, url = user_request.parts[0].content
        assert text == 'look at this'
        assert isinstance(png, BinaryContent)
        assert png.data == b'not-a-real-png'
        assert png.media_type == 'image/png'
        assert isinstance(jpeg, BinaryContent)
        assert jpeg.media_type == 'image/jpeg'
        assert isinstance(url, ImageUrl)
        assert url.url == 'https://example.com/shot.png'

    async def test_conversation_ending_with_assistant_rejected(self) -> None:
        model = PydanticAIChatModel(TestModel())
        with pytest.raises(ValueError, match='must end with a system or user message'):
            await model.ainvoke([UserMessage(content='hi'), AssistantMessage(content='done')])


class TestPydanticAIChatModel:
    async def test_text_completion_and_usage(self) -> None:
        model = PydanticAIChatModel(TestModel())
        result = await model.ainvoke([UserMessage(content='hello')])
        assert isinstance(result.completion, str)
        usage = result.usage
        assert usage is not None
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.prompt_cached_tokens is None

    async def test_structured_completion(self) -> None:
        model = PydanticAIChatModel(TestModel())
        result = await model.ainvoke([UserMessage(content='hello')], output_format=_Facts)
        assert isinstance(result.completion, _Facts)

    async def test_output_type_varies_per_call(self) -> None:
        """The output type is per call, so alternating between shapes keeps working on one model."""
        model = PydanticAIChatModel(TestModel())

        facts = await model.ainvoke([UserMessage(content='hello')], output_format=_Facts)
        text = await model.ainvoke([UserMessage(content='hello')])
        other = await model.ainvoke([UserMessage(content='hello')], output_format=_Other)

        assert isinstance(facts.completion, _Facts)
        assert isinstance(text.completion, str)
        assert isinstance(other.completion, _Other)

    def test_identity_properties(self) -> None:
        model = PydanticAIChatModel('test')
        assert model.provider == 'test'
        assert model.name == 'test'
        assert model.model == 'test'
        assert model.model_name == 'test'


class TestResolveChatModel:
    def test_none_passes_through(self) -> None:
        assert resolve_chat_model(None) is None

    def test_browser_use_model_passes_through(self) -> None:
        model = PydanticAIChatModel('test')
        assert resolve_chat_model(model) is model

    def test_model_name_string_is_wrapped(self) -> None:
        resolved = resolve_chat_model('test')
        assert isinstance(resolved, PydanticAIChatModel)

    def test_pydantic_ai_model_is_wrapped(self) -> None:
        resolved = resolve_chat_model(TestModel())
        assert isinstance(resolved, PydanticAIChatModel)
