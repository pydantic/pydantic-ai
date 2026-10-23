"""Tests for PydanticAIChatModel (browser-use chat model backed by a Pydantic AI model)."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Literal

import pytest
from browser_use.agent.service import Agent as BrowserUseAgent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser import BrowserSession
from browser_use.llm.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability, Instrumentation
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
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.browser_use import BrowserUse, PydanticAIChatModel, resolve_chat_model
from tests.harness.conftest import agent_run_names

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire


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

    @pytest.mark.usefixtures('instrument_all_agents')
    async def test_run_is_named_after_the_capability(self, capfire: CaptureLogfire) -> None:
        model = PydanticAIChatModel(TestModel())
        await model.ainvoke([UserMessage(content='hello')])

        assert agent_run_names(capfire) == ['browser_use']

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


class TestInheritedInstrumentation:
    @pytest.fixture(autouse=True)
    def browser_model_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercise the real factory and adapter without launching a browser or calling a provider."""
        monkeypatch.setenv('ANONYMIZED_TELEMETRY', 'false')

        async def run(self: BrowserUseAgent[None, BaseModel], max_steps: int = 500) -> AgentHistoryList[BaseModel]:
            await self.llm.ainvoke(
                [
                    UserMessage(
                        content=[
                            ContentPartTextParam(text='private page content'),
                            ContentPartImageParam(image_url=ImageURL(url=f'data:image/png;base64,{_PNG}')),
                        ]
                    )
                ]
            )
            return AgentHistoryList[BaseModel](history=[])

        async def kill(self: BrowserSession) -> None:
            pass

        monkeypatch.setattr(BrowserUseAgent, 'run', run)
        monkeypatch.setattr(BrowserSession, 'kill', kill)

    @pytest.mark.parametrize('global_enabled', [False, True])
    @pytest.mark.parametrize(
        ('source', 'include_content'),
        [
            ('capability', False),
            ('capability', True),
            ('run', False),
            ('dynamic', False),
            ('model', False),
            ('disabled', False),
        ],
    )
    async def test_host_instrumentation_is_inherited(
        self,
        capfire: CaptureLogfire,
        global_enabled: bool,
        source: Literal['capability', 'run', 'dynamic', 'model', 'disabled'],
        include_content: bool,
    ) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        settings = InstrumentationSettings(
            tracer_provider=provider,
            include_content=include_content,
            include_binary_content=False,
            include_model_request_parameters=False,
            version=6,
        )
        capabilities: list[AgentCapability[None]] = [BrowserUse()]
        run_capabilities: list[AgentCapability[None]] = []
        if source == 'capability':
            capabilities.append(Instrumentation(settings=settings))
        elif source == 'dynamic':
            run_capabilities.append(lambda ctx: Instrumentation(settings=settings))
        elif source == 'run':
            capabilities.append(Instrumentation())
            run_capabilities.append(Instrumentation(settings=settings))
        model = InstrumentedModel(TestModel(), settings) if source == 'model' else TestModel()
        agent = Agent(model, deps_type=None, capabilities=capabilities)
        if source == 'disabled':
            agent.instrument = False

        Agent.instrument_all(global_enabled)
        try:
            await agent.run('Browse.', capabilities=run_capabilities)
        finally:
            Agent.instrument_all(False)
            provider.shutdown()

        # Neither the host nor its browser model may fall back to the global provider.
        assert agent_run_names(capfire) == []
        spans = exporter.get_finished_spans()
        if source == 'disabled':
            assert spans == ()
        else:
            assert any(span.name == 'invoke_agent browser_use' for span in spans)
            assert sum(span.name == 'chat test' for span in spans) == 3  # Two host turns, one browser turn.
            attributes = [dict(span.attributes or {}) for span in spans]
            assert ('private page content' in str(attributes)) is include_content
            assert _PNG not in str(attributes)
            assert all('model_request_parameters' not in span_attributes for span_attributes in attributes)

    async def test_explicit_llm_keeps_its_own_instrumentation(self, capfire: CaptureLogfire) -> None:
        browser_model = InstrumentedModel(TestModel(), InstrumentationSettings(include_content=True))
        agent = Agent(
            TestModel(),
            capabilities=[
                Instrumentation(settings=InstrumentationSettings(include_content=False)),
                BrowserUse(llm=browser_model),
            ],
        )

        await agent.run('Browse.')

        assert 'browser_use' in agent_run_names(capfire)
        assert 'private page content' in str(capfire.exporter.exported_spans_as_dict())
