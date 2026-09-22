from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel
from pydantic_ai import AdvisorTool, Agent, NativeOutput, ToolOutput
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputSpec
from pydantic_ai.profiles import ModelProfile

from pydantic_ai_harness.advisor import Advisor

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class Decision(BaseModel):
    proceed: bool
    risk: Literal['low', 'high']


class TestAdvisorOutput:
    @pytest.mark.parametrize('max_uses', [None, 1])
    @pytest.mark.parametrize('output_type', [Decision, ToolOutput(Decision), NativeOutput(Decision)])
    async def test_returns_validated_output(self, output_type: OutputSpec[object], max_uses: int | None) -> None:
        def executor(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            description = info.function_tools[0].description
            assert description is not None and 'validated answer' in description
            returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
            if not returns:
                return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'Is this plan safe?'})])
            assert returns[0].content == Decision(proceed=False, risk='high')
            return ModelResponse(parts=[TextPart('Stop.')])

        def answer(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                assert not info.model_request_parameters.allow_text_output
                return ModelResponse(
                    parts=[ToolCallPart(info.output_tools[0].name, {'proceed': False, 'risk': 'high'})]
                )
            assert info.model_request_parameters.output_mode == 'native'
            return ModelResponse(parts=[TextPart('{"proceed": false, "risk": "high"}')])

        advisor_model = FunctionModel(answer, profile=ModelProfile(supports_json_schema_output=True))
        agent = Agent(
            FunctionModel(executor),
            capabilities=[Advisor(advisor_model, output_type=output_type, max_uses=max_uses)],
        )

        result = await agent.run('Review the plan.')

        assert result.output == 'Stop.'
        assert result.usage.requests == 3

    async def test_structured_instructions_do_not_request_prose(self) -> None:
        def advisor_model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.instructions is not None
            assert 'configured output format' in info.instructions
            assert 'concise, actionable advice' not in info.instructions
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': True})])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
            if not returns:
                return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'Proceed?'})])
            assert returns[0].content is True
            return ModelResponse(parts=[TextPart('Proceed.')])

        await Agent(
            FunctionModel(executor),
            capabilities=[Advisor(FunctionModel(advisor_model), output_type=bool, mode='local')],
        ).run('Review the plan.')

    async def test_custom_output_bypasses_compatible_native_provider(self) -> None:
        class AnthropicTestModel(TestModel):
            @property
            def system(self) -> str:
                return 'anthropic'

        executor = AnthropicTestModel(
            call_tools=[],
            profile=ModelProfile(supported_native_tools=frozenset({AdvisorTool})),
        )
        await Agent(
            executor,
            capabilities=[Advisor('anthropic:claude-opus-4-8', output_type=Decision)],
        ).run('Hello.')

        parameters = executor.last_model_request_parameters
        assert parameters is not None
        assert parameters.native_tools == []
        assert [tool.name for tool in parameters.function_tools] == ['advisor']

    def test_native_mode_rejects_custom_output(self) -> None:
        with pytest.raises(ValueError, match=r"Advisor.output_type is not supported in mode='native'"):
            Advisor('anthropic:claude-opus-4-8', mode='native', output_type=Decision)
