from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, NativeOutput, PromptedOutput, RunContext, ToolOutput
from pydantic_ai.capabilities import AbstractCapability, ModelFallback, SelectModel
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, RetryFeedbackPart, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage, UsageLimits
from pydantic_ai_harness.escalate_on_output_retry import EscalateOnOutputRetry, EscalationLevel


class Answer(BaseModel):
    number: int


@dataclass
class RetryRequest(AbstractCapability[object]):
    after: bool

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if not self.after and ctx.retry == 0:
            raise ModelRetry('Retry the request.')
        return request_context

    async def after_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if self.after and ctx.retry == 0:
            raise ModelRetry('Retry the request.')
        return response


class TestEscalateOnOutputRetry:
    @pytest.mark.parametrize('source', ['validator', 'empty', 'output_tool', 'function_tool', 'before', 'after'])
    async def test_retry_sources(
        self, source: Literal['validator', 'empty', 'output_tool', 'function_tool', 'before', 'after']
    ) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            if source == 'output_tool':
                return ModelResponse(parts=[ToolCallPart('final_result', {'number': 'bad'})])
            if source == 'function_tool' and len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('lookup', {})])
            return ModelResponse(parts=[] if source == 'empty' else [TextPart('draft')])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            if source == 'output_tool':
                return ModelResponse(parts=[ToolCallPart('final_result', {'number': 42})])
            return ModelResponse(parts=[TextPart('ready')])

        capability = EscalateOnOutputRetry[object](
            levels=[
                EscalationLevel(FunctionModel(primary, model_name='primary'), after_failures=1),
                EscalationLevel(FunctionModel(backup, model_name='backup')),
            ]
        )
        capabilities: list[AbstractCapability[object]] = [capability]
        if source in ('before', 'after'):
            capabilities.append(RetryRequest(after=source == 'after'))
        agent = Agent(
            capabilities=capabilities,
            output_type=Answer if source == 'output_tool' else str,
            retries={'output': capability.required_output_retries},
        )
        if source == 'validator':

            @agent.output_validator
            def validate(value: Answer | str) -> Answer | str:
                if value == 'draft':
                    raise ModelRetry('Return a finished answer.')
                return value

        if source == 'function_tool':

            @agent.tool_plain
            def lookup() -> str:
                raise ModelRetry('Retry lookup.')

        await agent.run('Answer.')
        assert calls == (
            ['primary', 'backup']
            if source in ('validator', 'empty', 'output_tool')
            else ['primary']
            if source == 'before'
            else ['primary', 'primary']
        )

    @pytest.mark.parametrize('mode', ['native', 'prompted', 'tool'])
    async def test_schema_validation(self, mode: Literal['native', 'prompted', 'tool']) -> None:
        calls: list[str] = []

        def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('call')
            args = {'number': 'invalid' if len(calls) == 1 else 42}
            return ModelResponse(
                parts=[ToolCallPart('final_result', args) if mode == 'tool' else TextPart(json.dumps(args))]
            )

        profile = ModelProfile(supports_json_schema_output=True)
        first = FunctionModel(response, model_name='first', profile=profile)
        second = FunctionModel(response, model_name='second', profile=profile)
        escalation = EscalateOnOutputRetry(levels=[EscalationLevel(first, after_failures=1), EscalationLevel(second)])
        output = NativeOutput(Answer) if mode == 'native' else PromptedOutput(Answer) if mode == 'prompted' else Answer
        result = await Agent(output_type=output, capabilities=[escalation]).run('Answer.')
        assert result.output == Answer(number=42)
        assert [m.model_name for m in result.all_messages() if isinstance(m, ModelResponse)] == ['first', 'second']

    async def test_thresholds_history_tools_and_usage(self) -> None:
        calls: list[str] = []
        tool_calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[TextPart('draft')], usage=RequestUsage(input_tokens=10, output_tokens=2))

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            if not tool_calls:
                return ModelResponse(parts=[ToolCallPart('lookup', {})])
            return ModelResponse(parts=[TextPart('ready')])

        def lookup() -> str:
            tool_calls.append('lookup')
            return 'evidence'

        escalation = EscalateOnOutputRetry(
            levels=[EscalationLevel(FunctionModel(primary), after_failures=2), EscalationLevel(FunctionModel(backup))]
        )
        agent = Agent(capabilities=[escalation], retries={'output': 3}, tools=[lookup])

        @agent.output_validator
        def validate(ctx: RunContext[object], value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            assert ctx.retry == 2
            return value

        result = await agent.run('Answer.')
        assert calls == ['primary', 'primary', 'backup', 'backup']
        assert tool_calls == ['lookup']
        assert result.usage.input_tokens >= 20
        feedback = [
            p
            for m in result.all_messages()
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, RetryFeedbackPart)
        ]
        assert [p.retry.attempt for p in feedback] == [1, 2]
        assert all(p.content == 'Finish the draft.' for p in feedback)
        with pytest.raises(UsageLimitExceeded, match='request_limit'):
            await agent.run('Answer.', usage_limits=UsageLimits(request_limit=2))

    @pytest.mark.parametrize('tool_budget', [0, 1, 3])
    async def test_output_tool_budget(self, tool_budget: int) -> None:
        calls: list[str] = []

        def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('request')
            return ModelResponse(parts=[ToolCallPart('final_result', {'number': 'bad'})])

        model = FunctionModel(invalid)
        agent = Agent(
            output_type=ToolOutput(Answer, max_retries=tool_budget),
            retries={'output': 1},
            capabilities=[
                EscalateOnOutputRetry(levels=[EscalationLevel(model, after_failures=1), EscalationLevel(model)])
            ],
        )
        with pytest.raises(UnexpectedModelBehavior, match=f'output retries \\({tool_budget}\\)'):
            await agent.run('Answer.')
        assert len(calls) == tool_budget + 1

    @pytest.mark.parametrize('selection', ['ladder', 'last', 'run', 'override'])
    async def test_selection_precedence(self, selection: Literal['ladder', 'last', 'run', 'override']) -> None:
        first = TestModel(custom_output_text='ladder')
        other = TestModel(custom_output_text='other')
        escalation = EscalateOnOutputRetry(levels=[EscalationLevel(first)])
        caps: list[AbstractCapability[object]] = [SelectModel(lambda ctx: other), escalation]
        if selection == 'last':
            caps.reverse()
        agent = Agent(other, capabilities=caps)
        if selection == 'override':
            with agent.override(model=other):
                result = await agent.run('Answer.', model=first)
        else:
            result = await agent.run('Answer.', model=other if selection == 'run' else None)
        assert result.output == ('ladder' if selection == 'ladder' else 'other')

    async def test_model_fallback_is_separate(self) -> None:
        calls: list[str] = []

        def failed(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('failed')
            raise ModelHTTPError(503, 'failed')

        def rejected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('rejected')
            return ModelResponse(parts=[])

        def valid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('valid')
            return ModelResponse(parts=[TextPart('ready')])

        agent = Agent(
            capabilities=[
                EscalateOnOutputRetry(
                    levels=[
                        EscalationLevel(FunctionModel(failed), after_failures=1),
                        EscalationLevel(FunctionModel(valid)),
                    ]
                ),
                ModelFallback(FunctionModel(rejected)),
            ]
        )
        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['failed', 'rejected', 'valid']

    @pytest.mark.parametrize(
        'levels', [[], [EscalationLevel('test'), EscalationLevel('test')], [EscalationLevel('test', after_failures=1)]]
    )
    def test_invalid_ladders(self, levels: list[EscalationLevel]) -> None:
        with pytest.raises(UserError):
            EscalateOnOutputRetry(levels=levels)

    def test_invalid_threshold(self) -> None:
        with pytest.raises(UserError, match='positive'):
            EscalationLevel('test', after_failures=0)

    async def test_tool_switch_does_not_reset_counters(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[ToolCallPart('first', {'number': 'bad'})])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            return ModelResponse(parts=[ToolCallPart('second' if len(calls) == 2 else 'first', {'number': 'bad'})])

        agent = Agent(
            output_type=[
                ToolOutput(Answer, name='first', max_retries=1),
                ToolOutput(Answer, name='second', max_retries=3),
            ],
            retries={'output': 1},
            capabilities=[
                EscalateOnOutputRetry(
                    levels=[
                        EscalationLevel(FunctionModel(primary), after_failures=1),
                        EscalationLevel(FunctionModel(backup)),
                    ]
                )
            ],
        )
        with pytest.raises(UnexpectedModelBehavior, match='output retries \\(1\\)'):
            await agent.run('Answer.')
        assert calls == ['primary', 'backup', 'backup']

    async def test_batch_can_cross_multiple_levels(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(
                parts=[
                    ToolCallPart('first', {'number': 'bad'}, 'one'),
                    ToolCallPart('second', {'number': 'bad'}, 'two'),
                ]
            )

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('final')
            return ModelResponse(parts=[ToolCallPart('first', {'number': 42})])

        agent = Agent(
            output_type=[ToolOutput(Answer, name='first'), ToolOutput(Answer, name='second')],
            retries={'output': 3},
            end_strategy='exhaustive',
            capabilities=[
                EscalateOnOutputRetry(
                    levels=[
                        EscalationLevel(FunctionModel(primary), after_failures=1),
                        EscalationLevel(TestModel(custom_output_args={'number': -1}), after_failures=1),
                        EscalationLevel(FunctionModel(backup)),
                    ]
                )
            ],
        )
        assert (await agent.run('Answer.')).output == Answer(number=42)
        assert calls == ['primary', 'final']

    @pytest.mark.parametrize('mode', ['native', 'prompted', 'tool'])
    async def test_output_function_retry(self, mode: Literal['native', 'prompted', 'tool']) -> None:
        calls: list[str] = []

        def output(number: int) -> int:
            if number == 0:
                raise ModelRetry('Return a positive number.')
            return number

        def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('call')
            args = {'number': 0 if len(calls) == 1 else 42}
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, args) if mode == 'tool' else TextPart(json.dumps(args))]
            )

        model = FunctionModel(response, profile=ModelProfile(supports_json_schema_output=True))
        agent = Agent(
            output_type=NativeOutput(output)
            if mode == 'native'
            else PromptedOutput(output)
            if mode == 'prompted'
            else ToolOutput(output),
            capabilities=[
                EscalateOnOutputRetry(levels=[EscalationLevel(model, after_failures=1), EscalationLevel(model)])
            ],
        )
        assert (await agent.run('Answer.')).output == 42
        assert calls == ['call', 'call']

    async def test_selected_model_prepares_its_own_output_mode_and_settings(self) -> None:
        seen: list[tuple[str, float | None]] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append((info.model_request_parameters.output_mode, (info.model_settings or {}).get('temperature')))
            return ModelResponse(parts=[ToolCallPart('final_result', {'number': 'bad'})])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append((info.model_request_parameters.output_mode, (info.model_settings or {}).get('temperature')))
            return ModelResponse(parts=[TextPart('{"number": 42}')])

        first = FunctionModel(primary, settings={'temperature': 0.1})
        second = FunctionModel(
            backup,
            settings={'temperature': 0.9},
            profile=ModelProfile(supports_json_schema_output=True, default_structured_output_mode='native'),
        )
        agent = Agent(
            output_type=Answer,
            capabilities=[
                EscalateOnOutputRetry(levels=[EscalationLevel(first, after_failures=1), EscalationLevel(second)])
            ],
        )
        assert (await agent.run('Answer.')).output == Answer(number=42)
        assert seen == [('tool', 0.1), ('native', 0.9)]
