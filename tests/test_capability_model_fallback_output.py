from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, NativeOutput, PromptedOutput, RunContext, ToolOutput
from pydantic_ai.capabilities import AbstractCapability, Hooks, ModelFallback, SelectModel
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryFeedbackPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage, UsageLimits


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


class TestOutputFallback:
    async def test_thresholds_can_be_configured_per_model(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[TextPart('draft')])

        def middle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('middle')
            return ModelResponse(parts=[TextPart('draft')])

        first = FunctionModel(primary, model_name='primary')
        second = FunctionModel(middle, model_name='middle')
        thresholds: dict[str, int] = {first.model_id: 2, second.model_id: 1}
        fallback = ModelFallback(second, TestModel(custom_output_text='ready'), output_retries=thresholds)
        thresholds[first.model_id] = 1
        agent = Agent(first, capabilities=[fallback], retries={'output': 3})

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['primary', 'primary', 'middle']

    @pytest.mark.parametrize('thresholds', [{}, {'unused:model': 1}])
    async def test_mapping_leaves_unlisted_models_unchanged(self, thresholds: dict[str, int]) -> None:
        agent = Agent(
            TestModel(custom_output_text='draft'),
            capabilities=[ModelFallback(TestModel(custom_output_text='ready'), output_retries=thresholds)],
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        with pytest.raises(UnexpectedModelBehavior, match='output retries'):
            await agent.run('Answer.')

    async def test_suspended_continuation_precedes_output_routing(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')  # pragma: no cover
            raise AssertionError('Continue the pinned response.')  # pragma: no cover

        def pinned_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('pinned')
            assert isinstance(messages[-1], ModelResponse) and messages[-1].state == 'suspended'
            return ModelResponse(parts=[TextPart('draft')])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            assert isinstance(messages[-1], ModelRequest)
            assert any(
                isinstance(part, UserPromptPart) and 'Finish the draft.' in part.content for part in messages[-1].parts
            )
            return ModelResponse(parts=[TextPart('ready')])

        pinned = FunctionModel(pinned_response, model_name='pinned')
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Answer.')]),
            ModelResponse(
                parts=[TextPart('partial')],
                state='suspended',
                provider_response_id='job-1',
                model_id=pinned.model_id,
            ),
        ]
        agent = Agent(
            FunctionModel(primary),
            capabilities=[ModelFallback(pinned, FunctionModel(backup), output_retries=1)],
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value.endswith('draft'):
                raise ModelRetry('Finish the draft.')
            return value

        result = await agent.run(message_history=history)
        assert result.output == 'ready'
        assert any(
            isinstance(part, RetryFeedbackPart)
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        assert calls == ['pinned', 'backup']

    async def test_producer_counts_survive_request_hook_retries(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            if calls.count('primary') == 2:
                raise ModelHTTPError(503, 'primary')
            return ModelResponse(parts=[TextPart('draft')])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            return ModelResponse(parts=[TextPart('request hook' if calls.count('backup') == 1 else 'ready')])

        def after_request(
            ctx: RunContext[object], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            if response.text == 'request hook':
                raise ModelRetry('Retry the request hook.')
            return response

        agent = Agent(
            FunctionModel(primary),
            capabilities=[
                ModelFallback(FunctionModel(backup), output_retries=2),
                Hooks(after_model_request=after_request),
            ],
            retries={'output': 3},
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['primary', 'primary', 'backup', 'primary', 'backup']

    @pytest.mark.parametrize('enabled', [False, True])
    async def test_output_fallback_is_opt_in(self, enabled: bool) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[TextPart('draft')])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            return ModelResponse(parts=[TextPart('ready')])

        candidate = FunctionModel(backup)
        fallback = ModelFallback(candidate, output_retries=1) if enabled else ModelFallback(candidate)
        agent = Agent(FunctionModel(primary), capabilities=[fallback])

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        if enabled:
            assert (await agent.run('Answer.')).output == 'ready'
            assert calls == ['primary', 'backup']
        else:
            with pytest.raises(UnexpectedModelBehavior, match='output retries'):
                await agent.run('Answer.')
            assert calls == ['primary', 'primary']

    async def test_request_fallback_does_not_return_to_a_retired_candidate(self) -> None:
        calls: list[str] = []
        tools: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[TextPart('draft')])

        def middle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('middle')
            if tools:
                raise ModelHTTPError(503, 'middle')
            return ModelResponse(parts=[ToolCallPart('lookup', {})])

        def final(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('final')
            return ModelResponse(parts=[TextPart('draft' if calls.count('final') == 1 else 'ready')])

        def lookup() -> str:
            tools.append('lookup')
            return 'evidence'

        agent = Agent(
            FunctionModel(primary),
            capabilities=[ModelFallback(FunctionModel(middle), FunctionModel(final), output_retries=1)],
            tools=[lookup],
            retries={'output': 2},
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['primary', 'middle', 'middle', 'final', 'final']
        assert tools == ['lookup']

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

        capability = ModelFallback(
            FunctionModel(primary, model_name='primary'), FunctionModel(backup, model_name='backup'), output_retries=1
        )
        capabilities: list[AbstractCapability[object]] = [capability]
        if source in ('before', 'after'):
            capabilities.append(RetryRequest(after=source == 'after'))
        agent = Agent(
            capabilities=capabilities,
            output_type=Answer if source == 'output_tool' else str,
            retries={'output': 1},
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
        escalation = ModelFallback(first, second, output_retries=1)
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

        escalation = ModelFallback(FunctionModel(primary), FunctionModel(backup), output_retries=2)
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
            capabilities=[ModelFallback(model, model, output_retries=1)],
        )
        with pytest.raises(UnexpectedModelBehavior, match=f'output retries \\({tool_budget}\\)'):
            await agent.run('Answer.')
        assert len(calls) == tool_budget + 1

    @pytest.mark.parametrize('selection', ['ladder', 'last', 'run', 'override'])
    async def test_selection_precedence(self, selection: Literal['ladder', 'last', 'run', 'override']) -> None:
        first = TestModel(custom_output_text='ladder')
        other = TestModel(custom_output_text='other')
        escalation = ModelFallback(first, output_retries=1)
        caps: list[AbstractCapability[object]] = [SelectModel(lambda ctx: other), escalation]
        if selection == 'last':
            caps.reverse()
        agent = Agent(other, capabilities=caps)
        if selection == 'override':
            with agent.override(model=other):
                result = await agent.run('Answer.', model=first)
        else:
            result = await agent.run('Answer.', model=other if selection == 'run' else None)
        assert result.output == 'other'

    async def test_request_and_output_failures_share_chain(self) -> None:
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
                ModelFallback(FunctionModel(failed), FunctionModel(rejected), FunctionModel(valid), output_retries=1),
            ]
        )
        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['failed', 'rejected', 'valid']

    @pytest.mark.parametrize('threshold', [-1, 0, True])
    def test_invalid_threshold(self, threshold: int) -> None:
        with pytest.raises(UserError, match='positive'):
            ModelFallback('test', output_retries=threshold)

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
            capabilities=[ModelFallback(FunctionModel(primary), FunctionModel(backup), output_retries=1)],
        )
        with pytest.raises(UnexpectedModelBehavior, match='output retries \\(1\\)'):
            await agent.run('Answer.')
        assert calls == ['primary', 'backup', 'backup']

    async def test_failed_siblings_do_not_skip_an_untried_candidate(self) -> None:
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

        def middle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('middle')
            return ModelResponse(parts=[ToolCallPart('first', {'number': 'bad'})])

        agent = Agent(
            output_type=[ToolOutput(Answer, name='first'), ToolOutput(Answer, name='second')],
            retries={'output': 3},
            end_strategy='exhaustive',
            capabilities=[
                ModelFallback(FunctionModel(primary), FunctionModel(middle), FunctionModel(backup), output_retries=1)
            ],
        )
        assert (await agent.run('Answer.')).output == Answer(number=42)
        assert calls == ['primary', 'middle', 'final']

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
            capabilities=[ModelFallback(model, model, output_retries=1)],
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
            capabilities=[ModelFallback(first, second, output_retries=1)],
        )
        assert (await agent.run('Answer.')).output == Answer(number=42)
        assert seen == [('tool', 0.1), ('native', 0.9)]
