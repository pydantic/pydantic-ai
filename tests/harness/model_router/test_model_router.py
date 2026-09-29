from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, _display
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.tools import DeferredToolResults
from pydantic_ai.usage import RequestUsage, UsageLimits
from pydantic_ai_harness.compaction import SummarizingCompaction
from pydantic_ai_harness.guardrails import GuardrailResult, InputGuardrail
from pydantic_ai_harness.model_router import ModelChoice, ModelRouter
from tests.conftest import IsDatetime, IsInstance, IsStr
from tests.harness._recording_durability import RecordingDurability
from tests.harness.conftest import agent_run_names

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.anyio


def _router_model(
    choice: str,
    *,
    provider_details: Mapping[str, Any] | None = None,
    inspect: Callable[[list[ModelMessage], AgentInfo], None] | None = None,
) -> FunctionModel:
    def route(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if inspect is not None:
            inspect(messages, info)
        output_tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(output_tool.name, {'choice': choice})],
            provider_details=dict(provider_details) if provider_details is not None else None,
        )

    return FunctionModel(route)


def _answer_model(answer: str) -> FunctionModel:
    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(answer)])

    return FunctionModel(respond)


def _router(
    router_model: FunctionModel,
    *,
    mode: Literal['once', 'per_step'] = 'once',
    probability_threshold: float | None = None,
    fast_model: FunctionModel | None = None,
    capable_model: FunctionModel | None = None,
) -> ModelRouter[object]:
    return ModelRouter(
        choices={
            'fast': ModelChoice(fast_model or _answer_model('fast'), 'Use for lookups.'),
            'capable': ModelChoice(capable_model or _answer_model('capable'), 'Use for difficult work.'),
        },
        router_model=router_model,
        default='capable',
        mode=mode,
        probability_threshold=probability_threshold,
    )


class TestModelRouter:
    @pytest.fixture
    def anyio_backend(self) -> str:
        return 'asyncio'

    async def test_router_receives_the_run_history_ending_with_the_routed_request(self) -> None:
        received: list[list[ModelMessage]] = []

        def inspect(messages: list[ModelMessage], _info: AgentInfo) -> None:
            received.append(messages)

        def main(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='call-1')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            capabilities=[
                _router(_router_model('fast', inspect=inspect), mode='per_step', fast_model=FunctionModel(main))
            ],
            instructions='Parent instructions.',
        )

        @agent.tool_plain
        def lookup() -> str:
            return 'lookup result'

        await agent.run('Find the capital of France')

        assert received == snapshot(
            [
                [
                    ModelRequest(
                        parts=[UserPromptPart(content='Find the capital of France', timestamp=IsDatetime())],
                        timestamp=IsDatetime(),
                        instructions='Which model should take the next step of this conversation?',
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    )
                ],
                [
                    ModelRequest(
                        parts=[UserPromptPart(content='Find the capital of France', timestamp=IsDatetime())],
                        timestamp=IsDatetime(),
                        instructions='Parent instructions.',
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    ),
                    ModelResponse(
                        parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id='call-1')],
                        usage=IsInstance(RequestUsage),
                        model_name=IsStr(),
                        timestamp=IsDatetime(),
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    ),
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name='lookup',
                                content='lookup result',
                                tool_call_id='call-1',
                                timestamp=IsDatetime(),
                            )
                        ],
                        timestamp=IsDatetime(),
                        instructions='Which model should take the next step of this conversation?',
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    ),
                ],
            ]
        )

    async def test_each_choice_description_is_in_the_output_schema(self) -> None:
        seen: list[AgentInfo] = []

        agent = Agent(capabilities=[_router(_router_model('fast', inspect=lambda _messages, info: seen.append(info)))])

        result = await agent.run('Find the capital of France')

        assert result.output == 'fast'
        assert result.usage.requests == 2
        assert seen[0].output_tools[0].parameters_json_schema == snapshot(
            {
                'properties': {
                    'choice': {
                        'anyOf': [
                            {'const': 'fast', 'description': 'Use for lookups.', 'type': 'string'},
                            {
                                'const': 'capable',
                                'description': 'Use for difficult work.',
                                'type': 'string',
                            },
                        ]
                    }
                },
                'required': ['choice'],
                'title': 'ModelRoute',
                'type': 'object',
            }
        )

    async def test_a_single_choice_still_routes(self) -> None:
        router = ModelRouter[object](
            choices={'only': ModelChoice(_answer_model('only'), 'Use for everything.')},
            router_model=_router_model('only'),
            default='only',
        )

        result = await Agent(capabilities=[router]).run('Choose')

        assert result.output == 'only'

    async def test_combining_routers_rebuilds_the_output_schema_from_the_merged_menu(self) -> None:
        first = ModelRouter[object](
            choices={'fast': ModelChoice(_answer_model('fast'), 'Use for lookups.')},
            router_model=_router_model('fast'),
            default='fast',
        )
        second = ModelRouter[object](
            choices={'capable': ModelChoice(_answer_model('capable'), 'Use for hard work.')},
            router_model=_router_model('fast'),
            default='capable',
        )

        result = await Agent(capabilities=[first, second]).run('Choose')

        assert result.output == 'fast'

    async def test_resuming_with_pending_tool_calls_routes_on_their_results(self) -> None:
        received: list[list[ModelMessage]] = []

        def inspect(messages: list[ModelMessage], _info: AgentInfo) -> None:
            received.append(messages)

        agent = Agent(capabilities=[_router(_router_model('fast', inspect=inspect))])

        @agent.tool_plain(requires_approval=True)
        def delete_file() -> str:
            return 'deleted'

        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Delete the file')]),
            ModelResponse(parts=[ToolCallPart('delete_file', {}, tool_call_id='call-1')]),
        ]
        result = await agent.run(
            message_history=history, deferred_tool_results=DeferredToolResults(approvals={'call-1': True})
        )

        assert result.output == 'fast'
        assert len(received) == 1, 'the bootstrap selection, which sees the pending response, does not route'
        last = received[0][-1]
        assert isinstance(last, ModelRequest)
        assert [type(part) for part in last.parts] == [ToolReturnPart]

    async def test_reserves_a_request_for_the_pending_parent_call(self) -> None:
        router = _router(_router_model('fast'))

        result = await Agent(capabilities=[router]).run('Choose', usage_limits=UsageLimits(request_limit=1))

        assert result.output == 'capable'
        assert result.usage.requests == 1

    async def test_once_reuses_choice_and_per_step_routes_again(self) -> None:
        async def run(mode: Literal['once', 'per_step']) -> tuple[str, int, int, int]:
            router_calls = 0
            fast_calls = 0
            capable_calls = 0

            def route(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                nonlocal router_calls
                router_calls += 1
                choice = 'fast' if router_calls == 1 else 'capable'
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'choice': choice})])

            def fast(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
                nonlocal fast_calls
                fast_calls += 1
                if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                    return ModelResponse(parts=[ToolCallPart('lookup', {})])
                return ModelResponse(parts=[TextPart('fast finished')])

            def capable(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
                nonlocal capable_calls
                capable_calls += 1
                return ModelResponse(parts=[TextPart('capable finished')])

            agent = Agent(
                capabilities=[
                    _router(
                        FunctionModel(route),
                        mode=mode,
                        fast_model=FunctionModel(fast),
                        capable_model=FunctionModel(capable),
                    )
                ]
            )

            @agent.tool_plain
            def lookup() -> str:
                return 'new information'

            result = await agent.run('Research this')
            return result.output, router_calls, fast_calls, capable_calls

        assert await run('once') == ('fast finished', 1, 2, 0)
        assert await run('per_step') == ('capable finished', 2, 1, 1)

    async def test_low_probability_uses_default(self) -> None:
        router = _router(
            _router_model('fast', provider_details={'probabilities': {'choice': {'fast': 0.49, 'capable': 0.51}}}),
            probability_threshold=0.5,
        )

        result = await Agent(capabilities=[router]).run('Choose')

        assert result.output == 'capable'

    @pytest.mark.parametrize(
        ('provider_details', 'expected'),
        [
            pytest.param(None, 'fast', id='missing'),
            pytest.param({'probabilities': 0.4}, 'fast', id='not-a-mapping'),
            pytest.param({'probabilities': {'other': {'fast': 0.4}}}, 'fast', id='other-field'),
            pytest.param({'probabilities': {'choice': {'capable': 0.4}}}, 'fast', id='pick-missing'),
            pytest.param({'probabilities': {'choice': {'fast': 'low'}}}, 'fast', id='non-numeric'),
            pytest.param({'probabilities': {'choice': {'fast': False}}}, 'fast', id='boolean'),
            pytest.param({'probabilities': {'choice': {'fast': 1}}}, 'fast', id='integer-above-threshold'),
            pytest.param({'probabilities': {'choice': {'fast': 0}}}, 'capable', id='integer-below-threshold'),
            pytest.param({'probabilities': {'choice': {'fast': 0.8}}}, 'fast', id='above-threshold'),
            pytest.param({'probabilities': {'choice': {'fast': 0.4}}}, 'capable', id='below-threshold'),
            pytest.param(
                {'confidence': {'choice': 0.9}, 'probabilities': {'choice': {'fast': 0.4}}},
                'capable',
                id='confidence-is-not-the-probability',
            ),
        ],
    )
    async def test_probability_shapes(self, provider_details: Mapping[str, Any] | None, expected: str) -> None:
        router = _router(
            _router_model('fast', provider_details=provider_details),
            probability_threshold=0.5,
        )

        result = await Agent(capabilities=[router]).run('Choose')

        assert result.output == expected

    async def test_router_error_uses_default(self) -> None:
        def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise RuntimeError('router unavailable')

        result = await Agent(capabilities=[_router(FunctionModel(fail))]).run('Choose')

        assert result.output == 'capable'

    async def test_concurrent_runs_have_independent_once_cache(self) -> None:
        calls = 0

        async def route(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'choice': 'fast'})])

        agent = Agent(capabilities=[_router(FunctionModel(route))])

        first, second = await asyncio.gather(agent.run('first'), agent.run('second'))

        assert (first.output, second.output, calls) == ('fast', 'fast', 2)

    async def test_span_records_pick_probability_and_fallback(
        self, capfire: CaptureLogfire, instrument_all_agents: None
    ) -> None:
        router = _router(
            _router_model('fast', provider_details={'probabilities': {'choice': {'fast': 0.4, 'capable': 0.6}}}),
            probability_threshold=0.5,
        )
        agent = Agent(capabilities=[router])

        await agent.run('Choose')

        spans = [span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'model_router.select']
        assert len(spans) == 1
        attributes = spans[0]['attributes']
        assert attributes['model_router.choice'] == 'capable'
        assert attributes['model_router.probability'] == 0.4
        assert attributes['model_router.fallback_reason'] == 'low_probability'
        assert attributes['model_router.mode'] == 'once'
        assert attributes['model_router.run_step'] == 1
        assert agent_run_names(capfire).count('model_router') == 1

    async def test_error_span_records_fallback(self, capfire: CaptureLogfire) -> None:
        def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise RuntimeError('router unavailable')

        agent = Agent(capabilities=[_router(FunctionModel(fail))])
        agent.instrument = InstrumentationSettings()

        await agent.run('Choose')

        span = next(span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'model_router.select')
        assert span['attributes']['model_router.choice'] == 'capable'
        assert span['attributes']['model_router.fallback_reason'] == 'error'
        assert span['attributes']['model_router.error.type'] == 'RuntimeError'

    async def test_a_user_error_from_the_router_request_propagates(self, capfire: CaptureLogfire) -> None:
        def reject(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise UserError('Files are not supported by this model.')

        agent = Agent(name='parent', capabilities=[_router(FunctionModel(reject)), Instrumentation()])

        with pytest.raises(UserError, match='Files are not supported'):
            await agent.run('Choose')

        span = next(span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'model_router.select')
        assert span['attributes']['model_router.error.type'] == 'UserError'

    async def test_routing_is_a_durable_operation(self) -> None:
        capable = _answer_model('capable')
        fast = _answer_model('fast')
        router = ModelRouter[object](
            choices={
                'fast': ModelChoice(fast, 'Use for lookups.'),
                'capable': ModelChoice(capable, 'Use for hard work.'),
            },
            router_model=_router_model('fast'),
            default='capable',
        )
        assert router.id == 'model_router'
        durability = RecordingDurability(models={'fast': fast})
        agent = Agent(capable, name='parent', capabilities=[router, durability])

        result = await agent.run('Choose')

        assert result.output == 'fast'
        assert result.usage.requests == 2, 'the router request crossed the boundary into the run usage'
        bound = RecordingDurability.from_agent(agent)
        assert bound is not None
        assert [name for name, _ in bound.calls] == [
            'parent__capability__model_router.route',
            'parent__model.request.fast',
        ]

    async def test_a_failed_durable_operation_uses_the_default(self, capfire: CaptureLogfire) -> None:
        capable = _answer_model('capable')
        router = ModelRouter[object](
            choices={
                'fast': ModelChoice(_answer_model('fast'), 'Use for lookups.'),
                'capable': ModelChoice(capable, 'Use for hard work.'),
            },
            router_model=_router_model('fast'),
            default='capable',
        )
        durability = RecordingDurability(fail_operations=frozenset({'parent__capability__model_router.route'}))
        agent = Agent(capable, name='parent', capabilities=[router, durability, Instrumentation()])

        result = await agent.run('Choose')

        assert result.output == 'capable'
        span = next(span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'model_router.select')
        assert span['attributes']['model_router.fallback_reason'] == 'error'
        assert span['attributes']['model_router.error.type'] == 'RuntimeError'

    async def test_the_router_run_follows_the_parent_instrumentation(self, capfire: CaptureLogfire) -> None:
        instrumented = Agent(name='parent', capabilities=[_router(_router_model('fast')), Instrumentation()])
        await instrumented.run('Choose')
        assert agent_run_names(capfire) == ['model_router', 'parent']

        capfire.exporter.clear()
        await Agent(name='parent', capabilities=[_router(_router_model('fast'))]).run('Choose')
        assert capfire.exporter.exported_spans_as_dict() == []

    async def test_the_router_run_does_not_show_the_first_run_banner(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_display, '_banner_displayed', False)
        for name in ('PYTEST_VERSION', 'CI', 'PYDANTIC_AI_NO_BANNER'):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv('AI_AGENT', 'test')

        await Agent(name='parent', capabilities=[_router(_router_model('fast'))]).run('Choose')

        banner = capsys.readouterr().err
        assert 'agent: parent' in banner
        assert 'model_router' not in banner

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            pytest.param({'choices': {}}, 'choices must not be empty', id='empty-choices'),
            pytest.param(
                {'choices': {'': ModelChoice('test', 'description')}},
                'choice names must not be empty',
                id='empty-name',
            ),
            pytest.param(
                {'choices': {'fast': ModelChoice('test', '')}},
                'choice descriptions must not be empty',
                id='empty-description',
            ),
            pytest.param({'default': 'missing'}, 'default must name a configured choice', id='unknown-default'),
            pytest.param({'mode': 'sometimes'}, 'mode must be', id='mode'),
            pytest.param({'probability_threshold': -0.1}, 'must be between 0 and 1', id='threshold-low'),
            pytest.param({'probability_threshold': 1.1}, 'must be between 0 and 1', id='threshold-high'),
        ],
    )
    async def test_validates_configuration(self, kwargs: dict[str, Any], message: str) -> None:
        options: dict[str, Any] = {
            'choices': {'fast': ModelChoice('test', 'description')},
            'router_model': _router_model('fast'),
            'default': 'fast',
            **kwargs,
        }

        with pytest.raises(UserError, match=message):
            ModelRouter(**options)

    async def test_an_unresolvable_router_model_is_rejected_at_construction(self) -> None:
        with pytest.raises(UserError, match='Unknown model'):
            ModelRouter(
                choices={'fast': ModelChoice('test', 'description')},
                router_model='nonexistent:model',
                default='fast',
            )

    async def test_not_agent_spec_serializable(self) -> None:
        assert ModelRouter.get_serialization_name() is None


class TestCompositionConstraints:
    """Pin what the README promises about routing's place in the step, which core's order fixes."""

    @pytest.fixture
    def anyio_backend(self) -> str:
        return 'asyncio'

    async def test_per_step_routing_reads_the_history_one_step_behind_compaction(self) -> None:
        """Compaction bounds the routing input, but the compacting step routes on the old history."""
        routing_inputs: list[list[ModelMessage]] = []
        steps = 6

        def inspect(messages: list[ModelMessage], _info: AgentInfo) -> None:
            routing_inputs.append(messages)

        step = 0

        def main(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal step
            step += 1
            if step < steps:
                return ModelResponse(parts=[ToolCallPart('work', {'n': step})])
            return ModelResponse(parts=[TextPart('done')])

        main_model = FunctionModel(main)
        router = ModelRouter[None](
            choices={'fast': ModelChoice(main_model, 'Use for everything.')},
            router_model=_router_model('fast', inspect=inspect),
            default='fast',
            mode='per_step',
        )
        compaction = SummarizingCompaction[None](
            model=_answer_model('SUMMARY'), max_messages=4, keep_messages=2, preserve_first_user_message=False
        )
        agent = Agent[None, str](main_model, deps_type=type(None), capabilities=[router, compaction])

        @agent.tool_plain
        def work(n: int) -> str:
            return f'PAYLOAD-{n} ' + 'z' * 200

        result = await agent.run('start the work')
        assert result.output == 'done'
        assert len(routing_inputs) == steps

        summarized = ['SUMMARY' in repr(routing_input) for routing_input in routing_inputs]
        assert summarized == [False, False, False, True, True, True], (
            'the step that compacts routes on the history compaction is about to replace'
        )
        # Past that step the router reads the compacted history, so its input stops growing.
        assert len(set(len(routing_input) for routing_input in routing_inputs[3:])) == 1
        assert max(len(routing_input) for routing_input in routing_inputs) == len(routing_inputs[2])

    async def test_an_input_guardrail_does_not_cover_the_router_request(self) -> None:
        """Selection precedes `wrap_model_request`, so the guard cannot gate the router's own call."""
        secret = 'my api_key is sk-TOPSECRET'
        routed_prompts: list[str] = []
        answered = False

        def inspect(messages: list[ModelMessage], _info: AgentInfo) -> None:
            prompt = messages[-1].parts[-1]
            assert isinstance(prompt, UserPromptPart)
            assert isinstance(prompt.content, str)
            routed_prompts.append(prompt.content)

        def answer(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:  # pragma: no cover
            nonlocal answered
            answered = True
            return ModelResponse(parts=[TextPart('answered')])

        def no_secrets(prompt: str) -> GuardrailResult:
            if 'api_key' in prompt.lower():
                return GuardrailResult.block('looks like an API key')
            return GuardrailResult.allow()  # pragma: no cover

        agent = Agent(
            capabilities=[
                _router(_router_model('fast', inspect=inspect), fast_model=FunctionModel(answer)),
                InputGuardrail(guard=no_secrets),
            ]
        )

        result = await agent.run(secret)

        assert result.output == 'looks like an API key'
        assert not answered, 'the guard blocked the parent call'
        assert len(routed_prompts) == 1, 'the router ran even though the parent call was skipped'
        assert secret in routed_prompts[0], 'the router was sent the unredacted prompt'
