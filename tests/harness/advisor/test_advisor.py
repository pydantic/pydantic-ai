from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest
from inline_snapshot import snapshot
from pydantic_ai import AdvisorTool, Agent
from pydantic_ai.capabilities import AbstractCapability, PrefixTools
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionDef, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.advisor import Advisor
from tests.conftest import agent_run_names  # pyright: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class _ProviderFunctionModel(FunctionModel):
    def __init__(
        self,
        provider: str,
        function: FunctionDef,
        *,
        supports_advisor: bool = False,
    ) -> None:
        profile = ModelProfile(supported_native_tools=frozenset({AdvisorTool}) if supports_advisor else frozenset())
        super().__init__(function, profile=profile)
        self._system_name = provider

    @property
    def system(self) -> str:
        return self._system_name


def _has_tool_return(messages: Sequence[ModelMessage]) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


_RESOLVES_DUPLICATE_IDS = hasattr(AbstractCapability, 'combine')
"""Whether the installed `pydantic-ai` resolves two capabilities sharing an `id` rather than raising.

Keyed on the hook itself rather than a version number: the release that introduces
`AbstractCapability.combine` (pydantic/pydantic-ai#7248) is not cut yet, so there is no number to
compare against, and asking whether the behaviour is present needs no bookkeeping once there is.
"""


class TestAdvisor:
    async def test_uses_native_advisor_for_matching_provider(self) -> None:
        seen: list[ModelRequestParameters] = []

        def executor(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('anthropic', executor, supports_advisor=True)
        agent = Agent(
            model,
            capabilities=[
                Advisor(
                    'anthropic:claude-opus-4-8',
                    mode='native',
                    max_tokens=2048,
                    caching='5m',
                    forward_history=True,
                )
            ],
        )

        await agent.run('Review this plan.')

        assert seen[0].function_tools == []
        assert seen[0].native_tools == [AdvisorTool(model='claude-opus-4-8', max_tokens=2048, caching='5m')]

    async def test_model_instance_uses_local_advisor_and_preserves_identity(self) -> None:
        advisor_prompts: list[str] = []
        advisor_settings: list[ModelSettings | None] = []

        def advisor_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            advisor_settings.append(info.model_settings)
            prompt = messages[-1].parts[-1]
            assert isinstance(prompt, UserPromptPart)
            assert isinstance(prompt.content, str)
            advisor_prompts.append(prompt.content)
            return ModelResponse(parts=[TextPart('Use a staged rollout.')])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not _has_tool_return(messages):
                return ModelResponse(
                    parts=[ToolCallPart('advisor', {'prompt': 'How should I deploy this?'}, tool_call_id='advisor-1')]
                )
            return ModelResponse(parts=[TextPart('I will use a staged rollout.')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        executor_model = _ProviderFunctionModel('anthropic', executor, supports_advisor=True)
        agent = Agent(executor_model, capabilities=[Advisor(advisor, max_tokens=2048)])

        await agent.run('Plan the deployment.')

        assert advisor_prompts == ['How should I deploy this?']
        assert advisor_settings == [ModelSettings(max_tokens=2048)]

    @pytest.mark.usefixtures('instrument_all_agents')
    async def test_local_advisor_run_is_named_after_the_capability(self, capfire: CaptureLogfire) -> None:
        def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('Use a staged rollout.')])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not _has_tool_return(messages):
                return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'How?'}, tool_call_id='advisor-1')])
            return ModelResponse(parts=[TextPart('done')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        agent = Agent(
            _ProviderFunctionModel('anthropic', executor, supports_advisor=True),
            name='outer',
            capabilities=[Advisor(advisor)],
        )

        await agent.run('Plan the deployment.')

        assert 'advisor' in agent_run_names(capfire)

    async def test_unsupported_native_profile_exposes_local_tool(self) -> None:
        seen: list[ModelRequestParameters] = []

        def executor(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('anthropic', executor)
        agent = Agent(model, capabilities=[Advisor('anthropic:claude-opus-4-8')])

        await agent.run('Review this plan.')

        assert seen[0].native_tools == []
        assert [tool.name for tool in seen[0].function_tools] == ['advisor']

    async def test_unsupported_native_profile_rejects_native_mode(self) -> None:
        model = _ProviderFunctionModel(
            'anthropic',
            lambda _messages, _info: ModelResponse(parts=[TextPart('done')]),
        )
        agent = Agent(model, capabilities=[Advisor('anthropic:claude-opus-4-8', mode='native')])

        with pytest.raises(UserError, match=r'Native tool\(s\).*not supported'):
            await agent.run('Review this plan.')

    @pytest.mark.parametrize(
        ('forward_history', 'expected_prompts'),
        [
            (False, ['What should I do?']),
            (True, ['Original request.', 'What should I do?']),
        ],
    )
    async def test_local_advisor_optionally_forwards_history(
        self,
        forward_history: bool,
        expected_prompts: list[str],
    ) -> None:
        advisor_prompts: list[str] = []

        def advisor_model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            advisor_prompts.extend(
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            )
            return ModelResponse(parts=[TextPart('advice')])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not _has_tool_return(messages):
                return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'What should I do?'})])
            return ModelResponse(parts=[TextPart('done')])

        advisor = FunctionModel(advisor_model)
        agent = Agent(
            FunctionModel(executor),
            capabilities=[Advisor(advisor, forward_history=forward_history)],
        )

        await agent.run('Original request.')

        assert advisor_prompts == expected_prompts

    @pytest.mark.parametrize('prefix', [None, 'consult'])
    async def test_max_uses_is_deterministic_for_parallel_calls(self, prefix: str | None) -> None:
        advisor_calls = 0
        tool_returns: list[str] = []
        tool_name = f'{prefix}_advisor' if prefix is not None else 'advisor'

        def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal advisor_calls
            advisor_calls += 1
            return ModelResponse(parts=[TextPart('first advice')])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal tool_returns
            if not _has_tool_return(messages):
                return ModelResponse(
                    parts=[
                        ToolCallPart(tool_name, {'prompt': 'first'}, tool_call_id='advisor-1'),
                        ToolCallPart(tool_name, {'prompt': 'second'}, tool_call_id='advisor-2'),
                    ]
                )
            tool_returns = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart) and isinstance(part.content, str)
            ]
            return ModelResponse(parts=[TextPart('done')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        capability: AbstractCapability[object] = Advisor(advisor, max_uses=1)
        if prefix is not None:
            capability = PrefixTools(capability, prefix=prefix)
        agent = Agent(FunctionModel(executor), capabilities=[capability])

        await agent.run('Ask twice.')

        assert advisor_calls == 1
        assert tool_returns == snapshot(
            [
                'first advice',
                'Advisor consultation limit reached for this model request. Continue without further advice.',
            ]
        )

    async def test_invalid_call_does_not_consume_max_uses(self) -> None:
        advisor_calls = 0
        executor_calls = 0

        def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal advisor_calls
            advisor_calls += 1
            return ModelResponse(parts=[TextPart('valid advice')])

        def executor(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal executor_calls
            executor_calls += 1
            if executor_calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('advisor', {}, tool_call_id='invalid'),
                        ToolCallPart('advisor', {'prompt': 'valid'}, tool_call_id='valid'),
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        agent = Agent(FunctionModel(executor), capabilities=[Advisor(advisor, max_uses=1)])

        await agent.run('Ask twice.')

        assert advisor_calls == 1

    async def test_max_uses_resets_for_each_executor_request(self) -> None:
        advisor_calls = 0
        executor_calls = 0

        def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal advisor_calls
            advisor_calls += 1
            return ModelResponse(parts=[TextPart(f'advice {advisor_calls}')])

        def executor(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal executor_calls
            executor_calls += 1
            if executor_calls <= 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'advisor',
                            {'prompt': f'question {executor_calls}'},
                            tool_call_id=f'advisor-{executor_calls}',
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        agent = Agent(FunctionModel(executor), capabilities=[Advisor(advisor, max_uses=1)])

        await agent.run('Ask on consecutive requests.')

        assert advisor_calls == 2

    async def test_concurrent_runs_have_independent_max_uses(self) -> None:
        advisor_calls = 0

        async def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal advisor_calls
            advisor_calls += 1
            await asyncio.sleep(0)
            return ModelResponse(parts=[TextPart('advice')])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not _has_tool_return(messages):
                return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'question'})])
            return ModelResponse(parts=[TextPart('done')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        agent = Agent(FunctionModel(executor), capabilities=[Advisor(advisor, max_uses=1)])

        await asyncio.gather(agent.run('first'), agent.run('second'))

        assert advisor_calls == 2

    async def test_openrouter_max_uses_selects_local_fallback(self) -> None:
        seen: list[ModelRequestParameters] = []

        def executor(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('openrouter', executor, supports_advisor=True)
        agent = Agent(
            model,
            capabilities=[Advisor('openrouter:anthropic/claude-opus-4.8', max_uses=1)],
        )

        await agent.run('Review this plan.')

        assert seen[0].native_tools == []
        assert [tool.name for tool in seen[0].function_tools] == ['advisor']

    async def test_openrouter_without_max_uses_uses_native_advisor(self) -> None:
        seen: list[ModelRequestParameters] = []

        def executor(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('openrouter', executor, supports_advisor=True)
        agent = Agent(
            model,
            capabilities=[Advisor('openrouter:anthropic/claude-opus-4.8')],
        )

        await agent.run('Review this plan.')

        assert seen[0].function_tools == []
        assert seen[0].native_tools == [AdvisorTool(model='anthropic/claude-opus-4.8')]

    async def test_local_advisor_shares_parent_usage_and_limits(self) -> None:
        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not _has_tool_return(messages):
                return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'Review this.'})])
            return ModelResponse(parts=[TextPart('done')])

        advisor = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('advice')]))
        agent = Agent(FunctionModel(executor), capabilities=[Advisor(advisor)])

        result = await agent.run('Ask for advice.')

        assert result.usage.requests == 3

        with pytest.raises(UsageLimitExceeded, match='request_limit'):
            await agent.run('Ask for advice.', usage_limits=UsageLimits(request_limit=1))

    async def test_local_unexpected_model_behavior_retries_executor(self) -> None:
        def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise UnexpectedModelBehavior('advisor did not return advice')

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart('continued without advice')])
            return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'Review this.'})])

        agent = Agent(FunctionModel(executor), capabilities=[Advisor(FunctionModel(advisor_model))])

        result = await agent.run('Ask for advice.')

        assert result.output == 'continued without advice'

    @pytest.mark.parametrize(
        ('executor_provider', 'advisor_model', 'required_provider'),
        [
            ('openrouter', 'anthropic:claude-opus-4-8', 'anthropic'),
            ('anthropic', 'openrouter:anthropic/claude-opus-4.8', 'openrouter'),
        ],
    )
    async def test_native_mode_rejects_cross_provider_model_id(
        self,
        executor_provider: str,
        advisor_model: str,
        required_provider: str,
    ) -> None:
        executor = _ProviderFunctionModel(
            executor_provider,
            lambda _messages, _info: ModelResponse(parts=[TextPart('done')]),
            supports_advisor=True,
        )
        agent = Agent(executor, capabilities=[Advisor(advisor_model, mode='native')])

        with pytest.raises(UserError, match=f'requires a {required_provider} executor'):
            await agent.run('Review this.')

    async def test_composes_duplicates_and_rejects_a_conflicting_tool_name(self) -> None:
        model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('done')]))

        # This branch pins the composing behaviour; `_RESOLVES_DUPLICATE_IDS` keeps the file
        # runnable against a released core that still raises (see its docstring).
        if _RESOLVES_DUPLICATE_IDS:  # pragma: no cover - depends on the installed core
            # An agent has one advisor, so two resolve to one rather than colliding -- which is
            # what lets two packaged harnesses that each carry an `Advisor` compose. The later
            # configuration wins where both state one.
            agent = Agent(model, capabilities=[Advisor(model, max_tokens=2048), Advisor(model, max_tokens=4096)])
            await agent.run('Review this.')
            merged = next(c for c in agent._root_capability.capabilities if isinstance(c, Advisor))  # pyright: ignore[reportPrivateUsage]
            assert merged.max_tokens == 4096
        else:  # pragma: no cover - depends on the installed core
            with pytest.raises(UserError, match="Capability id 'advisor' is used by multiple capabilities"):
                Agent(model, capabilities=[Advisor(model), Advisor(model)])

        def advisor(prompt: str) -> str:  # pragma: no cover
            return prompt

        agent = Agent(model, capabilities=[Advisor(model)], tools=[advisor])
        with pytest.raises(UserError, match='tool whose name conflicts'):
            await agent.run('Review this.')

    def test_rejects_invalid_options(self) -> None:
        model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('done')]))
        with pytest.raises(ValueError, match='Advisor.mode'):
            Advisor(model, mode='invalid')  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match='Advisor.max_uses must be at least 1'):
            Advisor(model, max_uses=0)
        with pytest.raises(ValueError, match='Advisor.max_tokens must be at least 1024'):
            Advisor(model, max_tokens=1023)
        with pytest.raises(ValueError, match="mode='native'.*model name"):
            Advisor(model, mode='native')
        with pytest.raises(ValueError, match="mode='native'.*model name"):
            Advisor('test', mode='native')
        with pytest.raises(ValueError, match="mode='native'.*model name"):
            Advisor('openai:gpt-5.4', mode='native')
        with pytest.raises(ValueError, match="mode='native'.*model name"):
            Advisor('anthropic:', mode='native')
        with pytest.raises(ValueError, match='not supported by OpenRouter'):
            Advisor('openrouter:anthropic/claude-opus-4.8', mode='native', max_uses=1)

    async def test_loads_string_model_from_agent_spec(self) -> None:
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [
                    {
                        'Advisor': {
                            'model': 'test',
                            'mode': 'local',
                        }
                    }
                ],
            },
            custom_capability_types=[Advisor],
        )

        result = await agent.run('Consult the advisor.')

        assert result.usage.requests == 3
