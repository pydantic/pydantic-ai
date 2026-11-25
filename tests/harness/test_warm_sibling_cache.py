"""Tests for `warm_sibling_cache` on `SubAgents` and `DynamicWorkflow`: siblings that start together go one at a time."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Literal

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Caching
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
from pydantic_ai_harness.sub_agents import SubAgent, SubAgents

from ..conftest import try_import

with try_import() as anthropic_installed:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

TASKS = ('alpha', 'beta', 'gamma')


class Siblings:
    """Records when each sibling's first request starts, and when the first one's first event goes out.

    The first sibling waits a moment for another to start before it streams its first event, so siblings that
    start together start inside that wait, and siblings that are held start after the event.
    """

    def __init__(self, *, fail_first: bool = False) -> None:
        self.log: list[str] = []
        self.fail_first = fail_first
        self.other_started = asyncio.Event()

    async def first_request(self, messages: list[ModelMessage]) -> str:
        (task,) = [part.content for part in messages[0].parts if isinstance(part, UserPromptPart)]
        assert isinstance(task, str)
        self.log.append(f'start {task}')
        if len(self.log) > 1:
            self.other_started.set()
            return task
        # real-wait: held siblings never start, so the first one can only give the others a moment to.
        with anyio.move_on_after(0.2):
            await self.other_started.wait()
        if self.fail_first:
            raise ModelAPIError('worker', 'provider down')
        self.log.append(f'first event {task}')
        return task

    def model(self, *, warms_siblings: bool = True, cache: bool = True) -> FunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(f'did {await self.first_request(messages)}')])

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield f'did {await self.first_request(messages)}'

        return FunctionModel(
            respond,
            stream_function=stream,
            profile=ModelProfile(supports_cache=True, prompt_cache_warms_siblings=warms_siblings),
            settings={'cache': True} if cache else None,
        )

    @property
    def held(self) -> bool:
        """Whether the siblings after the first started only after the first one's first event."""
        return self.log[1].startswith('first event')


def delegate_to_all(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart('delegate_task', {'agent_name': 'worker', 'task': task}, tool_call_id=task)
                for task in TASKS
            ]
        )
    return ModelResponse(parts=[TextPart('all done')])


async def drain(ctx: RunContext[Any], events: AsyncIterator[AgentStreamEvent]) -> None:
    async for _ in events:
        pass


def sub_agents_parent(
    worker: FunctionModel, *, mode: bool | Literal['auto'] = 'auto', streamed: bool = False
) -> Agent[None, str]:
    return Agent(
        FunctionModel(delegate_to_all),
        capabilities=[
            SubAgents(
                agents=[SubAgent(Agent(worker, name='worker'))],
                warm_sibling_cache=mode,
                contain_errors=True,
                # A handler makes the children stream their requests.
                event_stream_handler=drain if streamed else None,
            )
        ],
    )


class TestSubAgentsWarmSiblingCache:
    @pytest.mark.parametrize('streamed', [False, True], ids=['request', 'stream'])
    async def test_siblings_wait_for_the_first_ones_first_event(self, streamed: bool) -> None:
        siblings = Siblings()
        result = await sub_agents_parent(siblings.model(), streamed=streamed).run('go')
        assert result.output == 'all done'
        assert siblings.held
        assert sorted(entry for entry in siblings.log if entry.startswith('start')) == [f'start {t}' for t in TASKS]

    @pytest.mark.parametrize(
        ('warms_siblings', 'cache', 'mode', 'held'),
        [
            pytest.param(False, True, 'auto', False, id='profile-off'),
            pytest.param(True, False, 'auto', False, id='no-cache'),
            pytest.param(True, True, False, False, id='turned-off'),
            pytest.param(False, False, True, True, id='turned-on'),
        ],
    )
    async def test_policy(self, warms_siblings: bool, cache: bool, mode: bool | Literal['auto'], held: bool) -> None:
        siblings = Siblings()
        model = siblings.model(warms_siblings=warms_siblings, cache=cache)
        result = await sub_agents_parent(model, mode=mode).run('go')
        assert result.output == 'all done'
        assert siblings.held is held

    @pytest.mark.parametrize('streamed', [False, True], ids=['request', 'stream'])
    async def test_first_sibling_failing_releases_the_others(self, streamed: bool) -> None:
        siblings = Siblings(fail_first=True)
        # real-wait: well inside the 10 seconds a held sibling waits on its own, so this shows the failure released them.
        with anyio.fail_after(5):
            result = await sub_agents_parent(siblings.model(), streamed=streamed).run('go')
        assert result.output == 'all done'
        assert len(siblings.log) == len(TASKS)
        assert not any(entry.startswith('first event') for entry in siblings.log)

    async def test_only_the_first_request_is_held(self) -> None:
        siblings = Siblings()
        second_requests: list[str] = []

        async def two_steps(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            if len(messages) == 1:
                task = await siblings.first_request(messages)
                yield {0: DeltaToolCall(name='note', json_args=json.dumps({'text': task}))}
                return
            second_requests.append('second request')
            yield 'done'

        model = FunctionModel(
            stream_function=two_steps, profile=ModelProfile(supports_cache=True, prompt_cache_warms_siblings=True)
        )
        worker = Agent(model, name='worker', model_settings={'cache': True})

        @worker.tool_plain
        def note(text: str) -> str:
            return text

        parent = Agent(
            FunctionModel(delegate_to_all),
            capabilities=[SubAgents(agents=[SubAgent(worker)], event_stream_handler=drain)],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        assert siblings.held
        assert second_requests == ['second request'] * len(TASKS)


class TestDynamicWorkflowWarmSiblingCache:
    @pytest.mark.parametrize(('mode', 'held'), [('auto', True), (False, False)])
    async def test_gathered_calls_wait_for_the_first_ones_first_event(
        self, mode: bool | Literal['auto'], held: bool
    ) -> None:
        siblings = Siblings()
        script = 'import asyncio\nawait asyncio.gather(' + ', '.join(f'worker(task={task!r})' for task in TASKS) + ')'

        def orchestrate(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('run_workflow', {'code': script})])
            return ModelResponse(parts=[TextPart('all done')])

        worker = Agent(siblings.model(), name='worker', description='Does a task.')
        parent = Agent(
            FunctionModel(orchestrate), capabilities=[DynamicWorkflow(agents=[worker], warm_sibling_cache=mode)]
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        assert siblings.held is held


@pytest.fixture
def vcr_config(vcr_config: dict[str, object], pytestconfig: pytest.Config) -> dict[str, object]:
    """On replay, match each request's body too, so each sibling gets its own recorded response."""
    if pytestconfig.getoption('--record-mode', default=None) not in (None, 'none'):
        return vcr_config
    return {**vcr_config, 'match_on': ['method', 'uri', 'body']}


class RecordUsage(AbstractCapability[Any]):
    def __init__(self) -> None:
        self.usages: list[RequestUsage] = []

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        self.usages.append(response.usage)
        return response


# Long enough to be above Claude's minimum cacheable prompt.
POLICY = '\n'.join(
    f'{n}. When a task mentions topic {n}, answer in a single short sentence about it.' for n in range(400)
)


@pytest.mark.skipif(not anthropic_installed(), reason='anthropic not installed')
@pytest.mark.vcr
async def test_claude_siblings_read_the_first_ones_cached_prefix(
    allow_model_requests: None, anthropic_api_key: str
) -> None:
    usage = RecordUsage()
    worker = Agent(
        AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(api_key=anthropic_api_key)),
        name='worker',
        instructions=f'You do one small task.\n\n{POLICY}',
        model_settings={'thinking': False, 'max_tokens': 256},
        capabilities=[usage, Caching()],
    )
    parent = Agent(FunctionModel(delegate_to_all), capabilities=[SubAgents(agents=[SubAgent(worker)])])
    result = await parent.run('go')
    assert result.output == 'all done'
    # The first sibling writes the shared prefix, and the others, held until it streamed, read it.
    assert sorted((usage.cache_read_tokens, usage.cache_write_tokens) for usage in usage.usages) == snapshot(
        [(0, 10414), (10409, 4), (10409, 5)]
    )
