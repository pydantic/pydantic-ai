"""Tests for the ExaAgent capability, ExaAgentToolset, and agent_run_result."""

import json
from pathlib import Path

import httpx
import pytest
from exa_py.agent.types import (
    AgentError,
    AgentGroundingCitation,
    AgentGroundingEntry,
    AgentOutput,
    AgentRun,
    AgentRunStatus,
)
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai.exceptions import CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.exa import (
    RUN_ID_METADATA_KEY,
    ExaAgent,
    ExaAgentToolset,
    agent_run_result,
)


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _text(result: ToolReturn[str]) -> str:
    """The model-facing text of a tool result."""
    body = result.return_value
    assert isinstance(body, str)
    return body


def _run(
    status: AgentRunStatus = 'completed',
    *,
    run_id: str = 'run_1',
    text: str | None = None,
    structured: object = None,
    citations: list[tuple[str, str | None]] | None = None,
    error_message: str | None = None,
) -> AgentRun:
    """A real `exa_py` agent run with the fields the capability reads."""
    output: AgentOutput | None = None
    if text is not None or structured is not None:
        grounding = (
            [
                AgentGroundingEntry(
                    field='answer',
                    citations=[AgentGroundingCitation(url=url, title=title) for url, title in citations],
                )
            ]
            if citations
            else None
        )
        output = AgentOutput(text=text, structured=structured, grounding=grounding)
    error = AgentError(message=error_message) if error_message is not None else None
    return AgentRun(id=run_id, status=status, output=output, error=error)


class _FakeRuns:
    """In-memory `ExaAgentRuns` double: canned runs, recorded call arguments."""

    def __init__(
        self,
        *,
        created: AgentRun | None = None,
        finished: AgentRun | None = None,
        error: Exception | None = None,
        poll_error: Exception | None = None,
    ) -> None:
        self.created = created if created is not None else _run('queued')
        self.finished = finished if finished is not None else _run(text='Done.')
        self.error = error
        self.poll_error = poll_error
        self.create_calls: list[dict[str, object]] = []
        self.poll_calls: list[dict[str, object]] = []

    async def create(
        self,
        *,
        query: str,
        system_prompt: str | None = None,
        output_schema: 'dict[str, object] | type[BaseModel] | None' = None,
        effort: str | None = None,
        previous_run_id: str | None = None,
    ) -> AgentRun:
        if self.error is not None:
            raise self.error
        self.create_calls.append(
            {
                'query': query,
                'system_prompt': system_prompt,
                'output_schema': output_schema,
                'effort': effort,
                'previous_run_id': previous_run_id,
            }
        )
        return self.created

    async def poll_until_finished(
        self, run_id: str, *, poll_interval: int = 1000, timeout_ms: int = 3600000
    ) -> AgentRun:
        self.poll_calls.append({'run_id': run_id, 'poll_interval': poll_interval, 'timeout_ms': timeout_ms})
        if self.poll_error is not None:
            raise self.poll_error
        return self.finished


class _Facts(BaseModel):
    name: str
    founded_year: int


def _call_tool_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call `exa_agent` on the first turn, then answer with text (e.g. after a retry prompt)."""
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart(tool_name='exa_agent', args={'query': 'q'})])
    return ModelResponse(parts=[TextPart('done')])


def _retry_parts(messages: list[ModelMessage]) -> list[RetryPromptPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]


class TestExaAgentToolset:
    async def test_creates_run_and_defers(self) -> None:
        runs = _FakeRuns(created=_run('queued', run_id='run_42'))
        toolset = ExaAgent[None](runs=runs, effort='high', system_prompt='Be thorough.').get_toolset()
        assert isinstance(toolset, ExaAgentToolset)
        with pytest.raises(CallDeferred) as exc_info:
            await toolset.exa_agent('research task')
        metadata = exc_info.value.metadata
        assert metadata is not None
        assert metadata[RUN_ID_METADATA_KEY] == 'run_42'
        assert isinstance(metadata['exa_agent_owner_id'], str)
        assert runs.create_calls == [
            {
                'query': 'research task',
                'system_prompt': 'Be thorough.',
                'output_schema': None,
                'effort': 'high',
                'previous_run_id': None,
            }
        ]

    async def test_previous_run_id_forwarded(self) -> None:
        runs = _FakeRuns()
        with pytest.raises(CallDeferred):
            await ExaAgent[None](runs=runs).get_toolset().exa_agent('follow up', previous_run_id='run_0')
        assert runs.create_calls[0]['previous_run_id'] == 'run_0'

    async def test_model_output_schema_forwarded(self) -> None:
        runs = _FakeRuns()
        with pytest.raises(CallDeferred):
            await ExaAgent[None](runs=runs, output_schema=_Facts).get_toolset().exa_agent('q')
        assert runs.create_calls[0]['output_schema'] is _Facts

    async def test_non_2xx_becomes_model_retry(self) -> None:
        runs = _FakeRuns(error=ValueError('Request failed with status code 429: rate limited'))
        with pytest.raises(ModelRetry, match='Exa request failed: .*429'):
            await ExaAgent[None](runs=runs).get_toolset().exa_agent('q')

    async def test_auth_failure_propagates(self) -> None:
        runs = _FakeRuns(error=ValueError('Request failed with status code 401: invalid API key'))
        with pytest.raises(ValueError, match='status code 401'):
            await ExaAgent[None](runs=runs).get_toolset().exa_agent('q')

    async def test_network_failure_becomes_model_retry(self) -> None:
        runs = _FakeRuns(error=httpx.ConnectError('connection refused'))
        with pytest.raises(ModelRetry, match='Exa request failed: connection refused'):
            await ExaAgent[None](runs=runs).get_toolset().exa_agent('q')


class TestAgentRunResult:
    def test_completed_text_with_sources_and_run_id(self) -> None:
        run = _run(text='Answer.', citations=[('https://a.dev', 'A'), ('https://b.dev', None)])
        result = agent_run_result(run)
        assert _text(result) == (
            'Answer.\n\nRun ID: run_1 (pass as previous_run_id for follow-ups)\n\n'
            'Sources:\n- A: https://a.dev\n- (untitled): https://b.dev'
        )
        assert result.metadata == {
            RUN_ID_METADATA_KEY: 'run_1',
            'sources': [{'url': 'https://a.dev', 'title': 'A'}, {'url': 'https://b.dev', 'title': None}],
        }

    def test_completed_without_output(self) -> None:
        result = agent_run_result(_run())
        assert _text(result) == '(no text output)\n\nRun ID: run_1 (pass as previous_run_id for follow-ups)'
        assert result.metadata == {RUN_ID_METADATA_KEY: 'run_1', 'sources': []}

    def test_structured_output_validated_against_model(self) -> None:
        run = _run(structured={'name': 'Exa', 'founded_year': 2021})
        result = agent_run_result(run, output_schema=_Facts)
        assert _text(result).startswith('{"name":"Exa","founded_year":2021}')

    def test_structured_output_mismatch_raises_model_retry(self) -> None:
        run = _run(structured={'name': 'Exa'})
        with pytest.raises(ModelRetry, match='did not match the configured schema'):
            agent_run_result(run, output_schema=_Facts)

    def test_structured_output_with_dict_schema_rendered_as_json(self) -> None:
        run = _run(structured={'name': 'Exa'})
        result = agent_run_result(run, output_schema={'type': 'object'})
        assert _text(result).startswith('{"name": "Exa"}')

    def test_failed_run_reports_error(self) -> None:
        run = _run('failed', error_message='budget exceeded')
        result = agent_run_result(run)
        assert _text(result) == 'Exa agent run run_1 failed: budget exceeded'
        assert result.metadata == {RUN_ID_METADATA_KEY: 'run_1', 'sources': []}

    def test_failed_run_without_error_details(self) -> None:
        assert _text(agent_run_result(_run('cancelled'))) == 'Exa agent run run_1 cancelled: no error details'


class TestExaAgent:
    def test_default_runs_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('EXA_API_KEY', raising=False)
        with pytest.raises(UserError, match='EXA_API_KEY'):
            ExaAgent[None]().get_toolset()

    def test_default_runs_built_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('EXA_API_KEY', 'test-key')
        toolset = ExaAgent[None]().get_toolset()
        assert isinstance(toolset, ExaAgentToolset)

    def test_instructions_reference_the_tool(self) -> None:
        instructions = ExaAgent[None]().get_instructions()
        assert isinstance(instructions, str)
        assert '`exa_agent`' in instructions
        assert 'previous_run_id' in instructions

    def test_custom_guidance_replaces_default(self) -> None:
        assert ExaAgent[None](guidance='Delegate research.').get_instructions() == 'Delegate research.'

    def test_empty_guidance_disables_instructions(self) -> None:
        assert ExaAgent[None](guidance='').get_instructions() is None

    async def test_inline_execution_resolves_run_in_one_agent_run(self) -> None:
        runs = _FakeRuns(
            created=_run('queued', run_id='run_7'),
            finished=_run(text='Findings.', run_id='run_7', citations=[('https://a.dev', 'A')]),
        )
        agent = Agent(TestModel(), capabilities=[ExaAgent(runs=runs, poll_interval=5, timeout_ms=60_000)])

        result = await agent.run('Research something.')

        assert runs.poll_calls == [{'run_id': 'run_7', 'poll_interval': 5, 'timeout_ms': 60_000}]
        parts = [
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'exa_agent'
        ]
        expected = agent_run_result(runs.finished)
        assert [part.content for part in parts] == [_text(expected)]
        assert [part.metadata for part in parts] == [expected.metadata]

    async def test_deferred_calls_from_other_toolsets_left_unresolved(self) -> None:
        runs = _FakeRuns()
        capability = ExaAgent[None](runs=runs)
        requests = DeferredToolRequests(
            calls=[ToolCallPart(tool_name='other_tool', tool_call_id='c1')],
            metadata={'c1': {'other_key': 'x'}},
        )
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
        result = await capability.handle_deferred_tool_calls(ctx, requests=requests)
        assert result is None
        assert runs.poll_calls == []

    async def test_prefixed_capability_still_resolves_inline(self) -> None:
        runs = _FakeRuns(
            created=_run('queued', run_id='run_8'),
            finished=_run(text='Findings.', run_id='run_8'),
        )
        agent = Agent(TestModel(), capabilities=[PrefixTools(wrapped=ExaAgent(runs=runs), prefix='cb')])

        result = await agent.run('Research something.')

        assert runs.poll_calls == [{'run_id': 'run_8', 'poll_interval': 1000, 'timeout_ms': 3_600_000}]
        parts = [
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'cb_exa_agent'
        ]
        assert [part.content for part in parts] == [_text(agent_run_result(runs.finished))]

    async def test_other_instances_calls_left_unresolved(self) -> None:
        creator = ExaAgent[None](runs=_FakeRuns(created=_run('queued', run_id='run_a')))
        with pytest.raises(CallDeferred) as exc_info:
            await creator.get_toolset().exa_agent('task')
        metadata = exc_info.value.metadata
        assert metadata is not None

        other_runs = _FakeRuns()
        other = ExaAgent[None](runs=other_runs)
        requests = DeferredToolRequests(
            calls=[ToolCallPart(tool_name='exa_agent', tool_call_id='c1')],
            metadata={'c1': metadata},
        )
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
        result = await other.handle_deferred_tool_calls(ctx, requests=requests)
        assert result is None
        assert other_runs.poll_calls == []

    async def test_structured_mismatch_becomes_retry_prompt_and_run_continues(self) -> None:
        runs = _FakeRuns(
            created=_run('queued', run_id='run_m'),
            finished=_run(structured={'name': 'Exa'}, run_id='run_m'),
        )
        agent = Agent(
            FunctionModel(_call_tool_then_answer),
            capabilities=[ExaAgent(runs=runs, output_schema=_Facts)],
        )

        result = await agent.run('Research something.')

        assert result.output == 'done'
        [retry] = _retry_parts(result.all_messages())
        assert retry.tool_name == 'exa_agent'
        assert 'did not match the configured schema' in retry.model_response()

    async def test_poll_timeout_becomes_retry_prompt_with_run_id(self) -> None:
        runs = _FakeRuns(
            created=_run('queued', run_id='run_t'),
            poll_error=TimeoutError('Agent run run_t did not complete within 60000ms'),
        )
        agent = Agent(
            FunctionModel(_call_tool_then_answer),
            capabilities=[ExaAgent(runs=runs, timeout_ms=60_000)],
        )

        result = await agent.run('Research something.')

        assert result.output == 'done'
        [retry] = _retry_parts(result.all_messages())
        assert retry.tool_name == 'exa_agent'
        assert 'run_t' in retry.model_response()
        assert 'previous_run_id' in retry.model_response()

    async def test_poll_transient_error_returned_as_retry_result(self) -> None:
        creator_runs = _FakeRuns(created=_run('queued', run_id='run_v'))
        capability = ExaAgent[None](runs=creator_runs)
        with pytest.raises(CallDeferred) as exc_info:
            await capability.get_toolset().exa_agent('task')
        metadata = exc_info.value.metadata
        assert metadata is not None

        capability.runs = _FakeRuns(poll_error=ValueError('Request failed with status code 500: oops'))
        requests = DeferredToolRequests(
            calls=[ToolCallPart(tool_name='exa_agent', tool_call_id='c1')],
            metadata={'c1': metadata},
        )
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
        result = await capability.handle_deferred_tool_calls(ctx, requests=requests)
        assert result is not None
        retry = result.calls['c1']
        assert isinstance(retry, ModelRetry)
        assert 'run_v' in str(retry)

    async def test_poll_auth_failure_propagates(self) -> None:
        creator_runs = _FakeRuns(created=_run('queued', run_id='run_x'))
        capability = ExaAgent[None](runs=creator_runs)
        with pytest.raises(CallDeferred) as exc_info:
            await capability.get_toolset().exa_agent('task')
        metadata = exc_info.value.metadata
        assert metadata is not None

        capability.runs = _FakeRuns(poll_error=ValueError('Request failed with status code 401: invalid API key'))
        requests = DeferredToolRequests(
            calls=[ToolCallPart(tool_name='exa_agent', tool_call_id='c1')],
            metadata={'c1': metadata},
        )
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
        with pytest.raises(ValueError, match='status code 401'):
            await capability.handle_deferred_tool_calls(ctx, requests=requests)

    async def test_external_execution_bubbles_deferred_requests(self) -> None:
        runs = _FakeRuns(created=_run('queued', run_id='run_9'))
        agent = Agent(
            TestModel(),
            output_type=[str, DeferredToolRequests],
            capabilities=[ExaAgent(runs=runs, execution='external')],
        )

        result = await agent.run('Research something.')

        assert runs.poll_calls == []
        output = result.output
        assert isinstance(output, DeferredToolRequests)
        [call] = output.calls
        assert call.tool_name == 'exa_agent'
        assert output.metadata[call.tool_call_id][RUN_ID_METADATA_KEY] == 'run_9'


class TestAgentSpec:
    def test_spec_schema_includes_exa_agent(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([ExaAgent])
        assert 'ExaAgent' in json.dumps(schema)

    def test_from_spec_builds_capability(self) -> None:
        schema: dict[str, object] = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        capability = ExaAgent[None].from_spec(
            effort='low',
            execution='external',
            output_schema=schema,
            system_prompt='Be brief.',
            poll_interval=500,
            timeout_ms=120_000,
            guidance='Delegate.',
        )
        assert capability.effort == 'low'
        assert capability.execution == 'external'
        assert capability.output_schema == schema
        assert capability.system_prompt == 'Be brief.'
        assert capability.poll_interval == 500
        assert capability.timeout_ms == 120_000
        assert capability.guidance == 'Delegate.'
        assert capability.runs is None

    def test_agent_loads_from_spec_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('EXA_API_KEY', 'test-key')
        spec = tmp_path / 'agent.yaml'
        spec.write_text('model: test\ncapabilities:\n  - ExaAgent:\n      effort: low\n')
        agent = Agent.from_file(spec, custom_capability_types=[ExaAgent])
        assert isinstance(agent, Agent)
