"""`PruneToolResults`: the model replaces a tool result it just received with its own summary."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness import PruneToolResults
from pydantic_ai_harness.compaction import ClearToolResults
from tests.conftest import try_import
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire

JUNK = 'build/fingerprint.json: abc123\n' * 200


@dataclass
class Script:
    """A `FunctionModel` answering with `responses` in order, recording what each request sent."""

    responses: list[ModelResponse]
    seen: list[list[ModelMessage]] = field(default_factory=list[list[ModelMessage]])

    def model(self) -> FunctionModel:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.seen.append(list(messages))
            return self.responses[len(self.seen) - 1]

        return FunctionModel(respond)


def _agent(script: Script, *capabilities: AbstractCapability[None]) -> Agent[None, str]:
    def grep(pattern: str) -> str:
        """Search the repository."""
        return JUNK

    return Agent(script.model(), deps_type=type(None), tools=[grep], capabilities=list(capabilities))


def _prune(tool_call_id: str, summary: str = 'grep matched build fingerprints only') -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart('prune_tool_result', {'tool_call_id': tool_call_id, 'summary': summary}, tool_call_id='p1')]
    )


def _returns(messages: list[ModelMessage]) -> dict[str, str]:
    return {
        part.tool_call_id: str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }


GREP = ModelResponse(parts=[ToolCallPart('grep', {'pattern': 'TODO'}, tool_call_id='g1')])
DONE = ModelResponse(parts=[TextPart('done')])
PRUNED = '[tool result pruned] grep matched build fingerprints only'


class TestPruneToolResults:
    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_replaces_the_result_before_the_next_request(self) -> None:
        script = Script([GREP, _prune('g1'), DONE])

        result = await _agent(script, PruneToolResults()).run('find the TODOs')

        assert _returns(script.seen[1])['g1'] == JUNK
        assert _returns(script.seen[2]) == {
            'g1': PRUNED,
            'p1': 'Pruned: the result was replaced with your summary.',
        }
        assert _returns(result.all_messages())['g1'] == PRUNED

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_automatic_compaction_sees_the_pruned_history(self) -> None:
        @dataclass
        class Recorder(ClearToolResults[None]):
            seen: list[str] = field(default_factory=list[str])

            async def compact(self, messages: list[ModelMessage], ctx: RunContext[None]) -> list[ModelMessage]:
                self.seen.append(_returns(messages).get('g1', ''))
                return messages

        recorder = Recorder(max_messages=1)
        script = Script([GREP, _prune('g1'), DONE])

        await _agent(script, recorder, PruneToolResults()).run('find the TODOs')

        assert recorder.seen == [JUNK, PRUNED]

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_survives_a_run_paused_for_approval(self) -> None:
        deploy_call = ToolCallPart('deploy', {}, tool_call_id='d1')
        script = Script([GREP, ModelResponse(parts=[*_prune('g1').parts, deploy_call]), DONE])
        agent = Agent(
            script.model(),
            deps_type=type(None),
            output_type=[str, DeferredToolRequests],
            capabilities=[PruneToolResults()],
        )

        @agent.tool_plain
        def grep(pattern: str) -> str:
            """Search the repository."""
            return JUNK

        @agent.tool_plain(requires_approval=True)
        def deploy() -> str:
            """Deploy the app."""
            return 'deployed'

        paused = await agent.run('find the TODOs')
        assert isinstance(paused.output, DeferredToolRequests)

        await agent.run(
            message_history=paused.all_messages(), deferred_tool_results=DeferredToolResults(approvals={'d1': True})
        )

        assert _returns(script.seen[2])['g1'] == PRUNED

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_prunes_when_the_call_itself_was_approved_on_resume(self) -> None:
        @dataclass
        class ApproveEveryTool(AbstractCapability[None]):
            async def before_tool_execute(
                self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
            ) -> ValidatedToolArgs:
                if call.tool_name == 'prune_tool_result' and not ctx.tool_call_approved:
                    raise ApprovalRequired
                return args

        look = ToolCallPart('grep', {'pattern': 'x'}, tool_call_id='g2')
        script = Script([GREP, ModelResponse(parts=[*_prune('g1').parts, look]), DONE])
        agent = _agent(script, PruneToolResults(), ApproveEveryTool())

        paused = await agent.run('find the TODOs', output_type=[str, DeferredToolRequests])
        assert isinstance(paused.output, DeferredToolRequests)

        await agent.run(
            message_history=paused.all_messages(), deferred_tool_results=DeferredToolResults(approvals={'p1': True})
        )

        returns = _returns(script.seen[2])
        assert returns['g1'] == PRUNED
        assert returns['g2'] == JUNK

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_prunes_two_results_in_one_step(self) -> None:
        greps = ModelResponse(
            parts=[
                ToolCallPart('grep', {'pattern': 'a'}, tool_call_id='g1'),
                ToolCallPart('grep', {'pattern': 'b'}, tool_call_id='g2'),
            ]
        )
        prunes = ModelResponse(
            parts=[
                ToolCallPart('prune_tool_result', {'tool_call_id': 'g1', 'summary': 'a'}, tool_call_id='p1'),
                ToolCallPart('prune_tool_result', {'tool_call_id': 'g2', 'summary': 'b'}, tool_call_id='p2'),
            ]
        )
        script = Script([greps, prunes, DONE])

        await _agent(script, PruneToolResults()).run('find the TODOs')

        returns = _returns(script.seen[2])
        assert (returns['g1'], returns['g2']) == ('[tool result pruned] a', '[tool result pruned] b')

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_leaves_an_earlier_result_with_a_reused_id(self) -> None:
        look = ModelResponse(parts=[ToolCallPart('grep', {'pattern': 'x'}, tool_call_id='g2')])
        script = Script([GREP, look, GREP, _prune('g1'), DONE])

        result = await _agent(script, PruneToolResults()).run('find the TODOs')

        returns = [
            str(part.content)
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_call_id == 'g1'
        ]
        assert returns == [JUNK, PRUNED]

    async def test_refuses_a_result_from_an_earlier_step(self) -> None:
        look = ModelResponse(parts=[ToolCallPart('grep', {'pattern': 'x'}, tool_call_id='g2')])
        script = Script([GREP, look, _prune('g1'), DONE])

        result = await _agent(script, PruneToolResults()).run('find the TODOs')

        returns = _returns(result.all_messages())
        assert returns['g1'] == JUNK
        assert returns['p1'] == "Not pruned: 'g1' is not a result from your previous step."

    async def test_refuses_an_unknown_id_on_the_first_step(self) -> None:
        script = Script([_prune('nope'), DONE])

        result = await _agent(script, PruneToolResults()).run('find the TODOs')

        assert _returns(result.all_messages())['p1'] == ("Not pruned: 'nope' is not a result from your previous step.")

    async def test_refuses_a_typed_result(self) -> None:
        search = ToolSearchReturnPart(
            content={'discovered_tools': [{'name': f'tool_{i}'} for i in range(50)]}, tool_call_id='ts1'
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('find the TODOs')]),
            ModelResponse(parts=[ToolCallPart('search_tools', {'queries': ['todo']}, tool_call_id='ts1')]),
            ModelRequest(parts=[search]),
        ]
        script = Script([_prune('ts1'), DONE])

        result = await _agent(script, PruneToolResults()).run(message_history=history)

        assert _returns(result.all_messages())['p1'] == (
            "Not pruned: the result of 'ts1' is structured data that cannot be replaced."
        )
        assert result.all_messages()[2].parts == [search]

    async def test_refuses_a_summary_not_shorter_than_the_result(self) -> None:
        script = Script([GREP, _prune('g1', JUNK), DONE])

        result = await _agent(script, PruneToolResults()).run('find the TODOs')

        returns = _returns(result.all_messages())
        assert returns['g1'] == JUNK
        assert returns['p1'] == 'Not pruned: the summary is not shorter than the result.'


@pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed')
@pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
async def test_span_marks_the_prune_model_requested(capfire: CaptureLogfire) -> None:
    agent = _agent(Script([GREP, _prune('g1'), DONE]), PruneToolResults())
    agent.instrument = InstrumentationSettings()

    await agent.run('find the TODOs')

    [span] = [s for s in capfire.exporter.exported_spans_as_dict() if s['name'] == 'compact_messages']
    attributes = span['attributes']
    assert attributes['compaction.strategy'] == 'prune_tool_results'
    assert attributes['compaction.model_requested'] is True
    assert attributes['compaction.messages_before'] == attributes['compaction.messages_after'] == 4
    assert attributes['compaction.tokens_after'] < attributes['compaction.tokens_before']
