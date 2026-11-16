"""`PruneToolResults`: the model replaces a tool result it just received with its own summary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import PruneToolResults
from pydantic_ai_harness.compaction import ClearToolResults
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY

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


def _agent(script: Script, *capabilities: Any) -> Agent[None, str]:
    def grep(pattern: str) -> str:
        """Search the repository."""
        return JUNK

    return Agent(script.model(), tools=[grep], capabilities=list(capabilities))


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
            'p1': 'Pruned: the result will be replaced with your summary before your next turn.',
        }
        assert _returns(result.all_messages())['g1'] == PRUNED

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_runs_before_automatic_compaction(self) -> None:
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

    async def test_refuses_a_result_from_an_earlier_step(self) -> None:
        look = ModelResponse(parts=[ToolCallPart('grep', {'pattern': 'x'}, tool_call_id='g2')])
        script = Script([GREP, look, _prune('g1'), DONE])

        result = await _agent(script, PruneToolResults()).run('find the TODOs')

        returns = _returns(result.all_messages())
        assert returns['g1'] == JUNK
        assert returns['p1'] == "Not pruned: 'g1' is not a result from your previous step that can be pruned."

    async def test_refuses_an_unknown_id_on_the_first_step(self) -> None:
        script = Script([_prune('nope'), DONE])

        result = await _agent(script, PruneToolResults()).run('find the TODOs')

        assert _returns(result.all_messages())['p1'] == (
            "Not pruned: 'nope' is not a result from your previous step that can be pruned."
        )

    async def test_refuses_a_summary_not_shorter_than_the_result(self) -> None:
        script = Script([GREP, _prune('g1', JUNK), DONE])

        result = await _agent(script, PruneToolResults()).run('find the TODOs')

        returns = _returns(result.all_messages())
        assert returns['g1'] == JUNK
        assert returns['p1'] == 'Not pruned: the summary is not shorter than the result.'
