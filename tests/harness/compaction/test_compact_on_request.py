"""`CompactOnRequest`: the model asks for a compaction, and it runs before the next request."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai_harness.compaction import (
    CompactOnRequest,
    SlidingWindowCompaction,
    TieredCompaction,
    estimate_token_count,
)
from tests.conftest import try_import
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire


@dataclass
class Tail:
    """A strategy keeping the last two messages, recording each call's focus."""

    calls: list[str | None] = field(default_factory=list[str | None])

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[None]) -> list[ModelMessage]:
        self.calls.append(None)
        return messages[-2:]


@dataclass
class FocusedTail:
    """`Tail`, steerable with `with_focus`."""

    calls: list[str | None] = field(default_factory=list[str | None])
    focus: str | None = None

    def with_focus(self, focus: str) -> FocusedTail:
        return replace(self, focus=focus)

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[None]) -> list[ModelMessage]:
        self.calls.append(self.focus)
        return messages[-2:]


@dataclass
class Script:
    """A `FunctionModel` answering with `responses` in order, recording what each request sent."""

    responses: list[ModelResponse]
    seen: list[list[ModelMessage]] = field(default_factory=list[list[ModelMessage]])
    tools: list[str] = field(default_factory=list[str])

    def model(self) -> FunctionModel:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.seen.append(list(messages))
            self.tools = [tool.name for tool in info.function_tools]
            return self.responses[len(self.seen) - 1]

        return FunctionModel(respond)


def _compact(call_id: str = 'c1', **args: Any) -> ToolCallPart:
    return ToolCallPart('compact_context', args, tool_call_id=call_id)


def _history() -> list[ModelMessage]:
    return [
        ModelRequest.user_text_prompt('explore the repo'),
        ModelResponse(parts=[TextPart('found a lot ' * 50)]),
    ]


def _returns(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'compact_context'
    ]


def _agent(script: Script, *capabilities: Any) -> Agent[None, str]:
    def look() -> str:
        """Look around."""
        return 'looked'

    return Agent(script.model(), deps_type=type(None), tools=[look], capabilities=list(capabilities))


SCHEDULED = 'Compaction scheduled: older history will be compacted before your next turn.'


class TestCompactOnRequest:
    async def test_compacts_before_the_next_request_with_focus(self) -> None:
        strategy = FocusedTail()
        script = Script(
            [
                ModelResponse(parts=[_compact(focus='the plan'), ToolCallPart('look', {}, tool_call_id='l1')]),
                ModelResponse(parts=[TextPart('done')]),
            ]
        )

        result = await _agent(script, CompactOnRequest(strategy)).run('continue', message_history=_history())

        assert script.tools == ['look', 'compact_context']
        assert strategy.calls == ['the plan']
        # Compaction ran before the second request, not inside the tool call: the first request
        # saw the whole history, the second only the call and its return, still paired.
        assert len(script.seen[0]) == 3
        response, returns = script.seen[1]
        assert isinstance(response, ModelResponse)
        assert [part.tool_call_id for part in response.parts if isinstance(part, ToolCallPart)] == ['c1', 'l1']
        assert isinstance(returns, ModelRequest)
        assert {part.tool_call_id for part in returns.parts if isinstance(part, ToolReturnPart)} == {'c1', 'l1'}
        assert _returns(result.all_messages()) == [SCHEDULED]

    async def test_focus_is_ignored_by_a_strategy_that_cannot_honour_it(self) -> None:
        strategy = Tail()
        script = Script([ModelResponse(parts=[_compact(focus='the plan')]), ModelResponse(parts=[TextPart('done')])])

        # Above `min_tokens`, the request is accepted.
        await _agent(script, CompactOnRequest(strategy, min_tokens=1)).run('continue', message_history=_history())

        assert strategy.calls == [None]
        assert len(script.seen[1]) == 2

    async def test_below_min_tokens_is_refused_with_the_reason(self) -> None:
        strategy = Tail()
        script = Script([ModelResponse(parts=[_compact()]), ModelResponse(parts=[TextPart('done')])])

        result = await _agent(script, CompactOnRequest(strategy, min_tokens=10_000)).run(
            'continue', message_history=_history()
        )

        assert strategy.calls == []
        assert len(script.seen[1]) == 5
        [refusal] = _returns(result.all_messages())
        assert refusal.startswith('Not compacted: the history is about ')
        assert refusal.endswith(' tokens, below the minimum of 10000, so compacting would save little.')

    async def test_requests_past_max_per_run_are_refused(self) -> None:
        strategy = Tail()
        script = Script(
            [
                ModelResponse(parts=[_compact('c1')]),
                ModelResponse(parts=[_compact('c2')]),
                ModelResponse(parts=[TextPart('done')]),
            ]
        )
        agent = _agent(script, CompactOnRequest(strategy, max_per_run=1))

        result = await agent.run('continue', message_history=_history())

        assert strategy.calls == [None]
        assert _returns(result.all_messages())[-1] == (
            'Not compacted: the limit of 1 compaction requests for this run has been reached.'
        )

        # The count is per run.
        script.responses += [ModelResponse(parts=[_compact('c3')]), ModelResponse(parts=[TextPart('done')])]
        await agent.run('again')
        assert strategy.calls == [None, None]

    async def test_a_second_call_in_one_response_is_refused(self) -> None:
        strategy = Tail()
        script = Script(
            [ModelResponse(parts=[_compact('c1'), _compact('c2')]), ModelResponse(parts=[TextPart('done')])]
        )

        result = await _agent(script, CompactOnRequest(strategy)).run('continue', message_history=_history())

        assert strategy.calls == [None]
        assert sorted(_returns(result.all_messages())) == [
            SCHEDULED,
            'Not compacted again: a compaction is already scheduled before your next turn.',
        ]

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_stands_down_when_automatic_compaction_already_ran(self) -> None:
        strategy = Tail()
        script = Script(
            [
                ModelResponse(parts=[_compact('c1')]),
                ModelResponse(parts=[_compact('c2')]),
                ModelResponse(parts=[TextPart('done')]),
            ]
        )
        automatic = TieredCompaction(tiers=[SlidingWindowCompaction(max_messages=1, keep_messages=2)], target_tokens=1)

        # Listed first, it still runs after `TieredCompaction`, and sees that one compacted already.
        result = await _agent(script, CompactOnRequest(strategy, max_per_run=1), automatic).run(
            'continue', message_history=_history()
        )

        assert strategy.calls == []
        # A dropped request does not use up `max_per_run`.
        assert _returns(result.all_messages())[-1] == SCHEDULED

    def test_rejects_non_positive_guards(self) -> None:
        with pytest.raises(ValueError, match='min_tokens must be positive'):
            CompactOnRequest(Tail(), min_tokens=0)
        with pytest.raises(ValueError, match='max_per_run must be positive'):
            CompactOnRequest(Tail(), max_per_run=0)


@pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed')
async def test_span_marks_the_compaction_model_requested(capfire: CaptureLogfire) -> None:
    script = Script([ModelResponse(parts=[_compact()]), ModelResponse(parts=[TextPart('done')])])
    agent = _agent(script, CompactOnRequest(Tail(), tokenizer=len))
    agent.instrument = InstrumentationSettings()

    await agent.run('continue', message_history=_history())
    before = [*script.seen[0], *script.seen[1]]

    [span] = [s for s in capfire.exporter.exported_spans_as_dict() if s['name'] == 'compact_messages']
    attributes = span['attributes']
    assert attributes['compaction.strategy'] == 'Tail'
    assert attributes['compaction.model_requested'] is True
    assert attributes['compaction.messages_before'] == 5
    assert attributes['compaction.messages_after'] == 2
    # Measured with the configured tokenizer, not the characters-per-token heuristic.
    assert attributes['compaction.tokens_before'] == estimate_token_count(before, len)
    assert estimate_token_count(before, len) != estimate_token_count(before)
