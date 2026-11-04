"""`ClearToolResults` targeting: a `tools` allowlist, `keep_turns`, and a callable `placeholder`."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ToolReturn
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchReturnContent,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import ClearToolResults, DeduplicateFileReads, pin
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY

CLEARED = '[tool result cleared]'


def _first_line(part: ToolReturnPart) -> str:
    return str(part.content).partition('\n')[0]


def _call(tool_name: str, tool_call_id: str, args: str = '{}') -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name, args, tool_call_id=tool_call_id)])


def _return(tool_name: str, tool_call_id: str, content: str = 'result', metadata: object = None) -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name, content, tool_call_id=tool_call_id, metadata=metadata)])


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(text)])


def _turn(prompt: str, *tool_call_ids: str, tool_name: str = 'search') -> list[ModelMessage]:
    """A user turn in which the model calls `tool_name` once per id, then answers."""
    messages: list[ModelMessage] = [_user(prompt)]
    for tool_call_id in tool_call_ids:
        messages += [_call(tool_name, tool_call_id), _return(tool_name, tool_call_id, f'{tool_call_id} result')]
    return [*messages, ModelResponse(parts=[TextPart('answer')])]


def _returns(messages: list[ModelMessage]) -> dict[str, ToolReturnPart]:
    return {
        part.tool_call_id: part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }


def _contents(messages: list[ModelMessage]) -> dict[str, str]:
    return {tool_call_id: str(part.content) for tool_call_id, part in _returns(messages).items()}


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def _recording_agent(capability: ClearToolResults[None]) -> tuple[Agent[None, str], list[list[ModelMessage]]]:
    """An agent that answers at once and records the history each request sent."""
    sent: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(list(messages))
        return ModelResponse(parts=[TextPart('done')])

    return Agent(FunctionModel(respond), deps_type=type(None), capabilities=[capability]), sent


class TestClearToolResultsTargeting:
    def test_tools_and_exclude_tools_are_exclusive(self):
        with pytest.raises(UserError, match='Set at most one of tools or exclude_tools'):
            ClearToolResults(max_tokens=1, tools=['search'], exclude_tools=frozenset({'fetch'}))

    def test_tools_rejects_a_bare_string(self):
        # A `str` is a `Sequence[str]`, so `tools='search'` typechecks but would test substrings.
        with pytest.raises(UserError, match='tools takes a sequence of tool names'):
            ClearToolResults(max_tokens=1, tools='search')

    def test_keep_pairs_and_keep_turns_are_exclusive(self):
        with pytest.raises(UserError, match='Set at most one of keep_pairs or keep_turns'):
            ClearToolResults(max_tokens=1, keep_pairs=2, keep_turns=1)

    async def test_keep_turns_replaces_the_default_pair_count(self):
        # An explicit `keep_pairs=3` can't be told apart from the default, so `keep_turns` wins.
        messages = [*_turn('one', 'a', 'b', 'c', 'd'), _user('two')]
        cap = ClearToolResults[None](max_tokens=1, keep_pairs=3, keep_turns=0)
        assert _contents(await cap.compact(messages, _ctx())) == dict.fromkeys('abcd', CLEARED)

    def test_negative_keep_turns(self):
        with pytest.raises(ValueError, match='keep_turns must be non-negative'):
            ClearToolResults(max_tokens=1, keep_turns=-1)

    async def test_default_keeps_three_pairs(self):
        messages = _turn('go', 'a', 'b', 'c', 'd')
        cleared = await ClearToolResults[None](max_tokens=1).compact(messages, _ctx())
        assert _contents(cleared) == {'a': CLEARED, 'b': 'b result', 'c': 'c result', 'd': 'd result'}

    async def test_tools_clears_only_the_listed_tools(self):
        messages = [*_turn('go', 'a'), *_turn('more', 'b', tool_name='fetch')]
        cleared = await ClearToolResults[None](max_tokens=1, keep_pairs=0, tools=['search']).compact(messages, _ctx())
        assert _contents(cleared) == {'a': CLEARED, 'b': 'b result'}

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_first_line_placeholder_on_earlier_turns(self):
        """The issue's case: earlier turns' search results shrink to their first line, so their ids survive."""
        sent: list[list[ModelMessage]] = []
        placeholder_calls: list[str] = []

        def placeholder(part: ToolReturnPart) -> str:
            placeholder_calls.append(part.tool_call_id)
            return _first_line(part)

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            sent.append(list(messages))
            last = messages[-1]
            assert isinstance(last, ModelRequest)
            if isinstance(last.parts[-1], UserPromptPart):
                turn = len(
                    [m for m in messages if isinstance(m, ModelRequest) and isinstance(m.parts[-1], UserPromptPart)]
                )
                return ModelResponse(
                    parts=[
                        ToolCallPart('search', {'query': f'q{turn}'}, tool_call_id=f'search_{turn}'),
                        ToolCallPart('profile', {}, tool_call_id=f'profile_{turn}'),
                    ]
                )
            return ModelResponse(parts=[TextPart('answer')])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[ClearToolResults(max_tokens=1, tools=['search'], keep_turns=0, placeholder=placeholder)],
        )

        @agent.tool_plain
        def search(query: str) -> str:
            return f'id=doc-{query} title=Result for {query}\n' + 'body line\n' * 20

        @agent.tool_plain
        def profile() -> str:
            return 'name=Ada\nrole=admin'

        history = (await agent.run('first')).all_messages()
        # Nothing in the current turn is cleared, whatever the trigger says.
        assert placeholder_calls == []
        assert _first_line(_returns(sent[-1])['search_1']) == 'id=doc-q1 title=Result for q1'
        assert str(_returns(sent[-1])['search_1'].content).count('\n') == 21

        sent.clear()
        history = (await agent.run('second', message_history=history)).all_messages()
        assert _contents(sent[-1]) == snapshot(
            {
                'search_1': 'id=doc-q1 title=Result for q1',
                'profile_1': 'name=Ada\nrole=admin',
                'search_2': 'id=doc-q2 title=Result for q2\n' + 'body line\n' * 20,
                'profile_2': 'name=Ada\nrole=admin',
            }
        )
        # Cleared once, at the turn's first request: the turn's later request extends that history.
        assert sent[1][: len(sent[0])] == sent[0]
        assert placeholder_calls == ['search_1']

        sent.clear()
        await agent.run('third', message_history=history)
        # The already-cleared result isn't passed to the placeholder again.
        assert placeholder_calls == ['search_1', 'search_2']
        assert _contents(sent[-1])['search_1'] == 'id=doc-q1 title=Result for q1'

    async def test_callable_placeholder_runs_once_per_result(self):
        calls: list[str] = []

        def placeholder(part: ToolReturnPart) -> str:
            calls.append(part.tool_call_id)
            return f'[{part.tool_call_id} cleared]'

        cap = ClearToolResults[None](max_tokens=1, keep_pairs=0, placeholder=placeholder, clear_tool_inputs=True)
        messages = _turn('go', 'a', 'b')
        once = await cap.compact(messages, _ctx())
        twice = await cap.compact(once, _ctx())
        assert calls == ['a', 'b']
        assert twice == once
        assert _contents(twice) == {'a': '[a cleared]', 'b': '[b cleared]'}

    async def test_callable_placeholder_skips_results_that_cannot_be_marked(self):
        """A result whose metadata can't hold the clear mark would be cleared again from its own placeholder."""
        calls: list[str] = []

        def placeholder(part: ToolReturnPart) -> str:
            calls.append(part.tool_call_id)
            return 'cleared'

        search_content: ToolSearchReturnContent = {'discovered_tools': [{'name': 'alpha'}]}
        messages: list[ModelMessage] = [
            _user('go'),
            _call('fetch', 'opaque'),
            _return('fetch', 'opaque', metadata='opaque metadata'),
            _call('fetch', 'tagged'),
            _return('fetch', 'tagged', metadata={'source': 'cache'}),
            _call('search_tools', 'found'),
            ModelRequest(parts=[ToolSearchReturnPart(content=search_content, tool_call_id='found')]),
        ]
        cleared = await ClearToolResults[None](max_tokens=1, keep_pairs=0, placeholder=placeholder).compact(
            messages, _ctx()
        )
        returns = _returns(cleared)
        assert calls == ['tagged']
        assert returns['opaque'].content == 'result'
        assert returns['found'].content == search_content
        assert returns['tagged'].content == 'cleared'
        assert returns['tagged'].metadata == {
            'source': 'cache',
            'pydantic-ai-harness.compaction.cleared-tool-result.v1': True,
            'pydantic-ai-harness.compaction.cleared-by-placeholder-function.v1': True,
        }

        # A string placeholder clears the opaque one, as it always has.
        cleared = await ClearToolResults[None](max_tokens=1, keep_pairs=0).compact(messages, _ctx())
        assert _returns(cleared)['opaque'].content == CLEARED

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_callable_placeholder_with_max_messages_clears_in_batches(self):
        """`max_messages` counts from the last result the callable cleared, found by its mark."""
        sent: list[list[ModelMessage]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            sent.append(list(messages))
            step = len(sent)
            if step > 11:
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[ToolCallPart('lookup', {'q': step}, tool_call_id=f'c{step}')])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[
                ClearToolResults(max_messages=8, keep_pairs=2, placeholder=lambda part: f'[{part.tool_call_id}]')
            ],
        )

        @agent.tool_plain
        def lookup(q: int) -> str:
            return f'result {q}'

        await agent.run('go')

        rewritten = [i for i in range(1, len(sent)) if sent[i][: len(sent[i - 1])] != sent[i - 1]]
        assert rewritten == [4, 7, 10]
        assert list(_contents(sent[-1]).values())[-4:] == ['[c8]', 'result 9', 'result 10', 'result 11']

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_callable_placeholder_max_messages_ignores_other_strategies_clears(self):
        """`DeduplicateFileReads` marks the reads it clears too; those must not reset the `max_messages` count."""
        sent: list[list[ModelMessage]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            sent.append(list(messages))
            step = len(sent)
            if step > 12:
                return ModelResponse(parts=[TextPart('done')])
            if step % 2:
                return ModelResponse(parts=[ToolCallPart('lookup', {'q': step}, tool_call_id=f'c{step}')])
            return ModelResponse(parts=[ToolCallPart('read', {'path': 'a.py'}, tool_call_id=f'r{step}')])

        def file_key(call: ToolCallPart) -> str | None:
            return 'a.py' if call.tool_name == 'read' else None

        agent = Agent(
            FunctionModel(respond),
            capabilities=[
                DeduplicateFileReads(file_key=file_key),
                ClearToolResults(max_messages=8, keep_pairs=2, placeholder=lambda part: f'[{part.tool_call_id}]'),
            ],
        )

        @agent.tool_plain
        def lookup(q: int) -> str:
            return f'result {q}'

        @agent.tool_plain
        def read(path: str) -> str:
            return f'contents of {path}'

        await agent.run('go')

        contents = _contents(sent[-1])
        assert [contents[f'c{step}'] for step in (1, 3, 5)] == ['[c1]', '[c3]', '[c5]']

    async def test_keep_turns_keeps_the_last_completed_turns_and_the_current_one(self):
        messages = [*_turn('one', 'a'), *_turn('two', 'b'), *_turn('three', 'c'), _user('four')]
        cleared = await ClearToolResults[None](max_tokens=1, keep_turns=1).compact(messages, _ctx())
        assert _contents(cleared) == {'a': CLEARED, 'b': CLEARED, 'c': 'c result'}

    async def test_keep_turns_never_clears_the_current_turn(self):
        messages = [*_turn('one', 'a'), _user('two'), _call('search', 'b'), _return('search', 'b', 'b result')]
        cleared = await ClearToolResults[None](max_tokens=1, keep_turns=0).compact(messages, _ctx())
        assert _contents(cleared) == {'a': CLEARED, 'b': 'b result'}

    async def test_keep_turns_with_fewer_turns_or_none_clears_nothing(self):
        cap = ClearToolResults[None](max_tokens=1, keep_turns=2)
        messages = [*_turn('one', 'a'), *_turn('two', 'b')]
        assert await cap.compact(messages, _ctx()) is messages
        # Without a user prompt the whole history is the current turn.
        no_prompt: list[ModelMessage] = [_call('search', 'a'), _return('search', 'a')]
        assert await ClearToolResults[None](max_tokens=1, keep_turns=0).compact(no_prompt, _ctx()) is no_prompt

    async def test_keep_turns_ignores_harness_prompts_between_turns(self):
        """A pinned message is user-role content a harness capability added: it doesn't start a turn."""
        messages = [*_turn('one', 'a'), *_turn('two', 'b'), ModelRequest(parts=[pin('Remember this.')]), _user('three')]
        cleared = await ClearToolResults[None](max_tokens=1, keep_turns=1).compact(messages, _ctx())
        assert _contents(cleared) == {'a': CLEARED, 'b': 'b result'}

    async def test_keep_turns_does_not_reach_a_reused_tool_call_id_in_a_kept_turn(self):
        messages = [*_turn('one', 'same'), _user('two'), _call('search', 'same'), _return('search', 'same', 'new')]
        cleared = await ClearToolResults[None](max_tokens=1, keep_turns=0, clear_tool_inputs=True).compact(
            messages, _ctx()
        )
        returns = [
            str(part.content)
            for message in cleared
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert returns == [CLEARED, 'new']

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_keep_turns_ignores_user_content_a_tool_returns(self):
        """A `ToolReturn` with `content` sends it as a user prompt next to the result, inside the current turn."""
        sent: list[list[ModelMessage]] = []
        script = iter(['search', 'screenshot', 'search', None])

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            sent.append(list(messages))
            tool_name = next(script)
            if tool_name is None:
                return ModelResponse(parts=[TextPart('answer')])
            return ModelResponse(parts=[ToolCallPart(tool_name, {}, tool_call_id=f'{tool_name}_{len(sent)}')])

        agent = Agent(FunctionModel(respond), capabilities=[ClearToolResults(max_tokens=1, keep_turns=0)])

        @agent.tool_plain
        def search() -> str:
            return 'found'

        @agent.tool_plain
        def screenshot() -> ToolReturn:
            return ToolReturn('captured', content=['what the screen shows'])

        await agent.run('two', message_history=_turn('one', 'earlier'))
        assert _contents(sent[-1]) == {
            'earlier': CLEARED,
            'search_1': 'found',
            'screenshot_2': 'captured',
            'search_3': 'found',
        }

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_keep_turns_on_a_resumed_history_counts_its_last_turn_as_current(self):
        """A run resumed without a new prompt answers the history's last turn, so that turn is kept."""
        agent, sent = _recording_agent(ClearToolResults(max_tokens=1, keep_turns=0))
        history = [*_turn('one', 'a'), _user('two'), _call('search', 'b'), _return('search', 'b', 'b result')]
        await agent.run(message_history=history)
        assert _contents(sent[0]) == {'a': CLEARED, 'b': 'b result'}

    @pytest.mark.parametrize(
        ('kwargs', 'expected'),
        [
            pytest.param({'keep_turns': 0}, {'a': CLEARED, 'b': 'b result'}, id='keep_turns'),
            pytest.param({'tools': ['fetch']}, {'a': 'a result', 'b': 'b result'}, id='tools'),
        ],
    )
    async def test_loads_from_an_agent_spec(self, kwargs: dict[str, object], expected: dict[str, str]):
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'ClearToolResults': {'max_tokens': 1, **kwargs}}]},
            custom_capability_types=[ClearToolResults],
        )
        [cap] = [c for c in agent.root_capability.capabilities if isinstance(c, ClearToolResults)]
        messages = [*_turn('one', 'a'), *_turn('two', 'b')]
        assert _contents(await cap.compact(messages, _ctx())) == expected

    def test_agent_spec_accepts_only_a_string_placeholder(self):
        params = AgentSpec.model_json_schema_with_capabilities([ClearToolResults])['$defs'][
            'spec_params_ClearToolResults'
        ]['properties']
        assert params['placeholder'] == snapshot({'title': 'Placeholder', 'type': 'string'})
        assert {'tools', 'keep_turns', 'keep_pairs', 'exclude_tools'} <= set(params)
