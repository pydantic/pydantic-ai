"""Tests for the `Pixeltable` capability: configuration, spec loading, merging, and agent runs."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.conftest import try_import

with try_import() as imports_successful:
    import pixeltable as pxt

    from pydantic_ai_harness.pixeltable import Pixeltable, PixeltableToolset

    from .support import create_table, insert_rows

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pixeltable not installed')


def _loaded(agent: Agent[None, str]) -> list[Pixeltable[None]]:
    return [cap for cap in agent.root_capability.capabilities if isinstance(cap, Pixeltable)]


class TestPixeltableCapability:
    @pytest.mark.parametrize(
        ('kwargs', 'error'),
        [
            ({'tables': ['*', 'a.b']}, 'only entry'),
            ({'tables': 'my_app.doc_chunks'}, 'list of paths'),
            # An empty allowlist must not mean the whole catalog; that is an opt-in via ['*'].
            ({'tables': []}, 'non-empty allowlist'),
            ({'tables': ['']}, 'non-empty allowlist'),
            ({'tables': ['my_app.']}, 'tables entry'),
            ({'tables': ['a..b']}, 'tables entry'),
            ({'tables': ['a.b c']}, 'tables entry'),
            ({'tables': ['a'], 'max_rows': 0}, 'max_rows must be at least 1, got 0'),
            ({'tables': ['a'], 'max_chars': 0}, 'max_chars must be at least 1, got 0'),
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict[str, object], error: str) -> None:
        with pytest.raises(ValueError, match=error):
            Pixeltable(**kwargs)  # pyright: ignore[reportArgumentType]

    def test_toolset_validates_its_configuration_too(self) -> None:
        with pytest.raises(ValueError, match='only entry'):
            PixeltableToolset(tables=['*', 'a'], max_rows=10, max_chars=100)

    def test_toolset_defaults_its_caps_like_the_capability(self) -> None:
        # Building the toolset directly does not require repeating the capability's 20-row and 8000-char caps.
        toolset = PixeltableToolset(tables=['a'])
        assert sorted(toolset.tools) == ['describe_table', 'list_tables', 'query_table', 'similarity_search']

    def test_tables_are_normalized_sorted_and_deduplicated(self) -> None:
        assert Pixeltable(tables=['My_App/Doc_Chunks', '', 'b', 'b']).tables == ['b', 'my_app.doc_chunks']

    def test_instructions_name_the_allowlist(self) -> None:
        scoped = Pixeltable(tables=['my_app.doc_chunks', 'other']).get_instructions()
        assert scoped is not None
        assert scoped.endswith('You may only use these tables or prefixes: my_app.doc_chunks, other.')
        whole = Pixeltable(tables=['*']).get_instructions()
        assert whole is not None
        assert 'list_tables' in whole
        assert 'You may only use' not in whole

    def test_guidance_replaces_or_removes_instructions(self) -> None:
        assert Pixeltable(tables=['a'], guidance='Use the tables.').get_instructions() == 'Use the tables.'
        assert Pixeltable(tables=['a'], guidance='').get_instructions() is None

    def test_from_spec_positional_shorthand(self) -> None:
        # {"Pixeltable": ["dir.tbl"]} passes the allowlist as the single positional argument.
        spec = {'model': 'test', 'capabilities': [{'Pixeltable': ['my_app.doc_chunks']}]}
        agent = Agent.from_spec(spec, custom_capability_types=[Pixeltable])
        assert _loaded(agent)[0].tables == ['my_app.doc_chunks']

    def test_toolset_uses_the_capability_settings(self) -> None:
        toolset = Pixeltable(tables=['a'], id=None).get_toolset()
        assert isinstance(toolset, PixeltableToolset)
        assert toolset.id == 'pixeltable'
        assert sorted(toolset.tools) == ['describe_table', 'list_tables', 'query_table', 'similarity_search']
        assert Pixeltable(tables=['a'], id='catalog').get_toolset().id == 'catalog'


class TestPixeltableCombine:
    def test_same_configuration_combines(self) -> None:
        agent = Agent('test', capabilities=[Pixeltable(tables=['X/Y']), Pixeltable(tables=['x.y'])])
        assert [cap.tables for cap in _loaded(agent)] == [['x.y']]

    def test_reordered_allowlists_combine(self) -> None:
        agent = Agent('test', capabilities=[Pixeltable(tables=['a', 'b']), Pixeltable(tables=['b', 'a'])])
        assert [cap.tables for cap in _loaded(agent)] == [['a', 'b']]

    def test_different_allowlists_raise(self) -> None:
        # `tables` is an access boundary, so two that disagree need their own ids rather than a merge.
        with pytest.raises(UserError, match="disagree on 'tables'"):
            Agent('test', capabilities=[Pixeltable(tables=['x']), Pixeltable(tables=['x.y'])])


@pytest.fixture
def catalog(root: str) -> str:
    chunks = create_table(f'{root}.chunks', {'text': pxt.String, 'status': pxt.String})
    insert_rows(chunks, [{'text': 'cats sit on mats', 'status': 'open'}, {'text': 'dogs run', 'status': 'closed'}])
    create_table(f'{root}.secret', {'text': pxt.String})
    return root


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


class TestPixeltableAgent:
    async def test_agent_lists_queries_and_retries_outside_the_allowlist(self, catalog: str) -> None:
        chunks = f'{catalog}.chunks'
        calls = [
            ToolCallPart('list_tables', {}, tool_call_id='c1'),
            ToolCallPart('query_table', {'table': f'{catalog}.secret'}, tool_call_id='c2'),
            ToolCallPart('query_table', {'table': chunks, 'where': {'status': 'open'}}, tool_call_id='c3'),
        ]
        seen_instructions: list[str | None] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen_instructions.append(info.instructions)
            returns = _tool_returns(messages)
            if len(returns) < len(calls):
                return ModelResponse(parts=[calls[len(returns)]])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(model), capabilities=[Pixeltable(tables=[chunks])])
        result = await agent.run('Which chunks are open?')

        listed, refused, queried = _tool_returns(result.all_messages())
        assert isinstance(listed, ToolReturnPart)
        assert listed.content == {'tables': [chunks]}
        assert isinstance(refused, ToolReturnPart)
        assert refused.outcome == 'retried'
        assert 'not in the Pixeltable allowlist' in str(refused.content)
        assert isinstance(queried, ToolReturnPart)
        assert queried.content == {
            'table': chunks,
            'rows': [{'text': 'cats sit on mats', 'status': 'open'}],
            'truncated': False,
        }
        assert all(f'You may only use these tables or prefixes: {chunks}.' in str(text) for text in seen_instructions)
