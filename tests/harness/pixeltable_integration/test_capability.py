"""Tests for the `Pixeltable` capability: configuration, spec loading, merging, and agent runs."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pixeltable as pxt
import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import Memory
from pydantic_ai_harness.pixeltable import Pixeltable, PixeltableToolset

from .support import create_table, insert_rows


def _loaded(agent: Agent[None, str]) -> list[Pixeltable[None]]:
    return [cap for cap in agent.root_capability.capabilities if isinstance(cap, Pixeltable)]


def _narrow(capability: AbstractCapability[None]) -> Pixeltable[None]:
    assert isinstance(capability, Pixeltable)
    return capability


class TestPixeltableCapability:
    def test_tables_is_required(self) -> None:
        with pytest.raises(TypeError, match='tables'):
            Pixeltable()  # pyright: ignore[reportCallIssue]

    def test_star_must_be_the_only_entry(self) -> None:
        with pytest.raises(ValueError, match='only entry'):
            Pixeltable(tables=['*', 'a.b'])

    def test_tables_rejects_a_string(self) -> None:
        with pytest.raises(ValueError, match='list of paths'):
            Pixeltable(tables='my_app.doc_chunks')  # pyright: ignore[reportArgumentType]

    def test_tables_rejects_an_empty_allowlist(self) -> None:
        # None must not silently mean the whole catalog; that is an opt-in via ['*'].
        for tables in ([], ['']):
            with pytest.raises(ValueError, match='non-empty allowlist'):
                Pixeltable(tables=tables)

    def test_tables_rejects_bad_entries(self) -> None:
        for bad in (['my_app.'], [' '], ['a..b'], ['.hidden'], ['a.b c']):
            with pytest.raises(ValueError, match='tables entry'):
                Pixeltable(tables=bad)

    def test_tables_normalizes_slashes_and_case(self) -> None:
        assert Pixeltable(tables=['My_App/Doc_Chunks', '']).tables == ['my_app.doc_chunks']

    def test_caps_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match='max_rows must be at least 1, got 0'):
            Pixeltable(tables=['a'], max_rows=0)
        with pytest.raises(ValueError, match='max_chars must be at least 1, got 0'):
            Pixeltable(tables=['a'], max_chars=0)

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
        assert _narrow(Pixeltable.from_spec(['my_app.doc_chunks'])).tables == ['my_app.doc_chunks']
        spec = {'model': 'test', 'capabilities': [{'Pixeltable': ['my_app.doc_chunks']}]}
        agent = Agent.from_spec(spec, custom_capability_types=[Pixeltable])
        assert _loaded(agent)[0].tables == ['my_app.doc_chunks']
        with pytest.raises(TypeError):
            Pixeltable.from_spec(['a.b'], tables=['c.d'])

    def test_toolset_uses_the_capability_settings(self) -> None:
        toolset = Pixeltable(tables=['a'], id=None).get_toolset()
        assert isinstance(toolset, PixeltableToolset)
        assert toolset.id == 'pixeltable'
        assert sorted(toolset.tools) == ['describe_table', 'list_tables', 'query_table', 'similarity_search']
        assert Pixeltable(tables=['a'], id='catalog').get_toolset().id == 'catalog'

    def test_toolset_requires_an_allowlist(self) -> None:
        with pytest.raises(ValueError, match='allowlist'):
            PixeltableToolset(tables=[], max_rows=10, max_chars=100)


class TestPixeltableCombine:
    def test_combine_intersects_allowlists_and_tightens_caps(self) -> None:
        # tables is an access boundary, so a merge narrows: an entry survives only when every
        # capability covers it ('*' and directory prefixes cover the entries beneath them).
        merged = Pixeltable.combine(
            [
                Pixeltable(tables=['a.b', 'a.c'], max_rows=50, max_chars=100),
                Pixeltable(tables=['a.b'], max_rows=5, max_chars=900),
            ]
        )
        assert merged.tables == ['a.b']
        assert merged.max_rows == 5
        assert merged.max_chars == 100
        assert merged.guidance is None

    def test_combine_star_and_prefixes(self) -> None:
        assert Pixeltable.combine([Pixeltable(tables=['a.b']), Pixeltable(tables=['*'])]).tables == ['a.b']
        assert Pixeltable.combine([Pixeltable(tables=['*']), Pixeltable(tables=['*'])]).tables == ['*']
        assert Pixeltable.combine([Pixeltable(tables=['d']), Pixeltable(tables=['d.t1'])]).tables == ['d.t1']
        slashed = Pixeltable.combine([Pixeltable(tables=['d/t1']), Pixeltable(tables=['d.t1.v'])])
        assert slashed.tables == ['d.t1.v']
        # An entry covered by another surviving entry is redundant and dropped.
        nested = Pixeltable.combine([Pixeltable(tables=['d', 'd.t1']), Pixeltable(tables=['*'])])
        assert nested.tables == ['d']

    def test_combine_rejects_disjoint_allowlists(self) -> None:
        with pytest.raises(ValueError, match='share no allowed tables'):
            Pixeltable.combine([Pixeltable(tables=['a']), Pixeltable(tables=['b'])])

    def test_combine_rejects_other_capabilities(self) -> None:
        with pytest.raises(TypeError, match='only merges other Pixeltable capabilities'):
            Pixeltable.combine([Pixeltable(tables=['a']), Memory()])

    def test_combine_takes_the_latest_settings(self) -> None:
        merged = Pixeltable.combine(
            [
                Pixeltable(tables=['a'], guidance='first'),
                Pixeltable(tables=['a'], guidance='second', description='latest', defer_loading=True),
                Pixeltable(tables=['a']),
            ]
        )
        assert merged.guidance == 'second'
        assert merged.id == 'pixeltable'
        assert merged.description is not None
        assert 'list_tables' in merged.description
        assert merged.defer_loading is False

    def test_agent_merges_same_id_capabilities(self) -> None:
        agent = Agent('test', capabilities=[Pixeltable(tables=['x']), Pixeltable(tables=['x.y'], max_chars=10)])
        caps = _loaded(agent)
        assert len(caps) == 1
        assert caps[0].tables == ['x.y']
        assert caps[0].max_chars == 10


@pytest.fixture
def root() -> Iterator[str]:
    name = f'harness_pxt_cap_{uuid.uuid4().hex[:8]}'
    pxt.create_dir(name)
    chunks = create_table(f'{name}.chunks', {'text': pxt.String, 'status': pxt.String})
    insert_rows(chunks, [{'text': 'cats sit on mats', 'status': 'open'}, {'text': 'dogs run', 'status': 'closed'}])
    create_table(f'{name}.secret', {'text': pxt.String})
    yield name
    pxt.drop_dir(name, force=True)


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart | RetryPromptPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart | RetryPromptPart)
    ]


class TestPixeltableAgent:
    async def test_agent_lists_queries_and_retries_outside_the_allowlist(self, root: str) -> None:
        chunks = f'{root}.chunks'
        calls = [
            ToolCallPart('list_tables', {}, tool_call_id='c1'),
            ToolCallPart('query_table', {'table': f'{root}.secret'}, tool_call_id='c2'),
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
        assert isinstance(refused, RetryPromptPart)
        assert 'not in the Pixeltable allowlist' in str(refused.content)
        assert isinstance(queried, ToolReturnPart)
        assert queried.content == {
            'table': chunks,
            'rows': [{'text': 'cats sit on mats', 'status': 'open'}],
            'truncated': False,
        }
        assert all(f'You may only use these tables or prefixes: {chunks}.' in str(text) for text in seen_instructions)
