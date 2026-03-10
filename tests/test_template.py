"""Tests for TemplateStr and template resolution in Agent.from_spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import TypeAdapter

pytest.importorskip('pydantic_handlebars', reason='pydantic-handlebars not installed')

from pydantic_ai import Agent, TemplateStr
from pydantic_ai._run_context import RunContext
from pydantic_ai._template import validate_from_spec_args
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.instructions import Instructions
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import RunUsage


@dataclass
class MyDeps:
    name: str
    age: int = 25


def _make_run_context(deps: Any) -> RunContext[Any]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage(), prompt=None, messages=[])


# --- TemplateStr construction and rendering ---


class TestTemplateStr:
    def test_construct_with_deps_type(self) -> None:
        t = TemplateStr('Hello {{name}}', deps_type=MyDeps)
        assert repr(t) == "TemplateStr('Hello {{name}}')"
        assert str(t) == 'Hello {{name}}'

    def test_construct_with_deps_schema(self) -> None:
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        t = TemplateStr('Hello {{name}}', deps_schema=schema)
        assert repr(t) == "TemplateStr('Hello {{name}}')"

    def test_construct_without_deps(self) -> None:
        t = TemplateStr('Hello {{name}}')
        assert repr(t) == "TemplateStr('Hello {{name}}')"

    def test_render_with_typed_deps(self) -> None:
        t = TemplateStr('Hello {{name}}, age {{age}}', deps_type=MyDeps)
        ctx = _make_run_context(MyDeps(name='Alice', age=30))
        assert t(ctx) == 'Hello Alice, age 30'

    def test_render_with_schema_deps(self) -> None:
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        t = TemplateStr('Hello {{name}}', deps_schema=schema)
        ctx = _make_run_context(MyDeps(name='Bob'))
        assert t(ctx) == 'Hello Bob'

    def test_render_without_deps(self) -> None:
        t = TemplateStr('Hello {{name}}')
        ctx = _make_run_context({'name': 'Charlie'})
        assert t(ctx) == 'Hello Charlie'

    def test_render_no_deps_no_placeholders(self) -> None:
        t = TemplateStr('Hello world')
        ctx = _make_run_context(None)
        assert t(ctx) == 'Hello world'

    def test_invalid_field_with_deps_type(self) -> None:
        with pytest.raises(Exception):
            TemplateStr('Hello {{nonexistent}}', deps_type=MyDeps)

    def test_invalid_field_with_deps_schema(self) -> None:
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        with pytest.raises(Exception):
            TemplateStr('Hello {{nonexistent}}', deps_schema=schema)


# --- Pydantic validation (__get_pydantic_core_schema__) ---


TemplateStrOrStr = TemplateStr[Any] | str


class TestTemplateStrPydanticValidation:
    def test_template_string_becomes_template_str(self) -> None:
        ta: TypeAdapter[TemplateStrOrStr] = TypeAdapter(TemplateStrOrStr)
        result = ta.validate_python('Hello {{name}}', context={'deps_type': MyDeps})
        assert isinstance(result, TemplateStr)
        assert repr(result) == "TemplateStr('Hello {{name}}')"

    def test_plain_string_stays_str(self) -> None:
        ta: TypeAdapter[TemplateStrOrStr] = TypeAdapter(TemplateStrOrStr)
        result = ta.validate_python('Hello world', context={'deps_type': MyDeps})
        assert isinstance(result, str)
        assert result == 'Hello world'

    def test_template_str_passthrough(self) -> None:
        ta: TypeAdapter[TemplateStrOrStr] = TypeAdapter(TemplateStrOrStr)
        original = TemplateStr('Hello {{name}}', deps_type=MyDeps)
        result = ta.validate_python(original, context={'deps_type': MyDeps})
        assert result is original

    def test_list_of_templates_and_strings(self) -> None:
        ta: TypeAdapter[list[TemplateStrOrStr]] = TypeAdapter(list[TemplateStrOrStr])
        result = ta.validate_python(['Hello {{name}}', 'plain text'], context={'deps_type': MyDeps})
        assert isinstance(result[0], TemplateStr)
        assert isinstance(result[1], str)

    def test_serialization_round_trip(self) -> None:
        ta: TypeAdapter[TemplateStrOrStr] = TypeAdapter(TemplateStrOrStr)
        t = ta.validate_python('Hello {{name}}', context={'deps_type': MyDeps})
        serialized = ta.dump_python(t)
        assert serialized == 'Hello {{name}}'

    def test_without_context(self) -> None:
        """TemplateStr validation without context still compiles (no deps_type)."""
        ta: TypeAdapter[TemplateStrOrStr] = TypeAdapter(TemplateStrOrStr)
        result = ta.validate_python('Hello {{name}}')
        assert isinstance(result, TemplateStr)


# --- validate_from_spec_args ---


class TestValidateFromSpecArgs:
    def test_resolves_template_in_positional_arg(self) -> None:
        ctx: dict[str, Any] = {'deps_type': MyDeps}
        args, _kwargs = validate_from_spec_args(Instructions, ('Hello {{name}}',), {}, ctx)
        assert len(args) == 1
        assert isinstance(args[0], TemplateStr)

    def test_resolves_template_in_keyword_arg(self) -> None:
        ctx: dict[str, Any] = {'deps_type': MyDeps}
        _args, kwargs = validate_from_spec_args(Instructions, (), {'instructions': 'Hello {{name}}'}, ctx)
        assert isinstance(kwargs['instructions'], TemplateStr)

    def test_plain_string_unchanged(self) -> None:
        ctx: dict[str, Any] = {'deps_type': MyDeps}
        args, _kwargs = validate_from_spec_args(Instructions, ('Hello world',), {}, ctx)
        assert args == ('Hello world',)
        assert isinstance(args[0], str)
        assert not isinstance(args[0], TemplateStr)

    def test_no_template_str_in_hints_is_noop(self) -> None:
        """Capabilities without TemplateStr in from_spec hints are unchanged."""

        @dataclass
        class PlainCap(AbstractCapability[None]):
            value: str = ''

            @classmethod
            def from_spec(cls, value: str = '') -> PlainCap:
                return cls(value=value)

        ctx: dict[str, Any] = {'deps_type': MyDeps}
        args, _kwargs = validate_from_spec_args(PlainCap, ('Hello {{name}}',), {}, ctx)
        # Should not be converted since the hint is `str`, not `TemplateStr | str`
        assert args == ('Hello {{name}}',)
        assert isinstance(args[0], str)

    def test_with_deps_schema(self) -> None:
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        ctx: dict[str, Any] = {'deps_schema': schema}
        args, _kwargs = validate_from_spec_args(Instructions, ('Hello {{name}}',), {}, ctx)
        assert isinstance(args[0], TemplateStr)


# --- Agent.from_spec with deps_type / deps_schema ---


class TestAgentFromSpecDeps:
    def test_from_spec_with_deps_type(self) -> None:
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'Instructions': 'Hello {{name}}'}]},
            deps_type=MyDeps,
        )
        assert agent.model is not None

    def test_from_spec_with_deps_schema(self) -> None:
        agent = Agent.from_spec(
            {
                'model': 'test',
                'deps_schema': {
                    'type': 'object',
                    'properties': {'name': {'type': 'string'}},
                    'required': ['name'],
                },
                'capabilities': [{'Instructions': 'Hello {{name}}'}],
            },
        )
        assert agent.model is not None

    def test_from_spec_plain_string_no_deps(self) -> None:
        """Plain strings work without deps_type."""
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'Instructions': 'Hello world'}]},
        )
        assert agent.model is not None

    def test_from_spec_template_instructions_stored(self) -> None:
        """Template instructions are stored as TemplateStr, not plain strings."""
        agent: Agent[Any, str] = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'Instructions': 'Hello {{name}}'}]},
            deps_type=MyDeps,
        )
        # The agent normalizes instructions into _instructions list
        assert len(agent._instructions) == 1  # pyright: ignore[reportPrivateUsage]
        assert isinstance(agent._instructions[0], TemplateStr)  # pyright: ignore[reportPrivateUsage]

    def test_from_spec_without_deps_type_returns_agent_none(self) -> None:
        """Without deps_type, from_spec returns Agent[None, str]."""
        agent = Agent.from_spec({'model': 'test'})
        assert agent._deps_type is type(None)  # pyright: ignore[reportPrivateUsage]

    def test_from_spec_with_deps_type_sets_deps(self) -> None:
        agent: Agent[Any, str] = Agent.from_spec({'model': 'test'}, deps_type=MyDeps)
        assert agent._deps_type is MyDeps  # pyright: ignore[reportPrivateUsage]


# --- Integration: full agent run with templated instructions ---

pytestmark = [pytest.mark.anyio]


async def test_agent_run_with_template_instructions() -> None:
    """Full integration: run an agent with templated instructions and verify they render."""
    agent: Agent[MyDeps, str] = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'Instructions': 'You are helping {{name}}, age {{age}}.'}]},
        deps_type=MyDeps,
    )
    result = await agent.run('hi', deps=MyDeps(name='Alice', age=30))
    # The rendered instructions should appear in the first model request
    first_request = result.all_messages()[0]
    assert isinstance(first_request, ModelRequest)
    assert first_request.instructions == 'You are helping Alice, age 30.'
