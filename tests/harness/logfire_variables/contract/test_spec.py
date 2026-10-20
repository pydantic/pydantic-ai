"""The cross-language vectors, run against this core.

Everything that has to give the same answer in Python and TypeScript, because a Logfire project is
shared and the SDKs are not: which variable an agent's config lives in, what a code baseline says,
what a published value parses to, and what applying one does to a request. `spec/` is where those
answers live, and this is the half of the agreement this package keeps. A change to a rule changes
the file first, and both cores after.

This is a copy of the suite in logfire's `tests/agent_control/`, run against Pydantic AI's copy of
the contract in `pydantic_ai_harness.logfire._contract`. `spec/` is copied whole, and pinned by digest
below, so a rule cannot change here without changing everywhere. The merge vectors belong to a helper
this copy does not carry and are only digested.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import pytest

from pydantic_ai_harness.logfire._contract import (
    AgentConfig,
    AgentSupport,
    ApplyIssue,
    Block,
    Destination,
    Section,
    ToolDef,
    agent_variable_name,
    apply_instructions,
    apply_settings,
    apply_tool_definitions,
    build_baseline,
)
from pydantic_ai_harness.logfire._contract._reporting import reset_warned_messages

from .conftest import SPEC, expand_repeats

# A distinctive fragment of each warning, mapped to the code the vectors name it by. Both cores word
# their warnings for their own users; what has to match is which decision was taken.
CODES: list[tuple[str, str]] = [
    ('selects invalid model', 'invalid-model'),
    ('Managed instructions section contains', 'instructions-section-too-long'),
    ("instructions section is invalid -- instructions=''", 'instructions-section-empty'),
    ('instructions section has invalid container', 'instructions-invalid-container'),
    ('Managed instruction entry contains', 'instruction-entry-too-long'),
    ('has neither an `id` to address nor text to add', 'instruction-entry-empty'),
    ('Managed instruction entry', 'instruction-entry-invalid'),
    ('settings section has invalid container', 'settings-invalid-container'),
    ('which this version of the SDK does not recognize', 'setting-value-not-recognized'),
    ('has invalid value', 'setting-value-invalid'),
    ('tool definitions section has invalid container', 'tool-definitions-invalid-container'),
    ('tool definition override', 'tool-definition-invalid'),
    ('The agent runs with a request timeout', 'baseline-timeout-not-representable'),
    ('The agent runs with', 'baseline-value-not-describable'),
]


def code_of(message: str) -> str:
    for fragment, code in CODES:
        if fragment in message:
            return code
    raise AssertionError(f'no spec warning code covers {message!r}')  # pragma: no cover


@pytest.fixture(autouse=True)
def _record_warnings() -> Any:
    """Collect warnings rather than raising them, since half of every vector is what was warned."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield caught


def test_the_baseline_vectors(baseline_vectors: list[dict[str, Any]], _record_warnings: Any) -> None:
    caught: list[warnings.WarningMessage] = _record_warnings
    for vector in baseline_vectors:
        # Each vector is its own process as far as the once-per-process warning guard is concerned.
        reset_warned_messages()
        del caught[:]
        data: dict[str, Any] = vector['input']
        blocks = [
            Block(text=block['text'], id=block.get('id'), dynamic=block.get('dynamic', False))
            for block in data.get('blocks', [])
        ]
        tools = [
            ToolDef(
                name=tool['name'],
                description=tool.get('description'),
                parameters_json_schema=tool.get('parameters_json_schema', {}),
                toolset=tool.get('toolset'),
            )
            for tool in data.get('tools', [])
        ]
        baseline = build_baseline(
            instructions=blocks, model=data.get('model'), settings=data.get('settings'), tools=tools
        )
        assert baseline.model_dump(exclude_none=True) == vector['expected'], vector['name']
        assert [code_of(str(warning.message)) for warning in caught] == vector.get('warnings', []), vector['name']


def test_the_config_parsing_vectors(config_parsing_vectors: list[dict[str, Any]], _record_warnings: Any) -> None:
    caught: list[warnings.WarningMessage] = _record_warnings
    for vector in config_parsing_vectors:
        reset_warned_messages()
        del caught[:]
        config = AgentConfig.model_validate(expand_repeats(vector['input']))
        assert config.model_dump(exclude_none=True) == expand_repeats(vector['expected']), vector['name']
        assert [code_of(str(warning.message)) for warning in caught] == vector['warnings'], vector['name']
        if 'unrecognized_settings' in vector:
            assert config.settings is not None
            assert list(config.settings.unrecognized) == vector['unrecognized_settings'], vector['name']


def support_of(data: Any) -> AgentSupport | None:
    """The `support` an apply vector declares, or `None` for an adapter that declares nothing.

    Read here rather than in each vector's assertions because it is the same three lines in every
    apply file, and because the JSON is the wire form -- arrays, `accepts_additions` -- while the
    core's own is a frozenset of `Section` and a tuple of `Destination`.
    """
    if data is None:
        return None
    sections: list[Section] = data['sections']
    return AgentSupport(
        sections=frozenset(sections),
        destinations=tuple(
            Destination(
                id=destination['id'],
                default=destination.get('default', False),
                accepts_additions=destination.get('accepts_additions', True),
            )
            for destination in data.get('destinations', [])
        ),
        settings=frozenset(data.get('settings', [])),
    )


def issues_as_paths(issues: Sequence[ApplyIssue]) -> list[dict[str, Any]]:
    """Every issue as the fields it actually set, which is what the vectors compare.

    The message is left out on purpose: each core words it for its own users, and what has to match
    across the two is the decision and the path to what it was about.
    """
    return [
        {name: value for name, value in asdict(issue).items() if name != 'message' and value is not None}
        for issue in issues
    ]


def parts_of(parts: list[dict[str, Any]]) -> list[Block]:
    return [Block(text=part['text'], id=part.get('id'), dynamic=part.get('dynamic', False)) for part in parts]


def parts_as_json(blocks: Sequence[Block]) -> list[dict[str, Any]]:
    """Each part as the vectors write one: `id` even when absent, `dynamic` only when it is set."""
    return [{'id': block.id, 'text': block.text, **({'dynamic': True} if block.dynamic else {})} for block in blocks]


def tools_of(tools: list[dict[str, Any]]) -> list[ToolDef]:
    return [
        ToolDef(
            name=tool['name'],
            description=tool.get('description'),
            parameters_json_schema=tool.get('parameters_json_schema', {}),
            toolset=tool.get('toolset'),
        )
        for tool in tools
    ]


def tools_as_json(tools: Sequence[ToolDef]) -> list[dict[str, Any]]:
    """Each tool as the vectors write one, leaving out what it does not carry."""
    return [
        {
            'name': tool.name,
            **({'description': tool.description} if tool.description is not None else {}),
            **({'parameters_json_schema': tool.parameters_json_schema} if tool.parameters_json_schema else {}),
            **({'toolset': tool.toolset} if tool.toolset is not None else {}),
        }
        for tool in tools
    ]


def test_the_instructions_apply_vectors(instructions_apply_vectors: list[dict[str, Any]]) -> None:
    for vector in instructions_apply_vectors:
        data: dict[str, Any] = expand_repeats(vector['input'])
        expected: dict[str, Any] = expand_repeats(vector['expected'])
        applied = apply_instructions(parts_of(data.get('parts', [])), AgentConfig.model_validate(data['config']))
        assert parts_as_json(applied.blocks) == expected.get('parts', []), vector['name']
        assert issues_as_paths(applied.issues) == expected['issues'], vector['name']


def test_the_tools_apply_vectors(tools_apply_vectors: list[dict[str, Any]]) -> None:
    for vector in tools_apply_vectors:
        data: dict[str, Any] = expand_repeats(vector['input'])
        expected: dict[str, Any] = expand_repeats(vector['expected'])
        applied = apply_tool_definitions(
            tools_of(data.get('tools', [])),
            AgentConfig.model_validate(data['config']),
            reserved=data.get('reserved', ()),
            collision_scope=data.get('collision_scope', 'global'),
        )
        assert tools_as_json(applied.tools) == expected.get('tools', []), vector['name']
        if 'routes' in expected:
            assert applied.routes == expected['routes'], vector['name']
        assert issues_as_paths(applied.issues) == expected['issues'], vector['name']


def test_the_settings_apply_vectors(settings_apply_vectors: list[dict[str, Any]]) -> None:
    for vector in settings_apply_vectors:
        data: dict[str, Any] = expand_repeats(vector['input'])
        expected: dict[str, Any] = expand_repeats(vector['expected'])
        applied = apply_settings(AgentConfig.model_validate(data['config']), support=support_of(data.get('support')))
        assert applied.settings == expected.get('settings', {}), vector['name']
        assert issues_as_paths(applied.issues) == expected['issues'], vector['name']


def test_the_agent_name_vectors(agent_name_vectors: list[dict[str, Any]], _record_warnings: Any) -> None:
    caught: list[warnings.WarningMessage] = _record_warnings
    for vector in agent_name_vectors:
        del caught[:]
        try:
            variable_name = agent_variable_name(vector['input'])
        except ValueError:
            assert vector.get('error') == 'empty', vector['name']
        else:
            assert variable_name == vector['variable_name'], vector['name']
        warned = ['prefix-added-automatically' for warning in caught if 'prefix is added' in str(warning.message)]
        assert warned == ([vector['warning']] if vector.get('warning') else []), vector['name']
        assert len(caught) == len(warned), vector['name']


CANONICAL_SPEC_SHA256 = '662f91c7155aa745f22d171e9bf6251be6e44e309fbd92bdfe1cc84ddd328304'

SPEC_LOCKSTEP = (
    'The vectors in spec/ are one half of a contract with every other Agent Control SDK, and the '
    'canonical copy lives in the logfire repository under tests/agent_control/spec/: logfire-js and '
    'this package vendor the same files and pin the same digest. If this assertion fails because a '
    'rule deliberately changed, change it in the canonical copy first, copy the whole directory to '
    'every copy, and update the digest in each. If you did not mean to change a rule, the edit to '
    'spec/ is the bug.'
)


def spec_sha256() -> str:
    """A digest over every file in `spec/`, name and content, in a stable order.

    Line endings are normalized so a checkout that rewrites them still agrees with the digest the
    other SDK pins.
    """
    digest = hashlib.sha256()
    for path in sorted(SPEC.iterdir()):
        digest.update(path.name.encode())
        digest.update(b'\0')
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
        digest.update(b'\0')
    return digest.hexdigest()


def test_the_vectors_match_the_digest_every_copy_pins() -> None:
    assert spec_sha256() == CANONICAL_SPEC_SHA256, SPEC_LOCKSTEP
