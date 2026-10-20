"""The stored JSON schema is one half of a contract, so it is pinned rather than merely exercised."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from pydantic_ai_harness.logfire._contract import (
    AGENT_CONFIG_JSON_SCHEMA,
    SCHEMA_SHA256,
    AgentConfig,
    AgentConfigSettings,
    InstructionBlock,
    ToolDefinitionOverride,
    canonical_json,
)

LOCKSTEP = (
    'AGENT_CONFIG_JSON_SCHEMA is one half of a contract with the Logfire UI and with every other '
    'Agent Control SDK: update the TypeScript package and '
    'src/services/logfire-frontend/src/app/project/managed-agents/agent-config.ts in the Logfire '
    'platform repo in lockstep, or a UI-created variable and an SDK-created one will store different '
    'schemas for the same agent.'
)

CANONICAL_SCHEMA_SHA256 = 'e4c38488d6c46440dbf31939291e13fcd56814c1305bc9012cd2158733b0bbad'

INSTRUCTIONS_SCHEMA: dict[str, Any] = AGENT_CONFIG_JSON_SCHEMA['properties']['instructions']
SETTINGS_SCHEMA: dict[str, Any] = AGENT_CONFIG_JSON_SCHEMA['properties']['settings']
TOOL_OVERRIDE_SCHEMA: dict[str, Any] = AGENT_CONFIG_JSON_SCHEMA['properties']['tool_definitions']['items']

# The object branch of an `instructions` entry, the other branch being a bare string.
INSTRUCTION_BLOCK_SCHEMA: dict[str, Any] = next(
    option for option in INSTRUCTIONS_SCHEMA['anyOf'][1]['items']['anyOf'] if option['type'] == 'object'
)


def subschemas(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every object schema reachable from `schema`, itself included."""
    yield schema
    children: list[Any] = [
        *cast(dict[str, Any], schema.get('properties', {})).values(),
        schema.get('items'),
        schema.get('additionalProperties'),
        *cast(list[Any], schema.get('anyOf', [])),
    ]
    for child in children:
        if isinstance(child, dict):
            yield from subschemas(cast(dict[str, Any], child))


def test_schema_matches_the_digest_every_copy_pins() -> None:
    assert SCHEMA_SHA256 == CANONICAL_SCHEMA_SHA256, LOCKSTEP


def test_the_canonical_form_encodes_non_ascii_the_way_json_stringify_does() -> None:
    r"""The digest must not depend on which language computed it.

    `json.dumps` escapes non-ASCII to `\uXXXX` by default and `JSON.stringify` emits the character,
    so a schema carrying one would otherwise digest differently in the Python and TypeScript copies
    while being the same document -- drift in the canonical form rather than in the contract, which
    is the one kind this digest would be worst at explaining. The schema is ASCII today, so nothing
    but this test would notice if the canonical form stopped saying so.
    """
    assert canonical_json({'description': 'Z\u00fcrich'}) == '{"description":"Z\u00fcrich"}'.encode()


def test_every_model_field_is_described() -> None:
    assert set(AgentConfig.model_fields) == set(AGENT_CONFIG_JSON_SCHEMA['properties']), LOCKSTEP
    assert set(AgentConfigSettings.model_fields) == set(SETTINGS_SCHEMA['properties']), LOCKSTEP
    assert set(ToolDefinitionOverride.model_fields) == set(TOOL_OVERRIDE_SCHEMA['properties']), LOCKSTEP
    assert set(InstructionBlock.model_fields) == set(INSTRUCTION_BLOCK_SCHEMA['properties']), LOCKSTEP


def test_schema_is_permissive_at_every_level() -> None:
    # A closed schema anywhere would make the Logfire backend reject a write that `AgentConfig` is
    # built to tolerate, so the forward-compatibility contract lives in the stored schema too.
    for subschema in subschemas(AGENT_CONFIG_JSON_SCHEMA):
        assert subschema.get('additionalProperties') is not False, LOCKSTEP
        assert 'enum' not in subschema, LOCKSTEP
