"""Every core capability's agent spec schema publishes its parameters, and every form it publishes loads.

See `tests/spec_conformance.py` for what each check does. `tests/harness/test_spec_conformance.py` and
`tests/evals/test_spec_conformance.py` run the same checks over the Harness capabilities and the evaluators. Native
tools are covered here through `NativeTool`, whose forms take each of them. `OpenAICompaction` and `AnthropicCompaction`
load only as custom capability types, so they're swept as such where their provider SDKs are installed.

A parameter a class keeps out of its spec on purpose goes in `OPT_OUT` with the reason, and one a defect keeps out
goes in `KNOWN`, linked to the issue tracking it. `KNOWN_REJECTED` holds a class whose `from_spec` rejects some form
its schema publishes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pydantic_ai.agent.spec import AgentSpec, get_capability_registry

from .conftest import try_import
from .spec_conformance import JsonSchema, argument_errors, check_capability_forms_load, check_capability_parameters

with try_import() as compaction_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicCompaction
    from pydantic_ai.models.openai import OpenAICompaction

if TYPE_CHECKING:
    # Imported for type checking only, so a hint naming it doesn't resolve at runtime.
    from decimal import Decimal as TypeCheckingOnly

CUSTOM_TYPES = [OpenAICompaction, AnthropicCompaction] if compaction_imports_successful() else []
REGISTRY = get_capability_registry(CUSTOM_TYPES)

OPT_OUT: dict[str, dict[str, str]] = {
    'Instrumentation': {
        'settings': 'Its serializable options are spec parameters of their own.',
        'id': 'An agent has one instrumentation configuration, so a spec has nothing to name.',
    },
    'PrefixTools': {'wrapped': 'A spec gives the wrapped capability as a nested `capability` spec.'},
}

_DISAGREEMENT = 'https://github.com/pydantic/pydantic-ai/issues/13979'
KNOWN: dict[str, dict[str, str]] = {
    'BatchModelRequests': dict.fromkeys(('id', 'description', 'defer_loading'), _DISAGREEMENT),
    'Instrumentation': dict.fromkeys(('description', 'defer_loading'), _DISAGREEMENT),
    'NativeTool': dict.fromkeys(('id', 'description', 'defer_loading'), _DISAGREEMENT),
    'PrefixTools': dict.fromkeys(('id', 'description', 'defer_loading'), _DISAGREEMENT),
}

KNOWN_REJECTED: dict[str, str] = {
    'NativeTool': f'Its bare name gives `from_spec` no native tool to build: {_DISAGREEMENT}',
}


@pytest.fixture(scope='module')
def schema() -> JsonSchema:
    return AgentSpec.model_json_schema_with_capabilities(CUSTOM_TYPES)


@pytest.mark.parametrize('name', sorted(REGISTRY))
def test_schema_publishes_spec_parameters(name: str, schema: JsonSchema):
    check_capability_parameters(REGISTRY[name], schema, {**OPT_OUT.get(name, {}), **KNOWN.get(name, {})})


# The schema still publishes deprecated parameters, and loading one warns.
@pytest.mark.filterwarnings('ignore::pydantic_ai.PydanticAIDeprecationWarning')
@pytest.mark.parametrize('name', sorted(REGISTRY))
def test_spec_forms_load(name: str, schema: JsonSchema, tmp_path: Path):
    check_capability_forms_load(REGISTRY[name], schema, tmp_path, CUSTOM_TYPES, known_rejected=name in KNOWN_REJECTED)


def test_tables_name_registered_classes():
    tables = OPT_OUT.keys() | KNOWN.keys() | KNOWN_REJECTED.keys()
    assert tables <= REGISTRY.keys(), (
        f'Remove the entries for classes that are gone: {sorted(tables - REGISTRY.keys())}'
    )


def test_argument_errors_says_why_arguments_do_not_load():
    def target(count: int, *, hook: Callable[[], None] = print, later: TypeCheckingOnly | None = None) -> None:
        pass

    assert argument_errors(target, (1,), {}) == {}
    assert argument_errors(target, (), {'other': 1}) == {'': "missing a required argument: 'count'"}
    assert set(argument_errors(target, ('a',), {})) == {'count'}
    assert argument_errors(target, (1,), {'hook': None}) == {'hook': 'its type has no JSON form'}
    assert argument_errors(target, (1,), {'later': None})['later'].startswith('its type can not be checked (NameError')
