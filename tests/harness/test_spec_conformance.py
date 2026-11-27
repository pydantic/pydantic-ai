"""Every Harness capability's agent spec schema publishes its parameters, and every form it publishes loads.

See `tests/spec_conformance.py` for what each check does, and `tests/test_spec_conformance.py` for the core
capabilities. Every capability class in `pydantic_ai_harness` that a spec can name is checked, with the schema
`AgentSpec.model_json_schema_with_capabilities()` generates when it is a custom capability type.

A parameter a class keeps out of its spec on purpose goes in `OPT_OUT` with the reason, and one a defect keeps out
goes in `KNOWN`, linked to the issue tracking it. `KNOWN_UNLOADABLE` holds a parameter whose published schema accepts
values its type rejects, `KNOWN_REJECTED` a class whose `from_spec` rejects some form its schema publishes, and
`KNOWN_SCHEMA_FAILURES` a class whose schema can't be generated. Each expects the failure, so an entry has to go once
its class is fixed.

`NARROWED_FROM_SPEC` lists the classes whose `from_spec` restates `__init__` with narrower types, so a subclass
doesn't inherit it: see `pydantic_ai_harness._from_spec`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest
from pydantic import PydanticUserError
from pydantic_core import SchemaError

from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness.ask_user import AskUser

from ..capability_cases import shipped_capability_types
from ..spec_conformance import JsonSchema, check_capability_forms_load, check_capability_parameters
from .capability_cases import HARNESS_CAPABILITY_PACKAGES

SHIPPED, SKIPPED_MODULES = shipped_capability_types(HARNESS_CAPABILITY_PACKAGES)
CLASSES = {
    name: cls
    for name, cls in SHIPPED.items()
    if not inspect.isabstract(cls) and '__dataclass_fields__' in vars(cls) and cls.get_serialization_name() is not None
}
"""The classes a spec can name: concrete, decorated with `@dataclass` as the registry requires, and serializable."""

_RUNTIME_ONLY = 'An SDK object, which a spec has no form for.'
_NO_DEFERRED_WORKSPACE = 'A workspace capability rejects `defer_loading=True`: its workspace is selected first.'
OPT_OUT: dict[str, dict[str, str]] = {
    'BoxdSandbox': {'client': _RUNTIME_ONLY},
    'BrowserUse': dict.fromkeys(
        ('llm', 'browser_profile', 'output_schema'),
        "Kept out of specs by `from_spec`: a spec-loaded instance uses the host run's model, the default browser and "
        'prose output.',
    ),
    'DaytonaSandbox': {'client': _RUNTIME_ONLY},
    'E2BSandbox': {'client': _RUNTIME_ONLY},
    'KubernetesSandbox': {'defer_loading': _NO_DEFERRED_WORKSPACE},
    'ManagedPrompt': {'logfire_instance': 'A `Logfire` instance, which a spec has no form for.'},
    'ModalSandbox': {
        'client': _RUNTIME_ONLY,
        'defer_loading': _NO_DEFERRED_WORKSPACE,
        'workdir': 'The deprecated name of `working_dir`.',
    },
    'OpenSandbox': {'connection': _RUNTIME_ONLY, 'defer_loading': _NO_DEFERRED_WORKSPACE},
    'Permissions': {'remembered': 'The approvals remembered during runs, not configuration.'},
    'PlaywrightBrowser': {'storage_state': 'Session credentials, which `from_spec` keeps out of specs.'},
    'StackOne': {'auth': 'A credential, which `from_spec` keeps out of specs.'},
}

_DISAGREEMENT = 'https://github.com/pydantic/pydantic-ai/issues/13979'
KNOWN: dict[str, dict[str, str]] = {
    'CDPBrowser': {'policy': _DISAGREEMENT},
    'CodeMode': dict.fromkeys(('os_policy', 'monty_sandbox_url'), _DISAGREEMENT),
    'E2BSandbox': {'network': _DISAGREEMENT},
    'Memory': {'write_metadata': _DISAGREEMENT},
    'OpenSandbox': {'network_policy': _DISAGREEMENT},
    'Planning': {'descriptions': _DISAGREEMENT},
    'PlaywrightBrowser': {'policy': _DISAGREEMENT},
    'StepPersistence': dict.fromkeys(
        (
            'id',
            'description',
            'defer_loading',
            'agent_name',
            'run_id',
            'parent_run_id',
            'metadata',
            'capture_frontier',
        ),
        _DISAGREEMENT,
    ),
}

KNOWN_UNLOADABLE: dict[str, dict[str, str]] = {'ReportToolErrors': {'exceptions': _DISAGREEMENT}}

KNOWN_REJECTED: dict[str, str] = {
    'ManagedPrompt': f'A prompt named by string needs `default`, which the schema publishes as optional: {_DISAGREEMENT}',
}

_SCHEMA_FAILURE = 'https://github.com/pydantic/pydantic-ai/issues/13978'
_HOSTED_MCP = 'The spec surface for `client` and `auth` is pending a decision.'
_SCHEMA_ERRORS: dict[str, tuple[type[Exception], str]] = {
    _SCHEMA_FAILURE: (PydanticUserError, 'is not fully defined'),
    _HOSTED_MCP: (SchemaError, "'cls' must be valid as the first argument to 'isinstance'"),
}
KNOWN_SCHEMA_FAILURES: dict[str, str] = {
    'Coder': _SCHEMA_FAILURE,
    'ToolOutputLimits': _SCHEMA_FAILURE,
    **dict.fromkeys(
        ('DayAI', 'GitHub', 'Grain', 'Linear', 'LogfireMCP', 'Notion', 'Ordinal', 'PostHog', 'Pylon', 'Slack'),
        _HOSTED_MCP,
    ),
}

NARROWED_FROM_SPEC = ('AskUser', 'IPythonCodeMode', 'PromptInjectionDefender')

CHECKED = sorted(CLASSES.keys() - KNOWN_SCHEMA_FAILURES.keys())


@cache
def _schema(cls: type[AbstractCapability[None]]) -> JsonSchema:
    return AgentSpec.model_json_schema_with_capabilities([cls])


@pytest.mark.parametrize('name', CHECKED)
def test_schema_publishes_spec_parameters(name: str):
    cls = CLASSES[name]
    check_capability_parameters(cls, _schema(cls), {**OPT_OUT.get(name, {}), **KNOWN.get(name, {})})


# The schema still publishes deprecated parameters, and loading one warns.
@pytest.mark.filterwarnings('ignore::pydantic_ai.PydanticAIDeprecationWarning')
@pytest.mark.filterwarnings('ignore::pydantic_ai_harness.HarnessDeprecationWarning')
@pytest.mark.parametrize('name', CHECKED)
def test_spec_forms_load(name: str, tmp_path: Path):
    cls = CLASSES[name]
    check_capability_forms_load(
        cls, _schema(cls), tmp_path, [cls], KNOWN_UNLOADABLE.get(name, {}), known_rejected=name in KNOWN_REJECTED
    )


def _shipped_class(name: str) -> type[AbstractCapability[None]]:
    """The class a table names; when every module imported, a name with no class fails as a stale entry."""
    if name not in CLASSES and SKIPPED_MODULES:  # pragma: lax no cover
        pytest.skip(f'{name} needs an optional dependency group that is not installed')
    return CLASSES[name]


@pytest.mark.parametrize('name', sorted(KNOWN_SCHEMA_FAILURES))
def test_known_schema_failures(name: str):
    error, match = _SCHEMA_ERRORS[KNOWN_SCHEMA_FAILURES[name]]
    with pytest.raises(error, match=match):
        AgentSpec.model_json_schema_with_capabilities([_shipped_class(name)])


def test_tables_name_checked_classes():
    # Only meaningful when every module imported: a class whose optional group isn't installed looks deleted. Every
    # module imports in the `test harness browser-use` CI job, which runs this file.
    tables = {**OPT_OUT, **KNOWN, **KNOWN_UNLOADABLE, **KNOWN_REJECTED}.keys() | {*NARROWED_FROM_SPEC}
    if not SKIPPED_MODULES:  # pragma: lax no cover
        assert tables <= set(CHECKED), f'Remove the entries for classes that are gone: {sorted(tables - set(CHECKED))}'
    else:  # pragma: lax no cover
        pytest.skip(f'Modules that did not import: {", ".join(sorted(SKIPPED_MODULES))}')


@pytest.mark.parametrize('name', NARROWED_FROM_SPEC)
def test_narrowed_from_spec_keeps_init_parameters(name: str):
    """A narrowed `from_spec` takes the parameters `__init__` takes, of the same kinds and with the same defaults.

    Only the annotations may differ, so a spec-loaded instance can't drift from one built directly.
    """
    cls = _shipped_class(name)

    def parameters(signature: inspect.Signature) -> dict[str, tuple[object, object]]:
        return {p.name: (p.kind, p.default) for p in signature.parameters.values() if p.name != 'self'}

    assert parameters(inspect.signature(cls.from_spec)) == parameters(inspect.signature(cls.__init__))


@pytest.mark.parametrize('name', NARROWED_FROM_SPEC)
def test_subclass_loads_its_own_fields_and_defaults(name: str):
    """A subclass loads through the default `from_spec`, which takes the fields it adds and keeps the defaults it sets."""
    parent = _shipped_class(name)

    @dataclass(kw_only=True)
    class Custom(parent):
        extra: str = 'default'
        description: str | None = 'custom'

    required = {p.name: None for p in inspect.signature(parent.from_spec).parameters.values() if p.default is p.empty}
    loaded = Custom.from_spec(**required, extra='set')
    assert isinstance(loaded, Custom)
    assert (loaded.extra, loaded.description) == ('set', 'custom')


def test_subclass_keeps_the_from_spec_it_defines():
    """A subclass that defines its own `from_spec`, and a subclass of that, load through it."""

    @dataclass(kw_only=True)
    class Custom(AskUser[None]):
        @classmethod
        def from_spec(cls, *args: object, **kwargs: object) -> AskUser[None]:
            return cls(answerer=None, description='own')

    class Grandchild(Custom):
        pass

    loaded = Grandchild.from_spec()
    assert isinstance(loaded, Grandchild)
    assert loaded.description == 'own'
