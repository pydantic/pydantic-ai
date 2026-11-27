"""Tests for `CapabilityNamespace`: where an agent spec's capability names resolve to classes.

These are not VCR tests: a namespace is resolved while the spec is loaded, before any model request,
so there is nothing for a cassette to record.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from pydantic_ai import Agent, CapabilityNamespace
from pydantic_ai.agent.spec import AgentSpec, get_capability_registry
from pydantic_ai.capabilities import AbstractCapability, PrefixTools, Thinking

_CAPABILITY_MODULE = """
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability


@dataclass
class {name}(AbstractCapability):
    greeting: str = 'hello'
"""


@dataclass
class Greeting(AbstractCapability):
    greeting: str = 'hello'


ModuleWriter = Callable[[str, str], str]


@pytest.fixture
def write_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleWriter:
    """Write a module importable by name, and forget it again after the test."""
    monkeypatch.syspath_prepend(str(tmp_path))  # pyright: ignore[reportUnknownMemberType]

    def write(module_name: str, source: str) -> str:
        (tmp_path / f'{module_name}.py').write_text(textwrap.dedent(source))
        importlib.invalidate_caches()
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        return module_name

    return write


def _capability_module(write_module: ModuleWriter, module_name: str, class_name: str = 'Greeting') -> str:
    return write_module(module_name, _CAPABILITY_MODULE.format(name=class_name))


def _loaded(agent: Agent[object, str]) -> list[AbstractCapability[object]]:
    """The capabilities the spec added, without the ones every agent starts with."""
    defaults = {type(c) for c in Agent('test')._root_capability.capabilities}  # pyright: ignore[reportPrivateUsage]
    return [c for c in agent._root_capability.capabilities if type(c) not in defaults]  # pyright: ignore[reportPrivateUsage]


def _spec(capability: object) -> dict[str, object]:
    return {'model': 'test', 'capabilities': [capability]}


def test_names_imports_only_the_capabilities_a_spec_uses(write_module: ModuleWriter):
    used = _capability_module(write_module, 'ns_names_used')
    unused = _capability_module(write_module, 'ns_names_unused', 'Farewell')
    namespace = CapabilityNamespace(names={'Greeting': f'{used}:Greeting', 'Farewell': f'{unused}:Farewell'})

    agent = Agent.from_spec(_spec({'Greeting': {'greeting': 'hi'}}), capability_namespace=namespace)

    [capability] = _loaded(agent)
    assert type(capability).__module__ == used
    assert capability.greeting == 'hi'  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert used in sys.modules
    assert unused not in sys.modules


def test_entry_points_resolve_installed_capabilities(write_module: ModuleWriter, tmp_path: Path):
    module = _capability_module(write_module, 'ns_entry_points')
    dist_info = tmp_path / 'ns_capabilities-0.0.0.dist-info'
    dist_info.mkdir()
    (dist_info / 'METADATA').write_text('Metadata-Version: 2.1\nName: ns-capabilities\nVersion: 0.0.0\n')
    (dist_info / 'entry_points.txt').write_text(f'[pydantic_ai_tests.capabilities]\nGreeting = {module}:Greeting\n')
    importlib.invalidate_caches()

    agent = Agent.from_spec(
        _spec('Greeting'), capability_namespace=CapabilityNamespace(entry_points='pydantic_ai_tests.capabilities')
    )

    [capability] = _loaded(agent)
    assert type(capability).__module__ == module


def test_entry_points_ignore_extras_like_entry_point_load(write_module: ModuleWriter, tmp_path: Path):
    module = _capability_module(write_module, 'ns_ep_extras')
    dist_info = tmp_path / 'ns_extras-0.0.0.dist-info'
    dist_info.mkdir()
    (dist_info / 'METADATA').write_text('Metadata-Version: 2.1\nName: ns-extras\nVersion: 0.0.0\n')
    (dist_info / 'entry_points.txt').write_text(f'[pydantic_ai_tests.extras]\nGreeting = {module}:Greeting [fast]\n')
    importlib.invalidate_caches()

    agent = Agent.from_spec(
        _spec('Greeting'), capability_namespace=CapabilityNamespace(entry_points='pydantic_ai_tests.extras')
    )

    [capability] = _loaded(agent)
    assert type(capability).__module__ == module


def test_an_entry_point_naming_a_bare_module_is_not_a_capability(write_module: ModuleWriter, tmp_path: Path):
    module = _capability_module(write_module, 'ns_ep_bare')
    dist_info = tmp_path / 'ns_bare-0.0.0.dist-info'
    dist_info.mkdir()
    (dist_info / 'METADATA').write_text('Metadata-Version: 2.1\nName: ns-bare\nVersion: 0.0.0\n')
    (dist_info / 'entry_points.txt').write_text(f'[pydantic_ai_tests.bare]\nGreeting = {module}\n')
    importlib.invalidate_caches()

    with pytest.raises(
        ValueError,
        match=f"Capability 'Greeting' could not be loaded from `{module}`: .*must be subclasses of AbstractCapability",
    ):
        Agent.from_spec(
            _spec('Greeting'), capability_namespace=CapabilityNamespace(entry_points='pydantic_ai_tests.bare')
        )


def test_entry_points_with_two_targets_for_one_name_raise(write_module: ModuleWriter, tmp_path: Path):
    for dist, module in [('first', 'ns_ep_first'), ('second', 'ns_ep_second')]:
        _capability_module(write_module, module)
        dist_info = tmp_path / f'ns_{dist}-0.0.0.dist-info'
        dist_info.mkdir()
        (dist_info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: ns-{dist}\nVersion: 0.0.0\n')
        (dist_info / 'entry_points.txt').write_text(f'[pydantic_ai_tests.clash]\nGreeting = {module}:Greeting\n')
    importlib.invalidate_caches()

    # Installed packages fill this tier, so the clash breaks only a spec that uses the name.
    clashing = CapabilityNamespace(entry_points='pydantic_ai_tests.clash')
    registry = get_capability_registry(namespace=clashing)
    assert registry['Thinking'] is Thinking
    # No single class describes the name, so it is left out of the listing and the JSON schema.
    assert 'Greeting' not in registry
    assert 'Greeting' not in list(registry)
    schema = AgentSpec.model_json_schema_with_capabilities(capability_namespace=clashing)
    assert 'Greeting' not in str(schema)
    assert 'Thinking' in str(schema)

    with pytest.raises(
        ValueError, match=r"Two different capabilities are named 'Greeting' in the `pydantic_ai_tests\.clash`"
    ):
        Agent.from_spec(
            _spec('Greeting'), capability_namespace=CapabilityNamespace(entry_points='pydantic_ai_tests.clash')
        )

    # A higher tier settles it.
    settled = CapabilityNamespace(entry_points='pydantic_ai_tests.clash', names={'Greeting': f'{module}:Greeting'})
    assert get_capability_registry(namespace=settled)['Greeting'].__module__ == module


def test_modules_make_their_capabilities_available(write_module: ModuleWriter):
    module = write_module(
        'ns_modules_all',
        """
        from dataclasses import dataclass

        from pydantic_ai.capabilities import AbstractCapability

        __all__ = ['Greeting']


        @dataclass
        class Greeting(AbstractCapability):
            pass


        @dataclass
        class Unlisted(AbstractCapability):
            pass
        """,
    )

    registry = get_capability_registry(namespace=CapabilityNamespace(modules=[module]))

    assert registry['Greeting'].__module__ == module
    assert 'Unlisted' not in registry


def test_modules_without_all_offer_their_public_capabilities(write_module: ModuleWriter):
    module = write_module(
        'ns_modules_public',
        """
        from dataclasses import dataclass

        from pydantic_ai.capabilities import AbstractCapability, Thinking


        @dataclass
        class Greeting(AbstractCapability):
            pass


        @dataclass
        class _Private(AbstractCapability):
            pass


        class NotADataclass(AbstractCapability):
            pass


        @dataclass
        class OptedOut(AbstractCapability):
            @classmethod
            def get_serialization_name(cls):
                return None
        """,
    )

    registry = get_capability_registry(namespace=CapabilityNamespace(modules=[module]))

    assert registry['Greeting'].__module__ == module
    # Imported classes are not the module's own: `AbstractCapability` would otherwise become a spec name.
    assert {'_Private', 'NotADataclass', 'OptedOut', 'AbstractCapability'}.isdisjoint(registry)
    assert registry['Thinking'] is Thinking


def test_a_module_that_imports_a_built_in_does_not_clash_with_one_that_overrides_it(write_module: ModuleWriter):
    overriding = _capability_module(write_module, 'ns_modules_override', class_name='Thinking')
    wrapping = write_module(
        'ns_modules_wrapping',
        """
        from dataclasses import dataclass

        from pydantic_ai.capabilities import AbstractCapability, Thinking


        @dataclass
        class Careful(AbstractCapability):
            thinking: Thinking | None = None
        """,
    )

    registry = get_capability_registry(namespace=CapabilityNamespace(modules=[overriding, wrapping]))

    assert registry['Thinking'].__module__ == overriding
    assert registry['Careful'].__module__ == wrapping


def test_a_module_that_cannot_be_imported_says_where_it_came_from():
    with pytest.raises(
        ValueError, match='Capability module `ns_no_such_module` could not be imported: No module named'
    ):
        get_capability_registry(namespace=CapabilityNamespace(modules=['ns_no_such_module']))


def test_modules_with_two_classes_for_one_name_raise(write_module: ModuleWriter):
    first = _capability_module(write_module, 'ns_modules_first')
    second = _capability_module(write_module, 'ns_modules_second')

    with pytest.raises(ValueError, match="Two different capabilities are named 'Greeting' in `modules`"):
        get_capability_registry(namespace=CapabilityNamespace(modules=[first, second]))


def test_higher_tiers_override_lower_ones_and_built_ins(write_module: ModuleWriter):
    module = _capability_module(write_module, 'ns_tiers', 'Thinking')

    by_name = get_capability_registry(namespace=CapabilityNamespace(names={'Thinking': f'{module}:Thinking'}))
    assert by_name['Thinking'].__module__ == module

    @dataclass
    class Thinking(AbstractCapability):  # shadows the built-in on purpose
        pass

    by_type = get_capability_registry(
        namespace=CapabilityNamespace(types=[Thinking], names={'Thinking': f'{module}:Thinking'})
    )
    assert by_type['Thinking'] is Thinking


def test_custom_capability_types_and_namespace_types_share_a_tier():
    registry = get_capability_registry([Greeting], CapabilityNamespace(types=[PrefixTools]))
    assert registry['Greeting'] is Greeting

    # The same class through both doors is one capability, as when a namespace adds to a shared default.
    assert get_capability_registry([Greeting], CapabilityNamespace(types=[Greeting]))['Greeting'] is Greeting

    @dataclass
    class OtherGreeting(AbstractCapability):
        @classmethod
        def get_serialization_name(cls) -> str:
            return 'Greeting'

    with pytest.raises(ValueError, match="Duplicate capability class name: 'Greeting'"):
        get_capability_registry([Greeting], CapabilityNamespace(types=[OtherGreeting]))


def test_an_unknown_name_points_at_both_parameters():
    with pytest.raises(
        ValueError,
        match=r"Capability 'Missing' is not in the provided `custom_capability_types` or `capability_namespace`\.",
    ):
        Agent.from_spec(_spec('Missing'), capability_namespace=CapabilityNamespace())


def test_a_name_must_match_its_class(write_module: ModuleWriter):
    module = _capability_module(write_module, 'ns_mismatch')
    registry = get_capability_registry(namespace=CapabilityNamespace(names={'Hello': f'{module}:Greeting'}))

    with pytest.raises(
        ValueError, match="points at `ns_mismatch:Greeting`, whose serialization name is 'Greeting', not 'Hello'"
    ):
        registry['Hello']


@pytest.mark.parametrize(
    ('target', 'reason'),
    [
        ('ns_does_not_exist:Greeting', "No module named 'ns_does_not_exist'"),
        ('pydantic_ai:DoesNotExist', 'DoesNotExist'),
        ('pydantic_ai:Agent', 'must be subclasses of AbstractCapability'),
    ],
)
def test_a_target_that_is_not_a_capability_raises_on_use(target: str, reason: str):
    registry = get_capability_registry(namespace=CapabilityNamespace(names={'Greeting': target}))
    assert 'Greeting' in registry

    with pytest.raises(ValueError, match=f"Capability 'Greeting' could not be loaded from `{target}`: .*{reason}"):
        Agent.from_spec(_spec('Greeting'), capability_namespace=CapabilityNamespace(names={'Greeting': target}))


def test_nested_specs_resolve_through_the_same_namespace(write_module: ModuleWriter):
    module = _capability_module(write_module, 'ns_nested')

    agent = Agent.from_spec(
        _spec({'PrefixTools': {'prefix': 'greet', 'capability': 'Greeting'}}),
        capability_namespace=CapabilityNamespace(names={'Greeting': f'{module}:Greeting'}),
    )

    [prefix_tools] = _loaded(agent)
    assert isinstance(prefix_tools, PrefixTools)
    assert type(prefix_tools.wrapped).__module__ == module


def test_from_file_and_schema_accept_a_namespace(write_module: ModuleWriter, tmp_path: Path):
    module = _capability_module(write_module, 'ns_file')
    namespace = CapabilityNamespace(names={'Greeting': f'{module}:Greeting'})
    spec_path = tmp_path / 'agent.yaml'
    AgentSpec.model_validate(_spec({'Greeting': {'greeting': 'hi'}})).to_file(spec_path, capability_namespace=namespace)

    agent = Agent.from_file(spec_path, capability_namespace=namespace)

    [capability] = _loaded(agent)
    assert type(capability).__module__ == module
    schema = (tmp_path / 'agent_schema.json').read_text()
    assert 'Greeting' in schema
    assert 'Greeting' in str(AgentSpec.model_json_schema_with_capabilities(capability_namespace=namespace))


def test_registry_lists_names_without_importing_them():
    registry = get_capability_registry(namespace=CapabilityNamespace(names={'Greeting': 'ns_never_imported:Greeting'}))

    assert len(registry) == len(set(registry))
    assert 'Greeting' in list(registry)
    assert 'ns_never_imported' not in sys.modules


def test_a_target_that_is_not_a_dataclass_raises_on_use(write_module: ModuleWriter):
    module = write_module(
        'ns_not_dataclass',
        """
        from pydantic_ai.capabilities import AbstractCapability


        class Greeting(AbstractCapability):
            pass
        """,
    )
    registry = get_capability_registry(namespace=CapabilityNamespace(names={'Greeting': f'{module}:Greeting'}))

    with pytest.raises(ValueError, match='must be decorated with `@dataclass`'):
        registry['Greeting']


def test_modules_with_an_empty_all_offer_nothing(write_module: ModuleWriter):
    module = write_module(
        'ns_modules_empty_all',
        """
        from dataclasses import dataclass

        from pydantic_ai.capabilities import AbstractCapability

        __all__ = []


        @dataclass
        class Thinking(AbstractCapability):
            pass
        """,
    )

    registry = get_capability_registry(namespace=CapabilityNamespace(modules=[module]))

    assert registry['Thinking'] is Thinking
