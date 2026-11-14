"""Tests for the shared _spec.py module."""

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, Self

import pytest
from pydantic import BaseModel, GetPydanticSchema, JsonValue, ValidationError
from pydantic_core import SchemaError
from typing_extensions import TypeAliasType

from pydantic_ai import Agent
from pydantic_ai._spec import NamedSpec, build_registry, build_schema_types, load_from_registry
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, SetToolMetadata

from .conftest import try_import

with try_import() as jsonschema_imports_successful:
    from jsonschema.validators import validator_for


class TestNamedSpec:
    """Test NamedSpec parsing of all three short forms."""

    def test_bare_name(self):
        spec = NamedSpec.model_validate('MyClass')
        assert spec.name == 'MyClass'
        assert spec.arguments is None
        assert spec.args == ()
        assert spec.kwargs == {}

    def test_single_arg(self):
        spec = NamedSpec.model_validate({'MyClass': 'hello'})
        assert spec.name == 'MyClass'
        assert spec.arguments == ('hello',)
        assert spec.args == ('hello',)
        assert spec.kwargs == {}

    def test_kwargs(self):
        spec = NamedSpec.model_validate({'MyClass': {'a': 1, 'b': 2}})
        assert spec.name == 'MyClass'
        assert spec.arguments == {'a': 1, 'b': 2}
        assert spec.args == ()
        assert spec.kwargs == {'a': 1, 'b': 2}

    def test_single_non_string_arg(self):
        spec = NamedSpec.model_validate({'MyClass': 42})
        assert spec.name == 'MyClass'
        assert spec.arguments == (42,)

    def test_single_list_arg(self):
        spec = NamedSpec.model_validate({'MyClass': [1, 2, 3]})
        assert spec.name == 'MyClass'
        assert spec.arguments == ([1, 2, 3],)

    def test_long_form(self):
        spec = NamedSpec(name='MyClass', arguments={'x': 1, 'y': 2})
        assert spec.name == 'MyClass'
        assert spec.kwargs == {'x': 1, 'y': 2}

    def test_multiple_keys_error(self):
        with pytest.raises(Exception):
            NamedSpec.model_validate({'A': 1, 'B': 2})

    def test_serialize_short_form_no_args(self):
        spec = NamedSpec(name='MyClass', arguments=None)
        result = spec.model_dump(context={'use_short_form': True})
        assert result == 'MyClass'

    def test_serialize_short_form_single_arg(self):
        spec = NamedSpec(name='MyClass', arguments=('hello',))
        result = spec.model_dump(context={'use_short_form': True})
        assert result == {'MyClass': 'hello'}

    def test_serialize_short_form_kwargs(self):
        spec = NamedSpec(name='MyClass', arguments={'a': 1})
        result = spec.model_dump(context={'use_short_form': True})
        assert result == {'MyClass': {'a': 1}}


class _Base:
    @classmethod
    def get_name(cls) -> str | None:
        return cls.__name__


class Foo(_Base):
    def __init__(self, x: int = 0):
        self.x = x


class Bar(_Base):
    def __init__(self, msg: str):
        self.msg = msg


class OptedOut(_Base):
    @classmethod
    def get_name(cls) -> str | None:
        return None


class TestBuildRegistry:
    def test_basic(self):
        registry = build_registry(
            custom_types=[],
            defaults=[Foo, Bar],
            get_name=lambda c: c.get_name(),
            label='thing',
        )
        assert set(registry.keys()) == {'Foo', 'Bar'}

    def test_custom_types(self):
        registry = build_registry(
            custom_types=[Foo],
            defaults=[Bar],
            get_name=lambda c: c.get_name(),
            label='thing',
        )
        assert set(registry.keys()) == {'Foo', 'Bar'}

    def test_custom_overrides_default(self):
        class Foo2(_Base):
            @classmethod
            def get_name(cls) -> str | None:
                return 'Foo'

        registry = build_registry(
            custom_types=[Foo2],
            defaults=[Foo, Bar],
            get_name=lambda c: c.get_name(),
            label='thing',
        )
        assert registry['Foo'] is Foo2

    def test_duplicate_custom_types(self):
        with pytest.raises(ValueError, match="Duplicate thing class name: 'Foo'"):
            build_registry(
                custom_types=[Foo, Foo],
                defaults=[],
                get_name=lambda c: c.get_name(),
                label='thing',
            )

    def test_opted_out_default(self):
        registry = build_registry(
            custom_types=[],
            defaults=[Foo, OptedOut],
            get_name=lambda c: c.get_name(),
            label='thing',
        )
        assert set(registry.keys()) == {'Foo'}

    def test_opted_out_custom_raises(self):
        with pytest.raises(ValueError, match='has opted out of serialization'):
            build_registry(
                custom_types=[OptedOut],
                defaults=[],
                get_name=lambda c: c.get_name(),
                label='thing',
            )

    def test_validate_callback(self):
        def validate(cls: type[Any]) -> None:
            if cls.__name__ == 'Foo':  # pragma: no branch
                raise ValueError('Foo is not allowed')

        with pytest.raises(ValueError, match='Foo is not allowed'):
            build_registry(
                custom_types=[Foo],
                defaults=[],
                get_name=lambda c: c.get_name(),
                label='thing',
                validate=validate,
            )


class TestLoadFromRegistry:
    def test_success_no_args(self):
        registry: dict[str, type[Any]] = {'Foo': Foo}
        spec = NamedSpec(name='Foo', arguments=None)
        result = load_from_registry(registry, spec, label='thing', custom_types_param='custom_things')
        assert isinstance(result, Foo)
        assert result.x == 0

    def test_success_with_args(self):
        registry: dict[str, type[Any]] = {'Foo': Foo}
        spec = NamedSpec(name='Foo', arguments=(42,))
        result = load_from_registry(registry, spec, label='thing', custom_types_param='custom_things')
        assert isinstance(result, Foo)
        assert result.x == 42

    def test_success_with_kwargs(self):
        registry: dict[str, type[Any]] = {'Bar': Bar}
        spec = NamedSpec(name='Bar', arguments={'msg': 'hello'})
        result = load_from_registry(registry, spec, label='thing', custom_types_param='custom_things')
        assert isinstance(result, Bar)
        assert result.msg == 'hello'

    def test_not_found(self):
        registry: dict[str, type[Any]] = {'Foo': Foo}
        spec = NamedSpec(name='Baz', arguments=None)
        with pytest.raises(ValueError, match="Thing 'Baz' is not in the provided `custom_things`"):
            load_from_registry(registry, spec, label='thing', custom_types_param='custom_things')

    def test_instantiation_error(self):
        registry: dict[str, type[Any]] = {'Bar': Bar}
        spec = NamedSpec(name='Bar', arguments=None)  # Bar requires msg
        with pytest.raises(ValueError, match="Failed to instantiate thing 'Bar'"):
            load_from_registry(registry, spec, label='thing', custom_types_param='custom_things')

    def test_custom_instantiate(self):
        @dataclass
        class Baz:
            value: str

            @classmethod
            def create(cls, *args: Any, **kwargs: Any) -> 'Baz':
                return cls(value=f'custom:{args[0] if args else kwargs.get("value", "")}')

        registry: dict[str, type[Any]] = {'Baz': Baz}
        spec = NamedSpec(name='Baz', arguments=('test',))
        result = load_from_registry(
            registry,
            spec,
            label='thing',
            custom_types_param='custom_things',
            instantiate=lambda cls, args, kwargs: cls.create(*args, **kwargs),
        )
        assert result.value == 'custom:test'


@dataclass
class ObjectParam(AbstractCapability[None]):
    weights: dict[str, int] | None = None

    @classmethod
    def from_spec(cls, weights: dict[str, int] | None = None) -> Self:
        return cls(weights=weights)


class Limits(BaseModel):
    length: int


@dataclass
class ModelParam(AbstractCapability[None]):
    limits: Limits

    @classmethod
    def from_spec(cls, limits: Limits) -> Self:
        return cls(limits=Limits.model_validate(limits))


@dataclass
class WeightedParam(AbstractCapability[None]):
    weights: dict[str, int]
    label: str = 'w'


@dataclass
class AnyParam(AbstractCapability[None]):
    value: Any = None

    @classmethod
    def from_spec(cls, value: Any = None) -> Self:
        return cls(value=value)


ExternalRef = Annotated[
    dict[str, int],
    GetPydanticSchema(
        get_pydantic_json_schema=lambda core_schema, handler: {'$ref': 'https://example.com/schema.json'}
    ),
]
# A union that names itself through another alias: its JSON schema is a `$ref` cycle. Pyright rejects the
# self-reference, but Pydantic generates a schema for it.
IntOrStr = TypeAliasType('IntOrStr', 'int | StrOrInt')  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
StrOrInt = TypeAliasType('StrOrInt', 'str | IntOrStr')


@dataclass
class ExternalRefParam(AbstractCapability[None]):
    @classmethod
    def from_spec(cls, value: ExternalRef | None = None) -> Self:
        return cls()


@dataclass
class RefCycleParam(AbstractCapability[None]):
    @classmethod
    def from_spec(cls, value: IntOrStr | None = None) -> Self:  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
        return cls()


class TestBuildSchemaTypes:
    """Schema generation and spec loading make no model request, so these tests record no cassette."""

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    @pytest.mark.parametrize(
        ('arguments', 'tools', 'metadata'),
        [
            pytest.param(['a'], ['a'], {}, id='short-form-list'),
            pytest.param('all', 'all', {}, id='short-form-all'),
            pytest.param({'code_mode': True}, 'all', {'code_mode': True}, id='metadata-only'),
            pytest.param({'tools': ['a'], 'code_mode': True}, ['a'], {'code_mode': True}, id='tools-and-metadata'),
        ],
    )
    def test_variadic_keyword_capability_schema(
        self, arguments: JsonValue, tools: list[str] | Literal['all'], metadata: dict[str, JsonValue]
    ):
        """Every `SetToolMetadata` form the schema accepts loads, short forms included."""
        spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [{'SetToolMetadata': arguments}]}
        schema = AgentSpec.model_json_schema_with_capabilities()
        assert list(validator_for(schema)(schema).iter_errors(spec)) == []
        capabilities = Agent.from_spec(spec).root_capability.capabilities
        [capability] = [c for c in capabilities if isinstance(c, SetToolMetadata)]
        assert (capability.tools, capability.metadata) == (tools, metadata)

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_keyword_only_capability_schema(self):
        """Capability schemas agree with the keyword-only arguments accepted by Agent.from_spec."""

        @dataclass
        class KeywordCapability(AbstractCapability[None]):
            minimum: int = 0

            @classmethod
            def from_spec(cls, *, minimum: int) -> Self:
                return cls(minimum=minimum)

        spec = {'model': 'test', 'capabilities': [{'KeywordCapability': {'minimum': 4}}]}
        Agent.from_spec(spec, custom_capability_types=[KeywordCapability])
        schema = AgentSpec.model_json_schema_with_capabilities([KeywordCapability])
        validator = validator_for(schema)(schema)
        assert list(validator.iter_errors(spec)) == []
        scalar_spec = {'model': 'test', 'capabilities': [{'KeywordCapability': 4}]}
        assert list(validator.iter_errors(scalar_spec))
        with pytest.raises(ValueError, match='positional argument'):
            Agent.from_spec(scalar_spec, custom_capability_types=[KeywordCapability])

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    @pytest.mark.parametrize(
        ('arguments', 'weights'),
        [
            pytest.param({'weights': {'a': 1}}, {'a': 1}, id='keyword-form'),
            pytest.param(None, None, id='short-form'),
        ],
    )
    def test_object_param_capability_schema(self, arguments: JsonValue, weights: dict[str, int] | None):
        """Every form the schema accepts for a sole object-typed parameter loads as that parameter's value."""
        spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [{'ObjectParam': arguments}]}
        schema = AgentSpec.model_json_schema_with_capabilities([ObjectParam])
        assert list(validator_for(schema)(schema).iter_errors(spec)) == []
        capabilities = Agent.from_spec(spec, custom_capability_types=[ObjectParam]).root_capability.capabilities
        [capability] = [c for c in capabilities if isinstance(c, ObjectParam)]
        assert capability.weights == weights

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_object_param_capability_schema_rejects_object_short_form(self):
        """The short form rejects an object, since the loader reads one as keyword arguments."""
        schema = AgentSpec.model_json_schema_with_capabilities([ObjectParam])
        validator = validator_for(schema)(schema)

        unknown_keyword = {'model': 'test', 'capabilities': [{'ObjectParam': {'a': 1}}]}
        assert list(validator.iter_errors(unknown_keyword))
        with pytest.raises(ValueError, match="unexpected keyword argument 'a'"):
            Agent.from_spec(unknown_keyword, custom_capability_types=[ObjectParam])

        # `{'weights': 5}` is a valid `dict[str, int]`, but the loader would pass it as `weights=5`.
        mistyped = {'model': 'test', 'capabilities': [{'ObjectParam': {'weights': 5}}]}
        assert list(validator.iter_errors(mistyped))

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_model_param_capability_schema(self):
        """A sole model-typed parameter, whose schema is a `$ref`, takes its value in the keyword form only."""
        schema = AgentSpec.model_json_schema_with_capabilities([ModelParam])
        validator = validator_for(schema)(schema)
        keyword_form = {'model': 'test', 'capabilities': [{'ModelParam': {'limits': {'length': 4}}}]}
        assert list(validator.iter_errors(keyword_form)) == []
        capabilities = Agent.from_spec(keyword_form, custom_capability_types=[ModelParam]).root_capability.capabilities
        assert [c.limits for c in capabilities if isinstance(c, ModelParam)] == [Limits(length=4)]
        short_form = {'model': 'test', 'capabilities': [{'ModelParam': {'length': 4}}]}
        assert list(validator.iter_errors(short_form))

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    @pytest.mark.parametrize(
        ('arguments', 'valid', 'value'),
        [
            pytest.param(5, True, 5, id='short-form-scalar'),
            pytest.param({'value': {'a': 1}}, True, {'a': 1}, id='keyword-form-object'),
            pytest.param({'a': 1}, False, None, id='short-form-object'),
        ],
    )
    def test_any_param_capability_schema(self, arguments: JsonValue, valid: bool, value: JsonValue):
        """A sole `Any`-typed parameter, whose schema has no `type`, takes a string-keyed object in the keyword form only."""
        spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [{'AnyParam': arguments}]}
        schema = AgentSpec.model_json_schema_with_capabilities([AnyParam])
        assert (list(validator_for(schema)(schema).iter_errors(spec)) == []) is valid
        if valid:
            capabilities = Agent.from_spec(spec, custom_capability_types=[AnyParam]).root_capability.capabilities
            assert [c.value for c in capabilities if isinstance(c, AnyParam)] == [value]

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    @pytest.mark.parametrize(
        ('arguments', 'valid'),
        [
            pytest.param({'weights': {'a': 1}}, True, id='keyword-form'),
            pytest.param({'a': 1}, False, id='short-form-object'),
            pytest.param({}, False, id='short-form-empty'),
        ],
    )
    def test_multi_param_object_capability_schema(self, arguments: JsonValue, valid: bool):
        """A sole required object-typed parameter among several takes a string-keyed object in the keyword form only."""
        spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [{'WeightedParam': arguments}]}
        schema = AgentSpec.model_json_schema_with_capabilities([WeightedParam])
        assert (list(validator_for(schema)(schema).iter_errors(spec)) == []) is valid
        if valid:
            capabilities = Agent.from_spec(spec, custom_capability_types=[WeightedParam]).root_capability.capabilities
            assert [c.weights for c in capabilities if isinstance(c, WeightedParam)] == [{'a': 1}]
        else:
            with pytest.raises(ValueError, match="Failed to instantiate capability 'WeightedParam'"):
                Agent.from_spec(spec, custom_capability_types=[WeightedParam])

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_empty_capability_entry_schema(self):
        """The schema rejects an empty capability entry, which names no capability."""
        spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [{}]}
        schema = AgentSpec.model_json_schema_with_capabilities()
        assert list(validator_for(schema)(schema).iter_errors(spec))
        with pytest.raises(ValidationError, match='Field required'):
            Agent.from_spec(spec)

    def test_schema_error_param_capability_schema(self):
        """A short-form parameter type with no core schema raises from the full schema, which names the field."""

        class Session(Protocol):
            id: str

        @dataclass
        class SessionParam(AbstractCapability[None]):
            session: Session
            label: str = 'x'

        with pytest.raises(SchemaError, match='Field "capabilities"'):
            AgentSpec.model_json_schema_with_capabilities([SessionParam])

    @pytest.mark.parametrize('capability', [ExternalRefParam, RefCycleParam])
    def test_unresolved_ref_param_capability_schema(self, capability: type[AbstractCapability[None]]):
        """A sole parameter whose schema has a `$ref` outside `$defs`, or a `$ref` cycle, keeps its short form."""
        name = capability.get_serialization_name()
        schema = AgentSpec.model_json_schema_with_capabilities([capability])
        assert [key for key in schema['$defs'] if key.endswith(f'_{name}')] == [f'short_spec_{name}']
        spec = {'model': 'test', 'capabilities': [name]}
        capabilities = Agent.from_spec(spec, custom_capability_types=[capability]).root_capability.capabilities
        assert [type(c) for c in capabilities if isinstance(c, capability)] == [capability]

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_unrepresentable_param_capability_schema(self):
        """A sole parameter with no JSON schema drops the capability's short form instead of raising."""

        class Client: ...

        @dataclass
        class ClientParam(AbstractCapability[None]):
            @classmethod
            def from_spec(cls, client: Client | None = None) -> Self:
                return cls()

        schema = AgentSpec.model_json_schema_with_capabilities([ClientParam])
        assert [key for key in schema['$defs'] if key.endswith('_ClientParam')] == []
        name_only = {'model': 'test', 'capabilities': ['ClientParam']}
        assert list(validator_for(schema)(schema).iter_errors(name_only)) == []
        capabilities = Agent.from_spec(name_only, custom_capability_types=[ClientParam]).root_capability.capabilities
        assert [c for c in capabilities if isinstance(c, ClientParam)] == [ClientParam()]

    def test_no_params_class(self):
        class NoParams:
            def __init__(self): ...  # pragma: no cover

        registry = {'NoParams': NoParams}
        types = build_schema_types(registry)
        # Should include the Literal[name] form
        assert len(types) >= 1

    def test_single_param_class(self):
        class SingleParam:
            def __init__(self, x: int): ...  # pragma: no cover

        registry = {'SingleParam': SingleParam}
        types = build_schema_types(registry)
        assert len(types) >= 1

    def test_multi_param_class(self):
        class MultiParam:
            def __init__(self, x: int, y: str): ...  # pragma: no cover

        registry = {'MultiParam': MultiParam}
        types = build_schema_types(registry)
        assert len(types) >= 1

    def test_custom_schema_target(self):
        def my_factory(name: str, count: int = 1) -> None: ...  # pragma: no cover

        registry = {'MyClass': object}
        types = build_schema_types(
            registry,
            get_schema_target=lambda c: my_factory,
        )
        assert len(types) >= 1
