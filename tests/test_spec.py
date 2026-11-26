"""Tests for the shared _spec.py module."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Protocol, Self

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, GetPydanticSchema, JsonValue, TypeAdapter, ValidationError
from pydantic_core import SchemaError
from typing_extensions import TypeAliasType

from pydantic_ai import Agent
from pydantic_ai._spec import NamedSpec, build_registry, build_schema_types, load_from_registry
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, SetToolMetadata
from pydantic_ai.tools import AgentDepsT

from .conftest import try_import

with try_import() as jsonschema_imports_successful:
    from jsonschema.validators import validator_for

if TYPE_CHECKING:
    from decimal import Decimal


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


@dataclass
class CallableParam(AbstractCapability[None]):
    key: Callable[[str], str]
    limit: int = 1


@dataclass
class TypeVarGenericParam(AbstractCapability[AgentDepsT]):
    inner: AbstractCapability[AgentDepsT]
    limit: int = 1


@dataclass
class KeywordOnlyVarKeyword(AbstractCapability[None]):
    minimum: int = 0

    @classmethod
    def from_spec(cls, *, minimum: int, **kwargs: object) -> Self:
        return cls(minimum=minimum)


@dataclass
class KeywordOnlyAmongSeveralVarKeyword(AbstractCapability[None]):
    minimum: int = 0

    @classmethod
    def from_spec(cls, *, minimum: int, label: str | None = None, **unsupported: object) -> Self:
        return cls(minimum=minimum)


@dataclass
class OpenMetadata(AbstractCapability[None]):
    tools: list[str] | Literal['all'] | dict[str, JsonValue] = 'all'
    metadata: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])

    @classmethod
    def from_spec(
        cls, *, tools: list[str] | Literal['all'] | dict[str, JsonValue] = 'all', **metadata: JsonValue
    ) -> Self:
        return cls(tools=tools, metadata=metadata)


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
        ('capability', 'arguments', 'minimum'),
        [
            pytest.param(KeywordOnlyVarKeyword, {'minimum': 4}, 4, id='keyword-form'),
            pytest.param(KeywordOnlyVarKeyword, 4, None, id='short-form'),
            pytest.param(KeywordOnlyAmongSeveralVarKeyword, {'minimum': 4}, 4, id='among-several-keyword-form'),
            pytest.param(KeywordOnlyAmongSeveralVarKeyword, 4, None, id='among-several-short-form'),
        ],
    )
    def test_keyword_only_var_keyword_capability_schema(
        self,
        capability: type[KeywordOnlyVarKeyword] | type[KeywordOnlyAmongSeveralVarKeyword],
        arguments: JsonValue,
        minimum: int | None,
    ):
        """A keyword-only parameter of a target taking `**kwargs` takes its value in the keyword form only.

        The loader passes a short form's value positionally, which such a parameter rejects.
        """
        name = capability.__name__
        spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [{name: arguments}]}
        schema = AgentSpec.model_json_schema_with_capabilities([capability])
        assert (list(validator_for(schema)(schema).iter_errors(spec)) == []) is (minimum is not None)
        if minimum is None:
            with pytest.raises(ValueError, match='positional argument'):
                Agent.from_spec(spec, custom_capability_types=[capability])
        else:
            capabilities = Agent.from_spec(spec, custom_capability_types=[capability]).root_capability.capabilities
            assert [c.minimum for c in capabilities if isinstance(c, capability)] == [minimum]

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    @pytest.mark.parametrize(
        ('arguments', 'tools', 'metadata'),
        [
            pytest.param({'code_mode': True}, 'all', {'code_mode': True}, id='metadata-only'),
            pytest.param({'tools': ['a'], 'code_mode': True}, ['a'], {'code_mode': True}, id='tools-and-metadata'),
        ],
    )
    def test_open_metadata_capability_schema(
        self, arguments: JsonValue, tools: list[str] | Literal['all'], metadata: dict[str, JsonValue]
    ):
        """A target taking `**kwargs` keeps its short form for a keyword-only parameter whose type accepts an object.

        The loader reads that short form's object as keyword arguments, which the target takes.
        """
        spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [{'OpenMetadata': arguments}]}
        schema = AgentSpec.model_json_schema_with_capabilities([OpenMetadata])
        assert [key for key in schema['$defs'] if key.endswith('_OpenMetadata')] == ['short_spec_OpenMetadata']
        assert list(validator_for(schema)(schema).iter_errors(spec)) == []
        capabilities = Agent.from_spec(spec, custom_capability_types=[OpenMetadata]).root_capability.capabilities
        [capability] = [c for c in capabilities if isinstance(c, OpenMetadata)]
        assert (capability.tools, capability.metadata) == (tools, metadata)

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

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    @pytest.mark.parametrize('capability', [CallableParam, TypeVarGenericParam])
    def test_required_unrepresentable_param_capability_schema(self, capability: type[AbstractCapability[None]]):
        """A required parameter with no JSON form leaves the capability out of the schema, since no spec entry can load it.

        Not a VCR test: schema generation and `Agent.from_spec` run before any model request.
        """
        name = capability.__name__
        schema = AgentSpec.model_json_schema_with_capabilities([capability])
        assert [key for key in schema['$defs'] if key.endswith(f'_{name}')] == []
        validator = validator_for(schema)(schema)
        entries: list[JsonValue] = [name, {name: {'limit': 2}}]
        for entry in entries:
            spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [entry]}
            assert list(validator.iter_errors(spec))
            with pytest.raises(ValueError, match='missing 1 required positional argument'):
                Agent.from_spec(spec, custom_capability_types=[capability])

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_unannotated_required_param_capability_schema(self):
        """An unannotated required parameter accepts any value, so the schema rejects the name alone.

        Not a VCR test: schema generation and `Agent.from_spec` run before any model request.
        """

        @dataclass
        class UnannotatedParam(AbstractCapability[None]):
            threshold: int

            @classmethod
            def from_spec(cls, threshold) -> Self:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
                return cls(threshold=threshold)  # pyright: ignore[reportUnknownArgumentType]

        schema = AgentSpec.model_json_schema_with_capabilities([UnannotatedParam])
        validator = validator_for(schema)(schema)
        name_only = {'model': 'test', 'capabilities': ['UnannotatedParam']}
        assert list(validator.iter_errors(name_only))
        with pytest.raises(ValueError, match="missing 1 required positional argument: 'threshold'"):
            Agent.from_spec(name_only, custom_capability_types=[UnannotatedParam])
        entries: list[JsonValue] = [{'UnannotatedParam': 3}, {'UnannotatedParam': {'threshold': 3}}]
        for entry in entries:
            spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [entry]}
            assert list(validator.iter_errors(spec)) == []
            capabilities = Agent.from_spec(
                spec, custom_capability_types=[UnannotatedParam]
            ).root_capability.capabilities
            assert [c.threshold for c in capabilities if isinstance(c, UnannotatedParam)] == [3]

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_unannotated_optional_param_capability_schema(self):
        """An unannotated optional parameter leaves the schema of a sole object-typed parameter rejecting objects.

        Not a VCR test: schema generation and `Agent.from_spec` run before any model request.
        """

        @dataclass
        class WeightsWithFlag(AbstractCapability[None]):
            weights: dict[str, int]

            @classmethod
            def from_spec(cls, weights: dict[str, int], flag=False) -> Self:  # pyright: ignore[reportMissingParameterType]
                return cls(weights=weights)

        schema = AgentSpec.model_json_schema_with_capabilities([WeightsWithFlag])
        short_form_object = {'model': 'test', 'capabilities': [{'WeightsWithFlag': {'a': 1}}]}
        assert list(validator_for(schema)(schema).iter_errors(short_form_object))
        with pytest.raises(ValueError, match="unexpected keyword argument 'a'"):
            Agent.from_spec(short_form_object, custom_capability_types=[WeightsWithFlag])

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_type_checking_only_optional_hint_capability_schema(self):
        """An optional parameter whose hint resolves only under `TYPE_CHECKING` is left out; the others stay.

        Not a VCR test: schema generation and `Agent.from_spec` run before any model request.
        """

        @dataclass
        class PricedLabel(AbstractCapability[None]):
            label: str
            limit: int = 3
            rate: 'Decimal | None' = None

        schema = AgentSpec.model_json_schema_with_capabilities([PricedLabel])
        assert schema['$defs']['spec_params_PricedLabel'] == snapshot(
            {
                'additionalProperties': False,
                'properties': {
                    'label': {'title': 'Label', 'type': 'string'},
                    'limit': {'title': 'Limit', 'type': 'integer'},
                    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                    'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                    'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                },
                'required': ['label'],
                'title': 'spec_params_PricedLabel',
                'type': 'object',
            }
        )
        validator = validator_for(schema)(schema)
        entries: list[JsonValue] = [{'PricedLabel': 'x'}, {'PricedLabel': {'label': 'x', 'limit': 2}}]
        for entry in entries:
            spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [entry]}
            assert list(validator.iter_errors(spec)) == []
            capabilities = Agent.from_spec(spec, custom_capability_types=[PricedLabel]).root_capability.capabilities
            assert [c.label for c in capabilities if isinstance(c, PricedLabel)] == ['x']

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_type_checking_only_required_hint_capability_schema(self):
        """A required parameter whose hint resolves only under `TYPE_CHECKING` accepts any value, as if unannotated.

        Not a VCR test: schema generation and `Agent.from_spec` run before any model request.
        """

        @dataclass
        class Priced(AbstractCapability[None]):
            rate: 'Decimal'
            limit: int = 3

        schema = AgentSpec.model_json_schema_with_capabilities([Priced])
        assert schema['$defs']['spec_params_Priced'] == snapshot(
            {
                'additionalProperties': False,
                'properties': {
                    'limit': {'title': 'Limit', 'type': 'integer'},
                    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                    'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                    'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                    'rate': {'title': 'Rate'},
                },
                'required': ['rate'],
                'title': 'spec_params_Priced',
                'type': 'object',
            }
        )
        validator = validator_for(schema)(schema)
        name_only = {'model': 'test', 'capabilities': ['Priced']}
        assert list(validator.iter_errors(name_only))
        entries: list[JsonValue] = [{'Priced': 1.5}, {'Priced': {'rate': 1.5, 'limit': 2}}]
        for entry in entries:
            spec: dict[str, JsonValue] = {'model': 'test', 'capabilities': [entry]}
            assert list(validator.iter_errors(spec)) == []
            capabilities = Agent.from_spec(spec, custom_capability_types=[Priced]).root_capability.capabilities
            assert [c.rate for c in capabilities if isinstance(c, Priced)] == [1.5]

    @pytest.mark.skipif(not jsonschema_imports_successful(), reason='jsonschema not installed')
    def test_type_checking_only_hint_schema_error_capability_schema(self):
        """A class whose schema can't be built from the hints that resolve publishes its name alone, as before.

        Not a VCR test: schema generation and `Agent.from_spec` run before any model request.
        """

        class Session(Protocol):
            id: str

        @dataclass
        class PricedSession(AbstractCapability[None]):
            session: Session | None = None
            rate: 'Decimal | None' = None

        schema = AgentSpec.model_json_schema_with_capabilities([PricedSession])
        assert [key for key in schema['$defs'] if key.endswith('_PricedSession')] == []
        name_only = {'model': 'test', 'capabilities': ['PricedSession']}
        assert list(validator_for(schema)(schema).iter_errors(name_only)) == []
        capabilities = Agent.from_spec(name_only, custom_capability_types=[PricedSession]).root_capability.capabilities
        assert [c for c in capabilities if isinstance(c, PricedSession)] == [PricedSession()]

    def test_type_checking_only_hint_from_spec_schema(self):
        """A `from_spec` parameter whose hint resolves only under `TYPE_CHECKING` is left out; the others stay."""

        @dataclass
        class PricedFromSpec(AbstractCapability[None]):
            limit: int = 3

            @classmethod
            def from_spec(cls, limit: int = 3, rate: 'Decimal | None' = None) -> Self:
                return cls(limit=limit)  # pragma: no cover

        schema = AgentSpec.model_json_schema_with_capabilities([PricedFromSpec])
        assert {key: schema['$defs'][key] for key in schema['$defs'] if key.endswith('_PricedFromSpec')} == snapshot(
            {
                'short_spec_PricedFromSpec': {
                    'additionalProperties': False,
                    'properties': {'PricedFromSpec': {'title': 'Pricedfromspec', 'type': 'integer'}},
                    'required': ['PricedFromSpec'],
                    'title': 'short_spec_PricedFromSpec',
                    'type': 'object',
                }
            }
        )

    def test_type_checking_only_hint_inherited_from_spec_schema(self):
        """A class whose `__init__` hints don't all resolve keeps the schema of the `from_spec` override it inherits."""

        @dataclass
        class Limited(AbstractCapability[None]):
            limit: int = 3

            @classmethod
            def from_spec(cls, limit: int = 3) -> Self:
                return cls(limit=limit)  # pragma: no cover

        @dataclass
        class PricedLimited(Limited):
            rate: 'Decimal | None' = None

        schema = AgentSpec.model_json_schema_with_capabilities([PricedLimited])
        assert {key: schema['$defs'][key] for key in schema['$defs'] if key.endswith('_PricedLimited')} == snapshot(
            {
                'short_spec_PricedLimited': {
                    'additionalProperties': False,
                    'properties': {'PricedLimited': {'title': 'Pricedlimited', 'type': 'integer'}},
                    'required': ['PricedLimited'],
                    'title': 'short_spec_PricedLimited',
                    'type': 'object',
                }
            }
        )

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

    def test_dataclass_target_omits_non_init_annotations(self):
        """A class target's `ClassVar` and `init=False` annotations name no parameter, so no spec form takes them.

        Evals pass an evaluator class itself as its schema target.
        """

        @dataclass
        class Stateful:
            kind: ClassVar[str] = 'stateful'
            threshold: int
            label: str = 'x'
            derived: str = field(init=False)

        types = build_schema_types({'Stateful': Stateful})
        assert [TypeAdapter(t).json_schema() for t in types] == snapshot(
            [
                {
                    'additionalProperties': False,
                    'properties': {'Stateful': {'title': 'Stateful', 'type': 'integer'}},
                    'required': ['Stateful'],
                    'title': 'short_spec_Stateful',
                    'type': 'object',
                },
                {
                    '$defs': {
                        'spec_params_Stateful': {
                            'additionalProperties': False,
                            'properties': {
                                'threshold': {'title': 'Threshold', 'type': 'integer'},
                                'label': {'title': 'Label', 'type': 'string'},
                            },
                            'required': ['threshold'],
                            'title': 'spec_params_Stateful',
                            'type': 'object',
                        }
                    },
                    'additionalProperties': False,
                    'properties': {'Stateful': {'$ref': '#/$defs/spec_params_Stateful'}},
                    'required': ['Stateful'],
                    'title': 'spec_Stateful',
                    'type': 'object',
                },
            ]
        )

    def test_variadic_keyword_dataclass_target_omits_classvar(self):
        """A class target whose `__init__` takes `**kwargs` keeps every field in its schema, `init=False` included.

        `**kwargs` can take any field, but never a `ClassVar`. Each field is required, since the signature gives no
        defaults: https://github.com/pydantic/pydantic-ai/issues/13490.
        """

        @dataclass
        class Stateful:
            kind: ClassVar[str] = 'stateful'
            threshold: int
            derived: str = field(default='', init=False)

            def __init__(self, **kwargs: object) -> None: ...  # pragma: no cover

        types = build_schema_types({'Stateful': Stateful})
        assert [TypeAdapter(t).json_schema() for t in types] == snapshot(
            [
                {'const': 'Stateful', 'type': 'string'},
                {
                    '$defs': {
                        'spec_params_Stateful': {
                            'additionalProperties': False,
                            'properties': {
                                'threshold': {'title': 'Threshold', 'type': 'integer'},
                                'derived': {'title': 'Derived', 'type': 'string'},
                            },
                            'required': ['threshold', 'derived'],
                            'title': 'spec_params_Stateful',
                            'type': 'object',
                        }
                    },
                    'additionalProperties': False,
                    'properties': {'Stateful': {'$ref': '#/$defs/spec_params_Stateful'}},
                    'required': ['Stateful'],
                    'title': 'spec_Stateful',
                    'type': 'object',
                },
            ]
        )

    def test_custom_schema_target(self):
        def my_factory(name: str, count: int = 1) -> None: ...  # pragma: no cover

        registry = {'MyClass': object}
        types = build_schema_types(
            registry,
            get_schema_target=lambda c: my_factory,
        )
        assert len(types) >= 1
