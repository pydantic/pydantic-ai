"""Tests for the shared _spec.py module."""

from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai._spec import NamedSpec, build_registry, build_schema_types, load_from_registry


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
            if cls.__name__ == 'Foo':
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


class TestBuildSchemaTypes:
    def test_no_params_class(self):
        class NoParams:
            def __init__(self): ...

        registry = {'NoParams': NoParams}
        types = build_schema_types(registry)
        # Should include the Literal[name] form
        assert len(types) >= 1

    def test_single_param_class(self):
        class SingleParam:
            def __init__(self, x: int): ...

        registry = {'SingleParam': SingleParam}
        types = build_schema_types(registry)
        assert len(types) >= 1

    def test_multi_param_class(self):
        class MultiParam:
            def __init__(self, x: int, y: str): ...

        registry = {'MultiParam': MultiParam}
        types = build_schema_types(registry)
        assert len(types) >= 1

    def test_custom_schema_target(self):
        def my_factory(name: str, count: int = 1) -> None: ...

        registry = {'MyClass': object}
        types = build_schema_types(
            registry,
            get_schema_target=lambda c: my_factory,
        )
        assert len(types) >= 1
