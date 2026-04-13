"""Shared spec utilities for constructing objects from YAML/JSON/dict specifications.

This module provides the `NamedSpec` class (generalized from `EvaluatorSpec` in pydantic_evals)
and registry/loading utilities that can be reused by both the evaluator system and the capability system.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    RootModel,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import to_jsonable_python
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler
from typing_extensions import NotRequired, TypedDict

from pydantic_ai._utils import get_function_type_hints

if TYPE_CHECKING:
    from pydantic import ModelWrapValidatorHandler

T = TypeVar('T')


def serializes_as_string_keyed_dict(value: Any) -> bool:
    """Check if a value would serialize to a dict with all string keys.

    When serialize() uses the compact tuple form (arguments = (value,)), the serialized
    output becomes {Name: value}. On deserialization, _SerializedNamedSpec._args
    treats any dict with all-string keys as kwargs. This means a single positional argument
    that is itself a dict (like ModelSettings) would be incorrectly unpacked as kwargs
    on the round-trip. We avoid the compact form in this case.
    """
    jsonable = to_jsonable_python(value, serialize_unknown=True)
    return isinstance(jsonable, dict) and all(isinstance(k, str) for k in jsonable)  # pyright: ignore[reportUnknownVariableType]


class NamedSpec(BaseModel):
    """A specification for constructing a named object from serialized arguments.

    Supports three short forms:
    * `'MyClass'` — no arguments
    * `{'MyClass': single_arg}` — a single positional argument
    * `{'MyClass': {k1: v1, k2: v2}}` — keyword arguments
    """

    name: str
    """The name of the class to construct."""

    arguments: None | tuple[Any] | dict[str, Any]
    """The arguments to pass to the constructor.

    Can be None (no arguments), a tuple (a single positional argument), or a dict (keyword arguments).
    """

    @property
    def args(self) -> tuple[Any, ...]:
        """Get the positional arguments."""
        if isinstance(self.arguments, tuple):
            return self.arguments
        return ()

    @property
    def kwargs(self) -> dict[str, Any]:
        """Get the keyword arguments."""
        if isinstance(self.arguments, dict):
            return self.arguments
        return {}

    @model_validator(mode='wrap')
    @classmethod
    def deserialize(cls, value: Any, handler: ModelWrapValidatorHandler[NamedSpec]) -> NamedSpec:
        """Deserialize a NamedSpec from various formats."""
        try:
            return handler(value)
        except ValidationError as exc:
            try:
                deserialized = _SerializedNamedSpec.model_validate(value)
            except ValidationError:
                raise exc  # raise the original error
            return deserialized.to_named_spec(cls)

    @model_serializer(mode='wrap')
    def serialize(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> Any:
        """Serialize using the appropriate short-form if possible."""
        if isinstance(info.context, dict) and info.context.get('use_short_form'):  # pyright: ignore[reportUnknownMemberType]
            if self.arguments is None:
                return self.name
            elif isinstance(self.arguments, tuple):
                # A single positional arg that serializes as a string-keyed dict would be
                # misinterpreted as kwargs on deserialization. Fall back to the long form.
                if serializes_as_string_keyed_dict(self.arguments[0]):
                    return handler(self)
                return {self.name: self.arguments[0]}
            else:
                return {self.name: self.arguments}
        else:
            return handler(self)


class _SerializedNamedSpec(RootModel[str | dict[str, Any]]):
    """Internal class for handling the serialized form of a NamedSpec."""

    @field_validator('root')
    @classmethod
    def enforce_one_key(cls, value: str | dict[str, Any]) -> Any:
        """Enforce that the root value has exactly one key when it is a dict."""
        if isinstance(value, str):
            return value
        if len(value) != 1:
            raise ValueError(f'Expected a single key containing the class name, found keys {list(value.keys())}')
        return value

    @property
    def _name(self) -> str:
        if isinstance(self.root, str):
            return self.root
        return next(iter(self.root.keys()))

    @property
    def _args(self) -> None | tuple[Any] | dict[str, Any]:
        if isinstance(self.root, str):
            return None

        value = next(iter(self.root.values()))

        if isinstance(value, dict):
            keys: list[Any] = list(value.keys())  # pyright: ignore[reportUnknownArgumentType]
            if all(isinstance(k, str) for k in keys):
                return cast(dict[str, Any], value)

        # Anything else is passed as a single positional argument
        return (cast(Any, value),)

    def to_named_spec(self, cls: type[NamedSpec] = NamedSpec) -> NamedSpec:
        return cls(name=self._name, arguments=self._args)


class CapabilitySpec(NamedSpec):
    """A capability specification, distinguishable from other NamedSpec types for schema generation.

    In JSON schemas, fields typed as CapabilitySpec are replaced with the full
    capability Union (the same set of types used in `AgentSpec.capabilities`).
    """


def build_registry(
    *,
    custom_types: Sequence[type[T]],
    defaults: Sequence[type[T]],
    get_name: Callable[[type[T]], str | None],
    label: str,
    validate: Callable[[type[T]], None] | None = None,
) -> Mapping[str, type[T]]:
    """Create a registry of types from default and custom types.

    Args:
        custom_types: Additional classes to include in the registry.
        defaults: Default classes to include (can be overridden by custom types).
        get_name: Callable to get the serialization name from a class. Return None to opt out.
        label: Human-readable label for error messages.
        validate: Optional callback to validate each custom type.

    Returns:
        A mapping from names to classes.
    """
    registry: dict[str, type[T]] = {}

    for cls in custom_types:
        if validate is not None:
            validate(cls)
        name = get_name(cls)
        if name is None:
            raise ValueError(f'Custom {label} class {cls.__name__} has opted out of serialization (name is None)')
        if name in registry:
            raise ValueError(f'Duplicate {label} class name: {name!r}')
        registry[name] = cls

    for cls in defaults:
        name = get_name(cls)
        if name is not None:
            # Allow overriding the defaults with custom types without raising an error
            registry.setdefault(name, cls)

    return registry


def load_from_registry(
    registry: Mapping[str, type[T]],
    spec: NamedSpec,
    *,
    label: str,
    custom_types_param: str,
    context: str | None = None,
    instantiate: Callable[[type[T], tuple[Any, ...], dict[str, Any]], T] | None = None,
) -> T:
    """Load an object from the registry based on a specification.

    Args:
        registry: Mapping from names to classes.
        spec: Specification of the object to load.
        label: Human-readable label for error messages.
        custom_types_param: Name of the parameter for custom types, used in error messages.
        context: Optional context for error messages.
        instantiate: Optional callback to instantiate the class. Default: `cls(*args, **kwargs)`.

    Returns:
        An initialized instance.
    """
    cls = registry.get(spec.name)
    if cls is None:
        raise ValueError(
            f'{label.capitalize()} {spec.name!r} is not in the provided `{custom_types_param}`. Valid choices: {list(registry.keys())}.'
            f' If you are trying to use a custom {label}, you must include its type in the `{custom_types_param}` argument.'
        )
    try:
        if instantiate is not None:
            return instantiate(cls, spec.args, spec.kwargs)
        else:
            return cls(*spec.args, **spec.kwargs)
    except Exception as e:
        detail = f' for {context}' if context else ''
        raise ValueError(f'Failed to instantiate {label} {spec.name!r}{detail}: {e}') from e


def filter_serializable_type(tp: Any) -> Any | None:
    """Filter a type to only include members that can be represented in JSON schema.

    For Union types, removes non-serializable members (TypeVars, Callables).
    Returns None if the type is entirely non-serializable.
    """
    # TypeVar is not serializable
    if isinstance(tp, TypeVar):
        return None

    origin = typing.get_origin(tp)

    # Callable is not serializable
    if origin is Callable:
        return None

    # Union: filter members
    if origin is typing.Union or isinstance(tp, types.UnionType):
        args = typing.get_args(tp)
        filtered = [fa for a in args if (fa := filter_serializable_type(a)) is not None]
        if not filtered:
            return None
        if len(filtered) == 1:
            return filtered[0]
        return typing.Union[tuple(filtered)]  # noqa: UP007

    # Other generics (list[X], dict[X, Y]): all args must be serializable
    args = typing.get_args(tp)
    if args and any(filter_serializable_type(a) is None for a in args):
        return None

    return tp


def build_schema_types(
    registry: Mapping[str, type[Any]],
    *,
    get_schema_target: Callable[[type[Any]], Any] | None = None,
) -> list[Any]:
    """Build a list of schema types from a registry for JSON schema generation.

    Args:
        registry: Mapping from names to classes.
        get_schema_target: Optional callback to get the schema target (e.g. `from_spec` method)
            from a class. Default: use the class itself.

    Returns:
        A list of types suitable for use in a Union for JSON schema generation.
    """
    schema_types: list[Any] = []
    for name, cls in registry.items():
        target = get_schema_target(cls) if get_schema_target is not None else cls
        type_hints = get_function_type_hints(target)
        type_hints.pop('return', None)

        # Filter out non-serializable types (TypeVars, Callables) from unions
        type_hints = {k: fv for k, v in type_hints.items() if (fv := filter_serializable_type(v)) is not None}

        required_type_hints: dict[str, Any] = {}

        for p in inspect.signature(target).parameters.values():
            # Skip self/cls (unbound instance/class methods) and *args/**kwargs
            if p.name in ('self', 'cls') and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                type_hints.pop(p.name, None)
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                type_hints.pop(p.name, None)
                continue
            # Skip params whose type was entirely filtered out
            if p.name not in type_hints:
                continue
            type_hints.setdefault(p.name, Any)
            if p.default is not p.empty:
                type_hints[p.name] = NotRequired[type_hints[p.name]]
            else:
                required_type_hints[p.name] = type_hints[p.name]

        def _make_typed_dict(cls_name_prefix: str, fields: dict[str, Any]) -> Any:
            td = TypedDict(f'{cls_name_prefix}_{name}', fields)  # pyright: ignore[reportArgumentType]
            config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
            # TODO: Replace with pydantic.with_config once pydantic 2.11 is the min supported version
            td.__pydantic_config__ = config  # pyright: ignore[reportAttributeAccessIssue]
            return td

        # Shortest form: just the name
        if len(type_hints) == 0 or not required_type_hints:
            schema_types.append(Literal[name])

        # Short form: can be called with only one parameter
        if len(type_hints) == 1:
            [type_hint_type] = type_hints.values()
            schema_types.append(_make_typed_dict('short_spec', {name: type_hint_type}))
        elif len(required_type_hints) == 1:  # pragma: no branch
            [type_hint_type] = required_type_hints.values()
            schema_types.append(_make_typed_dict('short_spec', {name: type_hint_type}))

        # Long form: multiple parameters, possibly required
        if len(type_hints) > 1:
            params_td = _make_typed_dict('spec_params', type_hints)
            schema_types.append(_make_typed_dict('spec', {name: params_td}))

    return schema_types
