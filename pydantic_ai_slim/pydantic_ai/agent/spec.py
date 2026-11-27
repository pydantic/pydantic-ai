"""Agent specification for constructing agents from YAML/JSON/dict specs."""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast

from pydantic import BaseModel, Field, model_serializer
from pydantic_core import from_json, to_json
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler

from pydantic_ai._agent_graph import EndStrategy
from pydantic_ai._spec import CapabilitySpec, build_registry, build_schema_types
from pydantic_ai._utils import get_function_type_hints, is_str_dict
from pydantic_ai.agent.abstract import AgentRetries
from pydantic_ai.exceptions import UserError
from pydantic_ai.settings import ModelSettings
from pydantic_ai.template import TemplateStr

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import AbstractCapability

__all__ = ['CapabilityNamespace', 'CapabilitySpec']  # `CapabilitySpec` is re-exported from _spec

DEFAULT_SCHEMA_PATH_TEMPLATE = './{stem}_schema.json'
"""Default template for schema file paths, where {stem} is replaced with the spec filename stem."""

_YAML_SCHEMA_LINE_PREFIX = '# yaml-language-server: $schema='


class AgentSpec(BaseModel):
    """Specification for constructing an Agent from a dict/YAML/JSON."""

    # $schema is included to avoid validation fails from the `$schema` key, see `_add_json_schema` below for context
    json_schema_path: str | None = Field(default=None, alias='$schema')
    model: str | None = None
    name: str | None = None
    description: TemplateStr[Any] | str | None = None
    instructions: TemplateStr[Any] | str | list[TemplateStr[Any] | str] | None = None
    deps_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    model_settings: dict[str, Any] | None = None
    retries: int | AgentRetries | None = None
    end_strategy: EndStrategy = 'graceful'
    tool_timeout: float | None = None
    metadata: dict[str, Any] | None = None
    capabilities: list[CapabilitySpec] = []

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        fmt: Literal['yaml', 'json'] | None = None,
    ) -> AgentSpec:
        """Load an agent spec from a YAML or JSON file.

        Args:
            path: Path to the file to load.
            fmt: Format of the file. If None, inferred from file extension.

        Returns:
            A new AgentSpec instance.
        """
        path = Path(path)
        fmt = _infer_fmt(path, fmt)
        # Not `encoding='utf-8-sig'`: `read_text` returns '' for a lone b'\xef' or b'\xef\xbb' instead of raising.
        content = path.read_text(encoding='utf-8').removeprefix('\ufeff')
        return cls.from_text(content, fmt=fmt)

    @classmethod
    def from_text(
        cls,
        text: str,
        fmt: Literal['yaml', 'json'] = 'yaml',
    ) -> AgentSpec:
        """Parse YAML or JSON text into an AgentSpec.

        Args:
            text: The string content to parse.
            fmt: Format of the content. Must be either 'yaml' or 'json'.

        Returns:
            A new AgentSpec instance.
        """
        if fmt == 'json':
            data = from_json(text)
        else:
            try:
                import yaml
            except ImportError:  # pragma: no cover
                # Reaching this requires PyYAML to not be installed.
                raise ImportError(
                    'PyYAML is required to load YAML agent specs. Install it with: pip install "pydantic-ai-slim[spec]"'
                ) from None
            data = yaml.safe_load(text)
        if not is_str_dict(data):
            raise UserError(f'Agent spec must parse to an object, got {type(data).__name__}')
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        """Validate a dictionary into an AgentSpec.

        Args:
            data: Dictionary representation of the agent spec.

        Returns:
            A new AgentSpec instance.
        """
        return cls.model_validate(data)

    def to_file(
        self,
        path: Path | str,
        fmt: Literal['yaml', 'json'] | None = None,
        schema_path: Path | str | None = DEFAULT_SCHEMA_PATH_TEMPLATE,
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        capability_namespace: CapabilityNamespace | None = None,
    ) -> None:
        """Save the agent spec to a YAML or JSON file.

        Args:
            path: Path to save the spec to.
            fmt: Format to use. If None, inferred from file extension.
            schema_path: Path to save the JSON schema to. If None, no schema will be saved.
                Can be a string template with {stem} which will be replaced with the spec filename stem.
            custom_capability_types: Custom capability classes to include in the schema.
            capability_namespace: A namespace whose capabilities to include in the schema, which
                imports every class it names.
        """
        path = Path(path)
        fmt = _infer_fmt(path, fmt)

        schema_ref: str | None = None
        if schema_path is not None:
            if isinstance(schema_path, str):
                schema_path = Path(schema_path.format(stem=path.stem))

            if not schema_path.is_absolute():
                schema_ref = str(schema_path)
                schema_path = path.parent / schema_path
            elif schema_path.is_relative_to(path.parent):
                schema_ref = str(schema_path.relative_to(path.parent))
            else:
                schema_ref = str(schema_path)
            self._save_schema(schema_path, custom_capability_types, capability_namespace)

        context: dict[str, Any] = {'use_short_form': True}
        if fmt == 'yaml':
            try:
                import yaml
            except ImportError:  # pragma: no cover
                # Reaching this requires PyYAML to not be installed.
                raise ImportError(
                    'PyYAML is required to save YAML agent specs. Install it with: pip install "pydantic-ai-slim[spec]"'
                ) from None
            dumped_data = self.model_dump(mode='json', by_alias=True, context=context, exclude_defaults=True)
            content = yaml.dump(dumped_data, sort_keys=False, allow_unicode=True)
            if schema_ref:
                content = f'{_YAML_SCHEMA_LINE_PREFIX}{schema_ref}\n{content}'
            path.write_text(content, encoding='utf-8')
        else:
            context['$schema'] = schema_ref
            json_data = self.model_dump_json(indent=2, by_alias=True, context=context, exclude_defaults=True)
            path.write_text(json_data + '\n', encoding='utf-8')

    @model_serializer(mode='wrap')
    def _add_json_schema(self, nxt: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict[str, Any]:
        """Add the JSON schema path to the serialized output when provided via context."""
        context = cast(dict[str, Any] | None, info.context)
        if isinstance(context, dict) and (schema := context.get('$schema')):
            return {'$schema': schema} | nxt(self)
        return nxt(self)

    @classmethod
    def model_json_schema_with_capabilities(
        cls,
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        capability_namespace: CapabilityNamespace | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON schema for this agent spec type, including capability details.

        This is useful for generating a schema that can be used to validate YAML-format agent spec files.

        Args:
            custom_capability_types: Custom capability classes to include in the schema.
            capability_namespace: A namespace whose capabilities to include in the schema, which
                imports every class it names.

        Returns:
            A dictionary representing the JSON schema.
        """
        capability_schema_types = _build_capability_schema_types(
            get_capability_registry(custom_capability_types, capability_namespace)
        )

        # Build a schema-only model with the resolved capability union.
        # NOTE: This duplicates the field list from AgentSpec above. We can't inherit from
        # AgentSpec because the types intentionally differ for schema generation:
        # - TemplateStr is replaced with plain str (templates are just strings in YAML/JSON)
        # - capabilities uses a resolved Union of typed schema models instead of CapabilitySpec
        # - extra='forbid' enables strict validation in the generated schema
        # When adding or removing fields on AgentSpec, update this class to match.
        class _AgentSpecSchema(BaseModel, extra='forbid', arbitrary_types_allowed=True):
            model: str | None = None
            name: str | None = None
            description: str | None = None
            instructions: str | list[str] | None = None
            deps_schema: dict[str, Any] | None = None
            output_schema: dict[str, Any] | None = None
            model_settings: ModelSettings | None = None
            retries: int | AgentRetries | None = None
            end_strategy: EndStrategy = 'graceful'
            tool_timeout: float | None = None
            metadata: dict[str, Any] | None = None
            if capability_schema_types:  # pragma: no branch
                capabilities: list[Union[tuple(capability_schema_types)]] = []  # pyright: ignore[reportUnknownVariableType, reportInvalidTypeArguments, reportInvalidTypeForm]  # noqa: UP007

        json_schema = _AgentSpecSchema.model_json_schema()
        json_schema['title'] = 'AgentSpec'
        json_schema['properties']['$schema'] = {'type': 'string'}

        # ModelSettings should allow additional properties for provider-specific settings;
        # extra='forbid' on _AgentSpecSchema propagates additionalProperties:false to nested
        # types, so we remove it from ModelSettings.
        model_settings_def: dict[str, Any] = json_schema.get('$defs', {}).get('ModelSettings', {})
        model_settings_def.pop('additionalProperties', None)

        # Replace CapabilitySpec $refs with the capability items Union,
        # so nested capability fields (e.g. PrefixTools.capability) show
        # the same rich schema as the top-level capabilities array.
        cap_items_schema = json_schema['properties']['capabilities']['items']
        _replace_capability_spec_refs(json_schema, cap_items_schema)

        return json_schema

    @classmethod
    def _save_schema(
        cls,
        path: Path | str,
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        capability_namespace: CapabilityNamespace | None = None,
    ) -> None:
        """Save the JSON schema for this agent spec type to a file.

        Args:
            path: Path to save the schema to.
            custom_capability_types: Custom capability classes to include in the schema.
            capability_namespace: A namespace whose capabilities to include in the schema.
        """
        path = Path(path)
        json_schema = cls.model_json_schema_with_capabilities(custom_capability_types, capability_namespace)
        schema_content = to_json(json_schema, indent=2).decode() + '\n'
        if not path.exists() or path.read_text(encoding='utf-8') != schema_content:
            path.write_text(schema_content, encoding='utf-8')


def _infer_fmt(path: Path, fmt: Literal['yaml', 'json'] | None) -> Literal['yaml', 'json']:
    """Infer the format to use for a file based on its extension."""
    if fmt is not None:
        return fmt
    suffix = path.suffix.lower()
    if suffix in {'.yaml', '.yml'}:
        return 'yaml'
    elif suffix == '.json':
        return 'json'
    raise ValueError(
        f'Could not infer format for filename {path.name!r}. Use the `fmt` argument to specify the format.'
    )


@dataclass(frozen=True, kw_only=True)
class CapabilityNamespace:
    """Where the capability names in an agent spec resolve to classes.

    A spec names a capability by its bare class name, such as `- Coder: {}`, and never by an import
    path, so the caller decides which class each name means. A name resolves in the first of these
    tiers that has it, in this order: `types`, `names`, `entry_points`, `modules`, then the built-in
    capabilities. A higher tier therefore deliberately overrides a lower one, including a built-in,
    while two different classes for one name within a single tier raise an error.

    `names` and `entry_points` only point at a `module:attribute` target, and nothing is imported
    until a spec actually uses that name. `modules` are imported when the namespace is used, since
    finding the capabilities in a module means importing it.
    """

    types: Sequence[type[AbstractCapability[Any]]] = ()
    """Capability classes, as passed to `custom_capability_types`."""

    names: Mapping[str, str] = field(default_factory=dict[str, str])
    """Capability names mapped to the `module:attribute` target of their class, for example
    `{'Coder': 'pydantic_ai_harness.coder:Coder'}`."""

    entry_points: str | None = None
    """An [entry point](https://packaging.python.org/en/latest/specifications/entry-points/) group
    whose entries map capability names to `module:attribute` targets, so every installed package can
    contribute to it. Packages that publish capabilities register them in `'pydantic_ai.capabilities'`."""

    modules: Sequence[str] = ()
    """Modules whose capability classes are all made available: those listed in the module's
    `__all__`, or else its public attributes."""


_Target = Union['type[AbstractCapability[Any]]', str]
"""A registry entry: a class, or the `module:attribute` target it is imported from on first use."""


class _CapabilityRegistry(Mapping[str, 'type[AbstractCapability[Any]]']):
    """A capability registry whose `module:attribute` entries are imported on first lookup."""

    def __init__(self, entries: dict[str, _Target]) -> None:
        self._entries = entries

    def __getitem__(self, name: str) -> type[AbstractCapability[Any]]:
        entry = self._entries[name]
        if isinstance(entry, str):
            entry = _resolve_capability_target(name, entry)
            self._entries[name] = entry
        return entry

    def __contains__(self, name: object) -> bool:
        # `Mapping.__contains__` would look the name up, importing its target.
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


def _validate_capability_type(cls: object) -> type[AbstractCapability[Any]]:
    from pydantic_ai.capabilities.abstract import AbstractCapability

    if not (isinstance(cls, type) and issubclass(cls, AbstractCapability)):
        raise ValueError(f'All custom capability classes must be subclasses of AbstractCapability, but {cls} is not')
    if '__dataclass_fields__' not in cls.__dict__:
        raise ValueError(f'All custom capability classes must be decorated with `@dataclass`, but {cls} is not')
    return cast('type[AbstractCapability[Any]]', cls)


def _resolve_capability_target(name: str, target: str) -> type[AbstractCapability[Any]]:
    try:
        cls = _validate_capability_type(pkgutil.resolve_name(target))
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(f'Capability {name!r} could not be loaded from `{target}`: {e}') from e
    if cls.get_serialization_name() != name:
        raise ValueError(
            f'Capability {name!r} points at `{target}`, whose serialization name is'
            f' {cls.get_serialization_name()!r}, not {name!r}'
        )
    return cls


def _module_capability_types(module_name: str) -> list[type[AbstractCapability[Any]]]:
    from pydantic_ai.capabilities.abstract import AbstractCapability

    module = importlib.import_module(module_name)
    exported: Sequence[str] | None = getattr(module, '__all__', None)
    # An empty `__all__` exports nothing; only a missing one falls back to the public attributes.
    attribute_names: list[str] = list(
        exported if exported is not None else (n for n in vars(module) if not n.startswith('_'))
    )
    found: dict[type[AbstractCapability[Any]], None] = {}
    for attribute_name in attribute_names:
        value = getattr(module, attribute_name, None)
        if (
            isinstance(value, type)
            and issubclass(value, AbstractCapability)
            and '__dataclass_fields__' in value.__dict__
            and value.get_serialization_name() is not None
        ):
            found[value] = None
    return list(found)


def _named_tier(items: Sequence[tuple[str, _Target]], label: str) -> dict[str, _Target]:
    tier: dict[str, _Target] = {}
    for name, entry in items:
        if name in tier and tier[name] != entry:
            raise ValueError(f'Two different capabilities are named {name!r} in {label}: `{tier[name]}` and `{entry}`')
        tier[name] = entry
    return tier


def get_capability_registry(
    custom_types: Sequence[type[AbstractCapability[Any]]] = (),
    namespace: CapabilityNamespace | None = None,
) -> Mapping[str, type[AbstractCapability[Any]]]:
    """Create a registry of capability types from the built-ins, `custom_types` and a `namespace`."""
    from pydantic_ai.capabilities import CAPABILITY_TYPES

    namespace = namespace or CapabilityNamespace()
    types = build_registry(
        # The same class passed through both doors is one capability, as in the other tiers.
        custom_types=[_validate_capability_type(cls) for cls in dict.fromkeys((*custom_types, *namespace.types))],
        defaults=(),
        get_name=lambda cls: cls.get_serialization_name(),
        label='capability',
    )
    built_ins = {name: cls for cls in CAPABILITY_TYPES.values() if (name := cls.get_serialization_name()) is not None}

    entries: dict[str, _Target] = dict(built_ins)
    entries.update(
        _named_tier(
            [
                (name, cls)
                for module_name in namespace.modules
                for cls in _module_capability_types(module_name)
                if (name := cls.get_serialization_name()) is not None
            ],
            '`modules`',
        )
    )
    if namespace.entry_points is not None:
        entries.update(
            _named_tier(
                # `module:attr` without any `[extras]` suffix, which `EntryPoint.load()` ignores too.
                [
                    (ep.name, f'{ep.module}:{ep.attr}')
                    for ep in importlib.metadata.entry_points(group=namespace.entry_points)
                ],
                f'the `{namespace.entry_points}` entry points',
            )
        )
    entries.update(namespace.names)
    entries.update(types)
    return _CapabilityRegistry(entries)


class CapabilitySpecContext:
    """Holds the registry and instantiation callback for the current spec-loading scope."""

    __slots__ = ('registry', 'instantiate')

    def __init__(
        self,
        registry: Mapping[str, type[AbstractCapability[Any]]],
        instantiate: Callable[
            [type[AbstractCapability[Any]], tuple[Any, ...], dict[str, Any]], AbstractCapability[Any]
        ],
    ) -> None:
        self.registry = registry
        self.instantiate = instantiate


capability_spec_context: ContextVar[CapabilitySpecContext | None] = ContextVar('capability_spec_context', default=None)


def load_capability_from_nested_spec(spec: CapabilitySpec | dict[str, Any] | str) -> AbstractCapability[Any]:
    """Load a capability from a nested spec, reusing the current spec-loading context.

    When called inside `Agent.from_spec()` or `Agent._resolve_spec()`, this uses the same
    registry (including custom capability types) and template context as the outer loading.
    When called outside a spec-loading context, falls back to the default registry.

    This is intended for use in `from_spec()` methods of wrapper capabilities like
    [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] that need to instantiate
    a nested capability from a spec argument.
    """
    from pydantic_ai._spec import load_from_registry

    cap_spec = spec if isinstance(spec, CapabilitySpec) else CapabilitySpec.model_validate(spec)
    ctx = capability_spec_context.get()
    if ctx is not None:
        return load_from_registry(
            ctx.registry,
            cap_spec,
            label='capability',
            custom_types_param=('custom_capability_types', 'capability_namespace'),
            instantiate=ctx.instantiate,
        )
    else:
        return load_from_registry(
            get_capability_registry(),
            cap_spec,
            label='capability',
            custom_types_param=('custom_capability_types', 'capability_namespace'),
            instantiate=lambda cap_cls, args, kwargs: cap_cls.from_spec(*args, **kwargs),
        )


def _build_capability_schema_types(registry: Mapping[str, type[Any]]) -> list[Any]:
    """Build a list of schema types for capabilities from a registry."""

    def _get_schema_target(cls: type[Any]) -> Any:
        # When from_spec is not overridden, it delegates to cls(*args, **kwargs).
        # Use __init__ directly so build_schema_types sees the actual parameter types.
        # Fall back to from_spec if __init__ hints can't be resolved (e.g. TYPE_CHECKING imports).
        if 'from_spec' not in cls.__dict__:
            try:
                get_function_type_hints(cls.__init__)
                return cls.__init__
            except (NameError, TypeError, AttributeError):
                pass
        return cls.from_spec

    return build_schema_types(
        registry,
        get_schema_target=_get_schema_target,
    )


def _replace_capability_spec_refs(schema: dict[str, Any], cap_items_schema: dict[str, Any]) -> None:
    """Walk the schema and replace any $ref to CapabilitySpec with the capability items Union."""
    cap_ref = '#/$defs/CapabilitySpec'

    if schema.get('$ref') == cap_ref:
        schema.clear()
        schema.update(cap_items_schema)
        return
    for value in schema.values():
        if isinstance(value, dict):
            _replace_capability_spec_refs(cast(dict[str, Any], value), cap_items_schema)
        elif isinstance(value, list):
            for item in value:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(item, dict):
                    _replace_capability_spec_refs(cast(dict[str, Any], item), cap_items_schema)

    # Clean up the CapabilitySpec $def entry
    defs: dict[str, Any] = schema.get('$defs', {})
    defs.pop('CapabilitySpec', None)
