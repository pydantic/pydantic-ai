"""Agent specification for constructing agents from YAML/JSON/dict specs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast

from pydantic import BaseModel, Field, model_serializer
from pydantic_core import from_json, to_json
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler

from pydantic_ai._agent_graph import EndStrategy
from pydantic_ai._spec import CapabilitySpec, build_registry, build_schema_types
from pydantic_ai._template import TemplateStr
from pydantic_ai._utils import get_function_type_hints
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import AbstractCapability

__all__ = ['CapabilitySpec']  # re-exported from _spec

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
    retries: int = 1
    output_retries: int | None = None
    end_strategy: EndStrategy = 'early'
    tool_timeout: float | None = None
    instrument: bool | None = None
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
        content = path.read_text(encoding='utf-8')
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
            except ImportError:  # pragma: no cover — requires PyYAML to not be installed
                raise ImportError(
                    'PyYAML is required to load YAML agent specs. Install it with: pip install "pydantic-ai-slim[spec]"'
                ) from None
            data = yaml.safe_load(text)
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
    ) -> None:
        """Save the agent spec to a YAML or JSON file.

        Args:
            path: Path to save the spec to.
            fmt: Format to use. If None, inferred from file extension.
            schema_path: Path to save the JSON schema to. If None, no schema will be saved.
                Can be a string template with {stem} which will be replaced with the spec filename stem.
            custom_capability_types: Custom capability classes to include in the schema.
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
            else:  # pragma: no cover
                schema_ref = str(schema_path)
            self._save_schema(schema_path, custom_capability_types)

        context: dict[str, Any] = {'use_short_form': True}
        if fmt == 'yaml':
            try:
                import yaml
            except ImportError:  # pragma: no cover — requires PyYAML to not be installed
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
    ) -> dict[str, Any]:
        """Generate a JSON schema for this agent spec type, including capability details.

        This is useful for generating a schema that can be used to validate YAML-format agent spec files.

        Args:
            custom_capability_types: Custom capability classes to include in the schema.

        Returns:
            A dictionary representing the JSON schema.
        """
        capability_schema_types = _build_capability_schema_types(get_capability_registry(custom_capability_types))

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
            retries: int = 1
            output_retries: int | None = None
            end_strategy: EndStrategy = 'early'
            tool_timeout: float | None = None
            instrument: bool | None = None
            metadata: dict[str, Any] | None = None
            if capability_schema_types:  # pragma: no branch
                capabilities: list[Union[tuple(capability_schema_types)]] = []  # pyright: ignore  # noqa: UP007

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
    ) -> None:
        """Save the JSON schema for this agent spec type to a file.

        Args:
            path: Path to save the schema to.
            custom_capability_types: Custom capability classes to include in the schema.
        """
        path = Path(path)
        json_schema = cls.model_json_schema_with_capabilities(custom_capability_types)
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


def get_capability_registry(
    custom_types: Sequence[type[AbstractCapability[Any]]] = (),
) -> Mapping[str, type[AbstractCapability[Any]]]:
    """Create a registry of capability types from default and custom types."""
    from pydantic_ai.capabilities import CAPABILITY_TYPES
    from pydantic_ai.capabilities.abstract import AbstractCapability

    def _validate_capability(cls: type[AbstractCapability[Any]]) -> None:
        if not issubclass(cls, AbstractCapability):
            raise ValueError(
                f'All custom capability classes must be subclasses of AbstractCapability, but {cls} is not'
            )
        if '__dataclass_fields__' not in cls.__dict__:
            raise ValueError(f'All custom capability classes must be decorated with `@dataclass`, but {cls} is not')

    return build_registry(
        custom_types=custom_types,
        defaults=tuple(CAPABILITY_TYPES.values()),
        get_name=lambda cls: cls.get_serialization_name(),
        label='capability',
        validate=_validate_capability,
    )


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
            custom_types_param='custom_capability_types',
            instantiate=ctx.instantiate,
        )
    else:
        return load_from_registry(
            get_capability_registry(),
            cap_spec,
            label='capability',
            custom_types_param='custom_capability_types',
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
