"""Agent specification for constructing agents from YAML/JSON/dict specs."""

from __future__ import annotations

import json as _json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast

from pydantic import BaseModel, Field, model_serializer
from pydantic_core import to_json
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler

from pydantic_ai._agent_graph import EndStrategy
from pydantic_ai._spec import NamedSpec, build_registry, build_schema_types
from pydantic_ai._template import TemplateStr

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import AbstractCapability

CapabilitySpec = NamedSpec
"""The specification of a capability to be constructed.

Supports the same short forms as `EvaluatorSpec`:
* ``'MyCapability'`` — no arguments
* ``{'MyCapability': single_arg}`` — a single positional argument
* ``{'MyCapability': {k1: v1, k2: v2}}`` — keyword arguments
"""

DEFAULT_SCHEMA_PATH_TEMPLATE = './{stem}_schema.json'
"""Default template for schema file paths, where {stem} is replaced with the spec filename stem."""

_YAML_SCHEMA_LINE_PREFIX = '# yaml-language-server: $schema='


class AgentSpec(BaseModel):
    """Specification for constructing an Agent from a dict/YAML/JSON."""

    json_schema_path: str | None = Field(default=None, alias='$schema')
    model: str
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

        if fmt == 'json':
            data = _json.loads(content)
        else:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    'PyYAML is required to load YAML agent specs. Install it with: pip install "pydantic-ai-slim[cli]"'
                ) from None
            data = yaml.safe_load(content)

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
            except ImportError:
                raise ImportError(
                    'PyYAML is required to save YAML agent specs. Install it with: pip install "pydantic-ai-slim[cli]"'
                ) from None
            dumped_data = self.model_dump(mode='json', by_alias=True, context=context, exclude_defaults=True)
            content = yaml.dump(dumped_data, sort_keys=False)
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
        capability_schema_types = _build_capability_schema_types(_get_capability_registry(custom_capability_types))

        # Build a schema-only model with the resolved capability union
        class AgentSpec(BaseModel, extra='forbid'):
            model: str
            if capability_schema_types:  # pragma: no branch
                capabilities: list[Union[tuple(capability_schema_types)]] = []  # pyright: ignore  # noqa: UP007

        json_schema = AgentSpec.model_json_schema()
        json_schema['properties']['$schema'] = {'type': 'string'}
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


def _get_capability_registry(
    custom_types: Sequence[type[AbstractCapability[Any]]] = (),
) -> Mapping[str, type[AbstractCapability[Any]]]:
    """Create a registry of capability types from default and custom types."""
    from pydantic_ai.capabilities import DEFAULT_CAPABILITY_TYPES
    from pydantic_ai.capabilities.abstract import AbstractCapability

    def _validate_capability(cls: type[AbstractCapability[Any]]) -> None:
        if not issubclass(cls, AbstractCapability):
            raise ValueError(
                f'All custom capability classes must be subclasses of AbstractCapability, but {cls} is not'
            )

    return build_registry(
        custom_types=custom_types,
        defaults=DEFAULT_CAPABILITY_TYPES,
        get_name=lambda cls: cls.get_serialization_name(),
        label='capability',
        validate=_validate_capability,
    )


def _build_capability_schema_types(registry: Mapping[str, type[Any]]) -> list[Any]:
    """Build a list of schema types for capabilities from a registry."""
    return build_schema_types(
        registry,
        get_schema_target=lambda cls: cls.from_spec,
    )
