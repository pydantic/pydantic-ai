"""Agent specification for constructing agents from YAML/JSON/dict specs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from pydantic import BaseModel
from pydantic_core import to_json

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


class AgentSpec(BaseModel):
    """Specification for constructing an Agent from a dict/YAML/JSON."""

    model: str
    name: str | None = None
    description: TemplateStr[Any] | str | None = None
    instructions: TemplateStr[Any] | str | list[TemplateStr[Any] | str] | None = None
    deps_schema: dict[str, Any] | None = None
    model_settings: dict[str, Any] | None = None
    retries: int = 1
    output_retries: int | None = None
    end_strategy: EndStrategy = 'early'
    tool_timeout: float | None = None
    instrument: bool | None = None
    metadata: dict[str, Any] | None = None
    capabilities: list[CapabilitySpec] = []

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
