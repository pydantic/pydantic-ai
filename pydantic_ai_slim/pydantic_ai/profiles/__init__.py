from __future__ import annotations as _annotations

from dataclasses import dataclass, fields, replace
from typing import Callable, Self, Union

from typing_extensions import TypeAliasType

from pydantic_ai.profiles._json_schema import JsonSchemaTransformer


@dataclass
class ModelProfile:
    """Describes how requests to a specific model or family of models need to be formatted to get the best results, independent of the model and provider classes used."""

    json_schema_transformer: type[JsonSchemaTransformer] | None = None

    @classmethod
    def from_profile(cls, profile: ModelProfile | None) -> Self:
        """Build a ModelProfile subclass instance from a ModelProfile instance."""
        if isinstance(profile, cls):
            return profile
        return cls().update(profile)

    def update(self, profile: ModelProfile | None) -> Self:
        """Update this ModelProfile (subclass) instance with the values from another ModelProfile instance."""
        if not profile:
            return self
        field_names = set(f.name for f in fields(self))
        shallow_copied_dict = {
            field.name: getattr(profile, field.name) for field in fields(profile) if field.name in field_names
        }
        return replace(self, **shallow_copied_dict)


ModelProfileSpec = TypeAliasType(
    'ModelProfileSpec', Union[ModelProfile, Callable[[str], Union[ModelProfile, None]], None]
)

DEFAULT_PROFILE = ModelProfile()
