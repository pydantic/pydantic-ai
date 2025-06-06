from __future__ import annotations as _annotations

from dataclasses import dataclass, field, fields, replace
from typing import Callable, Literal, Union

from typing_extensions import Self

from ._json_schema import JsonSchemaTransformer


@dataclass
class ModelProfile:
    """Describes how requests to a specific model or family of models need to be constructed to get the best results, independent of the model and provider classes used."""

    json_schema_transformer: type[JsonSchemaTransformer] | None = None
    output_modes: set[Literal['tool', 'json_schema']] = field(default_factory=lambda: {'tool'})
    # TODO: Add docstrings
    default_output_mode: Literal['tool', 'json_schema', 'manual_json'] = 'tool'

    @classmethod
    def from_profile(cls, profile: ModelProfile | None) -> Self:
        """Build a ModelProfile subclass instance from a ModelProfile instance."""
        if isinstance(profile, cls):
            return profile
        return cls().update(profile)

    def update(self, profile: ModelProfile | None) -> Self:
        """Update this ModelProfile (subclass) instance with the non-default values from another ModelProfile instance."""
        if not profile:
            return self
        field_names = set(f.name for f in fields(self))
        non_default_attrs = {
            f.name: getattr(profile, f.name)
            for f in fields(profile)
            if f.name in field_names and getattr(profile, f.name) != f.default
        }
        return replace(self, **non_default_attrs)


ModelProfileSpec = Union[ModelProfile, Callable[[str], Union[ModelProfile, None]]]

DEFAULT_PROFILE = ModelProfile()
