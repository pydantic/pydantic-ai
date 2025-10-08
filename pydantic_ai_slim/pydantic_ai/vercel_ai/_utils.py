from abc import ABC
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

__all__ = 'ProviderMetadata', 'CamelBaseModel'

# technically this is recursive union of JSON types
# for to simplify validation, we call it Any
JSONValue = Any

# Provider metadata types
ProviderMetadata = dict[str, dict[str, JSONValue]]


class CamelBaseModel(BaseModel, ABC):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra='forbid')
