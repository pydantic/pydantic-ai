"""Shared models for the Bedrock feature-central test package."""

from __future__ import annotations as _annotations

from pydantic import BaseModel


class CityInfo(BaseModel):
    """Information about a city."""

    model_config = {'extra': 'forbid'}
    city: str
    country: str
    population: int


class Address(BaseModel):
    """A street address (no extra='forbid' — additionalProperties: false not in native schema)."""

    street: str
    city: str


class PersonQuery(BaseModel):
    """A person query with a nested Address object (no extra='forbid')."""

    name: str
    address: Address
