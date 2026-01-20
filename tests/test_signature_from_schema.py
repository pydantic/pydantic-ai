"""Tests for JSON schema to Python signature conversion.

Uses roundtrip testing: define functions with various signatures, generate JSON schemas,
convert back to Python signatures, and verify the output.

Tests mainly pydantic_ai_slim/pydantic_ai/_signature_from_schema.py
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict


class UserInput(BaseModel):
    """User input model for roundtrip testing."""

    name: str
    email: str


class Config(TypedDict):
    """Config TypedDict for roundtrip testing."""

    timeout: int
    retries: NotRequired[int]


@dataclass
class Point:
    """Point dataclass for roundtrip testing."""

    x: float
    y: float


# Models for recursive $defs and union tests


class Address(BaseModel):
    street: str
    city: str


class User(BaseModel):
    name: str
    address: Address


class TreeNode(BaseModel):
    value: str
    children: list[TreeNode] = []


class Circle(BaseModel):
    radius: float


class Rectangle(BaseModel):
    width: float
    height: float


class Country(BaseModel):
    name: str


class CompanyAddress(BaseModel):
    street: str
    country: Country


class Company(BaseModel):
    name: str
    headquarters: CompanyAddress


class ConfigModel(BaseModel):
    key: str


# TODO move this to tests/code_mode/test_signature_from_schema.py
# TODO design this as a large parametrized case in response to any and all gaps that coverage raises
