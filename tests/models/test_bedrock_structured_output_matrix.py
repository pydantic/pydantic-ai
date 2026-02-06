"""Structured output matrix tests for Bedrock Converse API.

Probes Bedrock's actual behavior for various JSON schema features under three modes:
  - Native output (NativeOutput)
  - Strict tools (strict=True)
  - Non-strict tools (strict=False)

Each test is annotated:
  - permanent (K)  — enforces known/final transformer behavior
  - decision-probe (P)  — exists to resolve ambiguous docs vs observed behavior

After VCR recording, decision-probe tests whose cassettes show 400 should be updated
to assert `pytest.raises(ModelHTTPError)` with `status_code == 400`.
"""

from __future__ import annotations as _annotations

from datetime import date, datetime, time, timedelta
from enum import Enum
from ipaddress import IPv4Address, IPv6Address
from typing import Annotated, Literal
from uuid import UUID

import pytest
from pydantic import AnyUrl, BaseModel, Field

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.output import NativeOutput

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skip(reason='Matrix suite is for reference only — will be removed before merging'),
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning'),
]

MODEL_ID = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'


# ── Pydantic models for test schemas ──────────────────────────────────


class SimpleModel(BaseModel):
    name: str
    age: int


class NumericConstraints(BaseModel):
    score: Annotated[float, Field(ge=0.0, le=100.0)]
    count: Annotated[int, Field(ge=0, le=1000)]


class MultipleOfOnly(BaseModel):
    value: Annotated[int, Field(multiple_of=5)]


class StringConstraints(BaseModel):
    username: Annotated[str, Field(min_length=2, max_length=50)]
    email: Annotated[str, Field(pattern=r'^[\w.-]+@[\w.-]+\.\w+$')]


class ArrayMinItems0(BaseModel):
    items: Annotated[list[str], Field(min_length=0)]


class ArrayMinItems1(BaseModel):
    items: Annotated[list[str], Field(min_length=1)]


class ArrayMinItems2(BaseModel):
    items: Annotated[list[str], Field(min_length=2)]


class ArrayMaxItemsOnly(BaseModel):
    items: Annotated[list[str], Field(max_length=5)]


class InnerModel(BaseModel):
    street: str
    city: str


class NestedModel(BaseModel):
    name: str
    address: InnerModel


class OptionalFields(BaseModel):
    name: str
    nickname: str | None = None


class Color(str, Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


class ScalarEnumModel(BaseModel):
    color: Color
    name: str


class UnionModel(BaseModel):
    value: int | str


class ConstModel(BaseModel):
    action: Literal['submit']
    data: str


class WithDefaults(BaseModel):
    name: str
    country: str = 'Unknown'
    count: int = 0


class TreeNode(BaseModel):
    value: str
    children: list[TreeNode] = []


# ── Section 1: Native Output tests ────────────────────────────────────


@pytest.mark.vcr()
async def test_native_multipleOf(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with multipleOf constraint — expected: reject (400)."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(MultipleOfOnly))
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Give me a value that is a multiple of 5.')
    assert exc_info.value.status_code == 400


@pytest.mark.vcr()
async def test_native_array_minItems_0(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with minItems=0 — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(ArrayMinItems0))
    result = await agent.run('Give me a list of fruits.')
    assert isinstance(result.output, ArrayMinItems0)


@pytest.mark.vcr()
async def test_native_array_minItems_1(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with minItems=1 — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(ArrayMinItems1))
    result = await agent.run('Give me a list of fruits.')
    assert isinstance(result.output, ArrayMinItems1)


@pytest.mark.vcr()
async def test_native_array_maxItems_only(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with maxItems only — expected: reject per docs (400)."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(ArrayMaxItemsOnly))
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Give me up to 5 fruits.')
    assert exc_info.value.status_code == 400


@pytest.mark.vcr()
async def test_native_scalar_enum(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with scalar enum — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(ScalarEnumModel))
    result = await agent.run('Pick a color and give me a name.')
    assert isinstance(result.output, ScalarEnumModel)


@pytest.mark.vcr()
async def test_native_const_literal(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with const/Literal — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(ConstModel))
    result = await agent.run('Submit data "hello world".')
    assert isinstance(result.output, ConstModel)
    assert result.output.action == 'submit'


@pytest.mark.vcr()
async def test_native_union_non_null(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with non-null union (int | str) — expected: accept or reject."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(UnionModel))
    result = await agent.run('Give me a value, either a number or a string.')
    assert isinstance(result.output, UnionModel)


@pytest.mark.vcr()
async def test_native_recursive_schema(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with recursive schema ($ref to self) — expected: reject (500, Bedrock crashes)."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(TreeNode))
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Create a tree: root with children A and B, where A has child C.')
    assert exc_info.value.status_code == 500


@pytest.mark.vcr()
async def test_native_with_defaults(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with default values — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(WithDefaults))
    result = await agent.run('Give me a person named Alice.')
    assert isinstance(result.output, WithDefaults)
    assert result.output.name == 'Alice'


@pytest.mark.vcr()
async def test_native_format_email(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with string constraints (minLength, pattern) — expected: accept or reject."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(StringConstraints))
    result = await agent.run('Give me username "alice" and email "alice@example.com".')
    assert isinstance(result.output, StringConstraints)


# ── Section 2: Strict Tool tests (strict=True) ────────────────────────


@pytest.mark.vcr()
async def test_strict_tool_string_constraints(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Strict tool with string constraints (minLength, maxLength) — expected: accept or reject."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def greet_user(
        username: Annotated[str, Field(min_length=2, max_length=50)],
    ) -> str:
        return f'Hello, {username}!'

    result = await agent.run('Greet the user Alice.')
    assert 'Alice' in result.output


@pytest.mark.vcr()
async def test_strict_tool_multipleOf(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Strict tool with multipleOf — expected: reject (400)."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def set_quantity(
        quantity: Annotated[int, Field(multiple_of=5)],
    ) -> str:
        return f'Quantity set to {quantity}'

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Set quantity to 25.')
    assert exc_info.value.status_code == 400


@pytest.mark.vcr()
async def test_strict_tool_array_minItems_gt1(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Strict tool with array minItems > 1 — expected: reject (400)."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def process_tags(
        tags: Annotated[list[str], Field(min_length=2)],
    ) -> str:
        return f'Tags: {", ".join(tags)}'

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Process tags: python, ai.')
    assert exc_info.value.status_code == 400


@pytest.mark.vcr()
async def test_strict_tool_array_maxItems(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Strict tool with array maxItems — expected: reject per docs (400)."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def select_items(
        items: Annotated[list[str], Field(max_length=5)],
    ) -> str:
        return f'Selected: {", ".join(items)}'

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Select items: apple, banana, cherry.')
    assert exc_info.value.status_code == 400


@pytest.mark.vcr()
async def test_strict_tool_scalar_enum(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Strict tool with scalar enum parameter — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def set_color(color: Color) -> str:
        return f'Color set to {color.value}'

    result = await agent.run('Set color to red.')
    assert 'red' in result.output.lower()


@pytest.mark.vcr()
async def test_strict_tool_const_literal(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Strict tool with Literal parameter — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def confirm_action(action: Literal['submit'], data: str) -> str:
        return f'Action {action}: {data}'

    result = await agent.run('Submit data "hello".')
    assert 'submit' in result.output.lower() or 'hello' in result.output.lower()


@pytest.mark.vcr()
async def test_strict_tool_nested_defs_ref(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Strict tool with nested model parameter ($defs/$ref) — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def register_person(name: str, address_street: str, address_city: str) -> str:
        return f'Registered {name} at {address_street}, {address_city}'

    result = await agent.run('Register Alice at 123 Main St, Springfield.')
    assert 'Alice' in result.output


@pytest.mark.vcr()
async def test_strict_tool_format_email(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Strict tool with string pattern constraint — expected: accept or reject."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def validate_email(
        email: Annotated[str, Field(pattern=r'^[\w.-]+@[\w.-]+\.\w+$')],
    ) -> str:
        return f'Email {email} is valid'

    result = await agent.run('Validate email alice@example.com.')
    assert 'alice@example.com' in result.output


# ── Section 3: Non-Strict Tool tests (strict=False) ───────────────────


@pytest.mark.vcr()
async def test_non_strict_tool_string_constraints(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Non-strict tool with string constraints — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=False)
    def greet_user(
        username: Annotated[str, Field(min_length=2)],
    ) -> str:
        return f'Hello, {username}!'

    result = await agent.run('Greet the user Alice.')
    assert 'Alice' in result.output


@pytest.mark.vcr()
async def test_non_strict_tool_array_minItems_gt1(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Non-strict tool with array minItems > 1 — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=False)
    def process_tags(
        tags: Annotated[list[str], Field(min_length=2)],
    ) -> str:
        return f'Tags: {", ".join(tags)}'

    result = await agent.run('Process tags: python, ai.')
    assert 'python' in result.output.lower()


@pytest.mark.vcr()
async def test_non_strict_tool_nested_defs_ref(  # permanent (K)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Non-strict tool with nested model parameter — expected: accept."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=False)
    def register_person(name: str, address_street: str, address_city: str) -> str:
        return f'Registered {name} at {address_street}, {address_city}'

    result = await agent.run('Register Alice at 123 Main St, Springfield.')
    assert 'Alice' in result.output


@pytest.mark.vcr()
async def test_non_strict_tool_additional_properties_true(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Non-strict tool with additionalProperties=true via **kwargs — expected: reject per docs (400)."""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=False)
    def collect_fields(name: str, **extras: str) -> str:
        # Extras force additionalProperties to be a schema (not false)
        return f'Collected {name} with {len(extras)} extras'

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Collect name Alice with extra fields.')
    assert exc_info.value.status_code == 400


# ── Section 4: String format tests (strict=True) ────────────────────
# Probes whether Bedrock accepts/rejects the JSON Schema "format" keyword
# under strict validation. Each format is tested individually.


class FormatDateTime(BaseModel):
    value: datetime


class FormatDate(BaseModel):
    value: date


class FormatTime(BaseModel):
    value: time


class FormatDuration(BaseModel):
    value: timedelta


class FormatIPv4(BaseModel):
    value: IPv4Address


class FormatIPv6(BaseModel):
    value: IPv6Address


class FormatUUID(BaseModel):
    value: UUID


class FormatURI(BaseModel):
    value: AnyUrl


@pytest.mark.vcr()
async def test_strict_native_format_datetime(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with format=date-time — probe: accept or reject?"""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(FormatDateTime))
    result = await agent.run('Give me a datetime value for January 1st 2025 at noon UTC.')
    assert isinstance(result.output, FormatDateTime)


@pytest.mark.vcr()
async def test_strict_native_format_date(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with format=date — probe: accept or reject?"""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(FormatDate))
    result = await agent.run('Give me the date January 1st 2025.')
    assert isinstance(result.output, FormatDate)


@pytest.mark.vcr()
async def test_strict_native_format_time(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with format=time — probe: accept or reject?"""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(FormatTime))
    result = await agent.run('Give me the time 12:30 PM.')
    assert isinstance(result.output, FormatTime)


@pytest.mark.vcr()
async def test_strict_native_format_duration(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with format=duration — probe: accept or reject?"""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(FormatDuration))
    result = await agent.run('Give me a duration of 2 hours and 30 minutes.')
    assert isinstance(result.output, FormatDuration)


@pytest.mark.vcr()
async def test_strict_native_format_ipv4(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with format=ipv4 — probe: accept or reject?"""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(FormatIPv4))
    result = await agent.run('Give me the IP address 192.168.1.1.')
    assert isinstance(result.output, FormatIPv4)


@pytest.mark.vcr()
async def test_strict_native_format_ipv6(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with format=ipv6 — probe: accept or reject?"""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(FormatIPv6))
    result = await agent.run('Give me the IPv6 address ::1.')
    assert isinstance(result.output, FormatIPv6)


@pytest.mark.vcr()
async def test_strict_native_format_uuid(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with format=uuid — probe: accept or reject?"""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(FormatUUID))
    result = await agent.run('Give me a UUID value.')
    assert isinstance(result.output, FormatUUID)


@pytest.mark.vcr()
async def test_strict_native_format_uri(  # decision-probe (P)
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with format=uri (not in OpenAI's compatible list) — probe: accept or reject?"""
    model = BedrockConverseModel(MODEL_ID, provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(FormatURI))
    result = await agent.run('Give me the URL https://example.com.')
    assert isinstance(result.output, FormatURI)
