"""VCR ground-truth tests for OpenAI model capabilities.

Probes each model via the Responses API to determine actual behavior,
then cross-checks against openai_model_profile() flags. This catches
regressions when OpenAI changes model behavior (e.g. which models
accept reasoning_effort='none').
"""

from __future__ import annotations as _annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from inline_snapshot import snapshot

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI, BadRequestError

    from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


@dataclass
class ProbeResult:
    status: str
    reasoning_tokens: int | None = None
    error_code: str | None = None
    error_message: str | None = None


def _extract_error(e: BadRequestError) -> tuple[str | None, str | None]:
    """Extract error code and message from the API error body.

    Parses the structured body rather than e.message, which includes
    SDK formatting that differs between live calls and VCR replay.
    """
    body: dict[str, object] = e.body if isinstance(e.body, dict) else {}  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    error_obj = body.get('error') if body else None
    if not isinstance(error_obj, dict):
        return e.code, None
    inner: dict[str, object] = error_obj  # pyright: ignore[reportUnknownVariableType]
    code = inner.get('code')
    message = inner.get('message')
    return (
        str(code) if isinstance(code, str) else None,
        str(message) if isinstance(message, str) else None,
    )


async def probe_reasoning_none(client: AsyncOpenAI, model: str) -> ProbeResult:
    """Probe whether a model accepts reasoning={"effort": "none"}."""
    try:
        response = await client.responses.create(
            model=model,
            input='Say "hi"',
            reasoning={'effort': 'none'},
            max_output_tokens=16,
            store=False,
        )
        reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens if response.usage else None
        return ProbeResult(status='success', reasoning_tokens=reasoning_tokens)
    except BadRequestError as e:
        code, message = _extract_error(e)
        return ProbeResult(status='error', error_code=code, error_message=message)


async def probe_temperature(client: AsyncOpenAI, model: str) -> ProbeResult:
    """Probe whether a model accepts temperature=0.5."""
    try:
        response = await client.responses.create(
            model=model,
            input='Say "hi"',
            temperature=0.5,
            max_output_tokens=16,
            store=False,
        )
        reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens if response.usage else None
        return ProbeResult(status='success', reasoning_tokens=reasoning_tokens)
    except BadRequestError as e:
        code, message = _extract_error(e)
        return ProbeResult(status='error', error_code=code, error_message=message)


@dataclass
class CapabilityCase:
    model: str
    reasoning_none_result: ProbeResult | None = None
    temperature_result: ProbeResult | None = None
    expected_supports_reasoning: bool = False
    expected_supports_reasoning_effort_none: bool = False


CASES = [
    # --- GPT-4.1 family (non-reasoning) ---
    CapabilityCase(
        model='gpt-4.1',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_parameter')),
        temperature_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
    ),
    CapabilityCase(
        model='gpt-4.1-mini',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_parameter')),
        temperature_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
    ),
    CapabilityCase(
        model='gpt-4.1-nano',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_parameter')),
        temperature_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
    ),
    # --- GPT-5 base (reasoning, no effort_none) ---
    CapabilityCase(
        model='gpt-5',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='gpt-5-chat-latest',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_parameter')),
        temperature_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
    ),
    CapabilityCase(
        model='gpt-5-codex',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='gpt-5-mini',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='gpt-5-nano',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='gpt-5-pro',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    # --- GPT-5.1 family ---
    CapabilityCase(
        model='gpt-5.1',
        reasoning_none_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
        temperature_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
        expected_supports_reasoning=True,
        expected_supports_reasoning_effort_none=True,
    ),
    CapabilityCase(
        model='gpt-5.1-chat-latest',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='gpt-5.1-codex',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='gpt-5.1-codex-max',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    # --- GPT-5.2 family ---
    CapabilityCase(
        model='gpt-5.2',
        reasoning_none_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
        temperature_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
        expected_supports_reasoning=True,
        expected_supports_reasoning_effort_none=True,
    ),
    CapabilityCase(
        model='gpt-5.2-chat-latest',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='gpt-5.2-pro',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    # --- GPT-5.3 family ---
    CapabilityCase(
        model='gpt-5.3-chat-latest',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    # --- GPT-5.4 ---
    CapabilityCase(
        model='gpt-5.4',
        reasoning_none_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
        temperature_result=snapshot(ProbeResult(status='success', reasoning_tokens=0)),
        expected_supports_reasoning=True,
        expected_supports_reasoning_effort_none=True,
    ),
    # --- o1 family ---
    CapabilityCase(
        model='o1',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='o1-pro',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    # --- o3 family ---
    CapabilityCase(
        model='o3',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='o3-mini',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='o3-pro',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='o3-deep-research',
        reasoning_none_result=snapshot(ProbeResult(status='error')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    # --- o4 family ---
    CapabilityCase(
        model='o4-mini',
        reasoning_none_result=snapshot(ProbeResult(status='error', error_code='unsupported_value')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
    CapabilityCase(
        model='o4-mini-deep-research',
        reasoning_none_result=snapshot(ProbeResult(status='error')),
        temperature_result=snapshot(ProbeResult(status='error')),
        expected_supports_reasoning=True,
    ),
]


@pytest.fixture
async def openai_client() -> AsyncIterator[AsyncOpenAI]:
    client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY', 'test-key'))
    yield client
    await client.close()


@pytest.mark.vcr()
@pytest.mark.parametrize('case', CASES, ids=lambda c: c.model)
async def test_model_capabilities(case: CapabilityCase, openai_client: AsyncOpenAI):
    reasoning_none_result = await probe_reasoning_none(openai_client, case.model)
    temperature_result = await probe_temperature(openai_client, case.model)

    assert reasoning_none_result == case.reasoning_none_result
    assert temperature_result == case.temperature_result

    profile = openai_model_profile(case.model)
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.openai_supports_reasoning is case.expected_supports_reasoning, (
        f'{case.model}: expected supports_reasoning={case.expected_supports_reasoning}, '
        f'got {profile.openai_supports_reasoning}'
    )
    assert profile.openai_supports_reasoning_effort_none is case.expected_supports_reasoning_effort_none, (
        f'{case.model}: expected supports_reasoning_effort_none={case.expected_supports_reasoning_effort_none}, '
        f'got {profile.openai_supports_reasoning_effort_none}'
    )
