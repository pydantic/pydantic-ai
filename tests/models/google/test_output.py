"""Tests for Google Gemini's `VALIDATED` function-calling mode (the `strict` tool flag).

On supported models (Gemini 2.5+), `VALIDATED` is the default — it enforces the declared schema with no
schema rewrites, so it's a safe silent improvement — and a caller opts a tool out with `strict=False`.

Test organization:
1. Mode resolution (unit, against a `MagicMock` client)
2. `strict` resolution via `GoogleJsonSchemaTransformer`
3. End-to-end wire contract (live recording)
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from pydantic_ai import Agent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition, ToolKind

from ..._inline_snapshot import snapshot
from ...cassette_utils import get_first_post_body
from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

if TYPE_CHECKING:
    GoogleModelFactory = Callable[..., GoogleModel]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def make_tool(name: str, *, strict: bool | None = None, kind: ToolKind = 'function') -> ToolDefinition:
    return ToolDefinition(
        name=name,
        parameters_json_schema={'type': 'object', 'properties': {}},
        strict=strict,
        kind=kind,
    )


# =============================================================================
# Mode resolution
# =============================================================================


STRICT_TOOL_CONFIG_CASES = [
    dict(
        # A supported model defaults to `VALIDATED` even with no `strict` flag set: the silent improvement.
        id='default-supported-model-uses-validated',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a'), make_tool('b')],
        settings={},
        expected_mode='VALIDATED',
    ),
    dict(
        id='unsupported-model-stays-auto',
        model='gemini-2.0-flash',
        function_tools=[make_tool('a')],
        settings={},
        expected_mode='AUTO',
    ),
    dict(
        id='explicit-strict-uses-validated',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a', strict=True), make_tool('b', strict=True)],
        settings={},
        expected_mode='VALIDATED',
    ),
    dict(
        # A single tool opting out with `strict=False` drops the whole request back to `AUTO`.
        id='opt-out-tool-stays-auto',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a'), make_tool('b', strict=False)],
        settings={},
        expected_mode='AUTO',
    ),
    dict(
        id='required-tool-choice-stays-any',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a')],
        settings={'tool_choice': 'required'},
        expected_mode='ANY',
    ),
    dict(
        id='none-tool-choice-stays-none',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a')],
        settings={'tool_choice': 'none'},
        expected_mode='NONE',
    ),
    dict(
        # `tool_defs` spans function *and* output tools; a default output tool doesn't block `VALIDATED`, so a
        # plain `output_type` still gets the benefit (no need to set `strict=True` on every tool).
        id='default-output-tool-uses-validated',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a')],
        output_tools=[make_tool('final_result', kind='output')],
        settings={},
        expected_mode='VALIDATED',
    ),
    dict(
        # An output tool opting out with `strict=False` drops the request to `AUTO`, same as a function tool.
        id='opt-out-output-tool-stays-auto',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a')],
        output_tools=[make_tool('final_result', strict=False, kind='output')],
        settings={},
        expected_mode='AUTO',
    ),
]


@pytest.mark.parametrize('case', STRICT_TOOL_CONFIG_CASES, ids=lambda c: c['id'])
def test_google_strict_tools_upgrade_auto_to_validated(case: dict[str, Any]):
    """On a supported model, `AUTO` is upgraded to Gemini's `VALIDATED` mode unless a tool (function *or*
    output) opts out with `strict=False`; `required`/`none` tool choices are never upgraded.

    Asserted on the request shape directly rather than via VCR: a cassette replay can't catch the mode we send,
    since it replays a recorded response without re-validating the request against the API.
    """
    m = GoogleModel(case['model'], provider=GoogleProvider(client=MagicMock()))
    params = ModelRequestParameters(
        function_tools=case['function_tools'],
        output_tools=case.get('output_tools', []),
        allow_text_output=True,
    )

    _, tool_config, _ = m._get_tool_config(params, case['settings'])  # pyright: ignore[reportPrivateUsage]

    assert tool_config is not None
    assert tool_config['function_calling_config']['mode'].name == case['expected_mode']  # pyright: ignore[reportTypedDictNotRequiredAccess,reportOptionalMemberAccess,reportOptionalSubscript,reportUnknownMemberType]


# =============================================================================
# `strict` resolution via `GoogleJsonSchemaTransformer`
# =============================================================================


def test_google_strict_resolution_via_transformer():
    """`GoogleJsonSchemaTransformer` treats every schema as `VALIDATED`-compatible (the mode needs no schema
    rewrites): `strict=None` resolves to `True` (VALIDATED-eligible), and an explicit `strict=False` is
    preserved as the per-tool opt-out."""
    m = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(client=MagicMock()))

    # `strict` left as `None` resolves to `True`: default-on, VALIDATED-eligible.
    params = m.customize_request_parameters(
        ModelRequestParameters(function_tools=[make_tool('a')], allow_text_output=True)
    )
    assert params.function_tools[0].strict is True

    # An explicit `strict=False` is preserved so the caller can opt the tool out of `VALIDATED`.
    params = m.customize_request_parameters(
        ModelRequestParameters(function_tools=[make_tool('a', strict=False)], allow_text_output=True)
    )
    assert params.function_tools[0].strict is False


# =============================================================================
# End-to-end wire contract
# =============================================================================


async def test_google_default_tools_use_validated_mode(
    allow_model_requests: None,
    google_model: GoogleModelFactory,
    vcr: Any,
):
    """On a supported model, function tools default to `VALIDATED` mode with no `strict` flag set, and Gemini
    accepts that enum end-to-end.

    The mode-resolution cases above assert the request shape against a `MagicMock` client; only a live
    recording proves `VALIDATED` is a wire value the API accepts (rather than 400-ing on it), so this test
    inspects the recorded request to confirm the mode we sent was `VALIDATED`.
    """
    agent = Agent(google_model('gemini-2.5-flash'))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'The weather in {city} is sunny and 24C.'

    @agent.tool_plain
    def get_time(city: str) -> str:
        return f'The time in {city} is 3pm.'

    result = await agent.run('What is the weather and the time in Paris? Use the tools.')
    assert result.output == snapshot('The weather in Paris is sunny and 24C. The time in Paris is 3pm.')

    first_request = get_first_post_body(vcr)
    assert first_request['toolConfig']['functionCallingConfig']['mode'] == 'VALIDATED'
    assert len(first_request['tools'][0]['functionDeclarations']) == 2
