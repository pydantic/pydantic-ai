"""Prompt-cache prefix stability across the UI-adapter round-trip seam.

A UI-driven app (Vercel AI / AG-UI) holds conversation history in the protocol's own
message shape and re-submits it every turn. The server turns it back into Pydantic AI
messages with `load_messages` and sends those to the provider. If that round-trip
(`ModelMessages -> dump_messages -> load_messages`) changes the *provider request bytes*,
the cacheable prefix moves and the provider's prompt cache silently misses — no error,
just a bigger bill and higher latency (the PR #4338 class of bug).

Each test records a real conversation, round-trips the resulting history through an adapter,
then sends the same follow-up prompt twice — once with the original history, once with the
round-tripped one — and compares the two recorded request bodies byte-for-byte. Equal bodies
mean the round-trip preserved the cacheable prefix. Coverage targets the PR #5873 wire-fidelity
risks: tool-call arg serialization and thinking signatures.

On real provider data the round-trips are byte-faithful, so these are regression guards. The one
real byte-change is AG-UI < 0.1.13 dropping `ThinkingPart` (no reasoning carrier before 0.1.13);
that test asserts the prefix *changes*, documenting the protocol-version limitation.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

from .conftest import try_import

with try_import() as ag_ui_imports_successful:
    from pydantic_ai.ui.ag_ui import AGUIAdapter

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


def _post_bodies(vcr: Cassette) -> list[bytes | str]:
    """Raw bodies of the POST requests recorded in `vcr`, in order."""
    return [request.body for request in vcr.requests if request.method == 'POST']  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _vercel_roundtrip(history: list[ModelMessage]) -> list[ModelMessage]:
    return VercelAIAdapter.load_messages(VercelAIAdapter.dump_messages(history))


def _ag_ui_roundtrip_latest(history: list[ModelMessage]) -> list[ModelMessage]:
    return AGUIAdapter.load_messages(AGUIAdapter.dump_messages(history, ag_ui_version='0.1.13'))


def _ag_ui_roundtrip_0_1_10(history: list[ModelMessage]) -> list[ModelMessage]:
    return AGUIAdapter.load_messages(AGUIAdapter.dump_messages(history, ag_ui_version='0.1.10'))


@pytest.mark.skipif(not openai_imports_successful, reason='openai not installed')
@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_vercel_roundtrip, id='vercel'),
        pytest.param(
            _ag_ui_roundtrip_latest,
            id='ag_ui',
            marks=pytest.mark.skipif(not ag_ui_imports_successful, reason='ag-ui-protocol not installed'),
        ),
    ],
)
async def test_openai_tool_call_roundtrip_wire_stable(
    allow_model_requests: None,
    openai_api_key: str,
    vcr: Cassette,
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
):
    """A real OpenAI tool call re-sends byte-identically after a UI round-trip.

    OpenAI carries `arguments` as a JSON string; a round-trip that reconstructed it differently
    would move the cacheable prefix under OpenAI's exact-prefix caching.
    """
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    generator = Agent(model)

    @generator.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    history = (await generator.run('What is the weather in Paris? Use the tool.')).all_messages()

    probe = Agent(model)
    await probe.run('Reply with exactly: OK', message_history=history)
    await probe.run('Reply with exactly: OK', message_history=roundtrip(history))

    original_body, roundtripped_body = _post_bodies(vcr)[-2:]
    assert roundtripped_body == original_body


@pytest.mark.skipif(not anthropic_imports_successful, reason='anthropic not installed')
@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_vercel_roundtrip, id='vercel'),
        pytest.param(
            _ag_ui_roundtrip_latest,
            id='ag_ui-0_1_13',
            marks=pytest.mark.skipif(not ag_ui_imports_successful, reason='ag-ui-protocol not installed'),
        ),
    ],
)
async def test_anthropic_thinking_roundtrip_wire_stable(
    allow_model_requests: None,
    anthropic_api_key: str,
    vcr: Cassette,
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
):
    """A real Anthropic thinking block (with its signature) re-sends byte-identically after a round-trip."""
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    generator = Agent(model, model_settings=settings)
    history = (await generator.run('Briefly: what is 17 * 23? Think first.')).all_messages()

    probe = Agent(model, model_settings=settings)
    await probe.run('Reply with exactly: OK', message_history=history)
    await probe.run('Reply with exactly: OK', message_history=roundtrip(history))

    original_body, roundtripped_body = _post_bodies(vcr)[-2:]
    assert roundtripped_body == original_body


@pytest.mark.skipif(not anthropic_imports_successful, reason='anthropic not installed')
@pytest.mark.skipif(not ag_ui_imports_successful, reason='ag-ui-protocol not installed')
async def test_anthropic_thinking_agui_0_1_10_drops_prefix(
    allow_model_requests: None,
    anthropic_api_key: str,
    vcr: Cassette,
):
    """AG-UI < 0.1.13 has no reasoning carrier, so dump drops the `ThinkingPart` and the re-sent
    prefix changes. Documents the protocol-version limitation (verified, not a defect to fix)."""
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    generator = Agent(model, model_settings=settings)
    history = (await generator.run('Briefly: what is 17 * 23? Think first.')).all_messages()

    probe = Agent(model, model_settings=settings)
    await probe.run('Reply with exactly: OK', message_history=history)
    await probe.run('Reply with exactly: OK', message_history=_ag_ui_roundtrip_0_1_10(history))

    original_body, roundtripped_body = _post_bodies(vcr)[-2:]
    assert roundtripped_body != original_body
