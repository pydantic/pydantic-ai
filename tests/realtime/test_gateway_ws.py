"""Cassette-backed tests for realtime models routed through the Pydantic AI Gateway.

These prove the gateway path end-to-end the way a user reaches it (`gateway/openai:...`): the model
connects to the gateway's realtime WebSocket, which relays the provider protocol verbatim, so the same
streamed part events and history come back as a direct connection. Recorded once against the live
gateway with `--record-mode=rewrite`, then replayed offline.

The gateway's OpenAI realtime relay is live, so `gateway/openai` is exercised end-to-end here. A full
`gateway/google` session can't be recorded yet: the gateway relay only matches the OpenAI-shaped
`/proxy/<route>/realtime` upgrade path, while the `google-genai` SDK dials the native Vertex Bidi path
(`/proxy/<route>/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent`), so the upgrade
never reaches the relay (and the Live model ids are unpriced in genai-prices, a second gate). That test
skips until the gateway accepts the Vertex Bidi upgrade path. The gateway handshake's URL derivation and
bearer-auth injection are pinned separately in `test_google.py` (`test_gateway_handshake_carries_bearer_auth`).
"""

from __future__ import annotations as _annotations

from typing import Any

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, SpeechPart

from ..conftest import try_import
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types

with try_import() as imports_successful:
    from pydantic_ai.providers import Provider
    from pydantic_ai.realtime import TurnCompleteEvent
    from pydantic_ai.realtime.google import GoogleRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed'),
]


async def test_gateway_openai_text_in_audio_out(
    gateway_openai_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """A `gateway/openai` session streams audio+transcript back through the gateway relay."""
    provider, _cassette = gateway_openai_ws_cassette
    # The provider carries the gateway base URL (host is region-encoded in the key), so the realtime
    # handshake dials the gateway rather than OpenAI directly — a genuine gateway round-trip.
    assert provider.base_url.endswith('/proxy/openai/')

    model = OpenAIRealtimeModel('gpt-realtime', provider=provider)
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        await session.send('Say a short greeting.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

    # The turn streams audio+transcript parts and ends cleanly, exactly as a direct OpenAI session does.
    assert 'PartStartEvent' in collapse_event_types(events)
    assert any(isinstance(event, TurnCompleteEvent) for event in events)
    response = session.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert any(isinstance(part, SpeechPart) for part in response.parts)


# Runs only while recording (route gated).
async def test_gateway_gemini_text_in_audio_out(  # pragma: no cover
    gateway_gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """A `gateway/google` session streams audio+transcript back through the gateway's Vertex relay.

    Skipped until the gateway's Vertex Live route is recordable (see the fixture docstring); wired so it
    records and runs the moment that route is available, without any further plumbing. The whole body is
    coverage-excluded because it never runs offline — the routing/URL/auth it would exercise are pinned
    offline by `test_inference.test_infer_realtime_model_gateway_google` and `test_google`'s handshake test.
    """
    provider, _cassette = gateway_gemini_ws_cassette
    assert provider.base_url.endswith('/proxy/google-vertex')

    model = GoogleRealtimeModel('gemini-live-2.5-flash', provider=provider)
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        await session.send('Say a short greeting.')
        with anyio.fail_after(30):
            async for event in session:
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

    assert any(isinstance(event, TurnCompleteEvent) for event in events)
