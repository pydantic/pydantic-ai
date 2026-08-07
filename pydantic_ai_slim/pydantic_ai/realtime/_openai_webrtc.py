"""Shared WebRTC signaling and sideband helpers for the OpenAI-protocol realtime models.

OpenAI and Azure OpenAI expose the same WebRTC surface (Gemini is WebSocket-only, and xAI has no
WebRTC/`call_id` sideband — it is WebSocket-only too), so the HTTP signaling lives here and is reused
by both [`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] and
[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel]:

1. **Mint a client secret** (`POST .../realtime/client_secrets`) so a browser can connect directly
   without a long-lived key.
2. **Relay a WebRTC offer** (`POST .../realtime/calls`) — the secure path where the server negotiates
   the call on the browser's behalf and reads the `call_id` from the response `Location` header.
3. The control-plane (sideband) WebSocket that attaches to the negotiated call by `call_id` is opened
   by the model's `connect_webrtc`, reusing the normal OpenAI codec and session state machine.
"""

from __future__ import annotations as _annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from .._utils import is_str_dict
from ..exceptions import ModelHTTPError, UnexpectedModelBehavior
from ._base import RealtimeClientSecret, WebRTCAnswer, WebRTCSession

if TYPE_CHECKING:
    import httpx


def parse_call_id(location: str | None) -> str | None:
    """Extract the realtime `call_id` from a `/realtime/calls` response `Location` header.

    OpenAI and Azure OpenAI return the created call at `Location: /v1/realtime/calls/rtc_...`, so the
    id is the last path segment. A `?call_id=...` query form is tolerated too, for robustness against
    minor gateway/proxy rewrites.
    """
    if not location:
        return None
    parsed = urlparse(location)
    if parsed.query and (query_values := parse_qs(parsed.query).get('call_id')):
        return query_values[0]
    path_parts = [part for part in parsed.path.split('/') if part]
    if len(path_parts) >= 2 and path_parts[-2] == 'calls':
        return path_parts[-1]
    return None


def _raise_for_status(response: httpx.Response, provider_name: str) -> None:
    """Raise a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] on a non-2xx WebRTC signaling response.

    Signaling failures (401/403 auth, 429 rate limit, 5xx outages) are ordinary provider HTTP errors, not
    unexpected model output, so they go through the standard HTTP exception hierarchy — callers can catch
    `ModelHTTPError`, read `status_code`, and apply their own retry policy — with the response body preserved.
    """
    if response.is_error:
        raise ModelHTTPError(
            status_code=response.status_code, model_name=provider_name, body=response.text.strip() or None
        )


async def mint_client_secret(
    *,
    http_client: httpx.AsyncClient,
    client_secrets_url: str,
    headers: dict[str, str],
    provider_name: str,
    session_config: dict[str, Any],
    expires_after_seconds: int | None,
) -> RealtimeClientSecret:
    """`POST .../realtime/client_secrets` to mint an ephemeral browser token bound to `session_config`."""
    payload: dict[str, Any] = {'session': session_config}
    if expires_after_seconds is not None:
        payload['expires_after'] = {'anchor': 'created_at', 'seconds': expires_after_seconds}
    response = await http_client.post(
        client_secrets_url,
        headers={**headers, 'Content-Type': 'application/json'},
        content=json.dumps(payload),
    )
    _raise_for_status(response, provider_name)
    data = response.json()
    if not is_str_dict(data) or not isinstance(value := data.get('value'), str):
        raise UnexpectedModelBehavior('Realtime client-secret response did not include a `value`.')
    expires_at = data.get('expires_at')
    if not isinstance(expires_at, int) or isinstance(expires_at, bool):
        raise UnexpectedModelBehavior('Realtime client-secret response did not include a numeric `expires_at`.')
    provider_details = {key: item for key, item in data.items() if key != 'value'}
    return RealtimeClientSecret(
        value=value,
        expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc),
        provider_details=provider_details or None,
    )


def _webrtc_answer_from_response(response: httpx.Response, provider_name: str) -> WebRTCAnswer:
    """Build a [`WebRTCAnswer`][pydantic_ai.realtime.WebRTCAnswer] from a `/realtime/calls` response.

    The created call's id comes back in the `Location` header (e.g. `/v1/realtime/calls/rtc_...`), not
    the SDP body, so it is parsed out and carried on the returned [`WebRTCSession`][pydantic_ai.realtime.WebRTCSession].
    """
    _raise_for_status(response, provider_name)
    location = response.headers.get('location')
    call_id = parse_call_id(location)
    if call_id is None:
        raise UnexpectedModelBehavior(
            'Realtime WebRTC negotiation did not return a parseable `call_id` in the `Location` header.'
        )
    return WebRTCAnswer(
        sdp=response.text,
        session=WebRTCSession(
            provider_name=provider_name,
            session_id=call_id,
            provider_details={'location': location} if location else None,
        ),
    )


async def answer_webrtc_offer(
    *,
    http_client: httpx.AsyncClient,
    calls_url: str,
    headers: dict[str, str],
    provider_name: str,
    sdp_offer: str,
    session_config: dict[str, Any],
) -> WebRTCAnswer:
    """`POST .../realtime/calls` with the browser's offer + session config, returning the answer and call handle.

    This is OpenAI's single-step relay: the offer and session config are sent as a `multipart/form-data`
    body authenticated with the server's own key, so no ephemeral token is involved. `httpx` generates
    the multipart boundary; the OpenAI SDK's own `realtime.calls.create` helper forces a boundary-less
    `Content-Type`, so the raw client is used. (Azure requires a different, two-step flow — see
    [`relay_sdp_offer`][pydantic_ai.realtime._openai_webrtc.relay_sdp_offer].)
    """
    response = await http_client.post(
        calls_url,
        headers={**headers, 'Accept': 'application/sdp'},
        files=[
            ('sdp', (None, sdp_offer, 'application/sdp')),
            ('session', (None, json.dumps(session_config), 'application/json')),
        ],
    )
    return _webrtc_answer_from_response(response, provider_name)


async def relay_sdp_offer(
    *,
    http_client: httpx.AsyncClient,
    calls_url: str,
    ephemeral_token: str,
    provider_name: str,
    sdp_offer: str,
) -> WebRTCAnswer:
    """`POST .../realtime/calls` with the raw SDP offer authenticated by an ephemeral client secret.

    Azure OpenAI's `/realtime/calls` rejects the resource api-key / Entra token with a 401 (`This
    operation requires ephemeral tokens`), and expects the offer as a raw `application/sdp` body rather
    than the multipart form OpenAI accepts. So Azure negotiates in two steps — mint a short-lived client
    secret (which binds the session config), then relay the offer with that secret as a bearer token.
    See <https://learn.microsoft.com/azure/ai-foundry/openai/how-to/realtime-audio-webrtc>.
    """
    response = await http_client.post(
        calls_url,
        headers={'Authorization': f'Bearer {ephemeral_token}', 'Content-Type': 'application/sdp'},
        content=sdp_offer,
    )
    return _webrtc_answer_from_response(response, provider_name)
