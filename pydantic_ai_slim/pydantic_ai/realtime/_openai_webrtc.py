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
from ..exceptions import UnexpectedModelBehavior
from ._base import RealtimeClientSecret, WebRTCAnswer, WebRTCCall

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


def _raise_for_status(response: httpx.Response, action: str) -> None:
    """Raise a clear error carrying the provider's response body on a non-2xx WebRTC signaling response."""
    if not response.is_error:
        return
    body = response.text.strip()
    message = f'{response.status_code} error {action}'
    if body:  # pragma: no branch - provider errors carry a body
        message = f'{message}: {body}'
    raise UnexpectedModelBehavior(message)


async def mint_client_secret(
    *,
    http_client: httpx.AsyncClient,
    client_secrets_url: str,
    headers: dict[str, str],
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
    _raise_for_status(response, 'minting realtime client secret')
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

    The offer and session config are sent as a `multipart/form-data` body so the server's API key
    stays server-side. `httpx` generates the multipart boundary; the OpenAI SDK's own
    `realtime.calls.create` helper forces a boundary-less `Content-Type`, so the raw client is used.
    The created call's id comes back in the `Location` header, not the SDP body.
    """
    response = await http_client.post(
        calls_url,
        headers={**headers, 'Accept': 'application/sdp'},
        files=[
            ('sdp', (None, sdp_offer, 'application/sdp')),
            ('session', (None, json.dumps(session_config), 'application/json')),
        ],
    )
    _raise_for_status(response, 'negotiating realtime WebRTC call')
    location = response.headers.get('location')
    call_id = parse_call_id(location)
    if call_id is None:
        raise UnexpectedModelBehavior(
            'Realtime WebRTC negotiation did not return a parseable `call_id` in the `Location` header.'
        )
    return WebRTCAnswer(
        sdp=response.text,
        call=WebRTCCall(
            provider_name=provider_name,
            call_id=call_id,
            provider_details={'location': location} if location else None,
        ),
    )
