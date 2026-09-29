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

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, StrictInt, ValidationError
from pydantic_core import to_json

from .._http import AsyncHTTPClient
from ..exceptions import ModelHTTPError, UnexpectedModelBehavior
from .model import RealtimeClientSecret, WebRTCAnswer, WebRTCSession

if TYPE_CHECKING:
    import httpx
    import httpx2


class _ClientSecretResponse(BaseModel):
    model_config = ConfigDict(extra='allow')

    value: str
    expires_at: StrictInt


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


def _raise_for_status(response: httpx.Response | httpx2.Response, model_name: str) -> None:
    """Raise a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] on a non-2xx WebRTC signaling response.

    Signaling failures (401/403 auth, 429 rate limit, 5xx outages) are ordinary provider HTTP errors, not
    unexpected model output, so they go through the standard HTTP exception hierarchy — callers can catch
    `ModelHTTPError`, read `status_code`, and apply their own retry policy — with the response body preserved.
    `model_name` carries the realtime model/deployment (not the provider), matching `ModelHTTPError`'s
    contract everywhere else, so an Azure user with several deployments can tell which one failed.
    """
    if not response.is_success:
        # `not is_success` (not `is_error`) so a 3xx redirect is rejected too: the signaling flow reads
        # a `call_id` from the `Location` header, and a redirect's `Location` would otherwise be mistaken
        # for a created call. `headers` is forwarded so a 429's `Retry-After` reaches `retry_after`.
        raise ModelHTTPError(
            status_code=response.status_code,
            model_name=model_name,
            body=response.text.strip() or None,
            headers=response.headers,
        )


async def mint_client_secret(
    *,
    http_client: AsyncHTTPClient,
    client_secrets_url: str,
    headers: dict[str, str],
    model_name: str,
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
        content=to_json(payload).decode(),
    )
    _raise_for_status(response, model_name)
    try:
        data = _ClientSecretResponse.model_validate_json(response.content)
    except ValidationError as e:
        if any(error['loc'] == ('value',) for error in e.errors()):
            raise UnexpectedModelBehavior('Realtime client-secret response did not include a `value`.') from e
        raise UnexpectedModelBehavior('Realtime client-secret response did not include a numeric `expires_at`.') from e
    provider_details = data.model_dump(exclude={'value'})
    try:
        expires_at = datetime.fromtimestamp(data.expires_at, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as e:
        # A numeric-but-unrepresentable `expires_at` (e.g. `10**100`) passes validation but overflows the
        # platform's timestamp range; surface it as unexpected output rather than a raw OverflowError/OSError.
        raise UnexpectedModelBehavior('Realtime client-secret response `expires_at` is out of range.') from e
    return RealtimeClientSecret(
        value=data.value,
        expires_at=expires_at,
        provider_details=provider_details or None,
    )


def _webrtc_answer_from_response(
    response: httpx.Response | httpx2.Response, provider_name: str, model_name: str
) -> WebRTCAnswer:
    """Build a [`WebRTCAnswer`][pydantic_ai.realtime.WebRTCAnswer] from a `/realtime/calls` response.

    The created call's id comes back in the `Location` header (e.g. `/v1/realtime/calls/rtc_...`), not
    the SDP body, so it is parsed out and carried on the returned [`WebRTCSession`][pydantic_ai.realtime.WebRTCSession].
    `provider_name` identifies the session (the attaching model must match); `model_name` names the model
    on a signaling `ModelHTTPError`.
    """
    _raise_for_status(response, model_name)
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
    http_client: AsyncHTTPClient,
    calls_url: str,
    headers: dict[str, str],
    provider_name: str,
    model_name: str,
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
            ('session', (None, to_json(session_config).decode(), 'application/json')),
        ],
    )
    return _webrtc_answer_from_response(response, provider_name, model_name)


async def relay_sdp_offer(
    *,
    http_client: AsyncHTTPClient,
    calls_url: str,
    ephemeral_token: str,
    provider_name: str,
    model_name: str,
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
    return _webrtc_answer_from_response(response, provider_name, model_name)
