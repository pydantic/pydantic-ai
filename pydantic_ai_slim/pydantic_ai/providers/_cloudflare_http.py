"""Native Workers AI HTTP error mapping."""

import httpx2

from .. import _model_errors
from .._utils import error_body
from ..exceptions import ModelHTTPError


async def send_request(client: httpx2.AsyncClient, request: httpx2.Request, *, model_name: str) -> httpx2.Response:
    try:
        response = await client.send(request)
    except httpx2.TransportError as e:
        raise _model_errors.connection_error(
            model_name, f'{type(e).__name__}: {e}', e, timeout=isinstance(e, httpx2.TimeoutException)
        ) from e
    if response.is_error:
        body: object = error_body(response)
        raise ModelHTTPError.for_category(
            _model_errors.http_status_category(response.status_code, str(body)),
            status_code=response.status_code,
            model_name=model_name,
            body=body,
            headers=dict(response.headers),
        )
    return response
