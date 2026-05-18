"""Single import surface for HTTP client types we hand to provider SDKs.

`openai`, `anthropic`, `mistralai`, `cohere`, `litellm` and others runtime-check
their `http_client` argument against `httpx.AsyncClient` — passing an
`httpx2.AsyncClient` fails with `TypeError: Invalid http_client argument`.

Anything we hand to those SDKs (or to libraries that hand it on to them, e.g.
the FastMCP `streamablehttp_client` factory) must therefore be the real `httpx`
class. When the ecosystem migrates to `httpx2`, swap the underlying import here
and call sites pick up the new types automatically.
"""

from __future__ import annotations

import httpx as _impl

AsyncClient = _impl.AsyncClient
Timeout = _impl.Timeout
Auth = _impl.Auth
Response = _impl.Response
Request = _impl.Request
URL = _impl.URL
HTTPStatusError = _impl.HTTPStatusError
HTTPTransport = _impl.HTTPTransport
AsyncHTTPTransport = _impl.AsyncHTTPTransport
BaseTransport = _impl.BaseTransport
AsyncBaseTransport = _impl.AsyncBaseTransport
MockTransport = _impl.MockTransport
Headers = _impl.Headers
Limits = _impl.Limits
TimeoutException = _impl.TimeoutException
ConnectError = _impl.ConnectError
ReadError = _impl.ReadError
RequestError = _impl.RequestError
TransportError = _impl.TransportError
StreamError = _impl.StreamError
StreamClosed = _impl.StreamClosed
ASGITransport = _impl.ASGITransport
AsyncByteStream = _impl.AsyncByteStream
Client = _impl.Client
