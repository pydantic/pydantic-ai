"""Server configuration models, in the JSON shape Code Puppy's `/mcp` form edits."""

import logging
import os
from collections.abc import Mapping
from string import Template
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, HttpUrl, TypeAdapter, model_validator

ServerName = Annotated[str, Field(pattern=r'^[A-Za-z][A-Za-z0-9-]{0,63}$')]
"""Underscores are reserved for the `server_tool` separator, so two pairs cannot produce one name."""

ServerType = Literal['stdio', 'http', 'sse']
OAUTH_TIMEOUT = 330.0
"""Seconds to allow for the initialize handshake when it includes a browser sign-in."""

# FastMCP logs the OAuth URL and callback server at INFO through its own handler, over the prompt.
# The browser opening is the signal; an explicit `FASTMCP_LOG_LEVEL` still wins for debugging.
if 'FASTMCP_LOG_LEVEL' not in os.environ:  # pragma: no branch
    logging.getLogger('fastmcp').setLevel(logging.WARNING)


class ServerSettings(BaseModel):
    """Common server options at the trusted configuration boundary."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    enabled: bool = True
    timeout: float | None = Field(default=None, gt=0)
    """Seconds allowed for the initialize handshake; core's default when unset."""


class StdioServer(ServerSettings):
    """A local program, launched without a shell by the MCP client."""

    type: Literal['stdio']
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None


class RemoteServer(ServerSettings):
    """An MCP endpoint reached over the network."""

    url: HttpUrl
    headers: dict[str, str] | None = None
    auth: Literal['oauth'] | None = None
    """`oauth` lets FastMCP run discovery, PKCE, and a browser sign-in on connect; tokens go to the keyring."""

    @model_validator(mode='after')
    def _oauth_rules(self) -> 'RemoteServer':
        if self.auth is None:
            return self
        if any(key.lower() == 'authorization' for key in self.headers or {}):
            raise ValueError('OAuth sets the Authorization header itself; remove it from headers')
        loopback = self.url.host in ('localhost', '127.0.0.1', '[::1]')
        if self.url.scheme != 'https' and not loopback:
            raise ValueError('OAuth needs an https URL, except for a loopback server')
        if self.url.username or self.url.password:
            raise ValueError('OAuth URLs cannot carry credentials')
        return self

    def init_timeout(self) -> float | None:
        """The configured timeout, or enough time for a browser sign-in when using OAuth."""
        return self.timeout or (OAUTH_TIMEOUT if self.auth else None)


class HTTPServer(RemoteServer):
    """A Streamable HTTP MCP endpoint."""

    type: Literal['http']


class SSEServer(RemoteServer):
    """A Server-Sent Events MCP endpoint, the transport older servers use."""

    type: Literal['sse']


def _legacy_key(value: object) -> object:
    """Plugin settings written before `/mcp install` named the discriminator `transport`."""
    if isinstance(value, dict):
        raw = _RAW.validate_python(value)
        if 'transport' in raw and 'type' not in raw:
            raw['type'] = raw.pop('transport')
        return raw
    return value


_RAW: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])

Server = Annotated[StdioServer | HTTPServer | SSEServer, Field(discriminator='type'), BeforeValidator(_legacy_key)]
Servers = dict[ServerName, Server]


class MCPSettings(BaseModel):
    """Plugin settings. Server names also prefix tool names to avoid cross-server collisions."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    servers: Servers = Field(default_factory=dict[str, StdioServer | HTTPServer | SSEServer])


def http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    follow_redirects: bool = False,
) -> httpx.AsyncClient:
    """Do not let a configured endpoint redirect MCP requests to another server.

    FastMCP passes `follow_redirects=True` to every client factory; it is accepted and overridden.
    """
    del follow_redirects
    return httpx.AsyncClient(
        headers=headers, timeout=timeout or httpx.Timeout(30, read=300), auth=auth, follow_redirects=False
    )


def references(server: Server) -> list[str]:
    """Environment variables named by `$VAR` in `env` values or `headers`, in first-use order."""
    values = server.env if isinstance(server, StdioServer) else server.headers
    # `Template.get_identifiers` is 3.11+; this is its implementation for the default pattern.
    matches = (match for value in (values or {}).values() for match in Template.pattern.finditer(value))
    names = (name for match in matches if (name := match['named'] or match['braced']))
    return list(dict.fromkeys(names))


def missing(server: Server) -> list[str]:
    """Referenced variables the current environment does not set."""
    return [name for name in references(server) if name not in os.environ]


def resolve(values: Mapping[str, str] | None) -> dict[str, str] | None:
    """Fill `$VAR` references from the environment at connect time, so saved files hold no secrets."""
    if values is None:
        return None
    return {key: Template(value).safe_substitute(os.environ) for key, value in values.items()}


def target(server: Server) -> str:
    """What the server runs or contacts, for listings. Env, headers, URL credentials, and queries are not shown."""
    if isinstance(server, StdioServer):
        return ' '.join([server.command, *server.args])
    url = server.url
    port = f':{url.port}' if url.port not in (None, 80, 443) else ''
    return f'{url.scheme}://{url.host}{port}{url.path or ""}'
