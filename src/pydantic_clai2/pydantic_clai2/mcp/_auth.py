"""The `Auth` base class the installed FastMCP accepts: legacy `httpx` under FastMCP 3, `httpx2` under FastMCP 4.

This workspace locks FastMCP 3, but an install that does not use the lock (`uvx --from git+...`, a fresh
`pip install`) can resolve FastMCP 4, whose transports reject a legacy `httpx.Auth` with
"Invalid "auth" argument". An auth that subclasses `MCPAuth` works under both; its flows only read
headers and status codes, which both generations share.
"""

from typing import TYPE_CHECKING

import httpx


def _auth_base() -> type[httpx.Auth]:
    try:
        import fastmcp

        major = int(fastmcp.__version__.split('.')[0])
    except (ImportError, ValueError, AttributeError):
        return httpx.Auth
    if major < 4:
        return httpx.Auth
    import httpx2

    return httpx2.Auth  # pyright: ignore[reportReturnType]


if TYPE_CHECKING:
    MCPAuth = httpx.Auth
else:
    MCPAuth = _auth_base()
