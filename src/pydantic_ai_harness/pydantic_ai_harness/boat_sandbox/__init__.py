"""Boat sandbox capability: gives agents an isolated cloud computer to work in.

`BoatSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `BoatSandboxBackend` is the Boat implementation of Pydantic AI's
workspace backend protocol, public for applications that want to create or attach to a sandbox
themselves and pass it to a run as `workspace=`. `BoatAPIError` is what a Boat failure that a
retry may fix, such as a rate limit or a server error, propagates as.
"""

from pydantic_ai_harness.boat_sandbox._api import BoatAPIError
from pydantic_ai_harness.boat_sandbox._backend import BoatSandboxBackend
from pydantic_ai_harness.boat_sandbox._capability import BoatSandbox

__all__ = [
    'BoatAPIError',
    'BoatSandbox',
    'BoatSandboxBackend',
]
