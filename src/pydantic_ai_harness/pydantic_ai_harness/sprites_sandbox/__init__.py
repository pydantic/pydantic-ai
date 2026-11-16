"""Fly.io Sprites sandbox capability: gives agents a persistent cloud computer to work in.

`SpritesSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `SpritesSandboxBackend` is the Sprites implementation of Pydantic AI's
workspace backend protocol, public for applications that want to create or attach to a Sprite
themselves and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.sprites_sandbox._backend import SpritesSandboxBackend
from pydantic_ai_harness.sprites_sandbox._capability import SpritesSandbox

__all__ = [
    'SpritesSandbox',
    'SpritesSandboxBackend',
]
