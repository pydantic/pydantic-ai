"""Local desktop capability: a screen on this machine for computer use.

`LocalDesktop` is the supported entry point. `LocalDesktopBackend` is its workspace backend, public for applications
that pass a workspace to a run as `workspace=`.
"""

from pydantic_ai_harness.local_desktop._backend import LocalDesktopBackend, VirtualScreen
from pydantic_ai_harness.local_desktop._capability import LocalDesktop

__all__ = [
    'LocalDesktop',
    'LocalDesktopBackend',
    'VirtualScreen',
]
