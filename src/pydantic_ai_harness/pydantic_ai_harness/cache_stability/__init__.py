"""Deprecated import location for `pydantic_ai_harness.warn_on_cache_busts`.

This capability was renamed; importing from here still works but emits a
`DeprecationWarning`. Import from `pydantic_ai_harness.warn_on_cache_busts` instead.
"""

from pydantic_ai_harness._warn import warn_module_renamed
from pydantic_ai_harness.warn_on_cache_busts import (
    CacheBustWarning,
    WarnOnCacheBusts,
)

CacheStabilityMonitor = WarnOnCacheBusts

warn_module_renamed('cache_stability', 'warn_on_cache_busts')

__all__ = [
    'CacheBustWarning',
    'CacheStabilityMonitor',
]
