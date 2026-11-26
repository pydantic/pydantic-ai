"""CDP Browser capability: a throwaway Chrome the model drives with raw Chrome DevTools Protocol commands."""

from pydantic_ai_harness._browser_egress import (
    DEFAULT_ALLOWLIST_REACH,
    DEFAULT_RESOLVED_KINDS,
    EgressPolicy,
    EgressRequest,
    RequestKind,
)
from pydantic_ai_harness.cdp_browser._capability import CDPBrowser
from pydantic_ai_harness.cdp_browser._toolset import DEFAULT_ALLOWED_CDP_METHODS, CDPBrowserToolset

__all__ = [
    'DEFAULT_ALLOWED_CDP_METHODS',
    'DEFAULT_ALLOWLIST_REACH',
    'DEFAULT_RESOLVED_KINDS',
    'CDPBrowser',
    'CDPBrowserToolset',
    'EgressPolicy',
    'EgressRequest',
    'RequestKind',
]
