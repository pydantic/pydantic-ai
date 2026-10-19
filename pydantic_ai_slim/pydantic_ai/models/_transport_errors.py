"""Describe a transport failure the provider SDK didn't wrap in its own exception."""

from __future__ import annotations as _annotations


def transport_error_message(e: Exception) -> str:
    """Return the message for the `ModelAPIError` raised in place of a transport failure.

    Some transport errors, like a bare `httpx.ReadTimeout()`, stringify to `''`, so fall back to the class name.
    """
    return str(e) or type(e).__name__
