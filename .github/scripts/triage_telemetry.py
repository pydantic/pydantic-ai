#!/usr/bin/env python3
"""Fail-soft Logfire event emission for the triage automation scripts.

Emission is best-effort by contract: without `LOGFIRE_TRIAGE_WRITE_TOKEN` or
the `logfire` package every helper is a no-op, and no emission failure may
ever break the GitHub write path it rides along with. A cloud token embeds
its Logfire region, so the SDK routes itself; `LOGFIRE_URL` (the same
variable every other Logfire emitter in this repository reads) overrides
that routing for tokens minted on a self-hosted instance.
"""

from __future__ import annotations

import os
import sys
from typing import Any

SERVICE_NAME = 'pydantic-ai-triage'

_instance: Any = None
_disabled = False


def _logfire() -> Any:
    """Configure Logfire on first use; return None whenever emission is unavailable."""
    global _instance, _disabled
    if _disabled or _instance is not None:
        return _instance
    token = os.environ.get('LOGFIRE_TRIAGE_WRITE_TOKEN')
    if not token:
        _disabled = True
        return None
    try:
        import logfire
    except Exception as exc:  # a broken transitive release must not break the write path
        _disabled = True
        print(f'logfire is unavailable ({type(exc).__name__}); skipping triage telemetry', file=sys.stderr)
        return None
    # Callers without the variable pass an empty string; only a real value
    # may override the token's own routing.
    base_url = os.environ.get('LOGFIRE_URL')
    try:
        logfire.configure(
            token=token,
            service_name=SERVICE_NAME,
            environment='github-actions',
            console=False,
            advanced=logfire.AdvancedOptions(base_url=base_url) if base_url else None,
        )
    except Exception as exc:  # telemetry must never break the GitHub write path
        _disabled = True
        print(f'triage telemetry configuration failed: {type(exc).__name__}', file=sys.stderr)
        return None
    _instance = logfire
    return _instance


def emit(name: str, **attributes: object) -> None:
    """Record one event, tagged with the enclosing GitHub Actions run for grouping."""
    instance = _logfire()
    if instance is None:
        return
    payload: dict[str, object] = {'github_run_id': os.environ.get('GITHUB_RUN_ID'), **attributes}
    try:
        instance.info(name, **payload)
    except Exception as exc:  # telemetry must never break the GitHub write path
        print(f'triage telemetry emission failed: {type(exc).__name__}', file=sys.stderr)
