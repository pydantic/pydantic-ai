#!/usr/bin/env python3
"""Fail-soft Logfire event emission for the triage automation scripts.

Emission is best-effort by contract: without `LOGFIRE_TRIAGE_WRITE_TOKEN` or
the `logfire` package every helper is a no-op, and no emission failure may
ever break the GitHub write path it rides along with. The token embeds its
Logfire region, so the SDK routes itself; there is deliberately no base-URL
plumbing here.
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
    try:
        logfire.configure(
            token=token,
            service_name=SERVICE_NAME,
            environment='github-actions',
            console=False,
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
