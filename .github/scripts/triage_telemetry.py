#!/usr/bin/env python3
"""Fail-soft Logfire event emission for the triage automation scripts.

Every routing decision -- including a decision to do nothing -- is recorded as
one Logfire event so skipped work stays auditable and maintainer corrections
can be joined back to the run that made the original call. Emission is
best-effort: without ``LOGFIRE_TRIAGE_WRITE_TOKEN`` or the ``logfire`` package
every helper is a no-op, and an emission failure never breaks the GitHub write
path it rides along with.
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
    except ImportError:
        _disabled = True
        print('logfire is not installed; skipping triage telemetry', file=sys.stderr)
        return None
    try:
        base_url = os.environ.get('LOGFIRE_URL')
        advanced = logfire.AdvancedOptions(base_url=base_url) if base_url else None
        logfire.configure(
            token=token,
            service_name=SERVICE_NAME,
            environment='github-actions',
            console=False,
            advanced=advanced,
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
