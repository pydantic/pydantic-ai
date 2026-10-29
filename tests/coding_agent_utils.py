"""What the tests clear so that the coding agent running `pytest` doesn't decide their outcome."""

from __future__ import annotations

import os

import pydantic_ai._display as _display


def agent_env_vars_in_scope() -> set[str]:
    """Every variable `detect_coding_agent` would read, so the suite decides rather than the shell.

    The agent running the tests sets some of these itself, so leaving any behind would make whether
    a banner appears depend on who ran `pytest`.
    """
    names: set[str] = set(_display._NAMED_AGENT_ENV_VARS)  # pyright: ignore[reportPrivateUsage]
    for _, signals in _display._CODING_AGENTS:  # pyright: ignore[reportPrivateUsage]
        for env_signal in signals:
            if env_signal.endswith('*'):
                names.update(name for name in os.environ if name.startswith(env_signal[:-1]))
            else:
                names.add(env_signal.partition('=')[0])
    return names
