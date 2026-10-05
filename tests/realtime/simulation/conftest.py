"""The simulator judges the new session core with its own invariants, rather than against the current one."""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai.realtime import _session as realtime_session  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def _shadow_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the new core in shadow, and let the simulator's invariants judge it (see `ShadowChecker`).

    Overrides the realtime suite's fixture of the same name, which compares the two cores' histories: the
    current core's known findings make them disagree on traces the simulator is built to find.
    """
    monkeypatch.setattr(realtime_session, '_CORE_MODE', 'shadow')
