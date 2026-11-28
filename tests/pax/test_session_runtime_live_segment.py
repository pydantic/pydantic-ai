"""Tests for SessionRuntime._live_segment counting sessions waiting on the user.

Fixes pydantic/pydantic-ai #14745 (hidden Pax sessions waiting on an
approval or a question show nothing in the terminal).

_live_segment only touches self.process.live.listing() and
self.terminal, so the runtime is built bare with object.__new__ and stub
attributes instead of a full SessionRuntime (which needs a loader).
"""

from pydantic_ai_pax.runtime.session_runtime import SessionRuntime


class _StubSession:
    """A live session as listing() returns it; only waiting_for matters here."""

    def __init__(self, waiting_for: str | None) -> None:
        self.waiting_for = waiting_for


class _StubLive:
    def __init__(self, sessions: list[_StubSession]) -> None:
        self._sessions = sessions

    def listing(self) -> list[_StubSession]:
        return list(self._sessions)


class _StubProcess:
    def __init__(self, sessions: list[_StubSession]) -> None:
        self.live = _StubLive(sessions)


def _runtime(sessions: list[_StubSession], *, terminal: object | None = object()) -> SessionRuntime:
    runtime = object.__new__(SessionRuntime)
    runtime.process = _StubProcess(sessions)
    runtime.terminal = terminal
    return runtime


def test_no_terminal_shows_nothing() -> None:
    runtime = _runtime([_StubSession('a tool approval'), _StubSession(None)], terminal=None)
    assert runtime._live_segment() == ''


def test_single_live_session_shows_nothing() -> None:
    runtime = _runtime([_StubSession('a tool approval')])
    assert runtime._live_segment() == ''


def test_two_live_none_waiting() -> None:
    runtime = _runtime([_StubSession(None), _StubSession(None)])
    assert runtime._live_segment() == '2 live'


def test_two_live_one_waiting() -> None:
    runtime = _runtime([_StubSession('a tool approval'), _StubSession(None)])
    assert runtime._live_segment() == '2 live · 1 waiting'


def test_three_live_two_waiting() -> None:
    runtime = _runtime([_StubSession('a tool approval'), _StubSession('an ask_user question'), _StubSession(None)])
    assert runtime._live_segment() == '3 live · 2 waiting'


# --- Approval reporting (not included: no anchor on current main) ---
# The draft patch sketched wrapping the await that blocks on an approval
# answer in SessionRuntime.ask_in_composer with
# self.live_session.wait_for_user('a tool approval') (modeled on
# QuestionStatus._asked in question_status.py). On current main
# (e0e2ff9), ask_in_composer exists nowhere in the repo (searched all
# .py files; no 'composer' reference in session_runtime.py), so that wrap
# has no anchor point and is intentionally left out of this PR. The
# counting change above still surfaces any wait reported through
# wait_for_user -- ask_user waits today, via QuestionStatus -- as soon as
# an approval path reports one the same way.
