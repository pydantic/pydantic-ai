"""Tests for the shared `ManagedVariableCapability` base (source `pydantic_ai_harness.logfire`).

Managed-variable capabilities like `ManagedPrompt` all derive their backing variable through the
base's `_build_managed_variable`: the `<prefix><name>` naming, hyphen-to-underscore normalization,
accidental-prefix stripping, and invalid-name rejection are identical across the three. Rather than
re-assert that shared contract once per capability, it is exercised here once through a minimal
concrete subclass. Each capability's own test module keeps only a smoke test that it wires its name,
prefix, value type, and default into this base correctly, plus its capability-specific branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import logfire
import pytest
from logfire.variables import Variable

from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability, resolution_reason

_PREFIX = 'var__'


@dataclass
class _StrVariable(ManagedVariableCapability[None, str]):
    """The smallest possible capability: a `str` variable declared from a bare name."""

    raw_name: str

    def __post_init__(self) -> None:
        self._resolved = self._new_resolved()
        self._variable = self._build_managed_variable(self.raw_name, prefix=_PREFIX, value_type=str, default='')


@dataclass
class _NamelessStrVariable(ManagedVariableCapability[None, str]):
    """A nameless `str` capability: defers its backing variable build to first run-time use."""

    def __post_init__(self) -> None:
        self._setup_variable(None, prefix=_PREFIX, value_type=str, default='')


def test_name_becomes_prefixed_variable_name() -> None:
    assert _StrVariable('greeting')._variable.name == 'var__greeting'


def test_hyphenated_name_is_normalized() -> None:
    assert _StrVariable('welcome-email')._variable.name == 'var__welcome_email'


def test_prefix_in_name_warns_and_is_stripped() -> None:
    with pytest.warns(UserWarning, match='added automatically') as caught:
        capability = _StrVariable('var__already_prefixed')
    assert capability._variable.name == 'var__already_prefixed'
    # The warning's filename should be this test module (the user's call site), not the
    # library's internal `_managed_variable.py`. Anchors the `stacklevel` against regressions.
    assert caught[0].filename == __file__


def test_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match='invalid variable name'):
        _StrVariable('has spaces')


def test_duplicate_construction_is_idempotent() -> None:
    # Each instance builds its own backing variable directly, so the same name can be declared
    # repeatedly (e.g. shared across agents) without the duplicate-registration error `logfire.var`
    # would raise.
    first = _StrVariable('shared')
    second = _StrVariable('shared')
    assert first._variable.name == second._variable.name == 'var__shared'


def test_explicit_logfire_instance_is_used() -> None:
    # Exercises the explicit-instance branch of variable construction (the default-instance branch is
    # covered by every other construction).
    capability = _StrVariable('with_instance', logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)
    assert capability._variable.name == 'var__with_instance'


def test_resolution_reason_falls_back_to_private_reason() -> None:
    # Older logfire SDKs (before the public `reason` attribute) expose only the private `_reason`;
    # `resolution_reason` reads that when there is no public `reason`.
    class _OldResolved:
        _reason = 'unrecognized_variable'

    assert resolution_reason(cast(Any, _OldResolved())) == 'unrecognized_variable'


def test_resolution_reason_prefers_public_reason() -> None:
    class _Resolved:
        reason = 'resolved'
        _reason = 'old'

    assert resolution_reason(cast(Any, _Resolved())) == 'resolved'


def test_resolution_reason_none_when_neither_attribute_present() -> None:
    # Neither the public `reason` nor the private `_reason` is available: nothing to report.
    class _NoReason:
        pass

    assert resolution_reason(cast(Any, _NoReason())) is None


def test_ensure_variable_returns_variable_built_while_awaiting_lock() -> None:
    # Double-checked lock: the outer check sees no backing variable, but a concurrent first run
    # finishes building it while this run waits for the build lock, so the second check inside the
    # lock returns that already-built variable rather than building a second one.
    capability = _NamelessStrVariable()
    assert capability._built_variable is None

    built = Variable('var__raced', type=str, default='', logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)

    class _RaceLock:
        """Stands in for the build lock, simulating the concurrent build completing on acquire."""

        def __enter__(self) -> _RaceLock:
            capability._variables_by_agent['racer'] = built
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    capability._build_lock = cast(Any, _RaceLock())
    assert capability._ensure_variable_for_agent(cast(Any, SimpleNamespace(name='racer'))) is built
