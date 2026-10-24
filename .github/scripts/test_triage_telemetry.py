"""The telemetry helper must observe the triage scripts without ever breaking them."""

from __future__ import annotations

import sys
from typing import Any

import pytest
import triage_telemetry


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triage_telemetry, '_instance', None)
    monkeypatch.setattr(triage_telemetry, '_disabled', False)
    monkeypatch.delenv('LOGFIRE_TRIAGE_WRITE_TOKEN', raising=False)
    monkeypatch.delenv('LOGFIRE_URL', raising=False)
    monkeypatch.delenv('GITHUB_RUN_ID', raising=False)


class FakeLogfire:
    class AdvancedOptions:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

    def __init__(self) -> None:
        self.configured: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.configure_error: Exception | None = None
        self.info_error: Exception | None = None

    def configure(self, **kwargs: Any) -> None:
        if self.configure_error is not None:
            raise self.configure_error
        self.configured.append(kwargs)

    def info(self, name: str, **attributes: Any) -> None:
        if self.info_error is not None:
            raise self.info_error
        self.events.append((name, attributes))


@pytest.fixture
def fake_logfire(monkeypatch: pytest.MonkeyPatch) -> FakeLogfire:
    fake = FakeLogfire()
    monkeypatch.setitem(sys.modules, 'logfire', fake)
    monkeypatch.setenv('LOGFIRE_TRIAGE_WRITE_TOKEN', 'write-token')
    return fake


def test_emit_is_a_no_op_without_a_token(monkeypatch: pytest.MonkeyPatch):
    fake = FakeLogfire()
    monkeypatch.setitem(sys.modules, 'logfire', fake)

    triage_telemetry.emit('router.sweep', repo='pydantic/pydantic-ai')

    assert fake.configured == []
    assert fake.events == []


def test_emit_configures_once_and_tags_the_workflow_run(fake_logfire: FakeLogfire, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GITHUB_RUN_ID', '4242')

    triage_telemetry.emit('router.sweep', repo='pydantic/pydantic-ai', selected=2)
    triage_telemetry.emit('router.decision', repo='pydantic/pydantic-ai', number=7)

    assert len(fake_logfire.configured) == 1
    assert fake_logfire.configured[0]['token'] == 'write-token'
    assert fake_logfire.configured[0]['advanced'] is None
    assert fake_logfire.events == [
        ('router.sweep', {'github_run_id': '4242', 'repo': 'pydantic/pydantic-ai', 'selected': 2}),
        ('router.decision', {'github_run_id': '4242', 'repo': 'pydantic/pydantic-ai', 'number': 7}),
    ]


def test_a_base_url_routes_to_the_self_hosted_instance(fake_logfire: FakeLogfire, monkeypatch: pytest.MonkeyPatch):
    """A self-hosted token cannot route itself; an empty variable must not override cloud routing."""
    monkeypatch.setenv('LOGFIRE_URL', 'https://logfire-eu.pydantic.info')

    triage_telemetry.emit('census.run', repo='pydantic/pydantic-ai')

    advanced = fake_logfire.configured[0]['advanced']
    assert advanced.base_url == 'https://logfire-eu.pydantic.info'


def test_an_empty_base_url_keeps_the_tokens_own_routing(fake_logfire: FakeLogfire, monkeypatch: pytest.MonkeyPatch):
    """Reusable-workflow callers without the `LOGFIRE_URL` variable pass an empty string."""
    monkeypatch.setenv('LOGFIRE_URL', '')

    triage_telemetry.emit('census.run', repo='pydantic/pydantic-ai')

    assert fake_logfire.configured[0]['advanced'] is None


def test_a_missing_logfire_package_never_raises(monkeypatch: pytest.MonkeyPatch):
    """The monitor's reconcile jobs run on bare stdlib Python without logfire installed."""
    monkeypatch.setenv('LOGFIRE_TRIAGE_WRITE_TOKEN', 'write-token')
    # A `None` entry makes `import logfire` raise ImportError.
    monkeypatch.setitem(sys.modules, 'logfire', None)

    triage_telemetry.emit('census.run', repo='pydantic/pydantic-ai')
    triage_telemetry.emit('census.run', repo='pydantic/pydantic-ai')


def test_a_configuration_failure_never_raises_or_retries(fake_logfire: FakeLogfire):
    fake_logfire.configure_error = RuntimeError('boom')

    triage_telemetry.emit('census.run', repo='pydantic/pydantic-ai')
    triage_telemetry.emit('census.run', repo='pydantic/pydantic-ai')

    assert fake_logfire.events == []


def test_an_emission_failure_never_raises(fake_logfire: FakeLogfire):
    fake_logfire.info_error = RuntimeError('boom')

    triage_telemetry.emit('census.run', repo='pydantic/pydantic-ai')
