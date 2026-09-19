"""Isolate settings and provider access for every CLAI test."""

from pathlib import Path

import keyring
import pytest
from pydantic_ai import models


@pytest.fixture
def anyio_backend() -> str:
    """CLAI's terminal and cancellation primitives require asyncio."""
    return 'asyncio'


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect default databases, including subprocesses, away from user data."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('CLAI_MODEL', raising=False)
    for name in (
        'LOGFIRE_TOKEN',
        'LOGFIRE_API_KEY',
        'OTEL_EXPORTER_OTLP_ENDPOINT',
        'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT',
        'OTEL_EXPORTER_OTLP_METRICS_ENDPOINT',
        'OTEL_EXPORTER_OTLP_LOGS_ENDPOINT',
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('LOGFIRE_CREDENTIALS_DIR', str(tmp_path / 'logfire'))
    monkeypatch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
    credentials: dict[tuple[str, str], str] = {}

    def get_password(service: str, account: str) -> str | None:
        return credentials.get((service, account))

    def set_password(service: str, account: str, value: str) -> None:
        credentials[service, account] = value

    monkeypatch.setattr(keyring, 'get_password', get_password)
    monkeypatch.setattr(keyring, 'set_password', set_password)
    monkeypatch.setenv('PYTHON_KEYRING_BACKEND', 'keyring.backends.null.Keyring')
