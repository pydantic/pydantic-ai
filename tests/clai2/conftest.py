"""Isolate settings and provider access for every CLAI test."""

from pathlib import Path

import keyring
import pytest
from pydantic_ai import models


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect default databases, including subprocesses, away from user data."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('CLAI_MODEL', raising=False)
    monkeypatch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)
    credentials: dict[tuple[str, str], str] = {}

    def get_password(service: str, account: str) -> str | None:
        return credentials.get((service, account))

    def set_password(service: str, account: str, value: str) -> None:
        credentials[service, account] = value

    monkeypatch.setattr(keyring, 'get_password', get_password)
    monkeypatch.setattr(keyring, 'set_password', set_password)
    monkeypatch.setenv('PYTHON_KEYRING_BACKEND', 'keyring.backends.null.Keyring')
