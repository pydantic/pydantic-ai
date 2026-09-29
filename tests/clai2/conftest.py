"""Isolate settings and provider access for every CLAI test."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import keyring
import pytest

from pydantic_ai import models


@pytest.fixture
def blockbuster_enabled() -> bool:
    """Not yet: the suite predates the detector. https://github.com/pydantic/pydantic-ai/issues/8821"""
    return False


@pytest.fixture
def anyio_backend() -> str:
    """CLAI's terminal and cancellation primitives require asyncio."""
    return 'asyncio'


@dataclass
class FakeGh:
    """The fake `gh`'s state: tokens per host, the next `auth login` outcome, and opened URLs."""

    state: Path
    opened: list[str] = field(default_factory=list[str])

    def sign_in(self, host: str = 'github.com', token: str = 'gho_saved') -> None:
        (self.state / host).write_text(token)

    def next_login(self, mode: str) -> None:
        """`ok`, `fail`, `hang` (after the code), `silent` (before it), `early`, `no-token`, or `stall-token` (then `auth token` hangs)."""
        (self.state / 'login').write_text(mode)

    def hang_on_token(self) -> None:
        (self.state / 'token-hang').write_text('')

    def calls(self) -> list[str]:
        path = self.state / 'calls'
        return path.read_text().splitlines() if path.exists() else []


@pytest.fixture(autouse=True)
def fake_gh(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    """Never run the real GitHub CLI or open a browser: either would touch the developer's own login."""
    fake = FakeGh(state=tmp_path_factory.mktemp('fake-gh'))
    monkeypatch.setenv('FAKE_GH_STATE', str(fake.state))
    script = Path(__file__).with_name('fake_gh_cli.py')
    monkeypatch.setattr('pydantic_clai2.gh_cli.gh_command', lambda: [sys.executable, str(script)])

    def open_browser(url: str) -> bool:
        fake.opened.append(url)
        return True

    monkeypatch.setattr('pydantic_clai2.github.OPEN_BROWSER', open_browser)
    return fake


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

    def delete_password(service: str, account: str) -> None:
        del credentials[service, account]

    monkeypatch.setattr(keyring, 'get_password', get_password)
    monkeypatch.setattr(keyring, 'set_password', set_password)
    monkeypatch.setattr(keyring, 'delete_password', delete_password)
    monkeypatch.setenv('PYTHON_KEYRING_BACKEND', 'keyring.backends.null.Keyring')
