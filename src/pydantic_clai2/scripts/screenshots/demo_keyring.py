"""A keyring backend for the README screenshots: a JSON file in the throwaway HOME.

`capture.sh` selects it with `PYTHON_KEYRING_BACKEND`, so CLAI saves its fake sign-ins the way it
does with an OS keyring, never touches the real one, and never shows a plaintext-fallback path.
"""

import json
from pathlib import Path

from jaraco.classes.properties import classproperty
from keyring.backend import KeyringBackend

_PATH = Path.home() / '.demo-keyring.json'


def _load() -> dict[str, str]:
    return json.loads(_PATH.read_text()) if _PATH.exists() else {}


class Keyring(KeyringBackend):
    """Passwords keyed by `service/username` in one file; for screenshots only."""

    @classproperty
    def priority(cls) -> float:
        """Any value: `PYTHON_KEYRING_BACKEND` selects this backend by name."""
        return 1

    def get_password(self, service: str, username: str) -> str | None:
        """Return the saved password, if any."""
        return _load().get(f'{service}/{username}')

    def set_password(self, service: str, username: str, password: str) -> None:
        """Save a password."""
        _PATH.write_text(json.dumps({**_load(), f'{service}/{username}': password}))

    def delete_password(self, service: str, username: str) -> None:
        """Forget a password."""
        _PATH.write_text(json.dumps({k: v for k, v in _load().items() if k != f'{service}/{username}'}))
