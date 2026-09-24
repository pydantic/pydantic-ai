"""Size-bounded keyring storage for OAuth token bundles, with a private file when no keyring exists."""

import os
import re
from pathlib import Path
from uuid import uuid4

import keyring
from keyring.errors import InitError, NoKeyringError, PasswordDeleteError
from pydantic_ai.exceptions import UserError

_SERVICE = 'pydantic-clai2'
_ACCOUNT = 'openai-codex'
_PREFIX = 'clai-chunks-v1:'
# Windows allows 2560 bytes per credential. Keyring writes UTF-16, so a
# 600-character chunk fits even if every character uses a surrogate pair.
_CHUNK_SIZE = 600
_MAX_BYTES = 2560
# A locked or failing keyring is not "no keyring": only these mean nothing is configured.
_NO_KEYRING = (NoKeyringError, InitError)


def credentials_path(*, account: str = _ACCOUNT) -> Path:
    """The private fallback file for one account, in the user's CLAI config directory.

    Credentials are per user, like keyring entries, so `--database` does not move them.
    Mirrors `SettingsStore`'s default directory rule, which cannot be imported here
    without constructing a database.
    """
    root = Path(os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'pydantic-clai2'
    return root / f'credentials-{account}.json'


def _chunk_services(*, value: str, account: str = _ACCOUNT) -> list[str]:
    if not value.startswith(_PREFIX):
        return []
    match = re.fullmatch(r'clai-chunks-v1:([0-9a-f]{32}):([1-9][0-9]{0,3})', value)
    if match is None:
        raise UserError(
            'Stored credentials are invalid. Reconnect through /add_model; for Codex run /login openai-codex.'
        )
    generation, count = match.groups()
    # Separate services avoid Windows keyring's multi-account collision handling.
    return [f'{_SERVICE}.{account}.{generation}.{index}' for index in range(int(count))]


def _write(*, service: str, value: str, account: str = _ACCOUNT) -> None:
    keyring.set_password(service, account, value)
    if keyring.get_password(service, account) != value:
        raise UserError('The credential backend did not retain the login. Configure an OS keyring backend.')


def _delete(*, services: list[str], account: str = _ACCOUNT) -> None:
    for service in services:
        try:
            keyring.delete_password(service, account)
        except PasswordDeleteError:
            pass  # A failed write may not have created the entry.


def write_private(*, path: Path, value: str) -> None:
    """Replace the file atomically, without ever writing through something already there.

    A unique staging name plus `O_EXCL` means a symlink planted where the staging file
    would go is refused rather than followed, concurrent saves cannot collide, and the
    `0600` mode applies to a file this call created rather than an attacker's choice.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(f'{path.name}.*.tmp'):
        stale.unlink(missing_ok=True)  # A killed write must not leave tokens on disk.
    staging = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            file.write(value)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)  # No-op once the replace succeeded.


def _fallback_file(*, account: str, fallback: Path | None) -> Path:
    return fallback if fallback is not None else credentials_path(account=account)


def load_codex_credentials(*, account: str = _ACCOUNT, fallback: Path | None = None) -> str | None:
    """Read from keyring, or from the fallback file when keyring is empty or absent."""
    try:
        value = _load_keyring(account=account)
    except _NO_KEYRING:
        value = None
    if value is not None:
        return value
    path = _fallback_file(account=account, fallback=fallback)
    return path.read_text(encoding='utf-8') if path.is_file() else None


def save_codex_credentials(*, value: str, account: str = _ACCOUNT, fallback: Path | None = None) -> None:
    """Write to keyring and drop any plaintext copy; write the file only when no keyring exists."""
    try:
        _save_keyring(value=value, account=account)
    except _NO_KEYRING:
        write_private(path=_fallback_file(account=account, fallback=fallback), value=value)
        return
    _fallback_file(account=account, fallback=fallback).unlink(missing_ok=True)


def delete_credentials(*, account: str = _ACCOUNT, fallback: Path | None = None) -> None:
    """Forget a login everywhere it may be: keyring entry, its chunks, and the fallback file."""
    try:
        value = keyring.get_password(_SERVICE, account)
    except _NO_KEYRING:
        value = None
    if value is not None:
        try:
            services = _chunk_services(value=value, account=account)
        except UserError:
            services = []
        _delete(services=[*services, _SERVICE], account=account)
    _fallback_file(account=account, fallback=fallback).unlink(missing_ok=True)


def _load_keyring(*, account: str = _ACCOUNT) -> str | None:
    """Read either a legacy single entry or a complete chunked token bundle."""
    value = keyring.get_password(_SERVICE, account)
    if value is None:
        return None
    services = _chunk_services(value=value, account=account)
    if not services:
        return value
    chunks: list[str] = []
    for service in services:
        chunk = keyring.get_password(service, account)
        if chunk is None:
            raise UserError(
                'Stored credentials are incomplete. Reconnect through /add_model; for Codex run /login openai-codex.'
            )
        chunks.append(chunk)
    return ''.join(chunks)


def _save_keyring(*, value: str, account: str = _ACCOUNT) -> None:
    """Publish verified chunks before replacing the current login's entry."""
    previous = keyring.get_password(_SERVICE, account)
    # A new login must also be able to replace a corrupt manifest.
    try:
        old_services = _chunk_services(value=previous or '', account=account)
    except UserError:
        old_services = []
    if len(value.encode('utf-16-le')) > _MAX_BYTES:
        chunks = [value[index : index + _CHUNK_SIZE] for index in range(0, len(value), _CHUNK_SIZE)]
        manifest = f'{_PREFIX}{uuid4().hex}:{len(chunks)}'
        services = _chunk_services(value=manifest, account=account)
        try:
            for service, chunk in zip(services, chunks, strict=True):
                _write(service=service, value=chunk, account=account)
        except Exception:
            _delete(services=services, account=account)
            raise
        value = manifest
    # Do not delete new chunks on an uncertain root write: the backend may have
    # committed it before reporting an error. The previous bundle stays intact.
    _write(service=_SERVICE, value=value, account=account)
    _delete(services=old_services, account=account)
