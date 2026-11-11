"""Credentials encrypted with one OS keyring key, with a private plaintext file when no keyring exists.

macOS asks before an app reads each keychain item, so one item per account (and per chunk of a large
bundle) meant one password prompt per credential at startup. The keyring now holds a single key, read at
most once per process; each account's secret is an encrypted file next to its lock file.
"""

import os
import re
import sqlite3
from collections.abc import Generator
from contextlib import closing, contextmanager
from functools import cache
from pathlib import Path
from uuid import uuid4

import keyring
from cryptography.fernet import Fernet, InvalidToken
from keyring.errors import InitError, KeyringError, NoKeyringError, PasswordDeleteError

from pydantic_ai.exceptions import UserError

_SERVICE = 'pydantic-clai2'
_ACCOUNT = 'openai-codex'
_KEY_ACCOUNT = 'encryption-key'
_PREFIX = 'clai-chunks-v1:'
_SLASH = '%2F'
"""How `/` is written in a credential file name; no account name CLAI stores otherwise contains `%`."""
# A locked or failing keyring is not "no keyring": only these mean nothing is configured.
_NO_KEYRING = (NoKeyringError, InitError)


def credentials_path(*, account: str = _ACCOUNT) -> Path:
    """The private fallback file for one account, in the user's CLAI config directory.

    Credentials are per user, like keyring entries, so `--database` does not move them.
    Mirrors `SettingsStore`'s default directory rule, which cannot be imported here
    without constructing a database.
    """
    root = Path(os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'pydantic-clai2'
    # A gateway profile such as `gateway/openai@work` stays one file in this folder, not a subfolder.
    return root / f'credentials-{account.replace("/", _SLASH)}.json'


_NO_OLDER_ENTRY: set[str] = set()
"""Default accounts already found to have no older per-entry login; this CLAI never writes one."""


def has_credentials(*, account: str) -> bool:
    """Whether a login is saved for `account`.

    A file answers without decrypting anything or reading the keyring. A default account without one
    may still have a login an older CLAI saved as its own keyring entry; that entry is moved into its
    file, as loading it would. Profiles came after per-entry logins, so they are never looked up, and a
    default account found without an entry is not looked up again this process, so turns read no entry.
    """
    encrypted, plaintext = _files(account=account, fallback=None)
    if encrypted.is_file() or plaintext.is_file():
        return True
    if '@' in account or account in _NO_OLDER_ENTRY:
        return False
    try:
        found = load_codex_credentials(account=account) is not None
    except (KeyringError, UserError):
        return False  # A locked keyring or a lost key: look again next time.
    if not found:
        _NO_OLDER_ENTRY.add(account)
    return found


def profile_accounts() -> list[str]:
    """Saved `PROVIDER@PROFILE` accounts, read from file names alone; nothing is decrypted."""
    paths = credentials_path(account='').parent.glob('credentials-*@*')
    names = {path.stem.removeprefix('credentials-') for path in paths if path.suffix in ('.enc', '.json')}
    return sorted(name.replace(_SLASH, '/') for name in names)


@contextmanager
def credential_lock(*, account: str, busy: str) -> Generator[None]:
    """Serialize one account's read-modify-write across processes; the SQLite lock file holds no secrets.

    `busy` is the error shown when the lock cannot be taken, so it can name what to close.
    """
    path = credentials_path(account=account).with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(sqlite3.connect(path, timeout=20)) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            yield
    except sqlite3.Error:
        raise UserError(busy) from None


def write_private(*, path: Path, value: str, replace: bool = True) -> None:
    """Publish the file atomically, without ever writing through something already there.

    A unique staging name plus `O_EXCL` means a symlink planted where the staging file
    would go is refused rather than followed, concurrent saves cannot collide, and the
    `0600` mode applies to a file this call created rather than an attacker's choice.
    `replace=False` raises `FileExistsError` instead of replacing a file another process wrote first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(f'{path.name}.*.tmp'):
        stale.unlink(missing_ok=True)  # A killed write must not leave tokens on disk.
    staging = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            file.write(value)
        if replace:
            os.replace(staging, path)
        else:
            os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)  # No-op once the replace succeeded.


@cache
def _stored_key() -> str | None:
    """The one keyring read a session needs. Cached, so the OS asks for permission at most once."""
    return keyring.get_password(_SERVICE, _KEY_ACCOUNT)


def _cipher() -> Fernet | None:
    """`None` also for a malformed key: loads then ask to reconnect, and the next save replaces it."""
    if (key := _stored_key()) is None:
        return None
    try:
        return Fernet(key)
    except ValueError:
        return None


def _create_cipher() -> Fernet:
    if (cipher := _cipher()) is not None:
        return cipher
    with credential_lock(account=_KEY_ACCOUNT, busy='Cannot lock the credential key. Close other CLAI sessions.'):
        _stored_key.cache_clear()  # Another process may have created it while this one waited.
        if (cipher := _cipher()) is not None:
            return cipher
        key = Fernet.generate_key().decode()
        keyring.set_password(_SERVICE, _KEY_ACCOUNT, key)
        _stored_key.cache_clear()
        if _stored_key() != key:
            raise UserError('The credential backend did not retain the key. Configure an OS keyring backend.')
        return Fernet(key)


def _files(*, account: str, fallback: Path | None) -> tuple[Path, Path]:
    """The encrypted file used with a keyring, and the plaintext file used without one."""
    plaintext = fallback if fallback is not None else credentials_path(account=account)
    return plaintext.with_suffix('.enc'), plaintext


def _decrypt(path: Path) -> str:
    if (cipher := _cipher()) is not None:
        try:
            return cipher.decrypt(path.read_bytes()).decode()
        except InvalidToken:
            pass
    raise UserError(
        'Stored credentials cannot be decrypted: the keyring lost their key. Reconnect through /model add, '
        'or /keys for saved API keys; for Codex run /login openai-codex.'
    )


def load_codex_credentials(*, account: str = _ACCOUNT, fallback: Path | None = None) -> str | None:
    """Read the encrypted file, a login saved per keyring entry by an older CLAI, or the plaintext fallback."""
    encrypted, plaintext = _files(account=account, fallback=fallback)
    try:
        if encrypted.is_file():
            return _decrypt(encrypted)
        root = keyring.get_password(_SERVICE, account)
    except _NO_KEYRING:
        root = None
    if root is not None:
        return _migrate(root=root, account=account, encrypted=encrypted)
    return plaintext.read_text(encoding='utf-8') if plaintext.is_file() else None


def save_codex_credentials(*, value: str, account: str = _ACCOUNT, fallback: Path | None = None) -> None:
    """Encrypt to a file and drop any other copy; write the plaintext file only when no keyring exists."""
    encrypted, plaintext = _files(account=account, fallback=fallback)
    try:
        cipher = _create_cipher()
    except _NO_KEYRING:
        write_private(path=plaintext, value=value)
        return
    write_private(path=encrypted, value=cipher.encrypt(value.encode()).decode())
    _delete_keyring(account=account)
    plaintext.unlink(missing_ok=True)


def replace_credentials(*, value: str, account: str) -> bool:
    """Replace a saved login with refreshed tokens; `False`, writing nothing, once there is none.

    Holds the account's lock, which signing out holds while it deletes, so a refresh that finishes
    after a sign-out cannot sign the account back in: it either saves first and is then deleted, or
    finds nothing to replace.
    """
    with credential_lock(account=account, busy=f'Another CLAI is updating the {account} login. Try again.'):
        encrypted, plaintext = _files(account=account, fallback=None)
        if not (encrypted.is_file() or plaintext.is_file()):
            return False
        save_codex_credentials(value=value, account=account)
        return True


def delete_credentials(*, account: str = _ACCOUNT, fallback: Path | None = None) -> None:
    """Forget a login everywhere it may be: encrypted file, older keyring entries, and the plaintext file."""
    encrypted, plaintext = _files(account=account, fallback=fallback)
    encrypted.unlink(missing_ok=True)
    try:
        _delete_keyring(account=account)
    except _NO_KEYRING:
        pass
    plaintext.unlink(missing_ok=True)


def _migrate(*, root: str, account: str, encrypted: Path) -> str:
    """Move an older per-entry login into its encrypted file, so the next session reads no entry but the key."""
    value = _join_chunks(root=root, account=account)
    try:
        write_private(path=encrypted, value=_create_cipher().encrypt(value.encode()).decode(), replace=False)
    except (FileExistsError, FileNotFoundError):
        # Another process saved or moved it first; its stale-file sweep may even have taken this staging file.
        return _decrypt(encrypted) if encrypted.is_file() else value
    _delete_entries(root=root, account=account)
    return value


def _chunk_services(*, value: str, account: str) -> list[str]:
    if not value.startswith(_PREFIX):
        return []
    match = re.fullmatch(r'clai-chunks-v1:([0-9a-f]{32}):([1-9][0-9]{0,3})', value)
    if match is None:
        raise UserError(
            'Stored credentials are invalid. Reconnect through /model add; for Codex run /login openai-codex.'
        )
    generation, count = match.groups()
    return [f'{_SERVICE}.{account}.{generation}.{index}' for index in range(int(count))]


def _join_chunks(*, root: str, account: str) -> str:
    """An older CLAI's login: the root entry itself, or a bundle chunked for Windows' per-entry size limit."""
    services = _chunk_services(value=root, account=account)
    if not services:
        return root
    chunks: list[str] = []
    for service in services:
        chunk = keyring.get_password(service, account)
        if chunk is None:
            raise UserError(
                'Stored credentials are incomplete. Reconnect through /model add; for Codex run /login openai-codex.'
            )
        chunks.append(chunk)
    return ''.join(chunks)


def _delete_keyring(*, account: str) -> None:
    if (root := keyring.get_password(_SERVICE, account)) is not None:
        _delete_entries(root=root, account=account)


def _delete_entries(*, root: str, account: str) -> None:
    """Drop an older CLAI's entry and its chunks, including a corrupt manifest's own entry."""
    try:
        services = _chunk_services(value=root, account=account)
    except UserError:
        services = []
    for service in [*services, _SERVICE]:
        try:
            keyring.delete_password(service, account)
        except PasswordDeleteError:
            pass  # A chunk may already be gone.
