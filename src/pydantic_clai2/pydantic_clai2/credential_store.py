"""Size-bounded keyring storage for OAuth token bundles."""

import re
from uuid import uuid4

import keyring
from keyring.errors import PasswordDeleteError
from pydantic_ai.exceptions import UserError

_SERVICE = 'pydantic-clai2'
_ACCOUNT = 'openai-codex'
_PREFIX = 'clai-chunks-v1:'
# Windows allows 2560 bytes per credential. Keyring writes UTF-16, so a
# 600-character chunk fits even if every character uses a surrogate pair.
_CHUNK_SIZE = 600
_MAX_BYTES = 2560


def _chunk_services(*, value: str, account: str = _ACCOUNT) -> list[str]:
    if not value.startswith(_PREFIX):
        return []
    match = re.fullmatch(r'clai-chunks-v1:([0-9a-f]{32}):([1-9][0-9]{0,3})', value)
    if match is None:
        raise UserError('Stored credentials are invalid. Reconnect through /model; for Codex run /login openai-codex.')
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


def load_codex_credentials(*, account: str = _ACCOUNT) -> str | None:
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
                'Stored credentials are incomplete. Reconnect through /model; for Codex run /login openai-codex.'
            )
        chunks.append(chunk)
    return ''.join(chunks)


def save_codex_credentials(*, value: str, account: str = _ACCOUNT) -> None:
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
