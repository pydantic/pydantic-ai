from __future__ import annotations as _annotations

import json
import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal

import anyio
from anyio.to_thread import run_sync
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, ValidationError

from .codex import CodexCredentials, CodexCredentialsError

_AUTH_FILE_VERSION = 1
_PROVIDER_KEY = 'codex'
_LOCK_TIMEOUT = 60
_LOCK_POLL_INTERVAL = 0.05


class _StoredCodexCredentials(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    access_token: str
    refresh_token: str
    id_token: str
    expires_at: datetime
    account_id: str
    revision: str
    account_is_fedramp: bool = False

    @classmethod
    def from_credentials(cls, credentials: CodexCredentials) -> _StoredCodexCredentials:
        return cls(
            access_token=credentials.access_token.get_secret_value(),
            refresh_token=credentials.refresh_token.get_secret_value(),
            id_token=credentials.id_token.get_secret_value(),
            expires_at=credentials.expires_at,
            account_id=credentials.account_id.get_secret_value(),
            revision=credentials.revision,
            account_is_fedramp=credentials.account_is_fedramp,
        )

    def to_credentials(self) -> CodexCredentials:
        return CodexCredentials(
            access_token=SecretStr(self.access_token),
            refresh_token=SecretStr(self.refresh_token),
            id_token=SecretStr(self.id_token),
            expires_at=self.expires_at,
            account_id=SecretStr(self.account_id),
            revision=self.revision,
            account_is_fedramp=self.account_is_fedramp,
        )


class _AuthFile(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    version: Literal[1] = _AUTH_FILE_VERSION
    providers: dict[str, JsonValue] = Field(default_factory=dict)


class FileCodexCredentialStore:
    """Private default file store backing [`CodexAuth`][pydantic_ai.auth.codex.CodexAuth]."""

    def __init__(self, path: Path | None = None) -> None:
        self._uses_default_path = path is None
        self.path = path or Path.home() / '.pydantic-ai' / 'auth.json'
        self._lock_path = self.path.with_name(f'{self.path.name}.lock')

    @asynccontextmanager
    async def exclusive(self) -> AsyncGenerator[None]:
        lock = FileLock(self._lock_path, mode=0o600, thread_local=False)
        try:
            await run_sync(self._prepare_directory)
            with anyio.fail_after(_LOCK_TIMEOUT):
                while True:
                    try:
                        with anyio.CancelScope(shield=True):
                            await run_sync(partial(lock.acquire, timeout=0))
                    except Timeout:
                        await anyio.sleep(_LOCK_POLL_INTERVAL)
                    else:
                        break
        except TimeoutError:
            raise CodexCredentialsError('Timed out waiting for exclusive access to Codex credentials.') from None
        except OSError as error:
            raise CodexCredentialsError('Unable to lock the Codex credential store.') from error

        try:
            if os.name != 'nt':  # pragma: no branch - platform-specific permission hardening
                try:
                    await run_sync(os.chmod, self._lock_path, 0o600)
                except OSError as error:
                    raise CodexCredentialsError('Unable to lock the Codex credential store.') from error
            yield
        finally:
            with anyio.CancelScope(shield=True):
                await run_sync(lock.release)

    async def load(self) -> CodexCredentials | None:
        try:
            return await run_sync(self._load_sync)
        except CodexCredentialsError:
            raise
        except OSError as error:
            raise CodexCredentialsError('Unable to read the Codex credential store.') from error

    async def save(self, credentials: CodexCredentials, *, expected_revision: str | None) -> bool:
        try:
            return await run_sync(self._save_sync, credentials, expected_revision)
        except CodexCredentialsError:
            raise
        except OSError as error:
            raise CodexCredentialsError('Unable to write the Codex credential store.') from error

    async def delete(self, *, expected_revision: str | None) -> bool:
        try:
            return await run_sync(self._delete_sync, expected_revision)
        except CodexCredentialsError:
            raise
        except OSError as error:
            raise CodexCredentialsError('Unable to update the Codex credential store.') from error

    def _prepare_directory(self) -> None:
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (self._uses_default_path or not parent_existed) and os.name != 'nt':
            os.chmod(self.path.parent, 0o700)

    def _load_sync(self) -> CodexCredentials | None:
        document = self._load_document()
        return self._get_record(document)

    def _save_sync(self, credentials: CodexCredentials, expected_revision: str | None) -> bool:
        document = self._load_document()
        current = self._get_record(document)
        current_revision = current.revision if current is not None else None
        if current_revision != expected_revision:
            return False

        providers = dict(document.providers)
        providers[_PROVIDER_KEY] = _StoredCodexCredentials.from_credentials(credentials).model_dump(mode='json')
        self._atomic_write(_AuthFile(version=_AUTH_FILE_VERSION, providers=providers))
        return True

    def _delete_sync(self, expected_revision: str | None) -> bool:
        document = self._load_document()
        current = self._get_record(document)
        if current is None:
            return False
        if current.revision != expected_revision:
            return False

        providers = dict(document.providers)
        del providers[_PROVIDER_KEY]
        self._atomic_write(_AuthFile(version=_AUTH_FILE_VERSION, providers=providers))
        return True

    def _load_document(self) -> _AuthFile:
        if not self.path.exists():
            return _AuthFile()
        if os.name != 'nt':  # pragma: no branch - platform-specific permission hardening
            os.chmod(self.path, 0o600)
        try:
            raw = json.loads(self.path.read_text(encoding='utf-8'))
            document = _AuthFile.model_validate(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
            pass
        else:
            return document
        raise CodexCredentialsError(
            'The Codex credential store is malformed or uses an unsupported schema version.'
        ) from None

    def _get_record(self, document: _AuthFile) -> CodexCredentials | None:
        raw = document.providers.get(_PROVIDER_KEY)
        if raw is None:
            return None
        try:
            record = _StoredCodexCredentials.model_validate(raw).to_credentials()
        except ValidationError:
            pass
        else:
            return record
        raise CodexCredentialsError('The stored Codex credential record is malformed.') from None

    def _atomic_write(self, document: _AuthFile) -> None:
        self._prepare_directory()
        content = json.dumps(document.model_dump(mode='json'), indent=2, sort_keys=True) + '\n'
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f'.{self.path.name}.',
            suffix='.tmp',
        )
        temporary_path = Path(temporary_name)
        try:
            if os.name != 'nt':  # pragma: no branch - platform-specific permission hardening
                os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, 'w', encoding='utf-8') as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
            if os.name != 'nt':  # pragma: no branch - platform-specific permission hardening
                os.chmod(self.path, 0o600)
                directory_descriptor = os.open(self.path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
