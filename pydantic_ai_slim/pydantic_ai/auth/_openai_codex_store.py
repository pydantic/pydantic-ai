from __future__ import annotations as _annotations

import errno
import json
import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, ValidationError

from .._utils import run_in_executor
from .openai_codex import OpenAICodexCredentials, OpenAICodexCredentialsError

_AUTH_FILE_VERSION = 1
_PROVIDER_KEY = 'openai-codex'
_LOCK_TIMEOUT = 60
_LOCK_POLL_INTERVAL = 0.05
# Windows has no `O_NOFOLLOW`; it also has no symbolic links an unprivileged user can plant here.
_O_NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)


class _StoredOpenAICodexCredentials(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    access_token: str
    refresh_token: str
    id_token: str
    expires_at: datetime
    account_id: str
    revision: str
    account_is_fedramp: bool = False

    @classmethod
    def from_credentials(cls, credentials: OpenAICodexCredentials) -> _StoredOpenAICodexCredentials:
        return cls(
            access_token=credentials.access_token.get_secret_value(),
            refresh_token=credentials.refresh_token.get_secret_value(),
            id_token=credentials.id_token.get_secret_value(),
            expires_at=credentials.expires_at,
            account_id=credentials.account_id.get_secret_value(),
            revision=credentials.revision,
            account_is_fedramp=credentials.account_is_fedramp,
        )

    def to_credentials(self) -> OpenAICodexCredentials:
        return OpenAICodexCredentials(
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


class OpenAICodexFileCredentialStore:
    """Private default file store backing [`OpenAICodexAuth`][pydantic_ai.auth.openai_codex.OpenAICodexAuth]."""

    def __init__(self, path: Path | None = None) -> None:
        self._uses_default_path = path is None
        self.path = path or Path.home() / '.pydantic-ai' / 'auth.json'
        self._lock_path = self.path.with_name(f'{self.path.name}.lock')

    @asynccontextmanager
    async def exclusive(self) -> AsyncGenerator[None]:
        # `filelock` is imported here rather than at module scope so that reading and writing the
        # store — and everything upstream of it, like constructing `OpenAICodexAuth` — works without the
        # `openai-codex` extra. Only cross-process locking actually needs it.
        try:
            from filelock import FileLock, Timeout
        except ImportError as _import_error:  # pragma: no cover
            raise ImportError(
                'Please install the `filelock` package to use the default OpenAI Codex credential store, '
                'you can use the `openai-codex` optional group — `pip install "pydantic-ai-slim[openai-codex]"`. '
                'Applications that supply their own `OpenAICodexCredentialStore` do not need it.'
            ) from _import_error

        lock = FileLock(self._lock_path, mode=0o600, thread_local=False)
        try:
            await run_in_executor(self._prepare_directory)
            with anyio.fail_after(_LOCK_TIMEOUT):
                while True:
                    try:
                        with anyio.CancelScope(shield=True):
                            await run_in_executor(lock.acquire, timeout=0)
                    except Timeout:
                        await anyio.sleep(_LOCK_POLL_INTERVAL)
                    else:
                        break
        except TimeoutError:
            raise OpenAICodexCredentialsError(
                'Timed out waiting for exclusive access to OpenAI Codex credentials.'
            ) from None
        except OSError as error:
            raise OpenAICodexCredentialsError('Unable to lock the OpenAI Codex credential store.') from error

        try:
            if os.name != 'nt':  # pragma: no branch
                try:
                    await run_in_executor(self._harden_lock_file)
                except OSError as error:
                    raise OpenAICodexCredentialsError('Unable to lock the OpenAI Codex credential store.') from error
            yield
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_executor(lock.release)

    async def load(self) -> OpenAICodexCredentials | None:
        try:
            return await run_in_executor(self._load_sync)
        except OpenAICodexCredentialsError:
            raise
        except OSError as error:
            raise OpenAICodexCredentialsError('Unable to read the OpenAI Codex credential store.') from error

    async def save(self, credentials: OpenAICodexCredentials, *, expected_revision: str | None) -> bool:
        try:
            return await run_in_executor(self._save_sync, credentials, expected_revision)
        except OpenAICodexCredentialsError:
            raise
        except OSError as error:
            raise OpenAICodexCredentialsError('Unable to write the OpenAI Codex credential store.') from error

    async def delete(self, *, expected_revision: str | None) -> bool:
        try:
            return await run_in_executor(self._delete_sync, expected_revision)
        except OpenAICodexCredentialsError:
            raise
        except OSError as error:
            raise OpenAICodexCredentialsError('Unable to update the OpenAI Codex credential store.') from error

    def _harden_lock_file(self) -> None:
        # Hardening the descriptor rather than the path, for the reason spelled out in
        # `_read_document`: `os.chmod` follows a symbolic link, so a co-local user who can write
        # the parent directory could plant the lock file as a link and redirect the mode change
        # onto an arbitrary file. `filelock` already opened the real lock file with `O_NOFOLLOW`.
        descriptor = os.open(self._lock_path, os.O_RDONLY | _O_NOFOLLOW)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)

    def _prepare_directory(self) -> None:
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (self._uses_default_path or not parent_existed) and os.name != 'nt':
            os.chmod(self.path.parent, 0o700)

    def _load_sync(self) -> OpenAICodexCredentials | None:
        document = self._load_document()
        return self._get_record(document)

    def _save_sync(self, credentials: OpenAICodexCredentials, expected_revision: str | None) -> bool:
        document = self._load_document()
        current = self._get_record(document)
        current_revision = current.revision if current is not None else None
        if current_revision != expected_revision:
            return False

        providers = dict(document.providers)
        providers[_PROVIDER_KEY] = _StoredOpenAICodexCredentials.from_credentials(credentials).model_dump(mode='json')
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

    def _read_document(self) -> bytes | None:
        """Open, harden and read the store through a single descriptor, or `None` if absent.

        Checking the path and then acting on it again leaves a window in which a co-local user who
        can write the parent directory swaps the file for a symbolic link, turning the `0o600`
        hardening into an arbitrary-file chmod. `O_NOFOLLOW` refuses the link at open time instead,
        and `fchmod` then applies to the same descriptor the content is read from. A link is
        refused rather than merely left unhardened because writes go through `os.replace`, which
        substitutes the link itself: the path could never be honored as an indirection anyway.
        """
        try:
            descriptor = os.open(self.path, os.O_RDONLY | _O_NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError as error:
            # `O_NOFOLLOW` on a symbolic link reports `ELOOP`, except on the BSDs, which use `EMLINK`.
            if error.errno not in (errno.ELOOP, errno.EMLINK):
                raise
            raise OpenAICodexCredentialsError(
                'The OpenAI Codex credential store path must not be a symbolic link.'
            ) from None
        with os.fdopen(descriptor, 'rb') as handle:
            if os.name != 'nt':  # pragma: no branch
                os.fchmod(descriptor, 0o600)
            return handle.read()

    def _load_document(self) -> _AuthFile:
        content = self._read_document()
        if content is None:
            return _AuthFile()
        try:
            raw = json.loads(content.decode('utf-8'))
            document = _AuthFile.model_validate(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
            pass
        else:
            return document
        raise OpenAICodexCredentialsError(
            'The OpenAI Codex credential store is malformed or uses an unsupported schema version.'
        ) from None

    def _get_record(self, document: _AuthFile) -> OpenAICodexCredentials | None:
        raw = document.providers.get(_PROVIDER_KEY)
        if raw is None:
            return None
        try:
            record = _StoredOpenAICodexCredentials.model_validate(raw).to_credentials()
        except ValidationError:
            pass
        else:
            return record
        # Raised outside the `except` block on purpose: `from None` only clears `__cause__`,
        # while `__context__` would still hold the `ValidationError` whose payload carries the
        # plaintext credential document. Raising here leaves `__context__` empty.
        raise OpenAICodexCredentialsError('The stored OpenAI Codex credential record is malformed.') from None

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
            # `os.fdopen` takes ownership of the descriptor, so whatever can fail ahead of it has to
            # close the raw descriptor itself; otherwise every failed save leaks one.
            try:
                if os.name != 'nt':  # pragma: no branch
                    os.fchmod(file_descriptor, 0o600)
                temporary_file = os.fdopen(file_descriptor, 'w', encoding='utf-8')
            except BaseException:
                os.close(file_descriptor)
                raise
            with temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        # Past `os.replace` the new document is live and the old revision is gone, so a failure here
        # must not surface as a failed write: the caller would retry with an `expected_revision` the
        # store no longer holds. Neither step is load-bearing — `os.replace` carries over the mode
        # `fchmod` already set, and the directory `fsync` only makes the rename durable — so the one
        # thing they must not do is undo a committed write.
        if os.name != 'nt':  # pragma: no branch
            with suppress(OSError):
                os.chmod(self.path, 0o600)
                directory_descriptor = os.open(self.path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
