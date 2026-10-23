"""S3, Google Cloud Storage and Azure Blob Storage as filesystems: workspace backends that store files."""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast
from urllib.parse import unquote, urlencode, urlsplit

import anyio
import anyio.to_thread

try:
    import obstore
    from obstore.exceptions import GenericError, PermissionDeniedError, UnauthenticatedError
    from obstore.store import AzureStore, GCSStore, S3Store
except ImportError as _import_error:
    raise ImportError(
        'Install `pydantic-ai-harness[cloud-storage]` to use cloud storage filesystems.'
    ) from _import_error

from pydantic_ai.workspaces import (
    FileEntry,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceRef,
    WorkspaceUnavailableError,
)

if TYPE_CHECKING:
    from obstore.store import AzureCredential, GCSCredential, ObjectStore, S3Credential

__all__ = ('AzureFilesystem', 'GCSFilesystem', 'S3Filesystem')

_T = TypeVar('_T')

_DIRECTORY_MARKER = '.pydantic-ai-dir'
"""Object that keeps an empty folder: object stores only have folders that hold something."""


@dataclass(frozen=True)
class _ObjectStoreFilesystem(WorkspaceBackend, SupportsFilesystem):
    """What the three clouds share: a bucket, or a folder in one, read and written with obstore.

    Paths start at `/`, which is the folder in `url` and the working directory. Folders exist while they hold
    something; an empty folder made here holds a hidden `.pydantic-ai-dir` object. There are no symlinks.
    """

    _schemes: ClassVar[tuple[str, ...]]
    _provider: ClassVar[str]

    url: str
    """The bucket or container, and an optional folder in it."""

    _: KW_ONLY

    # Each cloud types its own keys.
    credentials: Any = field(default=None, compare=False, repr=False)

    endpoint: str | None = None
    """The storage service's URL, for a compatible service or an emulator."""

    def __post_init__(self) -> None:
        scheme = urlsplit(self.url).scheme
        if scheme not in self._schemes:
            schemes = ' or '.join(f'`{scheme}://`' for scheme in self._schemes)
            raise ValueError(f'{type(self).__name__} URLs start with {schemes}, got {self.url!r}')
        container, folder = self._location
        if not container or any(part in ('', '.', '..') for part in folder.split('/') if folder):
            raise ValueError(
                f'{type(self).__name__} takes a bucket and a plain folder path, without `//`, `.` or `..`, '
                f'got {self.url!r}'
            )
        if not callable(self.credentials) and self.credentials is not None:
            check_expiry(self.credentials)

    @cached_property
    def _location(self) -> tuple[str, str]:
        """The bucket or container, and the folder in it, decoded once for file tools and mounts alike."""
        url = urlsplit(self.url)
        return url.netloc, _folder(url.path)

    @property
    def ref(self) -> WorkspaceRef:
        return WorkspaceRef(provider=self._provider, id=self.url)

    async def working_dir(self) -> str:
        return '/'

    def _config(self) -> dict[str, str]:
        """Settings for obstore beyond the URL and keys."""
        return {} if self.endpoint is None else {'endpoint': self.endpoint}

    @cached_property
    def _store(self) -> ObjectStore:
        credentials = self.credentials
        provider = credentials if callable(credentials) or credentials is None else lambda: credentials
        container, folder = self._location
        return self._open(
            container,
            prefix=folder or None,
            config=cast(Any, self._config()),
            client_options={'allow_http': True} if (self.endpoint or '').startswith('http://') else None,
            credential_provider=cast(Any, provider),
        )

    # obstore types `config` and the provider per cloud; each subclass names its store.
    _open: ClassVar[Callable[..., ObjectStore]]

    async def read_bytes(self, path: str) -> bytes:
        key = _key(path)
        try:
            return await self._call(lambda store: obstore.get(store, key).bytes().to_bytes(), path)
        except FileNotFoundError:
            if key and await self._has_children(key, path):
                raise IsADirectoryError(path) from None
            raise

    async def write_bytes(self, path: str, data: bytes) -> None:
        key = _key(path)
        if not key or await self._has_children(key, path):
            raise IsADirectoryError(path)
        await self._check_parents(key, path)
        await self._call(lambda store: obstore.put(store, key, data), path)

    async def stat(self, path: str) -> FileEntry:
        key = _key(path)
        name = posixpath.basename(key)
        if not key:
            return FileEntry(name=name, path='/', is_dir=True, size=None)
        size = await self._size(key, path)
        if size is not None:
            return FileEntry(name=name, path=path, is_dir=False, size=size)
        if await self._has_children(key, path):
            return FileEntry(name=name, path=path, is_dir=True, size=None)
        raise FileNotFoundError(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        key = _key(path)
        listing = await self._call(
            lambda store: obstore.list_with_delimiter(store, key or None, return_arrow=False),  # pyright: ignore[reportUnknownMemberType]
            path,
        )
        folders = {prefix.rsplit('/', 1)[-1] for prefix in listing['common_prefixes']}
        files = {
            name: meta['size']
            for meta in listing['objects']
            # A folder's own marker object (`dir/`, as consoles write it) lists as a file named like the folder.
            if (name := meta['path'].rsplit('/', 1)[-1]) != _DIRECTORY_MARKER and name not in folders
        }
        if key and not files and not folders and not listing['objects']:
            if await self._size(key, path) is not None:
                raise NotADirectoryError(path)
            raise FileNotFoundError(path)
        entries = [FileEntry(name=name, path=posixpath.join(path, name), is_dir=True, size=None) for name in folders]
        entries += [
            FileEntry(name=name, path=posixpath.join(path, name), is_dir=False, size=size)
            for name, size in files.items()
        ]
        return sorted(entries, key=lambda entry: entry.name)

    async def make_dir(self, path: str) -> None:
        key = _key(path)
        if not key:
            return
        if await self._size(key, path) is not None:
            raise FileExistsError(path)
        await self._check_parents(key, path)
        if not await self._has_children(key, path):
            await self._call(lambda store: obstore.put(store, f'{key}/{_DIRECTORY_MARKER}', b''), path)

    async def remove(self, path: str) -> None:
        key = _key(path)
        if not key:
            raise ValueError('cannot remove the root of a bucket')
        if await self._size(key, path) is not None:
            await self._call(lambda store: obstore.delete(store, key), path)
            return
        keys = await self._call(
            lambda store: [meta['path'] for batch in _list(store, key) for meta in batch],
            path,
        )
        if not keys:
            raise FileNotFoundError(path)
        await self._call(lambda store: obstore.delete(store, keys), path)

    async def exists(self, path: str) -> bool:
        try:
            await self.stat(path)
        except (FileNotFoundError, NotADirectoryError):
            return False
        return True

    async def _size(self, key: str, path: str) -> int | None:
        """The size of the object at `key`, or `None` when there is none."""
        try:
            return (await self._call(lambda store: obstore.head(store, key), path))['size']
        except FileNotFoundError:
            return None

    async def _has_children(self, key: str, path: str) -> bool:
        return await self._call(lambda store: any(batch for batch in _list(store, key, chunk_size=1)), path)

    async def _check_parents(self, key: str, path: str) -> None:
        """Refuse a path whose parent is a file, as a filesystem would."""
        parents = [key[:index] for index, char in enumerate(key) if char == '/']
        files: list[str] = []

        async def check(parent: str) -> None:
            if await self._size(parent, path) is not None:
                files.append(parent)

        async with anyio.create_task_group() as task_group:
            for parent in parents:
                task_group.start_soon(check, parent)
        if files:
            raise NotADirectoryError(path)

    async def _call(self, operation: Callable[[ObjectStore], _T], path: str) -> _T:
        # obstore's blocking API runs in a worker thread, the same way on asyncio and Trio. A cancelled
        # caller waits for the request to finish rather than leaving it running unowned.
        with _translate_errors(self.url, path):
            return await anyio.to_thread.run_sync(partial(operation, self._store))


@dataclass(frozen=True)
class S3Filesystem(_ObjectStoreFilesystem):
    """An S3 bucket, or a folder in one, as a filesystem; also S3-compatible services such as R2 and MinIO.

    ```python {test="skip"}
    from pydantic_ai_harness.cloud_storage import S3Filesystem

    S3Filesystem('s3://acme-data/exports/')  # keys from the environment
    S3Filesystem('s3://acme-data', endpoint='https://<account>.r2.cloudflarestorage.com', region='auto')  # R2
    ```

    Paths start at `/`, which is the folder in `url`. An empty folder holds a hidden `.pydantic-ai-dir` object.
    """

    _schemes = ('s3',)
    _provider = 's3'
    _open = S3Store

    _: KW_ONLY

    credentials: S3Credential | Callable[[], S3Credential] | None = field(default=None, compare=False, repr=False)
    """Keys, or a function returning fresh ones before they expire; `None` finds them in the environment, as the
    cloud's own SDK would."""

    region: str | None = None
    """The bucket's region."""

    share_long_lived_credentials: bool = False
    """Whether keys that never expire may be handed to a sandbox for a [`Mount`][pydantic_ai.workspaces.Mount] with
    `shell=True`, where its commands can read them. Without it, only keys with an `expires_at` are handed on."""

    def _config(self) -> dict[str, str]:
        config = super()._config()
        if self.region is not None:
            config['region'] = self.region
        return config

    def _rclone_remote(self) -> tuple[dict[str, str], str]:
        """The settings rclone needs for this bucket, without keys, and the folder in its terms."""
        remote = {'type': 's3', 'provider': 'Other' if self.endpoint else 'AWS', 'env_auth': 'false'}
        if self.endpoint:
            remote['endpoint'] = self.endpoint
        if self.region:
            remote['region'] = self.region
        return remote, '/'.join(part for part in self._location if part)

    def _rclone_keys(self, keys: Mapping[str, Any]) -> dict[str, str]:
        """The keys in rclone's terms."""
        if not {'access_key_id', 'secret_access_key'} <= keys.keys():
            raise ValueError('S3 keys need `access_key_id` and `secret_access_key`.')
        return {
            'access_key_id': keys['access_key_id'],
            'secret_access_key': keys['secret_access_key'],
            'session_token': keys.get('token') or '',
        }


@dataclass(frozen=True)
class GCSFilesystem(_ObjectStoreFilesystem):
    """A Google Cloud Storage bucket, or a folder in one, as a filesystem.

    ```python {test="skip"}
    from pydantic_ai_harness.cloud_storage import GCSFilesystem

    GCSFilesystem('gs://acme-data/exports/')  # keys from the environment
    ```

    Paths start at `/`, which is the folder in `url`. An empty folder holds a hidden `.pydantic-ai-dir` object.
    Commands can't see it: a [`Mount`][pydantic_ai.workspaces.Mount] of it with `commands=True` is refused.
    """

    _schemes = ('gs',)
    _provider = 'gcs'
    _open = GCSStore

    _: KW_ONLY

    credentials: GCSCredential | Callable[[], GCSCredential] | None = field(default=None, compare=False, repr=False)
    """Keys, or a function returning fresh ones before they expire; `None` finds them in the environment, as the
    cloud's own SDK would."""

    def _config(self) -> dict[str, str]:
        return {} if self.endpoint is None else {'base_url': self.endpoint}


@dataclass(frozen=True)
class AzureFilesystem(_ObjectStoreFilesystem):
    """An Azure Blob Storage container, or a folder in one, as a filesystem.

    ```python {test="skip"}
    from pydantic_ai_harness.cloud_storage import AzureFilesystem

    AzureFilesystem('https://<account>.blob.core.windows.net/<container>/exports/')  # keys from the environment
    AzureFilesystem('az://<container>/exports/', account='<account>')
    ```

    Paths start at `/`, which is the folder in `url`. An empty folder holds a hidden `.pydantic-ai-dir` object.
    """

    _schemes = ('az', 'https')
    _provider = 'azure'
    _open = AzureStore

    _: KW_ONLY

    credentials: AzureCredential | Callable[[], AzureCredential] | None = field(default=None, compare=False, repr=False)
    """Keys, or a function returning fresh ones before they expire; `None` finds them in the environment, as the
    cloud's own SDK would."""

    account: str | None = None
    """The storage account, for an `az://` URL; an `https://` URL names it."""

    share_long_lived_credentials: bool = False
    """Whether keys that never expire may be handed to a sandbox for a [`Mount`][pydantic_ai.workspaces.Mount] with
    `shell=True`, where its commands can read them. Without it, only keys with an `expires_at` are handed on."""

    def __post_init__(self) -> None:
        super().__post_init__()
        url = urlsplit(self.url)
        if url.scheme == 'https' and not url.netloc.endswith('.blob.core.windows.net'):
            raise ValueError(
                f'AzureFilesystem takes an `https://<account>.blob.core.windows.net/` URL, got {self.url!r}'
            )
        if url.scheme == 'az' and self.account is None:
            raise ValueError('AzureFilesystem needs `account=` with an `az://` URL')

    @cached_property
    def _location(self) -> tuple[str, str]:
        url = urlsplit(self.url)
        if url.scheme == 'az':
            return url.netloc, _folder(url.path)
        # `https://<account>.blob.core.windows.net/<container>/<folder>`
        container, _, folder = _folder(url.path).partition('/')
        return container, folder

    @property
    def _account(self) -> str:
        return self.account or urlsplit(self.url).netloc.split('.', 1)[0]

    def _config(self) -> dict[str, str]:
        config = super()._config()
        config['account_name'] = self._account
        return config

    def _rclone_remote(self) -> tuple[dict[str, str], str]:
        """The settings rclone needs for this container, without keys, and the folder in its terms."""
        remote = {'type': 'azureblob', 'account': self._account}
        if self.endpoint:
            remote['endpoint'] = self.endpoint
        return remote, '/'.join(part for part in self._location if part)

    def _rclone_keys(self, keys: Mapping[str, Any]) -> dict[str, str]:
        """The keys in rclone's terms: an account key, or a SAS token for the container."""
        if 'access_key' in keys:
            return {'key': keys['access_key']}
        if 'sas_token' in keys:
            token = keys['sas_token']
            query = token.lstrip('?') if isinstance(token, str) else urlencode(token)
            base = self.endpoint or f'https://{self._account}.blob.core.windows.net'
            return {'sas_url': f'{base.rstrip("/")}/{self._location[0]}?{query}'}
        raise ValueError('Azure keys for commands need an `access_key` or a `sas_token`.')


def _folder(path: str) -> str:
    """A URL's path without its leading and trailing `/`, decoded; a doubled `/` is kept, to be refused."""
    path = unquote(path)
    return (path[1:] if path.startswith('/') else path).rstrip('/')


def check_expiry(keys: Mapping[str, Any]) -> None:
    """Refuse an `expires_at` obstore can't compare: it must be a datetime with a time zone, or `None`."""
    expires_at = keys.get('expires_at')
    if expires_at is not None and (not isinstance(expires_at, datetime) or expires_at.tzinfo is None):
        raise ValueError(f'`expires_at` must be a datetime with a time zone, such as UTC, got {expires_at!r}')


def _list(store: ObjectStore, prefix: str, chunk_size: int = 1000) -> obstore.ListStream[Sequence[obstore.ObjectMeta]]:
    # obstore's Arrow overloads name a type its stubs leave unknown.
    return obstore.list(store, prefix, chunk_size=chunk_size, return_arrow=False)  # pyright: ignore[reportUnknownMemberType]


def _key(path: str) -> str:
    if not posixpath.isabs(path):
        raise ValueError(f'bucket paths are absolute, got {path!r}')
    return posixpath.normpath(path).lstrip('/')


@contextmanager
def _translate_errors(url: str, path: str) -> Generator[None]:
    try:
        yield
    except (PermissionDeniedError, UnauthenticatedError) as error:
        raise PermissionError(f'access to {path!r} in {url} was denied: {error}') from error
    except GenericError as error:
        # S3 and Azure name a missing bucket or container in the error; obstore has no type for it.
        if 'NoSuchBucket' in str(error) or 'ContainerNotFound' in str(error):
            raise WorkspaceUnavailableError(f'the bucket in {url} does not exist') from error
        raise
