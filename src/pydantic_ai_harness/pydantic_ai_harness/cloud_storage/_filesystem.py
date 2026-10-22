"""S3, Google Cloud Storage and Azure Blob Storage as filesystems: workspace backends that store files."""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast
from urllib.parse import urlsplit

import anyio
import anyio.to_thread

try:
    import obstore
    from obstore.exceptions import GenericError, PermissionDeniedError, UnauthenticatedError
    from obstore.store import from_url
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
        # One call for every cloud: obstore's overloads type `config` and the provider per cloud.
        return from_url(
            self.url,
            config=cast(Any, self._config()),
            client_options={'allow_http': True} if (self.endpoint or '').startswith('http://') else None,
            credential_provider=cast(Any, provider),
        )

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


@dataclass(frozen=True)
class GCSFilesystem(_ObjectStoreFilesystem):
    """A Google Cloud Storage bucket, or a folder in one, as a filesystem.

    ```python {test="skip"}
    from pydantic_ai_harness.cloud_storage import GCSFilesystem

    GCSFilesystem('gs://acme-data/exports/')  # keys from the environment
    ```

    Paths start at `/`, which is the folder in `url`. An empty folder holds a hidden `.pydantic-ai-dir` object.
    Shell commands can't see it: a [`Mount`][pydantic_ai.workspaces.Mount] of it with `shell=True` is refused.
    """

    _schemes = ('gs',)
    _provider = 'gcs'

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

    def _config(self) -> dict[str, str]:
        config = super()._config()
        if self.account is not None:
            config['account_name'] = self.account
        return config


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
