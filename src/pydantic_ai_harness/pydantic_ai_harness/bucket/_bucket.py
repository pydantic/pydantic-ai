"""A cloud storage bucket (S3, Google Cloud Storage, Azure Blob) as workspace storage."""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

import anyio
import anyio.to_thread

try:
    import obstore
    from obstore.exceptions import GenericError, PermissionDeniedError, UnauthenticatedError
    from obstore.store import from_url
except ImportError as _import_error:
    raise ImportError('Install `pydantic-ai-harness[bucket]` to use Bucket.') from _import_error

from pydantic_ai.workspaces import FileEntry, SupportsFilesystem, WorkspaceUnavailableError

if TYPE_CHECKING:
    from obstore.store import AzureCredential, GCSCredential, ObjectStore, S3Credential

__all__ = ('Bucket',)

_Credentials: TypeAlias = 'S3Credential | GCSCredential | AzureCredential'
"""Keys for the bucket's cloud, in obstore's shape."""

_T = TypeVar('_T')

_DIRECTORY_MARKER = '.pydantic-ai-dir'
"""Object that keeps an empty folder: object stores only have folders that hold something."""


@dataclass(frozen=True)
class Bucket(SupportsFilesystem):
    """A bucket, or a folder in one, as storage for a [`Mount`][pydantic_ai.workspaces.Mount].

    ```python {test="skip"}
    from pydantic_ai_harness.bucket import Bucket

    Bucket('s3://acme-data/exports/')  # AWS S3, keys from the environment
    Bucket('s3://acme-data', endpoint='https://<account>.r2.cloudflarestorage.com', region='auto')  # R2
    Bucket('gs://acme-data/exports/')  # Google Cloud Storage
    Bucket('https://<account>.blob.core.windows.net/<container>/exports/')  # Azure Blob Storage
    ```

    Paths start at `/`, which is the folder in `url`. Folders exist while they hold something; an empty folder
    made through this storage holds a hidden `.pydantic-ai-dir` object. There are no symlinks.
    """

    url: str
    """The bucket and an optional folder in it: `s3://`, `gs://`, `az://` or an Azure `https://` URL."""

    _: KW_ONLY

    credentials: _Credentials | Callable[[], _Credentials] | None = field(default=None, compare=False, repr=False)
    """Keys, or a function returning fresh ones before they expire; `None` finds them in the environment,
    as the cloud's own SDK would."""

    endpoint: str | None = None
    """The storage service's URL, for S3-compatible services (R2, MinIO) or an emulator."""

    region: str | None = None
    """The bucket's region, where the cloud needs one."""

    # Left out of the hash, which a dict can't take; it still counts for equality.
    options: Mapping[str, str] = field(default_factory=dict[str, str], hash=False)
    """Further [obstore store settings](https://developmentseed.org/obstore/latest/api/store/), such as
    `{'account_name': ...}` for Azure with `az://` URLs."""

    @cached_property
    def _store(self) -> ObjectStore:
        config: dict[str, Any] = dict(self.options)
        if self.endpoint is not None:
            config['base_url' if self.url.startswith('gs://') else 'endpoint'] = self.endpoint
        if self.region is not None:
            config['region'] = self.region
        credentials = self.credentials
        provider = credentials if callable(credentials) or credentials is None else lambda: credentials
        # One call for every cloud: obstore's overloads type `config` and the provider per cloud.
        return from_url(
            self.url,
            config=cast(Any, config),
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
