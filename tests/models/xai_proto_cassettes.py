"""Proto cassette utilities for xAI SDK (gRPC) tests.

Why this exists:
- `pytest-recording`/VCR only records HTTP. The xAI SDK uses gRPC, so VCR can't record/replay model calls.
- However, xAI responses are protobuf messages. We can serialize them and store them in YAML cassettes.

This is intentionally minimal for now:
- supports `chat.create(...).sample()` (non-streaming) responses
- supports `chat.create(...).stream()` by recording only `chunk.proto` bytes and reconstructing the aggregated
  `Response` via `Response.process_chunk()` during replay
- supports `files.upload(...)` with deterministic IDs for tests that pass `DocumentUrl`
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from ..conftest import try_import

with try_import() as imports_successful:
    import xai_sdk.chat as chat_types
    import yaml
    from google.protobuf.json_format import MessageToDict
    from google.protobuf.message import Message
    from xai_sdk import AsyncClient
    from xai_sdk.proto import chat_pb2


def _serialize_for_cassette(o: Any) -> Any:
    """Best-effort conversion of protobuf messages (and nested structures) to JSON-like dicts."""
    if isinstance(o, dict):
        return {str(k): _serialize_for_cassette(v) for k, v in cast(dict[str, Any], o).items()}
    if isinstance(o, list | tuple):
        return [_serialize_for_cassette(v) for v in cast(list[Any] | tuple[Any, ...], o)]
    if isinstance(o, Message):
        return MessageToDict(o, preserving_proto_field_name=True)
    return o


class XaiAsyncClientLike(Protocol):
    """A minimal protocol matching what `pydantic_ai` needs from an xAI client.

    We can't reliably type this as `xai_sdk.AsyncClient` because cassette replay uses a duck-typed client.
    """

    @property
    def chat(self) -> Any: ...

    @property
    def files(self) -> Any: ...


ProtoCassetteRecordMode = Literal['none', 'once', 'new_episodes', 'rewrite', 'all']


def _truthy_env(name: str) -> bool:
    v = __import__('os').getenv(name, '')
    return v.lower() in {'1', 'true', 'yes'}


def _normalize_record_mode(mode: str | None) -> ProtoCassetteRecordMode | None:
    """Normalize pytest-recording/VCR-ish record modes to a small supported set.

    Notes:
    - VCR uses: `none`, `once`, `new_episodes`, `all`
    - This repo frequently uses `rewrite` as a synonym for "overwrite cassette".
    """
    if mode is None:
        return None
    m = mode.strip().lower()
    if m in {'none', 'once', 'new_episodes', 'rewrite', 'all'}:
        return cast(ProtoCassetteRecordMode, m)
    raise ValueError(f'Unknown record mode: {mode!r}')


def _proto_cassette_plan(
    *,
    cassette_exists: bool,
    record_mode: str | None,
    env_record_flag: bool,
) -> Literal['replay', 'record', 'hybrid', 'error_missing']:
    """Decide replay vs record behavior for proto cassettes.

    This is intentionally pure/side-effect-free so it can be unit tested without xai-sdk.
    """
    normalized = _normalize_record_mode(record_mode)

    # Back-compat: previous behavior was a simple boolean env flag which meant "rewrite".
    if normalized is None and env_record_flag:
        normalized = 'rewrite'

    # Default behavior (when neither pytest flag nor env var is set) is "replay only".
    if normalized is None:
        normalized = 'none'

    if normalized == 'none':
        return 'replay' if cassette_exists else 'error_missing'
    if normalized == 'once':
        return 'replay' if cassette_exists else 'record'
    if normalized == 'new_episodes':
        return 'hybrid' if cassette_exists else 'record'
    if normalized in {'rewrite', 'all'}:
        return 'record'

    # This should be unreachable since `_normalize_record_mode` validates inputs.
    raise AssertionError(f'Unhandled record mode: {normalized!r}')  # pragma: no cover


@dataclass
class XaiSampleProtoCassette:
    """A simple cassette for a sequence of `chat.sample()` responses."""

    responses: list[bytes]
    # Recorded protobuf requests for `chat.create(...).sample()` calls (lossless).
    # NOTE: this is a serialized `GetCompletionsRequest` from the xAI SDK chat object.
    sample_requests: list[bytes] = field(default_factory=list)
    # Optional debug representation of the request kwargs, only recorded when enabled.
    sample_requests_json: list[dict[str, Any]] = field(default_factory=list)
    # Optional debug representation (only recorded when enabled), ignored by replay.
    responses_json: list[dict[str, Any]] = field(default_factory=list)
    # Streaming format: store only chunk protos and reconstruct response via `Response.process_chunk`.
    # Each entry corresponds to one `chat.stream()` call and contains the ordered chunks as serialized bytes.
    stream_chunks: list[list[bytes]] = field(default_factory=list)
    # Recorded protobuf requests for `chat.create(...).stream()` calls (lossless).
    stream_requests: list[bytes] = field(default_factory=list)
    # Optional debug representation of the request kwargs, only recorded when enabled.
    stream_requests_json: list[dict[str, Any]] = field(default_factory=list)
    # Optional debug representation (only recorded when enabled), ignored by replay.
    stream_chunks_json: list[list[dict[str, Any]]] = field(default_factory=list)
    version: int = 1

    @classmethod
    def load(cls, path: Path) -> XaiSampleProtoCassette:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        version = int(data.get('version', 1))
        responses = cast(list[bytes], data.get('responses', []))
        sample_requests = cast(list[bytes], data.get('sample_requests', []))
        sample_requests_json = cast(list[dict[str, Any]], data.get('sample_requests_json', []))
        responses_json = cast(list[dict[str, Any]], data.get('responses_json', []))
        stream_chunks = cast(list[list[bytes]], data.get('stream_chunks', []))
        stream_requests = cast(list[bytes], data.get('stream_requests', []))
        stream_requests_json = cast(list[dict[str, Any]], data.get('stream_requests_json', []))
        stream_chunks_json = cast(list[list[dict[str, Any]]], data.get('stream_chunks_json', []))
        return cls(
            responses=responses,
            sample_requests=sample_requests,
            sample_requests_json=sample_requests_json,
            responses_json=responses_json,
            stream_chunks=stream_chunks,
            stream_requests=stream_requests,
            stream_requests_json=stream_requests_json,
            stream_chunks_json=stream_chunks_json,
            version=version,
        )

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            'version': self.version,
        }
        # Only include non-streaming responses if present; most cassettes are streaming-only.
        if self.responses:
            data['responses'] = self.responses
        if self.sample_requests:
            data['sample_requests'] = self.sample_requests
        if self.sample_requests_json:
            data['sample_requests_json'] = self.sample_requests_json
        # Streaming is the default/primary format.
        if self.stream_chunks:
            data['stream_chunks'] = self.stream_chunks
        if self.stream_requests:
            data['stream_requests'] = self.stream_requests
        if self.stream_requests_json:
            data['stream_requests_json'] = self.stream_requests_json
        # Only include debug JSON fields if present to avoid bloating committed cassettes by default.
        if self.responses_json:
            data['responses_json'] = self.responses_json
        if self.stream_chunks_json:
            data['stream_chunks_json'] = self.stream_chunks_json
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


@dataclass
class _CassetteChatInstance:
    _client: XaiProtoCassetteClient

    async def sample(self) -> chat_types.Response:
        try:
            proto_bytes = self._client.cassette.responses[self._client.sample_idx]
        except IndexError as e:  # pragma: no cover
            raise IndexError(f'Cassette exhausted at index {self._client.sample_idx}') from e
        self._client.sample_idx += 1

        proto = chat_pb2.GetChatCompletionResponse()
        proto.ParseFromString(proto_bytes)
        return chat_types.Response(proto, index=None)

    def stream(self) -> Any:
        async def _aiter():
            if not self._client.cassette.stream_chunks:
                raise RuntimeError(
                    'xAI proto cassette is missing streaming data (`stream_chunks`).\n'
                    'Re-record this cassette with:\n'
                    '  XAI_API_KEY=... uv run pytest --record-mode=rewrite <test> -v'
                )
            try:
                chunk_list = self._client.cassette.stream_chunks[self._client.stream_idx]
            except IndexError as e:  # pragma: no cover
                raise IndexError(f'Cassette stream exhausted at index {self._client.stream_idx}') from e
            self._client.stream_idx += 1

            # Reconstruct the aggregated response by applying each chunk, mirroring the SDK behavior.
            aggregated = chat_types.Response(chat_pb2.GetChatCompletionResponse(), index=None)
            for chunk_bytes in chunk_list:
                chunk_proto = chat_pb2.GetChatCompletionChunk()
                chunk_proto.ParseFromString(chunk_bytes)
                aggregated.process_chunk(chunk_proto)
                yield aggregated, chat_types.Chunk(chunk_proto, index=None)

        return _aiter()


@dataclass
class XaiProtoCassetteClient:
    """Drop-in-ish xAI SDK client for replaying recorded protobuf responses."""

    cassette: XaiSampleProtoCassette
    # Important: these indices must be shared across `chat.create(...).sample()` calls,
    # otherwise multi-step agent runs will keep replaying the first response and can
    # get stuck in tool-call loops until `UsageLimits` fails (e.g. request_limit=50).
    sample_idx: int = 0
    stream_idx: int = 0

    @classmethod
    def from_path(cls, path: Path) -> XaiProtoCassetteClient:
        return cls(cassette=XaiSampleProtoCassette.load(path))

    @property
    def chat(self) -> Any:
        # We don't need to validate kwargs yet, but we keep the signature compatible.
        return type('Chat', (), {'create': self._chat_create})

    @property
    def files(self) -> Any:
        return type('Files', (), {'upload': self._files_upload})

    def _chat_create(self, *_args: Any, **_kwargs: Any) -> _CassetteChatInstance:
        return _CassetteChatInstance(self)

    async def _files_upload(self, data: bytes, filename: str) -> Any:
        # Deterministic ID; good enough for replay since we don't actually call the backend.
        # Keeping similar shape to the real SDK return value.
        file_id = f'file-{abs(hash((len(data), filename))) % 1_000_000:06d}'
        return type('UploadedFile', (), {'id': file_id})()


@dataclass
class XaiProtoCassetteHybridClient:
    """Replay from an existing cassette but record "new episodes" when the cassette runs out."""

    _inner: AsyncClient
    cassette: XaiSampleProtoCassette
    include_debug_json: bool = False
    sample_idx: int = 0
    stream_idx: int = 0
    dirty: bool = False

    @property
    def chat(self) -> Any:
        return type('Chat', (), {'create': self._chat_create})

    @property
    def files(self) -> Any:
        return type('Files', (), {'upload': self._inner.files.upload})

    def _chat_create(self, *args: Any, **kwargs: Any) -> Any:
        inner_chat = self._inner.chat.create(*args, **kwargs)
        include_debug_json = self.include_debug_json

        request_json: dict[str, Any] | None = None
        if include_debug_json:
            request_json = {
                'args': _serialize_for_cassette(list(args)),
                'kwargs': _serialize_for_cassette(kwargs),
            }

        client = self

        class _HybridChatInstance:
            async def sample(self) -> chat_types.Response:
                # Replay if we have a recorded response at this index.
                if client.sample_idx < len(client.cassette.responses):
                    proto_bytes = client.cassette.responses[client.sample_idx]
                    client.sample_idx += 1
                    proto = chat_pb2.GetChatCompletionResponse()
                    proto.ParseFromString(proto_bytes)
                    return chat_types.Response(proto, index=None)

                # Otherwise record a new episode.
                client.cassette.sample_requests.append(inner_chat.proto.SerializeToString())
                if request_json is not None:
                    client.cassette.sample_requests_json.append(request_json)
                response = await inner_chat.sample()
                client.cassette.responses.append(response.proto.SerializeToString())
                if include_debug_json:
                    client.cassette.responses_json.append(
                        MessageToDict(response.proto, preserving_proto_field_name=True)
                    )
                client.dirty = True
                client.sample_idx += 1
                return response

            def stream(self) -> Any:
                async def _aiter():
                    # Replay if we have recorded chunks at this index.
                    if client.stream_idx < len(client.cassette.stream_chunks):
                        chunk_list = client.cassette.stream_chunks[client.stream_idx]
                        client.stream_idx += 1

                        aggregated = chat_types.Response(chat_pb2.GetChatCompletionResponse(), index=None)
                        for chunk_bytes in chunk_list:
                            chunk_proto = chat_pb2.GetChatCompletionChunk()
                            chunk_proto.ParseFromString(chunk_bytes)
                            aggregated.process_chunk(chunk_proto)
                            yield aggregated, chat_types.Chunk(chunk_proto, index=None)
                        return

                    # Otherwise record a new streaming episode.
                    chunks: list[bytes] = []
                    chunks_json: list[dict[str, Any]] = []
                    client.cassette.stream_requests.append(inner_chat.proto.SerializeToString())
                    if request_json is not None:
                        client.cassette.stream_requests_json.append(request_json)
                    try:
                        async for response, chunk in inner_chat.stream():
                            chunks.append(chunk.proto.SerializeToString())
                            if include_debug_json:
                                chunks_json.append(
                                    {
                                        'chunk': MessageToDict(
                                            chunk.proto,
                                            preserving_proto_field_name=True,
                                        )
                                    }
                                )
                            yield response, chunk
                    finally:
                        client.cassette.stream_chunks.append(chunks)
                        if include_debug_json:
                            client.cassette.stream_chunks_json.append(chunks_json)
                        client.dirty = True
                        client.stream_idx += 1

                return _aiter()

        return _HybridChatInstance()


@dataclass
class XaiProtoRecorder:
    """Record `chat.sample()` responses as protobuf bytes.

    Usage:
        recorder = XaiProtoRecorder(real_client)
        ... run agent using recorder.client ...
        recorder.dump(path)
    """

    _inner: AsyncClient
    cassette: XaiSampleProtoCassette = field(default_factory=lambda: XaiSampleProtoCassette(responses=[]))
    include_debug_json: bool = False

    @property
    def client(self) -> Any:
        return self

    @property
    def chat(self) -> Any:
        return type('Chat', (), {'create': self._chat_create})

    @property
    def files(self) -> Any:
        return type('Files', (), {'upload': self._inner.files.upload})

    def dump(self, path: Path) -> None:
        self.cassette.dump(path)

    def _chat_create(self, *args: Any, **kwargs: Any) -> Any:
        inner_chat = self._inner.chat.create(*args, **kwargs)
        recorder = self
        include_debug_json = recorder.include_debug_json

        request_json: dict[str, Any] | None = None
        if include_debug_json:
            request_json = {
                'args': _serialize_for_cassette(list(args)),
                'kwargs': _serialize_for_cassette(kwargs),
            }

        class _RecorderChatInstance:
            async def sample(self) -> chat_types.Response:
                # Always record the request proto (lossless); only record JSON when debug is enabled.
                recorder.cassette.sample_requests.append(inner_chat.proto.SerializeToString())
                if request_json is not None:
                    recorder.cassette.sample_requests_json.append(request_json)
                response = await inner_chat.sample()
                recorder.cassette.responses.append(response.proto.SerializeToString())
                if include_debug_json:
                    recorder.cassette.responses_json.append(
                        MessageToDict(response.proto, preserving_proto_field_name=True)
                    )
                return response

            def stream(self) -> Any:
                async def _aiter():
                    chunks: list[bytes] = []
                    chunks_json: list[dict[str, Any]] = []
                    # Always record the request proto (lossless); only record JSON when debug is enabled.
                    recorder.cassette.stream_requests.append(inner_chat.proto.SerializeToString())
                    if request_json is not None:
                        recorder.cassette.stream_requests_json.append(request_json)
                    try:
                        async for response, chunk in inner_chat.stream():
                            chunks.append(chunk.proto.SerializeToString())
                            if include_debug_json:
                                chunks_json.append(
                                    {
                                        'chunk': MessageToDict(
                                            chunk.proto,
                                            preserving_proto_field_name=True,
                                        )
                                    }
                                )
                            yield response, chunk
                    finally:
                        # Ensure data is persisted even if the consumer stops early and closes the generator.
                        recorder.cassette.stream_chunks.append(chunks)
                        if include_debug_json:
                            recorder.cassette.stream_chunks_json.append(chunks_json)

                return _aiter()

        return _RecorderChatInstance()


@dataclass
class XaiProtoCassetteSession:
    """A session that provides an xAI client and optionally records to a cassette."""

    client: XaiAsyncClientLike
    cassette_path: Path
    cassette: XaiSampleProtoCassette | None = None
    dirty_check: Any | None = None

    def dump_if_recording(self) -> None:
        if self.cassette is None:
            return
        if self.dirty_check is None or bool(self.dirty_check()):
            self.cassette.dump(self.cassette_path)


def xai_proto_cassette_session(
    cassette_path: Path,
    record_mode: str | None = None,
    include_debug_json: bool = False,
) -> XaiProtoCassetteSession:
    """Create a cassette session (replay if cassette exists, otherwise record if enabled).

    Env vars:
    - `XAI_API_KEY`: required in record mode.
    - `XAI_BASE_URL`: optional; passed to `AsyncClient` if supported (useful for gateways/proxies).
    """

    if not xai_sdk_available():  # pragma: no cover
        raise RuntimeError('xai-sdk is not installed')

    plan = _proto_cassette_plan(
        cassette_exists=cassette_path.exists(),
        record_mode=record_mode,
        env_record_flag=_truthy_env('XAI_PROTO_CASSETTE_RECORD'),
    )
    if plan == 'replay':
        cassette = XaiSampleProtoCassette.load(cassette_path)
        return XaiProtoCassetteSession(
            client=cast(XaiAsyncClientLike, XaiProtoCassetteClient(cassette=cassette)),
            cassette_path=cassette_path,
            cassette=None,
        )

    if plan == 'error_missing':
        raise RuntimeError(
            'Missing xAI proto cassette.\n'
            f'Expected: {cassette_path}\n\n'
            'To record it (requires xai-sdk + network + creds):\n\n'
            'Example:\n'
            '  XAI_API_KEY=... [XAI_BASE_URL=...] uv run pytest --record-mode=rewrite <test> -v'
        )

    os = __import__('os')
    base_url = os.getenv('XAI_BASE_URL')
    try:
        api_key = os.environ['XAI_API_KEY']
    except KeyError as e:  # pragma: no cover
        raise RuntimeError('Set `XAI_API_KEY` to record xAI proto cassettes.') from e

    # Best-effort support for SDK variants with/without `base_url`.
    try:
        real_client = AsyncClient(api_key=api_key, base_url=base_url) if base_url else AsyncClient(api_key=api_key)  # type: ignore[call-arg]
    except TypeError:
        real_client = AsyncClient(api_key=api_key)

    if plan == 'hybrid':
        cassette = XaiSampleProtoCassette.load(cassette_path)
        hybrid = XaiProtoCassetteHybridClient(real_client, cassette=cassette, include_debug_json=include_debug_json)
        return XaiProtoCassetteSession(
            client=cast(XaiAsyncClientLike, hybrid),
            cassette_path=cassette_path,
            cassette=cassette,
            dirty_check=lambda: hybrid.dirty,
        )
    else:
        # plan == 'record'
        recorder = XaiProtoRecorder(real_client, include_debug_json=include_debug_json)
        return XaiProtoCassetteSession(
            client=cast(XaiAsyncClientLike, recorder.client),
            cassette_path=cassette_path,
            cassette=recorder.cassette,
        )


def xai_sdk_available() -> bool:
    return imports_successful()
