"""Proto cassette utilities for xAI SDK (gRPC) tests.

Why this exists:
- `pytest-recording`/VCR only records HTTP. The xAI SDK uses gRPC, so VCR can't record/replay model calls.
- However, xAI responses are protobuf messages. We can serialize them and store them in YAML cassettes.

This is intentionally minimal for now:
- supports `chat.create(...).sample()` (non-streaming) responses
- supports `chat.create(...).stream()` by recording only `chunk.proto` bytes and reconstructing the aggregated
  `Response` via `Response.process_chunk()` during replay
- supports `files.upload(...)` with deterministic IDs for tests that pass `DocumentUrl`

Cassette Format:
    The cassette stores an ordered list of request/response interactions for human readability.
    Each interaction pairs a request with its response, using `_sample` or `_stream` suffixes
    to align with the SDK methods (`chat.sample()` and `chat.stream()`).

    version: 1
    interactions:
    - request_sample:
        json: {...}    # Human-readable request (optional, for debugging)
        raw: !!binary  # Protobuf bytes (lossless)
      response_sample:
        json: {...}    # Human-readable response (optional, for debugging)
        raw: !!binary  # Protobuf bytes (lossless)
    - request_stream:
        json: {...}
        raw: !!binary
      response_stream:
        chunks_json: [{...}, ...]  # Human-readable chunks (optional)
        chunks_raw: [!!binary, ...]  # Protobuf chunk bytes (lossless)
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
    from xai_sdk import AsyncClient
    from xai_sdk.proto import chat_pb2


# ---------------------------------------------------------------------------
# Interaction dataclasses for v2 cassette format
# ---------------------------------------------------------------------------


@dataclass
class SampleInteraction:
    """A single `chat.sample()` request/response pair."""

    request_raw: bytes
    response_raw: bytes
    request_json: dict[str, Any] | None = None
    response_json: dict[str, Any] | None = None


@dataclass
class StreamInteraction:
    """A single `chat.stream()` request/response pair."""

    request_raw: bytes
    chunks_raw: list[bytes]
    request_json: dict[str, Any] | None = None
    chunks_json: list[dict[str, Any]] | None = None


# Union type for interactions (used for type hints in the ordered list)
Interaction = SampleInteraction | StreamInteraction


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
class XaiProtoCassette:
    """Cassette storing an ordered list of request/response interactions.

    Each interaction pairs a request with its response, using `SampleInteraction`
    for `chat.sample()` calls and `StreamInteraction` for `chat.stream()` calls.
    """

    interactions: list[Interaction] = field(default_factory=list[Interaction])
    version: int = 1

    @classmethod
    def load(cls, path: Path) -> XaiProtoCassette:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))

        interactions: list[Interaction] = []
        for item in data.get('interactions', []):
            if 'request_sample' in item:
                req = item['request_sample']
                resp = item['response_sample']
                interactions.append(
                    SampleInteraction(
                        request_raw=req['raw'],
                        response_raw=resp['raw'],
                        request_json=req.get('json'),
                        response_json=resp.get('json'),
                    )
                )
            elif 'request_stream' in item:
                req = item['request_stream']
                resp = item['response_stream']
                interactions.append(
                    StreamInteraction(
                        request_raw=req['raw'],
                        chunks_raw=resp['chunks_raw'],
                        request_json=req.get('json'),
                        chunks_json=resp.get('chunks_json'),
                    )
                )
        return cls(interactions=interactions)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        interactions_data: list[dict[str, Any]] = []

        for interaction in self.interactions:
            if isinstance(interaction, SampleInteraction):
                # Build request_sample: json first (if present), then raw
                req: dict[str, Any] = {}
                if interaction.request_json:
                    req['json'] = interaction.request_json
                req['raw'] = interaction.request_raw

                # Build response_sample: json first (if present), then raw
                resp: dict[str, Any] = {}
                if interaction.response_json:
                    resp['json'] = interaction.response_json
                resp['raw'] = interaction.response_raw

                interactions_data.append(
                    {
                        'request_sample': req,
                        'response_sample': resp,
                    }
                )

            elif isinstance(interaction, StreamInteraction):
                # Build request_stream: json first (if present), then raw
                req = {}
                if interaction.request_json:
                    req['json'] = interaction.request_json
                req['raw'] = interaction.request_raw

                # Build response_stream: chunks_json first (if present), then chunks_raw
                resp = {}
                if interaction.chunks_json:
                    resp['chunks_json'] = interaction.chunks_json
                resp['chunks_raw'] = interaction.chunks_raw

                interactions_data.append(
                    {
                        'request_stream': req,
                        'response_stream': resp,
                    }
                )

        data: dict[str, Any] = {
            'version': self.version,
            'interactions': interactions_data,
        }
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding='utf-8',
        )


# Backwards compatibility alias
XaiSampleProtoCassette = XaiProtoCassette


@dataclass
class _CassetteChatInstance:
    _client: XaiProtoCassetteClient
    _expected_type: Literal['sample', 'stream']

    async def sample(self) -> chat_types.Response:
        if self._expected_type != 'sample':
            raise RuntimeError(
                f'Cassette expects a stream() call at interaction {self._client.interaction_idx}, '
                f'but sample() was called.'
            )
        interaction = self._client.next_interaction()
        if not isinstance(interaction, SampleInteraction):  # pragma: no cover
            raise RuntimeError(f'Expected SampleInteraction, got {type(interaction).__name__}')

        proto = chat_pb2.GetChatCompletionResponse()
        proto.ParseFromString(interaction.response_raw)
        return chat_types.Response(proto, index=None)

    def stream(self) -> Any:
        if self._expected_type != 'stream':
            raise RuntimeError(
                f'Cassette expects a sample() call at interaction {self._client.interaction_idx}, '
                f'but stream() was called.'
            )
        interaction = self._client.next_interaction()

        async def _aiter():
            if not isinstance(interaction, StreamInteraction):  # pragma: no cover
                raise RuntimeError(f'Expected StreamInteraction, got {type(interaction).__name__}')

            # Reconstruct the aggregated response by applying each chunk, mirroring the SDK behavior.
            aggregated = chat_types.Response(chat_pb2.GetChatCompletionResponse(), index=None)
            for chunk_bytes in interaction.chunks_raw:
                chunk_proto = chat_pb2.GetChatCompletionChunk()
                chunk_proto.ParseFromString(chunk_bytes)
                aggregated.process_chunk(chunk_proto)
                yield aggregated, chat_types.Chunk(chunk_proto, index=None)

        return _aiter()


@dataclass
class XaiProtoCassetteClient:
    """Drop-in-ish xAI SDK client for replaying recorded protobuf responses."""

    cassette: XaiProtoCassette
    # Index into the ordered interactions list.
    interaction_idx: int = 0

    @classmethod
    def from_path(cls, path: Path) -> XaiProtoCassetteClient:
        return cls(cassette=XaiProtoCassette.load(path))

    def next_interaction(self) -> Interaction:
        if self.interaction_idx >= len(self.cassette.interactions):
            raise IndexError(
                f'Cassette exhausted at interaction {self.interaction_idx}.\n'
                'Re-record this cassette with:\n'
                '  XAI_API_KEY=... uv run pytest --record-mode=rewrite <test> -v'
            )
        interaction = self.cassette.interactions[self.interaction_idx]
        self.interaction_idx += 1
        return interaction

    def peek_interaction_type(self) -> Literal['sample', 'stream']:
        """Peek at the next interaction type without consuming it."""
        if self.interaction_idx >= len(self.cassette.interactions):
            raise IndexError(
                f'Cassette exhausted at interaction {self.interaction_idx}.\n'
                'Re-record this cassette with:\n'
                '  XAI_API_KEY=... uv run pytest --record-mode=rewrite <test> -v'
            )
        interaction = self.cassette.interactions[self.interaction_idx]
        return 'sample' if isinstance(interaction, SampleInteraction) else 'stream'

    @property
    def chat(self) -> Any:
        # We don't need to validate kwargs yet, but we keep the signature compatible.
        return type('Chat', (), {'create': self._chat_create})

    @property
    def files(self) -> Any:
        return type('Files', (), {'upload': self._files_upload})

    def _chat_create(self, *_args: Any, **_kwargs: Any) -> _CassetteChatInstance:
        expected_type = self.peek_interaction_type()
        return _CassetteChatInstance(self, expected_type)

    async def _files_upload(self, data: bytes, filename: str) -> Any:
        # Deterministic ID; good enough for replay since we don't actually call the backend.
        # Keeping similar shape to the real SDK return value.
        file_id = f'file-{abs(hash((len(data), filename))) % 1_000_000:06d}'
        return type('UploadedFile', (), {'id': file_id})()


@dataclass
class XaiProtoCassetteHybridClient:
    """Replay from an existing cassette but record "new episodes" when the cassette runs out."""

    _inner: AsyncClient
    cassette: XaiProtoCassette
    include_debug_json: bool = False
    interaction_idx: int = 0
    dirty: bool = False

    def _can_replay(self) -> bool:
        """Check if there are more recorded interactions to replay."""
        return self.interaction_idx < len(self.cassette.interactions)

    def _peek_interaction(self) -> Interaction | None:
        """Peek at the next interaction without consuming it."""
        if self.interaction_idx < len(self.cassette.interactions):
            return self.cassette.interactions[self.interaction_idx]
        return None

    def _consume_interaction(self) -> Interaction:
        """Consume and return the next interaction."""
        interaction = self.cassette.interactions[self.interaction_idx]
        self.interaction_idx += 1
        return interaction

    @property
    def chat(self) -> Any:
        return type('Chat', (), {'create': self._chat_create})

    @property
    def files(self) -> Any:
        return type('Files', (), {'upload': self._inner.files.upload})

    def _chat_create(self, *args: Any, **kwargs: Any) -> Any:
        inner_chat = self._inner.chat.create(*args, **kwargs)
        include_debug_json = self.include_debug_json
        client = self

        class _HybridChatInstance:
            async def sample(self) -> chat_types.Response:
                # Replay if we have a recorded SampleInteraction at this index.
                peeked = client._peek_interaction()
                if isinstance(peeked, SampleInteraction):
                    interaction = client._consume_interaction()
                    assert isinstance(interaction, SampleInteraction)
                    proto = chat_pb2.GetChatCompletionResponse()
                    proto.ParseFromString(interaction.response_raw)
                    return chat_types.Response(proto, index=None)

                # Otherwise record a new episode.
                request_raw = inner_chat.proto.SerializeToString()
                request_json: dict[str, Any] | None = None
                if include_debug_json:
                    request_json = MessageToDict(inner_chat.proto, preserving_proto_field_name=True)

                response = await inner_chat.sample()
                response_raw = response.proto.SerializeToString()

                response_json: dict[str, Any] | None = None
                if include_debug_json:
                    response_json = MessageToDict(response.proto, preserving_proto_field_name=True)

                client.cassette.interactions.append(
                    SampleInteraction(
                        request_raw=request_raw,
                        response_raw=response_raw,
                        request_json=request_json,
                        response_json=response_json,
                    )
                )
                client.interaction_idx += 1
                client.dirty = True
                return response

            def stream(self) -> Any:
                async def _aiter():
                    # Replay if we have a recorded StreamInteraction at this index.
                    peeked = client._peek_interaction()
                    if isinstance(peeked, StreamInteraction):
                        interaction = client._consume_interaction()
                        assert isinstance(interaction, StreamInteraction)

                        aggregated = chat_types.Response(chat_pb2.GetChatCompletionResponse(), index=None)
                        for chunk_bytes in interaction.chunks_raw:
                            chunk_proto = chat_pb2.GetChatCompletionChunk()
                            chunk_proto.ParseFromString(chunk_bytes)
                            aggregated.process_chunk(chunk_proto)
                            yield aggregated, chat_types.Chunk(chunk_proto, index=None)
                        return

                    # Otherwise record a new streaming episode.
                    request_raw = inner_chat.proto.SerializeToString()
                    request_json: dict[str, Any] | None = None
                    if include_debug_json:
                        request_json = MessageToDict(inner_chat.proto, preserving_proto_field_name=True)

                    chunks_raw: list[bytes] = []
                    chunks_json: list[dict[str, Any]] = []
                    try:
                        async for response, chunk in inner_chat.stream():
                            chunks_raw.append(chunk.proto.SerializeToString())
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
                        client.cassette.interactions.append(
                            StreamInteraction(
                                request_raw=request_raw,
                                chunks_raw=chunks_raw,
                                request_json=request_json,
                                chunks_json=chunks_json if include_debug_json else None,
                            )
                        )
                        client.interaction_idx += 1
                        client.dirty = True

                return _aiter()

        return _HybridChatInstance()


@dataclass
class XaiProtoRecorder:
    """Record `chat.sample()` and `chat.stream()` responses as protobuf bytes.

    Usage:
        recorder = XaiProtoRecorder(real_client)
        ... run agent using recorder.client ...
        recorder.dump(path)
    """

    _inner: AsyncClient
    cassette: XaiProtoCassette = field(default_factory=XaiProtoCassette)
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

        class _RecorderChatInstance:
            async def sample(self) -> chat_types.Response:
                request_raw = inner_chat.proto.SerializeToString()
                # Use MessageToDict for request JSON to get proper enum names
                request_json: dict[str, Any] | None = None
                if include_debug_json:
                    request_json = MessageToDict(inner_chat.proto, preserving_proto_field_name=True)
                response = await inner_chat.sample()
                response_raw = response.proto.SerializeToString()

                response_json: dict[str, Any] | None = None
                if include_debug_json:
                    response_json = MessageToDict(response.proto, preserving_proto_field_name=True)

                recorder.cassette.interactions.append(
                    SampleInteraction(
                        request_raw=request_raw,
                        response_raw=response_raw,
                        request_json=request_json,
                        response_json=response_json,
                    )
                )
                return response

            def stream(self) -> Any:
                async def _aiter():
                    request_raw = inner_chat.proto.SerializeToString()
                    # Use MessageToDict for request JSON to get proper enum names
                    request_json: dict[str, Any] | None = None
                    if include_debug_json:
                        request_json = MessageToDict(inner_chat.proto, preserving_proto_field_name=True)
                    chunks_raw: list[bytes] = []
                    chunks_json: list[dict[str, Any]] = []
                    try:
                        async for response, chunk in inner_chat.stream():
                            chunks_raw.append(chunk.proto.SerializeToString())
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
                        # Ensure data is persisted even if the consumer stops early.
                        recorder.cassette.interactions.append(
                            StreamInteraction(
                                request_raw=request_raw,
                                chunks_raw=chunks_raw,
                                request_json=request_json,
                                chunks_json=chunks_json if include_debug_json else None,
                            )
                        )

                return _aiter()

        return _RecorderChatInstance()


@dataclass
class XaiProtoCassetteSession:
    """A session that provides an xAI client and optionally records to a cassette."""

    client: XaiAsyncClientLike
    cassette_path: Path
    cassette: XaiProtoCassette | None = None
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
        cassette = XaiProtoCassette.load(cassette_path)
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
        cassette = XaiProtoCassette.load(cassette_path)
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
