from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx2
import pytest
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from vcr.cassette import Cassette

    from pydantic_ai.models.anthropic import AnthropicModel
    from tests.cassette_utils import CassetteContext

# `validate_json` parses through pydantic-core rather than the stdlib, and types the result without a cast.
_REQUEST_BODY_ADAPTER = TypeAdapter(dict[str, Any])


@dataclass
class RequestCapture:
    """Outbound request bodies, as the live code built them.

    A cassette records what was sent when it was recorded, and the default matchers ignore the body,
    so a request whose payload has since drifted still replays against its recording. httpx event
    hooks run inside `AsyncClient.send`, above the transport VCR patches, so they fire on replay too
    and see what is actually going out. Pass `capture.client` as a provider's `http_client` and
    snapshot a projection of `capture.body(...)` to pin the fields a test's claim rests on.
    """

    paths: list[str] = field(default_factory=list[str])
    raw_bodies: list[bytes] = field(default_factory=list[bytes])
    headers: list[httpx2.Headers] = field(default_factory=list[httpx2.Headers])
    client: httpx2.AsyncClient = field(init=False)

    def __post_init__(self) -> None:
        self.client = httpx2.AsyncClient(event_hooks={'request': [self._record]})

    async def _record(self, request: httpx2.Request) -> None:
        # Only the raw bytes are kept here: the hook runs on every request of every test that asks
        # for a capture, while a test typically inspects one of them. Parsing happens in `body`.
        self.paths.append(request.url.path)
        self.raw_bodies.append(request.read())
        # The cassette serializer strips `anthropic-*` headers, so the wire is the only place a test
        # can see beta gating.
        self.headers.append(request.headers)

    def bodies(self, path_suffix: str = '') -> list[dict[str, Any]]:
        """Every captured body whose URL path ends with `path_suffix`, parsed on demand."""
        return [
            _REQUEST_BODY_ADAPTER.validate_json(raw)
            for path, raw in zip(self.paths, self.raw_bodies)
            if path.endswith(path_suffix)
        ]

    def body(self, path_suffix: str = '', index: int = 0) -> dict[str, Any]:
        """The `index`th captured body whose URL path ends with `path_suffix`, parsed on demand."""
        matches = self.bodies(path_suffix)
        assert matches, f'no captured request matching {path_suffix!r}; saw {self.paths}'
        return matches[index]


@pytest.fixture
async def request_capture(anyio_backend: str) -> AsyncIterator[RequestCapture]:
    capture = RequestCapture()
    yield capture
    # Built directly rather than through `create_async_httpx2_client`, so the autouse
    # `close_httpx_clients` tracker never sees it and its pool would otherwise leak per test.
    await capture.client.aclose()


class AnthropicModelFactory(Protocol):
    def __call__(self, model_name: str, *, api_key: str | None = None, capture: bool = False) -> AnthropicModel: ...


@pytest.fixture
def anthropic_model(anthropic_api_key: str, request_capture: RequestCapture) -> AnthropicModelFactory:
    """Factory for Anthropic models in VCR-recorded integration tests.

    `capture=True` routes the model through the `request_capture` fixture's client, so the test can
    assert on the request as sent rather than as recorded. Both fixtures are function-scoped, so a
    test reading `request_capture` sees the same instance this wired in.
    """

    def _create_model(model_name: str, *, api_key: str | None = None, capture: bool = False) -> AnthropicModel:
        # Imported here rather than at module scope: this conftest also loads on shards installed
        # without the `anthropic` extra, where a top-level import would fail at collection.
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(
            api_key=api_key or anthropic_api_key, http_client=request_capture.client if capture else None
        )
        return AnthropicModel(model_name, provider=provider)

    return _create_model


def content_blocks(body: dict[str, Any], block_type: str) -> list[dict[str, Any]]:
    """Every content block of `block_type` a request's messages carry, in order.

    A block list is a flatter and more stable projection than the messages themselves: it survives a
    message being split or merged, so it pins how a block renders without churning on unrelated
    conversation-shape changes.
    """
    return [
        block
        for message in body['messages']
        if isinstance(message['content'], list)
        for block in message['content']
        if block.get('type') == block_type
    ]


def message_shape(body: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Each message's role and the types of its content blocks, dropping the payloads.

    The digest a history-rewriting test wants: it moves when compaction drops, reorders or re-wraps a
    turn, and stays put when only wording changes.
    """
    return [
        (
            message['role'],
            [block['type'] for block in message['content']] if isinstance(message['content'], list) else ['<str>'],
        )
        for message in body['messages']
    ]


def cache_breakpoints(body: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """The request-level `cache_control`, plus a path for every block carrying its own breakpoint.

    Where the breakpoints sit is the thing a caching test actually depends on: a breakpoint that
    moves silently re-processes the tail instead of reading from cache, with no error to notice.
    """
    blocks: list[str] = []
    for section in ('system', 'tools'):
        section_blocks: list[dict[str, Any]] = body[section] if isinstance(body.get(section), list) else []
        blocks += [f'{section}[{i}]' for i, block in enumerate(section_blocks) if block.get('cache_control')]
    blocks += [
        f'messages[{m}].content[{b}]'
        for m, message in enumerate(body['messages'])
        if isinstance(message['content'], list)
        for b, block in enumerate(message['content'])
        if block.get('cache_control')
    ]
    return body.get('cache_control'), blocks


@pytest.fixture(scope='function')
def cassette_ctx(request: pytest.FixtureRequest, vcr: Cassette) -> CassetteContext:
    """Unified cassette verification context for model tests.

    Returns a CassetteContext for tests with a 'provider' parameter, or for
    non-parametrized tests (defaulting to 'vcr' provider).
    """
    from tests.cassette_utils import CassetteContext

    provider = 'vcr'
    if callspec := getattr(request.node, 'callspec', None):  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        params = cast(dict[str, object], callspec.params)
        p = params.get('provider')
        if isinstance(p, str):  # pragma: no branch
            provider = p

    test_module: str = request.node.fspath.basename.replace('.py', '')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    test_dir = Path(request.node.fspath).parent  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    return CassetteContext(
        provider=provider,
        vcr=vcr,
        test_name=request.node.name,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        test_module=test_module,  # pyright: ignore[reportUnknownArgumentType]
        test_dir=test_dir,
    )
