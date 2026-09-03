from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pytest
from pydantic import JsonValue

from ..conftest import RequestCapture, try_import

if TYPE_CHECKING:
    from vcr.cassette import Cassette

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider
    from tests.cassette_utils import CassetteContext

with try_import() as google_imports:
    from google.genai import Client

    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider


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


def content_blocks(body: dict[str, JsonValue], block_type: str) -> list[dict[str, JsonValue]]:
    """Every content block of `block_type` a request's messages carry, in order.

    A block list is a flatter and more stable projection than the messages themselves: it survives a
    message being split or merged, so it pins how a block renders without churning on unrelated
    conversation-shape changes.
    """
    messages = body.get('messages')
    assert isinstance(messages, list)

    blocks: list[dict[str, JsonValue]] = []
    for message in messages:
        assert isinstance(message, dict)
        content = message.get('content')
        if isinstance(content, str):
            continue
        for block in json_objects(content):
            if block.get('type') == block_type:
                blocks.append(block)
    return blocks


def json_objects(value: JsonValue) -> list[dict[str, JsonValue]]:
    """Narrow a JSON array to objects, failing if the wire shape differs."""
    assert isinstance(value, list)
    objects: list[dict[str, JsonValue]] = []
    for item in value:
        assert isinstance(item, dict)
        objects.append(item)
    return objects


def message_shape(body: dict[str, JsonValue]) -> list[tuple[str, list[str]]]:
    """Each message's role and the types of its content blocks, dropping the payloads.

    The digest a history-rewriting test wants: it moves when compaction drops, reorders or re-wraps a
    turn, and stays put when only wording changes.
    """
    messages = body.get('messages')
    assert isinstance(messages, list)

    shape: list[tuple[str, list[str]]] = []
    for message in messages:
        assert isinstance(message, dict)
        role = message.get('role')
        assert isinstance(role, str)
        content = message.get('content')
        if isinstance(content, str):
            shape.append((role, ['<str>']))
            continue

        block_types: list[str] = []
        for block in json_objects(content):
            block_type = block.get('type')
            assert isinstance(block_type, str)
            block_types.append(block_type)
        shape.append((role, block_types))
    return shape


def cache_breakpoints(body: dict[str, JsonValue]) -> tuple[dict[str, JsonValue] | None, list[str]]:
    """The request-level `cache_control`, plus a path for every block carrying its own breakpoint.

    Where the breakpoints sit is the thing a caching test actually depends on: a breakpoint that
    moves silently re-processes the tail instead of reading from cache, with no error to notice.
    """
    cache_control = body.get('cache_control')
    assert cache_control is None or isinstance(cache_control, dict)

    blocks: list[str] = []
    for section in ('system', 'tools'):
        section_blocks = body.get(section)
        if section_blocks is None:
            continue
        for index, block in enumerate(json_objects(section_blocks)):
            if block.get('cache_control'):
                blocks.append(f'{section}[{index}]')

    messages = body.get('messages')
    assert isinstance(messages, list)
    for message_index, message in enumerate(messages):
        assert isinstance(message, dict)
        content = message.get('content')
        if isinstance(content, str):
            continue
        for block_index, block in enumerate(json_objects(content)):
            if block.get('cache_control'):
                blocks.append(f'messages[{message_index}].content[{block_index}]')
    return cache_control, blocks


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


@pytest.fixture
def vertex_client_google_provider() -> GoogleProvider:
    """A Vertex-backed `genai.Client` wrapped in `GoogleProvider`, the construction from #6792.

    `system` stays `'google'` while the transport is Google Cloud (Vertex), so transport
    (not the provider name) must drive Vertex-vs-Gemini-API behavior.
    """
    if not google_imports():  # pragma: lax no cover
        pytest.skip('google is not installed')

    return GoogleProvider(client=Client(vertexai=True, project='test-project', location='us-central1'))


@pytest.fixture
def gla_client_google_cloud_provider() -> GoogleCloudProvider:
    """A Gemini-Developer-API `genai.Client` wrapped in `GoogleCloudProvider`, the mirror of #6792.

    `system` stays `'google-cloud'` while the transport is the Gemini Developer API. `__init__`
    short-circuits on `client=` before it would force `vertexai=True`, so the two disagree in this
    direction too and every transport branch has to follow the client rather than the name.
    """
    if not google_imports():  # pragma: lax no cover
        pytest.skip('google is not installed')

    return GoogleCloudProvider(client=Client(vertexai=False, api_key='mock-api-key'))
