from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import pytest

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
