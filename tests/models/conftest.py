from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from vcr.cassette import Cassette

    from pydantic_ai.providers.google import GoogleProvider
    from tests.cassette_utils import CassetteContext


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
    try:
        from google.genai import Client

        from pydantic_ai.providers.google import GoogleProvider
    except ImportError:  # pragma: lax no cover
        pytest.skip('google is not installed')

    return GoogleProvider(client=Client(vertexai=True, project='test-project', location='us-central1'))
