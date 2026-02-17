from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from tests.cassette_utils import CassetteContext


@pytest.fixture(scope='function')
def cassette_ctx(request: pytest.FixtureRequest, vcr: Any) -> CassetteContext:
    """Unified cassette verification context for model tests.

    Returns a CassetteContext for tests with a 'provider' parameter, or for
    non-parametrized tests (defaulting to 'vcr' provider).
    """
    from tests.cassette_utils import CassetteContext

    provider = 'vcr'
    if callspec := getattr(request.node, 'callspec', None):  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        params = cast(dict[str, object], callspec.params)
        p = params.get('provider')
        if isinstance(p, str):
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
