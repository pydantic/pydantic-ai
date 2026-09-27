"""Opt-in search check against the E2B default template (without rg)."""

# pyright: basic, reportMissingImports=false

import os

import pytest

from pydantic_ai_harness.filesystem import FileSystemToolset

pytestmark = [pytest.mark.anyio, pytest.mark.e2b_live]


@pytest.mark.parametrize('anyio_backend', ['asyncio'])  # the E2B SDK needs asyncio
async def test_default_e2b_search_without_ripgrep() -> None:
    pytest.importorskip('e2b')
    if not os.getenv('E2B_API_KEY'):
        pytest.skip('E2B_API_KEY is required')
    from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

    backend = E2BSandboxBackend(sandbox_timeout=300)
    try:
        await backend.get_sandbox()
        await backend.write_bytes('/home/user/search-proof.txt', b'proof-marker\n')
        tools = FileSystemToolset(
            root_dir=None,
            allowed_patterns=[],
            denied_patterns=[],
            max_read_lines=200,
            max_list_results=100,
            max_search_results=10,
            max_find_results=10,
        )
        assert 'search-proof.txt:1:proof-marker' in await tools.search_files('proof-marker', workspace=backend)
    finally:
        if backend.ref is not None:
            # Kill only the sandbox created by this test, even if the assertion fails.
            import e2b

            await e2b.AsyncSandbox.kill(backend.ref.id)
