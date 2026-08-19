"""Shared collection rules for the Playwright capability tests."""

from __future__ import annotations

import importlib.util

# `playwright` is gated on the `playwright` extra, so slim CI runs (no extras) can't
# import these modules. Ignore them at collection. A conditional expression rather
# than an `if` statement: branch coverage traces statement arcs, and no single
# environment can take both arms of an install-dependent branch.
collect_ignore = ['test_playwright.py'] if importlib.util.find_spec('playwright') is None else []


if not collect_ignore:  # pragma: no branch
    import pytest

    from pydantic_ai_harness.playwright import _toolset as toolset_module

    @pytest.fixture(autouse=True)
    def no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the private-address block from resolving test hostnames for real.

        `decide` resolves any host that is not already an address, so without this
        every navigation in the suite would issue a live DNS query. The stub answers
        with a public address, which is the case that changes nothing: a lookup that
        does not answer is a refusal, so returning nothing here would block every
        navigation in the suite. The lookup is stubbed rather than the caching
        wrapper around it, so the cache still runs in every test and the tests that
        care about it can count lookups. The cache is module-level, so it is emptied
        between tests.
        """
        toolset_module._resolution_cache.clear()

        async def public_address(host: str) -> tuple[str, ...]:
            return ('93.184.216.34',)

        monkeypatch.setattr(toolset_module, '_getaddrinfo', public_address)
