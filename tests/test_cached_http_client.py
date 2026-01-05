"""Test for issue #3240: httpx.AsyncClient bound to different event loop error.

This reproduces the problem that occurs in Celery workers and Cloud Run Functions
where a cached httpx.AsyncClient from one event loop is used in a different event loop.

See: https://github.com/pydantic/pydantic-ai/issues/3240
"""

from __future__ import annotations

import asyncio

import pytest

from pydantic_ai.models import (
    cached_async_http_client,
    clear_cached_http_clients,
)


class TestCachedHttpClientEventLoopAwareness:
    """Tests for event-loop-aware caching (issue #3240).

    These tests verify that cached_async_http_client returns different clients
    for different event loops, preventing the 'bound to a different event loop' error.
    """

    def test_different_event_loops_get_different_clients(self):
        """Test that different event loops get different cached clients.

        This is the fix for issue #3240. After the fix:
        - Each event loop should get its own cached client
        - This prevents the 'bound to a different event loop' RuntimeError

        Before the fix, this test FAILS because both loops get the same client.
        After the fix, this test PASSES because each loop gets its own client.
        """
        # Clear any existing cache to start fresh
        clear_cached_http_clients()

        # --- Event loop 1: Create and cache the client ---
        loop1 = asyncio.new_event_loop()

        async def get_client_in_loop1():
            return cached_async_http_client(provider='test-issue-3240')

        try:
            client1 = loop1.run_until_complete(get_client_in_loop1())
            assert client1 is not None
            assert not client1.is_closed
        finally:
            loop1.run_until_complete(loop1.shutdown_asyncgens())
            loop1.close()

        # --- Event loop 2: Should get a DIFFERENT client ---
        loop2 = asyncio.new_event_loop()

        async def get_client_in_loop2():
            return cached_async_http_client(provider='test-issue-3240')

        try:
            client2 = loop2.run_until_complete(get_client_in_loop2())

            # CRITICAL ASSERTION: Different event loops should get different clients
            # Before the fix: client1 is client2 (SAME object) -> test FAILS
            # After the fix: client1 is not client2 (DIFFERENT objects) -> test PASSES
            assert client1 is not client2, (
                'Different event loops should get different cached clients. '
                'This is issue #3240 - the client from loop1 is being reused in loop2, '
                'which causes RuntimeError when the client tries to use its internal locks.'
            )
        finally:
            loop2.run_until_complete(loop2.shutdown_asyncgens())
            loop2.close()
            clear_cached_http_clients()

    def test_same_event_loop_gets_same_client(self):
        """Test that the same event loop gets the same cached client.

        This ensures we still have caching within the same event loop.
        """
        clear_cached_http_clients()

        loop = asyncio.new_event_loop()

        async def get_clients():
            client1 = cached_async_http_client(provider='test-same-loop')
            client2 = cached_async_http_client(provider='test-same-loop')
            return client1, client2

        try:
            client1, client2 = loop.run_until_complete(get_clients())
            # Same event loop, same provider -> same cached client
            assert client1 is client2
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            clear_cached_http_clients()

    def test_client_usable_after_loop_change(self):
        """Test that clients remain usable in their respective event loops.

        After the fix, each loop should have its own working client.
        Before the fix, trying to use the old client in a new loop raises RuntimeError.
        """
        clear_cached_http_clients()

        # --- Event loop 1 ---
        loop1 = asyncio.new_event_loop()

        async def work_in_loop1():
            client = cached_async_http_client(provider='test-usable')
            # Client should be usable - just check it's not closed
            assert not client.is_closed
            return client

        try:
            client1 = loop1.run_until_complete(work_in_loop1())
        finally:
            loop1.run_until_complete(loop1.shutdown_asyncgens())
            loop1.close()

        # --- Event loop 2 ---
        loop2 = asyncio.new_event_loop()

        async def work_in_loop2():
            client = cached_async_http_client(provider='test-usable')
            # This client should be usable in loop2
            # Before fix: this is the same client as loop1, bound to dead loop -> problems
            # After fix: this is a NEW client, bound to loop2 -> works fine
            assert not client.is_closed
            return client

        try:
            client2 = loop2.run_until_complete(work_in_loop2())
            # Verify we got different clients
            assert client1 is not client2
        finally:
            loop2.run_until_complete(loop2.shutdown_asyncgens())
            loop2.close()
            clear_cached_http_clients()


class TestCachedHttpClientBasicBehavior:
    """Tests for basic caching behavior (should pass before and after fix)."""

    @pytest.mark.anyio
    async def test_same_provider_returns_cached_client(self):
        """Test that the same provider returns the same cached client in the same loop."""
        client1 = cached_async_http_client(provider='test-same')
        client2 = cached_async_http_client(provider='test-same')
        assert client1 is client2

    @pytest.mark.anyio
    async def test_different_providers_return_different_clients(self):
        """Test that different providers return different clients."""
        client1 = cached_async_http_client(provider='provider-a')
        client2 = cached_async_http_client(provider='provider-b')
        assert client1 is not client2

    @pytest.mark.anyio
    async def test_different_timeouts_return_different_clients(self):
        """Test that different timeout settings return different clients."""
        client1 = cached_async_http_client(provider='test-timeout', timeout=100)
        client2 = cached_async_http_client(provider='test-timeout', timeout=200)
        assert client1 is not client2

    @pytest.mark.anyio
    async def test_client_has_correct_user_agent(self):
        """Test that cached client has the correct User-Agent header."""
        client = cached_async_http_client(provider='test-ua')
        assert 'pydantic-ai/' in client.headers.get('User-Agent', '')

    @pytest.mark.anyio
    async def test_closed_client_gets_replaced(self):
        """Test that a closed cached client gets replaced with a new one.

        This covers the branch where we detect a closed client in cache and create a new one.
        """
        # Get a cached client
        client1 = cached_async_http_client(provider='test-closed')
        assert not client1.is_closed

        # Close it manually
        await client1.aclose()
        assert client1.is_closed

        # Request again - should get a NEW client since the old one is closed
        client2 = cached_async_http_client(provider='test-closed')
        assert not client2.is_closed
        assert client1 is not client2
