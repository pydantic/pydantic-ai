import pytest
import httpx
from unittest.mock import patch
from pydantic_ai.mcp import MCPServerHTTP

@pytest.mark.anyio
def test_mcpserverhttp_extra_http_client_args(monkeypatch):
    """Test that extra_http_client_args is passed to the httpx_client_factory and merged correctly."""
    created_clients = []

    class DummyAsyncClient:
        def __init__(self, **kwargs):
            self._init_kwargs = kwargs
            created_clients.append(kwargs)
        async def aclose(self):
            pass

    monkeypatch.setattr("httpx.AsyncClient", DummyAsyncClient)

    class DummySseCtx:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        async def __aenter__(self):
            httpx_factory = self.kwargs["httpx_client_factory"]
            client = httpx_factory(headers={"from_factory": "yup"}, timeout=12, auth="theauth")
            
            class DummyStream:
                async def send(self, msg):
                    pass
                async def aclose(self):
                    pass
            read_stream = DummyStream()
            write_stream = DummyStream()
            return (read_stream, write_stream)
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr("pydantic_ai.mcp.sse_client", DummySseCtx)

    server = MCPServerHTTP(
        url="http://any-url",
        extra_http_client_args={"timeout": 8, "cool_option": 1, "headers": {"foo": "bar"}}
    )

    # Enter the actual client_streams context, which will call DummySseCtx (mocked for sse_client)
    import anyio
    async def run_streams():
        async with server.client_streams():
            pass
    anyio.run(run_streams)

    assert created_clients
    kwargs = created_clients[-1]
    assert kwargs["timeout"] == 12  # override by factory arg
    assert kwargs["cool_option"] == 1
    assert "headers" in kwargs
    assert kwargs["headers"]["foo"] == "bar"
    assert kwargs["headers"]["from_factory"] == "yup"
    assert kwargs["auth"] == "theauth"
