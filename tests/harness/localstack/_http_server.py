from __future__ import annotations

import socket
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread


@dataclass(frozen=True)
class HttpResponse:
    status: int = 200
    body: str = '{}'


@dataclass(frozen=True)
class RunningHttpServer:
    endpoint_url: str
    port: int
    paths: list[str]


class _RecordingHttpServer(ThreadingHTTPServer):
    def __init__(self, responses: Sequence[HttpResponse], port: int | None = None) -> None:
        self.responses = list(responses) or [HttpResponse()]
        self.paths: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                owner.paths.append(self.path)
                response_index = min(len(owner.paths) - 1, len(owner.responses) - 1)
                response = owner.responses[response_index]
                body = response.body.encode()
                self.send_response(response.status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        super().__init__(('localhost', port or 0), Handler)


@contextmanager
def http_server(responses: Sequence[HttpResponse], *, port: int | None = None) -> Generator[RunningHttpServer]:
    server = _RecordingHttpServer(responses, port)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bound_port = server.server_port
    try:
        yield RunningHttpServer(endpoint_url=f'http://localhost:{bound_port}', port=bound_port, paths=server.paths)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('localhost', 0))
        port = sock.getsockname()[1]
    assert isinstance(port, int)
    return port
