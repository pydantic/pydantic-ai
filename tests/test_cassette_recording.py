"""Recording a cassette sends every request live, even one that matches an interaction recorded earlier.

`tests/conftest.py` patches cassetter so it doesn't replay an interaction that has already been played while
recording. Without the patch, every later turn of a conversation, which POSTs to the same URL, would be answered
with the first recorded response.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx2
import pytest
from cassetter import use_cassette

from . import cassette_hooks


class _EchoHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers['content-length']))
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def echo_url() -> Iterator[str]:
    with ThreadingHTTPServer(('127.0.0.1', 0), _EchoHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f'http://127.0.0.1:{server.server_address[1]}/v1/messages'
        server.shutdown()
        thread.join()


def _post_all(url: str, cassette_path: Path, record_mode: str, payloads: list[int]) -> list[int]:
    # The URI normalizer every recorded test gets from `vcr_config`, which routes matching through a second copy of
    # the cassette.
    with (
        use_cassette(cassette_path, record_mode=record_mode, uri_normalizer=cassette_hooks.normalize_uri),
        httpx2.Client() as client,
    ):
        return [json.loads(client.post(url, json={'n': n}).content)['n'] for n in payloads]


@pytest.mark.parametrize('record_mode', ['once', 'all', 'rewrite', 'new_episodes'])
def test_recording_sends_repeated_requests_live(echo_url: str, tmp_path: Path, record_mode: str) -> None:
    cassette_path = tmp_path / 'cassette.yaml'

    assert _post_all(echo_url, cassette_path, record_mode, [1, 2, 3]) == [1, 2, 3]
    assert _post_all(echo_url, cassette_path, 'none', [1, 2, 3]) == [1, 2, 3]


def test_new_episodes_replays_unplayed_interactions_and_records_the_rest(echo_url: str, tmp_path: Path) -> None:
    cassette_path = tmp_path / 'cassette.yaml'
    _post_all(echo_url, cassette_path, 'once', [1])

    assert _post_all(echo_url, cassette_path, 'new_episodes', [2, 3]) == [1, 3]
    assert _post_all(echo_url, cassette_path, 'none', [0, 0]) == [1, 3]
