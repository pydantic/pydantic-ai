"""Bounded, latest-wins delivery to herdr's local Unix socket."""

import json
import logging
import socket
import threading
import time
from time import monotonic
from typing import Literal

from pydantic import JsonValue, TypeAdapter

_SOURCE = 'herdr:clai2'
_REPLY = TypeAdapter(dict[str, JsonValue])
_LOGGER = logging.getLogger(__name__)
Lane = Literal['state', 'session', 'activity', 'metadata', 'title', 'release']


class HerdrClient:
    """One plugin-owned worker; socket failures never escape into the agent."""

    def __init__(self, *, socket_path: str, pane_id: str, tab_id: str | None = None) -> None:
        self.socket_path = socket_path
        self.pane_id = pane_id
        self.tab_id = tab_id
        self._condition = threading.Condition()
        self._pending: dict[str, tuple[str, dict[str, JsonValue]]] = {}
        self._closing = False
        self._seq = time.time_ns() // 1000
        self._original_label: str | None = None
        self._last_label: str | None = None
        self._worker = threading.Thread(target=self._run, name='clai2-herdr', daemon=True)
        self._worker.start()

    def submit(self, lane: Lane, method: str, params: dict[str, JsonValue]) -> None:
        """Replace a mailbox slot without performing IO on the caller's thread."""
        with self._condition:
            if self._closing:
                return
            if lane == 'state':
                self._pending.pop('activity', None)
            self._pending[lane] = (method, params)
            self._condition.notify()

    def close(self) -> None:
        """Discard obsolete work, restore an owned tab label, and release once."""
        with self._condition:
            if not self._closing:
                self._closing = True
                self._pending.clear()
                self._pending['title'] = ('tab.rename', {'label': None})
                self._pending['release'] = ('pane.release_agent', {})
                self._condition.notify()
        # Each request has one total IO deadline, including retries and reads.
        # At most one in-flight request and three cleanup requests remain.
        self._worker.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    if self._closing:
                        return
                    self._condition.wait()
                lanes = ('state', 'session', 'activity', 'metadata', 'title', 'release')
                lane = next(k for k in lanes if k in self._pending)
                method, params = self._pending.pop(lane)
            if method == 'tab.rename':
                self._rename(params.get('label'))
            else:
                self._request(method, params)

    def _request(self, method: str, params: dict[str, JsonValue]) -> dict[str, JsonValue]:
        self._seq += 1
        if method.startswith('pane.'):
            params = {'pane_id': self.pane_id, 'source': _SOURCE, 'agent': 'clai2', 'seq': self._seq, **params}
        payload = (json.dumps({'id': f'{_SOURCE}:{self._seq}', 'method': method, 'params': params}) + '\n').encode()
        deadline = monotonic() + 0.5

        def remaining() -> float:
            timeout = deadline - monotonic()
            if timeout <= 0:
                raise TimeoutError('herdr request deadline exceeded')
            return timeout

        for _ in range(3):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(remaining())
                    connection.connect(self.socket_path)
                    connection.settimeout(remaining())
                    connection.sendall(payload)
                    response = bytearray()
                    while len(response) < 65536 and not response.endswith(b'\n'):
                        connection.settimeout(remaining())
                        chunk = connection.recv(4096)
                        if not chunk:
                            break
                        response.extend(chunk)
                    reply = _REPLY.validate_json(response)
                    if 'error' in reply:
                        _LOGGER.debug('herdr rejected %s: %s', method, reply['error'])
                        return {}
                    return reply
            except (OSError, ValueError):
                _LOGGER.debug('herdr delivery failed: %s', method, exc_info=True)
        return {}

    def _rename(self, label: JsonValue) -> None:
        if not self.tab_id:
            return
        reply = self._request('tab.get', {'tab_id': self.tab_id})
        result = reply.get('result')
        tab = result.get('tab') if isinstance(result, dict) else None
        if not isinstance(tab, dict) or tab.get('pane_count') != 1:
            return
        current = tab.get('label')
        if not isinstance(current, str):
            return
        # A user rename relinquishes ownership, including on subsequent title updates.
        if self._last_label is not None and current != self._last_label:
            return
        if isinstance(label, str):
            target = label
        elif self._original_label is not None:
            target = self._original_label
        else:
            return
        if current != target and not self._request('tab.rename', {'tab_id': self.tab_id, 'label': target}):
            return
        if isinstance(label, str):
            if self._original_label is None:
                self._original_label = current
            self._last_label = label
        else:
            self._original_label = self._last_label = None
