"""Low-overhead leak probe: stacks are recorded only when a socket transport is created or closed.

Loaded with `-p leakprobe`. Prints `[leakprobe]` lines to the real stderr.
"""

import asyncio
import asyncio.base_events
import asyncio.selector_events as se
import sys
import traceback

_test = '<collection>'
_closed_loops: dict[int, str] = {}


def _out(text: str) -> None:
    print(f'[leakprobe] {text}', file=sys.__stderr__, flush=True)


def _stack() -> str:
    frames = traceback.extract_stack()[:-2]
    keep = [f for f in frames if '_pytest' not in f.filename and 'pluggy' not in f.filename]
    return ''.join(traceback.format_list(keep[-22:]))


_orig_init = se._SelectorSocketTransport.__init__


def _init(self, loop, sock, protocol, *args, **kwargs):
    self._probe_created = (_test, id(loop), _stack())
    self._probe_closed = None
    _orig_init(self, loop, sock, protocol, *args, **kwargs)


_orig_close = se._SelectorTransport.close


def _close(self):
    if getattr(self, '_probe_closed', 1) is None:
        self._probe_closed = (_test, _stack())
    return _orig_close(self)


_orig_del = se._SelectorTransport.__del__


def _del(self, _warn=None):
    if self._sock is not None and hasattr(self, '_probe_created'):
        test, loop_id, created = self._probe_created
        _out(f'LEAKED {self!r} loop={loop_id:x} loop_closed_during={_closed_loops.get(loop_id)!r}')
        _out(f'  created during {test}:\n{created}')
        if self._probe_closed:
            closed_test, closed = self._probe_closed
            _out(f'  close() called during {closed_test}:\n{closed}')
        else:
            _out('  close() was never called')
    _orig_del(self)


se._SelectorSocketTransport.__init__ = _init
se._SelectorTransport.close = _close
se._SelectorTransport.__del__ = _del

_orig_loop_close = asyncio.base_events.BaseEventLoop.close


def _loop_close(self):
    if not self.is_closed():
        _closed_loops[id(self)] = _test
        ready = [h for h in self._ready if not h._cancelled]
        pending = [t for t in asyncio.all_tasks(self) if not t.done()]
        if ready or pending:
            _out(f'loop {id(self):x} closed during {_test} with {len(ready)} ready callback(s), {len(pending)} pending task(s)')
            for h in ready:
                _out(f'  ready: {h!r}'[:300])
            for t in pending:
                _out(f'  pending: {t!r}'[:300])
    return _orig_loop_close(self)


asyncio.base_events.BaseEventLoop.close = _loop_close


def pytest_runtest_logstart(nodeid, location):
    global _test
    _test = nodeid


def pytest_sessionfinish(session, exitstatus):
    global _test
    _test = '<session teardown>'
