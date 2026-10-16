"""`clai2 --desktop`: the web UI in a pywebview window run as a child process."""

import io
import sys
import types
from pathlib import Path

import pytest

from pydantic_clai2.cli import desktop
from pydantic_clai2.cli.desktop import TITLE, Bridge, main, open_window, require_webview


def missing(name: str) -> None:
    """`find_spec` when pywebview is not installed."""
    return None


def found(name: str) -> object:
    """`find_spec` when pywebview is installed."""
    return object()


def test_require_webview_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop, 'find_spec', missing)
    with pytest.raises(ValueError, match=r'pydantic-clai2\[desktop\]'):
        require_webview()
    monkeypatch.setattr(desktop, 'find_spec', found)
    require_webview()


async def test_open_window_hands_the_address_over_stdin(tmp_path: Path) -> None:
    """The token travels on the child's stdin, never its command line; its exit status comes back."""
    received = tmp_path / 'received'
    status = await open_window(
        'http://127.0.0.1:1/?token=secret', command=['/bin/sh', '-c', f'cat > {received}; exit 3']
    )
    assert status == 3
    assert received.read_text() == 'http://127.0.0.1:1/?token=secret\n'


def test_bridge_opens_only_web_addresses_and_keeps_tokens_off_command_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    def open_url(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(desktop, 'open_browser', open_url)
    bridge = Bridge(tmp_path)
    assert bridge.open_external('https://logfire.pydantic.dev/x?q=1') is True
    assert bridge.open_external('file:///etc/passwd') is False
    # A workspace's page carries its token: the browser gets a private page that forwards to it.
    assert bridge.open_external('http://127.0.0.1:2/?token=secret&x=<1>') is True
    page = tmp_path / 'open.html'
    assert opened == ['https://logfire.pydantic.dev/x?q=1', page.as_uri()]
    assert page.stat().st_mode & 0o777 == 0o600
    assert page.read_text() == (
        '<!doctype html><meta http-equiv="refresh" content="0;url=http://127.0.0.1:2/?token=secret&amp;x=&lt;1&gt;">'
    )


def test_main_shows_the_address_in_a_maximized_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    windows: list[tuple[tuple[object, ...], dict[str, object]]] = []
    started: list[dict[str, object]] = []

    def create_window(*args: object, **kwargs: object) -> None:
        windows.append((args, kwargs))

    def start(**kwargs: object) -> None:
        started.append(kwargs)

    fake = types.ModuleType('webview')
    setattr(fake, 'create_window', create_window)
    setattr(fake, 'start', start)
    monkeypatch.setitem(sys.modules, 'webview', fake)
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))

    main(io.StringIO('http://127.0.0.1:1/?token=secret\n'))

    [(args, options)] = windows
    assert args == (TITLE, 'http://127.0.0.1:1/?token=secret')
    assert isinstance(options.pop('js_api'), Bridge)
    assert options == {'maximized': True, 'min_size': (800, 500), 'text_select': True}
    [start_options] = started
    storage = tmp_path / 'pydantic-clai2' / 'desktop'
    assert storage.is_dir()
    assert start_options['private_mode'] is False and start_options['storage_path'] == str(storage)
    icon = start_options['icon']
    assert isinstance(icon, str) and Path(icon).is_file()


def test_main_refuses_anything_but_the_web_ui_address() -> None:
    with pytest.raises(SystemExit, match='Expected the web UI address'):
        main(io.StringIO('file:///etc/passwd\n'))
