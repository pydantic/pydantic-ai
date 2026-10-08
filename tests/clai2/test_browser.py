"""A command-line browser opener must not print over CLAI's display.

The openers here are real shell scripts run through `$BROWSER`, as VS Code's remote terminal sets it up:
what is under test is which of CLAI's file descriptors the opener process inherits.
"""

import sys
import time
import webbrowser
from pathlib import Path

import pytest

from pydantic_clai2.ui.browser import open_browser

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='the fake openers are POSIX shell scripts')

URL = 'https://example.com/sign-in'
NOISE = 'echo "(node:1) [DEP0169] DeprecationWarning: url.parse() behavior is not standardized" >&2\necho "opening $1"'


@pytest.fixture(autouse=True)
def fresh_webbrowser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `webbrowser` read the browsers and `$BROWSER` set by each test, instead of what it cached first."""
    for name in ('BROWSER', 'DISPLAY', 'WAYLAND_DISPLAY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(webbrowser, '_tryorder', None)
    monkeypatch.setattr(webbrowser, '_browsers', {})


@pytest.fixture
def opened(tmp_path: Path) -> Path:
    """Where the fake opener writes the URL it was given."""
    return tmp_path / 'opened'


def use_opener(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str, *, background: bool = False) -> None:
    script = tmp_path / 'opener'
    script.write_text(f'#!/bin/sh\n{body}\n', encoding='utf-8')
    script.chmod(0o755)
    # `webbrowser` runs a `$BROWSER` command ending in `&` in the background, as it does `xdg-open`.
    monkeypatch.setenv('BROWSER', f'{script} %s &' if background else str(script))


def test_opener_output_stays_off_the_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, opened: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    use_opener(monkeypatch, tmp_path, f'{NOISE}\necho "$1" > {opened}')
    assert open_browser(URL) is True
    assert opened.read_text(encoding='utf-8') == f'{URL}\n'
    assert capfd.readouterr() == ('', '')


# Like `webbrowser`'s own background openers, CLAI leaves the opener running and lets its `Popen` go.
@pytest.mark.filterwarnings('ignore:subprocess .* is still running:ResourceWarning')
def test_background_opener_output_stays_off_the_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, opened: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    use_opener(monkeypatch, tmp_path, f'{NOISE}\necho "$1" > {opened}\nexec sleep 1', background=True)
    assert open_browser(URL) is True
    deadline = time.monotonic() + 10
    while not opened.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert opened.read_text(encoding='utf-8') == f'{URL}\n'
    assert capfd.readouterr() == ('', '')


def test_failing_opener_reports_not_opened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    use_opener(monkeypatch, tmp_path, f'{NOISE}\nexit 1')
    assert open_browser(URL) is False
    assert capfd.readouterr() == ('', '')


def test_missing_opener_reports_not_opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('BROWSER', str(tmp_path / 'missing'))
    assert open_browser(URL) is False


def test_no_browser_reports_not_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webbrowser, '_tryorder', [])
    assert open_browser(URL) is False


def test_other_browsers_open_through_webbrowser() -> None:
    urls: list[str] = []

    class Recording(webbrowser.BaseBrowser):
        def open(self, url: str, new: int = 0, autoraise: bool = True) -> bool:
            urls.append(url)
            return True

    webbrowser.register('recording', None, Recording(), preferred=True)
    assert open_browser(URL) is True
    assert urls == [URL]
