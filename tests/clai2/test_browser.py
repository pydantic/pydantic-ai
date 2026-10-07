import subprocess
import sys
import webbrowser
from typing import Any

import pytest

from pydantic_clai2.ui.browser import open_browser

NOISY = 'import sys; sys.stdout.write("out"); sys.stderr.write("[DEP0169] url.parse()"); sys.exit(int(sys.argv[1] != "https://x"))'


def opener(monkeypatch: pytest.MonkeyPatch, browser: webbrowser.BaseBrowser) -> None:
    monkeypatch.setattr(webbrowser, 'get', lambda: browser)


@pytest.mark.parametrize(
    ('url', 'opened'),
    [('https://x', True), ('https://y', False)],
)
def test_a_command_line_opener_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str], url: str, opened: bool
) -> None:
    opener(monkeypatch, webbrowser.GenericBrowser([sys.executable, '-c', NOISY, '%s']))
    assert open_browser(url) is opened
    assert capfd.readouterr() == ('', '')


def test_a_background_opener_is_not_waited_for(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    started: list[subprocess.Popen[bytes]] = []

    class Recorded(subprocess.Popen[bytes]):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            started.append(self)

    monkeypatch.setattr(subprocess, 'Popen', Recorded)
    opener(monkeypatch, webbrowser.BackgroundBrowser([sys.executable, '-c', NOISY, '%s']))
    assert open_browser('https://y') is True
    assert [process.wait() for process in started] == [1]
    assert capfd.readouterr() == ('', '')


def test_a_missing_opener_is_not_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    opener(monkeypatch, webbrowser.GenericBrowser('/nonexistent/browser'))
    assert open_browser('https://x') is False


def test_other_browsers_open_themselves(monkeypatch: pytest.MonkeyPatch) -> None:
    class Recorder(webbrowser.BaseBrowser):
        def open(self, url: str, new: int = 0, autoraise: bool = True) -> bool:
            self.url = url
            return True

    browser = Recorder()
    opener(monkeypatch, browser)
    assert open_browser('https://x') is True
    assert browser.url == 'https://x'
