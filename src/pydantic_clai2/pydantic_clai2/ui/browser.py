"""Open links in the user's browser without letting the opener print over the terminal UI."""

import subprocess
import sys
import webbrowser


def open_browser(url: str) -> bool:
    """Open `url` like `webbrowser.open`, but discard the output of a command-line opener.

    `$BROWSER` and `xdg-open` run as plain commands that share CLAI's terminal, so whatever they print
    lands on top of the display: VS Code's remote `$BROWSER` helper, for one, is a Node script that prints
    deprecation warnings. Those commands run here with their streams sent to `os.devnull`; every other
    browser (macOS, Windows, and the GUI browsers `webbrowser` drives itself) opens through
    `webbrowser.open` as before.

    Returns whether the browser opened, as `webbrowser.open` does.
    """
    try:
        browser = webbrowser.get()
    except webbrowser.Error:
        return False
    if not isinstance(browser, webbrowser.GenericBrowser):
        return webbrowser.open(url)

    sys.audit('webbrowser.open', url)
    background = isinstance(browser, webbrowser.BackgroundBrowser)
    try:
        process = subprocess.Popen(
            [browser.name, *(arg.replace('%s', url) for arg in browser.args)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=background,
        )
    except OSError:
        return False
    # Mirrors `BackgroundBrowser.open` and `GenericBrowser.open`: a background opener counts as opened if it is
    # still running right after launch, a foreground one if it exits cleanly.
    return process.poll() is None if background else process.wait() == 0
