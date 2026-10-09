"""Open sign-in links without letting the opener print into the terminal UI."""

import subprocess
import webbrowser


def open_browser(url: str) -> bool:
    """`webbrowser.open`, except that a command-line opener's output is discarded.

    `$BROWSER` openers are often scripts: VS Code's remote helper is a Node script whose deprecation warnings
    would otherwise land in the transcript. Raises `webbrowser.Error` when no browser is available, as
    `webbrowser.open` does.
    """
    browser = webbrowser.get()
    if not isinstance(browser, webbrowser.GenericBrowser):
        return browser.open(url)
    command = [browser.name, *(arg.replace('%s', url) for arg in browser.args)]
    background = isinstance(browser, webbrowser.BackgroundBrowser)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=background,
        )
    except OSError:
        return False
    return background or process.wait() == 0
