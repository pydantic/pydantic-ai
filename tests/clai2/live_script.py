"""Drive a `termflow.live` app without a terminal."""

from termflow.live import KeyEvent, LiveApp


def frame(app: LiveApp, seconds: float = 0.0) -> str:
    """Advance `app` by `seconds` and return the screen it draws."""
    buffer = app.step(seconds)
    return '\n'.join(buffer.row_text(row) for row in range(buffer.height))


def press(app: LiveApp, *keys: str) -> str:
    """Press `keys` on `app`, typing the single characters among them, and return the next screen."""
    app.handle([KeyEvent(key, text=key) if len(key) == 1 else KeyEvent(key) for key in keys], 0.0, False)
    return frame(app)
