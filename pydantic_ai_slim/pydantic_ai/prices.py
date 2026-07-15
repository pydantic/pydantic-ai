"""Keep model pricing data fresh by fetching the latest `genai-prices` data in the background."""

from __future__ import annotations

from threading import Lock

from genai_prices import UpdatePrices as _UpdatePrices

__all__ = ('update_in_background',)

_updater: _UpdatePrices | None = None
_updater_lock = Lock()


def update_in_background() -> None:
    """Start fetching the latest model pricing data in the background.

    The updater is retained as a shared owner for the lifetime of the process, so repeated calls
    are safe and other libraries can independently acquire compatible ownership.
    Call this after any `os.fork()`; inheriting a running updater in a child process is unsupported.

    This is a fire-and-forget convenience wrapper. For shutdown, configuration, or waiting for
    the first fetch, use
    [`genai_prices.UpdatePrices`](https://github.com/pydantic/genai-prices) directly
    as a context manager or via its `start()`/`stop()` methods.

    Example:
    ```python {test="skip"}
    from pydantic_ai import prices

    prices.update_in_background()
    ```
    """
    global _updater
    with _updater_lock:
        if _updater is None:
            updater = _UpdatePrices()
            updater.start()
            # This helper deliberately has no stop API; retaining the object keeps its shared
            # ownership claim alive for the process lifetime.
            _updater = updater
