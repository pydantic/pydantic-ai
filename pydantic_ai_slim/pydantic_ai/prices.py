from __future__ import annotations

from genai_prices import UpdatePrices

_updater: UpdatePrices | None = None


def update_in_background() -> None:
    """Start fetching the latest model pricing data from GitHub in the background.

    Uses [`genai-prices`](https://github.com/pydantic/genai-prices)' `UpdatePrices` to download
    the latest pricing data in a background daemon thread, refreshing hourly.
    Call this once at application startup to keep pricing accurate for new models
    without waiting for a new `genai-prices` PyPI release.

    If the fetch fails (e.g. no network access), pricing silently falls back to
    the data bundled with the installed `genai-prices` package.

    This is a fire-and-forget convenience wrapper. If you need more control
    (e.g. stopping the thread, configuring the update interval, or waiting
    for the first fetch), use
    [`genai_prices.UpdatePrices`](https://github.com/pydantic/genai-prices) directly
    as a context manager or via its `start()`/`stop()` methods.

    Example:
    ```python {test="skip"}
    from pydantic_ai import prices

    prices.update_in_background()
    ```
    """
    global _updater
    if _updater is not None:
        return
    try:
        _updater = UpdatePrices()
        _updater.start()
    except Exception:
        _updater = None
