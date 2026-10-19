"""Keep model prices up to date by downloading the latest price list in the background."""

from __future__ import annotations

from genai_prices import UpdatePrices

__all__ = ('update_in_background',)


def update_in_background() -> UpdatePrices:
    """Download the latest model prices now, and again every hour, in the background.

    Pydantic AI bundles model prices at release time. Call this when your app starts to estimate
    costs for models released after you installed it.

    Downloads never block your code. If one fails, the last good price list stays in use and the
    failure is logged to the `genai-prices` logger.

    Returns the updater. Call `stop()` when your app shuts down.

    To download from your own URL or on a different schedule, use
    [`genai_prices.UpdatePrices`](https://github.com/pydantic/genai-prices/blob/main/packages/python/README.md#updateprices)
    directly. It shares one background download with this function.
    """
    updater = UpdatePrices()
    updater.start()
    return updater
