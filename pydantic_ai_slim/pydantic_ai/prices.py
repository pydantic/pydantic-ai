"""Keep model prices up to date by downloading the latest price list in the background."""

from __future__ import annotations

from genai_prices import UpdatePrices

__all__ = ('update_in_background',)


def update_in_background() -> UpdatePrices:
    """Download the latest model prices now, and again every hour, in the background.

    Pydantic AI ships with a price list that's only refreshed with each release, so a model that
    came out after your install has no cost until you upgrade. Call this once when your app
    starts to pick up new prices as they're published.

    Downloads never block your code. If one fails, the last good price list stays in use and the
    failure is logged to the `genai-prices` logger.

    Returns the updater, which you can use to wait for the first download or to stop updating:

    ```python {test="skip"}
    from pydantic_ai import prices

    updater = prices.update_in_background()
    updater.wait()  # block until the first download has finished
    updater.stop()  # stop updating, e.g. when your app shuts down
    ```

    It also works as a context manager, which stops updating on exit. If your app runs several
    worker processes, call this in each worker after it starts, not before the workers are forked.

    To download from your own URL or on a different schedule, use
    [`genai_prices.UpdatePrices`](https://github.com/pydantic/genai-prices/blob/main/packages/python/README.md#updateprices)
    directly. It shares one background download with this function.
    """
    updater = UpdatePrices()
    updater.start()
    return updater
