from __future__ import annotations

from genai_prices import UpdatePrices


def update_prices() -> None:
    """Start fetching the latest model pricing data from GitHub in the background.

    Uses [`genai-prices`](https://github.com/pydantic/genai-prices)' `UpdatePrices` to download
    the latest pricing data in a background daemon thread, refreshing hourly.
    Call this once at application startup to keep pricing accurate for new models
    without waiting for a new `genai-prices` PyPI release.

    If the fetch fails (e.g. no network access), pricing silently falls back to
    the data bundled with the installed `genai-prices` package.

    Example:
    ```python {test="skip"}
    import pydantic_ai

    pydantic_ai.update_prices()

    agent = pydantic_ai.Agent('openai:gpt-5.2')
    ```
    """
    try:
        UpdatePrices().start()
    except Exception:
        pass
