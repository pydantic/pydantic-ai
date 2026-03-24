import time
from typing import Callable, Any

class RetryHandler:
    """
    Handler for automatic retries in Pydantic AI agent runs.
    Addresses issue #3922.
    """
    @staticmethod
    def execute_with_retry(func: Callable, retries: int = 3, delay: float = 1.0) -> Any:
        last_exception = None
        for i in range(retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                print(f"Attempt {i+1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
        raise last_exception
