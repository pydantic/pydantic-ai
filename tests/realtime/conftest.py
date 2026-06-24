"""Realtime test configuration.

`test_finance_demo.py` imports the `pydantic_ai_examples` package, which isn't installed in the main
test environment (examples are import-checked separately via `tests/import_examples.py`). Skip
collecting it unless that package is available, so the unit-test job doesn't error on import.
"""

import importlib.util

collect_ignore: list[str] = []

if importlib.util.find_spec('pydantic_ai_examples') is None:
    collect_ignore.append('test_finance_demo.py')
