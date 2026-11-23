"""The encodings of a secret, and which of them each value redactor's docs promise to replace.

`Shell(secret_env=)` masks a value as written: its docs say a command that transforms the value, such as by JSON
escaping, is not masked. `BrowserUse(sensitive_data=)` replaces a value as written, its JSON-escaped form as schema
output writes it, and its `quote_plus` form: its docs say other encodings, such as `%20` for a space, are not
matched. Each redactor's tests run every encoding in `ENCODINGS` and fail on a cell that disagrees with `COVERED`,
whether the redactor stopped replacing an encoding or started replacing one: update its docs with the table.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal
from urllib.parse import quote, quote_plus

Redactor = Literal['Shell', 'BrowserUse']

SECRET = 'p@ss w"rd/é\\'
"""A value whose encodings below all differ from it, and from each other."""

ENCODINGS: dict[str, Callable[[str], str]] = {
    'as-written': lambda value: value,
    'json-escaped': lambda value: json.dumps(value, ensure_ascii=False)[1:-1],
    'json-escaped-ascii': lambda value: json.dumps(value)[1:-1],
    'quote-plus': quote_plus,
    'quote': quote,
}

COVERED: dict[Redactor, frozenset[str]] = {
    'Shell': frozenset({'as-written'}),
    'BrowserUse': frozenset({'as-written', 'json-escaped', 'quote-plus'}),
}
