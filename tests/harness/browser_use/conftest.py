"""Shared collection rules for the BrowserUse capability tests."""

from __future__ import annotations

import importlib.util

# The `browser-use` dependency is gated on the `browser-use` extra (and needs
# Python 3.11+), so slim CI runs can't import these modules. Ignore them at
# collection. A conditional expression rather than an `if` statement: branch
# coverage traces statement arcs, and no single environment can take both arms
# of an install-dependent branch.

collect_ignore = ['test_browser_use.py', 'test_model.py'] if importlib.util.find_spec('browser_use') is None else []
