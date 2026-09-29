"""Shared collection rules for the Exa capability tests."""

from __future__ import annotations

import importlib.util

# The `exa-py` dependency is gated on the `exa` extra, so slim CI runs (no extras)
# can't import these modules. Ignore them at collection. A conditional expression
# rather than an `if` statement: branch coverage traces statement arcs, and no
# single environment can take both arms of an install-dependent branch.
collect_ignore = ['test_exa.py', 'test_agent.py'] if importlib.util.find_spec('exa_py') is None else []
