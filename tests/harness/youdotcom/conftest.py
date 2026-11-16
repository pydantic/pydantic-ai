"""Shared collection rules for the You.com capability tests."""

from __future__ import annotations

import importlib.util

# The `youdotcom` dependency is gated on the `youdotcom` extra, so slim CI runs
# (no extras) can't import these modules. Ignore them at collection. A
# conditional expression rather than an `if` statement: branch coverage traces
# statement arcs, and no single environment can take both arms of an
# install-dependent branch.
collect_ignore = ['test_youdotcom.py'] if importlib.util.find_spec('youdotcom') is None else []
