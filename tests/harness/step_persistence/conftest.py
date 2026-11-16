"""Collection rules for the step-persistence tests."""

from __future__ import annotations

import importlib.util

# `pymongo` is gated on the `mongodb` extra, so an install without it can't import
# the Mongo store tests. Ignore them at collection then. A conditional expression
# rather than an `if` statement: branch coverage traces statement arcs, and no
# single environment can take both arms of an install-dependent branch.
collect_ignore = ['test_mongo.py'] if importlib.util.find_spec('pymongo') is None else []
