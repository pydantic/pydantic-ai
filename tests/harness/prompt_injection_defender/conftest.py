"""Shared collection rules for the prompt injection defender tests."""

from __future__ import annotations

import importlib.util

# The `stackone-defender` dependency needs the `prompt-injection-defender` extras,
# so slim CI runs without those extras can't import these modules.
# Ignore them at collection. A conditional expression rather than an
# `if` statement: branch coverage traces statement arcs, and no single environment can
# take both arms of an install-dependent branch.
collect_ignore = ['test_defender.py'] if importlib.util.find_spec('stackone_defender') is None else []
