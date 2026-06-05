"""Tests for the experimental-capability warning convention."""

from __future__ import annotations

import importlib
import warnings

import pytest

from pydantic_ai_harness.experimental import HarnessExperimentalWarning
from pydantic_ai_harness.experimental._warn import warn_experimental


class TestExperimentalWarning:
    def test_message_names_feature_and_carries_silence_snippet(self):
        with pytest.warns(HarnessExperimentalWarning) as rec:
            warn_experimental('compaction')
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert '`pydantic_ai_harness.experimental.compaction`' in msg
        # The message must hand the user the exact, category-wide silence line.
        assert "warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)" in msg

    def test_one_filter_silences_every_capability(self):
        # A single category filter mutes all experimental warnings — no per-capability lines.
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # baseline: any warning is an error
            warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
            warn_experimental('compaction')
            warn_experimental('some_future_capability')  # also silenced, same filter

    def test_importing_a_capability_warns(self):
        module = importlib.import_module('pydantic_ai_harness.experimental.compaction')
        with pytest.warns(HarnessExperimentalWarning):
            importlib.reload(module)
