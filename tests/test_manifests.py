"""Tests for `tests/manifests.py`, the check every classification table in the suite runs through.

These are unit tests: the primitive is test infrastructure with no public API to drive it through.
"""

from __future__ import annotations

import pytest

from .manifests import Exempt, assert_classified


def test_passes_when_every_member_is_classified():
    assert_classified({'a', 'b'}, {'a': Exempt('first'), 'b': Exempt('second')}, complete_walk=True, table_name='T')


def test_fails_on_an_unclassified_member():
    with pytest.raises(AssertionError, match=r"`T` has no entry for \['b'\]"):
        assert_classified({'a', 'b'}, {'a': Exempt('first')}, complete_walk=True, table_name='T')


def test_fails_on_an_entry_without_a_reason():
    with pytest.raises(AssertionError, match=r"`T` entries \['a'\] have no reason"):
        assert_classified({'a'}, {'a': Exempt('  ')}, complete_walk=True, table_name='T')


def test_fails_on_a_stale_entry_after_a_complete_walk():
    table = {'a': Exempt('first'), 'gone': Exempt('was a member')}
    with pytest.raises(AssertionError, match=r"`T` entries \['gone'\] no longer name a member"):
        assert_classified({'a'}, table, complete_walk=True, table_name='T')


def test_keeps_entries_an_incomplete_walk_may_have_skipped():
    assert_classified({'a'}, {'a': Exempt('first'), 'b': Exempt('skipped')}, complete_walk=False, table_name='T')


def test_names_types_and_tuple_keys_readably():
    with pytest.raises(AssertionError, match=r"`T` has no entry for \['Exempt.reason'\]"):
        assert_classified({(Exempt, 'reason')}, {}, complete_walk=True, table_name='T')
