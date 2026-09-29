from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import cache_cleanup
from cache_cleanup import CacheEntry, render_summary, select_caches_to_delete, total_bytes

GIB = 1024**3


def entry(
    cache_id: int,
    *,
    ref: str = 'refs/heads/main',
    key: str = 'setup-uv-2-ubuntu-24.04-3.13-aaaa',
    size_in_bytes: int = GIB,
    created_at: str = '2026-09-29T00:00:00Z',
) -> CacheEntry:
    return CacheEntry(
        cache_id=cache_id,
        ref=ref,
        key=key,
        size_in_bytes=size_in_bytes,
        created_at=created_at,
        last_accessed_at=created_at,
    )


@pytest.mark.parametrize(
    'key,expected_stem',
    [
        pytest.param(
            'setup-uv-2-x86_64-ubuntu-24.04-3.14-pruned-fd01b3f602f40de6e2b11e3420c2fc7f2-lowest-versions',
            'setup-uv-2-x86_64-ubuntu-24.04-3.14-pruned-lowest-versions',
            id='hash-before-the-cache-suffix',
        ),
        pytest.param(
            'temporal-cli-Linux-X64-locked-297fcb5c96893fdd53f71350a6459e26aba89360',
            'temporal-cli-Linux-X64-locked',
            id='hash-last',
        ),
        pytest.param(
            'codeql-overlay-base-database-1-d953d79b74456ce0-python-2.27.1-a060fe43',
            'codeql-overlay-base-database-1-python-2.27.1',
            id='two-hashes',
        ),
        pytest.param(
            'hf-Linux-stsb-bert-tiny-safetensors-f3cb857cba53019a20df283396bcca179cf051a4',
            'hf-Linux-stsb-bert-tiny-safetensors',
            id='pinned-revision',
        ),
        pytest.param('deadbeefcafe', 'deadbeefcafe', id='nothing-but-a-hash'),
    ],
)
def test_group_drops_every_hash_shaped_segment(key: str, expected_stem: str) -> None:
    assert entry(1, key=key).group == ('refs/heads/main', expected_stem)


def test_group_keeps_words_that_only_look_hexadecimal() -> None:
    # `locked`, `lowest` and `examples` are all short of the hash length or carry a non-hex letter.
    assert entry(1, key='setup-uv-2-locked-examples').group == ('refs/heads/main', 'setup-uv-2-locked-examples')


def test_closed_pull_request_caches_go_even_when_under_target() -> None:
    caches = [
        entry(1, ref='refs/pull/1/merge'),
        entry(2, ref='refs/pull/2/merge'),
        entry(3, ref='refs/heads/main'),
    ]

    doomed = select_caches_to_delete(
        caches,
        open_pull_request_refs={'refs/pull/2/merge'},
        target_bytes=100 * GIB,
        keep_per_group=2,
    )

    assert [item.cache_id for item in doomed] == [1]


def test_closed_pull_request_caches_go_largest_first() -> None:
    caches = [
        entry(1, ref='refs/pull/1/merge', size_in_bytes=GIB),
        entry(2, ref='refs/pull/2/merge', size_in_bytes=3 * GIB),
    ]

    doomed = select_caches_to_delete(caches, open_pull_request_refs=set(), target_bytes=0, keep_per_group=2)

    assert [item.cache_id for item in doomed] == [2, 1]


def test_superseded_entries_go_oldest_first_until_the_target_is_met() -> None:
    caches = [
        entry(1, created_at='2026-09-01T00:00:00Z'),
        entry(2, created_at='2026-09-02T00:00:00Z'),
        entry(3, created_at='2026-09-03T00:00:00Z'),
        entry(4, created_at='2026-09-04T00:00:00Z'),
    ]

    doomed = select_caches_to_delete(caches, open_pull_request_refs=set(), target_bytes=3 * GIB, keep_per_group=2)

    assert [item.cache_id for item in doomed] == [1]
    assert total_bytes(caches) - total_bytes(doomed) == 3 * GIB


def test_the_newest_entries_in_a_group_are_never_deleted() -> None:
    caches = [entry(index, created_at=f'2026-09-0{index}T00:00:00Z') for index in range(1, 5)]

    doomed = select_caches_to_delete(caches, open_pull_request_refs=set(), target_bytes=0, keep_per_group=2)

    assert [item.cache_id for item in doomed] == [1, 2]


def test_groups_are_kept_apart_by_ref_and_key_stem() -> None:
    caches = [
        entry(1, key='setup-uv-2-3.13-aaaaaaaa-lint', created_at='2026-09-01T00:00:00Z'),
        entry(2, key='setup-uv-2-3.13-bbbbbbbb-lint', created_at='2026-09-02T00:00:00Z'),
        entry(3, key='setup-uv-2-3.14-aaaaaaaa-lint', created_at='2026-09-01T00:00:00Z'),
        entry(4, ref='refs/heads/other', key='setup-uv-2-3.13-aaaaaaaa-lint', created_at='2026-09-01T00:00:00Z'),
    ]

    doomed = select_caches_to_delete(caches, open_pull_request_refs=set(), target_bytes=0, keep_per_group=1)

    # Only the `3.13` group on `main` has a second entry, so only its older half is superseded.
    assert [item.cache_id for item in doomed] == [1]


def test_nothing_is_deleted_when_the_repository_is_inside_its_budget() -> None:
    caches = [entry(1), entry(2, ref='refs/pull/7/merge')]

    doomed = select_caches_to_delete(
        caches, open_pull_request_refs={'refs/pull/7/merge'}, target_bytes=100 * GIB, keep_per_group=2
    )

    assert doomed == []


def test_summary_reports_the_before_and_after_totals() -> None:
    caches = [entry(1, size_in_bytes=2 * GIB), entry(2, size_in_bytes=GIB)]

    summary = render_summary(caches, [caches[1]], target_bytes=2 * GIB, keep_per_group=2, dry_run=True)

    assert '- Before: 3.00 GiB across 2 entries' in summary
    assert '- Would delete: 1.00 GiB across 1 entries' in summary
    assert '- After: 2.00 GiB (target 2.00 GiB)' in summary
    assert '| 1.00 GiB | `refs/heads/main` | `setup-uv-2-ubuntu-24.04-3.13-aaaa` |' in summary
    assert 'Still over target' not in summary


def test_summary_says_deleted_when_it_is_not_a_dry_run() -> None:
    assert 'Deleted: 0.00 GiB' in render_summary([], [], target_bytes=GIB, keep_per_group=2, dry_run=False)


def test_summary_explains_a_target_it_could_not_reach() -> None:
    caches = [entry(1, size_in_bytes=3 * GIB)]

    summary = render_summary(caches, [], target_bytes=GIB, keep_per_group=2, dry_run=True)

    assert '- Still over target: every group is down to its newest 2 entries.' in summary


@pytest.mark.parametrize(
    'value,items_key,expected_ids',
    [
        pytest.param({'actions_caches': [{'id': 1}]}, 'actions_caches', [1], id='wrapped'),
        pytest.param([{'id': 2}], None, [2], id='bare-list'),
    ],
)
def test_page_items_reads_both_response_shapes(
    value: cache_cleanup.JsonValue, items_key: str | None, expected_ids: list[int]
) -> None:
    assert [item['id'] for item in cache_cleanup._page_items(value, items_key)] == expected_ids  # pyright: ignore[reportPrivateUsage]


def test_page_items_rejects_an_unexpected_shape() -> None:
    with pytest.raises(RuntimeError, match='Unexpected paginated response shape'):
        cache_cleanup._page_items('nope', 'actions_caches')  # pyright: ignore[reportPrivateUsage]
