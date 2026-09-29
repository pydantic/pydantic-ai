"""Prune GitHub Actions caches so the repository stays inside its storage limit.

GitHub evicts least-recently-used caches once a repository passes its limit -- 10 GB unless
an admin raises it -- which turns every cache into a coin flip rather than a guarantee:
`setup-uv` installs swing between 19s and 211s, and the pinned sentence-transformers model
is re-downloaded by every eligible job in the run that loses the race. Dropping the entries
nothing can reach any more keeps the ones that still pay for themselves resident, and past
the free 10 GB it also stops the dead ones being billed.

Two classes of entry are pruned, in order:

1. Caches scoped to a pull request that is no longer open. A `refs/pull/N/merge` cache is
   only ever restored by a run on that pull request, so once it closes the entry is dead
   weight. These go unconditionally.
2. Entries superseded within their group -- same ref, same key but for its content hashes --
   beyond the newest `--keep-per-group`. A `setup-uv` key carries a lockfile hash and a
   CodeQL overlay key a commit sha, so every lockfile bump or merge mints a fresh set and
   strands the previous one. These go oldest-first, and only while the repository is over
   `--target-bytes`.

The newest entry in a group is never deleted, so the steady state keeps one warm cache per
ref and key.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import string
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# GitHub's per-repository limit is 10 GB until an admin raises it with
# `PUT /repos/{owner}/{repo}/actions/cache/storage-limit`. Aim below whatever that limit is,
# so a week's runs still have room to save their own caches without evicting anything --
# raise this alongside the limit rather than letting the gap close.
DEFAULT_TARGET_BYTES = 8 * 1024**3
# One warm entry per group is enough to serve the next run; a second covers a branch that
# forked before the most recent lockfile bump.
DEFAULT_KEEP_PER_GROUP = 2
PULL_REQUEST_REF_PREFIX = 'refs/pull/'
# Long enough that no word in a cache key qualifies -- the shortest real hash in this
# repository's keys is the CodeQL overlay's 8-character commit sha.
HASH_SEGMENT_MIN_LENGTH = 8

JsonValue = None | bool | int | float | str | list['JsonValue'] | dict[str, 'JsonValue']
JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class CacheEntry:
    cache_id: int
    ref: str
    key: str
    size_in_bytes: int
    created_at: str
    last_accessed_at: str

    @property
    def group(self) -> tuple[str, str]:
        """The ref and key stem this entry competes with its siblings for.

        Cache keys carry a content hash -- a lockfile hash for `setup-uv`, a commit sha for the
        CodeQL overlay, a pinned revision for the Hugging Face model -- but not always in the
        same place: `setup-uv` follows its hash with the job's cache suffix, so the hash lands
        mid-key. Dropping every hash-shaped segment groups an entry with the older entries it
        supersedes while keeping entries that differ by job apart. A key that is nothing but a
        hash is its own group.
        """
        stem = '-'.join(segment for segment in self.key.split('-') if not _is_hash_segment(segment))
        return (self.ref, stem or self.key)


class GitHubClient:
    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token
        self.ssl_context = _ssl_context()

    def list_caches(self) -> list[CacheEntry]:
        return [_cache_entry(item) for item in self._paginated('actions/caches', 'actions_caches')]

    def open_pull_request_refs(self) -> set[str]:
        refs: set[str] = set()
        for pull_request in self._paginated('pulls?state=open', None):
            number = pull_request.get('number')
            if isinstance(number, int):
                refs.add(f'{PULL_REQUEST_REF_PREFIX}{number}/merge')
        return refs

    def delete_cache(self, cache_id: int) -> None:
        self._request_json(f'actions/caches/{cache_id}', method='DELETE')

    def _request_json(self, path: str, *, method: str = 'GET') -> JsonValue:
        request = urllib.request.Request(
            self._url(path),
            method=method,
            headers={
                'Accept': 'application/vnd.github+json',
                'Authorization': f'Bearer {self.token}',
                'X-GitHub-Api-Version': '2022-11-28',
            },
        )
        with urllib.request.urlopen(request, timeout=30, context=self.ssl_context) as response:
            body = response.read()
        return json.loads(body) if body else None

    def _paginated(self, path: str, items_key: str | None) -> list[JsonObject]:
        parsed = urllib.parse.urlsplit(path)
        query = [item for item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True) if item[0] != 'page']
        query = [item for item in query if item[0] != 'per_page']
        query.append(('per_page', '100'))

        page = 1
        results: list[JsonObject] = []
        while True:
            page_path = urllib.parse.urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    urllib.parse.urlencode([*query, ('page', str(page))]),
                    '',
                )
            )
            items = _page_items(self._request_json(page_path), items_key)
            results.extend(items)
            if len(items) < 100:
                return results
            page += 1

    def _url(self, path: str) -> str:
        return f'https://api.github.com/repos/{self.repo}/{path}'


def select_caches_to_delete(
    caches: list[CacheEntry],
    *,
    open_pull_request_refs: set[str],
    target_bytes: int,
    keep_per_group: int,
) -> list[CacheEntry]:
    """Pick the caches to delete, in the order they should go."""
    doomed = [
        entry
        for entry in caches
        if entry.ref.startswith(PULL_REQUEST_REF_PREFIX) and entry.ref not in open_pull_request_refs
    ]
    doomed.sort(key=lambda entry: (-entry.size_in_bytes, entry.cache_id))

    remaining = total_bytes(caches) - total_bytes(doomed)
    if remaining <= target_bytes:
        return doomed

    doomed_ids = {entry.cache_id for entry in doomed}
    for superseded in _superseded(
        [entry for entry in caches if entry.cache_id not in doomed_ids], keep_per_group=keep_per_group
    ):
        if remaining <= target_bytes:
            break
        doomed.append(superseded)
        remaining -= superseded.size_in_bytes
    return doomed


def _superseded(caches: list[CacheEntry], *, keep_per_group: int) -> list[CacheEntry]:
    """Every entry beyond the newest `keep_per_group` in its group, oldest first."""
    groups: dict[tuple[str, str], list[CacheEntry]] = defaultdict(list)
    for entry in caches:
        groups[entry.group].append(entry)

    superseded: list[CacheEntry] = []
    for group in groups.values():
        group.sort(key=lambda entry: (entry.created_at, entry.cache_id), reverse=True)
        superseded.extend(group[keep_per_group:])
    superseded.sort(key=lambda entry: (entry.created_at, entry.cache_id))
    return superseded


def total_bytes(caches: list[CacheEntry]) -> int:
    return sum(entry.size_in_bytes for entry in caches)


def render_summary(
    caches: list[CacheEntry],
    doomed: list[CacheEntry],
    *,
    target_bytes: int,
    keep_per_group: int,
    dry_run: bool,
) -> str:
    before = total_bytes(caches)
    freed = total_bytes(doomed)
    verb = 'Would delete' if dry_run else 'Deleted'
    lines = [
        '## Actions cache cleanup',
        '',
        f'- Before: {_gib(before)} across {len(caches)} entries',
        f'- {verb}: {_gib(freed)} across {len(doomed)} entries',
        f'- After: {_gib(before - freed)} (target {_gib(target_bytes)})',
    ]
    if before - freed > target_bytes:
        lines.append(
            f'- Still over target: every group is down to its newest {keep_per_group} entries. '
            'Shrink what CI caches, or lower `--keep-per-group`.'
        )
    if doomed:
        lines += ['', '| Size | Ref | Key |', '| --- | --- | --- |']
        lines += [f'| {_gib(entry.size_in_bytes)} | `{entry.ref}` | `{entry.key}` |' for entry in doomed[:20]]
        if len(doomed) > 20:
            lines.append(f'| … | … | {len(doomed) - 20} more |')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target-bytes', type=int, default=DEFAULT_TARGET_BYTES)
    parser.add_argument('--keep-per-group', type=int, default=DEFAULT_KEEP_PER_GROUP)
    parser.add_argument('--dry-run', action='store_true', help='Report what would go without deleting anything.')
    args = parser.parse_args()

    client = _github_client_from_env()
    caches = client.list_caches()
    doomed = select_caches_to_delete(
        caches,
        open_pull_request_refs=client.open_pull_request_refs(),
        target_bytes=args.target_bytes,
        keep_per_group=args.keep_per_group,
    )

    if not args.dry_run:
        for entry in doomed:
            try:
                client.delete_cache(entry.cache_id)
            except urllib.error.HTTPError as error:  # pragma: no cover - depends on a live race
                # A cache GitHub evicted, or another run replaced, between listing and deleting.
                if error.code != 404:
                    raise
                print(f'Cache {entry.cache_id} ({entry.key}) was already gone')

    summary = render_summary(
        caches,
        doomed,
        target_bytes=args.target_bytes,
        keep_per_group=args.keep_per_group,
        dry_run=args.dry_run,
    )
    print(summary)
    if summary_path := os.getenv('GITHUB_STEP_SUMMARY'):
        with Path(summary_path).open('a', encoding='utf-8') as summary_file:
            summary_file.write(summary + '\n')


def _cache_entry(item: JsonObject) -> CacheEntry:
    return CacheEntry(
        cache_id=_expect_int(item.get('id'), 'id'),
        ref=_expect_str(item.get('ref'), 'ref'),
        key=_expect_str(item.get('key'), 'key'),
        size_in_bytes=_expect_int(item.get('size_in_bytes'), 'size_in_bytes'),
        created_at=_expect_str(item.get('created_at'), 'created_at'),
        last_accessed_at=_expect_str(item.get('last_accessed_at'), 'last_accessed_at'),
    )


def _page_items(value: JsonValue, items_key: str | None) -> list[JsonObject]:
    if items_key is None and isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if items_key is not None and isinstance(value, dict):
        items = value.get(items_key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    raise RuntimeError(f'Unexpected paginated response shape: {value!r}')


def _expect_int(value: JsonValue, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f'Expected an int for {field}, got {value!r}')
    return value


def _expect_str(value: JsonValue, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f'Expected a str for {field}, got {value!r}')
    return value


def _is_hash_segment(segment: str) -> bool:
    return len(segment) >= HASH_SEGMENT_MIN_LENGTH and all(character in string.hexdigits for character in segment)


def _gib(size_in_bytes: int) -> str:
    return f'{size_in_bytes / 1024**3:.2f} GiB'


def _github_client_from_env() -> GitHubClient:
    repo = os.getenv('GITHUB_REPOSITORY')
    token = os.getenv('GITHUB_TOKEN')
    if not repo:
        raise SystemExit('GITHUB_REPOSITORY is required')
    if not token:
        raise SystemExit('GITHUB_TOKEN is required')
    return GitHubClient(repo, token)


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


if __name__ == '__main__':
    main()
