"""Inventory tracked CE, reconcile reported reads, and confine weekly trimming edits.

Coverage receipts attest model reads; they cannot prove comprehension. Git objects,
not local overlays or filesystem symlinks, define the audit and its write boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypeGuard

BRANCH_PREFIX = 'gh-aw/weekly-ce-audit-'
REFERENCES = {
    '.macroscope/correctness/review-discipline.md',
    '.macroscope/ignore.md',
    'src/pydantic_ai_harness/gh-aw/pydantic.md',
    'src/pydantic_clai2/FILE_MAP.md',
    'src/pydantic_clai2/pydantic_clai2/customization.md',
}
PRIVATE_PARTS = {'local-notes', 'handoffs', 'memory', 'memories'}
PRIVATE_NAMES = {'PAPERCUTS.md', 'CLAUDE.local.md', 'issue-brief.md', 'pr-decisions.md', 'handoffs-index.md'}


@dataclass(frozen=True)
class Entry:
    """One tracked Markdown loading path and its canonical audit responsibility."""

    path: str
    scope: Literal['editable', 'report-only', 'alias', 'excluded']
    reason: str
    canonical: str | None = None
    sha256: str | None = None
    lines: int = 0


def classify(path: str) -> Literal['editable', 'report-only', 'excluded']:
    """Separate instruction content from private state and product documentation."""
    parts = PurePosixPath(path).parts
    if not path.endswith('.md') or PRIVATE_PARTS.intersection(parts) or parts[-1] in PRIVATE_NAMES:
        return 'excluded'
    ce = (
        parts[-1] in {'AGENTS.md', 'CLAUDE.md', 'SKILL.md'}
        or bool({'agent_docs', '.agents', '.claude'}.intersection(parts))
        or path.startswith('.github/')
        or path in REFERENCES
    )
    if not ce:
        return 'excluded'
    if path.startswith(('.github/', '.macroscope/', 'src/pydantic_ai_harness/gh-aw/')):
        return 'report-only'
    return 'editable'


def git(root: Path, *args: str) -> bytes:
    """Read git metadata or objects without invoking a shell."""
    return subprocess.check_output(['git', '-C', str(root), *args])


def inventory(root: Path, source: str) -> list[Entry]:
    """Inventory immutable tracked content and resolve aliases inside that tree."""
    objects: dict[str, tuple[str, str]] = {}
    for record in git(root, 'ls-tree', '-rz', source).split(b'\0'):
        if record:
            metadata, path = record.decode().split('\t', 1)
            mode, _, oid = metadata.split()
            objects[path] = mode, oid
    entries: list[Entry] = []
    for path, (mode, oid) in sorted(objects.items()):
        if not path.endswith('.md'):
            continue
        scope = classify(path)
        if scope == 'excluded':
            entries.append(Entry(path, scope, 'Private state, generated record, or product documentation'))
            continue
        canonical = path
        visited: set[str] = set()
        while mode == '120000':
            visited.add(canonical)
            target = git(root, 'cat-file', 'blob', oid).decode()
            canonical = posixpath.normpath(posixpath.join(posixpath.dirname(canonical), target))
            if canonical in visited or canonical not in objects or classify(canonical) == 'excluded':
                break
            mode, oid = objects[canonical]
        else:
            if mode == '100644' and canonical.endswith('.md'):
                body = git(root, 'cat-file', 'blob', oid)
                entries.append(
                    Entry(
                        path,
                        'alias' if canonical != path else scope,
                        'Tracked alias; review canonical content' if canonical != path else 'Tracked CE',
                        canonical,
                        hashlib.sha256(body).hexdigest(),
                        len(body.splitlines()),
                    )
                )
                continue
        entries.append(Entry(path, 'excluded', 'Unresolved, cyclic, external, or non-regular tracked alias'))
    return entries


def is_container(value: object) -> TypeGuard[dict[object, object] | list[object]]:
    """Narrow decoded JSON containers without introducing unknown element types."""
    return isinstance(value, (dict, list))


def mapping(value: object) -> dict[str, object]:
    """Validate an untrusted JSON object before accessing its fields."""
    if not is_container(value) or isinstance(value, list):
        raise ValueError('Expected an object with string keys')
    items: dict[object, object] = value
    if not all(isinstance(key, str) for key in items):
        raise ValueError('Expected an object with string keys')
    return {key: item for key, item in items.items() if isinstance(key, str)}


def existing_pr(payload: object, repository: str) -> int | None:
    """Match the workflow's open same-repository PR without claiming other owners."""
    if not is_container(payload) or isinstance(payload, dict):
        raise ValueError('Expected a complete open-PR list')
    items: list[object] = payload
    for item in items:
        pr = mapping(item)
        head = mapping(pr.get('head'))
        if head.get('repo') is None:
            continue
        repo = mapping(head.get('repo'))
        branch, number = head.get('ref'), pr.get('number')
        if (
            pr.get('state') == 'open'
            and repo.get('full_name') == repository
            and isinstance(branch, str)
            and branch.startswith(BRANCH_PREFIX)
            and isinstance(number, int)
        ):
            return number
    return None


def reconcile(entries: list[Entry], receipt: object, source: str) -> list[str]:
    """Return unread canonical paths, rejecting invented or inconsistent receipts."""
    data = mapping(receipt)
    if data.get('source') != source:
        raise ValueError('Receipt source does not match audited commit')
    reviewed = data.get('reviewed')
    if not is_container(reviewed) or isinstance(reviewed, dict):
        raise ValueError('Receipt reviewed must be a list')
    expected = {entry.path: entry for entry in entries if entry.scope in {'editable', 'report-only'}}
    seen: set[str] = set()
    complete: set[str] = set()
    rows: list[object] = reviewed
    for item in rows:
        row = mapping(item)
        path = row.get('path')
        if not isinstance(path, str) or path not in expected or path in seen:
            raise ValueError('Receipt contains an unknown, alias, excluded, or repeated path')
        seen.add(path)
        entry = expected[path]
        if row.get('sha256') != entry.sha256:
            raise ValueError(f'Receipt hash mismatch: {path}')
        ranges = row.get('ranges')
        if not is_container(ranges) or isinstance(ranges, dict):
            raise ValueError(f'Missing read ranges: {path}')
        next_line = 1
        spans: list[object] = ranges
        for span in spans:
            if not is_container(span) or isinstance(span, dict):
                raise ValueError(f'Invalid read range: {path}')
            bounds: list[object] = span
            if len(bounds) != 2:
                raise ValueError(f'Invalid read range: {path}')
            start, end = bounds
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or isinstance(start, bool)
                or isinstance(end, bool)
            ):
                raise ValueError(f'Invalid read range: {path}')
            if start != next_line or end < start or end > entry.lines:
                raise ValueError(f'Gapped, overlapping, or invalid read range: {path}')
            next_line = end + 1
        if next_line == entry.lines + 1:
            complete.add(path)
    return sorted(expected.keys() - complete)


def validate_edits(root: Path, source: str, entries: list[Entry], revision: str | None = None) -> list[str]:
    """Reject edits outside existing regular canonical instruction Markdown."""
    editable = {entry.path for entry in entries if entry.scope == 'editable'}
    changed = (
        git(
            root,
            'diff',
            '--no-ext-diff',
            '--no-textconv',
            '--name-only',
            '-z',
            source,
            *([revision] if revision else []),
            '--',
        )
        .decode()
        .split('\0')
    )
    untracked = [] if revision else git(root, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    paths = sorted({path for path in [*changed, *untracked] if path and not path.startswith('.ce-audit/')})
    for path in paths:
        if revision:
            metadata = git(root, 'ls-tree', revision, '--', path).decode()
            if path not in editable or not metadata.startswith('100644 blob '):
                raise ValueError(f'Edit outside existing canonical Markdown allowlist: {path}')
            git(root, 'show', f'{revision}:{path}').decode('utf-8')
            continue
        candidate = root / path
        if path not in editable or candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f'Edit outside existing canonical Markdown allowlist: {path}')
        if candidate.stat().st_mode & 0o111:
            raise ValueError(f'Executable mode change: {path}')
        candidate.read_text(encoding='utf-8')
    return paths


def submitted_branches(output: object) -> list[str]:
    """Validate the selected PR branches independently of the final checkout."""
    items = mapping(output).get('items')
    if not is_container(items) or isinstance(items, dict):
        raise ValueError('Invalid agent output items')
    branches: list[str] = []
    for item in items:
        row = mapping(item)
        if row.get('type') != 'create_pull_request':
            continue
        branch = row.get('branch')
        if not isinstance(branch, str) or not branch.startswith(BRANCH_PREFIX):
            raise ValueError('Invalid weekly audit branch')
        branches.append(f'refs/heads/{branch}')
    if len(branches) > 1:
        raise ValueError('Only one focused PR is permitted')
    return branches


def main() -> None:
    """Prepare an inventory or validate the agent's coverage and proposed edits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--source', required=True)
    parser.add_argument('--directory', type=Path, default=Path('.ce-audit'))
    parser.add_argument('--pull-requests', type=Path)
    parser.add_argument('--repository', default=os.environ.get('GITHUB_REPOSITORY', ''))
    parser.add_argument('--safe-outputs', type=Path)
    parser.add_argument('--agent-output', type=Path)
    args = parser.parse_args()
    entries = inventory(args.root, args.source)
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare':
        manifest = {'source': args.source, 'entries': [asdict(entry) for entry in entries]}
        (args.directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if args.pull_requests:
            payload: object = json.loads(args.pull_requests.read_text())
            number = existing_pr(payload, args.repository)
            if number is not None:
                if args.safe_outputs is None:
                    raise ValueError('Cannot skip inference without a safe-output path')
                with args.safe_outputs.open('a') as stream:
                    stream.write(
                        json.dumps({'type': 'noop', 'message': f'Existing weekly CE audit PR #{number}'}) + '\n'
                    )
        return
    paths = validate_edits(args.root, args.source, entries)
    branches: list[str] = []
    if args.agent_output:
        output: object = json.loads(args.agent_output.read_text())
        branches = submitted_branches(output)
        for branch in branches:
            git(args.root, 'merge-base', '--is-ancestor', args.source, branch)
            paths = sorted(set(paths) | set(validate_edits(args.root, args.source, entries, branch)))
    receipt_path = args.directory / 'receipt.json'
    unread = sorted(entry.path for entry in entries if entry.scope in {'editable', 'report-only'})
    if receipt_path.is_file():
        receipt: object = json.loads(receipt_path.read_text())
        unread = reconcile(entries, receipt, args.source)
    summary = [
        '# Weekly CE audit coverage',
        f'Source: `{args.source}`',
        f'Reported canonical coverage: {"incomplete" if unread else "complete"}.',
        'A receipt attests reads; structural validation does not prove comprehension.',
        f'Canonical files: {sum(entry.scope in {"editable", "report-only"} for entry in entries)}.',
        f'Alias loading paths: {sum(entry.scope == "alias" for entry in entries)}.',
        '## Report-only paths',
        *[f'- `{entry.path}`' for entry in entries if entry.scope == 'report-only'],
        '## Unread or incompletely reported paths',
        *[f'- `{path}`' for path in unread],
        '## Proposed edits',
        *[f'- `{path}`' for path in paths],
    ]
    (args.directory / 'summary.md').write_text('\n'.join(summary) + '\n')
    if (paths or branches) and unread:
        raise ValueError('Incomplete audit cannot publish trimming edits')


if __name__ == '__main__':
    main()
