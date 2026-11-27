#!/usr/bin/env python3
"""Prepare a committed review snapshot and its added-line provenance (stdlib only)."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

CODE_PATHS = (':(glob,exclude)**/cassettes/**', ':(glob,exclude)**/*.cassette.yaml')
CASSETTE_PATHS = (':(glob)**/cassettes/**', ':(glob)**/*.cassette.yaml')
DIFF_FLAGS = (
    '--no-ext-diff',
    '--no-textconv',
    '--no-color',
    '--src-prefix=a/',
    '--dst-prefix=b/',
    '--find-renames',
    '--diff-algorithm=myers',
    '--ignore-submodules=none',
    '--submodule=short',
)
HUNK = re.compile(rb'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
BLAME = re.compile(rb'([0-9a-f]{40}|[0-9a-f]{64}) \d+ (\d+)(?: \d+)?')


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ['git', '-C', str(repo), '-c', 'core.quotePath=true', *args],
        check=True,
        capture_output=True,
    ).stdout


def added_lines(diff: bytes) -> dict[str, list[int]]:  # noqa: C901
    """Read hunk bodies before file headers; source text can resemble a header."""
    added: dict[str, list[int]] = {}
    path: str | None = None
    gitlink = False
    old_left = new_left = line_number = 0
    for line in diff.removesuffix(b'\n').split(b'\n'):
        if line.startswith(b'\\ No newline at end of file'):
            continue
        if old_left or new_left:
            prefix = line[:1]
            if prefix not in (b' ', b'+', b'-'):
                raise ValueError('Malformed diff: unfinished hunk')
            if prefix != b'+':
                old_left -= 1
            if prefix != b'-':
                if prefix == b'+' and not gitlink:
                    if path is None:
                        raise ValueError('Added line without a destination path')
                    added.setdefault(path, []).append(line_number)
                new_left -= 1
                line_number += 1
            if old_left < 0 or new_left < 0:
                raise ValueError('Malformed diff: hunk counts do not match')
        elif line.startswith(b'diff --git '):
            path = None
            gitlink = False
        elif line in (b'new file mode 160000', b'deleted file mode 160000') or (
            line.startswith(b'index ') and line.endswith(b' 160000')
        ):
            gitlink = True
        elif line.startswith(b'+++ '):
            raw = line[4:].split(b'\t', 1)[0]
            if raw.startswith(b'"'):
                # core.quotePath=true emits ASCII C strings; bytes literals decode
                # Git's octal UTF-8 escapes without interpreting source text.
                decoded = ast.literal_eval('b' + raw.decode('ascii'))
                if not isinstance(decoded, bytes):
                    raise ValueError('Malformed quoted path')
                raw = decoded
            if raw == b'/dev/null':
                path = None
            elif raw.startswith(b'b/'):
                path = raw[2:].decode('utf-8', errors='surrogateescape')
            else:
                raise ValueError('Diff destination lacks b/ prefix')
        elif match := HUNK.match(line):
            old_left = int(match[2]) if match[2] is not None else 1
            new_left = int(match[4]) if match[4] is not None else 1
            line_number = int(match[3])
    if old_left or new_left:
        raise ValueError('Malformed diff: truncated hunk')
    return added


def blame_map(repo: Path, base: str, head: str, diff: bytes) -> tuple[bytes, Counter[str]]:
    commits = set(git(repo, 'rev-list', f'{base}..{head}').splitlines())
    rows: list[str] = []
    counts: Counter[str] = Counter()
    for path, numbers in sorted(added_lines(diff).items()):
        blame = git(
            repo,
            '-c',
            'blame.ignoreRevsFile=',
            'blame',
            '--line-porcelain',
            '--no-textconv',
            head,
            '--',
            path,
        )
        origins = {int(m[2]): m[1] for line in blame.split(b'\n') if (m := BLAME.fullmatch(line))}
        label = json.dumps(path) if any(c.isspace() or c in '\\"' for c in path) else path
        for number in numbers:
            commit = origins.get(number)
            if commit is None:
                raise ValueError(f'Blame missing for {path}:{number} at {head}')
            tag = 'NEW' if commit in commits else 'PRE-EXISTING'
            rows.append(f'{label}:{number} {tag} {commit.decode("ascii")}\n')
            counts[tag] += 1
    return ''.join(rows).encode('utf-8', errors='surrogateescape'), counts


def prepare(repo: Path, base_ref: str, head_ref: str, output: Path) -> Counter[str]:
    head = git(repo, 'rev-parse', '--verify', f'{head_ref}^{{commit}}').decode().strip()
    base = git(repo, 'rev-parse', '--verify', f'{base_ref}^{{commit}}').decode().strip()
    if git(repo, 'ls-files', '--unmerged', '-z'):
        raise ValueError('Resolve and commit the unmerged index before reviewing')
    for operation in (
        'MERGE_HEAD',
        'CHERRY_PICK_HEAD',
        'REVERT_HEAD',
        'rebase-merge',
        'rebase-apply',
    ):
        location = git(repo, 'rev-parse', '--git-path', operation).decode().strip()
        if (repo / location).exists():
            raise ValueError(f'Finish {operation} before reviewing')
    # A conflicted merge has no settled tree for reviewers to assess.
    git(repo, 'merge-tree', '--write-tree', base, head)
    bases = git(repo, 'merge-base', '--all', base, head).splitlines()
    if len(bases) != 1:
        raise ValueError('Review requires one unambiguous merge base')
    merge_base = bases[0].decode('ascii')
    diff_args = ('diff', *DIFF_FLAGS, merge_base, head)
    diff = git(repo, *diff_args, '--patch', '--unified=3', '--', *CODE_PATHS)
    mapping, counts = blame_map(repo, merge_base, head, diff)
    files = git(repo, *diff_args, '--name-only')
    cassettes = git(repo, *diff_args, '--name-status', '--', *CASSETTE_PATHS)
    artifacts = {
        'base-ref.txt': (base_ref + '\n').encode(),
        'base-sha.txt': (base + '\n').encode(),
        'merge-base.txt': (merge_base + '\n').encode(),
        'head-sha.txt': (head + '\n').encode(),
        'branch-files.txt': files,
        'code.diff': diff,
        'code.blame-map.txt': mapping,
        'cassettes.list': cassettes,
        'diff-hash.txt': ('sha256:' + hashlib.sha256(diff).hexdigest() + '\n').encode(),
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output) as temporary:
        staging = Path(temporary)
        for name, contents in artifacts.items():
            (staging / name).write_bytes(contents)
        for name in artifacts:
            (staging / name).replace(output / name)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--base', required=True, help='Fetched canonical base ref')
    parser.add_argument('--head', default='HEAD', help='Commit to review (default: HEAD)')
    parser.add_argument('--output', type=Path, help='Defaults to worktree-local review-branch/tmp')
    args = parser.parse_args(argv)
    try:
        repo = Path(git(args.repo, 'rev-parse', '--show-toplevel').decode().strip())
        output = args.output or repo / '.agents/skills/review-branch/tmp'
        counts = prepare(repo, args.base, args.head, output)
    except subprocess.CalledProcessError as error:
        detail = error.stderr or error.stdout or b''
        print(detail.decode(errors='replace').strip() or str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, SyntaxError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({'output': str(output), 'added_lines': dict(counts)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
