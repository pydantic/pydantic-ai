"""Fail when a change adds a model adapter `try` that handles an SDK's status error but not the transport's.

A user can mount a retrying `httpx` transport under any provider that takes an HTTP client. Once it gives up on an
error status, it raises `HTTPStatusError` itself, and most SDKs let that through unwrapped. A `try` that maps the
SDK's status error, like `openai.APIStatusError` or `google.genai.errors.APIError`, therefore misses the same status
when the transport gave up on it, unless it handles `HTTPStatusError` too; so does a `suppress()` naming one. Each such
`try` or `suppress()` in a model, embedding or image adapter counts, and a change may not raise a file's count. What
the base already has is allowed, so existing sites pass until a change adds to them.

`tests/test_model_adapter_conformance.py` sends a gave-up status to every adapter entry point that calls an SDK; this
check also reaches the SDK calls behind no entry point of their own, like a file upload or a cleanup after a failure.
A provider's `_errors.py`, which holds its shared error mapping, isn't counted. botocore's `ClientError` and gRPC's
`RpcError` aren't counted either: neither SDK sends through `httpx`.

During a merge, `--staged` reports a file only when its count is higher than in both parents.

Usage:
    uv run scripts/check_status_error_catches.py --staged     # the index against HEAD, for pre-commit
    uv run scripts/check_status_error_catches.py --base REF   # the working tree against REF
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from _ratchet import Change, Options, escape, merge_head, parse_args, read_changes, worse_than_both

ROOTS = (
    'pydantic_ai_slim/pydantic_ai/models',
    'pydantic_ai_slim/pydantic_ai/embeddings',
    'pydantic_ai_slim/pydantic_ai/images',
)

SDK_STATUS_ERRORS = frozenset(
    {'APIStatusError', 'APIError', 'ApiError', 'SDKError', 'HfHubHTTPError', 'TypeSafeAPIError'}
)
"""The status errors of the SDKs that send through `httpx`: `openai`, `anthropic` and `groq`; `google.genai`'s
`errors.APIError`; `cohere`; `mistralai`; `huggingface_hub`; `typesafe_sdk`."""

GOOGLE_STATUS_ERRORS = frozenset({'ClientError', 'ServerError'})
"""`google.genai`'s `errors.ClientError` and `errors.ServerError`, counted only as `errors.<name>`: a bare `ClientError`
in a Bedrock adapter is botocore's."""

ADVICE = (
    'a retrying `httpx` transport raises `HTTPStatusError` for a status it gave up on, which the SDK lets through: '
    "handle it in the same `try`, as `_model_errors.transport_status_error` maps it, or map the provider's errors in "
    'its `_errors.py`'
)


@dataclass(frozen=True)
class _Finding:
    line: int
    text: str


@dataclass(frozen=True)
class _Growth:
    path: str
    old: int
    new: int
    added: list[_Finding]


def _named(node: ast.expr | None) -> Iterator[str]:
    """The exception names an `except` clause or `suppress()` argument names, qualified as written."""
    if isinstance(node, ast.Tuple):
        for element in node.elts:
            yield from _named(element)
    elif isinstance(node, ast.Attribute | ast.Name):
        yield ast.unparse(node)


def _is_sdk_status_error(name: str) -> bool:
    qualifier, _, last = name.rpartition('.')
    google_errors = qualifier == 'errors' or qualifier.endswith('.errors')
    return last in SDK_STATUS_ERRORS or (google_errors and last in GOOGLE_STATUS_ERRORS)


def _unpaired(names: list[str]) -> bool:
    return any(_is_sdk_status_error(n) for n in names) and not any(n.endswith('HTTPStatusError') for n in names)


def _findings(source: str | None) -> list[_Finding]:
    if source is None:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings: list[_Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try | ast.TryStar):
            names = [name for handler in node.handlers for name in _named(handler.type)]
            if _unpaired(names):
                findings.append(_Finding(node.lineno, f'except {", ".join(names)}'))
        elif isinstance(node, ast.Call) and ast.unparse(node.func).endswith('suppress'):
            names = [name for arg in node.args for name in _named(arg)]
            if _unpaired(names):
                findings.append(_Finding(node.lineno, f'suppress({", ".join(names)})'))
    return sorted(findings, key=lambda f: f.line)


def _counted(path: str | None) -> bool:
    return path is not None and PurePosixPath(path).name != '_errors.py'


def _growth(changes: list[Change]) -> list[_Growth]:
    growth: list[_Growth] = []
    for change in changes:
        if not _counted(change.new_path):
            continue
        old = _findings(change.old_source) if _counted(change.old_path) else []
        new = _findings(change.new_source)
        if len(new) <= len(old):
            continue
        before = Counter(f.text for f in old)
        added: list[_Finding] = []
        for finding in new:
            if before[finding.text]:
                before[finding.text] -= 1
            else:
                added.append(finding)
        growth.append(_Growth(change.new_path or '', len(old), len(new), added))
    return growth


def _check(options: Options) -> list[_Growth]:
    return _growth(read_changes(options, ROOTS))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check the SDK status-error handlers a change adds to the model adapters.')
    growth = _check(options)
    theirs = merge_head() if options.staged else None
    if theirs:
        growth = worse_than_both(
            growth, _check(replace(options, base=theirs)), key=lambda g: g.path, before=lambda g: g.old
        )
    for g in growth:
        print(f'{g.path}: {g.old} -> {g.new} SDK status-error handlers without `HTTPStatusError`')
        for f in g.added:
            print(f'    {g.path}:{f.line}: {f.text}')
            if options.github:
                title = escape(f'Status-error handlers {g.old} -> {g.new}', prop=True)
                print(f'::error file={escape(g.path, prop=True)},line={f.line},title={title}::{escape(ADVICE)}')
    if growth:
        print(f'\n{ADVICE}.')
    return 1 if growth else 0


if __name__ == '__main__':
    sys.exit(main())
