"""Guard the invariant that makes agent-run spans report the usage their own run produced.

`_usage_attribution` credits an increment to the run that made it, which is what lets agent-run
spans be summed without counting a nested run twice. A `RunUsage` field incremented directly still
reaches the caller's total and the usage limits, so nothing fails loudly — the tokens just go
missing from the run's span. That is invisible until someone sums spans, so it is guarded here
rather than left to review.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from pydantic_ai import _usage_attribution
from pydantic_ai.usage import RequestUsage, RunUsage

PACKAGE_ROOT = Path(_usage_attribution.__file__).parent

# `requests`/`tool_calls` bumped in place, or tokens folded in with `incr`.
BARE_MUTATION = re.compile(r'\.(?:requests|tool_calls)\s*\+=|\.incr\(')

# A line that mutates something other than a live run's usage says so, and why, rather than being
# waived by path: the files holding the real sites are also where a new one is most likely to land.
MARKER = '# usage-attribution: '

# `usage.py` defines the mutating API and `_usage_attribution.py` is its one caller.
OWNERS = {Path('usage.py'), Path('_usage_attribution.py')}


def find_bare_mutations(root: Path) -> list[str]:
    """Return `path:line: source` for each unmarked in-place usage mutation under `root`."""
    offenders: list[str] = []
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if relative in OWNERS:
            continue
        lines = path.read_text().splitlines()
        for number, line in enumerate(lines, start=1):
            # The marker sits on the line, or just above it when that would overrun the line length.
            preceding = lines[number - 2] if number > 1 else ''
            if BARE_MUTATION.search(line) and MARKER not in line and MARKER not in preceding:
                offenders.append(f'{relative}:{number}: {line.strip()}')
    return offenders


def test_no_bare_run_usage_mutation_outside_the_recorder() -> None:
    """Every increment of a live run's usage must go through `_usage_attribution`."""
    assert find_bare_mutations(PACKAGE_ROOT) == [], (
        "Increment a run's usage through `_usage_attribution.record_*` so it is also credited to "
        "the run that produced it; a bare increment leaves the tokens off that run's span. If the "
        f"target is not a live run's usage, say so with `{MARKER}<reason>` on the line. Offending "
        'lines:\n' + '\n'.join(find_bare_mutations(PACKAGE_ROOT))
    )


@pytest.mark.parametrize(
    'source,expected',
    [
        ('usage.requests += 1', ['probe.py:1: usage.requests += 1']),
        ('usage.tool_calls += 1', ['probe.py:1: usage.tool_calls += 1']),
        ('usage.incr(response.usage)', ['probe.py:1: usage.incr(response.usage)']),
        (f'usage.requests += 1  {MARKER}a copy, for a check only', []),
        (f'{MARKER}a copy, for a check only\nusage.requests += 1', []),
        ('usage.requests = 1', []),
    ],
    ids=['requests', 'tool_calls', 'incr', 'marked-inline', 'marked-above', 'not-in-place'],
)
def test_the_guard_catches_what_it_claims_to(tmp_path: Path, source: str, expected: list[str]) -> None:
    """The guard is only worth having if it fires, so prove it does before trusting a clean run."""
    (tmp_path / 'probe.py').write_text(source + '\n')
    assert find_bare_mutations(tmp_path) == expected


def _record_usage(usage: RunUsage) -> None:
    _usage_attribution.record_usage(usage, RequestUsage(input_tokens=3, output_tokens=1))


@pytest.mark.parametrize(
    'record,expected',
    [
        (_usage_attribution.record_request, RunUsage(requests=1)),
        (_usage_attribution.record_tool_call, RunUsage(tool_calls=1)),
        (_record_usage, RunUsage(input_tokens=3, output_tokens=1)),
    ],
    ids=['request', 'tool_call', 'usage'],
)
def test_record_credits_the_target_and_the_innermost_run(
    record: Callable[[RunUsage], None], expected: RunUsage
) -> None:
    """Each `record_*` reaches the run's own usage and the run that produced it, and no other.

    A nested run's requests belong to the nested run; the enclosing one reports its own, so that
    summing both spans gives the total instead of counting the inner run twice.
    """
    shared = RunUsage()
    outer = RunUsage()
    inner = RunUsage()

    with _usage_attribution.accumulate(outer):
        with _usage_attribution.accumulate(inner):
            record(shared)
        # Crediting is handed back to the enclosing run when the nested one ends.
        record(shared)

    assert shared == expected + expected
    assert inner == expected
    assert outer == expected


def test_record_outside_any_run_only_touches_the_target() -> None:
    """With no span open there is nothing to attribute to, and the target is still incremented."""
    usage = RunUsage()
    _usage_attribution.record_request(usage)
    assert usage == RunUsage(requests=1)


def test_accumulators_do_not_leak_to_siblings() -> None:
    """A sibling's accumulator is not the one credited here, which is the whole point."""
    first = RunUsage()
    second = RunUsage()

    with _usage_attribution.accumulate(first):
        _usage_attribution.record_request(RunUsage())
    with _usage_attribution.accumulate(second):
        _usage_attribution.record_usage(RunUsage(), RequestUsage(input_tokens=7))

    assert first == RunUsage(requests=1)
    assert second == RunUsage(input_tokens=7)
