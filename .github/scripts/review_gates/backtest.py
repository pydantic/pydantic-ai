"""Judge the gates on a labelled set of historical PR heads, and report where they would have stood.

The set holds the CI Review findings that were real bugs, each at the head it was posted on, and PRs
CI Review approved with no finding. Nothing is posted. Every item's probabilities are written out, so a
bar can be picked offline: an item flagged on a clean PR is judged by hand, not counted as a false alarm.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from pydantic_ai.models import infer_model

from .gates import FLAG_BAR, GATES, MODEL, evaluate
from .github import GitHub
from .prep import prepare
from .reach import reach_map

# A finding counts as located when an item covers its line, give or take this many lines.
_LINE_SLACK = 5
_MARGIN_BUCKETS = ((-1.0, -FLAG_BAR), (-FLAG_BAR, 0.0), (0.0, FLAG_BAR), (FLAG_BAR, 1.0))


class LabelledFinding(BaseModel):
    """A CI Review finding that was a real bug, at the head it was posted on."""

    idx: int
    pr: int
    head: str
    comment: str
    path: str
    line: int | None
    summary: str


class CleanHead(BaseModel):
    """A PR head CI Review approved with no finding."""

    pr: int
    head: str


class BacktestSet(BaseModel):
    """The labelled set, as `backtest_set.json` stores it."""

    window: list[str]
    findings: list[LabelledFinding]
    clean: list[CleanHead]


class Row(BaseModel):
    """One judged item."""

    pr: int
    head: str
    clean_pr: bool
    gate: str
    path: str
    start: int
    end: int
    label: str
    outcome: str
    yes: dict[str, float]
    # How far the answers lean toward the defect: the least of each field's margin, signed toward it.
    defect_margin: float | None
    # The labelled findings this item, or the sibling it is judged against, covers.
    finding_idx: list[int]
    error: str | None = None


async def backtest(set_path: Path, out: Path, token: str, root: Path) -> int:
    """Judge every head in the set, write each item to `out` as JSONL and the report beside it."""
    labelled = BacktestSet.model_validate_json(set_path.read_text(encoding='utf-8'))
    github = GitHub.connect('pydantic/pydantic-ai', token)
    model = infer_model(MODEL)
    reach = reach_map(root)
    heads = {(f.pr, f.head): False for f in labelled.findings} | {(c.pr, c.head): True for c in labelled.clean}
    rows: list[Row] = []
    try:
        for (number, head), clean in heads.items():
            prepared = await prepare(github, await github.pull_request(number), head)
            for result in await evaluate(prepared, model, root=root, reach=reach, keys=set(GATES)):
                defect = GATES[result.gate].defect
                for j in result.judgements:
                    location = j.item.location
                    spans = [s for s in (location, j.item.sibling) if s is not None]
                    covered = [
                        f.idx
                        for f in labelled.findings
                        if (f.pr, f.head) == (number, head)
                        and f.line is not None
                        and any(
                            s.path == f.path and s.start - _LINE_SLACK <= f.line <= s.end + _LINE_SLACK for s in spans
                        )
                    ]
                    margin = (
                        min((p - 0.5) * 2 * (1 if defect[name] else -1) for name, p in j.yes.items()) if j.yes else None
                    )
                    rows.append(
                        Row(
                            pr=number,
                            head=head,
                            clean_pr=clean,
                            gate=result.gate,
                            path=location.path,
                            start=location.start,
                            end=location.end,
                            label=j.item.label[:300],
                            outcome=j.outcome,
                            yes=j.yes,
                            defect_margin=margin,
                            finding_idx=covered,
                            error=j.error,
                        )
                    )
            print(f'PR #{number} at {head[:12]}: {sum(r.pr == number and r.head == head for r in rows)} item(s)')
    finally:
        await github.aclose()
    out.write_text(''.join(row.model_dump_json() + '\n' for row in rows), encoding='utf-8')
    report = summarize(labelled, rows)
    out.with_suffix('.md').write_text(report, encoding='utf-8')
    print(report)
    return 0


def summarize(labelled: BacktestSet, rows: list[Row]) -> str:
    """Per-gate counts, where each labelled finding landed, and the spread of defect margins."""
    lines = [
        f'Bar {FLAG_BAR}, model `{MODEL}`.',
        '',
        '| gate | items | findings | unsure | not judged | flagged on clean PRs |',
    ]
    lines.append('|---|---|---|---|---|---|')
    for key in GATES:
        gate_rows = [r for r in rows if r.gate == key]
        counts = Counter(r.outcome for r in gate_rows)
        flagged_clean = sum(r.outcome == 'finding' and r.clean_pr for r in gate_rows)
        not_judged = counts['error'] + counts['declined']
        lines.append(
            f'| `{key}` | {len(gate_rows)} | {counts["finding"]} | {counts["unsure"]} | {not_judged} | {flagged_clean} |'
        )
    lines += ['', '| finding | summary | located by | outcome |', '|---|---|---|---|']
    for finding in labelled.findings:
        covering = [r for r in rows if finding.idx in r.finding_idx]
        located = ', '.join(sorted({r.gate for r in covering})) or 'not located'
        outcomes = ', '.join(sorted({r.outcome for r in covering})) or '-'
        lines.append(f'| [{finding.idx}]({finding.comment}) | {finding.summary} | {located} | {outcomes} |')
    lines += ['', 'Spread of the defect margin (signed toward the defect; a finding needs at least the bar):', '']
    lines += ['| gate | items on | ' + ' | '.join(f'[{lo:+.1f}, {hi:+.1f})' for lo, hi in _MARGIN_BUCKETS) + ' |']
    lines.append('|---|---|' + '---|' * len(_MARGIN_BUCKETS))
    for key in GATES:
        for clean, name in ((False, 'finding PRs'), (True, 'clean PRs')):
            margins = [
                r.defect_margin for r in rows if r.gate == key and r.clean_pr == clean and r.defect_margin is not None
            ]
            buckets = [sum(lo <= m < hi or (hi == 1.0 and m == 1.0) for m in margins) for lo, hi in _MARGIN_BUCKETS]
            lines.append(f'| `{key}` | {name} | ' + ' | '.join(map(str, buckets)) + ' |')
    flagged = [r for r in rows if r.outcome == 'finding' and r.clean_pr]
    if flagged:
        lines += ['', 'Flagged on clean PRs, to judge by hand:', '']
        lines += [f'- #{r.pr} `{r.path}:{r.start}-{r.end}` ({r.gate}): {json.dumps(r.label)}' for r in flagged]
    return '\n'.join(lines) + '\n'
