"""A gate's result as a check run: its conclusion, summary and annotations.

The gates are advisory. A finding, an unsure item, an item not judged or anything left out concludes
`neutral`; only a fully judged gate with nothing to report concludes `success`. Text from the pull
request is shown without links or mentions, so a check can neither ping anyone nor point anywhere.
"""

from __future__ import annotations

import re

from .gates import FLAG_BAR, GATES, MODEL, GateResult, Judgement
from .github import Annotation, CheckOutput, Conclusion

# The API takes at most 50 annotations a request, and a summary of at most 65,535 characters.
_ANNOTATION_LIMIT = 50
_SUMMARY_CHARS = 60_000
_LABEL_CHARS = 300
_MARKDOWN_LINK = re.compile(r'\[([^\]]*)\]\([^)]*\)')
_URL = re.compile(r'\b(?:https?|ftp)://\S+|\bwww\.\S+', re.IGNORECASE)
_MENTION = re.compile(r'@(?=[\w-])')


def sanitize(text: str) -> str:
    """`text` with links and URLs removed, mentions defused, and backticks dropped so it fits in a code span."""
    text = _URL.sub('<link removed>', _MARKDOWN_LINK.sub(r'\1', text))
    return _MENTION.sub('@\N{ZERO WIDTH SPACE}', text).replace('`', "'")


def conclusion(result: GateResult) -> Conclusion:
    """`success` only for a fully judged gate whose every item is clean."""
    clean = all(j.outcome == 'clean' for j in result.judgements)
    return 'success' if clean and not result.notes else 'neutral'


def check_output(result: GateResult) -> CheckOutput:
    """The title, summary and annotations of a gate's check run."""
    gate = GATES[result.gate]
    by_outcome = {
        outcome: [j for j in result.judgements if j.outcome == outcome]
        for outcome in ('finding', 'unsure', 'error', 'declined')
    }
    findings, unsure = by_outcome['finding'], by_outcome['unsure']
    not_judged = by_outcome['error'] + by_outcome['declined']
    title = (
        f'{len(findings)} finding(s), {len(unsure)} unsure, of {len(result.judgements)} item(s)'
        if findings or unsure
        else f'{len(result.judgements)} item(s), none flagged'
    )
    lines = [
        f'Advisory: this check never blocks a merge. Each item was judged by `{MODEL}`; it is a finding when every '
        f'answer gives the defect at a confidence of {FLAG_BAR} or more, and unsure below that.',
    ]
    for heading, judgements in (('Findings', findings), ('Unsure', unsure), ('Not judged', not_judged)):
        if judgements:
            lines += ['', f'### {heading} ({len(judgements)})', *(_line(gate.defect, j) for j in judgements)]
    if result.notes:
        lines += ['', '### Not covered', *(f'- {sanitize(note)}' for note in result.notes)]
    summary = '\n'.join(lines)
    if len(summary) > _SUMMARY_CHARS:
        summary = summary[:_SUMMARY_CHARS] + '\n\n(summary cut at the API limit)'
    annotations: list[Annotation] = [
        {
            'path': j.item.location.path,
            'start_line': j.item.location.start,
            'end_line': j.item.location.end,
            'annotation_level': 'warning',
            'title': gate.name,
            'message': sanitize(j.item.label)[:_LABEL_CHARS],
        }
        for j in findings[:_ANNOTATION_LIMIT]
    ]
    return {'title': title, 'summary': summary, 'annotations': annotations}


def _line(defect: dict[str, bool], judgement: Judgement) -> str:
    location = judgement.item.location
    where = f'`{sanitize(location.path)}:{location.start}-{location.end}`'
    if judgement.error is not None:
        return f'- {where} {judgement.outcome}: {sanitize(judgement.error)[:_LABEL_CHARS]}'
    answers = ', '.join(
        f'`{name}` {p >= 0.5} at {abs(p - 0.5) * 2:.2f}{" (defect)" if (p >= 0.5) == defect[name] else ""}'
        for name, p in judgement.yes.items()
    )
    return f'- {where} `{sanitize(judgement.item.label)[:_LABEL_CHARS]}`: {answers}'
