"""Fail when a change adds or edits a hard rule in an agent instruction without naming what enforces it.

A hard rule is a list item, or a sentence outside a list, that says must, never, always, do not or don't. It ends
with an enforcer tag: [gate: `<path>`] names the check that fails a change breaking the rule, and `[review]`
says only a reviewer can. The tag decides where the rule's enforcement lives, at the moment someone writes the rule.

Only rules a change adds or edits need a tag; rules the base already has pass until someone edits their prose. An edit
inside code spans alone, such as a moved path, is not a new rule, and a rule that moves between two changed files keeps
its standing. During a merge, `--staged` reports a rule only when it is new
against both parents. The base-ref comparison is `scripts/_ratchet.py`.

Usage:
    uv run scripts/check_enforcer_tags.py --staged                  # the index against HEAD, for pre-commit
    uv run scripts/check_enforcer_tags.py --base REF                # the working tree against REF
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from _ratchet import Change, Options, escape, merge_head, parse_args, read_changes, worse_than_both

GUIDE = 'AGENTS.md#coding-guidelines'
ROOTS = ('AGENTS.md', 'agent_docs', 'tests/AGENTS.md', '.agents/skills', '.claude/skills')
# Under the skill roots only the skill itself states rules; templates and helpers do not.
_SKILL_ROOTS = ('.agents/skills/', '.claude/skills/')

_FENCE = re.compile(r'^[ \t]*(`{3,}|~{3,})')
_LIST_ITEM = re.compile(r'^[ \t]*(?:[-*+]|\d+[.)])[ \t]+')
_SKIPPED_LINE = re.compile(r'^[ \t]*(?:#|\||<!--.*-->[ \t]*$)')
_TAG = re.compile(r'[ \t]*\[(?:review|gate: `(?P<gate>[^`\n]+)`)\][ \t]*$')
_ANY_TAG = re.compile(r'[ \t]*\[(?:review|gate\b)[^\]\n]*\][ \t]*$')
_SENTENCE_END = re.compile(r'[.!?:;]["\')*_`]*[ \t]*$')
_CODE_SPAN = re.compile(r'(`+)(?:(?!\n[ \t]*\n)[\s\S])+?\1')
_HARD = re.compile(r"\b(?:must|never|always|do not|don't)\b", re.IGNORECASE)


@dataclass
class Rule:
    """A list item, or a sentence outside a list, that may state a rule."""

    line: int
    """The unit's last line, where its tag goes."""
    text: str


@dataclass
class Finding:
    """A new hard rule whose enforcer tag is missing or does not resolve."""

    path: str
    line: int
    message: str


def units(source: str) -> list[Rule]:
    """Split Markdown into list items and sentences outside lists, skipping code, headings, tables and comments."""
    lines = source.split('\n')
    start = 0
    if lines and lines[0] == '---':
        start = next((i + 1 for i, line in enumerate(lines[1:], 1) if line == '---'), 0)
    found: list[Rule] = []
    current: list[str] = []
    in_list = False
    fence: str | None = None

    def close(end: int) -> None:
        if current:
            found.append(Rule(end, '\n'.join(current)))
            current.clear()

    for number, line in enumerate(lines[start:], start + 1):
        if fence:
            fence = None if line.strip().startswith(fence) else fence
            continue
        if match := _FENCE.match(line):
            close(number - 1)
            fence = match[1]
            continue
        if not line.strip() or _SKIPPED_LINE.match(line):
            close(number - 1)
            in_list = False
            continue
        if _LIST_ITEM.match(line):
            close(number - 1)
            in_list = True
        elif current and not in_list and _SENTENCE_END.search(_ANY_TAG.sub('', current[-1])):
            close(number - 1)
        current.append(line)
    close(len(lines))
    return found


def _key(rule: Rule) -> str:
    # An edit inside code spans, such as a moved path or a renamed identifier, leaves the rule's layer as it was.
    return ' '.join(_CODE_SPAN.sub('<code>', _ANY_TAG.sub('', rule.text)).split())


def _is_hard(rule: Rule) -> bool:
    return bool(_HARD.search(_CODE_SPAN.sub('', rule.text)))


def _problem(rule: Rule) -> str | None:
    """Why a new hard rule's tag does not pass, or `None` when it does."""
    tail = rule.text.rsplit('\n', 1)[-1]
    if tag := _TAG.search(tail):
        gate = tag['gate']
        if gate is None or Path(gate).exists():
            return None
        return f'the gate `{gate}` does not exist'
    if _ANY_TAG.search(tail):
        return 'write the tag as `[review]` or [gate: `<path>`], with the path in backticks'
    return 'end it with [gate: `<path>`], naming the check that fails a change breaking it, or `[review]`'


def findings(changes: Sequence[Change]) -> list[Finding]:
    """The hard rules the changes add or edit without a passing enforcer tag."""
    checked = [
        c
        for c in changes
        if c.new_path
        and not os.path.islink(c.new_path)
        and (not c.new_path.startswith(_SKILL_ROOTS) or c.new_path.endswith('/SKILL.md'))
    ]
    before = {_key(rule) for c in changes if c.old_source for rule in units(c.old_source)}
    found: list[Finding] = []
    for change in checked:
        assert change.new_path and change.new_source is not None
        for rule in units(change.new_source):
            if _is_hard(rule) and _key(rule) not in before and (problem := _problem(rule)):
                found.append(Finding(change.new_path, rule.line, problem))
    return found


def _check(options: Options) -> list[Finding]:
    return findings(read_changes(options, ROOTS, suffix='.md'))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check that the hard rules a change adds to agent instructions name their enforcer.')
    found = _check(options)
    theirs = merge_head() if options.staged else None
    if theirs:
        found = worse_than_both(
            found, _check(replace(options, base=theirs)), key=lambda f: (f.path, f.message), before=lambda f: f.line
        )
    for f in found:
        if options.github:
            title = escape('Enforcer tag', prop=True)
            print(f'::error file={escape(f.path, prop=True)},line={f.line},title={title}::{escape(f.message)}')
        else:
            print(f'{f.path}:{f.line}: hard rule without an enforcer tag: {f.message}')
    if found:
        print(f'See {GUIDE} for when a rule is a gate and when it is review.')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
