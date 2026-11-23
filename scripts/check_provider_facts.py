"""Fail when a change adds a name to a registered provider fact set without a `# fact:` tag for its recorded probe.

A registered set, listed in `REGISTERED`, is a module-level set, tuple or dict under `pydantic_ai/providers/`,
`profiles/` or `models/` that writes a provider fact by name: an exclusion or denylist, a mandatory-reasoning table,
an error-code map or an exact-name allowlist. Every string written in its value is a member. A member is tagged when
its line carries `# fact: <id>`, at its end or alone on the line above, and `<id>` names a `ProviderFact` row in
`tests/providers/facts/registry.py`, whose recorded probe proves the fact (proposal R32). Every other member is
untagged, and a change may not raise a file's count of untagged members. What the base already has is allowed, so
existing members pass until a change adds to them.

A member whose name the file had as often before, or that the change removed from another changed file, moved rather
than was added: renaming a set, or merging two sets into one, adds nothing. Each `# fact:` tag the change adds that
names no row fails.

During a merge, `--staged` reports a file only when its count is higher than in both parents.

Usage:
    uv run scripts/check_provider_facts.py --staged                 # the index against HEAD, for pre-commit
    uv run scripts/check_provider_facts.py --base REF               # the working tree against REF
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from _ratchet import Change, Options, escape, git, merge_head, parse_args, read_changes, worse_than_both

FACTS = 'tests/providers/facts/registry.py'
_PACKAGE = 'pydantic_ai_slim/pydantic_ai'
ROOTS = (f'{_PACKAGE}/providers', f'{_PACKAGE}/profiles', f'{_PACKAGE}/models')

REGISTERED: dict[str, frozenset[str]] = {
    f'{_PACKAGE}/models/_anthropic_schema_limits.py': frozenset({'_SCHEMA_TOO_COMPLEX_MESSAGES'}),
    f'{_PACKAGE}/models/anthropic/_errors.py': frozenset(
        {'ERROR_TYPE_STATUS_CODES', '_ERROR_TYPE_CATEGORIES', '_THREAD_REPLAY_ERROR_CODES'}
    ),
    f'{_PACKAGE}/models/bedrock.py': frozenset({'_STREAM_EXCEPTION_STATUS_CODES'}),
    f'{_PACKAGE}/models/openai/_model_names.py': frozenset({'DEPRECATED_OPENAI_MODELS'}),
    f'{_PACKAGE}/profiles/google.py': frozenset({'_MODEL_THINKING_LEVELS', '_REALTIME_MODEL_THINKING_LEVELS'}),
    f'{_PACKAGE}/profiles/grok.py': frozenset(
        {
            '_GROK_3_MINI_DISABLEABLE_MODELS',
            '_GROK_43_REASONING_MODELS',
            '_GROK_45_REASONING_MODELS',
            '_GROK_47_REASONING_MODELS',
        }
    ),
    f'{_PACKAGE}/profiles/minimax.py': frozenset({'_ALWAYS_THINKING_MODELS', '_NON_THINKING_MODELS'}),
    f'{_PACKAGE}/profiles/nvidia.py': frozenset({'_HYBRID_THINKING_PREFIXES', '_OPT_IN_THINKING_PREFIXES'}),
    f'{_PACKAGE}/profiles/openai.py': frozenset({'_REASONING_SUPPORT_BY_PREFIX'}),
    f'{_PACKAGE}/profiles/thinkingmachines.py': frozenset({'_HYBRID_THINKING_MODELS'}),
    f'{_PACKAGE}/profiles/zai.py': frozenset({'_ALWAYS_THINKING_MODEL_PREFIXES', '_REASONING_EFFORT_MODEL_PREFIXES'}),
    f'{_PACKAGE}/providers/_bedrock_model_names.py': frozenset(
        {'_STRUCTURED_OUTPUT_UNSUPPORTED_CLAUDE_MODEL_PREFIXES'}
    ),
    f'{_PACKAGE}/providers/bedrock.py': frozenset({'_BEDROCK_CONVERSE_GPT_MODELS'}),
    f'{_PACKAGE}/providers/mistral.py': frozenset({'_ADJUSTABLE_REASONING_MODELS', '_LARGE_4_MODELS'}),
    f'{_PACKAGE}/providers/moonshotai.py': frozenset({'_THINKING_FALSE_BODY_BY_PREFIX'}),
    f'{_PACKAGE}/providers/openrouter.py': frozenset(
        {
            '_FORCED_TOOL_CHOICE_REJECTED_WHILE_REASONING_MODEL_PREFIXES',
            '_LOW_REASONING_FLOOR_MODEL_PREFIXES',
            '_MANDATORY_REASONING_MODEL_PREFIXES',
        }
    ),
    f'{_PACKAGE}/providers/vercel.py': frozenset({'_OPEN_WEIGHT_REASONING_EFFORTS'}),
}
"""The registered sets, by the module that defines them."""

_ADVICE = (
    'a hand-written provider fact needs a recorded probe: record a test that shows the provider behaving as the set '
    f'says, add a `ProviderFact` row naming it to `{FACTS}`, and mark the line with `# fact: <id>`'
)

_TAG = re.compile(r'#[ \t]*fact[ \t]*:[ \t]*(?P<id>\S*)')


@dataclass
class _Member:
    line: int
    name: str
    symbol: str


@dataclass
class _Tag:
    line: int
    """The line it marks: its own, or the next one for a comment alone on its line."""
    fact: str
    text: str
    """The comment's source line, stripped, which pairs it across the two sides of a change."""


@dataclass
class _Scan:
    untagged: list[_Member] = field(default_factory=list[_Member])
    strings: Counter[str] = field(default_factory=Counter[str])
    """Every string constant in the module."""
    tags: list[_Tag] = field(default_factory=list[_Tag])
    members: Counter[str] = field(default_factory=Counter[str])
    """Every member name of the module's registered sets, tagged or not."""


@dataclass
class _Growth:
    path: str
    old: int
    new: int
    added: list[_Member]
    """The untagged members the change added to this file, without those that moved."""


def _fact_ids(source: str) -> frozenset[str]:
    """The `id` of each `ProviderFact(...)` row in the registry module's `source`."""
    return frozenset(
        keyword.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ProviderFact'
        for keyword in node.keywords
        if keyword.arg == 'id' and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str)
    )


def _members(tree: ast.Module, symbols: frozenset[str]) -> list[_Member]:
    """Every string written in the value of a module-level assignment to one of `symbols`."""
    found: list[_Member] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in symbols:
                found += [
                    _Member(n.lineno, n.value, target.id)
                    for n in ast.walk(value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]
    return found


def _tags(source: str) -> list[_Tag]:
    found: list[_Tag] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        marker = _TAG.match(token.string) if token.type == tokenize.COMMENT else None
        if marker:
            alone = not token.line[: token.start[1]].strip()
            # A comment alone on its line marks the next line.
            line = token.start[0] + 1 if alone else token.start[0]
            found.append(_Tag(line, marker['id'], token.line.strip()))
    return found


def _scan(source: str | None, path: str, facts: frozenset[str]) -> _Scan:
    if source is None:
        return _Scan()
    try:
        tree = ast.parse(source)
        found = _tags(source)
    except (SyntaxError, tokenize.TokenError):
        return _Scan()
    tagged = {t.line for t in found if t.fact in facts}
    members = _members(tree, REGISTERED.get(path, frozenset()))
    untagged = [m for m in members if m.line not in tagged]
    strings = Counter(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    return _Scan(untagged, strings, found, Counter(m.name for m in members))


def _file_growth(path: str, old: _Scan, new: _Scan, moves: Counter[str]) -> _Growth | None:
    """How far a change raised a file's untagged count, less the members that only moved.

    A member is new unless the file already had its name as often, or `moves`, the untagged members the change removed
    from the registered sets of changed files, holds it; each one taken from `moves` is used up.
    """
    gained = new.strings - old.strings
    paired = Counter(m.name for m in old.untagged)
    added: list[_Member] = []
    for m in new.untagged:
        if paired[m.name]:
            paired[m.name] -= 1
        elif gained[m.name]:
            gained[m.name] -= 1
            if moves[m.name]:
                moves[m.name] -= 1
            else:
                added.append(m)
    if not added or len(new.untagged) <= len(old.untagged):
        return None
    return _Growth(path, len(old.untagged), len(new.untagged), added)


def _added_tags(path: str, old: _Scan, new: _Scan) -> list[tuple[str, _Tag]]:
    before = Counter(t.text for t in old.tags)
    added: list[tuple[str, _Tag]] = []
    for t in new.tags:
        if before[t.text]:
            before[t.text] -= 1
        else:
            added.append((path, t))
    return added


def _growth(changes: list[Change], facts: frozenset[str]) -> tuple[list[_Growth], list[tuple[str, _Tag]]]:
    """Each file a change raised the untagged count of, and each `# fact:` tag it added."""
    scans = [
        (c, _scan(c.old_source, c.old_path or '', facts), _scan(c.new_source, c.new_path or '', facts)) for c in changes
    ]
    # Only an untagged name that leaves its file's registered sets can move into another file's. A name removed from
    # anywhere else, or one that stays but gains a tag, has not moved.
    moves: Counter[str] = Counter()
    for _, old, new in scans:
        moves.update(Counter(m.name for m in old.untagged) - new.members)
    growth: list[_Growth] = []
    added_tags: list[tuple[str, _Tag]] = []
    for change, old, new in scans:
        path = change.new_path or ''
        added_tags += _added_tags(path, old, new)
        grown = _file_growth(path, old, new, moves)
        if grown:
            growth.append(grown)
    return growth, added_tags


def _print_growth(g: _Growth, github: bool) -> None:
    print(f'{g.path}: untagged {g.old} -> {g.new}')
    for m in g.added:
        print(f'    {g.path}:{m.line}: {m.name!r} in {m.symbol}')
        if github:
            title = escape(f'Provider facts: untagged {g.old} -> {g.new}', prop=True)
            print(f'::error file={escape(g.path, prop=True)},line={m.line},title={title}::{escape(_ADVICE)}')


def _report_tags(added: list[tuple[str, _Tag]], facts: frozenset[str], github: bool) -> bool:
    """Print each `# fact:` tag the change added that names no row; return whether there is one."""
    unknown = [(path, t) for path, t in added if t.fact not in facts]
    for path, t in unknown:
        message = f'`# fact: {t.fact}` names no `ProviderFact` row in {FACTS}'
        print(f'{path}:{t.line}: {message}')
        if github:
            print(f'::error file={escape(path, prop=True)},line={t.line},title=Provider facts::{escape(message)}')
    return bool(unknown)


def _check(options: Options, facts: frozenset[str]) -> tuple[list[_Growth], list[tuple[str, _Tag]]]:
    return _growth(read_changes(options, ROOTS), facts)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check the provider fact set members a change adds have a recorded probe.')
    # The registry is read from the same side as the change, so a row and its first tag can land together.
    try:
        registry = git('show', f':{FACTS}').decode() if options.staged else Path(FACTS).read_text(encoding='utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        registry = ''
    facts = _fact_ids(registry)
    growth, added_tags = _check(options, facts)
    theirs = merge_head() if options.staged else None
    if theirs:
        their_growth, _ = _check(replace(options, base=theirs), facts)
        growth = worse_than_both(growth, their_growth, key=lambda g: g.path, before=lambda g: g.old)
    unknown = _report_tags(added_tags, facts, options.github)
    for g in growth:
        _print_growth(g, options.github)
    if growth:
        print(f'\n{_ADVICE}.')
    return 1 if growth or unknown else 0


if __name__ == '__main__':
    sys.exit(main())
