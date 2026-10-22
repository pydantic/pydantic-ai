"""The three launch gates: the items each one judges, and the decision question it asks of every item.

A decision model judges an item; it does not search for one. Scripts reduce the pull request to items,
each a short text with one or two yes/no questions, and an item is a finding only when the model gives
the defect answer at or above `FLAG_BAR` on every question.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter

from pydantic_ai import Agent, BoolCriteria, Choices
from pydantic_ai.exceptions import ContentFilterError, ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.models import Model

from .locators import Location, locate
from .prep import Function, Prepared, SourceFile, python_functions
from .reach import reach_of

MODEL = 'openai-decisions:gpt-6-luna'
# The starting bar from the design; it is tuned from the `decide` spans of the advisory week.
FLAG_BAR = 0.6
ITEM_CAP = 40
_CLAIM_CAP = 80
# About 6k tokens (an estimate of four characters a token): beyond it an item is cut and says so.
_ITEM_CHARS = 24_000
_DECISION_CONCURRENCY = 16
_FRAMING = 'Pull requests to Pydantic AI, an open-source Python agent framework.'

GateKey = Literal['pr-claims', 'unaffected-users', 'sibling-handling']
Outcome = Literal['finding', 'unsure', 'clean', 'error', 'declined']
T = TypeVar('T', bound=BaseModel)
AnswerT = TypeVar('AnswerT')


class ClaimCheck(BaseModel):
    """Check a pull request's claim against the code it changed."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    holds: Annotated[
        bool,
        BoolCriteria(
            true='For every input the claim names, the changed function returns or does what the claim says.',
            false='For at least one input the claim names, the changed function returns or does something else.',
        ),
    ]
    """Does the changed function do what the claim says?"""


class SharedChange(BaseModel):
    """Judge whether a change made for one surface moves behavior on others."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    alters_results: Annotated[
        bool,
        BoolCriteria(
            true='Code that used this before the change now gets a different result, error, limit or default.',
            false='Existing callers get the same results; the change only adds, refactors, or renames private code.',
        ),
    ]
    """Does this change alter what existing callers get?"""

    confined_to_target: Annotated[
        bool,
        BoolCriteria(
            true='The changed code only runs for the provider, model or feature the pull request is about.',
            false='The changed code also runs for providers, models or features the pull request is not about.',
        ),
    ]
    """Is this change confined to what the pull request is about?"""


class SiblingGap(BaseModel):
    """Judge a difference between two pieces of code that should handle the same thing."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    needs_it: Annotated[
        bool,
        BoolCriteria(
            true=(
                'Something the sibling handles reaches this code and goes wrong here: an error is lost, '
                'a value falls through, or state is left stale.'
            ),
            false=(
                'This code does not need it: that input or event never reaches it, or a caller or default '
                'already covers it.'
            ),
        ),
    ]
    """Does this code need the handling its sibling has?"""


@dataclass(frozen=True)
class Gate(Generic[T]):
    """One check run: the question asked of each item, and the answers that mark a defect."""

    key: GateKey
    name: str
    question: type[T]
    instructions: str
    # The answer to each field that marks a defect; an item is a finding only when every field gives it.
    defect: dict[str, bool]
    answers: Callable[[T], dict[str, bool]]

    @property
    def fingerprint(self) -> str:
        """Changes whenever the question, the framing, the model or the bar does, so a changed gate re-runs."""
        asked = json.dumps([self.question.model_json_schema(), self.instructions, MODEL, FLAG_BAR], sort_keys=True)
        return hashlib.sha256(asked.encode()).hexdigest()[:12]


PR_CLAIMS = Gate(
    'pr-claims',
    'Gate: PR claims match the diff',
    ClaimCheck,
    f'{_FRAMING} The text is one claim from a pull request and the changed function it describes.',
    {'holds': False},
    lambda o: {'holds': o.holds},
)
UNAFFECTED_USERS = Gate(
    'unaffected-users',
    'Gate: no behavior change for unaffected users',
    SharedChange,
    f'{_FRAMING} The text is one change from a pull request, what the pull request is about, and who runs the '
    'changed code.',
    {'alters_results': True, 'confined_to_target': False},
    lambda o: {'alters_results': o.alters_results, 'confined_to_target': o.confined_to_target},
)
SIBLING_HANDLING = Gate(
    'sibling-handling',
    'Gate: sibling code handled alike',
    SiblingGap,
    f'{_FRAMING} The text is code a pull request added or changed, sibling code that handles the same thing, '
    'and the difference a script found between them.',
    {'needs_it': True},
    lambda o: {'needs_it': o.needs_it},
)
GATES: dict[GateKey, Gate[ClaimCheck] | Gate[SharedChange] | Gate[SiblingGap]] = {
    g.key: g for g in (PR_CLAIMS, UNAFFECTED_USERS, SIBLING_HANDLING)
}


@dataclass(frozen=True)
class Item:
    """One thing to judge: where it is, a label for the check, and the text the model reads."""

    location: Location
    label: str
    text: str
    # Whether a section of `text` was cut to fit the item budget.
    cut: bool = False
    # Another place the item is about: a locator's sibling, or the docs lines a claim came from.
    related: Location | None = None


@dataclass(frozen=True)
class Judgement:
    """An item's outcome at the flag bar."""

    item: Item
    outcome: Outcome
    # The probability of yes to each field, as the model gave it; empty when no answer came back.
    yes: dict[str, float] = field(default_factory=dict[str, float])
    error: str | None = None


@dataclass
class GateResult:
    """Every judgement one gate made on one head."""

    gate: GateKey
    judgements: list[Judgement]
    # What the gate could not cover: capped items, truncated text, unasked questions.
    notes: list[str]


_CONFIDENCE = TypeAdapter(dict[str, float])


def classify(defect: dict[str, bool], yes: dict[str, float], bar: float = FLAG_BAR) -> Outcome:
    """`finding` when every field gives the defect answer at `bar` or above; `clean` when any is that sure of the other."""
    answers = {name: yes.get(name, 0.5) >= 0.5 for name in defect}
    sure = {name: abs(yes.get(name, 0.5) - 0.5) * 2 >= bar for name in defect}
    if any(answers[name] != defect[name] and sure[name] for name in defect):
        return 'clean'
    if all(answers[name] == defect[name] and sure[name] for name in defect):
        return 'finding'
    return 'unsure'


async def evaluate(
    prepared: Prepared, model: Model, *, root: Path, reach: dict[str, str], keys: set[GateKey]
) -> list[GateResult]:
    """Judge every item of each gate in `keys`."""
    semaphore = asyncio.Semaphore(_DECISION_CONCURRENCY)
    runs: list[asyncio.Task[GateResult]] = []
    async with asyncio.TaskGroup() as group:
        if 'pr-claims' in keys:
            runs.append(group.create_task(_claims_result(prepared, model, semaphore)))
        if 'unaffected-users' in keys:
            items = shared_items(prepared, reach)
            runs.append(group.create_task(_result(UNAFFECTED_USERS, model, semaphore, items, prepared.partial)))
        if 'sibling-handling' in keys:
            items = sibling_items(prepared, root)
            runs.append(group.create_task(_result(SIBLING_HANDLING, model, semaphore, items, prepared.partial)))
    return [run.result() for run in runs]


async def _claims_result(prepared: Prepared, model: Model, semaphore: asyncio.Semaphore) -> GateResult:
    items, notes = await claim_items(prepared, model, semaphore)
    return await _result(PR_CLAIMS, model, semaphore, items, [*prepared.partial, *notes])


async def _result(
    gate: Gate[T], model: Model, semaphore: asyncio.Semaphore, items: list[Item], notes: list[str]
) -> GateResult:
    if len(items) > ITEM_CAP:
        notes = [*notes, f'{len(items) - ITEM_CAP} item(s) over the cap of {ITEM_CAP} were not judged']
        items = items[:ITEM_CAP]
    if cut := sum(item.cut for item in items):
        notes = [*notes, f'{cut} item(s) were cut to fit the item budget of {_ITEM_CHARS} characters']
    agent = Agent(model, output_type=gate.question, instructions=gate.instructions)
    async with asyncio.TaskGroup() as group:
        judgements = [group.create_task(_judge(gate, agent, semaphore, item)) for item in items]
    return GateResult(gate.key, [j.result() for j in judgements], notes)


async def _judge(gate: Gate[T], agent: Agent[None, T], semaphore: asyncio.Semaphore, item: Item) -> Judgement:
    try:
        async with semaphore:
            result = await agent.run(item.text)
    except ContentFilterError as e:
        return Judgement(item, 'declined', error=str(e))
    except (ModelAPIError, UnexpectedModelBehavior) as e:
        return Judgement(item, 'error', error=str(e))
    answers = gate.answers(result.output)
    confidence = _CONFIDENCE.validate_python((result.response.provider_details or {}).get('confidence', {}))
    yes = {name: 0.5 + confidence.get(name, 0.0) / 2 * (1 if answer else -1) for name, answer in answers.items()}
    return Judgement(item, classify(gate.defect, yes), yes)


# --- M1: PR claims match the diff ---------------------------------------------------------------

_HTML_COMMENT = re.compile(r'<!--.*?-->', re.DOTALL)
_DETAILS = re.compile(r'<details>.*?</details>', re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r'^\s*```.*?^\s*```', re.DOTALL | re.MULTILINE)
_MARKDOWN_LINK = re.compile(r'\[([^\]]*)\]\([^)]*\)')
_URL = re.compile(r'https?://\S+')
_ISSUE_REF = re.compile(r'(?:\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+)?#\d+\b', re.IGNORECASE)
_SKIPPED_LINE = re.compile(r'^\s*(?:#|>|- \[[ xX]\]|\|)')
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z`])')
_IDENTIFIER = re.compile(r'`([^`]+)`|\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|[a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b')
_MIN_CLAIM_WORDS = 4
_TEST_PATH = re.compile(r'(^|/)tests?/|(^|/)test_[^/]*\.py$|_test\.py$')


@dataclass(frozen=True)
class _Claim:
    source: str
    text: str
    # Set for a docstring, which describes the function it belongs to.
    function: tuple[SourceFile, Function] | None = None
    # The docs lines a docs claim came from.
    lines: Location | None = None


async def claim_items(prepared: Prepared, model: Model, semaphore: asyncio.Semaphore) -> tuple[list[Item], list[str]]:
    """One item per behavior claim and the changed function it describes."""
    notes: list[str] = []
    # A claim describes library behavior, which a test function exercises but does not provide.
    changed = [
        (file, function)
        for file in prepared.files
        if not _TEST_PATH.search(file.path)
        for function in file.changed_functions()
    ]
    claims = _claims(prepared)
    if len(claims) > _CLAIM_CAP:
        notes.append(f'{len(claims) - _CLAIM_CAP} claim(s) over the cap of {_CLAIM_CAP} were not read')
        claims = claims[:_CLAIM_CAP]
    behavior = Agent(
        model,
        output_type=bool,
        instructions=(
            f'{_FRAMING} The text is one sentence from a pull request. Does it state what code does or how it behaves?'
        ),
    )
    # A docstring describes the function it belongs to, so it is neither filtered nor paired.
    async with asyncio.TaskGroup() as group:
        keeps = [group.create_task(_ask(behavior, semaphore, c.text)) for c in claims if c.function is None]
    kept = [task.result() for task in keeps]
    unasked = kept.count(None)
    pairs = [(c, *c.function) for c in claims if c.function is not None]
    to_pick: list[_Claim] = []
    for claim, keep in zip((c for c in claims if c.function is None), kept, strict=True):
        if not keep:
            continue
        if matched := _matching_functions(claim.text, changed):
            pairs.extend((claim, file, function) for file, function in matched)
        elif changed:
            to_pick.append(claim)
    if to_pick:
        options = {f'{file.path}::{function.qualname}': (file, function) for file, function in changed[:254]}
        described = {key: f'`{function.qualname}` in `{file.path}`' for key, (file, function) in options.items()}
        picker = Agent(
            model,
            output_type=Choices({**described, 'none': 'None of these functions.'}),
            instructions=f'{_FRAMING} The text is one claim from a pull request. Which changed function does it describe?',
        )
        async with asyncio.TaskGroup() as group:
            picks = [group.create_task(_ask(picker, semaphore, claim.text)) for claim in to_pick]
        for claim, task in zip(to_pick, picks, strict=True):
            unasked += (picked := task.result()) is None
            if picked in options:
                pairs.append((claim, *options[picked]))
    if unasked:
        notes.append(f'{unasked} claim(s) could not be classified or paired and were not judged')
    return [_claim_item(claim, file, function) for claim, file, function in pairs], notes


def _claims(prepared: Prepared) -> list[_Claim]:
    pull = prepared.pull_request
    claims = [_Claim('the PR title', pull.title)]
    claims += [_Claim('the PR description', sentence) for sentence in _sentences(pull.body or '')]
    for file in prepared.files:
        if file.path.startswith('docs/') and file.path.endswith('.md'):
            for start, end, added in _added_runs(file):
                where = Location(file.path, start, end)
                claims += [_Claim(f'the docs page `{file.path}`', s, lines=where) for s in _sentences(added)]
        if _TEST_PATH.search(file.path):
            continue
        base_docstrings = {
            f.qualname: ast.get_docstring(f.node)
            for f in (python_functions(file.base) if file.base is not None else [])
        }
        for function in file.changed_functions():
            docstring = ast.get_docstring(function.node)
            if docstring and docstring != base_docstrings.get(function.qualname):
                claims.append(_Claim(f'the docstring of `{function.qualname}`', docstring, (file, function)))
    return [c for c in claims if len(c.text.split()) >= _MIN_CLAIM_WORDS]


def _added_runs(file: SourceFile) -> list[tuple[int, int, str]]:
    """Each run of consecutive added lines, with its first and last head line."""
    runs: list[tuple[int, int, str]] = []
    for line in file.diff:
        if line.kind != '+':
            continue
        if runs and runs[-1][1] == line.new - 1:
            start, _, text = runs[-1]
            runs[-1] = (start, line.new, f'{text}\n{line.text}')
        else:
            runs.append((line.new, line.new, line.text))
    return runs


def _sentences(markdown: str) -> list[str]:
    """The prose sentences of `markdown`, without checklists, headings, quotes, tables, code, links or issue refs."""
    text = _FENCE.sub('', _DETAILS.sub('', _HTML_COMMENT.sub('', markdown)))
    text = _ISSUE_REF.sub('', _URL.sub('', _MARKDOWN_LINK.sub(r'\1', text)))
    sentences: list[str] = []
    for line in text.splitlines():
        if _SKIPPED_LINE.match(line):
            continue
        line = line.strip().removeprefix('- ').removeprefix('* ').strip()
        sentences += [s.strip() for s in _SENTENCE_END.split(line) if s.strip()]
    return sentences


async def _ask(agent: Agent[None, AnswerT], semaphore: asyncio.Semaphore, text: str) -> AnswerT | None:
    """The agent's answer about `text`, or `None` when it could not be asked."""
    try:
        async with semaphore:
            return (await agent.run(text)).output
    except (ModelAPIError, UnexpectedModelBehavior):
        return None


def _matching_functions(text: str, changed: list[tuple[SourceFile, Function]]) -> list[tuple[SourceFile, Function]]:
    named = {part for match in _IDENTIFIER.finditer(text) for part in re.split(r'\W+', match.group(0)) if part}
    return [(file, function) for file, function in changed if function.node.name in named][:3]


def _claim_item(claim: _Claim, file: SourceFile, function: Function) -> Item:
    text, cut = _sections(
        ('Claim', f'(from {claim.source}) {claim.text}'),
        ('Changed function', f'`{function.qualname}` in `{file.path}`\n{file.lines(function.start, function.end)}'),
        ('Change', file.hunk(function.start, function.end)),
    )
    location = Location(file.path, function.start, function.end)
    return Item(location, f'{claim.source}: {claim.text}', text, cut, claim.lines)


# --- M2: no behavior change for unaffected users ------------------------------------------------

_RANKED_REMOVAL = re.compile(r'^\s*(?:if|elif|return|raise)\b')


def shared_items(prepared: Prepared, reach: dict[str, str]) -> list[Item]:
    """One item per changed site in a shared file, the riskiest kinds of change first."""
    pull = prepared.pull_request
    about = '\n'.join(
        [
            pull.title,
            *(f'Label `{label}`.' for label in prepared.labels if label.startswith('pkg:')),
            ' '.join(_sentences(pull.body or '')),
            *(f'Linked issue #{i.number}: {i.title}. {" ".join(_sentences(i.body or ""))}' for i in prepared.issues),
        ]
    )
    ranked: list[tuple[int, Item]] = []
    for file in prepared.files:
        who = reach_of(file.path, reach)
        if who is None or file.head is None or not file.path.endswith('.py'):
            continue
        functions = file.head_functions()
        module_assignments = _module_assignment_lines(file.head)
        for start, end in _sites(file):
            enclosing = next((f for f in functions if f.start <= start <= f.end), None)
            context = (
                f'`{enclosing.qualname}`\n{file.lines(enclosing.start, enclosing.end)}'
                if enclosing
                else f'module level\n{file.lines(max(1, start - 5), end + 5)}'
            )
            text, cut = _sections(
                ('Pull request is about', about),
                ('Change', f'in `{file.path}`, {context}'),
                ('Hunk', file.hunk(start, end)),
                ('This code runs for', who),
            )
            item = Item(Location(file.path, start, end), f'change in `{file.path}`', text, cut)
            ranked.append((_rank(file, start, end, enclosing, module_assignments), item))
    return [item for _, item in sorted(ranked, key=lambda pair: pair[0])]


def _sites(file: SourceFile) -> list[tuple[int, int]]:
    """Runs of touched head lines, merged when fewer than four lines apart."""
    sites: list[tuple[int, int]] = []
    for line in sorted(file.touched):
        if sites and line - sites[-1][1] < 4:
            sites[-1] = (sites[-1][0], line)
        else:
            sites.append((line, line))
    return sites


def _module_assignment_lines(text: str) -> set[int]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    return {
        n
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.end_lineno is not None
        for n in range(node.lineno, node.end_lineno + 1)
    }


def _rank(file: SourceFile, start: int, end: int, enclosing: Function | None, constants: set[int]) -> int:
    """0 for a module constant or default argument, 1 for a removed branch or early exit, 2 for anything else."""
    if any(start <= n <= end for n in constants):
        return 0
    if enclosing and enclosing.node.body and start < enclosing.node.body[0].lineno:
        arguments = enclosing.node.args
        if arguments.defaults or any(d is not None for d in arguments.kw_defaults):
            return 0
    if any(_RANKED_REMOVAL.match(line.text) for line in file.diff if line.kind == '-' and start <= line.new <= end):
        return 1
    return 2


# --- M4: sibling code handled alike -------------------------------------------------------------


def sibling_items(prepared: Prepared, root: Path) -> list[Item]:
    """One item per pair the locators find."""
    items: list[Item] = []
    for pair in locate(prepared.files, root):
        text, cut = _sections(
            ('This code', f'`{pair.this.path}` lines {pair.this.start}-{pair.this.end}\n{pair.this_code}'),
            ('Sibling', f'`{pair.sibling.path}` lines {pair.sibling.start}-{pair.sibling.end}\n{pair.sibling_code}'),
            ('Difference found by a script', pair.difference),
        )
        items.append(Item(pair.this, f'locator {pair.locator}: {pair.difference}', text, cut, pair.sibling))
    return items


def _sections(*sections: tuple[str, str]) -> tuple[str, bool]:
    """Labelled sections within the item budget, and whether any had to be cut.

    Shorter sections are given what they need first, so only the longest are cut, to an equal share.
    """
    allowed: dict[str, int] = {}
    remaining = _ITEM_CHARS
    for left, (label, body) in enumerate(sorted(sections, key=lambda section: len(section[1]))):
        allowed[label] = min(len(body), remaining // (len(sections) - left))
        remaining -= allowed[label]
    text = '\n\n'.join(
        f'{label}:\n{body[: allowed[label]]}{" [... cut]" if len(body) > allowed[label] else ""}'
        for label, body in sections
    )
    return text, any(len(body) > allowed[label] for label, body in sections)
