"""Drift tests: the /review-branch suite against the code at `HEAD`.

The reviewer prompts and `preconditions.py` hardcode repo paths, surface triggers and family
lists. Each test here recomputes one of them from this checkout's `HEAD` and fails, naming the
stale entry, when the suite no longer matches. The replay needs the last month of history.

The weekly `Review Branch Drift` workflow runs them on `main`; no pull request runs them.

Run: uv run pytest .agents/skills/review-branch/test_drift.py  (`-m 'not drift_replay'` skips the replay)
"""

from __future__ import annotations

import ast
import itertools
import os
import re
import subprocess
import sys
import time
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import preconditions  # noqa: E402
import prepare  # noqa: E402
from mechanical.models import PACKAGE_ROOTS  # noqa: E402

REPO = HERE.parents[2]
REVIEWERS = REPO / '.agents/agents'
PROMPTS = (*sorted(REVIEWERS.glob('review-*.md')), HERE / 'SKILL.md')
REF = 'HEAD'
SLIM = preconditions.SLIM
HARNESS = 'src/pydantic_ai_harness/pydantic_ai_harness/'


@dataclass
class Tree:
    """Every file and directory of one tree, for literal and glob lookups."""

    files: list[str]
    paths: set[str] = field(init=False)
    suffixes: set[str] = field(init=False)

    def __post_init__(self) -> None:
        dirs = {'/'.join(parts[:i]) for parts in (f.split('/') for f in self.files) for i in range(1, len(parts))}
        self.paths = set(self.files) | dirs
        self.suffixes = {'/'.join(parts[i:]) for parts in (p.split('/') for p in self.paths) for i in range(len(parts))}

    def has(self, pattern: str, *, anchored: bool) -> bool:
        pattern = pattern.rstrip('/')
        if not GLOB_CHARS.search(pattern):
            return pattern in (self.paths if anchored else self.suffixes)
        regex = re.compile(('^' if anchored else '(^|/)') + glob_regex(pattern) + '$')
        return any(regex.search(path) for path in self.paths)


@dataclass
class Checkout:
    repo: Path
    sha: str
    tree: Tree
    top: set[str]

    def git(self, *args: str) -> bytes:
        return preconditions.git(self.repo, *args)

    def read(self, paths: list[str]) -> dict[str, bytes]:
        batch = subprocess.run(
            ['git', '-C', str(self.repo), 'cat-file', '--batch'],
            input=''.join(f'{self.sha}:{path}\n' for path in paths).encode(),
            check=True,
            capture_output=True,
        ).stdout
        contents: dict[str, bytes] = {}
        position = 0
        for path in paths:
            newline = batch.index(b'\n', position)
            size = int(batch[position:newline].split()[2])
            contents[path] = batch[newline + 1 : newline + 1 + size]
            position = newline + size + 2
        return contents


@pytest.fixture(scope='module')
def checkout() -> Checkout:
    sha = preconditions.git(REPO, 'rev-parse', '--verify', f'{REF}^{{commit}}').decode().strip()
    files = preconditions.git(REPO, 'ls-tree', '-r', '--name-only', '-z', sha).decode().split('\0')[:-1]
    return Checkout(REPO, sha, Tree(files), {path.split('/')[0] for path in files})


@pytest.fixture(scope='module')
def working_tree() -> Tree:
    """The working tree, untracked files included, so a path a prompt gained before its commit resolves."""
    listing = subprocess.run(
        ['git', '-C', str(REPO), 'ls-files', '-co', '--exclude-standard', '-z', '.'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return Tree([path for path in listing.split('\0') if path])


# ---------------------------------------------------------------------------------------------
# Path mentions
# ---------------------------------------------------------------------------------------------

BACKTICKED = re.compile(r'`([^`\n]+)`')
GLOB_CHARS = re.compile(r'[*<>{}\[\]]|\.\.\.')
BRACES = re.compile(r'\{([^{}]*,[^{}]*)\}')
EXTENSIONS = ('.py', '.md', '.yml', '.yaml', '.json', '.toml', '.txt', '.diff', '.list', '.sql')
# Files a review run writes into the worktree: the `tmp/` artifacts of prepare.py and
# preconditions.py, the run state, and the branch-context records made from templates.
WORKTREE_STATE = re.compile(
    r'^(\.agents/skills/review-branch/)?(tmp/.*|review-state\.json|preconditions\.json|code\.diff|cassettes\.list'
    r'|(base-ref|base-sha|merge-base|head-sha|branch-files|diff-hash|code\.blame-map|pr-ready-for-docs|spec-url'
    r'|core-symbols-touched|wrapped-class-files|new-py-files|ambient-branch-lines|sibling-family-files'
    r'|peer-type-branches|durable-exec-entries)\.txt)$'
    r'|^(\.claude/skills/)?(branch-context/)?(pr-decisions|issue-brief)\.md$'
)
# Paths a prompt names on purpose although `HEAD` lacks them.
NAMED_BUT_ABSENT = {
    'durable_exec/restate/': 'integration-impact says Restate has no in-tree package',
    '.../airflow/': 'integration-impact says Airflow has no in-tree package',
    'toolsets/fastmcp.py': 'integration-impact says `FastMCPToolset` was dropped for `mcp.py`',
    'pydantic-clai2/src/pydantic_clai2/': 'precedent maps the pre-monorepo pydantic-ai-harness repo layout',
    'pydantic-clai2/tests/': 'precedent maps the pre-monorepo pydantic-ai-harness repo layout',
    'openai/types/responses/*.py': 'runtime-behavior names the OpenAI SDK source',
    'routes/': 'spec-conformance lists path names that signal spec work in any diff',
    'protocol/': 'spec-conformance lists path names that signal spec work in any diff',
    'serializers/': 'spec-conformance lists path names that signal spec work in any diff',
    'chat_completions.py': 'spec-conformance lists path names that signal spec work in any diff',
    'tests/test_chat_completions.py::test_tool_call_streaming': 'an illustrative row of the spec-conformance report',
    '_internal/': 'public-api names the shape of a private subpackage',
}


def glob_regex(pattern: str) -> str:
    """Translate a prompt's path glob: `*`, `**`, `[!x]`, `...`, and `<placeholder>`."""
    out: list[str] = []
    index = 0
    while index < len(pattern):
        rest = pattern[index:]
        if rest.startswith(('**', '...')):
            out.append('.*')
            index += 2 if rest.startswith('**') else 3
        elif rest[0] == '*':
            out.append('[^/]*')
            index += 1
        elif rest[0] in '<{' and (close := rest.find('>' if rest[0] == '<' else '}')) > 0:
            out.append('.+' if rest[1:close] == '...' else '[^/]+')
            index += close + 1
        elif rest[0] == '[' and (close := rest.find(']')) > 0:
            body = rest[1:close]
            out.append('[' + ('^' + body[1:] if body.startswith('!') else body) + ']')
            index += close + 1
        else:
            out.append(re.escape(rest[0]))
            index += 1
    return ''.join(out)


def expand_braces(word: str) -> list[str]:
    match = BRACES.search(word)
    if match is None:
        return [word]
    return [
        expanded
        for option in match[1].split(',')
        for expanded in expand_braces(word[: match.start()] + option + word[match.end() :])
    ]


@dataclass(frozen=True)
class Mention:
    """One backticked path, braces expanded, with its pytest node or heading anchor split off."""

    text: str
    path: str
    node: str
    anchor: str


def mentions(text: str, top: set[str]) -> Iterator[Mention]:
    for span in BACKTICKED.findall(text):
        for word in span.split():
            word = word.strip('\'"(),;')
            # An unbalanced `<` or `>` is one word of a placeholder with spaces, `<file per tests/AGENTS.md>`.
            if '://' in word or word.startswith(('~', '/')) or word.count('<') != word.count('>'):
                continue
            for expanded in expand_braces(word):
                path, _, node = expanded.partition('::')
                path, _, anchor = path.partition('#')
                if ':' in path or not path.strip('.*/'):
                    continue
                anchored = path.startswith(('.../', '$REVIEW_SKILL/', '.claude/')) or path.split('/')[0] in top
                if anchored or (path.endswith(EXTENSIONS) and path[0] != '.') or ('/' in path and path.endswith('/')):
                    yield Mention(expanded, path, node, anchor)


def resolves(path: str, checkout: Checkout, working_tree: Tree) -> bool:
    if WORKTREE_STATE.match(path):
        return True
    if path.startswith('$REVIEW_SKILL/'):
        return working_tree.has('.agents/skills/review-branch/' + path.removeprefix('$REVIEW_SKILL/'), anchored=True)
    if path.startswith('.../'):
        return checkout.tree.has(path[4:], anchored=False)
    if path.split('/')[0] in checkout.top:
        return checkout.tree.has(path, anchored=True) or working_tree.has(path, anchored=True)
    return checkout.tree.has(path, anchored=False) or working_tree.has(path, anchored=False)


def heading_slugs(markdown: str) -> set[str]:
    slugs = set(re.findall(r'\{#([\w-]+)\}', markdown))
    for heading in re.findall(r'^#+\s+(.+?)\s*$', markdown, re.MULTILINE):
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', heading).replace('`', '').lower()
        slugs.add(re.sub(r'\s', '-', re.sub(r'[^\w\- ]', '', text)))
    return slugs


def test_prompt_paths_exist(checkout: Checkout, working_tree: Tree) -> None:
    missing: list[str] = []
    for prompt in PROMPTS:
        for mention in mentions(prompt.read_text(), checkout.top):
            if mention.text in NAMED_BUT_ABSENT:
                continue
            if not resolves(mention.path, checkout, working_tree):
                missing.append(f'{prompt.name}: `{mention.text}` matches no path')
                continue
            if GLOB_CHARS.search(mention.path) or not (mention.node or mention.anchor):
                continue
            if mention.path not in checkout.tree.files:
                missing.append(f'{prompt.name}: `{mention.text}` names a test or anchor in a directory')
                continue
            text = checkout.read([mention.path])[mention.path].decode()
            if mention.node and not re.search(rf'def {re.escape(mention.node)}\b', text):
                missing.append(f'{prompt.name}: `{mention.text}` names a test the file does not define')
            if mention.anchor and mention.anchor not in heading_slugs(text):
                missing.append(f'{prompt.name}: `{mention.text}` names a heading the page does not have')
    assert missing == [], f'stale paths at {REF} {checkout.sha[:10]}:\n' + '\n'.join(missing)


def test_absent_paths_are_still_named_and_still_absent(checkout: Checkout, working_tree: Tree) -> None:
    named = {mention.text: mention for prompt in PROMPTS for mention in mentions(prompt.read_text(), checkout.top)}
    stale = [
        text for text in NAMED_BUT_ABSENT if text not in named or resolves(named[text].path, checkout, working_tree)
    ]
    assert stale == [], f'NAMED_BUT_ABSENT entries no prompt names, or that exist again: {stale}'


# ---------------------------------------------------------------------------------------------
# Families, computed from code
# ---------------------------------------------------------------------------------------------

FIRST_PARTY = {'pydantic_ai', 'pydantic_ai_harness', 'pydantic_clai2', 'pydantic_graph', 'pydantic_evals', 'clai'}
SIBLING = 'review-sibling-consistency'
INTEGRATION = 'review-integration-impact'
SUSPEND = 'review-suspend-resume'


@dataclass(frozen=True)
class Family:
    """Every production subclass of `bases` (defining file, class name), under `within` when set."""

    bases: tuple[tuple[str, str], ...]
    reviewers: tuple[str, ...]
    within: str = ''


FAMILIES = {
    'Model subclasses': Family(((SLIM + 'models/__init__.py', 'Model'),), (SIBLING,)),
    'realtime models': Family(((SLIM + 'realtime/model.py', 'RealtimeModel'),), (SIBLING, INTEGRATION)),
    'embedding models': Family(((SLIM + 'embeddings/base.py', 'EmbeddingModel'),), (SIBLING,)),
    'image generation models': Family(((SLIM + 'images/base.py', 'ImageGenerationModel'),), (SIBLING,)),
    'harness sandbox backends': Family(((SLIM + 'workspaces/protocol.py', 'WorkspaceBackend'),), (SIBLING,), HARNESS),
    'durability backends': Family(
        (
            (SLIM + 'durable_exec/_operation_backend.py', 'DurableOperationBackend'),
            (SLIM + 'durable_exec/_base.py', 'BaseDurabilityCapability'),
        ),
        (SIBLING, INTEGRATION, SUSPEND),
    ),
    'UI adapters': Family(((SLIM + 'ui/_adapter.py', 'UIAdapter'),), (SIBLING, INTEGRATION, SUSPEND)),
    'wrapper agents': Family(((SLIM + 'agent/wrapper.py', 'WrapperAgent'),), (INTEGRATION,)),
}
ClassKey = tuple[str, str]


def base_name(base: ast.expr, external: set[str]) -> str | None:
    """The class name a base expression refers to, or None for a third-party base."""
    if isinstance(base, ast.Subscript):
        base = base.value
    root = base
    while isinstance(root, ast.Attribute):
        root = root.value
    if not isinstance(root, ast.Name) or root.id in external:
        return None
    return base.attr if isinstance(base, ast.Attribute) else root.id


@pytest.fixture(scope='module')
def family_members(checkout: Checkout) -> dict[str, set[ClassKey]]:
    """AST subclass scan over the production packages; a base resolves to its own file, then its package."""
    sources = checkout.read([p for p in checkout.tree.files if p.endswith('.py') and p.startswith(PACKAGE_ROOTS)])
    defined: defaultdict[str, list[ClassKey]] = defaultdict(list)
    bases_of: dict[ClassKey, list[str]] = {}
    for path, source in sources.items():
        tree = ast.parse(source)
        external: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external |= {
                    a.asname or a.name.split('.')[0] for a in node.names if a.name.split('.')[0] not in FIRST_PARTY
                }
            elif (
                isinstance(node, ast.ImportFrom)
                and not node.level
                and (node.module or '').split('.')[0] not in FIRST_PARTY
            ):
                external |= {a.asname or a.name for a in node.names}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                defined[node.name].append((path, node.name))
                bases_of[path, node.name] = [name for base in node.bases if (name := base_name(base, external))]
    children: defaultdict[ClassKey, set[ClassKey]] = defaultdict(set)
    for key, bases in bases_of.items():
        for base in bases:
            candidates = [c for c in defined.get(base, []) if c != key]
            package = [c for c in candidates if c[0].startswith(next(r for r in PACKAGE_ROOTS if key[0].startswith(r)))]
            for parent in [c for c in candidates if c[0] == key[0]] or package or candidates:
                children[parent].add(key)

    members: dict[str, set[ClassKey]] = {}
    for name, family in FAMILIES.items():
        missing = [base for base in family.bases if base not in bases_of]
        assert missing == [], f'{name}: base class moved, update FAMILIES: {missing}'
        found: set[ClassKey] = set()
        stack = list(family.bases)
        while stack:
            for child in children[stack.pop()] - found:
                found.add(child)
                stack.append(child)
        members[name] = {key for key in found if key[0].startswith(family.within)}
    return members


def too_broad(regex: re.Pattern[str]) -> bool:
    """A pattern that reaches an arbitrary module of a package names the package, not a family."""
    return any(
        regex.search(root + probe) for root in PACKAGE_ROOTS for probe in ('zz_probe.py', 'zz_probe/zz_probe.py')
    )


def family_patterns(text: str, top: set[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for mention in mentions(text, top):
        path = mention.path.removeprefix('.../').rstrip('/')
        anchored = not mention.path.startswith('.../') and path.split('/')[0] in top
        regex = re.compile(('^' if anchored else '(^|/)') + glob_regex(path) + '(/|$)')
        if not too_broad(regex):
            patterns.append(regex)
    return patterns


def test_family_lists_name_every_member(checkout: Checkout, family_members: dict[str, set[ClassKey]]) -> None:
    unnamed: list[str] = []
    for family_name, family in FAMILIES.items():
        assert family_members[family_name], f'{family_name}: the scan found no member; the base or `within` is wrong'
        for reviewer in family.reviewers:
            text = (REVIEWERS / f'{reviewer}.md').read_text()
            patterns = family_patterns(text, checkout.top)
            for path, name in sorted(family_members[family_name]):
                if not re.search(rf'\b{re.escape(name)}\b', text) and not any(p.search(path) for p in patterns):
                    unnamed.append(f'{reviewer}: {family_name} member `{name}` ({path})')
    assert unnamed == [], (
        'family members the lists neither name nor cover by a path pattern; add the member or its family '
        f'pattern at {REF} {checkout.sha[:10]}:\n' + '\n'.join(unnamed)
    )


# ---------------------------------------------------------------------------------------------
# Surface ownership
# ---------------------------------------------------------------------------------------------

# Surfaces no lane trigger opens on purpose, each with its reason.
OWNERLESS = {
    'docs/': 'review-docs runs on every branch, so docs need no path trigger',
    'examples/pydantic_ai_examples/': 'demo apps outside PACKAGE_ROOTS; the lanes that run on every code change review them',
    SLIM + 'common_tools/': 'independent leaf tools; `web_fetch.py` opens review-trust-boundary by name',
    SLIM + 'ext/': 'one LangChain adapter; the package lanes review it',
}


def lanes(path: str) -> set[str]:
    """The lanes a one-line change to `path` dispatches."""
    diff = f'diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-x = 0\n+x = 1\n'.encode()
    pre = preconditions.compute([preconditions.Change('M', path)], diff, [], None)
    return {lane for lane, (run, _) in preconditions.decide(pre).items() if run}


def test_every_surface_has_an_owning_lane(checkout: Checkout, family_members: dict[str, set[ClassKey]]) -> None:
    """A surface is owned when a change under it opens a lane a change one level up does not.

    Workspace packages, `.github/workflows/` and `docs/` compare against a root-level file. Slim
    subpackages compare against a slim root module. Harness directories that hold a family member
    (a sandbox or durability backend) compare against a harness root module; the other harness
    directories are capabilities, owned by the package lanes.
    """
    root = lanes('drift_probe.txt')
    surfaces: dict[str, set[str]] = {'.github/workflows/': root, 'docs/': root}
    pyproject = tomllib.loads(checkout.read(['pyproject.toml'])['pyproject.toml'].decode())
    for member in pyproject['tool']['uv']['workspace']['members']:
        # A regular package has `__init__.py`; `examples/pydantic_ai_examples/` is a namespace package.
        modules = [
            m for p in checkout.tree.files if (m := re.fullmatch(rf'({re.escape(member)}/[^./][^/]*/)([^/]+\.py)', p))
        ]
        packages = {m[1] for m in modules if m[2] == '__init__.py'} or {m[1] for m in modules}
        assert packages, f'workspace member {member} has no package directory'
        surfaces |= dict.fromkeys(packages, root)
    slim = lanes(SLIM + 'drift_probe.py')
    for path in checkout.tree.paths:
        if re.fullmatch(rf'{re.escape(SLIM)}[^._/][^/]*', path) and path not in checkout.tree.files:
            surfaces[path + '/'] = slim
    harness = lanes(HARNESS + 'drift_probe.py')
    for members in family_members.values():
        for path, _ in members:
            if path.startswith(HARNESS) and '/' in path.removeprefix(HARNESS):
                surfaces[HARNESS + path.removeprefix(HARNESS).split('/')[0] + '/'] = harness

    wrong: list[str] = []
    for surface, baseline in sorted(surfaces.items()):
        extension = '.yml' if surface.startswith('.github/') else '.md' if surface.startswith('docs/') else '.py'
        owners = lanes(surface + 'drift_probe' + extension) - baseline
        if not owners and surface not in OWNERLESS:
            wrong.append(f'{surface}: no lane trigger opens on it')
        elif owners and surface in OWNERLESS:
            wrong.append(f'{surface}: OWNERLESS, but {sorted(owners)} now open on it')
    wrong += [f'{surface}: OWNERLESS entry is no longer a surface' for surface in OWNERLESS if surface not in surfaces]
    assert wrong == [], f'surface ownership at {REF} {checkout.sha[:10]}:\n' + '\n'.join(wrong)


# ---------------------------------------------------------------------------------------------
# Dispatch replay
# ---------------------------------------------------------------------------------------------

REPLAY_DAYS = 30
# Lanes the replay cannot open, each with its reason.
NOT_REPLAYABLE = {
    'review-spec-coverage': 'opens on a spec URL in the PR body, and a squash commit carries no PR body',
}


@pytest.mark.drift_replay
def test_every_lane_dispatches_on_recent_commits(checkout: Checkout) -> None:
    commits = checkout.git('rev-list', '--first-parent', f'--since={REPLAY_DAYS}.days', checkout.sha).decode().split()
    assert commits, f'no {REF} commit in the last {REPLAY_DAYS} days; fetch the full history'

    def dispatched(commit: str) -> set[str]:
        diff_args = ('diff', *prepare.DIFF_FLAGS, f'{commit}^', commit)
        changes = preconditions.name_status(checkout.repo, f'{commit}^', commit)
        diff = checkout.git(*diff_args, '--patch', '--unified=3', '--', *prepare.CODE_PATHS)
        cassettes = checkout.git(*diff_args, '--name-status', '--', *prepare.CASSETTE_PATHS).decode().splitlines()
        pre = preconditions.compute(changes, diff, [line for line in cassettes if line.strip()], None)
        return {lane for lane, (run, _) in preconditions.decide(pre).items() if run}

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        counts = Counter(itertools.chain.from_iterable(pool.map(dispatched, commits)))
    roster = [lane for wave in preconditions.WAVES.values() for lane in wave]
    dead = [lane for lane in roster if counts[lane] == 0 and lane not in NOT_REPLAYABLE]
    revived = [lane for lane in NOT_REPLAYABLE if counts[lane]]
    table = '\n'.join(f'  {lane}: {counts[lane]}' for lane in roster)
    report = f'{len(commits)} commits in {time.monotonic() - started:.0f}s at {REF} {checkout.sha[:10]}:\n{table}'
    assert dead == [], f'lanes that never dispatched: {dead}\n{report}'
    assert revived == [], f'NOT_REPLAYABLE lanes that now dispatch: {revived}\n{report}'
