"""Select the PR Review Checks for a pull request and the files each check reviews."""

from __future__ import annotations

import codecs
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class ReviewCheck:
    """One check. Its prompt fragment is `.github/pr-review-checks/prompts/<name>.md`."""

    name: str
    title: str
    scope: str
    """Completes "This check covers …" in the shared prompt."""
    security_rubric: bool = True
    """Put `security.md`, the vulnerability rubric, before the check's own fragment."""
    # Permissive starting values: tune them down from the steps and durations each review records.
    max_steps: int = 100
    """At this step OpenCode stops offering tools, and the review ends `INCOMPLETE`."""
    max_minutes: int = 25
    """After this many minutes the runner stops the review and reports it `INCOMPLETE`."""


@dataclass(frozen=True, kw_only=True)
class SpecialistCheck(ReviewCheck):
    """A check selected by the files it owns, and optionally by tokens in other files' changed lines."""

    paths: re.Pattern[str]
    """A changed file whose path or previous path matches selects this check and is reviewed by it."""
    diff_tokens: re.Pattern[str] | None = None
    """An added or removed line matching this, in a file under `token_paths`, also selects that file."""
    token_paths: re.Pattern[str] | None = None


@dataclass(frozen=True)
class ChangedFile:
    """A file the pull request changes, with its added and removed lines."""

    path: str
    previous_path: str
    changed_lines: tuple[str, ...]

    def matches(self, pattern: re.Pattern[str]) -> bool:
        return any(pattern.fullmatch(path) for path in (self.path, self.previous_path) if path)


# Checkout, context gathering and tool setup run before a review's own clock starts.
SETUP_MINUTES = 10

GENERAL = ReviewCheck(
    name='general',
    title='General vulnerability review',
    scope='every changed source, workflow and configuration file that no other security check owns',
)

SPECIALIST_CHECKS: tuple[SpecialistCheck, ...] = (
    SpecialistCheck(
        name='url-fetch',
        title='URL fetch and egress',
        scope='URL fetching, downloads, provider file references and browser egress',
        paths=re.compile(
            r'pydantic_ai_slim/pydantic_ai/(_ssrf\.py|common_tools/web_fetch\.py'
            r'|capabilities/(web_fetch|file_understanding)\.py|realtime/_utils\.py)'
            r'|src/pydantic_ai_harness/pydantic_ai_harness/(playwright|browser_use)/.*'
            r'|tests/test_(ssrf|web_fetch)\.py'
        ),
        diff_tokens=re.compile(
            r'download_item\(|safe_download\(|force_download|allow_local|UploadedFile\(|_CLOUD_METADATA'
        ),
        token_paths=re.compile(r'(pydantic_ai_slim|src)/.*\.py'),
    ),
    SpecialistCheck(
        name='ui',
        title='UI adapters and local web UI',
        scope="the UI adapters' client/server trust boundary and CLAI 2's local web control UI",
        paths=re.compile(
            r'pydantic_ai_slim/pydantic_ai/(ui/.*|messages\.py)'
            r'|src/pydantic_clai2/(web-ui/.*|pydantic_clai2/builtin_plugins/web/.*|pydantic_clai2/cli/web\.py'
            r'|clai2_web_README\.md|scripts/check_web_assets\.py)'
            r'|docs/(ui/.*|input\.md|web\.md|harness/clai2\.md)'
            r'|tests/(test_vercel_ai|test_ag_ui|test_ui|test_ui_web)\.py|tests/clai2/(test_web[^/]*\.py|browser/.*)'
        ),
    ),
    SpecialistCheck(
        name='telemetry',
        title='Telemetry',
        scope='instrumentation, spans, logs and the content controls on them',
        paths=re.compile(
            r'pydantic_ai_slim/pydantic_ai/(_instrumentation|_otel_messages|capabilities/instrumentation'
            r'|models/instrumented|embeddings/instrumented|images/instrumented|realtime/_instrumentation)\.py'
            r'|src/pydantic_ai_harness/pydantic_ai_harness/logfire/.*'
            r'|src/pydantic_clai2/pydantic_clai2/runtime/agent_instrumentation\.py'
        ),
        diff_tokens=re.compile(r'otel_|include_content|gen_ai\.|set_attribute|record_exception|span_name'),
        token_paths=re.compile(r'pydantic_ai_slim/pydantic_ai/(messages\.py|durable_exec/.*)'),
    ),
    SpecialistCheck(
        name='sandboxes',
        title='Sandboxes and Code Mode',
        scope='sandboxes, workspaces, shells and Code Mode execution boundaries',
        paths=re.compile(
            r'src/pydantic_ai_harness/pydantic_ai_harness/'
            r'(bubblewrap_sandbox|code_mode|ipython_code_mode|ssh_workspace|shell)/.*'
        ),
    ),
    SpecialistCheck(
        name='tests-and-dependencies',
        title='Tests and dependencies',
        scope='test changes, cassettes, dependency manifests and lockfiles',
        security_rubric=False,
        paths=re.compile(
            r'tests/.*|(.*/)?(pyproject\.toml|uv\.lock|package-lock\.json)'
            r'|\.github/scripts/[^/]*\.lock|\.github/aw/actions-lock\.json'
        ),
    ),
)

CHECKS: tuple[ReviewCheck, ...] = (GENERAL, *SPECIALIST_CHECKS)

# Mirrors `is_non_test_content` in the `classify` job of `ci.yml`.
_CONTENT = (
    'README.md',
    'clai/README.md',
    'src/pydantic_clai2/README.md',
    'src/pydantic_ai_harness/README.md',
    *(f'docs/*.{ext}' for ext in ('md', 'mdc', 'png', 'svg', 'ico', 'jpg', 'jpeg', 'gif', 'webp', 'avif')),
    'examples/*.md',
    'pydantic_ai_slim/*.md',
    'pydantic_graph/*.md',
    'pydantic_evals/*.md',
    '.agents/*.md',
    '.agents/*.tsv',
    '.claude/*.md',
    'agent_docs/*.md',
    '*/AGENTS.md',
    '*/CLAUDE.md',
)
# Compiled from a workflow source that the general check reviews.
_GENERATED = re.compile(r'\.github/workflows/.*\.lock\.yml')


def is_content(path: str) -> bool:
    """Documentation and agent instructions, which the general check does not review."""
    if path.startswith(('tests/', '.github/', '.macroscope/')):
        return False
    return any(fnmatchcase(path, pattern) for pattern in _CONTENT) or (path.endswith('.md') and '/' not in path)


def _general_reviews(file: ChangedFile) -> bool:
    paths = [path for path in (file.path, file.previous_path) if path]
    return any(not is_content(path) and not _GENERATED.fullmatch(path) for path in paths)


def select(files: list[ChangedFile]) -> dict[str, list[str]]:
    """Map each selected check to the paths it reviews, in `CHECKS` order."""
    selected: dict[str, list[str]] = {}
    owned: set[str] = set()
    for check in SPECIALIST_CHECKS:
        by_path = [file.path for file in files if file.matches(check.paths)]
        owned.update(by_path)
        by_token = [
            file.path
            for file in files
            if check.diff_tokens
            and check.token_paths
            and file.path not in by_path
            and file.matches(check.token_paths)
            and any(check.diff_tokens.search(line) for line in file.changed_lines)
        ]
        if reviewed := by_path + by_token:
            selected[check.name] = reviewed
    general = [file.path for file in files if file.path not in owned and _general_reviews(file)]
    if general:
        return {GENERAL.name: general, **selected}
    return selected


def _patch_path(field: str, prefix: str) -> str:
    """A patch header's path without its `a/` or `b/` prefix, or '' for `/dev/null`."""
    # `core.quotePath=false` still C-quotes a path that holds a quote, a backslash or a control character.
    if field.startswith('"'):
        field = codecs.escape_decode(field[1:-1])[0].decode('utf-8', 'replace')
    return field.removeprefix(prefix) if field.startswith(prefix) else ''


def changed_files(repository: Path, base: str, head: str) -> list[ChangedFile]:
    """Read the pull request's changes from git, so no file-count API cap can hide one."""
    git = ['git', '-C', str(repository), '-c', 'core.quotePath=false']
    merge_base = subprocess.run(
        [*git, 'merge-base', base, head], check=True, capture_output=True, text=True
    ).stdout.strip()
    # `--no-ext-diff` and `--no-textconv` keep the pull request's `.gitattributes` from running a command,
    # and `--text` keeps a `-diff` attribute from hiding a file's changed lines.
    diff = ['diff', '--no-ext-diff', '--no-textconv', '--no-color', '-M', merge_base, head]
    status = subprocess.run([*git, *diff, '--name-status', '-z'], check=True, capture_output=True, text=True).stdout
    fields = status.split('\0')
    files: dict[str, ChangedFile] = {}
    index = 0
    while index < len(fields) - 1:
        code = fields[index]
        if code.startswith(('R', 'C')):
            previous, path = fields[index + 1], fields[index + 2]
            index += 3
        else:
            previous, path = '', fields[index + 1]
            index += 2
        files[path] = ChangedFile(path, previous if code.startswith('R') else '', ())
    patch = subprocess.run(
        [*git, *diff, '--text', '-U0'], check=True, capture_output=True, text=True, errors='replace'
    ).stdout
    lines: dict[str, list[str]] = {}
    in_header = False
    old_path = new_path = ''
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            in_header, old_path, new_path = True, '', ''
        elif in_header and line.startswith('--- '):
            old_path = _patch_path(line[4:], 'a/')
        elif in_header and line.startswith('+++ '):
            new_path = _patch_path(line[4:], 'b/') or old_path
        elif line.startswith('@@'):
            in_header = False
        elif not in_header and new_path and line[:1] in ('+', '-'):
            lines.setdefault(new_path, []).append(line[1:])
    return [ChangedFile(file.path, file.previous_path, tuple(lines.get(file.path, ()))) for file in files.values()]


def check_named(name: str) -> ReviewCheck:
    """Return the configured check with this name."""
    return next(entry for entry in CHECKS if entry.name == name)


def main() -> None:
    """Print the review matrix as JSON: one `{"check": name, "job_minutes": n}` entry per selected check."""
    repository, base, head = sys.argv[1:4]
    selection = select(changed_files(Path(repository), base, head))
    matrix = [{'check': name, 'job_minutes': check_named(name).max_minutes + SETUP_MINUTES} for name in selection]
    print(json.dumps(matrix, separators=(',', ':')))


if __name__ == '__main__':
    main()
