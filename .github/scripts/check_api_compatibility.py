from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

PACKAGES = {
    'pydantic_ai': 'pydantic_ai_slim',
    'pydantic_graph': 'pydantic_graph',
    'pydantic_evals': 'pydantic_evals',
    'clai': 'clai',
    'pydantic_ai_harness': 'src/pydantic_ai_harness',
    'pydantic_clai2': 'src/pydantic_clai2',
}
# The `clai2` web UI and the `workspaces` plugin behind it are application code, not an SDK for others to build on,
# so their changes need no compatibility waiver. Each excludes every module in the package.
EXCLUDED_PACKAGES = (
    'pydantic_clai2.builtin_plugins.web',
    'pydantic_clai2.builtin_plugins.workspaces',
)
# Run from its own tool environment: `griffe` 1.x, which `prefect` accepts, installs over `griffelib`'s `griffe`
# module, and `griffecli` crashes against it. A bump of either package can reword or add findings, which changes
# their fingerprints.
GRIFFECLI_VERSION = '2.3.0'
# Without it Griffe treats an import its module's `__all__` doesn't list as private, so a
# `from ._module import Name as Name` re-export such as `pydantic_ai.models.KnownModelName` is never checked.
GRIFFE_PUBLIC_REDUNDANT_ALIASES_VERSION = '0.3.2'
_FINDING_RE = re.compile(r'^(?P<path>.*):(?P<line>\d+): (?P<message>.*)$')
_ATTRIBUTE_VALUE_CHANGED = ': Attribute value was changed: '


class Waiver(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    against: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    fingerprint: Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{64}$')]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    pull_request: Annotated[str, StringConstraints(pattern=r'^https://github\.com/pydantic/pydantic-ai/pull/\d+$')]


class ModelNameLiteral(BaseModel):
    """A type alias listing model names for IDE autocompletion, whose `Literal` members can change without a waiver.

    Removing a name from an alias without a `str |` arm is still a permitted compatibility impact: it needs the
    `compatibility impact` label and a release note.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    path: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    symbol: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GrowingCollection(BaseModel):
    """A public `dict` or `set` constant that registers entries, such as CLAI's settings, which can gain entries without a waiver.

    Removing an entry or changing an entry's value still needs a waiver.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    path: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    symbol: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Allowlist(BaseModel):
    model_config = ConfigDict(extra='forbid')

    allowed_breakages: list[Waiver]
    model_name_literals: list[ModelNameLiteral] = []
    growing_collections: list[GrowingCollection] = []


@dataclass(frozen=True)
class Finding:
    package: str
    path: str
    line: int
    message: str

    @property
    def fingerprint(self) -> str:
        value = f'{self.package}\0{self.path}\0{self.message}'.encode()
        return hashlib.sha256(value).hexdigest()

    @property
    def symbol(self) -> str:
        return self.message.partition(': ')[0]


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare the public Python API with a released Git tag.')
    parser.add_argument('--against', required=True, help='Released Git tag to compare against.')
    parser.add_argument(
        '--allowlist',
        type=Path,
        default=Path('.github/api-compatibility-allowlist.json'),
        help='Exact reviewed compatibility-impact waivers.',
    )
    args = parser.parse_args()

    waivers, model_name_literals, growing_collections = load_allowlist(args.allowlist)
    used_waivers: set[tuple[str, str]] = set()
    failed = False
    for package, search_path in PACKAGES.items():
        findings = run_griffe(package, search_path, args.against)
        for finding in findings:
            if is_excluded(finding, search_path):
                continue
            if (finding.path, finding.symbol) in model_name_literals and (
                changes := literal_member_changes(finding.message)
            ):
                added, removed = changes
                message = (
                    f'{finding.symbol}: model names added: {sorted(added)}, removed: {sorted(removed)}. '
                    'Model-name alias changes need no waiver.'
                )
                if removed:
                    message += (
                        ' Removing a name from an alias without a `str |` arm is a permitted compatibility impact: '
                        'add the `compatibility impact` label and a release note.'
                    )
                emit_annotation('warning' if removed else 'notice', finding, message)
                continue
            if (
                (finding.path, finding.symbol) in growing_collections
                and (changes := collection_entry_changes(finding.message)) is not None
                and not changes[1]
            ):
                message = (
                    f'{finding.symbol}: entries added: {changes[0]}. Additions to a growing collection need no waiver.'
                )
                emit_annotation('notice', finding, message)
                continue
            waiver_key = (args.against, finding.fingerprint)
            waiver = waivers.get(waiver_key)
            if waiver is None:
                failed = True
                emit_annotation(
                    'error',
                    finding,
                    f'{finding.message} [fingerprint: {finding.fingerprint}]. '
                    'Preserve compatibility or follow the allowed compatibility-impact process.',
                )
            else:
                used_waivers.add(waiver_key)
                emit_annotation(
                    'warning',
                    finding,
                    f'{finding.message} Allowed by {waiver.pull_request}: {waiver.reason}',
                )

    unused_waivers = {key for key in waivers if key[0] == args.against} - used_waivers
    for _, fingerprint in sorted(unused_waivers):
        failed = True
        print(f'::error::Unused API compatibility waiver for {args.against}: {fingerprint}')

    if failed:
        raise SystemExit(1)


def is_excluded(finding: Finding, search_path: str) -> bool:
    """Whether the finding is in a file of one of `EXCLUDED_PACKAGES` under `search_path`."""
    path = Path(finding.path)
    return any(path.is_relative_to(Path(search_path, *package.split('.'))) for package in EXCLUDED_PACKAGES)


def load_allowlist(
    path: Path,
) -> tuple[dict[tuple[str, str], Waiver], set[tuple[str, str]], set[tuple[str, str]]]:
    allowlist = Allowlist.model_validate_json(path.read_text(encoding='utf-8'))
    waivers: dict[tuple[str, str], Waiver] = {}
    for waiver in allowlist.allowed_breakages:
        key = (waiver.against, waiver.fingerprint)
        if key in waivers:
            raise ValueError(f'{path}: duplicate waiver for {waiver.against}: {waiver.fingerprint}')
        waivers[key] = waiver
    return (
        waivers,
        {(entry.path, entry.symbol) for entry in allowlist.model_name_literals},
        {(entry.path, entry.symbol) for entry in allowlist.growing_collections},
    )


def literal_member_changes(message: str) -> tuple[set[str], set[str]] | None:
    """Return the names a value change adds and removes, if it changes nothing but `Literal` string members."""
    _, _, values = message.partition(_ATTRIBUTE_VALUE_CHANGED)
    # Griffe prints `<old> -> <new>`. A split inside a string member leaves a side that doesn't parse.
    for separator in re.finditer(' -> ', values):
        try:
            old = _literal_members(values[: separator.start()])
            new = _literal_members(values[separator.end() :])
        except SyntaxError:
            continue
        if old is None or new is None or old[0] != new[0]:
            return None
        return new[1] - old[1], old[1] - new[1]
    return None


def collection_entry_changes(message: str) -> tuple[list[str], list[str]] | None:
    """Return the entries a value change adds and removes, if both sides are `dict` or `set` literals of one kind.

    An entry whose value changed counts as removed, with its new value added.
    """
    _, _, values = message.partition(_ATTRIBUTE_VALUE_CHANGED)
    # Griffe prints `<old> -> <new>`. A split inside a string entry leaves a side that doesn't parse.
    for separator in re.finditer(' -> ', values):
        try:
            old = _collection_entries(values[: separator.start()])
            new = _collection_entries(values[separator.end() :])
        except (SyntaxError, ValueError, TypeError):
            continue
        if old is None or new is None or old[0] != new[0]:
            return None
        return sorted(new[1] - old[1]), sorted(old[1] - new[1])
    return None


def _collection_entries(value: str) -> tuple[str, set[str]] | None:
    """The kind of a `dict`, `set`, or `frozenset` literal, and its entries as source text."""
    node = ast.parse(value, mode='eval').body
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ('set', 'frozenset')
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    if isinstance(node, ast.Dict):
        items: dict[object, object] = ast.literal_eval(node)
        return 'dict', {f'{key!r}: {item!r}' for key, item in items.items()}
    if isinstance(node, ast.Set):
        members: set[object] = ast.literal_eval(node)
        return 'set', {repr(member) for member in members}
    return None


def _literal_members(value: str) -> tuple[tuple[str, ...], set[str]] | None:
    """Split a type alias value into the shape around its `Literal` arms and their string members."""
    node = ast.parse(value, mode='eval').body
    alias_name = ''
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'TypeAliasType':
        if len(node.args) != 2 or node.keywords:
            return None
        alias_name = ast.dump(node.args[0])
        node = node.args[1]
    shape: list[str] = []
    members: set[str] = set()
    arms = [node]
    while arms:
        arm = arms.pop()
        if isinstance(arm, ast.BinOp) and isinstance(arm.op, ast.BitOr):
            arms += (arm.left, arm.right)
        elif isinstance(arm, ast.Subscript) and isinstance(arm.value, ast.Name) and arm.value.id == 'Literal':
            elements = arm.slice.elts if isinstance(arm.slice, ast.Tuple) else [arm.slice]
            for element in elements:
                if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
                    return None
                members.add(element.value)
        else:
            shape.append(ast.dump(arm))
    return (alias_name, *sorted(shape)), members


def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
    result = subprocess.run(
        [
            'uvx',
            '--quiet',
            '--from',
            f'griffecli=={GRIFFECLI_VERSION}',
            '--with',
            f'griffe-public-redundant-aliases=={GRIFFE_PUBLIC_REDUNDANT_ALIASES_VERSION}',
            'griffecli',
            'check',
            package,
            '--search',
            search_path,
            '--against',
            against,
            '--extensions',
            'griffe_public_redundant_aliases',
            '--format',
            'oneline',
            '--no-color',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    # `griffecli` prints findings to stderr and exits 1 when there are any, but an uncaught exception also exits 1.
    # Accept only output that parses and agrees with the exit code, so a crash can never read as a clean check.
    try:
        findings = parse_findings(package, result.stderr)
    except ValueError:
        findings = None
    if findings is None or result.returncode != (1 if findings else 0):
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f'griffecli check {package} failed with exit code {result.returncode}')
    return findings


def parse_findings(package: str, output: str) -> list[Finding]:
    findings: list[Finding] = []
    for output_line in output.splitlines():
        match = _FINDING_RE.fullmatch(output_line)
        if match is None:
            raise ValueError(f'Unexpected Griffe output: {output_line!r}')
        findings.append(
            Finding(
                package=package,
                path=match.group('path'),
                line=int(match.group('line')),
                message=match.group('message'),
            )
        )
    return findings


def emit_annotation(level: str, finding: Finding, message: str) -> None:
    escaped = message.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    print(f'::{level} file={finding.path},line={finding.line}::{escaped}')


if __name__ == '__main__':
    main()
