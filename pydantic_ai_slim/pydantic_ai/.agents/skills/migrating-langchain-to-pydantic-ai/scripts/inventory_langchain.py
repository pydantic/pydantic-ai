#!/usr/bin/env python3
"""Inventory LangChain, LangGraph, and LangSmith usage and flag Deep Agents for routing."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tokenize
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

FRAMEWORK_PREFIXES = ('langchain', 'langgraph', 'langsmith', 'deepagents')

IGNORED_PARTS = {
    '.git',
    '.mypy_cache',
    '.pytest_cache',
    '.ruff_cache',
    '.tox',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'node_modules',
    'site-packages',
}

CONFIG_NAMES = {
    'Pipfile',
    'Pipfile.lock',
    'environment.yml',
    'environment.yaml',
    'langgraph.json',
    'pdm.lock',
    'poetry.lock',
    'pyproject.toml',
    'setup.cfg',
    'setup.py',
    'uv.lock',
}

IMPORT_CATEGORIES = (
    ('deepagents', 'harness'),
    ('langgraph.checkpoint', 'persistence'),
    ('langgraph.store', 'persistence'),
    ('langgraph.types', 'graph-control'),
    ('langgraph', 'graph'),
    ('langsmith', 'observability-evals'),
    ('langchain.agents.middleware', 'middleware'),
    ('langchain.agents.structured_output', 'structured-output'),
    ('langchain.agents', 'agent'),
    ('langchain_core.tools', 'tools'),
    ('langchain.tools', 'tools'),
    ('langchain_community.tools', 'tools'),
    ('langchain_community.agent_toolkits', 'tools'),
    ('langchain_core.messages', 'messages'),
    ('langchain_core.prompts', 'prompts'),
    ('langchain_core.runnables', 'lcel-runtime'),
    ('langchain_core.retrievers', 'retrieval'),
    ('langchain_core.vectorstores', 'retrieval'),
    ('langchain.memory', 'memory'),
    ('langchain_', 'models-integrations'),
    ('langchain', 'other-langchain'),
)

CALL_CATEGORIES = {
    'create_agent': 'agent',
    'create_react_agent': 'agent',
    'create_tool_calling_agent': 'agent',
    'AgentExecutor': 'agent',
    'create_deep_agent': 'harness',
    'StateGraph': 'graph',
    'MessageGraph': 'graph',
    'ToolNode': 'graph-tools',
    'Command': 'graph-control',
    'Send': 'graph-control',
    'interrupt': 'human-in-loop',
    'bind_tools': 'model-tools',
    'with_structured_output': 'structured-output',
    'with_retry': 'retries',
    'invoke': 'invocation',
    'ainvoke': 'invocation',
    'stream': 'streaming',
    'astream': 'streaming',
    'astream_events': 'streaming',
    'compile': 'graph',
}

GENERIC_METHOD_CALLS = {'ainvoke', 'astream', 'astream_events', 'compile', 'invoke', 'stream'}
GENERIC_RECEIVER_HINTS = (
    'agent',
    'chain',
    'executor',
    'graph',
    'model',
    'researcher',
    'runnable',
    'runtime',
    'subgraph',
    'supervisor',
    'workflow',
)
GENERIC_RECEIVER_EXACT_NAMES = {'app'}


@dataclass(frozen=True)
class Finding:
    """One framework usage found in the target repository."""

    path: str
    line: int
    category: str
    kind: str
    symbol: str


def import_category(module: str) -> str:
    """Classify a framework module by migration concern."""
    for prefix, category in IMPORT_CATEGORIES:
        if module == prefix or module.startswith(f'{prefix}.') or module.startswith(prefix):
            return category
    return 'other'


def dotted_name(node: ast.AST) -> str:
    """Return the dotted name represented by an AST expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f'{prefix}.{node.attr}' if prefix else node.attr
    return ''


class InventoryVisitor(ast.NodeVisitor):
    """Collect framework imports, decorators, and calls from one module."""

    def __init__(
        self,
        relative_path: str,
        *,
        has_framework_import: bool,
        call_aliases: dict[str, str],
    ) -> None:
        self.relative_path = relative_path
        self.has_framework_import = has_framework_import
        self.call_aliases = call_aliases
        self.findings: list[Finding] = []

    def add(self, node: ast.AST, category: str, kind: str, symbol: str) -> None:
        self.findings.append(
            Finding(
                path=self.relative_path,
                line=getattr(node, 'lineno', 0),
                category=category,
                kind=kind,
                symbol=symbol,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.startswith(('langchain', 'langgraph', 'langsmith', 'deepagents')):
                self.add(node, import_category(alias.name), 'import', alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ''
        if module.startswith(('langchain', 'langgraph', 'langsmith', 'deepagents')):
            names = ', '.join(alias.name for alias in node.names)
            self.add(node, import_category(module), 'import', f'{module}: {names}')
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        leaf = name.rsplit('.', 1)[-1]
        canonical_leaf = self.call_aliases.get(leaf, leaf)
        category = CALL_CATEGORIES.get(canonical_leaf)
        if canonical_leaf in {'__import__', 'import_module'} and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                if first_arg.value.startswith(FRAMEWORK_PREFIXES):
                    self.add(node, 'dynamic-import', 'call', f'{name}({first_arg.value!r})')
        if leaf in GENERIC_METHOD_CALLS and leaf != 'compile':
            receiver = name.rsplit('.', 1)[0].lower() if '.' in name else ''
            receiver_leaf = receiver.rsplit('.', 1)[-1]
            if receiver_leaf not in GENERIC_RECEIVER_EXACT_NAMES and not any(
                hint in receiver for hint in GENERIC_RECEIVER_HINTS
            ):
                self.generic_visit(node)
                return
        if leaf == 'compile':
            receiver = name.rsplit('.', 1)[0].lower() if '.' in name else ''
            if 'graph' not in receiver and 'builder' not in receiver:
                self.generic_visit(node)
                return
        if leaf in GENERIC_METHOD_CALLS and not self.has_framework_import:
            if category:
                self.add(node, f'candidate-{category}', 'candidate-call', name)
            self.generic_visit(node)
            return
        if category:
            self.add(node, category, 'call', name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_decorators(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_decorators(node)
        self.generic_visit(node)

    def _visit_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            name = dotted_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
            leaf = name.rsplit('.', 1)[-1]
            if self.has_framework_import and self.call_aliases.get(leaf, leaf) == 'tool':
                self.add(decorator, 'tools', 'decorator', name)


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield relevant Python source files below a scan root."""
    if root.is_file():
        if root.suffix == '.py':
            yield root
        return
    for path in sorted(root.rglob('*.py')):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if not any(part in IGNORED_PARTS for part in relative_parts):
            yield path


def is_config_file(path: Path) -> bool:
    """Return whether a file may declare or embed framework dependencies."""
    return (
        path.name in CONFIG_NAMES
        or (path.name.startswith('requirements') and path.suffix in {'.in', '.txt'})
        or path.suffix == '.ipynb'
    )


def iter_config_files(root: Path) -> Iterable[Path]:
    """Yield relevant notebooks and dependency or configuration files."""
    if root.is_file():
        if is_config_file(root):
            yield root
        return
    for path in sorted(candidate for candidate in root.rglob('*') if candidate.is_file()):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if not any(part in IGNORED_PARTS for part in relative_parts) and is_config_file(path):
            yield path


def call_aliases(tree: ast.AST) -> dict[str, str]:
    """Map locally aliased framework imports to their original names."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ''
        if not module.startswith(FRAMEWORK_PREFIXES):
            continue
        for alias in node.names:
            aliases[alias.asname or alias.name] = alias.name
    return aliases


def scan_config_file(path: Path, relative_path: str) -> list[Finding]:
    """Find framework text in one notebook or dependency file."""
    findings: list[Finding] = []
    lines = path.read_text(encoding='utf-8').splitlines()
    if path.suffix == '.lock' or path.name in {'uv.lock', 'Pipfile.lock'}:
        for prefix in FRAMEWORK_PREFIXES:
            matching_lines = [line_number for line_number, line in enumerate(lines, start=1) if prefix in line.lower()]
            if matching_lines:
                findings.append(
                    Finding(
                        relative_path,
                        matching_lines[0],
                        'dependency-config',
                        'text-summary',
                        f'contains {prefix} on {len(matching_lines)} line(s)',
                    )
                )
        return findings
    for line_number, line in enumerate(lines, start=1):
        lowered = line.lower()
        matched = next((prefix for prefix in FRAMEWORK_PREFIXES if prefix in lowered), None)
        if matched is None:
            continue
        category = 'notebook' if path.suffix == '.ipynb' else 'dependency-config'
        symbol = ' '.join(line.strip().split())
        if len(symbol) > 180:
            symbol = symbol[:177] + '...'
        findings.append(Finding(relative_path, line_number, category, 'text', symbol))
    return findings


def scan(root: Path) -> tuple[list[Finding], list[dict[str, object]], int, int]:
    """Scan a target and return findings, errors, and file counts."""
    findings: list[Finding] = []
    errors: list[dict[str, object]] = []
    python_files_scanned = 0
    config_files_scanned = 0
    base = root if root.is_dir() else root.parent
    for path in iter_python_files(root):
        python_files_scanned += 1
        relative_path = str(path.relative_to(base))
        try:
            with tokenize.open(path) as source_file:
                tree = ast.parse(source_file.read(), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            error: dict[str, object] = {
                'path': relative_path,
                'line': getattr(exc, 'lineno', 0) or 0,
                'error': f'{type(exc).__name__}: {exc}',
            }
            if isinstance(exc, SyntaxError):
                error['hint'] = (
                    'The source may require a newer Python grammar. Re-run this scanner '
                    'with a Python version supported by the target repository.'
                )
            errors.append(error)
            continue
        has_framework_import = any(
            (isinstance(node, ast.Import) and any(alias.name.startswith(FRAMEWORK_PREFIXES) for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and (node.module or '').startswith(FRAMEWORK_PREFIXES))
            for node in ast.walk(tree)
        )
        visitor = InventoryVisitor(
            relative_path,
            has_framework_import=has_framework_import,
            call_aliases=call_aliases(tree),
        )
        visitor.visit(tree)
        findings.extend(visitor.findings)
    for path in iter_config_files(root):
        config_files_scanned += 1
        relative_path = str(path.relative_to(base))
        try:
            findings.extend(scan_config_file(path, relative_path))
        except (OSError, UnicodeError) as exc:
            errors.append(
                {
                    'path': relative_path,
                    'line': 0,
                    'error': f'{type(exc).__name__}: {exc}',
                }
            )
    findings.sort(key=lambda item: (item.path, item.line, item.category, item.symbol))
    return findings, errors, python_files_scanned, config_files_scanned


def build_report(
    root: Path,
    findings: list[Finding],
    errors: list[dict[str, object]],
    python_files_scanned: int,
    config_files_scanned: int,
) -> dict[str, object]:
    """Build the structured inventory report."""
    category_counts = Counter(finding.category for finding in findings)
    file_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for finding in findings:
        file_counts[finding.path][finding.category] += 1
    hotspots = [
        {
            'path': path,
            'count': sum(counts.values()),
            'categories': dict(sorted(counts.items())),
        }
        for path, counts in sorted(file_counts.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    ]
    routing = []
    if category_counts['harness'] or any(
        finding.category == 'dependency-config' and 'deepagents' in finding.symbol.lower() for finding in findings
    ):
        routing.append(
            {
                'condition': 'Deep Agents usage detected',
                'skill': 'migrate-deep-agents-to-pydantic-ai',
                'reason': 'Use the dedicated skill for create_deep_agent, Harness, sandbox, and deployment semantics.',
            }
        )
    return {
        'root': str(root.resolve()),
        'scanner_python': (f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'),
        'python_files_scanned': python_files_scanned,
        'config_files_scanned': config_files_scanned,
        'files_with_findings': len(file_counts),
        'finding_count': len(findings),
        'category_counts': dict(sorted(category_counts.items())),
        'hotspots': hotspots,
        'findings': [asdict(finding) for finding in findings],
        'parse_errors': errors,
        'routing': routing,
        'limitations': [
            'Candidate generic calls are name/receiver heuristics and require manual confirmation.',
            'JavaScript/TypeScript, generated code, and arbitrary plugin registries are not scanned.',
            'A clean report is not sufficient evidence to remove dependencies.',
        ],
    }


def markdown_report(report: dict[str, object], *, details: bool, max_hotspots: int) -> str:
    """Render a structured inventory report as Markdown."""
    lines = [
        '# LangChain Migration Inventory',
        '',
        f'- Root: `{report["root"]}`',
        f'- Scanner Python: {report["scanner_python"]}',
        f'- Python files scanned: {report["python_files_scanned"]}',
        f'- Notebook/dependency/config files scanned: {report["config_files_scanned"]}',
        f'- Files with findings: {report["files_with_findings"]}',
        f'- Findings: {report["finding_count"]}',
        f'- Parse errors: {len(report["parse_errors"])}',
        '- Candidate calls are heuristic and require manual confirmation.',
        '',
        '## Categories',
        '',
        '| Category | Count |',
        '|---|---:|',
    ]
    category_counts = report['category_counts']
    assert isinstance(category_counts, dict)
    if category_counts:
        lines.extend(f'| {category} | {count} |' for category, count in category_counts.items())
    else:
        lines.append('| none | 0 |')

    lines.extend(['', '## Hotspots', '', '| File | Findings | Categories |', '|---|---:|---|'])
    hotspots = report['hotspots']
    assert isinstance(hotspots, list)
    for hotspot in hotspots[:max_hotspots]:
        assert isinstance(hotspot, dict)
        categories = hotspot['categories']
        assert isinstance(categories, dict)
        rendered_categories = ', '.join(f'{key}={value}' for key, value in categories.items())
        path = str(hotspot['path']).replace('|', '\\|')
        lines.append(f'| `{path}` | {hotspot["count"]} | {rendered_categories} |')
    if not hotspots:
        lines.append('| none | 0 | |')

    routing = report['routing']
    assert isinstance(routing, list)
    if routing:
        lines.extend(['', '## Route to another skill', ''])
        for route in routing:
            assert isinstance(route, dict)
            lines.append(f'- `{route["condition"]}` → `{route["skill"]}`: {route["reason"]}')

    if details:
        lines.extend(['', '## Findings', '', '| File | Line | Category | Kind | Symbol |', '|---|---:|---|---|---|'])
        findings = report['findings']
        assert isinstance(findings, list)
        for finding in findings:
            assert isinstance(finding, dict)
            path = str(finding['path']).replace('|', '\\|')
            symbol = str(finding['symbol']).replace('|', '\\|').replace('`', "'")
            lines.append(f'| `{path}` | {finding["line"]} | {finding["category"]} | {finding["kind"]} | `{symbol}` |')

    parse_errors = report['parse_errors']
    assert isinstance(parse_errors, list)
    if parse_errors:
        lines.extend(['', '## Parse errors', ''])
        for error in parse_errors:
            assert isinstance(error, dict)
            lines.append(f'- `{error["path"]}:{error["line"]}` — {error["error"]}')
            if hint := error.get('hint'):
                lines.append(f'  - Hint: {hint}')
    return '\n'.join(lines) + '\n'


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Inventory Python framework usage before a LangChain-to-Pydantic-AI migration.',
        epilog=(
            'Run with a Python version supported by the target repository so ast.parse() '
            'understands its syntax. Example: uv run --python 3.12 python '
            'scripts/inventory_langchain.py /path/to/repository --format json'
        ),
    )
    parser.add_argument('root', nargs='?', default='.', help='Repository, directory, or Python file')
    parser.add_argument('--format', choices=('markdown', 'json'), default='markdown')
    parser.add_argument('--details', action='store_true', help='Include every finding in Markdown')
    parser.add_argument('--max-hotspots', type=int, default=30, help='Maximum hotspot rows')
    parser.add_argument(
        '--allow-parse-errors',
        action='store_true',
        help='Exit successfully despite unreadable or unparsable files (exploratory use only)',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the inventory command and return its exit status."""
    args = parse_args(argv or sys.argv[1:])
    root = Path(args.root).expanduser()
    if not root.exists():
        print(f'error: path does not exist: {root}', file=sys.stderr)
        return 2
    if args.max_hotspots < 0:
        print('error: --max-hotspots must be non-negative', file=sys.stderr)
        return 2

    findings, errors, python_files_scanned, config_files_scanned = scan(root)
    report = build_report(
        root,
        findings,
        errors,
        python_files_scanned,
        config_files_scanned,
    )
    if args.format == 'json':
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(markdown_report(report, details=args.details, max_hotspots=args.max_hotspots), end='')
    if errors and not args.allow_parse_errors:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
