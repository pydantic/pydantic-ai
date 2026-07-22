"""Mode B: dual-path durable wrapper method coverage (AST + string presence)."""

from __future__ import annotations

import ast
from pathlib import Path

from ..models import Finding, ScanContext, Severity

CHECK = 'mode_b'

BACKENDS = ('temporal', 'dbos', 'prefect')
WRAPPER_CLASS = {
    'temporal': 'TemporalAgent',
    'dbos': 'DBOSAgent',
    'prefect': 'PrefectAgent',
}

# Methods that wrappers intentionally may not mirror 1:1 (properties / factories).
# Keep small; expand via mode_b_allowlist.txt if needed.
DEFAULT_ALLOWLIST = frozenset(
    {
        'from_spec',
        'from_file',
        'instrument_all',
        'instrument',
        'render_description',
        'deps_type',
        'output_type',
        'root_capability',
        'system_prompt_parts',
        'output_json_schema',
        # wrapper-specific
        'temporal_activities',
    }
)


def _public_methods(tree: ast.AST, class_name: str) -> set[str]:
    methods: set[str] = set()
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = item.name
                    if name.startswith('_'):
                        continue
                    methods.add(name)
            break
    return methods


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding='utf-8', errors='replace'))
    except (OSError, SyntaxError):
        return None


def _load_allowlist(repo: Path, package_dir: Path | None) -> frozenset[str]:
    names = set(DEFAULT_ALLOWLIST)
    candidates: list[Path] = []
    if package_dir is not None:
        candidates.append(package_dir / 'mode_b_allowlist.txt')
    candidates.append(repo / 'mode_b_allowlist.txt')
    for c in candidates:
        if c.is_file():
            for line in c.read_text(encoding='utf-8', errors='replace').splitlines():
                s = line.strip()
                if s and not s.startswith('#'):
                    names.add(s)
            break
    return frozenset(names)


def _find_agent_file(repo: Path) -> Path | None:
    candidates = [
        repo / 'pydantic_ai_slim/pydantic_ai/agent/__init__.py',
        repo / 'pydantic_ai_slim/pydantic_ai/agent.py',
    ]
    for c in candidates:
        if c.is_file():
            return c
    # agent/*.py search
    agent_dir = repo / 'pydantic_ai_slim/pydantic_ai/agent'
    if agent_dir.is_dir():
        for p in sorted(agent_dir.glob('*.py')):
            tree = _parse(p)
            if tree and _public_methods(tree, 'Agent'):
                return p
    return None


def _package_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def run(ctx: ScanContext) -> list[Finding]:
    findings: list[Finding] = []
    repo = ctx.repo
    de = repo / 'pydantic_ai_slim/pydantic_ai/durable_exec'
    allow = _load_allowlist(repo, _package_dir())

    # Dual-path file existence
    for backend in BACKENDS:
        bdir = de / backend
        for fname in ('_agent.py', '_durability.py'):
            fpath = bdir / fname
            if not fpath.is_file():
                findings.append(
                    Finding(
                        check=CHECK,
                        severity=Severity.ERROR,
                        path=f'pydantic_ai_slim/pydantic_ai/durable_exec/{backend}/{fname}',
                        message=f'missing dual-path file for {backend}',
                        rule_id='mode_b.missing_dual_path_file',
                    )
                )

    restate = de / 'restate'
    if restate.exists():
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.INFO,
                path='pydantic_ai_slim/pydantic_ai/durable_exec/restate',
                message='`durable_exec/restate/` package directory exists (expected removed)',
                rule_id='mode_b.restate_present',
            )
        )

    agent_path = _find_agent_file(repo)
    if agent_path is None:
        findings.append(
            Finding(
                check=CHECK,
                severity=Severity.WARNING,
                path='pydantic_ai_slim/pydantic_ai/agent',
                message='could not locate primary `Agent` class for Mode B comparison',
                rule_id='mode_b.agent_not_found',
            )
        )
        return findings

    agent_tree = _parse(agent_path)
    abstract_path = repo / 'pydantic_ai_slim/pydantic_ai/agent/abstract.py'
    abstract_methods: set[str] = set()
    if abstract_path.is_file():
        at = _parse(abstract_path)
        if at:
            abstract_methods = _public_methods(at, 'AbstractAgent')

    agent_methods: set[str] = set()
    if agent_tree:
        agent_methods = _public_methods(agent_tree, 'Agent')
    # Union with AbstractAgent — wrappers typically override AbstractAgent surface
    core_surface = (agent_methods | abstract_methods) - allow

    wrapper_methods: dict[str, set[str]] = {}
    durability_blobs: dict[str, str] = {}
    for backend in BACKENDS:
        ap = de / backend / '_agent.py'
        dp = de / backend / '_durability.py'
        methods: set[str] = set()
        if ap.is_file():
            tree = _parse(ap)
            if tree:
                methods = _public_methods(tree, WRAPPER_CLASS[backend])
        wrapper_methods[backend] = methods - allow
        if dp.is_file():
            try:
                durability_blobs[backend] = dp.read_text(encoding='utf-8', errors='replace')
            except OSError:
                durability_blobs[backend] = ''
        else:
            durability_blobs[backend] = ''

    # Sibling inconsistency: method on some wrappers but not all
    all_wrapper_names = set().union(*wrapper_methods.values()) if wrapper_methods else set()
    for name in sorted(all_wrapper_names):
        present = [b for b in BACKENDS if name in wrapper_methods.get(b, set())]
        missing = [b for b in BACKENDS if name not in wrapper_methods.get(b, set())]
        if present and missing:
            findings.append(
                Finding(
                    check=CHECK,
                    severity=Severity.WARNING,
                    path='pydantic_ai_slim/pydantic_ai/durable_exec',
                    message=(
                        f'wrapper method `{name}` present on {", ".join(present)} '
                        f'but missing on {", ".join(missing)}'
                    ),
                    rule_id='mode_b.wrapper_sibling_gap',
                )
            )

    # Agent methods absent from all three wrappers AND from durability modules
    for name in sorted(core_surface):
        on_any_wrapper = any(name in wrapper_methods.get(b, set()) for b in BACKENDS)
        in_any_durability = any(name in durability_blobs.get(b, '') for b in BACKENDS)
        if not on_any_wrapper and not in_any_durability:
            # Too noisy for every property — only report run* / iter / override class of I/O methods
            if not _is_high_signal_method(name):
                continue
            findings.append(
                Finding(
                    check=CHECK,
                    severity=Severity.INFO,
                    path=ctx.rel(agent_path),
                    message=(
                        f'Agent/AbstractAgent public method `{name}` not found on any durable '
                        f'wrapper class or `*_durability.py` string surface'
                    ),
                    rule_id='mode_b.unwrapped_agent_method',
                )
            )

    # Note: we intentionally do NOT flag "wrapper method not mentioned in *_durability.py"
    # by string search — durability binds via for_agent/activity registration, not method-name
    # text, so that check is almost always noise under dual-path.

    # MCPToolset existence (anchor for Mode B surface)
    mcp = repo / 'pydantic_ai_slim/pydantic_ai/mcp.py'
    if mcp.is_file():
        tree = _parse(mcp)
        if tree and not _public_methods(tree, 'MCPToolset') and 'class MCPToolset' not in mcp.read_text(
            encoding='utf-8', errors='replace'
        ):
            findings.append(
                Finding(
                    check=CHECK,
                    severity=Severity.WARNING,
                    path=ctx.rel(mcp),
                    message='`MCPToolset` class not found in mcp.py',
                    rule_id='mode_b.mcp_toolset_missing',
                )
            )

    return findings


def _is_high_signal_method(name: str) -> bool:
    if name in {
        'run',
        'run_sync',
        'run_stream',
        'run_stream_sync',
        'run_stream_events',
        'iter',
        'override',
        'toolsets',
        'model',
        'name',
        'event_stream_handler',
        'description',
        'aclose',
    }:
        return True
    if name.startswith('run_') or name.startswith('iter'):
        return True
    return False
