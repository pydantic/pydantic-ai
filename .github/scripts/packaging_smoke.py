"""Install each published workspace package on its own and import every module it ships.

`uv.lock` installs every workspace package, extra and dev group into one environment, so a shipped module that
imports a package its own distribution does not declare passes CI and fails a clean install. In
https://github.com/pydantic/pydantic-ai/pull/9620, `pydantic-evals` and `pydantic-ai-harness` imported `sniffio`,
which AnyIO 4.12 stopped installing.

This builds the wheels, then installs each published package bare and once per extra into a fresh virtual
environment. The other workspace packages resolve from the built wheels, and everything else from PyPI, as a
user's install resolves it: no lock, no workspace settings. Each environment then imports every module the
package ships. A module passes when it imports, or when it raises an `ImportError` whose message names an extra of
a published package to install, such as `pip install "pydantic-ai-slim[mcp]"`. Anything else fails the shape:
above all a `ModuleNotFoundError` with no install hint, and a hint that names the extra the shape installed, since
following it changes nothing. That hint still passes when a marker left one of the extra's requirements out on
this interpreter, such as `pydantic-ai-slim[voyageai]` on Python 3.14. `BASELINE` lists the failures `main` had
when this check was added; it only shrinks.

Run it with `uv run --no-project --with packaging python .github/scripts/packaging_smoke.py`. Each environment
runs the interpreter that runs this script, so markers evaluated here match the environments.
"""

from __future__ import annotations

import argparse
import email
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# The script runs without the project's environment, where `typing_extensions` may be absent; nothing validates this
# `TypedDict` with pydantic.
from typing import TypedDict  # noqa: TID251

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]

EXCLUDED_DISTRIBUTIONS: dict[str, str] = {
    'pydantic-ai': (
        'Ships no modules of its own. `clai` installs it bare, and each of its extras forwards to a '
        '`pydantic-ai-slim` extra whose own shape imports the modules it enables.'
    ),
    'pydantic-ai-examples': (
        'Runnable example scripts rather than a library: many build a provider model at import, which needs an API key.'
    ),
}

# Extras that cannot install on Linux at all, keyed by shape label. An extra whose requirements a marker leaves out
# on this interpreter installs nothing extra and still runs.
EXCLUDED_SHAPES: dict[str, str] = {}

SKIPPED_MODULES: dict[str, str] = {
    'pydantic_ai.workspaces.conformance': (
        'A pytest suite for workspace backend authors, run under pytest, which its docstring names as a requirement.'
    ),
}

GUARDED = "private module, imported only behind its extra's install hint"
MCP_SAMPLING = 'known defect: `pydantic_ai.models.mcp_sampling` needs `mcp` and gives no install hint'
OPENAI_OAUTH = 'known defect: `providers.openai_chatgpt` and `providers.openai_codex` give no install hint for `openai`'
REALTIME_OPENAI = 'known defect: `realtime.azure` and `realtime.openai_live` give no install hint for their extra'
WEB_UI = 'known defect: `Agent.to_web()`, `clai web` and `pydantic_ai.ui.DEFAULT_HTML_URL` give no hint for `web`'
HARNESS_HTTP = 'known defect: the harness module needs `httpx` or `websockets` and gives no install hint'
HARNESS_HINT = 'known defect: the harness module gives no install hint for its extra'
HARNESS_MONGO = 'known defect: `MongoMediaStore` and `MongoStepStore` give no install hint for `mongodb`'

# Every import failure on `main` at the time this smoke test was added, as (module, the module it could not import,
# or the exception's class when there is none). The list only shrinks: a failure not listed here fails its shape,
# and a full run fails on an entry no shape reports any more, so a fix removes its entry. A known defect is tracked
# for a fix; a guarded private module stays, because no import path reaches it without its extra.
BASELINE: dict[tuple[str, str], str] = {
    ('pydantic_ai._cli.web', 'starlette'): WEB_UI,
    ('pydantic_ai._mcp', 'mcp'): MCP_SAMPLING,
    ('pydantic_ai._mcp_compat', 'mcp'): MCP_SAMPLING,
    ('pydantic_ai.durable_exec.temporal._replay_safe_tracer_provider', 'opentelemetry.sdk'): GUARDED,
    ('pydantic_ai.images._xai_geometry', 'xai_sdk'): GUARDED,
    ('pydantic_ai.models._google_batch', 'google'): GUARDED,
    ('pydantic_ai.models._google_batch', 'google.genai'): GUARDED,
    ('pydantic_ai.models.mcp_sampling', 'mcp'): MCP_SAMPLING,
    ('pydantic_ai.models.openai._responses_websocket', 'websockets'): GUARDED,
    ('pydantic_ai.providers._oauth_provider', 'openai'): OPENAI_OAUTH,
    ('pydantic_ai.providers._openai_compatible', 'openai'): OPENAI_OAUTH,
    ('pydantic_ai.providers.openai_chatgpt', 'openai'): OPENAI_OAUTH,
    ('pydantic_ai.providers.openai_codex', 'openai'): OPENAI_OAUTH,
    ('pydantic_ai.realtime._openai_lifecycle', 'openai'): GUARDED,
    ('pydantic_ai.realtime._openai_lifecycle', 'websockets'): GUARDED,
    ('pydantic_ai.realtime._openai_protocol', 'openai'): REALTIME_OPENAI,
    ('pydantic_ai.realtime._openai_protocol', 'websockets'): REALTIME_OPENAI,
    ('pydantic_ai.realtime.azure', 'openai'): REALTIME_OPENAI,
    ('pydantic_ai.realtime.azure', 'websockets'): REALTIME_OPENAI,
    ('pydantic_ai.realtime.openai_live', 'websockets'): REALTIME_OPENAI,
    ('pydantic_ai.ui._web', 'starlette'): WEB_UI,
    ('pydantic_ai_harness._mongo_driver_info', 'pymongo'): HARNESS_MONGO,
    ('pydantic_ai_harness._monty_turns', 'pydantic_monty'): GUARDED,
    ('pydantic_ai_harness.exa', 'httpx'): HARNESS_HTTP,
    ('pydantic_ai_harness.experimental.acp', 'acp'): HARNESS_HINT,
    ('pydantic_ai_harness.logfire', 'logfire'): HARNESS_HINT,
    ('pydantic_ai_harness.media._mongo', 'bson'): HARNESS_MONGO,
    # Raises a `UserError` that names the `pydantic-ai-slim` extra to install.
    ('pydantic_ai_harness.researcher._agent', 'bs4'): GUARDED,
    ('pydantic_ai_harness.researcher._agent', 'duckduckgo_search'): GUARDED,
    ('pydantic_ai_harness.sprites_sandbox', 'httpx'): HARNESS_HTTP,
    ('pydantic_ai_harness.sprites_sandbox', 'websockets'): HARNESS_HTTP,
    ('pydantic_ai_harness.step_persistence._mongo', 'pymongo'): HARNESS_MONGO,
    ('pydantic_ai_harness.youdotcom', 'httpx'): HARNESS_HTTP,
    ('pydantic_evals.otel._context_in_memory_span_exporter', 'opentelemetry.sdk'): GUARDED,
}

# Names a requirement with extras, such as `pydantic-ai-slim[mcp]` or `pydantic-ai-slim[duckduckgo,web-fetch]`.
INSTALL_HINT = re.compile(r'([A-Za-z0-9][A-Za-z0-9._-]*)\[([A-Za-z0-9._, -]+)\]')

IMPORT_TIMEOUT_SECONDS = 900

# Runs in each environment's interpreter: imports every module under the given top-level packages and writes one
# JSON report. `os._exit` keeps a thread some import started from holding the interpreter open.
IMPORT_EVERY_MODULE = """
import importlib, json, os, pkgutil, sys, warnings

warnings.simplefilter('ignore')
report_path, skipped, *packages = sys.argv[1:]
skipped = set(filter(None, skipped.split(',')))
report = {'imported': 0, 'failures': []}


def missing_module(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ImportError) and exc.name:
            return exc.name
        exc = exc.__cause__ or exc.__context__
    return None


def attempt(name):
    try:
        importlib.import_module(name)
    except (Exception, SystemExit) as exc:
        report['failures'].append({
            'module': name,
            'exception': type(exc).__name__,
            'message': str(exc),
            'missing': missing_module(exc),
            'import_error': isinstance(exc, ImportError),
        })
        return False
    report['imported'] += 1
    return True


for package in packages:
    if not attempt(package):
        continue
    path = getattr(sys.modules[package], '__path__', None)
    if path is None:
        continue
    for info in pkgutil.walk_packages(path, f'{package}.', onerror=lambda name: None):
        if info.name.rpartition('.')[2] != '__main__' and info.name not in skipped:
            attempt(info.name)

with open(report_path, 'w') as f:
    json.dump(report, f)
sys.stdout.flush()
os._exit(0)
"""


@dataclass(frozen=True)
class Shape:
    """One install of a published distribution: bare, or with one extra."""

    distribution: str
    extra: str | None = None

    @property
    def label(self) -> str:
        """The requirement the shape installs, such as `pydantic-ai-slim[mcp]`."""
        return self.distribution if self.extra is None else f'{self.distribution}[{self.extra}]'


@dataclass(frozen=True)
class Wheel:
    """A built wheel and the metadata the smoke test reads from it."""

    path: Path
    name: str
    extras: tuple[str, ...]
    requirements: tuple[Requirement, ...]
    packages: tuple[str, ...]


@dataclass(frozen=True)
class ImportFailure:
    """A module that raised on import, as the environment's interpreter reported it."""

    module: str
    exception: str
    message: str
    missing: str | None
    import_error: bool

    @property
    def key(self) -> tuple[str, str]:
        """The failure's `BASELINE` key."""
        return self.module, self.missing or self.exception


@dataclass(frozen=True)
class ShapeResult:
    """How one shape installed and imported."""

    shape: Shape
    install_seconds: float
    import_seconds: float
    imported: int
    hints: int
    problems: tuple[str, ...]
    baselined: frozenset[tuple[str, str]] = frozenset()


def published_distributions(root: Path) -> set[str]:
    """The root project and every workspace member, as canonical distribution names."""
    pyproject = tomllib.loads((root / 'pyproject.toml').read_text())
    members: list[str] = pyproject['tool']['uv']['workspace']['members']
    names: set[str] = {canonicalize_name(pyproject['project']['name'])}
    for member in members:
        names.add(canonicalize_name(tomllib.loads((root / member / 'pyproject.toml').read_text())['project']['name']))
    return names


def read_wheel(path: Path) -> Wheel:
    """The distribution name, extras, requirements and top-level import packages of the wheel at `path`."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if re.fullmatch(r'[^/]+\.dist-info/METADATA', name))
        metadata = email.message_from_bytes(archive.read(metadata_name))
    packages = {name.split('/')[0] for name in names if re.fullmatch(r'[A-Za-z_]\w*/__init__\.py', name)}
    packages |= {name.removesuffix('.py') for name in names if re.fullmatch(r'[A-Za-z_]\w*\.py', name)}
    return Wheel(
        path=path,
        name=canonicalize_name(str(metadata['Name'])),
        extras=tuple(canonicalize_name(str(extra)) for extra in metadata.get_all('Provides-Extra', [])),
        requirements=tuple(Requirement(str(value)) for value in metadata.get_all('Requires-Dist', [])),
        packages=tuple(sorted(packages)),
    )


def enumerate_shapes(wheels: Iterable[Wheel]) -> list[Shape]:
    """Each published distribution bare, then once per extra, leaving out the excluded ones."""
    shapes: list[Shape] = []
    for wheel in sorted(wheels, key=lambda wheel: wheel.name):
        if wheel.name in EXCLUDED_DISTRIBUTIONS:
            continue
        shapes.append(Shape(wheel.name))
        shapes.extend(Shape(wheel.name, extra) for extra in sorted(wheel.extras))
    return [shape for shape in shapes if shape.label not in EXCLUDED_SHAPES]


def select_shapes(
    shapes: Sequence[Shape], *, packages: Collection[str], labels: Collection[str], shard: tuple[int, int]
) -> list[Shape]:
    """The shapes `packages` and `labels` name (all when both are empty), then every `count`th from `index`."""
    unknown = set(packages) - {shape.distribution for shape in shapes}
    unknown |= set(labels) - {shape.label for shape in shapes}
    if unknown:
        raise SystemExit(f'no such package or shape: {", ".join(sorted(unknown))}')
    chosen = [
        shape for shape in shapes if not (packages or labels) or shape.distribution in packages or shape.label in labels
    ]
    index, count = shard
    selected = chosen[index - 1 :: count]
    if not selected:
        raise SystemExit(f'shard {index}/{count} of {len(chosen)} shapes selects none')
    return selected


def installs_completely(wheel: Wheel, extra: str) -> bool:
    """Whether every requirement `extra` adds applies on this interpreter, so no marker left one out."""
    extra_marker = f'extra == "{extra}"'
    for requirement in wheel.requirements:
        marker = requirement.marker
        if marker is not None and extra_marker in str(marker) and not marker.evaluate({'extra': extra}):
            return False
    return True


def failure_problem(
    failure: ImportFailure, *, extras: Mapping[str, Collection[str]], installed: tuple[str, str] | None
) -> str | None:
    """Why `failure` fails its shape, or `None` when it is an install hint the shape does not satisfy.

    `extras` maps each published distribution to its extras. `installed` is the shape's own distribution and extra
    when installing that extra left none of its requirements out.
    """
    if not failure.import_error:
        return f'raised {failure.exception}, not an ImportError'
    hinted: set[tuple[str, str]] = set()
    for match in INSTALL_HINT.finditer(failure.message):
        distribution = canonicalize_name(match.group(1))
        if distribution in extras:
            hinted.update((distribution, canonicalize_name(extra)) for extra in match.group(2).split(','))
    if not hinted:
        return 'no install hint names an extra of a published package'
    if unknown := sorted(f'{name}[{extra}]' for name, extra in hinted if extra not in extras[name]):
        return f'the install hint names {", ".join(unknown)}, which is not an extra'
    if installed in hinted:
        return 'the install hint names the extra this shape installed'
    return None


def build_wheels(root: Path, out: Path) -> None:
    """Build every workspace package's wheel into `out`, as the release job builds them."""
    started = time.monotonic()
    subprocess.run(
        ['uv', 'build', '--all-packages', '--wheel', '--no-sources', '--quiet', '--out-dir', str(out)],
        cwd=root,
        check=True,
    )
    print(f'built the wheels in {time.monotonic() - started:.0f}s', flush=True)


def read_wheels(directory: Path, root: Path) -> dict[str, Wheel]:
    """The wheels in `directory` by distribution name, refusing a set that misses a published distribution."""
    wheels: dict[str, Wheel] = {}
    for path in sorted(directory.glob('*.whl')):
        wheel = read_wheel(path.resolve())
        if wheel.name in wheels:
            raise SystemExit(f'{directory} holds more than one wheel of {wheel.name}')
        wheels[wheel.name] = wheel
    if missing := published_distributions(root) - wheels.keys():
        raise SystemExit(f'{directory} holds no wheel of {", ".join(sorted(missing))}')
    return wheels


def tail(output: str, lines: int = 20) -> str:
    """The last `lines` lines of a command's output."""
    return '\n'.join(output.strip().splitlines()[-lines:])


def run_shape(shape: Shape, wheels: Mapping[str, Wheel], work: Path) -> ShapeResult:
    """Install `shape` into a fresh virtual environment under `work`, import its modules, and remove it."""
    wheel = wheels[shape.distribution]
    venv = work / re.sub(r'[^A-Za-z0-9]+', '-', shape.label)
    python = venv / 'bin' / 'python'
    started = time.monotonic()
    try:
        install = subprocess.run(
            ['uv', 'venv', '--no-config', '--quiet', '--python', sys.executable, str(venv)],
            capture_output=True,
            text=True,
        )
        if install.returncode == 0:
            # `--no-config` keeps the workspace's and the user's uv settings, such as `exclude-newer`, out of the
            # install. CPU PyTorch wheels import the same modules as the CUDA ones, at a fraction of the download.
            install = subprocess.run(
                [
                    'uv',
                    'pip',
                    'install',
                    '--no-config',
                    '--quiet',
                    '--python',
                    str(python),
                    '--find-links',
                    str(wheel.path.parent),
                    '--torch-backend',
                    'cpu',
                    f'{shape.label} @ {wheel.path.as_uri()}',
                ],
                cwd=work,
                capture_output=True,
                text=True,
            )
        installed_at = time.monotonic()
        if install.returncode != 0:
            return ShapeResult(shape, installed_at - started, 0, 0, 0, (f'install failed:\n{tail(install.stderr)}',))
        return import_every_module(shape, wheel, wheels, python, venv, installed_at - started)
    finally:
        shutil.rmtree(venv, ignore_errors=True)


def import_every_module(
    shape: Shape, wheel: Wheel, wheels: Mapping[str, Wheel], python: Path, venv: Path, install_seconds: float
) -> ShapeResult:
    """Import every module of `wheel` in the environment of `python`, and judge each failure."""
    report_path = venv / 'import-report.json'
    started = time.monotonic()
    try:
        child = subprocess.run(
            [
                str(python),
                '-I',
                '-c',
                IMPORT_EVERY_MODULE,
                str(report_path),
                ','.join(SKIPPED_MODULES),
                *wheel.packages,
            ],
            cwd=venv,
            capture_output=True,
            text=True,
            timeout=IMPORT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        problem = f'importing the modules took longer than {IMPORT_TIMEOUT_SECONDS}s'
        return ShapeResult(shape, install_seconds, time.monotonic() - started, 0, 0, (problem,))
    import_seconds = time.monotonic() - started
    if child.returncode != 0 or not report_path.exists():
        problem = f'the import run exited {child.returncode}:\n{tail(child.stderr)}'
        return ShapeResult(shape, install_seconds, import_seconds, 0, 0, (problem,))
    report = json.loads(report_path.read_text())
    failures = [ImportFailure(**failure) for failure in report['failures']]
    extras = {name: other.extras for name, other in wheels.items()}
    installed = (
        (shape.distribution, shape.extra)
        if shape.extra is not None and installs_completely(wheel, shape.extra)
        else None
    )
    problems: list[str] = []
    baselined: set[tuple[str, str]] = set()
    for failure in failures:
        if (problem := failure_problem(failure, extras=extras, installed=installed)) is None:
            continue
        if failure.key in BASELINE:
            baselined.add(failure.key)
            continue
        missing = f' (missing `{failure.missing}`)' if failure.missing else ''
        message = next(iter(failure.message.strip().splitlines()), '')
        problems.append(f'{failure.module}: {failure.exception}: {message}{missing}: {problem}')
    hints = len(failures) - len(problems) - len(baselined)
    return ShapeResult(
        shape, install_seconds, import_seconds, report['imported'], hints, tuple(problems), frozenset(baselined)
    )


class ShardReport(TypedDict):
    """What `--report` writes: the shard, whether a filter narrowed its shapes, and the entries they reported."""

    shard: list[int]
    filtered: bool
    baselined: list[list[str]]


def stale_entries(reports: Sequence[ShardReport]) -> list[tuple[str, str]]:
    """The `BASELINE` entries no shape reported, from the `--report` files of every shard of one full run."""
    shards: set[tuple[int, int]] = set()
    seen: set[tuple[str, str]] = set()
    for report in reports:
        if report['filtered']:
            raise SystemExit('a report comes from a run filtered by --package or --shape, which runs too few shapes')
        index, count = report['shard']
        shards.add((index, count))
        seen.update((module, missing) for module, missing in report['baselined'])
    counts = {count for _, count in shards}
    if len(counts) != 1 or shards != {(index, count) for count in counts for index in range(1, count + 1)}:
        raise SystemExit(f'the reports hold shards {sorted(shards)}, not every shard of one run')
    return sorted(BASELINE.keys() - seen)


def parse_shard(value: str) -> tuple[int, int]:
    """Parse `I/N`, shard `I` of `N`."""
    match = re.fullmatch(r'(\d+)/(\d+)', value)
    if match is None or not 1 <= int(match.group(1)) <= int(match.group(2)):
        raise argparse.ArgumentTypeError(f'expected I/N with 1 <= I <= N, got {value!r}')
    return int(match.group(1)), int(match.group(2))


def main(argv: Sequence[str] | None = None) -> int:
    """Build or read the wheels, run the selected shapes, and report every failure."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0] if __doc__ else None)
    parser.add_argument('--root', type=Path, default=ROOT, help='the workspace to build (default: this checkout)')
    parser.add_argument('--wheels', type=Path, help='use the wheels already built in this directory')
    parser.add_argument('--package', action='append', default=[], help='run every shape of this distribution')
    parser.add_argument('--shape', action='append', default=[], help='run this shape, such as pydantic-ai-slim[mcp]')
    parser.add_argument('--shard', type=parse_shard, default=(1, 1), help='run shard I/N of the selected shapes')
    parser.add_argument('--jobs', type=int, default=2, help='shapes to install and import at once')
    parser.add_argument('--report', type=Path, help='write the baseline entries the shapes reported to this file')
    parser.add_argument(
        '--stale-check',
        type=Path,
        nargs='+',
        metavar='REPORT',
        help='instead of running shapes, fail on a baseline entry that no shard of one full run reported',
    )
    args = parser.parse_args(argv)

    if args.stale_check:
        reports: list[ShardReport] = [json.loads(path.read_text()) for path in args.stale_check]
        return report_stale(stale_entries(reports))

    with tempfile.TemporaryDirectory(prefix='packaging-smoke-') as temp:
        work = Path(temp)
        wheel_dir: Path | None = args.wheels
        if wheel_dir is None:
            wheel_dir = work / 'wheels'
            build_wheels(args.root, wheel_dir)
        wheels = read_wheels(wheel_dir, args.root)
        shapes = select_shapes(
            enumerate_shapes(wheels.values()),
            packages={canonicalize_name(name) for name in args.package},
            labels=set(args.shape),
            shard=args.shard,
        )
        print(f'running {len(shapes)} shapes, {args.jobs} at a time', flush=True)
        results: list[ShapeResult] = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for future in as_completed(pool.submit(run_shape, shape, wheels, work) for shape in shapes):
                result = future.result()
                results.append(result)
                status = 'FAIL' if result.problems else 'PASS'
                print(
                    f'{status} {result.shape.label}: {result.imported} modules imported, {result.hints} install hints'
                    f' (install {result.install_seconds:.1f}s, import {result.import_seconds:.1f}s)',
                    flush=True,
                )

    failed = sorted((result for result in results if result.problems), key=lambda result: result.shape.label)
    for result in failed:
        print(f'\n{result.shape.label}:')
        for problem in result.problems:
            print(f'  {problem}')
    if failed:
        print(f'\n{len(failed)} of {len(results)} shapes failed', file=sys.stderr)
    report: ShardReport = {
        'shard': list(args.shard),
        'filtered': bool(args.package or args.shape),
        'baselined': [list(key) for key in sorted({key for result in results for key in result.baselined})],
    }
    if args.report is not None:
        args.report.write_text(json.dumps(report))
    stale = 0 if report['filtered'] or args.shard != (1, 1) else report_stale(stale_entries([report]))
    return 1 if failed or stale else 0


def report_stale(stale: Sequence[tuple[str, str]]) -> int:
    """Print each stale `BASELINE` entry, and return the exit status they give."""
    for module, missing in stale:
        print(f'{module} now imports without `{missing}` failing: remove its `BASELINE` entry', file=sys.stderr)
    return 1 if stale else 0


if __name__ == '__main__':
    sys.exit(main())
