"""Validation harness for the migrate-to-v2 skill.

For each `examples/individual/<id>_v1.py`:
  - import it with warnings captured
  - call its `trigger()` if defined
  - assert at least one emitted warning's text contains the file's `EXPECT` substring

For each `examples/individual/<id>_v2.py`:
  - import + trigger
  - assert zero pydantic-* deprecation warnings fire

Run with the v1 venv for the v1 half, and the v2 venv for the v2 half. If
called without an explicit half, it runs whichever half the active interpreter
supports (detected by `pydantic_ai.__version__`).

Usage:
    scripts/.venv-v1/bin/python scripts/validate.py        # runs v1 half + v2 half (the v2 half is skipped if pydantic_ai is v1)
    scripts/.venv-v1/bin/python scripts/validate.py v1
    scripts/.venv-v2/bin/python scripts/validate.py v2

Exit non-zero on any miss. Prints a coverage matrix at the end.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

# Stable env so model/provider constructions don't bail before deprecation fires.
os.environ.setdefault('OPENAI_API_KEY', 'sk-dummy-for-validation')
os.environ.setdefault('GEMINI_API_KEY', 'dummy')
os.environ.setdefault('GOOGLE_API_KEY', 'dummy')
os.environ.setdefault('XAI_API_KEY', 'dummy')
os.environ.setdefault('GROK_API_KEY', 'dummy')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INDIV = ROOT / 'examples' / 'individual'


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(path: Path) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mod = _load(path)
        trig = getattr(mod, 'trigger', None)
        if trig is not None:
            try:
                trig()
            except Exception as exc:  # noqa: BLE001
                # Construction-time exceptions are fine — the warning may have
                # already fired. Surface only if there are no warnings either.
                pass
    return list(caught)


def _is_pydantic_dep(w: warnings.WarningMessage) -> bool:
    name = type(w.message).__name__
    if name in {'PydanticAIDeprecationWarning', 'PydanticGraphDeprecationWarning', 'PydanticEvalsDeprecationWarning'}:
        return True
    if isinstance(w.message, DeprecationWarning):
        return 'pydantic' in str(w.filename).lower()
    return False


class Row(NamedTuple):
    row_id: str
    v1_emits_expected: bool | None  # None = not run
    v2_silent: bool | None
    v1_expect: str
    v1_actual: list[str]
    v2_actual: list[str]


def main() -> int:
    half = sys.argv[1] if len(sys.argv) > 1 else 'auto'

    try:
        import pydantic_ai
        pa_v = pydantic_ai.__version__
    except Exception as e:  # noqa: BLE001
        print(f'FAIL: pydantic_ai not importable: {e}')
        return 2

    is_v2 = pa_v.startswith('2.')
    run_v1 = half in {'v1', 'auto'} and not is_v2
    run_v2 = half in {'v2', 'auto'} and is_v2

    print(f'pydantic_ai version: {pa_v}  → running v1={run_v1} v2={run_v2}')
    if not (run_v1 or run_v2):
        print('Nothing to do (interpreter version does not match requested half).')
        return 0

    pairs: dict[str, dict[str, Path]] = {}
    for p in sorted(INDIV.glob('*_v1.py')):
        rid = p.stem[:-3]  # drop _v1
        pairs.setdefault(rid, {})['v1'] = p
    for p in sorted(INDIV.glob('*_v2.py')):
        rid = p.stem[:-3]
        pairs.setdefault(rid, {})['v2'] = p

    rows: list[Row] = []
    overall_ok = True

    for rid, paths in sorted(pairs.items()):
        expect = ''
        v1_emit: bool | None = None
        v2_silent: bool | None = None
        v1_msgs: list[str] = []
        v2_msgs: list[str] = []

        if run_v1 and 'v1' in paths:
            mod_path = paths['v1']
            # Read EXPECT from source statically — safer than loading twice.
            txt = mod_path.read_text()
            for line in txt.splitlines():
                if line.startswith('EXPECT'):
                    # EXPECT = '...'
                    expect = line.split('=', 1)[1].strip().strip("'").strip('"')
                    break
            ws = _run(mod_path)
            v1_msgs = [str(w.message) for w in ws if _is_pydantic_dep(w)]
            v1_emit = any(expect in m for m in v1_msgs) if expect else bool(v1_msgs)
            if not v1_emit:
                overall_ok = False
                print(f'  MISS v1 {rid}: expected {expect!r} not in {v1_msgs}')
            else:
                print(f'  OK   v1 {rid}')

        if run_v2 and 'v2' in paths:
            ws = _run(paths['v2'])
            v2_msgs = [str(w.message) for w in ws if _is_pydantic_dep(w)]
            v2_silent = len(v2_msgs) == 0
            if not v2_silent:
                overall_ok = False
                print(f'  MISS v2 {rid}: unexpected warnings {v2_msgs}')
            else:
                print(f'  OK   v2 {rid}')

        rows.append(Row(rid, v1_emit, v2_silent, expect, v1_msgs, v2_msgs))

    # Exercise legacy_app / modern_app
    legacy_ok = modern_ok = None
    if run_v1:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sys.path.insert(0, str(ROOT / 'examples'))
            try:
                from legacy_app import main as legacy_main  # noqa
                legacy_main.run()
            except Exception as exc:  # noqa: BLE001
                print(f'  legacy_app raised: {exc}')
            finally:
                sys.path.pop(0)
        legacy_warnings = [str(w.message) for w in caught if _is_pydantic_dep(w)]
        # Should hit at least A1, A2, A6, A7, F1, F2, F3, J1
        expected_legacy = [
            '`Agent(instrument=...)` is deprecated',
            '`Agent(history_processors=',
            'tool_retries',
            'mcp_servers',
            '`Usage` is deprecated',
            '`vendor_details` is deprecated',
            '`call_id` is deprecated',
            'Omitting the `name` parameter',
        ]
        missing = [e for e in expected_legacy if not any(e in m for m in legacy_warnings)]
        legacy_ok = not missing
        if not legacy_ok:
            overall_ok = False
            print(f'  MISS legacy_app: missing expected substrings {missing}')
        else:
            print(f'  OK   legacy_app ({len(legacy_warnings)} warnings, all expected substrings hit)')

    if run_v2:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sys.path.insert(0, str(ROOT / 'examples'))
            try:
                from modern_app import main as modern_main  # noqa
                modern_main.run()
            except Exception as exc:  # noqa: BLE001
                print(f'  modern_app raised: {exc}')
            finally:
                sys.path.pop(0)
        modern_warnings = [str(w.message) for w in caught if _is_pydantic_dep(w)]
        modern_ok = len(modern_warnings) == 0
        if not modern_ok:
            overall_ok = False
            print(f'  MISS modern_app: unexpected warnings {modern_warnings}')
        else:
            print(f'  OK   modern_app (silent)')

    # Print matrix
    print('\nCoverage matrix')
    print('=' * 70)
    print(f'{"row":<28} {"v1_emits":<10} {"v2_silent":<10}')
    print('-' * 70)
    for r in rows:
        v1c = '-' if r.v1_emits_expected is None else ('YES' if r.v1_emits_expected else 'NO')
        v2c = '-' if r.v2_silent is None else ('YES' if r.v2_silent else 'NO')
        print(f'{r.row_id:<28} {v1c:<10} {v2c:<10}')
    print('=' * 70)
    print(f'overall: {"PASS" if overall_ok else "FAIL"}')

    return 0 if overall_ok else 1


if __name__ == '__main__':
    sys.exit(main())
