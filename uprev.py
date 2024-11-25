"""Update the pydantic-ai version number everywhere.

Because we have multiple functions which depend on one-another,
we have to update the version number in:
* 3 places in pyproject.toml
* 1 place in pydantic_ai_examples/pyproject.toml

Usage:

    uv run uprev.py <new_version_number>
"""
import re
import sys

from pathlib import Path

ROOT_DIR = Path(__file__).parent

try:
    version = sys.argv[1]
except IndexError:
    print('Usage: uv run uprev.py <new_version_number>', file=sys.stderr)
    sys.exit(1)


old_version: str | None = None


def sub_version(m: re.Match[str]) -> str:
    global old_version
    prefix = m.group(1)
    quote = m.group(2)
    old_version = m.group(3)
    return f'{prefix}{quote}{version}{quote}'


root_pp = ROOT_DIR / 'pyproject.toml'
root_pp_text = root_pp.read_text()
root_pp_text, _ = re.subn(r'^(version ?= ?)(["\'])(.+)\2$', sub_version, root_pp_text, 1, flags=re.M)

if old_version is None:
    print('ERROR: Could not find version in root pyproject.toml', file=sys.stderr)
    sys.exit(1)

print(f'Updating version from {old_version!r} to {version!r}')


def replace_deps_version(text: str) -> tuple[str, int]:
    return re.subn(
        '(pydantic-ai-.+?==)' + re.escape(old_version),
        r'\g<1>' + version,
        text,
        count=5,
    )


root_pp_text, count_root = replace_deps_version(root_pp_text)

examples_pp = ROOT_DIR / 'pydantic_ai_examples' / 'pyproject.toml'
examples_pp_text = examples_pp.read_text()
examples_pp_text, count_ex = replace_deps_version(examples_pp_text)

if count_root == 2 and count_ex == 1:
    root_pp.write_text(root_pp_text)
    examples_pp.write_text(examples_pp_text)
    print(f'SUCCESS: replaced the following version occurrences\n  3 in root {root_pp}\n  1 in {examples_pp}')
else:
    print(
        f'ERROR: found {count_root} version references in {root_pp_text} (expected 3) '
        f'and {count_ex} in {examples_pp} (expected 1)',
        file=sys.stderr
    )
    sys.exit(1)
