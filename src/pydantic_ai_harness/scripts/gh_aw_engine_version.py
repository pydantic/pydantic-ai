# /// script
# requires-python = ">=3.10"
# dependencies = ["packaging>=24", "pydantic>=2", "pyyaml>=6.0.2"]
# ///
"""Validate `gh-aw/pydantic.md` and print the `pydantic-ai-harness` version its engine pins.

GitHub Agentic Workflows resolves the imported definition at compile time and vendors its
frontmatter into the workflow it generates, so those fields are a published contract rather
than repo-local config: `engine.id` keys the entry in gh-aw's engine catalog, and
`engine.version` is the release the generated workflow installs from PyPI at run time. A
renamed field breaks the catalog entry, and a version that was never published breaks every
run of it at install time.

Consumers import the file from `main`, so a merge reaches them on their next compile and
nothing downstream re-checks it. `--published` is therefore part of the pull request gate
rather than a release step: it asks PyPI the one question the file cannot answer about
itself.

The lint job and the release reminder job both call this, which is why the checks live here
rather than inlined as shell twice.

Inline dependency metadata, so `uv run --script scripts/gh_aw_engine_version.py` works
without a project sync: neither caller needs any other part of the harness.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

import yaml
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFINITION = Path(__file__).resolve().parent.parent / 'gh-aw' / 'pydantic.md'
# Bounds each socket operation rather than the whole request, which is enough to keep a
# connection that is accepted but never answered from holding a runner until the job
# default.
PYPI_TIMEOUT_SECONDS = 30


class _Engine(BaseModel):
    model_config = ConfigDict(extra='ignore')

    engine_id: Literal['pydantic-ai'] = Field(alias='id')
    # Not `str | float`: an unquoted `0.21.0` is a YAML float and `0.21` loses a
    # component on the way back to text, so the quoting is part of the contract.
    version: str

    @field_validator('version')
    @classmethod
    def _pep_440(cls, value: str) -> str:
        # gh-aw interpolates this into `pydantic-ai-harness[cli]==<version>`, and the
        # publication check below interpolates it into a PyPI URL. Anything that is not a
        # version is a broken install for consumers, and a value carrying `/` or `?`
        # reaches a different PyPI endpoint than the one the check means to ask about.
        #
        # Rejected rather than normalized, both here and for the surrounding
        # whitespace `Version` would otherwise accept: gh-aw reads the same bytes this
        # file holds, so a guard that checks a cleaned-up copy is checking a string
        # that never ships.
        if value != value.strip():
            raise ValueError('must not be padded with whitespace, which gh-aw would install verbatim')
        try:
            Version(value)
        except InvalidVersion as exc:
            raise ValueError(f'is not a PEP 440 version: {exc}') from exc
        return value


class _Frontmatter(BaseModel):
    model_config = ConfigDict(extra='ignore')

    engine: _Engine


def _frontmatter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0] != '---':
        raise ValueError(f'{DEFINITION} must open with YAML frontmatter delimited by `---`.')
    closing = next((index for index, line in enumerate(lines[1:], start=1) if line == '---'), None)
    if closing is None:
        raise ValueError(f'{DEFINITION} has unclosed YAML frontmatter.')
    return '\n'.join(lines[1:closing])


def engine_version() -> str:
    """Return `engine.version` once the fields gh-aw and its consumers read are known good."""
    parsed: object = yaml.safe_load(_frontmatter(DEFINITION.read_text(encoding='utf-8')))
    return _Frontmatter.model_validate(parsed).engine.version


def unpublished_reason(version: str) -> str | None:
    """Return why PyPI does not serve `version` as a release, or `None` when it does."""
    url = f'https://pypi.org/pypi/pydantic-ai-harness/{version}/json'
    try:
        with urllib.request.urlopen(url, timeout=PYPI_TIMEOUT_SECONDS) as response:
            status: int = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except OSError as exc:
        # A failed request is not evidence that the version is missing, so it is reported
        # as the network error it is rather than as an unpublished pin.
        return f'Could not ask PyPI whether pydantic-ai-harness {version} is published: {exc}'

    if status != 200:
        return (
            f'{DEFINITION} pins `engine.version: {version}`, which PyPI does not serve '
            f'(HTTP {status}). gh-aw installs that version at run time, so a workflow '
            f'compiled against this definition would fail before the agent starts.'
        )
    return None


def main() -> int:
    """Print the pinned version, or explain on stderr why the definition cannot be trusted."""
    parser = argparse.ArgumentParser(description='Validate the gh-aw engine definition.')
    parser.add_argument(
        '--published',
        action='store_true',
        help='also require the pinned version to be a release PyPI serves',
    )
    published: bool = parser.parse_args().published

    try:
        version = engine_version()
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
        print(f'{DEFINITION} is not a usable gh-aw engine definition: {exc}', file=sys.stderr)
        return 1

    if published and (reason := unpublished_reason(version)) is not None:
        print(reason, file=sys.stderr)
        return 1

    print(version)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
