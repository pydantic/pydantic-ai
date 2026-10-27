"""Tests for the CI provenance and case content hash attributes that `Dataset.evaluate` puts on its spans.

These attributes exist for consumers that compare experiment runs (such as a "this run vs. the base branch"
pull request comment), so the tests pin the exact span attributes a consumer would read.
"""

from __future__ import annotations as _annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals import Case, Dataset

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
    pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed'),
]

_PROVENANCE_PREFIXES = ('logfire.experiment.git.', 'logfire.experiment.ci.')
_HASH_PREFIX = 'logfire.experiment.case.'

_GITHUB_ENV = {
    'GITHUB_ACTIONS': 'true',
    'GITHUB_SERVER_URL': 'https://github.com',
    'GITHUB_REPOSITORY': 'acme/widgets',
    'GITHUB_RUN_ID': '1234567890',
}


@pytest.fixture(autouse=True)
def _clear_github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test from a clean slate, since the suite itself runs in GitHub Actions."""
    for name in list(os.environ):
        if name.startswith('GITHUB_'):
            monkeypatch.delenv(name)


async def _task(inputs: Any) -> str:
    return 'ok'


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)


async def _provenance(capfire: CaptureLogfire) -> dict[str, Any]:
    await Dataset[Any, Any, Any](name='provenance', cases=[Case(inputs='x')]).evaluate(_task, progress=False)
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    [experiment_span] = [s for s in spans if s['name'] == 'evaluate {name}']
    return {k: v for k, v in experiment_span['attributes'].items() if k.startswith(_PROVENANCE_PREFIXES)}


async def _case_hashes(capfire: CaptureLogfire, cases: list[Case[Any, Any, Any]]) -> dict[str, dict[str, Any]]:
    await Dataset[Any, Any, Any](name='provenance', cases=cases).evaluate(_task, progress=False)
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    return {
        s['attributes']['case_name']: {
            k.removeprefix(_HASH_PREFIX): v for k, v in s['attributes'].items() if k.startswith(_HASH_PREFIX)
        }
        for s in spans
        if s['name'] == 'case: {case_name}'
    }


async def test_not_in_ci(capfire: CaptureLogfire):
    assert await _provenance(capfire) == {}


async def test_github_push(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_EVENT_NAME': 'push',
            'GITHUB_SHA': 'abc123',
            'GITHUB_REF': 'refs/heads/main',
            'GITHUB_REF_NAME': 'main',
            'GITHUB_REF_TYPE': 'branch',
        },
    )
    assert await _provenance(capfire) == snapshot(
        {
            'logfire.experiment.ci.run_url': 'https://github.com/acme/widgets/actions/runs/1234567890',
            'logfire.experiment.git.commit_sha': 'abc123',
            'logfire.experiment.git.branch': 'main',
        }
    )


async def test_github_tag_push_has_no_branch(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_EVENT_NAME': 'push',
            'GITHUB_SHA': 'abc123',
            'GITHUB_REF': 'refs/tags/v1.0.0',
            'GITHUB_REF_NAME': 'v1.0.0',
            'GITHUB_REF_TYPE': 'tag',
        },
    )
    assert await _provenance(capfire) == snapshot(
        {
            'logfire.experiment.ci.run_url': 'https://github.com/acme/widgets/actions/runs/1234567890',
            'logfire.experiment.git.commit_sha': 'abc123',
        }
    )


@pytest.fixture
def pull_request_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `pull_request` event, where `GITHUB_SHA`/`GITHUB_REF_NAME` point at the synthetic merge commit."""
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_EVENT_NAME': 'pull_request',
            'GITHUB_SHA': 'merge-commit-sha',
            'GITHUB_REF': 'refs/pull/42/merge',
            'GITHUB_REF_NAME': '42/merge',
            'GITHUB_REF_TYPE': 'branch',
            'GITHUB_HEAD_REF': 'feature-x',
            'GITHUB_BASE_REF': 'main',
        },
    )


@pytest.mark.usefixtures('pull_request_env')
async def test_github_pull_request(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    event_path = tmp_path / 'event.json'
    event_path.write_text(
        json.dumps(
            {
                'action': 'synchronize',
                'number': 42,
                'pull_request': {
                    'number': 42,
                    'head': {'ref': 'feature-x', 'sha': 'head-sha'},
                    'base': {'ref': 'main', 'sha': 'base-sha'},
                },
            }
        )
    )
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(event_path))

    assert await _provenance(capfire) == snapshot(
        {
            'logfire.experiment.ci.run_url': 'https://github.com/acme/widgets/actions/runs/1234567890',
            'logfire.experiment.git.commit_sha': 'head-sha',
            'logfire.experiment.git.branch': 'feature-x',
            'logfire.experiment.git.base_branch': 'main',
            'logfire.experiment.ci.pull_request': 42,
        }
    )


@pytest.mark.usefixtures('pull_request_env')
@pytest.mark.parametrize('readable_payload', [True, False])
async def test_github_pull_request_target(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, readable_payload: bool
):
    """`pull_request_target` checks out the base branch by default, so the PR head isn't reported as the code under test."""
    if readable_payload:
        event_path = tmp_path / 'event.json'
        event_path.write_text(
            json.dumps(
                {
                    'pull_request': {
                        'number': 42,
                        'head': {'ref': 'feature-x', 'sha': 'head-sha'},
                        'base': {'ref': 'main'},
                    }
                }
            )
        )
        monkeypatch.setenv('GITHUB_EVENT_PATH', str(event_path))
    # Unlike `pull_request`, these point at the base branch rather than a merge commit.
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'pull_request_target')
    monkeypatch.setenv('GITHUB_SHA', 'base-tip-sha')
    monkeypatch.setenv('GITHUB_REF', 'refs/heads/main')
    monkeypatch.setenv('GITHUB_REF_NAME', 'main')

    expected: dict[str, Any] = {
        'logfire.experiment.ci.run_url': 'https://github.com/acme/widgets/actions/runs/1234567890',
        'logfire.experiment.git.base_branch': 'main',
    }
    if readable_payload:
        expected['logfire.experiment.ci.pull_request'] = 42
    assert await _provenance(capfire) == expected


@pytest.mark.usefixtures('pull_request_env')
@pytest.mark.parametrize('event_file', ['missing', 'not json', '{"pull_request": {"number": "not a number"}}'])
async def test_github_pull_request_unreadable_event(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, event_file: str
):
    """Without a readable payload the head commit is unknown, so no SHA is reported rather than the merge commit's."""
    event_path = tmp_path / 'event.json'
    if event_file != 'missing':
        event_path.write_text(event_file)
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(event_path))

    assert await _provenance(capfire) == snapshot(
        {
            'logfire.experiment.ci.run_url': 'https://github.com/acme/widgets/actions/runs/1234567890',
            'logfire.experiment.git.branch': 'feature-x',
            'logfire.experiment.git.base_branch': 'main',
            'logfire.experiment.ci.pull_request': 42,
        }
    )


async def test_github_minimal_env(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    """Missing variables omit their attributes instead of producing partial values."""
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('GITHUB_HEAD_REF', 'feature-x')
    monkeypatch.setenv('GITHUB_REF', 'refs/heads/feature-x')
    assert await _provenance(capfire) == snapshot({'logfire.experiment.git.branch': 'feature-x'})


@pytest.mark.parametrize('value', ['false', 'FALSE', '0'])
async def test_opt_out(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, value: str):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_SHA': 'abc123',
            'PYDANTIC_EVALS_GIT_BRANCH': 'main',
            'PYDANTIC_EVALS_CI_METADATA': value,
        },
    )
    assert await _provenance(capfire) == {}


async def test_overrides_outside_github(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYDANTIC_EVALS_GIT_COMMIT_SHA', 'def456')
    monkeypatch.setenv('PYDANTIC_EVALS_GIT_BRANCH', 'release')
    assert await _provenance(capfire) == snapshot(
        {'logfire.experiment.git.commit_sha': 'def456', 'logfire.experiment.git.branch': 'release'}
    )


async def test_overrides_win_over_github(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_SHA': 'abc123',
            'GITHUB_REF_NAME': 'main',
            'GITHUB_REF_TYPE': 'branch',
            'PYDANTIC_EVALS_GIT_COMMIT_SHA': 'def456',
            'PYDANTIC_EVALS_GIT_BRANCH': 'release',
        },
    )
    assert await _provenance(capfire) == snapshot(
        {
            'logfire.experiment.ci.run_url': 'https://github.com/acme/widgets/actions/runs/1234567890',
            'logfire.experiment.git.commit_sha': 'def456',
            'logfire.experiment.git.branch': 'release',
        }
    )


@dataclass
class _Inputs:
    query: str


async def test_case_hashes_are_stable(capfire: CaptureLogfire):
    """Hashes ignore dict key order and set ordering, and are pinned so they stay stable across processes and versions.

    Sets of strings iterate in a per-process order, so the pinned values would drift if the hash depended on it.
    """
    hashes = await _case_hashes(
        capfire,
        [
            Case(name='a', inputs={'x': 1, 'y': [1, 2]}, metadata={'tags': {'b', 'a', 'c'}}),
            Case(name='b', inputs={'y': [1, 2], 'x': 1}, metadata={'tags': {'c', 'a', 'b'}}),
            Case(name='dataclass', inputs=_Inputs(query='q'), expected_output=b'\xff'),
        ],
    )
    assert hashes['a'] == hashes['b']
    # A case without an expected output or metadata still gets a hash (of JSON `null`), so adding one shows up as a change.
    assert hashes['a']['expected_output_hash'] == hashlib.sha256(b'null').hexdigest()
    assert hashes == snapshot(
        {
            'a': {
                'inputs_hash': 'aa996729eb221630bcaf0d778199df99b2297bdaecbb2de0ca354caaec8d36f5',
                'expected_output_hash': '74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b',
                'metadata_hash': '66d4b0b771cafa995b17e34abd5e9e19d84189afc87e3f1f3554bcccb84fc091',
            },
            'b': {
                'inputs_hash': 'aa996729eb221630bcaf0d778199df99b2297bdaecbb2de0ca354caaec8d36f5',
                'expected_output_hash': '74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b',
                'metadata_hash': '66d4b0b771cafa995b17e34abd5e9e19d84189afc87e3f1f3554bcccb84fc091',
            },
            'dataclass': {
                'inputs_hash': 'cf11c31927352aeaa5b9ef86ce8f95a954bcd8e6a415f8cd3922101d92dba903',
                'expected_output_hash': '38347093a61793b352fc1d9ac0409035dd459d59e6e9c6b18818400bb40d9d21',
                'metadata_hash': '74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b',
            },
        }
    )


async def test_case_hashes_distinguish_changes(capfire: CaptureLogfire):
    hashes = await _case_hashes(
        capfire,
        [
            Case(name='a', inputs={'x': 1}, expected_output='yes'),
            Case(name='b', inputs={'x': 2}, expected_output='no'),
            Case(name='c', inputs={'x': 1}),
        ],
    )
    assert hashes['a']['inputs_hash'] != hashes['b']['inputs_hash']
    assert hashes['a']['inputs_hash'] == hashes['c']['inputs_hash']
    assert len({h['expected_output_hash'] for h in hashes.values()}) == 3


async def test_case_hashes_omitted_when_unserializable(capfire: CaptureLogfire):
    """A value with no JSON serialization gets no hash, which consumers treat as possibly changed."""
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    hashes = await _case_hashes(
        capfire,
        [
            Case(name='object', inputs=object(), expected_output='yes'),
            Case(name='cyclic', inputs='x', metadata={'self': cyclic}),
            Case(name='unsupported_key', inputs={frozenset({1}): 'x'}, expected_output='yes'),
            Case(name='surrogate', inputs={'t': 'bad \ud800'}, expected_output='yes'),
        ],
    )
    assert {name: sorted(h) for name, h in hashes.items()} == snapshot(
        {
            'object': ['expected_output_hash', 'metadata_hash'],
            'cyclic': ['expected_output_hash', 'inputs_hash'],
            'unsupported_key': ['expected_output_hash', 'metadata_hash'],
            'surrogate': ['expected_output_hash', 'metadata_hash'],
        }
    )
