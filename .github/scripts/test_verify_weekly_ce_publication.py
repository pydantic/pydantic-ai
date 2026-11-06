"""Tests for the weekly CE audit publication verifier."""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest
from verify_weekly_ce_publication import verify

PROPOSAL = {'title': 'Trim duplicate wording', 'body': 'A small docs change.', 'branch': 'gh-aw/weekly-ce-audit-x'}


def _write(path: pathlib.Path, items: list[dict[str, object]]) -> pathlib.Path:
    path.write_text(json.dumps({'items': items}), encoding='utf-8')
    return path


def _reviewed(path: pathlib.Path) -> pathlib.Path:
    path.write_text(json.dumps({**PROPOSAL, 'issue_exemption': 'small-doc-improvement'}), encoding='utf-8')
    return path


def test_matching_publication_is_accepted(tmp_path: pathlib.Path) -> None:
    verify(
        _write(tmp_path / 'agent_output.json', [{'type': 'create_pull_request', **PROPOSAL}]),
        _reviewed(tmp_path / 'proposal.json'),
    )


def test_noop_only_is_accepted(tmp_path: pathlib.Path) -> None:
    verify(_write(tmp_path / 'agent_output.json', [{'type': 'noop'}]), tmp_path / 'missing.json')


def test_missing_ingested_output_fails_closed(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match='missing'):
        verify(tmp_path / 'missing.json', tmp_path / 'missing-proposal.json')


def test_output_without_an_items_list_fails_closed(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'agent_output.json'
    path.write_text(json.dumps({'unexpected': []}), encoding='utf-8')
    with pytest.raises(ValueError, match='items'):
        verify(path, tmp_path / 'missing-proposal.json')


def test_noop_with_a_pull_request_is_rejected(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match='noop'):
        verify(
            _write(
                tmp_path / 'agent_output.json',
                [{'type': 'create_pull_request', **PROPOSAL}, {'type': 'noop'}],
            ),
            _reviewed(tmp_path / 'proposal.json'),
        )


def test_publication_without_a_reviewed_proposal_is_rejected(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match='reviewed proposal'):
        verify(
            _write(tmp_path / 'agent_output.json', [{'type': 'create_pull_request', **PROPOSAL}]),
            tmp_path / 'missing.json',
        )


@pytest.mark.parametrize('field', ['title', 'body', 'branch'])
def test_publication_must_match_the_reviewed_proposal(tmp_path: pathlib.Path, field: str) -> None:
    changed = {**PROPOSAL, field: 'different'}
    with pytest.raises(ValueError, match='differs'):
        verify(
            _write(tmp_path / 'agent_output.json', [{'type': 'create_pull_request', **changed}]),
            _reviewed(tmp_path / 'proposal.json'),
        )


def test_only_one_pull_request_is_permitted(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match='Only one'):
        verify(
            _write(
                tmp_path / 'agent_output.json',
                [{'type': 'create_pull_request', **PROPOSAL}, {'type': 'create_pull_request', **PROPOSAL}],
            ),
            _reviewed(tmp_path / 'proposal.json'),
        )


def test_proposal_survives_the_artifact_upload_and_download(tmp_path: pathlib.Path) -> None:
    """The agent job uploads `/tmp/gh-aw/agent/` and the safe-outputs job downloads it to its root."""
    agent_root = tmp_path / 'gh-aw' / 'agent'
    agent_root.mkdir(parents=True)
    (agent_root / 'reviewed-proposal.json').write_text(
        json.dumps({**PROPOSAL, 'issue_exemption': 'small-doc-improvement'}), encoding='utf-8'
    )

    artifact = tmp_path / 'artifact'
    shutil.copytree(agent_root, artifact)  # upload-artifact stores a directory's contents at the root

    download_root = tmp_path / 'downloaded'
    shutil.copytree(artifact, download_root)
    _write(download_root / 'agent_output.json', [{'type': 'create_pull_request', **PROPOSAL}])

    verify(download_root / 'agent_output.json', download_root / 'reviewed-proposal.json')

    _write(
        download_root / 'agent_output.json',
        [{'type': 'create_pull_request', **{**PROPOSAL, 'title': 'unreviewed'}}],
    )
    with pytest.raises(ValueError, match='differs'):
        verify(download_root / 'agent_output.json', download_root / 'reviewed-proposal.json')


def test_unexpected_published_fields_are_rejected(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match='unexpected fields'):
        verify(
            _write(
                tmp_path / 'agent_output.json',
                [{'type': 'create_pull_request', **PROPOSAL, 'base': 'other-branch'}],
            ),
            _reviewed(tmp_path / 'proposal.json'),
        )
