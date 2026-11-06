"""Tests for the weekly CE audit publication verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from verify_weekly_ce_publication import verify

PROPOSAL = {'title': 'Trim duplicate wording', 'body': 'A small docs change.', 'branch': 'gh-aw/weekly-ce-audit-x'}


def _write(path: Path, items: list[dict[str, object]]) -> Path:
    path.write_text('\n'.join(json.dumps(item) for item in items) + '\n', encoding='utf-8')
    return path


def _reviewed(path: Path) -> Path:
    path.write_text(json.dumps({**PROPOSAL, 'issue_exemption': 'small-doc-improvement'}), encoding='utf-8')
    return path


def test_matching_publication_is_accepted(tmp_path: Path) -> None:
    verify(
        _write(tmp_path / 'safe.jsonl', [{'type': 'create_pull_request', **PROPOSAL}]),
        _reviewed(tmp_path / 'proposal.json'),
    )


def test_noop_only_is_accepted(tmp_path: Path) -> None:
    verify(_write(tmp_path / 'safe.jsonl', [{'type': 'noop'}]), tmp_path / 'missing.json')


def test_missing_safe_outputs_file_means_no_publication(tmp_path: Path) -> None:
    verify(tmp_path / 'missing.jsonl', tmp_path / 'missing.json')


def test_noop_with_a_pull_request_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='noop'):
        verify(
            _write(tmp_path / 'safe.jsonl', [{'type': 'create_pull_request', **PROPOSAL}, {'type': 'noop'}]),
            _reviewed(tmp_path / 'proposal.json'),
        )


def test_publication_without_a_reviewed_proposal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='reviewed proposal'):
        verify(
            _write(tmp_path / 'safe.jsonl', [{'type': 'create_pull_request', **PROPOSAL}]), tmp_path / 'missing.json'
        )


@pytest.mark.parametrize('field', ['title', 'body', 'branch'])
def test_publication_must_match_the_reviewed_proposal(tmp_path: Path, field: str) -> None:
    changed = {**PROPOSAL, field: 'different'}
    with pytest.raises(ValueError, match='differs'):
        verify(
            _write(tmp_path / 'safe.jsonl', [{'type': 'create_pull_request', **changed}]),
            _reviewed(tmp_path / 'proposal.json'),
        )


def test_only_one_pull_request_is_permitted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Only one'):
        verify(
            _write(
                tmp_path / 'safe.jsonl',
                [{'type': 'create_pull_request', **PROPOSAL}, {'type': 'create_pull_request', **PROPOSAL}],
            ),
            _reviewed(tmp_path / 'proposal.json'),
        )
