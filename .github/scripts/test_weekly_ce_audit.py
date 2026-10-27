"""Exercise the weekly CE audit's inventory, coverage and publication boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from weekly_ce_audit import (
    Entry,
    classify,
    existing_pr,
    git,
    inventory,
    main,
    reconcile,
    submitted_branches,
    validate_edits,
)


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    """Create a real tracked fixture with canonical, private and aliased paths."""
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    for path in (
        'AGENTS.md',
        '.agents/skills/example/SKILL.md',
        '.github/workflows/audit.md',
        'docs/AGENTS.md',
        'docs/guide.md',
        'local-notes/private.md',
        'src/pydantic_ai_harness/gh-aw/pydantic.md',
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('First line\nSecond line\n')
    (tmp_path / 'CLAUDE.md').symlink_to('AGENTS.md')
    (tmp_path / '.agents/skills/example/alias.md').symlink_to('SKILL.md')
    (tmp_path / '.agents/skills/example/external.md').symlink_to('../../../../outside.md')
    (tmp_path / '.agents/skills/example/cycle.md').symlink_to('cycle.md')
    git(tmp_path, 'add', '.')
    git(
        tmp_path,
        '-c',
        'user.name=Audit Test',
        '-c',
        'user.email=audit@example.com',
        '-c',
        'core.hooksPath=/dev/null',
        'commit',
        '--quiet',
        '-m',
        'Fixture',
    )
    return tmp_path, git(tmp_path, 'rev-parse', 'HEAD').decode().strip()


@pytest.mark.parametrize(
    ('path', 'scope'),
    [
        ('docs/harness/AGENTS.md', 'editable'),
        ('pydantic_ai_slim/pydantic_ai/.agents/skills/example/references/tools.md', 'editable'),
        ('src/pydantic_clai2/FILE_MAP.md', 'editable'),
        ('.macroscope/correctness/review-discipline.md', 'report-only'),
        ('.github/workflows/shared/engine-zai.md', 'report-only'),
        ('src/pydantic_ai_harness/gh-aw/pydantic.md', 'report-only'),
        ('docs/tools.md', 'excluded'),
        ('.claude/skills/branch-context/issue-brief.md', 'excluded'),
        ('.agents/memory/notes.md', 'excluded'),
        ('PAPERCUTS.md', 'excluded'),
        ('CLAUDE.local.md', 'excluded'),
        ('.github/workflows/audit.lock.yml', 'excluded'),
    ],
)
def test_classification(path: str, scope: str):
    """Keep workflow configuration report-only and product/private content excluded."""
    assert classify(path) == scope


def test_inventory_uses_tracked_objects_and_resolves_aliases(repository: tuple[Path, str]):
    """Read committed objects even when the working tree contains an overlay."""
    root, source = repository
    (root / 'AGENTS.md').write_text('Uncommitted overlay\n')
    entries = {entry.path: entry for entry in inventory(root, source)}
    canonical = entries['AGENTS.md']
    assert canonical.lines == 2
    assert entries['CLAUDE.md'].scope == 'alias'
    assert entries['CLAUDE.md'].canonical == 'AGENTS.md'
    assert entries['CLAUDE.md'].sha256 == canonical.sha256
    assert entries['.agents/skills/example/alias.md'].canonical == '.agents/skills/example/SKILL.md'
    assert entries['.agents/skills/example/external.md'].scope == 'excluded'
    assert entries['.agents/skills/example/cycle.md'].scope == 'excluded'
    assert entries['docs/guide.md'].scope == 'excluded'
    assert entries['local-notes/private.md'].sha256 is None


def test_receipt_reconciles_canonical_report_only_and_paged_reads(repository: tuple[Path, str]):
    """Require complete paged canonical reads including report-only paths."""
    root, source = repository
    entries = inventory(root, source)
    rows = [
        {'path': entry.path, 'sha256': entry.sha256, 'ranges': [[1, 1], [2, 2]]}
        for entry in entries
        if entry.scope in {'editable', 'report-only'}
    ]
    receipt = {'source': source, 'reviewed': rows}
    assert reconcile(entries, receipt, source) == []
    removed = rows.pop()
    assert reconcile(entries, receipt, source) == [removed['path']]
    rows.append({**removed, 'ranges': [[1, 1]]})
    assert reconcile(entries, receipt, source) == [removed['path']]


@pytest.mark.parametrize(
    'row',
    [
        {'path': 'AGENTS.md', 'sha256': 'hash', 'ranges': [[2, 2]]},
        {'path': 'AGENTS.md', 'sha256': 'hash', 'ranges': [[1, 2], [2, 2]]},
        {'path': 'AGENTS.md', 'sha256': 'hash', 'ranges': [[1, 3]]},
        {'path': 'AGENTS.md', 'sha256': 'wrong', 'ranges': [[1, 2]]},
        {'path': 'CLAUDE.md', 'sha256': 'hash', 'ranges': [[1, 2]]},
        {'path': 'docs/guide.md', 'sha256': 'hash', 'ranges': [[1, 2]]},
    ],
)
def test_receipt_rejects_invalid_coverage(row: dict[str, object]):
    """Reject stale hashes, gaps, overlaps and noncanonical receipt entries."""
    entries = [Entry('AGENTS.md', 'editable', 'Tracked CE', 'AGENTS.md', 'hash', 2)]
    with pytest.raises(ValueError):
        reconcile(entries, {'source': 'source', 'reviewed': [row]}, 'source')
    with pytest.raises(ValueError, match='source'):
        reconcile(entries, {'source': 'wrong', 'reviewed': []}, 'source')


def test_duplicate_guard_recognizes_only_this_workflows_open_same_repo_pr():
    """Avoid taking ownership of another workflow or a contributor fork."""
    pr = {
        'number': 123,
        'state': 'open',
        'title': '[weekly-ce-audit] Trim duplicate instructions',
        'head': {'ref': 'gh-aw/weekly-ce-audit-123', 'repo': {'full_name': 'pydantic/pydantic-ai'}},
    }
    assert existing_pr([pr], 'pydantic/pydantic-ai') == 123
    assert existing_pr([{**pr, 'title': 'A maintainer renamed this PR'}], 'pydantic/pydantic-ai') == 123
    for override in (
        {'state': 'closed'},
        {'head': {'ref': 'another-owner', 'repo': {'full_name': 'pydantic/pydantic-ai'}}},
        {'head': {'ref': 'gh-aw/weekly-ce-audit-123', 'repo': {'full_name': 'other/fork'}}},
        {'head': {'ref': 'deleted-fork', 'repo': None}},
    ):
        assert existing_pr([{**pr, **override}], 'pydantic/pydantic-ai') is None


@pytest.mark.parametrize(
    'path',
    ['.github/workflows/audit.md', 'docs/guide.md', 'new.md', 'src/pydantic_ai_harness/gh-aw/pydantic.md'],
)
def test_write_boundary_rejects_report_only_product_and_new_files(repository: tuple[Path, str], path: str):
    """Confine publication to existing canonical instruction Markdown."""
    root, source = repository
    (root / path).write_text('Unauthorized change\n')
    with pytest.raises(ValueError, match='allowlist'):
        validate_edits(root, source, inventory(root, source))


def test_write_boundary_rejects_symlink_replacement_and_deletion(repository: tuple[Path, str]):
    """Preserve tracked alias and regular-file boundaries."""
    root, source = repository
    entries = inventory(root, source)
    target = root / 'AGENTS.md'
    target.unlink()
    with pytest.raises(ValueError, match='allowlist'):
        validate_edits(root, source, entries)
    target.symlink_to('docs/guide.md')
    with pytest.raises(ValueError, match='allowlist'):
        validate_edits(root, source, entries)


def test_incomplete_audit_cannot_publish_edits(repository: tuple[Path, str], monkeypatch: pytest.MonkeyPatch):
    """Allow an incomplete noop while refusing a PR with incomplete coverage."""
    root, source = repository
    directory = root / '.ce-audit'
    monkeypatch.setattr(
        sys, 'argv', ['audit', 'prepare', '--root', str(root), '--source', source, '--directory', str(directory)]
    )
    main()
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['source'] == source
    monkeypatch.setattr(
        sys, 'argv', ['audit', 'validate', '--root', str(root), '--source', source, '--directory', str(directory)]
    )
    main()
    assert 'incomplete' in (directory / 'summary.md').read_text()
    (root / 'AGENTS.md').write_text('A focused trim\n')
    with pytest.raises(ValueError, match='Incomplete audit'):
        main()
    receipt = {
        'source': source,
        'reviewed': [
            {'path': entry['path'], 'sha256': entry['sha256'], 'ranges': [[1, entry['lines']]]}
            for entry in manifest['entries']
            if entry['scope'] in {'editable', 'report-only'}
        ],
    }
    (directory / 'receipt.json').write_text(json.dumps(receipt))
    main()
    assert 'Reported canonical coverage: complete.' in (directory / 'summary.md').read_text()


def test_workflow_retains_separate_pr_sink_and_protected_review():
    """Retain least-privilege agent access and mandatory protected-file review."""
    workflow = Path(__file__).parents[1] / 'workflows/pydantic-ai-weekly-ce-audit.md'
    config = yaml.safe_load(workflow.read_text().split('---', 2)[1])
    outputs = config['safe-outputs']
    pr = outputs['create-pull-request']
    assert config['permissions'] == {'contents': 'read', 'issues': 'read', 'pull-requests': 'read'}
    assert config['concurrency']['cancel-in-progress'] is False
    assert pr['max'] == 1 and pr['draft'] is False and pr['fallback-as-issue'] is False
    assert pr['protected-files'] == 'request_review'
    assert '.github/**' in pr['excluded-files']
    assert 'create-issue' not in outputs
    assert outputs['steps'][0]['env']['AGENT_RESULT'] == '${{ needs.agent.result }}'
    assert outputs['steps'][0]['run'] == 'test "$AGENT_RESULT" = success'
    assert outputs['threat-detection']['engine']['env']['CLAUDE_CODE_ENABLE_TELEMETRY'] == '1'
    preflight = config['pre-agent-steps'][0]['run']
    assert '--slurp' in preflight and '--jq' not in preflight
    assert "jq 'add'" in preflight


def test_preflight_writes_noop_before_inference(repository: tuple[Path, str], monkeypatch: pytest.MonkeyPatch):
    """An existing workflow PR produces a terminal safe output without invoking a model."""
    root, source = repository
    prs = root / '.ce-audit/prs.json'
    prs.parent.mkdir()
    prs.write_text(
        json.dumps(
            [
                {
                    'number': 123,
                    'state': 'open',
                    'title': '[weekly-ce-audit] Trim duplicate instructions',
                    'head': {'ref': 'gh-aw/weekly-ce-audit-123', 'repo': {'full_name': 'pydantic/pydantic-ai'}},
                }
            ]
        )
    )
    sink = root / '.ce-audit/safeoutputs.jsonl'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'audit',
            'prepare',
            '--root',
            str(root),
            '--source',
            source,
            '--directory',
            str(prs.parent),
            '--pull-requests',
            str(prs),
            '--repository',
            'pydantic/pydantic-ai',
            '--safe-outputs',
            str(sink),
        ],
    )
    main()
    assert json.loads(sink.read_text()) == {'type': 'noop', 'message': 'Existing weekly CE audit PR #123'}


def test_selected_branch_cannot_bypass_boundary_by_switching_checkout(repository: tuple[Path, str]):
    """Validate the published branch even when the agent restores a clean checkout."""
    root, source = repository
    branch = 'gh-aw/weekly-ce-audit-test'
    git(root, 'checkout', '-b', branch)
    (root / 'docs/guide.md').write_text('Unauthorized product edit\n')
    git(root, 'add', '.')
    git(
        root,
        '-c',
        'user.name=Audit Test',
        '-c',
        'user.email=audit@example.com',
        '-c',
        'core.hooksPath=/dev/null',
        'commit',
        '--quiet',
        '-m',
        'Candidate',
    )
    git(root, 'checkout', '--detach', source)
    assert validate_edits(root, source, inventory(root, source)) == []
    branches = submitted_branches({'items': [{'type': 'create_pull_request', 'branch': branch}]})
    with pytest.raises(ValueError, match='allowlist'):
        validate_edits(root, source, inventory(root, source), branches[0])


def test_compiled_publication_gate_and_health_dependencies():
    """Require validated agent success before the compiled privileged handler runs."""
    path = Path(__file__).parents[1] / 'workflows/pydantic-ai-weekly-ce-audit.lock.yml'
    jobs = yaml.safe_load(path.read_text())['jobs']
    assert jobs['activation']['needs'] == 'provider_health'
    assert 'needs' not in jobs['provider_health']
    steps = jobs['safe_outputs']['steps']
    gate = next(
        i
        for i, step in enumerate(steps)
        if step.get('name') == 'Require successful coverage and edit validation before publication'
    )
    handler = next(i for i, step in enumerate(steps) if step.get('name') == 'Process Safe Outputs')
    assert gate < handler
    detection = next(step for step in jobs['detection']['steps'] if step.get('id') == 'detection_agentic_execution')
    assert str(detection['env']['CLAUDE_CODE_ENABLE_TELEMETRY']) == '1'
    assert detection['env']['OTEL_SERVICE_NAME'] == 'gh-aw.threat-detection'
    assert str(detection['env']['GH_AW_HARNESS_MAX_RETRIES']) == '3'
