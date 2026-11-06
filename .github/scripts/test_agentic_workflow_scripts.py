"""Offline tests for the repository's agentic-workflow host scripts.

The composition and its policies live in `test_gh_aw_agent.py`; this module
covers the context prefetches and the shared workflow fragments they feed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


def _run_context_prefetch(
    tmp_path: Path,
    *,
    script: str = 'prefetch-github-context.sh',
    gh_script: str | None = None,
    env_vars: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake_timeout = bin_dir / 'timeout'
    fake_timeout.write_text('#!/bin/sh\nshift\nexec "$@"\n', encoding='utf-8')
    fake_timeout.chmod(0o755)
    if gh_script is not None:
        fake_gh = bin_dir / 'gh'
        fake_gh.write_text(f'#!/bin/sh\n{gh_script}', encoding='utf-8')
        fake_gh.chmod(0o755)

    agent_dir = tmp_path / 'agent'
    env = os.environ.copy()
    env.pop('GH_TOKEN', None)
    env.pop('GITHUB_TOKEN', None)
    env.pop('GITHUB_REPOSITORY', None)
    env.update(
        PATH=f'{bin_dir}:{env["PATH"]}',
        GH_AW_AGENT_DIR=str(agent_dir),
    )
    env.update(env_vars or {})
    result = subprocess.run(
        ['bash', Path(__file__).with_name(script)], text=True, capture_output=True, check=False, env=env
    )
    return result, agent_dir


_SUCCESSFUL_GH_PREFETCH = """\
case "$1:$2" in
  issue:list) printf '%s\\n' '[{"number":1,"title":"Issue"}]' ;;
  pr:list) printf '%s\\n' '[{"number":2,"title":"PR"}]' ;;
  *) exit 8 ;;
esac
"""


@pytest.mark.parametrize(
    ('env_vars', 'expected_token'),
    [
        ({'GH_TOKEN': 'preferred-token', 'GITHUB_TOKEN': 'fallback-token'}, 'preferred-token'),
        ({'GITHUB_TOKEN': 'fallback-token'}, 'fallback-token'),
    ],
)
def test_prefetch_github_context_selects_token(tmp_path: Path, env_vars: dict[str, str], expected_token: str):
    result, _ = _run_context_prefetch(
        tmp_path,
        gh_script=f'[ "$GH_TOKEN" = "{expected_token}" ] || exit 9\n' + _SUCCESSFUL_GH_PREFETCH,
        env_vars={**env_vars, 'GITHUB_REPOSITORY': 'pydantic/pydantic-ai'},
    )

    assert result.returncode == 0
    assert 'Could not prefetch' not in result.stdout


def test_prefetch_github_context_without_credential_is_non_fatal(tmp_path: Path):
    result, agent_dir = _run_context_prefetch(tmp_path)

    assert result.returncode == 0
    assert 'No GitHub credential' in result.stdout
    assert not (agent_dir / 'github-context/open-issues.json').exists()
    assert not (agent_dir / 'github-context/open-pull-requests.json').exists()


def test_prefetch_github_context_without_repository_is_non_fatal(tmp_path: Path):
    result, agent_dir = _run_context_prefetch(tmp_path, env_vars={'GH_TOKEN': 'test-token'})

    assert result.returncode == 0
    assert 'GITHUB_REPOSITORY is unavailable' in result.stdout
    assert not (agent_dir / 'github-context/open-issues.json').exists()
    assert not (agent_dir / 'github-context/open-pull-requests.json').exists()


@pytest.mark.parametrize(
    ('failed_command', 'preserved_file', 'missing_file', 'warning'),
    [
        (
            'issue:list',
            'open-pull-requests.json',
            'open-issues.json',
            'Could not prefetch open issues',
        ),
        (
            'pr:list',
            'open-issues.json',
            'open-pull-requests.json',
            'Could not prefetch open pull requests',
        ),
    ],
)
def test_prefetch_github_context_preserves_independent_corpus(
    tmp_path: Path,
    failed_command: str,
    preserved_file: str,
    missing_file: str,
    warning: str,
):
    result, agent_dir = _run_context_prefetch(
        tmp_path,
        gh_script=f"""\
[ "$1:$2" = "{failed_command}" ] && exit 7
{_SUCCESSFUL_GH_PREFETCH}
""",
        env_vars={'GH_TOKEN': 'test-token', 'GITHUB_REPOSITORY': 'pydantic/pydantic-ai'},
    )

    context_dir = agent_dir / 'github-context'
    assert result.returncode == 0
    assert warning in result.stdout
    assert (context_dir / preserved_file).exists()
    assert not (context_dir / missing_file).exists()


@pytest.mark.parametrize('malformed_command', ['issue:list', 'pr:list'])
def test_prefetch_github_context_rejects_malformed_corpus(tmp_path: Path, malformed_command: str):
    result, agent_dir = _run_context_prefetch(
        tmp_path,
        gh_script=f"""\
if [ "$1:$2" = "{malformed_command}" ]; then
  printf '%s\\n' '{{"message":"unexpected response"}}'
  exit 0
fi
{_SUCCESSFUL_GH_PREFETCH}
""",
        env_vars={'GH_TOKEN': 'test-token', 'GITHUB_REPOSITORY': 'pydantic/pydantic-ai'},
    )

    rejected_corpus = 'open-issues.json' if malformed_command == 'issue:list' else 'open-pull-requests.json'
    preserved_corpus = 'open-pull-requests.json' if malformed_command == 'issue:list' else 'open-issues.json'
    assert result.returncode == 0
    assert 'Could not prefetch' in result.stdout
    assert not (agent_dir / 'github-context' / rejected_corpus).exists()
    assert (agent_dir / 'github-context' / preserved_corpus).exists()


# 1,277 open issues filled a cap of 1000, which dropped the corpus from every issue-filing sweep.
@pytest.mark.parametrize(('open_issues', 'kept'), [(1277, True), (10000, False)])
def test_prefetch_github_context_rejects_only_a_capped_corpus(tmp_path: Path, open_issues: int, kept: bool):
    result, agent_dir = _run_context_prefetch(
        tmp_path,
        gh_script=f"""\
if [ "$1:$2" = "issue:list" ]; then
  python3 -c 'import json; print(json.dumps([{{}}] * {open_issues}))'
  exit 0
fi
"""
        + _SUCCESSFUL_GH_PREFETCH,
        env_vars={'GH_TOKEN': 'test-token', 'GITHUB_REPOSITORY': 'pydantic/pydantic-ai'},
    )

    context_dir = agent_dir / 'github-context'
    assert result.returncode == 0
    assert ('Could not prefetch open issues' in result.stdout) is not kept
    assert (context_dir / 'open-issues.json').exists() is kept
    assert (context_dir / 'open-pull-requests.json').exists()


# GitHub search returns at most 1,000 results whatever `--limit` asks for, so a search-sorted list
# of 1,237 open issues kept only the 1,000 least recently updated. This fake caps search at 5.
_OPEN_ISSUES_GH = """\
[ "$1:$2" = "issue:list" ] || exit 8
python3 - {open_issues} "$@" <<'EOF'
import json, sys
open_issues, args = int(sys.argv[1]), sys.argv[2:]
returned = min(open_issues, int(args[args.index('--limit') + 1]), 5 if '--search' in args else open_issues)
print(json.dumps([
    {{'number': n, 'title': f'Issue {{n}}', 'updatedAt': f'2026-01-{{n:02d}}T00:00:00Z', 'createdAt': '2026-01-01T00:00:00Z', 'labels': []}}
    for n in range(returned, 0, -1)
]))
EOF
"""


def test_prefetch_open_issues_keeps_every_open_issue_oldest_update_first(tmp_path: Path):
    result, agent_dir = _run_context_prefetch(
        tmp_path,
        script='prefetch-open-issues.sh',
        gh_script=_OPEN_ISSUES_GH.format(open_issues=7),
        env_vars={'GITHUB_REPOSITORY': 'pydantic/pydantic-ai', 'ISSUE_LIMIT': '20'},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rows = (agent_dir / 'open-issues.tsv').read_text(encoding='utf-8').splitlines()[1:]
    assert [row.split('\t')[0] for row in rows] == ['1', '2', '3', '4', '5', '6', '7']
    assert len(list((agent_dir / 'issues/all').iterdir())) == 7


def test_prefetch_open_issues_fails_on_a_capped_list(tmp_path: Path):
    result, agent_dir = _run_context_prefetch(
        tmp_path,
        script='prefetch-open-issues.sh',
        gh_script=_OPEN_ISSUES_GH.format(open_issues=20),
        env_vars={'GITHUB_REPOSITORY': 'pydantic/pydantic-ai', 'ISSUE_LIMIT': '20'},
    )

    assert result.returncode != 0
    assert '::error::' in result.stdout
    assert not list((agent_dir / 'issues/all').iterdir())


@pytest.mark.parametrize(
    'prompt_name',
    [
        'pydantic-ai-bug-hunter.md',
        'pydantic-ai-docs-drift.md',
        'pydantic-ai-provider-mapping-sweep.md',
        'pydantic-ai-provider-parity-explore.md',
        'pydantic-ai-regression-detector.md',
        'pydantic-ai-roundtrip-sweep.md',
        'pydantic-ai-streaming-resilience-sweep.md',
    ],
)
def test_issue_filing_prompts_use_prefetched_context(prompt_name: str):
    prompt = Path(__file__).parent.parent / 'workflows' / 'shared' / 'prompts' / prompt_name
    text = prompt.read_text(encoding='utf-8')

    assert '/tmp/gh-aw/agent/github-context/open-issues.json' in text
    assert 'gh issue list' not in text
    assert 'gh pr list' not in text
    assert 'gh api --paginate' not in text

    workflow_name = prompt_name.removesuffix('.md')
    workflow = prompt.parent.parent.parent / f'{workflow_name}.md'
    assert '  - shared/issue-filing-context.md' in workflow.read_text(encoding='utf-8')


def test_shared_context_setup_is_scoped_and_uses_runtime_paths():
    shared = Path(__file__).parent.parent / 'workflows' / 'shared' / 'engine-pydantic-ai.md'
    shared_text = shared.read_text(encoding='utf-8')
    context_text = shared.with_name('issue-filing-context.md').read_text(encoding='utf-8')
    tool_hints = shared.with_name('tool-hints.md').read_text(encoding='utf-8')

    assert 'Stage the workflow agent composition' in shared_text
    assert 'Install tools for the AWF sandbox' in shared_text
    assert 'Forward agent spans to the OTLP endpoint' in shared_text
    assert 'GH_TOKEN' not in shared_text.split('post-steps:')[0]
    assert 'run: bash .github/scripts/prefetch-github-context.sh' in context_text
    assert '      GH_TOKEN: ${{ github.token }}' in context_text
    assert '$GITHUB_WORKSPACE/.review-context/' in tool_hints

    from gh_aw_agent._agent import INSTRUCTIONS

    assert '$GITHUB_WORKSPACE/.review-context/' in INSTRUCTIONS


def test_shared_tool_hints_route_safe_outputs_to_the_offered_tools():
    tool_hints = (Path(__file__).parent.parent / 'workflows' / 'shared' / 'tool-hints.md').read_text(encoding='utf-8')
    assert 'safeoutputs_noop' in tool_hints
    assert 'mcp__safeoutputs__' not in tool_hints
