"""Execute the PR Bots concurrency rule and the bot-review close-out helper with a recording gh fake."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

ROOT = Path(__file__).parents[2]
REPO = 'pydantic/pydantic-ai'
PULL = f'repos/{REPO}/pulls/123'
HEAD = '1' * 40


GH_STUB = r"""
gh() {
  local endpoint='' filter='' payload='' method=GET paginate=false slurp=false previous=''
  for arg in "$@"; do
    case "$arg" in repos/*) endpoint=$arg ;; --paginate) paginate=true ;; --slurp) slurp=true ;; esac
    [ "$previous" != '--jq' ] || filter=$arg
    if [ "$previous" = '--method' ] || [ "$previous" = '-X' ]; then method=$arg; fi
    previous=$arg
  done
  printf '%s %s paginate=%s slurp=%s\n' "$method" "${endpoint:-graphql}" "$paginate" "$slurp" >> "$CALLS"
  [ "${FAIL_ENDPOINT:-none}" != "${endpoint:-graphql}" ] || return 1
  if [ "$2" = graphql ]; then
    payload=$GRAPHQL_PAGES
    if [ "$paginate" != true ] || [ "$slurp" != true ]; then
      payload=$(jq -c '.[0]' <<<"$payload")
    fi
  elif [ "$endpoint" = "$PULL" ]; then
    payload=$PULL_JSON
  elif [ "$endpoint" = "$PULL/reviews" ]; then
    payload=$REVIEW_PAGES
    if [ "$paginate" != true ] || [ "$slurp" != true ]; then
      payload=$(jq -c '.[0]' <<<"$payload")
    fi
  elif [ "$endpoint" = "$PULL/reviews/101" ]; then
    payload=$REVIEW_JSON
  elif [ "$endpoint" = "$PULL/reviews/101/dismissals" ] && [ "$method" = PUT ]; then
    cat > "$DISMISS_BODY"
    payload='{"state":"DISMISSED"}'
  else
    echo "unexpected gh request: $*" >&2
    return 91
  fi
  if [ -n "$filter" ]; then jq -r "$filter" <<<"$payload"; else printf '%s\n' "$payload"; fi
}
"""


def run_shell(script: str, tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env: dict[str, str] = {
        **os.environ,
        'REPO': REPO,
        'PULL': PULL,
        'SHA': HEAD,
        'REVIEW_PAGES': '[[]]',
        'PULL_JSON': json.dumps({'state': 'open', 'head': {'sha': HEAD}}),
        'CALLS': str(tmp_path / 'calls'),
        'DISMISS_BODY': str(tmp_path / 'dismiss-body'),
        **overrides,
    }
    return subprocess.run(['bash', '-c', GH_STUB + '\n' + script], text=True, capture_output=True, env=env)


def evaluate_expression(expression: str, values: dict[str, str | bool | int]) -> object:
    expression = expression.strip().removeprefix('${{').removesuffix('}}')
    for name in sorted(values, key=len, reverse=True):
        expression = expression.replace(name, repr(values[name]))
    return eval(expression.replace('&&', ' and ').replace('||', ' or ').strip(), {'__builtins__': {}})


@pytest.mark.parametrize(
    ('action', 'label', 'isolated', 'cancels'),
    [
        ('labeled', 'douwebot', False, False),
        ('labeled', 'douwebot:review-agent-config', False, False),
        ('labeled', 'size: M', True, False),
        ('labeled', 'docs', True, False),
        ('unlabeled', 'douwebot', True, False),
        ('unlabeled', 'douwebot:review-agent-config', False, True),
        ('synchronize', '', False, True),
    ],
)
def test_unrelated_label_events_cannot_evict_pending_reviews(action: str, label: str, isolated: bool, cancels: bool):
    workflow = TypeAdapter(dict[str | bool, object]).validate_python(
        yaml.safe_load((ROOT / '.github/workflows/bots.yml').read_text())
    )
    concurrency = TypeAdapter(dict[str, str]).validate_python(workflow['concurrency'])
    values: dict[str, str | bool | int] = {
        'github.event.action': action,
        'github.event.label.name': label,
        "format('-noop-{0}', github.run_id)": '-noop-456',
    }
    suffix = concurrency['group'].split('}}', 1)[1]
    assert evaluate_expression(suffix, values) == ('-noop-456' if isolated else '')
    assert evaluate_expression(concurrency['cancel-in-progress'], values) == cancels


def threads_page(
    *,
    resolved: bool = True,
    head: str = HEAD,
    author: str = 'github-actions',
    reply_author: str | None = None,
    comments_complete: bool = True,
) -> dict[str, object]:
    comments: list[dict[str, dict[str, str]]] = [{'author': {'login': author}}]
    if reply_author is not None:
        comments.append({'author': {'login': reply_author}})
    return {
        'data': {
            'repository': {
                'pullRequest': {
                    'state': 'OPEN',
                    'headRefOid': head,
                    'reviewThreads': {
                        'nodes': [
                            {
                                'isResolved': resolved,
                                'comments': {
                                    'totalCount': len(comments) + (0 if comments_complete else 1),
                                    'nodes': comments,
                                },
                            }
                        ]
                    },
                }
            }
        }
    }


@pytest.mark.parametrize(
    'case',
    [
        'addressed',
        'human',
        'approved',
        'unresolved',
        'later-unresolved',
        'human-thread-bot-reply',
        'incomplete-comments',
        'unrelated-human-thread',
        'changed-page',
        'changed-final',
        'api-error',
    ],
)
def test_closeout_dismisses_only_verified_addressed_bot_verdicts(tmp_path: Path, case: str):
    reason = tmp_path / 'reason'
    reason.write_text('> Agent attribution\n\nVerified the fixes and resolved every blocking finding.\n')
    review: dict[str, object] = {
        'id': 101,
        'state': 'CHANGES_REQUESTED',
        'user': {'type': 'Bot', 'login': 'github-actions[bot]'},
    }
    pages = [threads_page()]
    live: dict[str, object] = {'state': 'open', 'head': {'sha': HEAD}}
    if case == 'human':
        review['user'] = {'type': 'User', 'login': 'maintainer'}
    elif case == 'approved':
        review['state'] = 'APPROVED'
    elif case == 'unresolved':
        pages = [threads_page(resolved=False)]
    elif case == 'later-unresolved':
        pages.append(threads_page(resolved=False))
    elif case == 'human-thread-bot-reply':
        pages = [threads_page(resolved=False, author='maintainer', reply_author='github-actions')]
    elif case == 'incomplete-comments':
        pages = [threads_page(resolved=False, author='maintainer', comments_complete=False)]
    elif case == 'unrelated-human-thread':
        pages = [threads_page(resolved=False, author='maintainer')]
    elif case == 'changed-page':
        pages.append(threads_page(head='2' * 40))
    elif case == 'changed-final':
        live['head'] = {'sha': '2' * 40}
    script = 'source "$HELPER" "$REPO" 123 101 "$SHA" "$REASON_FILE" --all-blockers-addressed'
    result = run_shell(
        script,
        tmp_path,
        HELPER=str(ROOT / '.agents/skills/pushing-commits-to-the-repo/dismiss-addressed-bot-review'),
        REASON_FILE=str(reason),
        REVIEW_JSON=json.dumps(review),
        GRAPHQL_PAGES=json.dumps(pages),
        PULL_JSON=json.dumps(live),
        FAIL_ENDPOINT=f'{PULL}/reviews/101' if case == 'api-error' else 'none',
    )
    dismissed = case in {'addressed', 'unrelated-human-thread'}
    assert result.returncode == (0 if dismissed else 1), result.stderr
    assert (tmp_path / 'dismiss-body').exists() == dismissed
    if dismissed:
        body = TypeAdapter(dict[str, str]).validate_json((tmp_path / 'dismiss-body').read_text())
        assert body['event'] == 'DISMISS'
        assert body['message'].startswith('> Agent attribution')
        assert f'Verified at `{HEAD}`' in body['message']


def test_closeout_requires_explicit_full_assessment(tmp_path: Path):
    result = run_shell(
        'source "$HELPER" "$REPO" 123 101 "$SHA" missing-reason',
        tmp_path,
        HELPER=str(ROOT / '.agents/skills/pushing-commits-to-the-repo/dismiss-addressed-bot-review'),
    )
    assert result.returncode == 2
    assert result.stderr.strip()
    assert not (tmp_path / 'calls').exists()


@pytest.mark.parametrize('case', ['repo', 'number', 'review-id', 'head', 'missing-reason', 'blank-reason'])
def test_invalid_closeout_inputs_cannot_dismiss_a_review(tmp_path: Path, case: str):
    reason = tmp_path / 'reason'
    reason.write_text('   ' if case == 'blank-reason' else 'Verified all blocking findings.')
    arguments: dict[str, str] = {
        'CLOSE_REPO': '../other/repo' if case == 'repo' else REPO,
        'CLOSE_PR': '0' if case == 'number' else '123',
        'CLOSE_REVIEW': 'invalid' if case == 'review-id' else '101',
        'CLOSE_HEAD': 'main' if case == 'head' else HEAD,
        'CLOSE_REASON': str(tmp_path / 'missing') if case == 'missing-reason' else str(reason),
    }
    result = run_shell(
        'source "$HELPER" "$CLOSE_REPO" "$CLOSE_PR" "$CLOSE_REVIEW" "$CLOSE_HEAD" "$CLOSE_REASON" --all-blockers-addressed',
        tmp_path,
        HELPER=str(ROOT / '.agents/skills/pushing-commits-to-the-repo/dismiss-addressed-bot-review'),
        **arguments,
    )
    assert result.returncode == 2
    assert result.stderr.strip()
    assert not (tmp_path / 'calls').exists()
