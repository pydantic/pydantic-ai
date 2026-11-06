"""Run the OpenCode fallback with read-only tools against the pinned PR checkout."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from publish_douwebot_review import ReviewResult

MODEL = 'opencode-go/deepseek-v4.1-flash'


def check_symlinks(root: Path, *, skip_pr_head: bool = False) -> None:
    """Reject links which escape their checkout, do not resolve, or form a cycle."""
    root = root.resolve(strict=True)
    for directory, directories, files in os.walk(root, followlinks=False):
        if skip_pr_head and Path(directory) == root and 'pr-head' in directories:
            directories.remove('pr-head')
        if '.git' in directories:
            directories.remove('.git')
        for name in directories + files:
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            try:
                target = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError(f'Invalid symlink in review checkout: {path.relative_to(root)}') from exc
            if not target.is_relative_to(root):
                raise ValueError(f'Symlink escapes review checkout: {path.relative_to(root)}')
            if target.is_dir() and path.parent.is_relative_to(target):
                raise ValueError(f'Symlink loops through an ancestor: {path.relative_to(root)}')


def main() -> None:
    """Run the reviewer and save its event stream for validation."""
    workspace = Path(os.environ['GITHUB_WORKSPACE']).resolve(strict=True)
    check_symlinks(workspace, skip_pr_head=True)
    check_symlinks(workspace / 'pr-head')
    api_key = os.environ['OPENCODE_API_KEY']
    if not api_key:
        raise ValueError('OPENCODE_API_KEY is empty')
    prompt = os.environ['REVIEW_PROMPT']
    before, marker, after = prompt.partition('Use the `gh` CLI only when you need additional information')
    if not marker:
        raise ValueError('Review prompt is missing the GitHub tool guidance')
    _, marker, after = after.partition('Be careful about loading large diffs')
    if not marker:
        raise ValueError('Review prompt is missing the diff guidance')
    prompt = before + 'Live GitHub and Git commands are unavailable; use the prefetched context.\n\n' + marker + after
    before, marker, after = prompt.partition('Before returning the structured result, run `gh pr view')
    if not marker or '\n' not in after:
        raise ValueError('Review prompt is missing the head-check guidance')
    prompt = before + 'The trusted publisher checks the live PR head before publication.' + after[after.index('\n') :]
    prompt += (
        '\n\nOpenCode has read, grep, and glob tools only. Shell, live GitHub and Git commands are unavailable. '
        'Use the prefetched review context. The trusted publisher checks the live PR head. '
        'If missing information prevents a complete review, return INCOMPLETE. '
        'Return only one JSON object matching this schema: '
        + json.dumps(ReviewResult.model_json_schema(), separators=(',', ':'))
    )
    runner_temp = Path(os.environ['RUNNER_TEMP'])
    # The wildcard deny covers external_directory for direct outside reads. OpenCode checks paths
    # lexically, so check_symlinks above also rejects links that could escape either checkout.
    permissions: list[dict[str, str]] = [
        {'action': '*', 'resource': '*', 'effect': 'deny'},
        *({'action': action, 'resource': '*', 'effect': 'allow'} for action in ('read', 'grep', 'glob')),
    ]
    policies: list[dict[str, str]] = [
        {'action': 'provider.use', 'resource': '*', 'effect': 'deny'},
        {'action': 'provider.use', 'resource': 'opencode-go', 'effect': 'allow'},
        {'action': 'plugin.use', 'resource': '*', 'effect': 'deny'},
        {'action': 'mcp.use', 'resource': '*', 'effect': 'deny'},
        {'action': 'tool.use', 'resource': '*', 'effect': 'deny'},
        *({'action': 'tool.use', 'resource': f'{action}:*', 'effect': 'allow'} for action in ('read', 'grep', 'glob')),
    ]
    config: dict[str, object] = {
        'plugins': ['-opencode.browser'],
        'snapshots': False,
        'default_agent': 'reviewer',
        'permissions': permissions,
        'agents': {
            'reviewer': {'mode': 'primary', 'permissions': permissions},
            'title': {'model': MODEL},
            'compaction': {'model': MODEL},
        },
        'policies': policies,
    }

    with tempfile.TemporaryDirectory(dir=runner_temp, prefix='douwebot-opencode-') as temp:
        temp_root = Path(temp)
        config_dir = temp_root / 'config' / 'opencode'
        config_dir.mkdir(parents=True)
        (config_dir / 'opencode.json').write_text(json.dumps(config), encoding='utf-8')
        env = {
            'HOME': str(temp_root),
            'PATH': os.environ['PATH'],
            'XDG_CONFIG_HOME': str(temp_root / 'config'),
            'XDG_DATA_HOME': str(temp_root / 'data'),
            'XDG_CACHE_HOME': str(temp_root / 'cache'),
            'OPENCODE_API_KEY': api_key,
        }
        with (runner_temp / 'douwebot-opencode.ndjson').open('w', encoding='utf-8') as output:
            subprocess.run(
                ['opencode', 'run', '--standalone', '--model', MODEL, '--agent', 'reviewer', '--format', 'json'],
                input=prompt,
                text=True,
                cwd=workspace,
                env=env,
                stdout=output,
                check=True,
            )


if __name__ == '__main__':
    main()
