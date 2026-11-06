"""Verify the OpenCode fallback's checkout boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

from douwebot_opencode_run import check_symlinks


def test_internal_symlinks_are_allowed(tmp_path: Path) -> None:
    """Existing links between files and directories in a checkout are safe."""
    root = tmp_path / 'checkout'
    (root / 'directory').mkdir(parents=True)
    (root / 'directory' / 'target').write_text('content')
    (root / 'file-link').symlink_to('directory/target')
    (root / 'directory-link').symlink_to('directory', target_is_directory=True)
    check_symlinks(root)


@pytest.mark.parametrize('kind', ['external', 'dangling', 'cycle', 'ancestor'])
def test_unsafe_symlink_fails_closed(tmp_path: Path, kind: str) -> None:
    """A review cannot start with links that escape or cannot resolve safely."""
    root = tmp_path / 'checkout'
    root.mkdir()
    if kind == 'external':
        (tmp_path / 'outside').write_text('secret')
        (root / 'link').symlink_to('../outside')
    elif kind == 'dangling':
        (root / 'link').symlink_to('missing')
    elif kind == 'cycle':
        (root / 'link').symlink_to('other')
        (root / 'other').symlink_to('link')
    else:
        (root / 'link').symlink_to('.', target_is_directory=True)
    with pytest.raises(ValueError, match=r'[Ss]ymlink'):
        check_symlinks(root)


def test_pr_checkout_is_checked_separately(tmp_path: Path) -> None:
    """An untrusted head cannot link into the surrounding base checkout."""
    root = tmp_path / 'base'
    head = root / 'pr-head'
    head.mkdir(parents=True)
    (root / 'trusted-file').write_text('base')
    (head / 'link').symlink_to('../trusted-file')
    check_symlinks(root, skip_pr_head=True)
    with pytest.raises(ValueError, match='escapes'):
        check_symlinks(head)


def test_launcher_sends_shared_review_and_restricts_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback keeps the review rubric and does not pass GitHub credentials to OpenCode."""
    import json
    import subprocess

    import douwebot_opencode_run
    import yaml

    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'pr-head').mkdir()
    runner_temp = tmp_path / 'runner-temp'
    runner_temp.mkdir()
    workflow_path = Path(__file__).parent.parent / 'workflows' / 'bots.yml'
    workflow = yaml.load(workflow_path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    review_steps = workflow['jobs']['review']['steps']
    prompt = next(step['with']['prompt'] for step in review_steps if step.get('id') == 'review')
    monkeypatch.setenv('GITHUB_WORKSPACE', str(workspace))
    monkeypatch.setenv('RUNNER_TEMP', str(runner_temp))
    monkeypatch.setenv('REVIEW_PROMPT', prompt)
    monkeypatch.setenv('OPENCODE_API_KEY', 'fake-key')
    monkeypatch.setenv('GH_TOKEN', 'fake-github-token')
    invoked: list[dict[str, object]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = TypeAdapter(dict[str, str]).validate_python(kwargs['env'])
        config_home = env['XDG_CONFIG_HOME']
        config = json.loads((Path(config_home) / 'opencode' / 'opencode.json').read_text())
        assert config['agents']['title']['model'] == 'opencode-go/deepseek-v4.1-flash'
        assert config['agents']['compaction']['model'] == 'opencode-go/deepseek-v4.1-flash'
        invoked.append({'args': args, **kwargs})
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, 'run', fake_run)
    douwebot_opencode_run.main()
    assert len(invoked) == 1
    call = invoked[0]
    assert call['args'] == [
        'opencode',
        'run',
        '--standalone',
        '--model',
        'opencode-go/deepseek-v4.1-flash',
        '--agent',
        'reviewer',
        '--format',
        'json',
    ]
    assert isinstance(call['input'], str)
    assert 'What to look for' in call['input']
    assert 'Live GitHub and Git commands are unavailable' in call['input']
    assert 'Before returning the structured result, run `gh pr view' not in call['input']
    assert 'Use the `gh` CLI only when' not in call['input']
    assert isinstance(call['env'], dict)
    assert call['env']['OPENCODE_API_KEY'] == 'fake-key'
    assert 'GH_TOKEN' not in call['env']
