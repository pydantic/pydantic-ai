"""Run OpenCode with read-only tools; `main` runs the douwebot fallback review against the pinned PR checkout."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, TypeVar

from publish_douwebot_review import ReviewEngine, ReviewResult
from pydantic import TypeAdapter, ValidationError

T = TypeVar('T')
_EVENT = TypeAdapter(dict[str, object])
# `provider.go-limit` is a spent OpenCode Go usage window; `provider.quota` is a quota error OpenCode cannot name.
QUOTA_ERRORS = frozenset({'provider.go-limit', 'provider.quota'})
# The shape of every error type OpenCode 2.2.5 sets, such as `provider.auth` or `unknown`.
ERROR_TYPE = re.compile(r'[a-z][a-z.-]*')


@dataclass(frozen=True)
class Provider:
    """An OpenCode provider of the review model, the variable holding its key, and its douwebot receipt engine."""

    name: str
    model: str
    key_variable: str
    review_engine: ReviewEngine


OPENCODE_GO = Provider('opencode-go', 'opencode-go/deepseek-v4.1-flash', 'OPENCODE_API_KEY', 'opencode')
DEEPINFRA = Provider(
    'deepinfra', 'deepinfra/deepseek-ai/DeepSeek-V4.1-Flash', 'DEEPINFRA_API_KEY', 'opencode-deepinfra'
)


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


def read_only_config(*, agent: str, provider: Provider, steps: int | None = None) -> dict[str, object]:
    """Allow only the read, grep and glob tools, and only `provider`."""
    # The wildcard deny covers external_directory for direct outside reads. OpenCode checks paths
    # lexically, so callers run check_symlinks to reject links that could escape `cwd`.
    permissions: list[dict[str, str]] = [
        {'action': '*', 'resource': '*', 'effect': 'deny'},
        *({'action': action, 'resource': '*', 'effect': 'allow'} for action in ('read', 'grep', 'glob')),
    ]
    policies: list[dict[str, str]] = [
        {'action': 'provider.use', 'resource': '*', 'effect': 'deny'},
        {'action': 'provider.use', 'resource': provider.name, 'effect': 'allow'},
        {'action': 'plugin.use', 'resource': '*', 'effect': 'deny'},
        {'action': 'mcp.use', 'resource': '*', 'effect': 'deny'},
        {'action': 'tool.use', 'resource': '*', 'effect': 'deny'},
        *({'action': 'tool.use', 'resource': f'{action}:*', 'effect': 'allow'} for action in ('read', 'grep', 'glob')),
    ]
    agent_config: dict[str, object] = {'mode': 'primary', 'permissions': permissions}
    if steps is not None:
        agent_config['steps'] = steps
    return {
        'plugins': ['-opencode.browser'],
        'snapshots': False,
        'default_agent': agent,
        'permissions': permissions,
        'agents': {
            agent: agent_config,
            'title': {'model': provider.model},
            'compaction': {'model': provider.model},
        },
        'policies': policies,
    }


def run_with_fallback(
    launch: Callable[[list[str], dict[str, str], TextIO], T],
    *,
    agent: str,
    output: Path,
    steps: int | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> tuple[T, Provider]:
    """Launch OpenCode on Go, and again on DeepInfra when a Go quota limit refused the run before it finished a step.

    `launch` runs the command in the environment it is given and writes the event stream to the stream it is given,
    which `output` holds after each run. Each run gets a config that allows only its provider, and an environment
    that holds only that provider's key. A run whose process fails prints its model and OpenCode error type.
    """

    def run_on(provider: Provider, api_key: str) -> tuple[T, Provider]:
        with (
            tempfile.TemporaryDirectory(dir=os.environ['RUNNER_TEMP'], prefix='opencode-') as temp,
            output.open('w', encoding='utf-8') as stream,
        ):
            temp_root = Path(temp)
            config_dir = temp_root / 'config' / 'opencode'
            config_dir.mkdir(parents=True)
            config = read_only_config(agent=agent, provider=provider, steps=steps)
            (config_dir / 'opencode.json').write_text(json.dumps(config), encoding='utf-8')
            env = {
                'HOME': str(temp_root),
                'PATH': os.environ['PATH'],
                'XDG_CONFIG_HOME': str(temp_root / 'config'),
                'XDG_DATA_HOME': str(temp_root / 'data'),
                'XDG_CACHE_HOME': str(temp_root / 'cache'),
                provider.key_variable: api_key,
                **(extra_env or {}),
            }
            command = [
                'opencode',
                'run',
                '--standalone',
                '--model',
                provider.model,
                '--agent',
                agent,
                '--format',
                'json',
            ]
            result = launch(command, env, stream)
        print(f'OpenCode run served by {provider.model}.')
        return result, provider

    def report_failure(provider: Provider) -> tuple[str | None, bool]:
        """Print the failed run's model, error type and status; return that type and whether the run finished a step."""
        try:
            # Only LF ends an event, as in `douwebot_opencode_result.final_text`.
            events = [_EVENT.validate_json(line) for line in output.read_text(encoding='utf-8').split('\n') if line]
            errors = [_EVENT.validate_python(event.get('error')) for event in events if event.get('type') == 'error']
        except ValidationError:
            # A malformed stream names no error. Its validation error would copy the stream into the public log.
            events, errors = [], []
        error = errors[-1] if errors else {}
        # Job logs are public, and the error's message and response body can repeat the provider's response.
        # OpenCode names its own error types; print only a value shaped like one.
        kind, status = error.get('type'), error.get('status')
        if not isinstance(kind, str) or not ERROR_TYPE.fullmatch(kind):
            kind = None
        described = f'error type {kind}' if kind else 'no error type'
        if isinstance(status, int):
            described += f', status {status}'
        print(f'OpenCode run on {provider.model} failed with {described}.')
        return kind, any(event.get('type') == 'step_finish' for event in events)

    go_key = os.environ[OPENCODE_GO.key_variable]
    if not go_key:
        raise ValueError(f'{OPENCODE_GO.key_variable} is empty')
    try:
        return run_on(OPENCODE_GO, go_key)
    except subprocess.CalledProcessError:
        kind, finished = report_failure(OPENCODE_GO)
        fallback_key = os.environ.get(DEEPINFRA.key_variable, '')
        # OpenCode 2.2.5 does not retry a quota rejection. It writes `step_start`, then an `error` event whose type
        # `toSessionError` sets, and no `step_finish`:
        # https://github.com/anomalyco/opencode/blob/v2.2.5/packages/core/src/session/to-session-error.ts
        if not fallback_key or finished or kind not in QUOTA_ERRORS:
            raise
        print('OpenCode Go refused the run before its first step; retrying on DeepInfra.')
    try:
        return run_on(DEEPINFRA, fallback_key)
    except subprocess.CalledProcessError:
        report_failure(DEEPINFRA)
        raise


def run_read_only(prompt: str, *, cwd: Path, agent: str, output: Path, timeout: float | None = None) -> Provider:
    """Run OpenCode in `cwd` with read, grep, and glob tools only, saving its event stream to `output`.

    `timeout` bounds both runs together when Go refuses the first. Returns the provider that served the run.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    _, provider = run_with_fallback(
        lambda command, env, stream: subprocess.run(
            command,
            input=prompt,
            text=True,
            cwd=cwd,
            env=env,
            stdout=stream,
            check=True,
            timeout=None if deadline is None else deadline - time.monotonic(),
        ),
        agent=agent,
        output=output,
    )
    return provider


def main() -> None:
    """Run the reviewer and save its event stream for validation."""
    workspace = Path(os.environ['GITHUB_WORKSPACE']).resolve(strict=True)
    check_symlinks(workspace, skip_pr_head=True)
    check_symlinks(workspace / 'pr-head')
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
        '\n\nOpenCode has read, grep, and glob tools only. Shell is unavailable. '
        'If missing information prevents a complete review, return INCOMPLETE. '
        'Return only one JSON object matching this schema: '
        + json.dumps(ReviewResult.model_json_schema(), separators=(',', ':'))
    )
    provider = run_read_only(
        prompt,
        cwd=workspace,
        agent='reviewer',
        output=Path(os.environ['RUNNER_TEMP']) / 'douwebot-opencode.ndjson',
    )
    with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as github_output:
        github_output.write(f'engine={provider.review_engine}\n')


if __name__ == '__main__':
    main()
