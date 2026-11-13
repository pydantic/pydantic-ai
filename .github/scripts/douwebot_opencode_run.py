"""Run OpenCode with read-only tools; `main` runs the douwebot fallback review against the pinned PR checkout."""

from __future__ import annotations

import http.client
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO, TypeVar
from urllib.parse import quote

from publish_douwebot_review import ReviewEngine, ReviewResult
from pydantic import TypeAdapter, ValidationError

T = TypeVar('T')
_EVENT = TypeAdapter(dict[str, object])
_TOKENS = TypeAdapter(dict[str, float | dict[str, float]])
# The shape of every error type OpenCode 2.2.5 sets, such as `provider.auth` or `unknown`.
ERROR_TYPE = re.compile(r'[a-z][a-z.-]*')
# The shape of a provider's own error code, such as Z.AI's `1302` request-rate limit or `1305` overload.
PROVIDER_CODE = re.compile(r'[0-9]{1,6}')
LOGFIRE_URL = 'https://logfire-api.pydantic.dev'


@dataclass(frozen=True)
class Provider:
    """An OpenCode provider of a review model, the account key it runs with, and its douwebot receipt engine."""

    name: str
    model: str
    key_variable: str
    """The variable OpenCode reads the provider's key from."""
    secret_variable: str
    """The workflow variable that holds this account's key."""
    plan: str
    """The subscription the job log names for this account."""
    review_engine: ReviewEngine
    handover_errors: frozenset[str]
    """OpenCode error types that hand a run this provider refused to the next provider in the chain."""
    restart_on_rate_limit: bool = False
    """Whether a `provider.rate-limit` after the run finished a step hands the run to the next provider anew."""


# OpenCode 2.2.5 types a spent OpenCode Go usage window `provider.go-limit`, and a quota it cannot name
# `provider.quota`. It also types any 402, and a Z.AI 429 whose message names a plan, quota or balance limit, such as
# codes 1308, 1309 and 1310, `provider.quota`. It retries any other 429, such as 1302 rate limits and 1305 overload,
# and types one that outlasts its retries `provider.rate-limit`: `classify` in
# https://github.com/anomalyco/opencode/blob/v2.2.5/packages/ai/src/provider-error.ts
OPENCODE_GO = Provider(
    'opencode-go',
    'opencode-go/deepseek-v4.1-flash',
    'OPENCODE_API_KEY',
    'OPENCODE_API_KEY',
    'OpenCode Go',
    'opencode',
    frozenset({'provider.go-limit', 'provider.quota'}),
)
# Each Coding Plan account hands a run its quota or rate limit refused to the other account. A rate limit after the
# run took steps restarts the run on the other account too.
CODING_PLAN_HANDOVER_ERRORS = frozenset({'provider.quota', 'provider.rate-limit'})
# The GLM Coding Plan, on its own coding endpoint. OpenCode 2.2.5 reads its key from `ZHIPU_API_KEY` only.
ZAI = Provider(
    'zai-coding-plan',
    'zai-coding-plan/glm-5.3-flash',
    'ZHIPU_API_KEY',
    'ZAI_API_KEY',
    'the primary Coding Plan',
    'opencode-zai',
    CODING_PLAN_HANDOVER_ERRORS,
    restart_on_rate_limit=True,
)
ZAI_BACKUP = Provider(
    'zai-coding-plan',
    'zai-coding-plan/glm-5.3-flash',
    'ZHIPU_API_KEY',
    'ZAI_BACKUP_API_KEY',
    'the backup Coding Plan',
    'opencode-zai',
    CODING_PLAN_HANDOVER_ERRORS,
    restart_on_rate_limit=True,
)


@dataclass
class Attempt:
    """One run of the fallback chain. A run whose process failed also records its stream's last `error` event."""

    provider: Provider
    started: int = field(default_factory=time.time_ns)
    """When the run started, in nanoseconds since the epoch."""
    ended: int = 0
    """When the run's process exited or was interrupted, in nanoseconds since the epoch; 0 while it runs."""
    steps: int | None = None
    """The `step_finish` events in the run's stream; `None` until the run's process ends."""
    tokens: int | None = None
    """The tokens those events report, every kind summed; `None` until the run's process ends."""
    error_type: str | None = None
    status: int | None = None
    code: str | None = None
    """The provider's own error code from the response body, such as Z.AI's `1302`."""
    finished: bool | None = None
    """Whether the failed run's stream holds a `step_finish` event; `None` while the run has not failed."""


def step_tokens(line: str) -> int | None:
    """Return the tokens a `step_finish` event reports, or None for any other event."""
    try:
        event = _EVENT.validate_json(line)
        if event.get('type') != 'step_finish':
            return None
        part = _EVENT.validate_python(event.get('part'))
        tokens = _TOKENS.validate_python(part.get('tokens', {}))
    except ValidationError:
        # A validation error repeats its input, a stream line can hold model text, and job logs are public.
        raise ValueError('OpenCode wrote an invalid stream event') from None
    total = 0.0
    for value in tokens.values():
        total += sum(value.values()) if isinstance(value, dict) else value
    return int(total)


def logfire_endpoint() -> str:
    """The Logfire base URL from `LOGFIRE_URL`, or Logfire's default."""
    return (os.environ.get('LOGFIRE_URL') or LOGFIRE_URL).rstrip('/')


def otlp_span(
    name: str,
    attributes: Mapping[str, str | int | bool],
    started: int,
    ended: int,
    *,
    trace_id: str,
    span_id: str,
    parent_id: str | None = None,
    failed: bool,
) -> dict[str, object]:
    """Build one OTLP/JSON span; `failed` sets its status to error."""
    return {
        'traceId': trace_id,
        'spanId': span_id,
        **({'parentSpanId': parent_id} if parent_id else {}),
        'name': name,
        'kind': 1,
        'startTimeUnixNano': str(started),
        'endTimeUnixNano': str(ended),
        'attributes': [
            {
                'key': key,
                'value': {'boolValue': value}
                if isinstance(value, bool)
                else {'intValue': str(value)}
                if isinstance(value, int)
                else {'stringValue': value},
            }
            for key, value in attributes.items()
        ],
        # OTLP's `STATUS_CODE_ERROR`.
        **({'status': {'code': 2}} if failed else {}),
    }


def export_spans(token: str, service: str, spans: list[dict[str, object]], *, timeout: float = 30) -> None:
    """Send `spans` to Logfire over OTLP/HTTP as `service`; a failed export only warns."""
    resource = {'attributes': [{'key': 'service.name', 'value': {'stringValue': service}}]}
    body = {'resourceSpans': [{'resource': resource, 'scopeSpans': [{'scope': {'name': service}, 'spans': spans}]}]}
    request = urllib.request.Request(
        f'{logfire_endpoint()}/v1/traces',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': token},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            pass
    # Callers export while a failed run's own exception propagates, so a failed export must not replace it.
    except (OSError, ValueError, http.client.HTTPException) as error:
        print(f'::warning::Could not export spans to Logfire: {error}')


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
    config: dict[str, object] = {
        'plugins': ['-opencode.browser'],
        'snapshots': False,
        'default_agent': agent,
        'permissions': permissions,
        'agents': {
            agent: agent_config,
            'title': {'model': provider.model},
            'compaction': {'model': provider.model},
            'summary': {'model': provider.model},
        },
        'policies': policies,
    }
    return config


def _stream_usage(output: Path) -> tuple[int, int]:
    """Count a run's `step_finish` events and their tokens, skipping every line that is not a valid event."""
    steps = tokens = 0
    for line in output.read_bytes().decode(errors='replace').split('\n'):
        try:
            step = step_tokens(line)
        except ValueError:
            continue
        if step is not None:
            steps += 1
            tokens += step
    return steps, tokens


def _export_run(
    token: str, attempt: Attempt, resource: Mapping[str, str], *, trace_id: str, parent_id: str | None, served: bool
) -> None:
    """Export the `OpenCode run` span of one ended run, with its provider, usage and how it failed, if `token` is set."""
    if not token:
        return
    fields = {
        'steps': attempt.steps,
        'tokens': attempt.tokens,
        'error_type': attempt.error_type,
        'error_status': attempt.status,
        'error_code': attempt.code,
        'step_finish': attempt.finished,
    }
    values = {
        **resource,
        'opencode_fallback.provider': attempt.provider.name,
        **{f'opencode_fallback.{key}': value for key, value in fields.items() if value is not None},
    }
    span = otlp_span(
        'OpenCode run',
        values,
        attempt.started,
        attempt.ended,
        trace_id=trace_id,
        span_id=secrets.token_hex(8),
        parent_id=parent_id,
        failed=not served,
    )
    # The export waits between the runs of its chain, inside the time budget of the run after it.
    export_spans(token, 'opencode-fallback', [span], timeout=5)


def run_with_fallback(
    launch: Callable[[list[str], dict[str, str], TextIO], T],
    *,
    agent: str,
    output: Path,
    steps: int | None = None,
    attributes: Mapping[str, str] | None = None,
    matrix_index: int = 0,
    attempts: list[Attempt] | None = None,
    trace_id: str | None = None,
    parent_id: str | None = None,
) -> tuple[T, Provider]:
    """Launch OpenCode on Go, then on one Z.AI Coding Plan account, then on the other, while each refuses the run.

    A provider hands the run to the next one whose key is set only when it refused the run with one of its
    `handover_errors` before the run finished a step, or, with `restart_on_rate_limit`, rate-limited it after. The next
    provider runs it from scratch, and each provider runs it at most once. The workflow's run number plus
    `matrix_index`, the run's place in its job matrix, picks the account tried first: the primary when the sum is even,
    the backup when it is odd. `launch` runs the command in the environment it is given and writes the event stream to
    the stream it is given, which `output` holds after each run. Each run gets a config that allows only its provider,
    and an environment whose only provider key is that account's. A run whose process fails prints its model, OpenCode
    error type, status and provider code. Each run appends an `Attempt` to `attempts`, so a caller can report a chain
    that raised. The returned result and `output` come from the run that served, so a run the chain moved past shows
    only as its `Attempt`, whatever steps and tokens it spent.

    With `LOGFIRE_TOKEN` set, OpenCode exports its spans and logs to Logfire over OTLP/HTTP. Their resource attributes
    name the repository, the workflow run, its attempt and job, then `attributes`, then the run's plan and its place in
    `attempts`, counting from 1, as `opencode_fallback.plan` and `opencode_fallback.attempt`. As each run ends, the
    chain also exports an `OpenCode run` span with those attributes and the run's `Attempt`, in trace `trace_id`, or a
    new trace, under span `parent_id`.
    """
    token = os.environ.get('LOGFIRE_TOKEN', '')
    attempts = [] if attempts is None else attempts
    trace_id = trace_id or secrets.token_hex(16)

    def run_on(provider: Provider, api_key: str) -> tuple[T, Provider]:
        attempt = Attempt(provider)
        attempts.append(attempt)
        resource = (
            {
                'github.repository': os.environ['GITHUB_REPOSITORY'],
                'github.run_id': os.environ['GITHUB_RUN_ID'],
                'github.run_attempt': os.environ['GITHUB_RUN_ATTEMPT'],
                'github.job': os.environ['GITHUB_JOB'],
                **(attributes or {}),
                'opencode_fallback.plan': provider.plan,
                'opencode_fallback.attempt': str(len(attempts)),
            }
            if token
            else {}
        )
        served = False
        try:
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
                }
                if token:
                    # OpenCode's spans and logs carry no prompt or completion text.
                    env |= {
                        'OTEL_EXPORTER_OTLP_ENDPOINT': logfire_endpoint(),
                        'OTEL_EXPORTER_OTLP_HEADERS': f'Authorization={token}',
                        'OTEL_RESOURCE_ATTRIBUTES': ','.join(
                            f'{key}={quote(value)}' for key, value in resource.items()
                        ),
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
                try:
                    result = launch(command, env, stream)
                    served = True
                finally:
                    # The next run overwrites the stream, and a failed run's usage is recorded nowhere else.
                    stream.flush()
                    attempt.steps, attempt.tokens = _stream_usage(output)
        except subprocess.CalledProcessError:
            report_failure(attempt)
            raise
        finally:
            attempt.ended = time.time_ns()
            # Each run exports its span as it ends, so neither a later run's failure nor a cancellation drops it.
            _export_run(token, attempt, resource, trace_id=trace_id, parent_id=parent_id, served=served)
        print(f'OpenCode run served by {provider.model} on {provider.plan}.')
        return result, provider

    def report_failure(attempt: Attempt) -> None:
        """Print the failed run's error type, status and provider code; record them and whether it finished a step."""
        try:
            # Only LF ends an event, as in `douwebot_opencode_result.final_text`.
            events = [_EVENT.validate_json(line) for line in output.read_text(encoding='utf-8').split('\n') if line]
            errors = [_EVENT.validate_python(event.get('error')) for event in events if event.get('type') == 'error']
        except ValueError:
            # A malformed or undecodable stream names no error, and the caller re-raises the process failure.
            # A validation error would copy the stream into the public log.
            events, errors = [], []
        error = errors[-1] if errors else {}
        try:
            # Z.AI's response body is `{"error": {"code": "1302", "message": ...}}`.
            body = _EVENT.validate_json(str(_EVENT.validate_python(error.get('response')).get('body')))
            code = str(_EVENT.validate_python(body.get('error')).get('code'))
        except ValidationError:
            code = ''
        # Job logs are public, and the error's message and response body can repeat the provider's response.
        # OpenCode names its own error types; print only a value shaped like one, and only a code of one to six digits.
        kind, status = error.get('type'), error.get('status')
        attempt.error_type = kind if isinstance(kind, str) and ERROR_TYPE.fullmatch(kind) else None
        attempt.status = status if isinstance(status, int) else None
        attempt.code = code if PROVIDER_CODE.fullmatch(code) else None
        attempt.finished = any(event.get('type') == 'step_finish' for event in events)
        described = f'error type {attempt.error_type}' if attempt.error_type else 'no error type'
        if attempt.status is not None:
            described += f', status {attempt.status}'
        if attempt.code:
            described += f', code {attempt.code}'
        print(f'OpenCode run on {attempt.provider.model} on {attempt.provider.plan} failed with {described}.')

    if not os.environ[OPENCODE_GO.secret_variable]:
        raise ValueError(f'{OPENCODE_GO.secret_variable} is empty')
    # A Z.AI account without a key is left out, so a run never waits on a provider it cannot reach.
    plans = [provider for provider in (ZAI, ZAI_BACKUP) if os.environ.get(provider.secret_variable)]
    # Run numbers count up within a workflow, so a burst of runs, and the checks of one matrix, alternate accounts.
    if (int(os.environ['GITHUB_RUN_NUMBER']) + matrix_index) % 2:
        plans.reverse()
    chain = [OPENCODE_GO, *plans]
    for provider, successor in zip(chain, chain[1:]):
        try:
            return run_on(provider, os.environ[provider.secret_variable])
        except subprocess.CalledProcessError:
            attempt = attempts[-1]
            # OpenCode 2.2.5 does not retry a quota rejection. It writes `step_start`, then an `error` event whose
            # type `toSessionError` sets, and no `step_finish`:
            # https://github.com/anomalyco/opencode/blob/v2.2.5/packages/core/src/session/to-session-error.ts
            if not attempt.finished and attempt.error_type in provider.handover_errors:
                print(
                    f'{provider.model} refused the run before its first step; '
                    f'retrying on {successor.model} on {successor.plan}.'
                )
            elif attempt.finished and provider.restart_on_rate_limit and attempt.error_type == 'provider.rate-limit':
                print(
                    f'{provider.model} rate-limited the run after it took steps; '
                    f'restarting the run on {successor.model} on {successor.plan}.'
                )
            else:
                raise
    return run_on(chain[-1], os.environ[chain[-1].secret_variable])


def run_read_only(
    prompt: str,
    *,
    cwd: Path,
    agent: str,
    output: Path,
    timeout: float | None = None,
    attempts: list[Attempt] | None = None,
) -> Provider:
    """Run OpenCode in `cwd` with read, grep, and glob tools only, saving its event stream to `output`.

    `timeout` bounds every run of the fallback chain together. Returns the provider that served the run, and appends
    each run to `attempts` as `run_with_fallback` does.
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
        attempts=attempts,
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
