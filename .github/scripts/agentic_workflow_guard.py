"""Static policy checks for the `gh-aw` agentic workflows in `.github/workflows/`.

Every check here encodes a defect that actually reached `main` and burned model
budget before anyone noticed, documented in #6766:

- `dangling_needs` — `pydantic-ai-ui-security-review` gated `activation` on
  `needs.detect.outputs.touched` while `detect` was not one of its `needs`. The
  expression silently evaluated to empty, so the whole chain skipped for a month.
  A job skipped by `if:` reports *success*, so the required check stayed green
  while the security review never ran.
- `safe_output_job_max` — `record-attention-decision` relied on the default
  `max: 1`, so 9 of every 10 classifications were dropped into an `errors` array
  nobody read and the host script then failed the run.
- `prompt_paths` — the review prompt told the agent to read
  `/tmp/gh-aw/.review-context/`, outside the `Read` tool's root. Every run wasted
  ~15 tool calls rediscovering its own context via `bash cat`.
- `timeout_declared` — a sweep silently needs a wall-clock bound; `roundtrip-sweep`
  spent three weeks timing out one minute under its limit, filing nothing.
- `job_timeout_env` — the shim reads its wall-clock budget from
  `PYDANTIC_AI_JOB_TIMEOUT_MINUTES`, so every workflow must declare it and keep it
  equal to `timeout-minutes`. Absent, the shim assumed 30: `stale-issues-finder`
  asked for 60 and only ever used 28, and `attention-triage` asked for 20 and was
  killed before it could emit anything.
- `compiler_versions` — a partial `gh aw compile` leaves locks on mixed compiler
  versions, which is how source and lock drift apart unnoticed.
- `compiled_runner_contract` — every generated agent job must keep the command-line
  contract required by the Pydantic AI runner shim.
- `awf_binary_version` — `engine.command` makes gh-aw skip its own AWF install, so
  `shared/pre-steps.md` re-runs the installer with a hand-written version. #8041 moved
  gh-aw to a release bundling AWF v0.27.44 without touching that pin, and the v0.27.42
  binary rejected the v0.27.44 config (`config.apiProxy.providers is not supported`).
  The agent job died before the model started, on every workflow using the shim.
- `compiler_version_compatibility` — the compiler version in every generated lock
  must not appear in gh-aw's live blocked-version policy.
- `lock_regenerated` — `.github/workflows/AGENTS.md` requires a recompiled
  `*.lock.yml` in the same change as its `*.md` source. GitHub Actions runs the
  lock, so an un-recompiled source is a silent no-op.

Run over the repo with `python .github/scripts/agentic_workflow_guard.py check`,
or as policy tests via `test_agentic_workflow_guard.py`.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shlex
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOWS_DIR = Path('.github/workflows')
AGENTIC_GLOB = 'pydantic-ai-*.md'

# gh-aw stages the agent's workspace no-exec and roots its file tools at the
# checkout, so absolute paths under this prefix are unreadable by `Read`.
SANDBOX_PREFIX = '/tmp/gh-aw/'

# Paths the agent is told about but reads through `Bash`/`jq` rather than the file
# tools, which is fine — `Bash` is not rooted at the checkout. `/tmp/gh-aw/bin` is
# gh-aw's exec-able launcher path; the rest is the prefetched GitHub corpus that
# `shared/tool-hints.md` documents with `jq` invocations (#6509).
#
# These stay outside `$GITHUB_WORKSPACE` while PR review context was moved into it
# (#6766 F3), so an agent that reaches for `Read` here still hits the old failure.
# Worth revisiting if a sweep is seen burning turns on them.
# Every entry ends in `/`, including the one naming a single file, and the candidate
# path gets one appended before comparison. That makes the match land on a whole path
# component in both directions: without it `bin` also allows `bindings/`, and
# `open-issues.tsv` also allows `open-issues.tsv.bak`.
PROMPT_PATH_ALLOWLIST_PREFIXES = (
    '/tmp/gh-aw/bin/',
    '/tmp/gh-aw/agent/github-context/',
    '/tmp/gh-aw/agent/open-issues.tsv/',
    '/tmp/gh-aw/agent/issues/',
)

# A fenced block opens with three or more backticks or tildes, indented at most three
# spaces, optionally followed by an info string. `~~~` and the indent allowance are both
# CommonMark and both appear in the wild. The trailing group is the info string, which
# only an *opener* may carry — see `check_prompt_paths`.
FENCE_MARKER = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')

# Mirrors `pydantic_ai_gh_aw_shim.cli`, which this script deliberately does not import:
# the shim pulls in the whole agent runtime, and the guard must stay a lightweight
# stdlib+yaml script. `test_job_timeout_constants_match_the_shim` pins them together.
DEFAULT_JOB_TIMEOUT_MINS = 30
JOB_TIMEOUT_HEADROOM_MINS = 2

# Run to the end of the path token rather than to the end of a filename-ish character
# class. A class that stops early truncates `/tmp/gh-aw/bin:secret/x` to `/tmp/gh-aw/bin`,
# which then matches the allowlist and lets the real path through unflagged. Stopping at
# whitespace, backtick, quote and the markdown-link delimiters is enough to end the token
# wherever a prompt actually writes one.
SANDBOX_PATH = re.compile(rf'{re.escape(SANDBOX_PREFIX)}[^\s`\'"()\[\]<>]*')

# The version argument of the hand-written AWF install step. Both the script path and
# the version may be quoted, and the capture must not swallow those quotes: a pin that
# only differs from the compiled version by a pair of them is the same pin.
AWF_INSTALL_VERSION = re.compile(r'install_awf_binary\.sh"?\s+"?([^"\s]+)')

NEEDS_REFERENCE = re.compile(r'\bneeds\.([A-Za-z_][A-Za-z0-9_-]*)')
EXPRESSION_BLOCK = re.compile(r'\$\{\{(.*?)\}\}', re.DOTALL)


@dataclass(frozen=True)
class Violation:
    """A single policy failure, rendered as one line of CI output."""

    path: str
    check: str
    message: str

    def __str__(self) -> str:
        return f'{self.path}: [{self.check}] {self.message}'


def _as_mapping(value: object) -> dict[str, Any]:
    """Coerce a parsed-YAML value to a string-keyed mapping.

    YAML 1.1 parses a bare `on:` key as the boolean `True`, so keys are stringified
    with that one special case mapped back.
    """
    if not isinstance(value, dict):
        return {}
    mapping = cast(dict[Any, Any], value)
    return {('on' if key is True else str(key)): item for key, item in mapping.items()}


def _as_strings(value: object) -> set[str]:
    """Coerce a parsed-YAML scalar-or-sequence to a set of strings."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in cast(list[Any], value)}
    return set()


def parse_frontmatter(source: Path) -> dict[str, Any]:
    """Return the YAML frontmatter of a gh-aw workflow source, or `{}` if absent."""
    text = source.read_text(encoding='utf-8')
    match = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
    if match is None:
        return {}
    return _as_mapping(yaml.safe_load(match.group(1)))


def parse_prompt_body(source: Path) -> str:
    """Return the markdown body (the agent-facing prompt) of a gh-aw source."""
    text = source.read_text(encoding='utf-8')
    match = re.match(r'^---\n.*?\n---\n(.*)$', text, re.DOTALL)
    return match.group(1) if match else text


def _expressions_of(job: dict[str, Any]) -> list[tuple[str, str]]:
    """Return `(field, expression)` pairs whose `needs.*` refs must resolve.

    Every one of these silently evaluates to empty when it references a job that is not
    a dependency, rather than failing loudly. `if:` skips the job or step outright — and
    a job skipped by `if:` reports success. `outputs:`, `env:` and `with:` are worse in
    kind: nothing skips, the step just runs with an empty value, so a wrong action call
    or a wrong shell variable goes through looking healthy. gh-aw emits exactly that
    shape, e.g. `GH_AW_NEEDS_DETECT_OUTPUTS_TOUCHED: ${{ needs.detect.outputs.touched }}`.
    """
    expressions: list[tuple[str, str]] = []

    def add(field: str, value: Any) -> None:
        if isinstance(value, str):
            expressions.append((field, value))

    add('if', job.get('if'))
    for name, value in _as_mapping(job.get('outputs')).items():
        add(f'outputs.{name}', value)
    for name, value in _as_mapping(job.get('env')).items():
        add(f'env.{name}', value)
    for name, value in _as_mapping(job.get('with')).items():
        add(f'with.{name}', value)

    steps = job.get('steps')
    for index, raw_step in enumerate(cast(list[Any], steps) if isinstance(steps, list) else []):
        step = _as_mapping(raw_step)
        label = str(step.get('name') or step.get('uses') or f'#{index}')
        add(f'steps[{label}].if', step.get('if'))
        add(f'steps[{label}].run', step.get('run'))
        for name, value in _as_mapping(step.get('env')).items():
            add(f'steps[{label}].env.{name}', value)
        for name, value in _as_mapping(step.get('with')).items():
            add(f'steps[{label}].with.{name}', value)
    return expressions


def check_dangling_needs(lock: Path) -> list[Violation]:
    """Every `needs.<job>` reference must name a declared dependency of that job."""
    workflow = _as_mapping(yaml.safe_load(lock.read_text(encoding='utf-8')))

    violations: list[Violation] = []
    for job_name, raw_job in _as_mapping(workflow.get('jobs')).items():
        job = _as_mapping(raw_job)
        declared = _as_strings(job.get('needs'))
        for field, expression in _expressions_of(job):
            # `if:` is the one field GitHub evaluates as an expression without `${{ }}`.
            # Everywhere else a bare `needs.foo` is literal text — most often inside a
            # `run:` shell script — so only interpolated blocks count, or the check would
            # flag prose and jq programs that merely spell the word.
            if field == 'if' or field.endswith('.if'):
                scanned = expression
            else:
                scanned = ' '.join(EXPRESSION_BLOCK.findall(expression))
            for referenced in sorted(set(NEEDS_REFERENCE.findall(scanned))):
                if referenced not in declared:
                    violations.append(
                        Violation(
                            str(lock),
                            'dangling-needs',
                            f'job `{job_name}` references `needs.{referenced}` in `{field}:` but '
                            f'`{referenced}` is not in its `needs:` ({sorted(declared) or "none"}). '
                            'The expression evaluates to empty and the job silently skips — '
                            'and a job skipped by `if:` reports success.',
                        )
                    )
    return violations


def check_safe_output_job_max(source: Path) -> list[Violation]:
    """Custom `safe-outputs.jobs.*` entries must declare `max:` explicitly."""
    safe_outputs = _as_mapping(parse_frontmatter(source).get('safe-outputs'))
    return [
        Violation(
            str(source),
            'safe-output-job-max',
            f'safe-output job `{job_name}` does not declare `max:`. The default is 1, and extra '
            'items are dropped into an `errors` array that nothing reads — set it explicitly '
            'even when 1 is correct, so the bound is a decision rather than an accident.',
        )
        for job_name, job in _as_mapping(safe_outputs.get('jobs')).items()
        if 'max' not in _as_mapping(job)
    ]


def check_prompt_paths(source: Path) -> list[Violation]:
    """Prompts must not instruct the agent to read paths outside the workspace.

    Only prose is scanned. A path inside a fenced code block is being handed to a
    shell, and `Bash` can read anywhere — the F3 defect was prose telling the agent to
    `Read` a path its *file tools* reject. Flagging shell snippets too would condemn
    the documented `jq /tmp/gh-aw/agent/github-context/...` corpus reads, which work.
    """
    violations: list[Violation] = []
    fence: str | None = None
    for lineno, line in enumerate(parse_prompt_body(source).splitlines(), start=1):
        marker = FENCE_MARKER.match(line)
        if marker is not None:
            run, info = marker.group(1), marker.group(2)
            # CommonMark: a closer uses the same character, is at least as long as its
            # opener, and carries no info string. Getting any of those wrong leaves the
            # scanner out of step with the real block structure — and once it believes it
            # is inside a fence it skips every later line, turning a real violation into a
            # silent PASS. A ```bash line inside an open block is content, not a closer.
            if fence is None:
                fence = run
                continue
            if run[0] == fence[0] and len(run) >= len(fence) and not info.strip():
                fence = None
                continue
        if fence is not None:
            continue
        for match in SANDBOX_PATH.finditer(line):
            # Trailing sentence punctuation belongs to the prose, not the path. Stripped
            # after matching rather than excluded from the class, so that `:` and `.`
            # inside a path still keep the token whole.
            path = match.group(0).rstrip('.,;:!?')
            # Normalise for the comparison only, never for the message: the author needs
            # to see the path as they wrote it to find it again. `/tmp/gh-aw/bin/../` is
            # textually under an allowlisted directory but resolves outside it, which is
            # the very class of path this rejects. Compare with a trailing separator too,
            # so a directory entry matches the directory itself and everything under it
            # but not a sibling sharing its opening characters — bare `bin` would
            # otherwise allow `/tmp/gh-aw/bindings/`.
            if f'{posixpath.normpath(path)}/'.startswith(PROMPT_PATH_ALLOWLIST_PREFIXES):
                continue
            violations.append(
                Violation(
                    str(source),
                    'prompt-path-outside-workspace',
                    f"line {lineno}: prompt references `{path}`, which is outside the agent's file-tool "
                    'root — `Read` rejects it with "resolves outside the root directory" and the agent '
                    'burns turns rediscovering the file. Stage it under `$GITHUB_WORKSPACE` instead.',
                )
            )
    return violations


def check_timeout_declared(source: Path) -> list[Violation]:
    """Agentic workflows must bound their own wall clock."""
    if 'timeout-minutes' in parse_frontmatter(source):
        return []
    return [
        Violation(
            str(source),
            'timeout-declared',
            'no `timeout-minutes:` in frontmatter. Without an explicit bound an agent can spend a '
            'full run and be killed with nothing to show for it.',
        )
    ]


def check_job_timeout_env(source: Path) -> list[Violation]:
    """`PYDANTIC_AI_JOB_TIMEOUT_MINUTES`, when set, must equal a usable `timeout-minutes`.

    The shim derives the agent's own wall-clock budget from that env var, because
    gh-aw's `GH_AW_TIMEOUT_MINUTES` is set only on the failure-handler step and never
    reaches the agent container. If the two drift, the agent either stops early and
    wastes the time it was granted, or overruns and is killed with nothing emitted.
    """
    frontmatter = parse_frontmatter(source)
    timeout = frontmatter.get('timeout-minutes')
    if timeout is None:
        # `timeout-declared` already reports this, and every message below would quote
        # `None` as the value to copy into `env:`.
        return []
    if isinstance(timeout, int) and timeout <= JOB_TIMEOUT_HEADROOM_MINS:
        return [
            Violation(
                str(source),
                'job-timeout-too-short',
                f'`timeout-minutes: {timeout}` leaves the agent no budget. The shim reserves '
                f'{JOB_TIMEOUT_HEADROOM_MINS} minutes for container teardown and artifact upload, so it '
                f'falls back to {DEFAULT_JOB_TIMEOUT_MINS} minutes rather than granting a non-positive '
                'one — the agent then runs well past the point Actions kills the job, and emits nothing. '
                f'Give the job more than {JOB_TIMEOUT_HEADROOM_MINS} minutes.',
            )
        ]
    declared = _as_mapping(frontmatter.get('env')).get('PYDANTIC_AI_JOB_TIMEOUT_MINUTES')
    if declared is None:
        return [
            Violation(
                str(source),
                'job-timeout-env-missing',
                'no `PYDANTIC_AI_JOB_TIMEOUT_MINUTES` in `env:`. Without it the shim falls back to a '
                f'30-minute assumption, so a job with `timeout-minutes: {timeout}` either overruns and '
                'is killed with nothing emitted, or silently never uses the time it was granted. '
                f'Set it to `"{timeout}"`.',
            )
        ]
    if str(declared) == str(timeout):
        return []
    return [
        Violation(
            str(source),
            'job-timeout-env-mismatch',
            f'`PYDANTIC_AI_JOB_TIMEOUT_MINUTES` is `{declared}` but `timeout-minutes` is `{timeout}`. '
            "The shim derives the agent's budget from the env var, so a mismatch either wastes the "
            'granted time or gets the agent killed mid-run. Keep them equal.',
        )
    ]


def check_compiler_versions(locks: list[Path]) -> list[Violation]:
    """All locks must be compiled by the same gh-aw version."""
    versions: dict[str, list[str]] = {}
    for lock in locks:
        first_line = lock.read_text(encoding='utf-8').split('\n', 1)[0]
        _, _, payload = first_line.partition('gh-aw-metadata: ')
        if not payload:
            continue
        try:
            # `_as_mapping`, not `.get` directly: a payload that parses as valid JSON but
            # isn't an object (a bare string, `null`) would otherwise raise `AttributeError`
            # and take down the whole lint job instead of reporting a policy violation.
            version = str(_as_mapping(json.loads(payload)).get('compiler_version', 'unknown'))
        except json.JSONDecodeError:
            continue
        versions.setdefault(version, []).append(lock.name)

    if len(versions) <= 1:
        return []
    summary = '; '.join(f'{version}: {", ".join(sorted(names))}' for version, names in sorted(versions.items()))
    return [
        Violation(
            str(WORKFLOWS_DIR),
            'compiler-version-drift',
            f'locks were built by different gh-aw versions ({summary}). Run `gh aw compile` so every '
            'lock is regenerated together.',
        )
    ]


def check_compiled_runner_contract(lock: Path) -> list[Violation]:
    """The generated agent command must retain the interface our runner implements."""
    workflow = _as_mapping(yaml.safe_load(lock.read_text(encoding='utf-8')))
    agent = _as_mapping(_as_mapping(workflow.get('jobs')).get('agent'))
    steps = agent.get('steps')
    commands: list[list[str]] = []
    for raw_step in cast(list[Any], steps) if isinstance(steps, list) else []:
        run = _as_mapping(raw_step).get('run')
        if not isinstance(run, str) or 'pydantic-ai-runner-launch' not in run:
            continue
        with suppress(ValueError):
            outer_arguments = shlex.split(run)
            for index, argument in enumerate(outer_arguments[:-1]):
                if argument != '-c' or 'pydantic-ai-runner-launch' not in outer_arguments[index + 1]:
                    continue
                with suppress(ValueError):
                    inner_arguments = shlex.split(outer_arguments[index + 1])
                    for runner_index, inner_argument in enumerate(inner_arguments):
                        if inner_argument.endswith('/pydantic-ai-runner-launch'):
                            commands.append(inner_arguments[runner_index:])
    if len(commands) != 1:
        return [
            Violation(
                str(lock),
                'compiled-runner-contract',
                'the `agent` job must contain exactly one invocation of '
                '`pydantic-ai-runner-launch`. The custom runner is the compatibility '
                'boundary between gh-aw and Pydantic AI.',
            )
        ]

    command = commands[0]
    missing: list[str] = []
    if '--output-format' not in command or command.index('--output-format') == len(command) - 1:
        missing.append('`--output-format stream-json`')
    elif command[command.index('--output-format') + 1] != 'stream-json':
        missing.append('`--output-format stream-json`')
    if '--mcp-config' not in command or command.index('--mcp-config') == len(command) - 1:
        missing.append('`--mcp-config`')
    if '--allowed-tools' not in command or command.index('--allowed-tools') == len(command) - 1:
        missing.append('`mcp__safeoutputs` in `--allowed-tools`')
    elif 'mcp__safeoutputs' not in command[command.index('--allowed-tools') + 1].split(','):
        missing.append('`mcp__safeoutputs` in `--allowed-tools`')
    if not missing:
        return []
    return [
        Violation(
            str(lock),
            'compiled-runner-contract',
            f'the generated Pydantic AI runner invocation is missing {", ".join(missing)}. '
            'Recompiling changed the gh-aw interface that the custom runner implements.',
        )
    ]


def check_awf_binary_version(lock: Path) -> list[Violation]:
    """The hand-pinned AWF binary must be the one gh-aw compiled the lock against.

    `GH_AW_INFO_AWF_VERSION` is what gh-aw bundles and what it writes the firewall
    config against; the install step's argument is maintained by hand because
    `engine.command` makes gh-aw skip its own install. A gh-aw upgrade moves the
    first and leaves the second, and the older binary then refuses the newer config.

    A lock that pins a version but declares none also fails: the comparison has lost
    its reference, and passing there would disarm the check on a gh-aw rename.
    """
    workflow = _as_mapping(yaml.safe_load(lock.read_text(encoding='utf-8')))
    pinned: set[str] = set()
    bundled: set[str] = set()
    for raw_job in _as_mapping(workflow.get('jobs')).values():
        steps = _as_mapping(raw_job).get('steps')
        for raw_step in cast(list[Any], steps) if isinstance(steps, list) else []:
            step = _as_mapping(raw_step)
            run = step.get('run')
            # `finditer`, not `search`: one `run:` block can invoke the installer more than
            # once, and the last call is the binary the agent ends up with. Every invocation
            # has to match, so stopping at the first would let a stale later one through.
            if isinstance(run, str):
                pinned.update(match.group(1) for match in AWF_INSTALL_VERSION.finditer(run))
            version = _as_mapping(step.get('env')).get('GH_AW_INFO_AWF_VERSION')
            if isinstance(version, str):
                bundled.add(version)

    # No install step means this lock does not use the shim, so there is no hand-written
    # pin to drift.
    if not pinned:
        return []
    # A pin with nothing to compare it against is the check losing its reference, not a
    # pass. Returning empty here would let a gh-aw rename of `GH_AW_INFO_AWF_VERSION`
    # disarm this check silently, and the next skew would ship green exactly as #8041 did.
    if not bundled:
        return [
            Violation(
                str(lock),
                'awf-binary-version',
                f'the AWF install step pins {", ".join(sorted(pinned))} but the lock declares no '
                '`GH_AW_INFO_AWF_VERSION` to check it against. gh-aw stopped reporting the AWF '
                'version it bundles, so this check needs a new source before the pin can be trusted.',
            )
        ]
    unexpected = sorted(pinned - bundled)
    if not unexpected:
        return []
    return [
        Violation(
            str(lock),
            'awf-binary-version',
            f'the AWF install step pins {", ".join(unexpected)} but gh-aw compiled this lock against '
            f'{", ".join(sorted(bundled))}. Update the version in the `pre-steps` that re-run '
            '`install_awf_binary.sh` and recompile; an older binary rejects the newer firewall config.',
        )
    ]


def check_compiler_version_compatibility(locks: list[Path], compatibility: dict[str, Any]) -> list[Violation]:
    """Compiled gh-aw versions must not be revoked by the upstream runtime policy."""
    blocked_raw = compatibility.get('blockedVersions')
    blocked = _as_strings(blocked_raw)
    if not isinstance(blocked_raw, list):
        return [
            Violation(
                'gh-aw compatibility policy',
                'compiler-compatibility-policy-invalid',
                '`blockedVersions` is missing or is not a list; refusing to pass without '
                'the policy enforced by gh-aw activation jobs.',
            )
        ]
    violations: list[Violation] = []
    for lock in locks:
        first_line = lock.read_text(encoding='utf-8').split('\n', 1)[0]
        _, _, payload = first_line.partition('gh-aw-metadata: ')
        if not payload:
            continue
        try:
            version = _as_mapping(json.loads(payload)).get('compiler_version')
        except json.JSONDecodeError:
            continue
        if isinstance(version, str) and version in blocked:
            violations.append(
                Violation(
                    str(lock),
                    'compiler-version-blocked',
                    f'compiler version `{version}` is blocked by gh-aw and the activation job will fail. '
                    'Upgrade gh-aw and recompile every agentic workflow.',
                )
            )
    return violations


def check_lock_regenerated(changed: list[str], workflows_dir: Path = WORKFLOWS_DIR) -> list[Violation]:
    """A changed `.md` source (or shared import) must ship its recompiled lock.

    Actions runs the `.lock.yml`, never the `.md`, so a source edit without its
    recompiled lock is a silent no-op that leaves the two drifted apart.
    """
    changed_set = set(changed)
    violations: list[Violation] = []

    # Union of sources still on disk and sources named in the changeset: a deleted `.md`
    # is gone from the glob, so globbing alone would let its orphaned lock through and
    # Actions would keep running a workflow whose source no longer exists.
    changed_sources = {
        path for path in changed_set if Path(path).match(str(workflows_dir / AGENTIC_GLOB)) and '/shared/' not in path
    }
    sources = {str(path) for path in workflows_dir.glob(AGENTIC_GLOB)} | changed_sources

    for source_path in sorted(sources):
        source = Path(source_path)
        lock = source.with_suffix('.lock.yml')
        if source_path not in changed_set or str(lock) in changed_set:
            continue
        if source.exists():
            violations.append(
                Violation(
                    source_path,
                    'lock-not-regenerated',
                    f'source changed but `{lock.name}` did not. Run `gh aw compile` and commit the '
                    'regenerated lock in the same change.',
                )
            )
        else:
            violations.append(
                Violation(
                    source_path,
                    'lock-not-regenerated',
                    f'source was deleted but `{lock.name}` was left behind. Actions runs the lock, so '
                    'the workflow would keep running with no source. Delete the lock in the same change.',
                )
            )

    changed_shared = {path for path in changed_set if path.startswith(str(workflows_dir / 'shared'))}
    for shared in sorted(changed_shared):
        name = Path(shared).name
        for source in sorted(workflows_dir.glob(AGENTIC_GLOB)):
            lock = source.with_suffix('.lock.yml')
            imports = _as_strings(parse_frontmatter(source).get('imports'))
            if f'shared/{name}' in imports and str(lock) not in changed_set:
                violations.append(
                    Violation(
                        shared,
                        'lock-not-regenerated',
                        f'shared import changed but `{lock.name}`, which imports it, was not '
                        'regenerated. Run `gh aw compile` and commit every affected lock.',
                    )
                )
    return violations


def changed_files(base_ref: str) -> list[str]:
    """Return paths changed relative to `base_ref` (empty if git can't resolve it)."""
    try:
        completed = subprocess.run(
            ['git', 'diff', '--name-only', f'{base_ref}...HEAD'],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [line for line in completed.stdout.splitlines() if line]


def run_checks(
    workflows_dir: Path = WORKFLOWS_DIR,
    changed: list[str] | None = None,
    compatibility: dict[str, Any] | None = None,
) -> list[Violation]:
    """Run every static check; `changed` enables the lock-freshness check."""
    sources = sorted(workflows_dir.glob(AGENTIC_GLOB))
    locks = sorted(workflows_dir.glob('*.lock.yml'))
    # Resolve `shared/` under the caller's `workflows_dir`, not the module global, so a
    # custom root scans its own fragments rather than whatever sits under the cwd.
    # `rglob`, not `glob`: the F3 defect this scan exists to catch lived in
    # `shared/prompts/pydantic-ai-pr-review.md`. Those prompts are fetched at run time
    # rather than compiled in, so they change with no lock diff and are the least
    # reviewed files here — the last ones a path check should skip.
    shared_dir = workflows_dir / 'shared'
    shared = sorted(shared_dir.rglob('*.md')) if shared_dir.is_dir() else []

    violations: list[Violation] = []
    for lock in locks:
        violations += check_dangling_needs(lock)
        violations += check_compiled_runner_contract(lock)
        violations += check_awf_binary_version(lock)
    for source in sources:
        violations += check_safe_output_job_max(source)
        violations += check_timeout_declared(source)
        violations += check_job_timeout_env(source)
    for markdown in [*sources, *shared]:
        violations += check_prompt_paths(markdown)
    violations += check_compiler_versions(locks)
    if compatibility is not None:
        violations += check_compiler_version_compatibility(locks, compatibility)
    if changed:
        violations += check_lock_regenerated(changed, workflows_dir)
    return violations


def main(argv: list[str] | None = None) -> int:
    """Print every violation to stderr and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check'])
    parser.add_argument('--base-ref', default='', help='git ref to diff against for lock-freshness')
    parser.add_argument(
        '--changed-file-list',
        default='',
        help=(
            'file holding one changed path per line, for lock-freshness. Prefer this in CI, where the '
            'checkout is shallow and a `git diff` range cannot resolve: feed it `gh pr view --json files`.'
        ),
    )
    parser.add_argument(
        '--compatibility-file',
        type=Path,
        help='gh-aw compatibility policy downloaded from `.github/aw/compat.json` in github/gh-aw',
    )
    args = parser.parse_args(argv)

    if args.changed_file_list:
        # `splitlines`, not `split`: the producer writes one path per line, and a path
        # containing a space would otherwise become two paths that match nothing — the
        # real file's lock freshness then goes unchecked while the run still passes.
        text = Path(args.changed_file_list).read_text(encoding='utf-8')
        changed = [line for line in text.splitlines() if line.strip()]
    elif args.base_ref:
        changed = changed_files(args.base_ref)
    else:
        changed = None

    if args.compatibility_file is None:
        violations = run_checks(changed=changed)
    else:
        try:
            compatibility = _as_mapping(json.loads(args.compatibility_file.read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError) as error:
            print(f'{args.compatibility_file}: [compiler-compatibility-policy-invalid] {error}', file=sys.stderr)
            return 1
        violations = run_checks(changed=changed, compatibility=compatibility)
    for violation in violations:
        print(violation, file=sys.stderr)
    if violations:
        print(f'\n{len(violations)} agentic-workflow policy violation(s).', file=sys.stderr)
        return 1
    print('Agentic workflow policy checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
