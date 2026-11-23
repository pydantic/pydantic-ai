"""Check the shell and the trust rules of every workflow gh-aw does not generate.

actionlint (`.pre-commit-config.yaml`) checks events, expressions and job graphs, and `agentic_workflow_guard.py` checks
the gh-aw locks. This checks what neither sees, each rule a defect that shipped:

- `shell-default`: a workflow without `defaults.run.shell: bash`. Without it, Linux steps run `bash -e` without
  `pipefail`, so a failed `gh` on the left of a pipe reads as an empty result.
- `error-as-data`: `$(gh ... || echo ...)` or the same around `curl`. `gh api` prints an HTTP error's body to stdout,
  so the substitution captures that body, plus the fallback, as data. On
  https://github.com/pydantic/pydantic-ai/pull/9373 a rate-limited role lookup read as no role and PR Guard closed a
  maintainer's PR.
- `pipeline-condition`: an `if`, `elif`, `while` or `until` whose pipeline starts with `gh` or `curl`. The condition
  reads the last command's status, so a failed call reads as no match: until
  https://github.com/pydantic/pydantic-ai/pull/9281 the gh-aw pin reminder went on as if no issue asked for the bump.
- `credential-less-actor`: a job that reads a secret on `pull_request`, or in a `credential_dependent` workflow of
  `.github/trust.toml`, and does not exclude each actor class that runs without secrets.
  https://github.com/pydantic/pydantic-ai/pull/10213 added the Dependabot exclusion `Agent Provider Health` missed.
- `bot-registry`: a `<login>[bot]` that `.github/trust.toml` does not list, or a bot allowlist that differs from its
  role there. PR Guard knew pydanty but not montybot until https://github.com/pydantic/pydantic-ai/pull/13242.
- `write-wildcard`: an agent's `--allowedTools` entry that runs a write command on any target, such as
  `Bash(gh pr comment:*)`, without an `unbound_write_tools` row. https://github.com/pydantic/pydantic-ai/pull/12357
  bound the fork job's comments to its own PR.
- `unpinned-image`: a container image without a digest, so the image a job runs can change without a commit.
  https://github.com/pydantic/pydantic-ai/pull/10216 is one of several fixes that moved a service image between
  registries.

Usage:
    uv run python .github/scripts/workflow_policy.py [ROOT] [--trust TRUST_TOML]
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter
from typing_extensions import TypeAliasType

# gh-aw generates these; `agentic_workflow_guard.py` checks them.
GENERATED = re.compile(r'(\.lock\.yml|^agentics-maintenance\.yml)$')

# Commands that change something on GitHub. An `--allowedTools` entry covers one when its command is the write command
# or a prefix of it, so `Bash(gh pr:*)` covers `gh pr comment`.
WRITE_COMMANDS = (
    'gh api',
    'gh issue close',
    'gh issue comment',
    'gh issue create',
    'gh issue edit',
    'gh label',
    'gh pr close',
    'gh pr comment',
    'gh pr create',
    'gh pr edit',
    'gh pr merge',
    'gh pr review',
    'gh release',
    'git push',
)

# `yaml.BaseLoader` reads every scalar as a string.
Node = TypeAliasType('Node', 'str | list[Node] | dict[str, Node]')
_NODE: TypeAdapter[Node] = TypeAdapter(Node)
Role: TypeAlias = Literal['claude-trigger', 'team-app']


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class _ActorClass(_Strict):
    conditions: list[str]


class _PullRequest(_Strict):
    never_runs_on: list[str]


class _CredentialDependent(_Strict):
    workflow: str
    reason: str


class _Bot(_Strict):
    roles: list[Role]


class _UnboundWriteTool(_Strict):
    workflow: str
    job: str
    tool: str
    reason: str


class Trust(_Strict):
    """`.github/trust.toml`."""

    credential_less: dict[str, _ActorClass]
    pull_request: _PullRequest
    credential_dependent: list[_CredentialDependent]
    bots: dict[str, _Bot]
    unbound_write_tools: list[_UnboundWriteTool]


@dataclass(frozen=True)
class Finding:
    """One rule a workflow breaks."""

    path: str
    rule: str
    message: str

    def __str__(self) -> str:
        return f'{self.path}: {self.rule}: {self.message}'


@dataclass(frozen=True)
class Workflow:
    """A workflow file and its parsed YAML, every scalar a string."""

    path: Path
    text: str
    data: dict[str, Node]

    @classmethod
    def parse(cls, path: Path, text: str) -> Workflow:
        return cls(path, text, _mapping(_NODE.validate_python(yaml.load(text, Loader=yaml.BaseLoader))))

    @property
    def jobs(self) -> dict[str, dict[str, Node]]:
        return {name: _mapping(job) for name, job in _mapping(self.data.get('jobs')).items()}


def _mapping(node: Node | None) -> dict[str, Node]:
    return node if isinstance(node, dict) else {}


def _list(node: Node | None) -> list[Node]:
    if isinstance(node, list):
        return node
    return [] if node is None or node == '' else [node]


def _strings(node: Node) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    else:
        for item in node.values() if isinstance(node, dict) else node:
            yield from _strings(item)


def _steps(job: dict[str, Node]) -> list[dict[str, Node]]:
    return [_mapping(step) for step in _list(job.get('steps'))]


def _normalized(text: str) -> str:
    return ' '.join(text.split())


def _excerpt(text: str) -> str:
    text = _normalized(text)
    return text if len(text) <= 80 else f'{text[:77]}...'


# A heredoc opener, not a `<<<` here-string. Its body is data, not shell.
_HEREDOC = re.compile(r'(?<!<)<<-?(?!<)\s*[\'"]?(\w+)[\'"]?')
_QUOTED = re.compile(r'\'[^\']*\'|"(?:\\.|[^"\\])*"')


def _shell_lines(script: str) -> list[str]:
    """The lines of a `run:` script that hold shell: continuations joined, comment lines and heredoc bodies dropped."""
    lines: list[str] = []
    heredoc: str | None = None
    for line in re.sub(r'\\\n[ \t]*', ' ', script).splitlines():
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
        elif not line.lstrip().startswith('#'):
            lines.append(line)
            if match := _HEREDOC.search(_QUOTED.sub("''", line)):
                heredoc = match.group(1)
    return lines


def _substitutions(script: str) -> list[str]:
    """The body of every `$(...)` in `script`, nested ones included, with single-quoted text emptied.

    A single-quoted string outside double quotes is literal: a `$(` in it starts nothing, and a jq program's
    parentheses in it do not count.
    """
    code = _QUOTED.sub(lambda match: match.group() if match.group().startswith('"') else "''", script)
    bodies: list[str] = []
    for start in (match.end() for match in re.finditer(r'\$\((?!\()', code)):
        depth, index = 1, start
        while index < len(code) and depth:
            depth += {'(': 1, ')': -1}.get(code[index], 0)
            index += 1
        bodies.append(code[start : index - 1])
    return bodies


def _run_scripts(workflow: Workflow) -> Iterator[tuple[str, str, str]]:
    """Each job's `run:` scripts, as the job name, step name and the script's shell lines joined."""
    for job_name, job in workflow.jobs.items():
        for index, step in enumerate(_steps(job)):
            run = step.get('run')
            if isinstance(run, str):
                name = step.get('name')
                yield job_name, name if isinstance(name, str) else f'step {index + 1}', '\n'.join(_shell_lines(run))


def check_shell_default(workflow: Workflow) -> list[Finding]:
    """A workflow must run its steps under `bash`, which adds `pipefail`, unless a step or job says otherwise."""
    shell = _mapping(_mapping(workflow.data.get('defaults')).get('run')).get('shell')
    if shell == 'bash':
        return []
    return [Finding(workflow.path.name, 'shell-default', 'set `defaults: run: shell: bash` at the workflow level')]


def check_error_as_data(workflow: Workflow) -> list[Finding]:
    """No `$(gh ... || echo ...)` or `$(curl ... || echo ...)`: the substitution captures the error body as data."""
    findings: list[Finding] = []
    for job, step, script in _run_scripts(workflow):
        for body in _substitutions(script):
            code = _QUOTED.sub("''", body)
            if re.match(r'\s*(gh|curl)\s', code) and re.search(r'\|\|\s*(echo|printf|true|:)\b', code):
                findings.append(
                    Finding(
                        workflow.path.name,
                        'error-as-data',
                        f'{job} / {step}: `$({_excerpt(body)})` captures the error output; '
                        "test the command's status outside the substitution, as `repo_role` in `pr-guard.yml` does",
                    )
                )
    return findings


_CONDITION = re.compile(r'(?:^|[;&]|\b(?:then|do|else)\b)\s*(?:if|elif|while|until)\s+(?:!\s+)?(gh|curl)\s([^;]*)')


def check_pipeline_condition(workflow: Workflow) -> list[Finding]:
    """No condition whose pipeline starts with `gh` or `curl`: the last command's status hides a failed call."""
    findings: list[Finding] = []
    for job, step, script in _run_scripts(workflow):
        for line in script.splitlines():
            for match in _CONDITION.finditer(_QUOTED.sub("''", line)):
                if re.search(r'(?<!\|)\|(?!\|)', match.group(2)):
                    findings.append(
                        Finding(
                            workflow.path.name,
                            'pipeline-condition',
                            f'{job} / {step}: a failed `{match.group(1)}` reads as no match in `{_excerpt(line)}`; '
                            'capture its output first, then test that',
                        )
                    )
    return findings


_SECRET = re.compile(r'\bsecrets\.(?!GITHUB_TOKEN\b)(\w+)|\bsecrets\[')
# Status functions that let a job run when a job it needs did not succeed.
_RELAXED = re.compile(r'\b(always|cancelled|failure)\(\)')


def _condition_chain(jobs: dict[str, dict[str, Node]], name: str) -> str:
    """The `if:` of a job and of each job it needs to have run, normalized and joined.

    Without a status function, a job runs only when every job it needs succeeded. With `always()`, `cancelled()` or
    `failure()`, it runs regardless, unless its `if:` reads a needed job's success or outputs.
    """
    job = jobs.get(name, {})
    condition = job.get('if', '')
    condition = _normalized(condition) if isinstance(condition, str) else ''
    needs = [need for need in _list(job.get('needs')) if isinstance(need, str)]
    if _RELAXED.search(condition):
        needs = [
            need
            for need in needs
            if re.search(rf'needs\.{re.escape(need)}\.(result\s*==\s*.success.|outputs\.)', condition)
        ]
    return ' && '.join([condition, *(_condition_chain(jobs, need) for need in needs)])


def check_credential_less_actor(workflow: Workflow, trust: Trust) -> list[Finding]:
    """A job that reads a secret where a credential-less actor can trigger it excludes every such actor class."""
    on = workflow.data.get('on', '')
    dependent = any(row.workflow == workflow.path.name for row in trust.credential_dependent)
    if not dependent and 'pull_request' not in (on if isinstance(on, dict | list) else [on]):
        return []
    findings: list[Finding] = []
    jobs = workflow.jobs
    for name, job in jobs.items():
        texts = [*_strings(job), *_strings(workflow.data.get('env', ''))]
        secrets = sorted({match.group(1) or 'secrets[...]' for text in texts for match in _SECRET.finditer(text)})
        if job.get('secrets') == 'inherit':
            secrets.append('secrets: inherit')
        if not secrets:
            continue
        chain = _condition_chain(jobs, name)
        if any(_normalized(condition) in chain for condition in trust.pull_request.never_runs_on):
            continue
        for actor, actor_class in trust.credential_less.items():
            if not any(_normalized(condition) in chain for condition in actor_class.conditions):
                findings.append(
                    Finding(
                        workflow.path.name,
                        'credential-less-actor',
                        f'{name} reads {", ".join(secrets)}, but no `if:` it depends on excludes {actor} runs, '
                        f'which get no secrets; add one of: {"; ".join(actor_class.conditions)}',
                    )
                )
    return findings


_BOT = re.compile(r'\b[A-Za-z0-9][A-Za-z0-9-]*\[bot\]')
_TEAM_APP_LOGINS = re.compile(r'\bTEAM_APP_LOGINS=\(([^)]*)\)')


def _allowlist_sites(workflows: list[Workflow], sources: dict[str, str]) -> Iterator[tuple[str, Role, set[str]]]:
    """Each bot allowlist, as where it is, the role it lists and its logins."""
    for workflow in workflows:
        for job in workflow.jobs.values():
            for step in _steps(job):
                allowed = _mapping(step.get('with')).get('allowed_bots')
                if isinstance(allowed, str):
                    logins = {login.strip().removesuffix('[bot]') + '[bot]' for login in allowed.split(',')}
                    yield workflow.path.name, 'claude-trigger', logins
    for path, text in sources.items():
        for match in _TEAM_APP_LOGINS.finditer(text):
            yield path, 'team-app', set(_BOT.findall(match.group(1)))


def check_bot_registry(workflows: list[Workflow], sources: dict[str, str], trust: Trust) -> list[Finding]:
    """Every bot `sources` name is in `.github/trust.toml`, and every allowlist holds exactly its role's bots there."""
    findings: list[Finding] = []
    for path, text in sources.items():
        for login in sorted(set(_BOT.findall(text)) - trust.bots.keys()):
            findings.append(Finding(path, 'bot-registry', f'add `{login}` to `[bots]` in `.github/trust.toml`'))
    sited: set[Role] = set()
    for path, role, logins in _allowlist_sites(workflows, sources):
        sited.add(role)
        expected = {login for login, bot in trust.bots.items() if role in bot.roles}
        if logins != expected:
            findings.append(
                Finding(
                    path,
                    'bot-registry',
                    f'the `{role}` allowlist holds {sorted(logins)}, but `.github/trust.toml` gives that role to '
                    f'{sorted(expected)}',
                )
            )
    for role in sorted({role for bot in trust.bots.values() for role in bot.roles} - sited):
        findings.append(
            Finding('.github/trust.toml', 'bot-registry', f'no allowlist lists the `{role}` bots, so nothing uses them')
        )
    return findings


_ALLOWED_TOOLS = re.compile(r'--allowedTools\s+"([^"]*)"')


def check_write_wildcard(workflows: list[Workflow], trust: Trust) -> list[Finding]:
    """An agent's write command is bound to its target, unless `unbound_write_tools` says why the job needs it."""
    rows = {(row.workflow, row.job, row.tool) for row in trust.unbound_write_tools}
    seen: set[tuple[str, str, str]] = set()
    findings: list[Finding] = []
    for workflow in workflows:
        for name, job in workflow.jobs.items():
            for text in _strings(job):
                for tools in _ALLOWED_TOOLS.findall(text):
                    for command in re.findall(r'Bash\(([^)]*):\*\)', tools):
                        if not any(write == command or write.startswith(f'{command} ') for write in WRITE_COMMANDS):
                            continue
                        key = (workflow.path.name, name, command)
                        seen.add(key)
                        if key not in rows:
                            findings.append(
                                Finding(
                                    workflow.path.name,
                                    'write-wildcard',
                                    f'{name}: `Bash({command}:*)` lets the agent write to any target; bind it, as '
                                    '`Bash(gh pr comment ${{ github.event.pull_request.number }}:*)`, or add an '
                                    '`unbound_write_tools` row saying why the job needs it',
                                )
                            )
    for workflow_name, job, tool in sorted(rows - seen):
        findings.append(
            Finding(
                '.github/trust.toml',
                'write-wildcard',
                f'the `unbound_write_tools` row for `{tool}` in {workflow_name} / {job} matches no tool',
            )
        )
    return findings


def check_unpinned_image(workflow: Workflow) -> list[Finding]:
    """Every container image a workflow pulls names its digest."""
    images: list[tuple[str, str]] = []
    for name, job in workflow.jobs.items():
        container = job.get('container')
        container_image = _mapping(container).get('image', container)
        if isinstance(container_image, str):
            images.append((name, container_image))
        for service in _mapping(job.get('services')).values():
            image = _mapping(service).get('image')
            if isinstance(image, str):
                images.append((name, image))
        for step in _steps(job):
            uses = step.get('uses')
            if isinstance(uses, str) and uses.startswith('docker://'):
                images.append((name, uses))
    for job, _, script in _run_scripts(workflow):
        images.extend((job, line.strip()) for line in script.splitlines() if re.search(r'\bdocker run\b', line))
    return [
        Finding(
            workflow.path.name,
            'unpinned-image',
            f'{job}: pin `{_excerpt(image)}` to its digest, as `name:tag@sha256:...`',
        )
        for job, image in images
        if '@sha256:' not in image
    ]


def load(root: Path) -> tuple[list[Workflow], dict[str, str]]:
    """The non-generated workflows under `root`, and the shell scripts under `.github/scripts` by path."""
    workflows = [
        Workflow.parse(path, path.read_text(encoding='utf-8'))
        for path in sorted((root / '.github' / 'workflows').glob('*.yml'))
        if not GENERATED.search(path.name)
    ]
    scripts = {
        str(path.relative_to(root)): path.read_text(encoding='utf-8')
        for path in sorted((root / '.github' / 'scripts').glob('*.sh'))
    }
    return workflows, scripts


def check(root: Path, trust: Trust) -> list[Finding]:
    """Every finding over the workflows under `root`."""
    workflows, scripts = load(root)
    findings: list[Finding] = []
    for workflow in workflows:
        findings += check_shell_default(workflow)
        findings += check_error_as_data(workflow)
        findings += check_pipeline_condition(workflow)
        findings += check_credential_less_actor(workflow, trust)
        findings += check_unpinned_image(workflow)
    sources = {**{workflow.path.name: workflow.text for workflow in workflows}, **scripts}
    findings += check_bot_registry(workflows, sources, trust)
    findings += check_write_wildcard(workflows, trust)
    return findings


def read_trust(path: Path) -> Trust:
    """Parse `.github/trust.toml`."""
    return Trust.model_validate(tomllib.loads(path.read_text(encoding='utf-8')))


def main(argv: Sequence[str] | None = None) -> int:
    """Print every finding and return 1 when there is one."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0] if __doc__ else None)
    parser.add_argument('root', nargs='?', type=Path, default=Path('.'), help='the repository checkout to check')
    parser.add_argument('--trust', type=Path, help='the registry to check against; default: ROOT/.github/trust.toml')
    args = parser.parse_args(argv)
    root: Path = args.root
    trust_path: Path | None = args.trust
    findings = check(root, read_trust(trust_path or root / '.github' / 'trust.toml'))
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
