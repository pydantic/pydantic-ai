"""Tests for `workflow_policy.py`.

Every workflow passes, and each rule rejects the configuration that shipped the defect it encodes, reconstructed from
the parent commit of the pull request that fixed it, and accepts the fix.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from workflow_policy import (
    Finding,
    Trust,
    Workflow,
    check,
    check_bot_registry,
    check_credential_less_actor,
    check_error_as_data,
    check_pipeline_condition,
    check_shell_default,
    check_unpinned_image,
    check_write_wildcard,
    main,
    read_trust,
)

ROOT = Path(__file__).parents[2]
TRUST = read_trust(ROOT / '.github' / 'trust.toml')


def _job(steps: str, extra: str = '') -> str:
    return f'on: pull_request\njobs:\n  job:\n    runs-on: ubuntu-latest\n{extra}    steps:\n{steps}'


def _run(script: str) -> str:
    body = ''.join(f'          {line}\n' for line in script.splitlines())
    return _job(f'      - name: step\n        run: |\n{body}')


def test_every_workflow_passes():
    assert check(ROOT, TRUST) == []


def _workflow_checks(
    rule_check: Callable[[Workflow], list[Finding]], name: str = 'workflow.yml'
) -> Callable[[str], list[str]]:
    return lambda text: [finding.rule for finding in rule_check(Workflow.parse(Path(name), text))]


def _with_rows(rows: list[dict[str, str]]) -> Trust:
    return Trust.model_validate({**TRUST.model_dump(), 'unbound_write_tools': rows})


CASES = [
    pytest.param(
        _workflow_checks(check_error_as_data),
        _run(
            'AUTHOR_ROLE=$(gh api "repos/${REPO}/collaborators/${PR_AUTHOR}/permission" --jq \'.role_name\''
            ' 2>/dev/null || echo "unknown")'
        ),
        _run(
            'repo_role() {\n'
            '  local role\n'
            '  role=$(gh api "repos/${REPO}/collaborators/$1/permission" --jq \'.role_name\' 2>/dev/null) || return 1\n'
            '  echo "$role"\n'
            '}'
        ),
        'error-as-data',
        id='https://github.com/pydantic/pydantic-ai/pull/9373',
    ),
    pytest.param(
        _workflow_checks(check_error_as_data),
        _run('X="$(curl -fsS "$URL" || printf \'{}\')"'),
        _run("echo '$(gh api x || echo y)'\ncat <<EOF\n$(gh api x || echo y)\nEOF"),
        'error-as-data',
        id='quoted-substitution-and-heredoc',
    ),
    pytest.param(
        _workflow_checks(check_pipeline_condition),
        _run(
            'if gh issue list --state open --search "${title} in:title" \\\n'
            '  --json title --jq \'.[].title\' | grep -qxF "${title}"; then\n'
            '  exit 0\n'
            'fi'
        ),
        _run(
            'existing_titles=$(gh issue list --state open --search "${title} in:title" \\\n'
            "  --json title --jq '.[].title')\n"
            'if grep -qxF "${title}" <<< "$existing_titles"; then\n'
            '  exit 0\n'
            'fi'
        ),
        'pipeline-condition',
        id='https://github.com/pydantic/pydantic-ai/pull/9281',
    ),
    pytest.param(
        _workflow_checks(lambda workflow: check_credential_less_actor(workflow, TRUST), 'agent-provider-health.yml'),
        'on: [workflow_run]\njobs:\n  monitor:\n'
        "    if: github.event_name != 'workflow_run' || github.event.workflow_run.event != 'pull_request' ||"
        ' github.event.workflow_run.head_repository.full_name == github.repository\n'
        '    runs-on: ubuntu-latest\n    env:\n      ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}\n'
        '    steps:\n      - run: python3 .github/scripts/agent_provider_health.py\n',
        'on: [workflow_run]\njobs:\n  triage:\n'
        "    if: github.event.workflow_run.event != 'pull_request' ||"
        ' (github.event.workflow_run.head_repository.full_name == github.repository &&'
        " github.event.workflow_run.actor.login != 'dependabot[bot]')\n"
        '    runs-on: ubuntu-latest\n    steps:\n      - run: echo\n'
        '  monitor:\n    needs: triage\n'
        "    if: ${{ !cancelled() && needs.triage.outputs.reconcile == 'true' }}\n"
        '    runs-on: ubuntu-latest\n    env:\n      ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}\n'
        '    steps:\n      - run: python3 .github/scripts/agent_provider_health.py\n',
        'credential-less-actor',
        id='https://github.com/pydantic/pydantic-ai/pull/10213',
    ),
    pytest.param(
        _workflow_checks(lambda workflow: check_credential_less_actor(workflow, TRUST)),
        _job("      - run: deploy\n        env:\n          KEY: ${{ secrets['DEPLOY_KEY'] }}\n"),
        _job(
            "      - run: deploy\n        env:\n          KEY: ${{ secrets['DEPLOY_KEY'] }}\n",
            extra="    if: github.event_name == 'push'\n",
        ),
        'credential-less-actor',
        id='never-runs-on-pull-request',
    ),
    pytest.param(
        _workflow_checks(lambda workflow: check_write_wildcard([workflow], _with_rows([]))),
        _job(
            '      - uses: anthropics/claude-code-action@v1\n        with:\n          claude_args: |\n'
            '            --allowedTools "Bash(gh pr comment:*),Bash(gh pr diff:*)"\n'
        ),
        _job(
            '      - uses: anthropics/claude-code-action@v1\n        with:\n          claude_args: |\n'
            '            --allowedTools "Bash(gh pr comment ${{ github.event.pull_request.number }}:*),'
            'Bash(gh pr diff:*)"\n'
        ),
        'write-wildcard',
        id='https://github.com/pydantic/pydantic-ai/pull/12357',
    ),
    pytest.param(
        _workflow_checks(check_unpinned_image),
        _job(
            '      - run: pytest\n',
            extra='    services:\n      postgres:\n        image: public.ecr.aws/docker/library/postgres:16\n',
        ),
        _job(
            '      - run: pytest\n',
            extra='    services:\n      postgres:\n        image: postgres:16@sha256:'
            '223efeb196329b55b3b9a13db144956411dbc18e727e59f681c3e0aede6475cf\n',
        ),
        'unpinned-image',
        id='https://github.com/pydantic/pydantic-ai/pull/10216',
    ),
    pytest.param(
        _workflow_checks(check_shell_default),
        _job('      - run: gh api x | base64 -d > script.sh\n'),
        'defaults:\n  run:\n    shell: bash\n' + _job('      - run: gh api x | base64 -d > script.sh\n'),
        'shell-default',
        id='shell-default',
    ),
]


@pytest.mark.parametrize(('rules', 'broken', 'fixed', 'rule'), CASES)
def test_each_rule_rejects_the_shipped_defect_and_accepts_its_fix(
    rules: Callable[[str], list[str]], broken: str, fixed: str, rule: str
):
    assert rule in rules(broken)
    assert rules(fixed) == []


def _registry(*sources: str) -> list[str]:
    workflows = [Workflow.parse(Path(f'{index}.yml'), text) for index, text in enumerate(sources)]
    findings = check_bot_registry(workflows, {workflow.path.name: workflow.text for workflow in workflows}, TRUST)
    return [finding.message for finding in findings]


_CLAUDE_TRIGGER = _job(
    '      - uses: anthropics/claude-code-action@v1\n        with:\n          allowed_bots: "{bots}"\n'
)
_TEAM_APPS = _run('TEAM_APP_LOGINS=("pydanty[bot]" "montybot-pydantic[bot]")')


def test_a_bot_allowlist_holds_exactly_its_roles_bots():
    # https://github.com/pydantic/pydantic-ai/pull/9685 let Dependabot trigger `category-classify`.
    assert _registry(_CLAUDE_TRIGGER.replace('{bots}', 'pydanty'), _TEAM_APPS) == [
        "the `claude-trigger` allowlist holds ['pydanty[bot]'], but `.github/trust.toml` gives that role to "
        "['dependabot[bot]', 'pydanty[bot]']"
    ]
    assert _registry(_CLAUDE_TRIGGER.replace('{bots}', 'pydanty,dependabot'), _TEAM_APPS) == []


def test_every_bot_and_role_is_registered_and_used():
    # Before https://github.com/pydantic/pydantic-ai/pull/13242, PR Guard named only pydanty.
    assert _registry(_CLAUDE_TRIGGER.replace('{bots}', 'pydanty,dependabot'), _run('PYDANTY_LOGIN="pydanty[bot]"')) == [
        'no allowlist lists the `team-app` bots, so nothing uses them'
    ]
    assert _registry(
        _CLAUDE_TRIGGER.replace('{bots}', 'pydanty,dependabot'), _TEAM_APPS, _run('[ "$ACTOR" = "new-app[bot]" ]')
    ) == ['add `new-app[bot]` to `[bots]` in `.github/trust.toml`']


def test_an_unbound_write_tool_row_must_match_a_tool():
    unused = _with_rows([{'workflow': 'workflow.yml', 'job': 'job', 'tool': 'git push', 'reason': 'x'}])
    assert check_write_wildcard([Workflow.parse(Path('workflow.yml'), _job('      - run: echo\n'))], unused) == [
        Finding(
            '.github/trust.toml',
            'write-wildcard',
            'the `unbound_write_tools` row for `git push` in workflow.yml / job matches no tool',
        )
    ]


def test_main_reports_findings_against_a_given_registry(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    workflows = tmp_path / '.github' / 'workflows'
    workflows.mkdir(parents=True)
    (workflows / 'ci.yml').write_text(_run('X=$(gh api x || true)'), encoding='utf-8')
    (workflows / 'ci.lock.yml').write_text(_run('X=$(gh api x || true)'), encoding='utf-8')

    assert main([str(tmp_path), '--trust', str(ROOT / '.github' / 'trust.toml')]) == 1
    output = capsys.readouterr().out.splitlines()
    assert [line.split(': ', 2)[:2] for line in output] == [
        ['ci.yml', 'shell-default'],
        ['ci.yml', 'error-as-data'],
        ['.github/trust.toml', 'bot-registry'],
        ['.github/trust.toml', 'bot-registry'],
        ['.github/trust.toml', 'write-wildcard'],
        ['.github/trust.toml', 'write-wildcard'],
    ]
