"""Tests for the agentic workflow policy guard.

Each check has a regression test built from the *actual* pre-fix configuration
that shipped to `main` (reconstructed from the parent of #6761), so the guard is
verified against the defects it exists to prevent rather than against invented
shapes. The final test asserts the live repository is clean.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

import agentic_workflow_guard
from agent_provider_health import AGENT_START_HEALTH_STEPS
from agentic_workflow_guard import (
    HOST_RESULT_JOB,
    Violation,
    changed_files,
    check_ai_credits_accounting,
    check_assigned_alert_metadata_gate,
    check_awf_binary_version,
    check_compiled_runner_contract,
    check_compiler_version_compatibility,
    check_compiler_versions,
    check_daily_aic_guardrail,
    check_dangling_needs,
    check_engine_sys_path_order,
    check_job_timeout_env,
    check_lock_regenerated,
    check_logfire_wiring,
    check_prompt_paths,
    check_provider_engine_config,
    check_provider_health_agent_start,
    check_provider_health_host_result,
    check_provider_health_identity,
    check_provider_health_monitor,
    check_provider_health_recovery_wiring,
    check_provider_health_wiring,
    check_safe_output_job_max,
    check_timeout_declared,
    compiled_awf_configs,
    run_checks,
)

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOWS_DIR = REPO_ROOT / '.github' / 'workflows'


def test_ci_review_compiled_membership_admits_installed_app_without_widening_roles():
    lock = yaml.safe_load((WORKFLOWS_DIR / 'pydantic-ai-pr-review.lock.yml').read_text(encoding='utf-8'))
    steps = lock['jobs']['pre_activation']['steps']
    check_membership = next(step for step in steps if step.get('id') == 'check_membership')

    assert check_membership['env'] == {
        'GH_AW_REQUIRED_ROLES': 'admin,maintainer,write',
        'GH_AW_ALLOWED_BOTS': 'pydantic-docs',
    }


@pytest.mark.parametrize(
    'body,commit_id,expected',
    [
        ('<!-- gh-aw-workflow-call-id: pydantic/pydantic-ai/pydantic-ai-pr-review -->', 'current-head', 1),
        ('[gh-aw-workflow-call-id]: # "pydantic%2Fpydantic-ai%2Fpydantic-ai-pr-review"', 'current-head', 1),
        ('<!-- gh-aw-workflow-call-id: pydantic/pydantic-ai/pydantic-ai-ui-security-review -->', 'current-head', 0),
        ('[gh-aw-workflow-call-id]: # "pydantic%2Fpydantic-ai%2Fpydantic-ai-ui-security-review"', 'current-head', 0),
        ('<!-- gh-aw-workflow-call-id: pydantic/pydantic-ai/pydantic-ai-pr-review -->', 'older-head', 0),
        ('[gh-aw-workflow-call-id]: # "pydantic%2Fpydantic-ai%2Fpydantic-ai-pr-review"', 'older-head', 0),
        ('No findings.', 'current-head', 0),
        (None, 'current-head', 0),
    ],
)
def test_ci_review_deduplicates_both_footer_formats(body: str | None, commit_id: str, expected: int):
    lock = yaml.safe_load((WORKFLOWS_DIR / 'pydantic-ai-pr-review.lock.yml').read_text(encoding='utf-8'))
    gate = next(step for step in lock['jobs']['eligibility']['steps'] if step.get('id') == 'gate')
    query = re.search(
        r"""jq --arg sha "\$HEAD_SHA" --arg caller "\$REVIEW_CALL_ID" '(.+?)'\)""", gate['run'], re.DOTALL
    )
    assert query is not None
    caller = gate['env']['REVIEW_CALL_ID'].replace('${{ github.repository }}', 'pydantic/pydantic-ai')
    # `$REVIEWS`: every page of reviews, already flattened by `jq 'add // []'`.
    reviews: list[dict[str, object]] = [
        {'commit_id': commit_id, 'body': body, 'user': {'login': 'github-actions[bot]'}},
        {'commit_id': 'current-head', 'body': 'Another review from the same bot.'},
    ]
    # Exercise the compiled jq program with its production interpreter.
    result = subprocess.run(
        ['jq', '--arg', 'sha', 'current-head', '--arg', 'caller', caller, query.group(1)],
        input=json.dumps(reviews),
        text=True,
        capture_output=True,
        check=True,
    )
    assert int(result.stdout) == expected


def test_ci_review_stale_preflight_writes_noop_before_safe_outputs_setup(tmp_path: Path):
    """The host preflight runs before safe-output setup, outside the model/VCR path."""
    lock = yaml.safe_load((WORKFLOWS_DIR / 'pydantic-ai-pr-review.lock.yml').read_text(encoding='utf-8'))
    steps = lock['jobs']['agent']['steps']
    preflight = next(step for step in steps if step.get('name') == 'Skip a PR that closed or moved on while queued')
    prepare = next(step for step in steps if step.get('name') == 'Prepare Safe Outputs Directories')
    safe_outputs = tmp_path / 'gh-aw' / 'safeoutputs' / 'outputs.jsonl'
    assert not safe_outputs.parent.exists()

    gh = tmp_path / 'gh'
    gh.write_text('#!/bin/sh\nprintf "closed at new-head\\n"\n', encoding='utf-8')
    gh.chmod(0o755)
    search_path = os.environ['PATH']
    env: dict[str, str] = {
        **os.environ,
        'PATH': f'{tmp_path}:{search_path}',
        'RUNNER_TEMP': str(tmp_path),
        'SAFE_OUTPUTS': str(safe_outputs),
        'PR': 'repos/pydantic/pydantic-ai/pulls/10450',
        'HEAD_SHA': 'old-head',
        'GITHUB_STEP_SUMMARY': str(tmp_path / 'summary.md'),
        'GITHUB_OUTPUT': str(tmp_path / 'github-output'),
    }
    result = subprocess.run(['bash', '-c', preflight['run']], env=env, text=True, capture_output=True, check=True)
    assert safe_outputs.exists(), result.stdout + result.stderr
    original = safe_outputs.read_bytes()
    assert json.loads(original) == {
        'type': 'noop',
        'message': 'Not reviewing old-head: the PR is now closed at new-head.',
    }
    # Gates `Check out the PR head`, which would refuse a head the branch has moved past.
    assert (tmp_path / 'github-output').read_text(encoding='utf-8') == 'skipped=true\n'

    subprocess.run(['bash', '-c', prepare['run']], env=env, check=True)
    assert safe_outputs.read_bytes() == original


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


# --- dangling-needs (the `ui-security-review` defect, #6766 F7) ---------------


def test_dangling_needs_catches_the_ui_security_review_defect(tmp_path: Path):
    """`activation` gated on `needs.detect` without depending on `detect`.

    This is the exact shape that shipped: the expression evaluated to empty, the
    whole chain skipped, and because a job skipped by `if:` reports success the
    required check stayed green for a month while the review never ran.
    """
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  detect:
    needs: activation
    runs-on: ubuntu-latest
  activation:
    needs:
      - fetch_dynamic_prompt
      - pre_activation
    if: needs.pre_activation.outputs.activated == 'true' && needs.detect.outputs.touched == 'true'
    runs-on: ubuntu-latest
  fetch_dynamic_prompt:
    runs-on: ubuntu-latest
  pre_activation:
    runs-on: ubuntu-latest
""",
    )

    violations = check_dangling_needs(lock)

    assert [v.check for v in violations] == ['dangling-needs']
    assert 'job `activation` references `needs.detect`' in violations[0].message


def test_dangling_needs_accepts_the_repaired_graph(tmp_path: Path):
    """The #6761 fix — `detect` first, `activation` depending on it — is clean."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  detect:
    runs-on: ubuntu-latest
  activation:
    needs:
      - detect
      - pre_activation
    if: needs.pre_activation.outputs.activated == 'true' && needs.detect.outputs.touched == 'true'
    runs-on: ubuntu-latest
  pre_activation:
    runs-on: ubuntu-latest
""",
    )

    assert check_dangling_needs(lock) == []


def test_dangling_needs_checks_outputs_as_well_as_if(tmp_path: Path):
    """`outputs:` silently resolves to empty for a non-dependency, same as `if:`."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  build:
    runs-on: ubuntu-latest
  publish:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ needs.build.outputs.digest }}
""",
    )

    violations = check_dangling_needs(lock)

    assert len(violations) == 1
    assert 'in `outputs.digest:`' in violations[0].message


def test_dangling_needs_checks_step_conditions(tmp_path: Path):
    """A step `if:` naming a non-dependency skips just as silently as a job.

    The job around it still reports success, so the skipped step is invisible —
    the F7 failure mode one scope deeper.
    """
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  setup:
    runs-on: ubuntu-latest
  detection:
    needs:
      - setup
    runs-on: ubuntu-latest
    steps:
      - name: Collect patch
        if: needs.agent.outputs.has_patch == 'true'
        run: echo hi
""",
    )

    violations = check_dangling_needs(lock)

    assert [v.check for v in violations] == ['dangling-needs']
    assert 'steps[Collect patch].if' in violations[0].message
    assert 'needs.agent' in violations[0].message


def test_dangling_needs_accepts_a_step_condition_on_a_real_dependency(tmp_path: Path):
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  agent:
    runs-on: ubuntu-latest
  detection:
    needs:
      - agent
    runs-on: ubuntu-latest
    steps:
      - name: Collect patch
        if: needs.agent.outputs.has_patch == 'true'
        run: echo hi
""",
    )

    assert check_dangling_needs(lock) == []


def test_dangling_needs_ignores_a_lock_without_jobs(tmp_path: Path):
    assert check_dangling_needs(_write(tmp_path / 'w.lock.yml', 'name: nothing\n')) == []


def test_dangling_needs_checks_step_env_and_with(tmp_path: Path):
    """These don't skip — the step runs with an empty value, which is worse.

    gh-aw emits exactly this shape (`GH_AW_NEEDS_DETECT_OUTPUTS_TOUCHED`), so a
    dangling ref here means a wrong action call or shell variable going through
    while everything reports success.
    """
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  setup:
    runs-on: ubuntu-latest
  detection:
    needs:
      - setup
    runs-on: ubuntu-latest
    env:
      JOB_LEVEL: ${{ needs.missing_job.outputs.value }}
    steps:
      - name: Report
        env:
          GH_AW_NEEDS_DETECT_OUTPUTS_TOUCHED: ${{ needs.detect.outputs.touched }}
        uses: actions/github-script@v9
        with:
          script: ${{ needs.agent.outputs.text }}
""",
    )

    violations = check_dangling_needs(lock)

    assert {v.check for v in violations} == {'dangling-needs'}
    fields = sorted(message.split('` in `')[1].split(':`')[0] for message in (v.message for v in violations))
    assert fields == [
        'env.JOB_LEVEL',
        'steps[Report].env.GH_AW_NEEDS_DETECT_OUTPUTS_TOUCHED',
        'steps[Report].with.script',
    ]


def test_dangling_needs_ignores_a_bare_reference_in_a_shell_script(tmp_path: Path):
    """Outside `if:`, GitHub only evaluates `needs.*` inside `${{ }}`.

    A `run:` block that merely spells the word — a comment, a jq program — is literal
    text, so flagging it would make the check unusable on real workflows.
    """
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Explain
        run: |
          # this job needs.detect to have run first, see the docs
          echo 'needs.agent.outputs.text is set elsewhere'
""",
    )

    assert check_dangling_needs(lock) == []


def test_dangling_needs_flags_an_interpolated_reference_in_a_shell_script(tmp_path: Path):
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Emit
        run: echo "${{ needs.detect.outputs.touched }}"
""",
    )

    violations = check_dangling_needs(lock)

    assert [v.check for v in violations] == ['dangling-needs']
    assert 'steps[Emit].run' in violations[0].message


# --- safe-output max (the `attention-triage` defect, #6766 F5) ----------------


def test_safe_output_job_max_catches_the_attention_triage_defect(tmp_path: Path):
    """A safe-output job with no `max:` silently truncates to one item."""
    source = _write(
        tmp_path / 'w.md',
        """---
timeout-minutes: 30
safe-outputs:
  jobs:
    record-attention-decision:
      description: "Classify one bounded candidate."
      runs-on: ubuntu-latest
---
prompt
""",
    )

    violations = check_safe_output_job_max(source)

    assert [v.check for v in violations] == ['safe-output-job-max']
    assert 'record-attention-decision' in violations[0].message


def test_safe_output_job_max_accepts_an_explicit_bound(tmp_path: Path):
    source = _write(
        tmp_path / 'w.md',
        """---
safe-outputs:
  jobs:
    record-attention-decision:
      max: 10
      runs-on: ubuntu-latest
---
prompt
""",
    )

    assert check_safe_output_job_max(source) == []


def test_safe_output_job_max_ignores_builtin_safe_outputs(tmp_path: Path):
    """Built-in types like `create-issue` are not custom jobs; gh-aw bounds them."""
    source = _write(
        tmp_path / 'w.md',
        """---
safe-outputs:
  create-issue:
    title-prefix: "[sweep] "
---
prompt
""",
    )

    assert check_safe_output_job_max(source) == []


# --- prompt paths (the review-context defect, #6766 F3) ----------------------


@pytest.mark.parametrize(
    'path', ['/tmp/gh-aw/.review-context/', '/tmp/gh-aw/agent/open-issues.tsv', '/tmp/gh-aw/agent/issues/1.json']
)
def test_prompt_paths_catches_the_review_context_defect(tmp_path: Path, path: str):
    """The prompt told the agent to read a path its file tools cannot open."""
    source = _write(
        tmp_path / 'shared.md',
        f"""---
name: x
---

A pre-agent step wrote everything you need to `{path}`.
**Read these files instead of calling the GitHub API.**
""",
    )

    violations = check_prompt_paths(source)

    assert [v.check for v in violations] == ['prompt-path-outside-workspace']
    assert path in violations[0].message


def test_prompt_paths_accepts_a_workspace_relative_path(tmp_path: Path):
    source = _write(
        tmp_path / 'shared.md',
        """---
name: x
---

A pre-agent step wrote everything you need to `.review-context/` at the root of
the checked-out repository.
""",
    )

    assert check_prompt_paths(source) == []


def test_prompt_paths_allows_the_launcher_staging_directory(tmp_path: Path):
    """`/tmp/gh-aw/bin` is gh-aw's exec-able launcher path, never read by the agent."""
    source = _write(
        tmp_path / 'shared.md',
        """---
name: x
---

The launcher is staged into gh-aw's exec-able `/tmp/gh-aw/bin` path.
""",
    )

    assert check_prompt_paths(source) == []


def test_prompt_paths_does_not_allow_a_sibling_of_the_launcher_directory(tmp_path: Path):
    """`bin` must not open up `bindings/` — the allowlist is a directory, not a prefix."""
    source = _write(
        tmp_path / 'shared.md',
        """---
name: x
---

Read the staged secrets from `/tmp/gh-aw/bindings/secret.json`.
""",
    )

    violations = check_prompt_paths(source)

    assert [v.check for v in violations] == ['prompt-path-outside-workspace']


def test_prompt_paths_keeps_scanning_after_a_nested_fence(tmp_path: Path):
    """A fence nested inside a longer one must not leave the scanner stuck.

    Toggling on every ``` reads the inner block's opener as this block's closer and
    its closer as a new opener, so everything after is treated as fenced and skipped —
    a false PASS in the check whose whole job is catching a silent failure. CommonMark
    closes a fence only on the same character, at least as long as the opener.
    """
    source = _write(
        tmp_path / 'shared.md',
        """---
name: x
---

````markdown
Wrap a snippet for the agent to copy:

```bash
echo hi
```
````

A pre-agent step wrote everything you need to `/tmp/gh-aw/.review-context/`.
""",
    )

    violations = check_prompt_paths(source)

    assert [v.check for v in violations] == ['prompt-path-outside-workspace']


@pytest.mark.parametrize(
    'path',
    [
        # Truncating the match at `:` would leave `/tmp/gh-aw/bin`, which is allowlisted.
        '/tmp/gh-aw/bin:secret/context.json',
        '/tmp/gh-aw/agent/github-context.bak/1.json',
        '/tmp/gh-aw/agent/github-context-private/1.json',
    ],
)
def test_prompt_paths_flags_a_near_miss_of_an_allowlist_entry(tmp_path: Path, path: str):
    """An allowlist entry must match a whole path component, never a bare prefix."""
    source = _write(tmp_path / 'shared.md', f'---\nname: x\n---\n\nRead {path} for context.\n')

    violations = check_prompt_paths(source)

    assert [v.check for v in violations] == ['prompt-path-outside-workspace']


@pytest.mark.parametrize(
    'path',
    [
        '/tmp/gh-aw/bin',
        '/tmp/gh-aw/bin/pydantic-ai-runner-launch',
        '/tmp/gh-aw/agent/github-context/x.json',
    ],
)
def test_prompt_paths_allows_every_documented_corpus_path(tmp_path: Path, path: str):
    """The allowlist's own entries, and paths under them, must keep passing."""
    source = _write(tmp_path / 'shared.md', f'---\nname: x\n---\n\nRead {path} for context.\n')

    assert check_prompt_paths(source) == []


def test_prompt_paths_resolves_traversal_out_of_an_allowlisted_directory(tmp_path: Path):
    """Textually under `bin/`, but it resolves to the review-context path F3 was about."""
    source = _write(
        tmp_path / 'shared.md',
        """---
name: x
---

Read `/tmp/gh-aw/bin/../.review-context/pr.json` for context.
""",
    )

    violations = check_prompt_paths(source)

    assert [v.check for v in violations] == ['prompt-path-outside-workspace']


def test_prompt_paths_treats_an_info_string_line_as_content_not_a_closer(tmp_path: Path):
    """CommonMark: a closer carries no info string, so ```bash inside a block is content.

    Reading it as a closer puts the scanner one block out of step — the real closer then
    reads as an opener and everything after it is skipped, which is a silent PASS.
    """
    source = _write(
        tmp_path / 'shared.md',
        """---
name: x
---

```markdown
```bash
echo hi
```

A pre-agent step wrote everything you need to `/tmp/gh-aw/.review-context/`.
""",
    )

    violations = check_prompt_paths(source)

    assert [v.check for v in violations] == ['prompt-path-outside-workspace']


def test_prompt_paths_ignores_a_tilde_fenced_snippet(tmp_path: Path):
    """`~~~` is a CommonMark fence too, and shell inside one is still shell."""
    source = _write(
        tmp_path / 'shared.md',
        """---
name: x
---

~~~bash
jq '.[] | {number}' /tmp/gh-aw/agent/some-corpus.json
~~~
""",
    )

    assert check_prompt_paths(source) == []


def test_prompt_paths_ignores_shell_snippets(tmp_path: Path):
    """A path inside a fenced block goes to `Bash`, which is not rooted at the checkout.

    Flagging these would condemn the documented `jq` reads of the prefetched GitHub
    corpus, which work fine.
    """
    source = _write(
        tmp_path / 'shared.md',
        """---
name: x
---

Filter the local corpus:

```bash
jq '.[] | {number}' /tmp/gh-aw/agent/some-corpus.json
```
""",
    )

    assert check_prompt_paths(source) == []


def test_prompt_paths_ignores_frontmatter(tmp_path: Path):
    """Frontmatter is config, not agent-facing prompt text."""
    source = _write(
        tmp_path / 'shared.md',
        """---
pre-agent-steps:
  - run: install -m 755 launcher /tmp/gh-aw/.review-context/x
---

prompt body with no paths
""",
    )

    assert check_prompt_paths(source) == []


# --- job-timeout env consistency ----------------------------------------------


def test_job_timeout_env_flags_a_mismatch(tmp_path: Path):
    """A drifted budget makes the agent stop early or get killed mid-run."""
    source = _write(
        tmp_path / 'w.md',
        '---\ntimeout-minutes: 45\nenv:\n  PYDANTIC_AI_JOB_TIMEOUT_MINUTES: "30"\n---\nprompt\n',
    )

    violations = check_job_timeout_env(source)

    assert [v.check for v in violations] == ['job-timeout-env-mismatch']


def test_job_timeout_env_accepts_matching_values(tmp_path: Path):
    source = _write(
        tmp_path / 'w.md',
        '---\ntimeout-minutes: 45\nenv:\n  PYDANTIC_AI_JOB_TIMEOUT_MINUTES: "45"\n---\nprompt\n',
    )

    assert check_job_timeout_env(source) == []


def test_job_timeout_env_is_required(tmp_path: Path):
    """Absent, the shim assumes 30 — which silently broke two live workflows.

    `stale-issues-finder` asked for 60 minutes and only ever used 28;
    `attention-triage` asked for 20 and was killed before it could emit anything.
    """
    source = _write(tmp_path / 'w.md', '---\ntimeout-minutes: 60\n---\nprompt\n')

    violations = check_job_timeout_env(source)

    assert [v.check for v in violations] == ['job-timeout-env-missing']
    assert '"60"' in violations[0].message


@pytest.mark.parametrize('minutes', ['0', '1', '2'])
def test_job_timeout_env_rejects_a_timeout_with_no_room_for_the_agent(tmp_path: Path, minutes: str):
    """Below the shim's headroom the two disagree on what a valid budget is.

    `_run_timeout_secs` substitutes `DEFAULT_JOB_TIMEOUT_MINS` for anything at or under
    the headroom, so a matching pair like `timeout-minutes: 2` / `"2"` would pass the
    equality check above and then run the agent for 28 minutes on a job Actions kills
    at 2 — the "killed mid-flight with nothing to show" mode this guard exists to stop.
    """
    source = _write(
        tmp_path / 'w.md',
        f'---\ntimeout-minutes: {minutes}\nenv:\n  PYDANTIC_AI_JOB_TIMEOUT_MINUTES: "{minutes}"\n---\nprompt\n',
    )

    violations = check_job_timeout_env(source)

    assert [v.check for v in violations] == ['job-timeout-too-short']


def test_job_timeout_env_defers_to_timeout_declared_when_no_timeout_is_set(tmp_path: Path):
    """Without a `timeout-minutes` every message here would quote `None` as the fix."""
    source = _write(tmp_path / 'w.md', '---\nname: x\n---\nprompt\n')

    assert check_job_timeout_env(source) == []
    assert [v.check for v in check_timeout_declared(source)] == ['timeout-declared']


# --- timeout, compiler drift, lock freshness ---------------------------------


def test_timeout_declared_requires_a_wall_clock_bound(tmp_path: Path):
    source = _write(tmp_path / 'w.md', '---\nname: x\n---\nprompt\n')

    violations = check_timeout_declared(source)

    assert [v.check for v in violations] == ['timeout-declared']


def test_timeout_declared_accepts_an_explicit_bound(tmp_path: Path):
    source = _write(tmp_path / 'w.md', '---\ntimeout-minutes: 30\n---\nprompt\n')

    assert check_timeout_declared(source) == []


def test_compiler_versions_flags_a_partial_recompile(tmp_path: Path):
    old = _write(tmp_path / 'a.lock.yml', '# gh-aw-metadata: {"compiler_version":"v0.74.8"}\njobs: {}\n')
    new = _write(tmp_path / 'b.lock.yml', '# gh-aw-metadata: {"compiler_version":"v0.83.4"}\njobs: {}\n')

    violations = check_compiler_versions([old, new])

    assert [v.check for v in violations] == ['compiler-version-drift']
    assert 'v0.74.8' in violations[0].message and 'v0.83.4' in violations[0].message


def test_compiler_versions_accepts_a_uniform_set(tmp_path: Path):
    a = _write(tmp_path / 'a.lock.yml', '# gh-aw-metadata: {"compiler_version":"v0.83.4"}\njobs: {}\n')
    b = _write(tmp_path / 'b.lock.yml', '# gh-aw-metadata: {"compiler_version":"v0.83.4"}\njobs: {}\n')

    assert check_compiler_versions([a, b]) == []


def test_compiled_runner_contract_rejects_the_legacy_runner(tmp_path: Path):
    """A legacy runner must not silently bypass CLAI 2 and canonical capture."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        'jobs: {agent: {steps: [{id: agentic_execution, run: pydantic-ai-runner-launch}]}}',
    )
    violations = check_compiled_runner_contract(lock)
    assert [violation.check for violation in violations] == ['compiled-runner-contract']
    assert 'public CLAI 2 harness' in violations[0].message


_PUBLIC_ENGINE_LOCK = """
jobs:
  agent:
    env:
      GH_AW_ENGINE_VERSION: '0.64.0'
    steps:
      - id: agentic_execution
        run: >-
          awf --exclude-env ANTHROPIC_API_KEY
          --capture-command-output '${RUNNER_TEMP}/gh-aw-protected-capture-${GH_AW_PROTECTED_CAPTURE_ID}'
          -- /bin/bash -c 'exec node /tmp/gh-aw/actions/pydantic-ai_harness.cjs clai2'
        env:
          PAI_AGENT: pydantic_ai_gh_aw_shim.agent:agent
          PAI_RESULT_METADATA: pydantic_ai_gh_aw_shim.agent:result_metadata
          PAI_OTEL_EXPORT_PATH: /tmp/gh-aw/pydantic-ai-private-otel.jsonl
          PAI_MODEL: anthropic/${{ vars.GH_AW_MODEL }}
          ANTHROPIC_API_KEY: ${{ steps.agent_start_health.outputs.account == 'backup' && secrets.ZAI_BACKUP_API_KEY || secrets.ZAI_API_KEY }}
          XDG_CONFIG_HOME: /tmp/gh-aw/clai2-config
          PYTHONPATH: /tmp/gh-aw/workflow-python
          PYDANTIC_AI_TASK_ID: stable-task
          GH_AW_PROTECTED_SESSION_CAPTURE: 1
          GH_AW_PROTECTED_CAPTURE_ID: run-attempt-job
"""


def test_compiled_runner_contract_accepts_the_public_engine(tmp_path: Path):
    """The public engine retains policy, model and proxy credential exclusion."""
    lock = _write(tmp_path / 'w.lock.yml', _PUBLIC_ENGINE_LOCK)
    assert check_compiled_runner_contract(lock) == []


@pytest.mark.parametrize('version', ['0.59.0', '0.64.0', '0.65.1', 'invalid'])
def test_compiled_runner_contract_allows_compatible_runtime_upgrades(tmp_path: Path, version: str):
    lock = _write(tmp_path / 'w.lock.yml', _PUBLIC_ENGINE_LOCK.replace('0.64.0', version))
    violations = check_compiled_runner_contract(lock)
    assert bool(violations) == (version in {'0.59.0', 'invalid'})


@pytest.mark.parametrize(
    'credential',
    [
        pytest.param('${{ secrets.ZAI_API_KEY }}', id='primary-only'),
        pytest.param(
            "${{ steps.agent_start_health.outputs.account == 'backup' && secrets.ZAI_API_KEY || secrets.ZAI_BACKUP_API_KEY }}",
            id='swapped-accounts',
        ),
        pytest.param(
            "${{ needs.provider_health.outputs.account == 'backup' && secrets.ZAI_BACKUP_API_KEY || secrets.ZAI_API_KEY }}",
            id='stale-gate-account',
        ),
    ],
)
def test_compiled_runner_contract_requires_the_account_selected_key(tmp_path: Path, credential: str):
    """The proxy gets the backup key exactly when the agent-start recheck selected the backup account."""
    workflow = yaml.safe_load(_PUBLIC_ENGINE_LOCK)
    workflow['jobs']['agent']['steps'][0]['env']['ANTHROPIC_API_KEY'] = credential
    lock = _write(tmp_path / 'w.lock.yml', yaml.safe_dump(workflow))
    violations = check_compiled_runner_contract(lock)
    assert [violation.check for violation in violations] == ['compiled-runner-contract']
    assert 'account-selected Z.ai key' in violations[0].message


@pytest.mark.parametrize('variable', ['GH_AW_PROTECTED_SESSION_CAPTURE', 'GH_AW_PROTECTED_CAPTURE_ID'])
def test_compiled_runner_contract_requires_protected_capture(tmp_path: Path, variable: str):
    workflow = yaml.safe_load(_PUBLIC_ENGINE_LOCK)
    del workflow['jobs']['agent']['steps'][0]['env'][variable]
    lock = _write(tmp_path / 'w.lock.yml', yaml.safe_dump(workflow))
    violations = check_compiled_runner_contract(lock)
    assert [violation.check for violation in violations] == ['compiled-runner-contract']
    assert 'native protected session capture' in violations[0].message


@pytest.mark.parametrize('variable', ['PAI_RESULT_METADATA', 'PAI_OTEL_EXPORT_PATH'])
def test_compiled_runner_contract_requires_policy_observability(tmp_path: Path, variable: str):
    """Compiling must retain the policy's authenticated outcome and private span path."""
    workflow = yaml.safe_load(_PUBLIC_ENGINE_LOCK)
    del workflow['jobs']['agent']['steps'][0]['env'][variable]
    lock = _write(tmp_path / 'w.lock.yml', yaml.safe_dump(workflow))
    violations = check_compiled_runner_contract(lock)
    assert [violation.check for violation in violations] == ['compiled-runner-contract']


@pytest.mark.parametrize('pythonpath', [None, '/tmp/other-policy'])
def test_compiled_runner_contract_rejects_an_unavailable_policy(tmp_path: Path, pythonpath: str | None):
    workflow = yaml.safe_load(_PUBLIC_ENGINE_LOCK)
    env = workflow['jobs']['agent']['steps'][0]['env']
    if pythonpath is None:
        del env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = pythonpath
    lock = _write(tmp_path / 'w.lock.yml', yaml.safe_dump(workflow))
    violations = check_compiled_runner_contract(lock)

    assert [violation.check for violation in violations] == ['compiled-runner-contract']
    assert 'staged workflow-policy import path' in violations[0].message


@pytest.mark.parametrize(
    'command',
    [
        "echo 'pydantic-ai_harness.cjs clai2 --exclude-env ANTHROPIC_API_KEY'\nawf -- /tmp/legacy-launcher",
        'awf -- /bin/bash -c \'echo "--exclude-env ANTHROPIC_API_KEY"; node /tmp/gh-aw/actions/pydantic-ai_harness.cjs clai2\'',
        'awf --exclude-env ANTHROPIC_API_KEY -- /bin/bash -c \'echo "pydantic-ai_harness.cjs clai2"; /tmp/legacy-launcher\'',
        '# pydantic-ai_harness.cjs clai2 --exclude-env ANTHROPIC_API_KEY\nawf -- /tmp/legacy-launcher',
        "awf -- /bin/bash -c 'node /tmp/gh-aw/actions/pydantic-ai_harness.cjs clai2' --exclude-env ANTHROPIC_API_KEY",
    ],
)
def test_compiled_runner_contract_rejects_decoy_command_text(tmp_path: Path, command: str):
    """Only executed commands and outer AWF arguments establish the contract."""
    workflow = yaml.safe_load(_PUBLIC_ENGINE_LOCK)
    workflow['jobs']['agent']['steps'][0]['run'] = command
    lock = _write(tmp_path / 'w.lock.yml', yaml.safe_dump(workflow))
    assert [v.check for v in check_compiled_runner_contract(lock)] == ['compiled-runner-contract']


@pytest.mark.parametrize('lock', sorted((Path(__file__).parent.parent / 'workflows').glob('pydantic-ai-*.lock.yml')))
def test_each_workflow_exports_primary_and_detector_telemetry(lock: Path):
    """Every GLM producer receives compiled telemetry and verified accounting."""
    workflow = yaml.safe_load(lock.read_text(encoding='utf-8'))
    assert workflow['env']['OTEL_EXPORTER_OTLP_ENDPOINT']
    assert workflow['env']['OTEL_EXPORTER_OTLP_HEADERS']
    detector = next(
        step for step in workflow['jobs']['detection']['steps'] if step.get('id') == 'detection_agentic_execution'
    )
    assert 'gh_aw_opencode_detect.py' in detector['run']
    assert detector['env']['OPENCODE_API_KEY'] == '${{ secrets.OPENCODE_API_KEY }}'
    assert detector['env']['ZAI_API_KEY'] == '${{ secrets.ZAI_API_KEY }}'
    assert detector['env']['ZAI_BACKUP_API_KEY'] == '${{ secrets.ZAI_BACKUP_API_KEY }}'
    assert 'DEEPINFRA_API_KEY' not in detector['env']
    assert detector['continue-on-error'] is True
    reporter = next(
        step
        for step in workflow['jobs']['detection']['steps']
        if step.get('name') == 'Report detector usage and outcome'
    )
    assert reporter['if'] == "always() && steps.detection_guard.outputs.run_detection == 'true'"
    assert reporter['env']['DETECTOR_OUTCOME'] == '${{ steps.detection_agentic_execution.outcome }}'
    assert reporter['env']['DETECTOR_JOB_STATUS'] == '${{ job.status }}'
    assert reporter['env']['OTEL_SERVICE_NAME'] == 'gh-aw.threat-detection'
    assert reporter['continue-on-error'] is True
    assert 'gh-aw.detector.result' in reporter['with']['script']
    configs = compiled_awf_configs(lock)
    assert len(configs) == 1
    for config in configs:
        proxy = config['apiProxy']
        assert isinstance(proxy, dict)
        assert proxy['providers']
        assert proxy['defaultAiCreditsPricing']
        network: dict[str, object] = TypeAdapter(dict[str, object]).validate_python(config['network'])
        domains = network['allowDomains']
        assert isinstance(domains, list)
        assert all(
            host in domains
            for host in (
                'logfire-api.pydantic.dev',
                'logfire-eu.pydantic.dev',
                'logfire-us.pydantic.dev',
                'logfire-api.pydantic.info',
                'logfire-eu.pydantic.info',
                'logfire-us.pydantic.info',
            )
        )
    assert check_ai_credits_accounting(lock) == []


# --- AWF binary pin (the #8041 skew) ------------------------------------------

_AWF_LOCK = """
jobs:
  agent:
    steps:
      - name: Setup Scripts
        run: bash setup.sh
        env:
          GH_AW_INFO_AWF_VERSION: "v0.27.44"
      - name: Install AWF firewall binary (skipped by custom engine.command)
        run: bash "${{RUNNER_TEMP}}/gh-aw/actions/install_awf_binary.sh" {pinned}
"""


def test_awf_binary_version_catches_the_gh_aw_upgrade_skew(tmp_path: Path):
    """The shape #8041 shipped: gh-aw moved to AWF v0.27.44, the hand-written pin stayed.

    The v0.27.42 binary then rejected the v0.27.44 firewall config with
    `config.apiProxy.providers is not supported`, so the agent job exited before
    the model ever started.
    """
    lock = _write(tmp_path / 'w.lock.yml', _AWF_LOCK.format(pinned='v0.27.42'))

    violations = check_awf_binary_version(lock)

    assert [v.check for v in violations] == ['awf-binary-version']
    assert 'v0.27.42' in violations[0].message and 'v0.27.44' in violations[0].message


def test_awf_binary_version_accepts_a_matched_pin(tmp_path: Path):
    lock = _write(tmp_path / 'w.lock.yml', _AWF_LOCK.format(pinned='v0.27.44'))

    assert check_awf_binary_version(lock) == []


def test_awf_binary_version_accepts_a_quoted_version(tmp_path: Path):
    """The pin is hand-written, so a quoted argument is a shape a maintainer will produce."""
    lock = _write(tmp_path / 'w.lock.yml', _AWF_LOCK.format(pinned='"v0.27.44"'))

    assert check_awf_binary_version(lock) == []


def test_awf_binary_version_reads_every_installer_invocation_in_one_step(tmp_path: Path):
    """The last install wins, so a stale second call must not hide behind a correct first."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  agent:
    steps:
      - name: Setup Scripts
        env:
          GH_AW_INFO_AWF_VERSION: "v0.27.44"
      - name: Install AWF firewall binary (skipped by custom engine.command)
        run: |
          bash "${RUNNER_TEMP}/gh-aw/actions/install_awf_binary.sh" v0.27.44
          bash "${RUNNER_TEMP}/gh-aw/actions/install_awf_binary.sh" v0.27.42
""",
    )

    violations = check_awf_binary_version(lock)

    assert [v.check for v in violations] == ['awf-binary-version']
    assert 'v0.27.42' in violations[0].message


def test_awf_binary_version_ignores_a_lock_without_the_hand_written_install(tmp_path: Path):
    """Only workflows setting `engine.command` re-run the installer themselves."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  agent:
    steps:
      - name: Setup Scripts
        env:
          GH_AW_INFO_AWF_VERSION: "v0.27.44"
      - name: Run agent
        run: awf -- /bin/bash -c 'agent'
  detect:
    runs-on: ubuntu-latest
""",
    )

    assert check_awf_binary_version(lock) == []


def test_awf_binary_version_rejects_a_pin_with_no_declared_version_to_check(tmp_path: Path):
    """A pin gh-aw no longer declares a version for is an unchecked pin, not a passing one.

    Returning empty here is how the check would disarm itself if gh-aw ever renamed
    `GH_AW_INFO_AWF_VERSION` — and the next skew would then ship green, as #8041 did.
    """
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  agent:
    steps:
      - run: bash "${RUNNER_TEMP}/gh-aw/actions/install_awf_binary.sh" v0.27.42
""",
    )

    violations = check_awf_binary_version(lock)

    assert [v.check for v in violations] == ['awf-binary-version']
    assert 'GH_AW_INFO_AWF_VERSION' in violations[0].message


@pytest.mark.parametrize('schema_version', ['v0.28.31', 'v0.28.44'])
def test_awf_binary_version_compares_the_schema_when_info_is_empty(tmp_path: Path, schema_version: str):
    """Custom engines emit an empty info version; their actual config supplies the reference."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        f"""
jobs:
  agent:
    steps:
      - name: Setup Scripts
        env:
          GH_AW_INFO_AWF_VERSION: ""
      - name: Install AWF binary
        run: bash "${{RUNNER_TEMP}}/gh-aw/actions/install_awf_binary.sh" v0.28.31
      - name: Write firewall config
        run: >-
          printf '%s' '{{"$schema":"https://github.com/github/gh-aw-firewall/releases/download/{schema_version}/config.schema.json"}}'
""",
    )

    violations = check_awf_binary_version(lock)

    if schema_version == 'v0.28.31':
        assert violations == []
    else:
        assert [v.check for v in violations] == ['awf-binary-version']
        assert 'v0.28.31' in violations[0].message
        assert 'v0.28.44' in violations[0].message
        assert 'Recompile with a gh-aw release' in violations[0].message


# --- AI-credits accounting (the overlay that 403'd every agent job) -----------


@pytest.mark.parametrize('per_million', [False, True])
@pytest.mark.parametrize('stale_fallback', [False, True])
def test_ai_credits_accounting_checks_glm_overlay_units(tmp_path: Path, per_million: bool, stale_fallback: bool):
    """The same published rates use different units in the two pricing settings."""
    unit = 1 if per_million else 1_000_000
    config: dict[str, object] = {
        'apiProxy': {
            'defaultAiCreditsPricing': {'input': 0.6, 'cachedInput': 0.03, 'output': 2.4}
            if stale_fallback
            else {'input': 0.15, 'cachedInput': 0.03, 'output': 0.50},
            'providers': {
                'anthropic': {
                    'models': {
                        'glm-5.3-flash': {
                            'cost': {'input': 0.15 / unit, 'cache_read': 0.03 / unit, 'output': 0.50 / unit}
                        }
                    }
                }
            },
        }
    }
    workflow: dict[str, object] = {'jobs': {'agent': {'steps': [{'run': f"printf '%s' '{json.dumps(config)}'"}]}}}
    lock = _write(tmp_path / 'w.lock.yml', yaml.safe_dump(workflow))
    assert bool(check_ai_credits_accounting(lock)) is (per_million or stale_fallback)


def test_ai_credits_accounting_rejects_a_priced_model_in_the_awf_config(tmp_path: Path):
    """The shape that shipped: `models.providers` compiled into `apiProxy.providers`.

    AWF then charged `MiniMax-M3` against a 10,000-credit cap no frontmatter field can
    raise, and every agent job 403'd with `ai_credits_limit_exceeded` on its first call.
    """
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  agent:
    steps:
      - run: printf '%s' '{"apiProxy":{"enabled":true,"maxRuns":500,"providers":{"anthropic":{"models":{"MiniMax-M3":{"cost":{"input":0.6}}}}}}}'
""",
    )

    violations = check_ai_credits_accounting(lock)

    assert [v.check for v in violations] == ['ai-credits-accounting']
    assert '`models.providers`' in violations[0].message


def test_ai_credits_accounting_reads_the_escaped_nested_config(tmp_path: Path):
    """The detection job's config is escaped one level deeper than the agent's."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  detection:
    steps:
      - run: printf '%s' '{\\"apiProxy\\":{\\"maxRuns\\":500,\\"providers\\":{\\"anthropic\\":{}}}}'
""",
    )

    assert [v.check for v in check_ai_credits_accounting(lock)] == ['ai-credits-accounting']


def test_ai_credits_accounting_ignores_the_informational_model_costs_env(tmp_path: Path):
    """`GH_AW_INFO_MODEL_COSTS` carries the same overlay but configures nothing."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  agent:
    steps:
      - name: Setup Scripts
        env:
          GH_AW_INFO_MODEL_COSTS: '{"providers":{"anthropic":{"models":{"MiniMax-M3":{"cost":{"input":0.6}}}}}}'
      - run: printf '%s' '{"apiProxy":{"enabled":true,"maxRuns":500}}'
""",
    )

    assert check_ai_credits_accounting(lock) == []


# --- daily AI-credits guardrail (the artifact scan that drained `GITHUB_TOKEN`) -


def test_daily_aic_guardrail_rejects_the_compiled_default(tmp_path: Path):
    """The shape every lock carried before `max-daily-ai-credits: -1`, as gh-aw v0.86 compiles it."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  activation:
    env:
      GH_AW_MAX_DAILY_AI_CREDITS: ${{ vars.GH_AW_DEFAULT_MAX_DAILY_AI_CREDITS || '5000' }}
    steps:
      - name: Check daily workflow token guardrail
        id: daily-effective-workflow-guardrail
        if: ${{ env.GH_AW_MAX_DAILY_AI_CREDITS != '' }}
        uses: actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3 # v9.0.0
        with:
          script: |
            const { main } = require('${{ runner.temp }}/gh-aw/actions/check_daily_aic_workflow_guardrail.cjs');
            await main();
""",
    )

    violations = check_daily_aic_guardrail(lock)

    assert [v.check for v in violations] == ['daily-aic-guardrail']
    assert '`max-daily-ai-credits: -1`' in violations[0].message


def test_daily_aic_guardrail_rejects_the_renamed_step_after_a_compiler_upgrade(tmp_path: Path):
    """gh-aw v0.92.6 compiles the step under a new id; it still runs the same guardrail script."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  activation:
    steps:
      - name: Check daily workflow token guardrail
        id: daily-ai-credits-workflow-guardrail
        if: ${{ env.GH_AW_MAX_DAILY_AI_CREDITS != '' }}
        uses: actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3 # v9.0.0
        with:
          script: |
            const path = require('path');
            const actionsDir = path.join(process.env.RUNNER_TEMP, 'gh-aw', 'actions');
            const { main } = require(path.join(actionsDir, 'check_daily_aic_workflow_guardrail.cjs'));
            await main();
""",
    )

    assert [v.check for v in check_daily_aic_guardrail(lock)] == ['daily-aic-guardrail']


def test_daily_aic_guardrail_rejects_a_disable_set_only_in_an_import(tmp_path: Path):
    """An imported `-1` drops the threshold env but still compiles the step."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  activation:
    steps:
      - name: Check daily workflow token guardrail
        id: daily-effective-workflow-guardrail
        if: ${{ env.GH_AW_MAX_DAILY_AI_CREDITS != '' }}
        with:
          script: |
            const { main } = require('${{ runner.temp }}/gh-aw/actions/check_daily_aic_workflow_guardrail.cjs');
""",
    )

    assert [v.check for v in check_daily_aic_guardrail(lock)] == ['daily-aic-guardrail']


def test_daily_aic_guardrail_accepts_a_lock_without_the_step(tmp_path: Path):
    lock = _write(
        tmp_path / 'w.lock.yml',
        """
jobs:
  activation:
    steps:
      - name: Check workflow lock file
        id: check-lock-file
        with:
          script: |
            const { main } = require('${{ runner.temp }}/gh-aw/actions/check_workflow_timestamp_api.cjs');
""",
    )

    assert check_daily_aic_guardrail(lock) == []


# --- engine wrapper `sys.path` order (the pre-#12818 engine pin, #12815) --------


def _harness_lock(resolve_agent: str) -> str:
    """The harness step every lock compiles, around the wrapper's `PAI_AGENT` import branch."""
    return f"""
jobs:
  agent:
    steps:
      - name: Write Pydantic AI harness script
        run: |
          cat <<'GHAW_HARNESS_SCRIPT_3c7b9f1a_EOF' > "${{RUNNER_TEMP}}/gh-aw/actions/pydantic-ai_harness.cjs"
          const AGENT_MODULE = `import importlib
          import os
          import sys

          _configured_agent = os.environ.get('PAI_AGENT', '').strip()
          if _configured_agent:
{resolve_agent}
              agent = getattr(importlib.import_module(_module_name), _attribute)
          `;
          const DEFAULT_AGENT = "gh_aw_agent:agent";
          GHAW_HARNESS_SCRIPT_3c7b9f1a_EOF
"""


def test_engine_sys_path_order_rejects_the_pre_12818_wrapper(tmp_path: Path):
    """The wrapper every lock compiled from engine revision `764a75c329`."""
    lock = _write(
        tmp_path / 'w.lock.yml',
        _harness_lock(
            """\
              sys.path.insert(0, os.getcwd())
              _module_name, _separator, _attribute = _configured_agent.rpartition(':')"""
        ),
    )

    violations = check_engine_sys_path_order(lock)

    assert [v.check for v in violations] == ['engine-sys-path-order']
    assert '#12818' in violations[0].message


def test_engine_sys_path_order_rejects_the_pre_12818_wrapper_in_a_double_quoted_run(tmp_path: Path):
    """A double-quoted `run` escapes the wrapper's line breaks, so only the parsed step has statement lines."""
    run = (
        'cat <<\'GHAW_HARNESS_SCRIPT_EOF\' > "${RUNNER_TEMP}/gh-aw/actions/pydantic-ai_harness.cjs"\n'
        'const AGENT_MODULE = `import os\nimport sys\n\nsys.path.insert(0, os.getcwd())\n`;\n'
        'GHAW_HARNESS_SCRIPT_EOF\n'
    )
    lock = _write(tmp_path / 'w.lock.yml', f'jobs:\n  agent:\n    steps:\n      - run: {json.dumps(run)}\n')

    assert [v.check for v in check_engine_sys_path_order(lock)] == ['engine-sys-path-order']


def test_engine_sys_path_order_accepts_the_12818_wrapper(tmp_path: Path):
    lock = _write(
        tmp_path / 'w.lock.yml',
        _harness_lock(
            """\
              _module_name, _separator, _attribute = _configured_agent.rpartition(':')
              # The checkout joins sys.path, after installed packages, only when nothing installed has the target's
              # top-level name. sys.path.insert(0, ...) would put it first.
              _top_level = _module_name.partition('.')[0]
              if _top_level not in sys.modules and find_spec(_top_level) is None:
                  sys.path.append(os.getcwd())"""
        ),
    )

    assert check_engine_sys_path_order(lock) == []


def test_engine_sys_path_order_rejects_a_lock_without_the_wrapper(tmp_path: Path):
    """A renamed wrapper must fail the check rather than leave it passing on nothing."""
    lock = _write(tmp_path / 'w.lock.yml', _harness_lock('').replace('const AGENT_MODULE', 'const WRAPPER'))

    assert [v.check for v in check_engine_sys_path_order(lock)] == ['engine-sys-path-order']


def test_compiler_version_compatibility_rejects_a_blocked_version(tmp_path: Path):
    lock = _write(tmp_path / 'w.lock.yml', '# gh-aw-metadata: {"compiler_version":"v0.83.4"}\njobs: {}\n')

    violations = check_compiler_version_compatibility([lock], {'blockedVersions': ['v0.83.4']})

    assert [v.check for v in violations] == ['compiler-version-blocked']
    assert '`v0.83.4`' in violations[0].message


def test_compiler_version_compatibility_accepts_an_unblocked_version(tmp_path: Path):
    lock = _write(tmp_path / 'w.lock.yml', '# gh-aw-metadata: {"compiler_version":"v0.86.2"}\njobs: {}\n')

    assert check_compiler_version_compatibility([lock], {'blockedVersions': ['v0.83.4']}) == []


@pytest.fixture
def workflows_dir(tmp_path: Path) -> Path:
    """A minimal workflows tree: one agentic source importing one shared fragment."""
    workflows = tmp_path / '.github' / 'workflows'
    _write(
        workflows / 'pydantic-ai-sweep.md',
        '---\ntimeout-minutes: 30\nimports:\n  - shared/rigor.md\n---\nprompt\n',
    )
    _write(workflows / 'pydantic-ai-sweep.lock.yml', 'jobs: {}\n')
    _write(workflows / 'shared' / 'rigor.md', '---\nname: rigor\n---\nbody\n')
    return workflows


def test_lock_regenerated_flags_a_source_edited_without_its_lock(workflows_dir: Path):
    changed = [str(workflows_dir / 'pydantic-ai-sweep.md')]

    violations = check_lock_regenerated(changed, workflows_dir)

    assert [v.check for v in violations] == ['lock-not-regenerated']
    assert 'pydantic-ai-sweep.lock.yml' in violations[0].message


def test_lock_regenerated_accepts_a_source_and_lock_changed_together(workflows_dir: Path):
    changed = [
        str(workflows_dir / 'pydantic-ai-sweep.md'),
        str(workflows_dir / 'pydantic-ai-sweep.lock.yml'),
    ]

    assert check_lock_regenerated(changed, workflows_dir) == []


def test_lock_regenerated_follows_shared_imports(workflows_dir: Path):
    """A shared fragment is inlined at compile time, so its importers must recompile."""
    changed = [str(workflows_dir / 'shared' / 'rigor.md')]

    violations = check_lock_regenerated(changed, workflows_dir)

    assert [v.check for v in violations] == ['lock-not-regenerated']
    assert 'pydantic-ai-sweep.lock.yml' in violations[0].message


def test_lock_regenerated_flags_a_deleted_source_with_an_orphaned_lock(workflows_dir: Path):
    """Actions runs the lock, so a lock outliving its source keeps running with no source."""
    source = workflows_dir / 'pydantic-ai-gone.md'
    changed = [str(source)]  # named in the changeset but absent from disk == deleted

    violations = check_lock_regenerated(changed, workflows_dir)

    assert [v.check for v in violations] == ['lock-not-regenerated']
    assert 'was deleted' in violations[0].message


def test_lock_regenerated_accepts_a_source_and_lock_deleted_together(workflows_dir: Path):
    changed = [str(workflows_dir / 'pydantic-ai-gone.md'), str(workflows_dir / 'pydantic-ai-gone.lock.yml')]

    assert check_lock_regenerated(changed, workflows_dir) == []


def test_lock_regenerated_accepts_a_source_renamed_with_its_lock(workflows_dir: Path):
    """A legitimate rename must not trip the orphan check.

    `ci.yml` feeds `previous_filename` through for renamed entries, so the old source
    appears in the changeset alongside the new one. The old lock is there too — as a
    `renamed` entry's `previous_filename`, or as a `removed` entry's `filename` — so
    the pair still reconciles and nothing is flagged.
    """
    _write(workflows_dir / 'pydantic-ai-bar.md', '---\ntimeout-minutes: 30\n---\nprompt\n')
    _write(workflows_dir / 'pydantic-ai-bar.lock.yml', 'jobs: {}\n')
    changed = [
        str(workflows_dir / 'pydantic-ai-bar.md'),
        str(workflows_dir / 'pydantic-ai-foo.md'),  # previous_filename of the renamed source
        str(workflows_dir / 'pydantic-ai-bar.lock.yml'),
        str(workflows_dir / 'pydantic-ai-foo.lock.yml'),  # old lock: renamed away or removed
    ]

    assert check_lock_regenerated(changed, workflows_dir) == []


def test_lock_regenerated_flags_a_rename_that_strands_the_old_lock(workflows_dir: Path):
    """Renaming the source but leaving the old lock in the repo is the orphan case.

    Actions keeps running `pydantic-ai-foo.lock.yml` even though no source produces it.
    """
    _write(workflows_dir / 'pydantic-ai-bar.md', '---\ntimeout-minutes: 30\n---\nprompt\n')
    _write(workflows_dir / 'pydantic-ai-bar.lock.yml', 'jobs: {}\n')
    changed = [
        str(workflows_dir / 'pydantic-ai-bar.md'),
        str(workflows_dir / 'pydantic-ai-foo.md'),
        str(workflows_dir / 'pydantic-ai-bar.lock.yml'),
    ]

    violations = check_lock_regenerated(changed, workflows_dir)

    assert [v.check for v in violations] == ['lock-not-regenerated']
    assert 'pydantic-ai-foo.lock.yml' in violations[0].message


def test_lock_regenerated_ignores_an_unimported_shared_fragment(workflows_dir: Path):
    _write(workflows_dir / 'shared' / 'unused.md', '---\nname: unused\n---\nbody\n')

    assert check_lock_regenerated([str(workflows_dir / 'shared' / 'unused.md')], workflows_dir) == []


# --- rendering, parsing edge cases, and the CLI -------------------------------


def test_violation_renders_as_one_line():

    assert str(Violation('a/b.yml', 'some-check', 'went wrong')) == 'a/b.yml: [some-check] went wrong'


def test_prompt_paths_handles_a_file_without_frontmatter(tmp_path: Path):
    """A bare markdown file is all prompt, so the whole file is scanned."""
    source = _write(tmp_path / 'plain.md', 'read /tmp/gh-aw/.review-context/x\n')

    assert [v.check for v in check_prompt_paths(source)] == ['prompt-path-outside-workspace']


def test_compiler_versions_skips_locks_without_parseable_metadata(tmp_path: Path):
    """A missing or malformed metadata header is not drift; other checks cover those."""
    missing = _write(tmp_path / 'a.lock.yml', 'jobs: {}\n')
    malformed = _write(tmp_path / 'b.lock.yml', '# gh-aw-metadata: {not json\njobs: {}\n')
    valid = _write(tmp_path / 'c.lock.yml', '# gh-aw-metadata: {"compiler_version":"v0.83.4"}\njobs: {}\n')

    assert check_compiler_versions([missing, malformed, valid]) == []


@pytest.mark.parametrize('payload', ['"just-a-string"', 'null', '[1, 2]'])
def test_compiler_versions_survives_valid_json_that_is_not_an_object(tmp_path: Path, payload: str):
    """Reaching `.get` on a non-object would take the whole lint job down with a traceback."""
    lock = _write(tmp_path / 'a.lock.yml', f'# gh-aw-metadata: {payload}\njobs: {{}}\n')
    valid = _write(tmp_path / 'b.lock.yml', '# gh-aw-metadata: {"compiler_version":"v0.83.4"}\njobs: {}\n')

    violations = check_compiler_versions([lock, valid])

    assert [v.check for v in violations] == ['compiler-version-drift']


def test_main_reads_a_changed_file_list_one_path_per_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Splitting on whitespace would tear any path containing a space into two."""
    listing = tmp_path / 'changed.txt'
    listing.write_text('.github/workflows/pydantic-ai-my workflow.md\n\n.github/workflows/other.md\n')
    seen: list[list[str] | None] = []

    def record(changed: list[str] | None = None) -> list[Violation]:
        seen.append(changed)
        return []

    monkeypatch.setattr(agentic_workflow_guard, 'run_checks', record)

    assert agentic_workflow_guard.main(['check', '--changed-file-list', str(listing)]) == 0
    assert seen == [['.github/workflows/pydantic-ai-my workflow.md', '.github/workflows/other.md']]


def test_main_checks_versions_against_a_compatibility_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    policy = _write(tmp_path / 'compat.json', '{"blockedVersions":["v0.83.4"]}\n')
    seen: list[dict[str, object] | None] = []

    def record(
        workflows_dir: Path = WORKFLOWS_DIR,
        changed: list[str] | None = None,
        compatibility: dict[str, object] | None = None,
    ) -> list[Violation]:
        seen.append(compatibility)
        return []

    monkeypatch.setattr(agentic_workflow_guard, 'run_checks', record)

    assert agentic_workflow_guard.main(['check', '--compatibility-file', str(policy)]) == 0
    assert seen == [{'blockedVersions': ['v0.83.4']}]


def test_changed_files_returns_empty_for_an_unresolvable_ref():

    assert changed_files('definitely-not-a-ref-8f3a2b') == []


def test_main_reports_success_on_a_clean_tree(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):

    def no_violations(workflows_dir: Path = WORKFLOWS_DIR, changed: list[str] | None = None) -> list[Violation]:
        return []

    monkeypatch.setattr(agentic_workflow_guard, 'run_checks', no_violations)

    assert agentic_workflow_guard.main(['check']) == 0
    assert 'passed' in capsys.readouterr().out


def test_main_exits_nonzero_and_prints_each_violation(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):

    def one_violation(workflows_dir: Path = WORKFLOWS_DIR, changed: list[str] | None = None) -> list[Violation]:
        assert changed == ['x.md'], '--base-ref must feed the lock-freshness check'
        return [Violation('w.lock.yml', 'dangling-needs', 'boom')]

    def fake_diff(base_ref: str) -> list[str]:
        return ['x.md']

    monkeypatch.setattr(agentic_workflow_guard, 'changed_files', fake_diff)
    monkeypatch.setattr(agentic_workflow_guard, 'run_checks', one_violation)

    assert agentic_workflow_guard.main(['check', '--base-ref', 'origin/main']) == 1
    err = capsys.readouterr().err
    assert 'w.lock.yml: [dangling-needs] boom' in err
    assert '1 agentic-workflow policy violation(s).' in err


def test_zai_workflow_requires_logfire_import(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    source = _write(
        workflows / 'pydantic-ai-attention-triage.md',
        '---\nimports:\n  - shared/engine-zai.md\n---\nPrompt\n',
    )
    lock = _write(source.with_suffix('.lock.yml'), 'env: {}\n')

    assert {violation.check for violation in check_logfire_wiring(workflows)} == {
        'logfire-import',
        'logfire-compiled',
        'logfire-network',
    }

    _write(
        source,
        source.read_text(encoding='utf-8').replace(
            '  - shared/engine-zai.md', '  - shared/engine-zai.md\n  - shared/otel-logfire.md'
        ),
    )
    compiled_lock = """env:
  OTEL_EXPORTER_OTLP_ENDPOINT: https://logfire-api.pydantic.dev
  OTEL_EXPORTER_OTLP_HEADERS: Authorization=${{ secrets.LOGFIRE_TOKEN }}
  GH_AW_OTLP_ENDPOINTS: '[{"url":"https://logfire-api.pydantic.dev"}]'
jobs:
  agent:
    steps:
      - env:
          GH_AW_ALLOWED_DOMAINS: logfire-api.pydantic.dev,logfire-eu.pydantic.dev,logfire-us.pydantic.dev,logfire-api.pydantic.info,logfire-eu.pydantic.info,logfire-us.pydantic.info
"""
    _write(lock, compiled_lock)
    assert check_logfire_wiring(workflows) == []

    _write(lock, compiled_lock.replace(',logfire-us.pydantic.info', ''))
    assert [violation.check for violation in check_logfire_wiring(workflows)] == ['logfire-network']

    _write(lock, 'env:\n  OTEL_EXPORTER_OTLP_ENDPOINT: https://logfire-api.pydantic.dev\n')
    assert {violation.check for violation in check_logfire_wiring(workflows)} == {
        'logfire-compiled',
        'logfire-network',
    }


# --- the live repository ------------------------------------------------------


def test_repository_agentic_workflows_satisfy_policy():
    """The checked-in workflows must pass every check.

    This is the test that actually gates PRs; the cases above only prove each
    check detects the defect it was written for.
    """
    violations = run_checks(WORKFLOWS_DIR)

    assert violations == [], 'agentic workflow policy violations:\n' + '\n'.join(str(v) for v in violations)


def test_run_checks_scans_shared_fragments_under_the_given_root(tmp_path: Path):
    """`shared/` must resolve under the caller's root, not the module global.

    Otherwise a custom `workflows_dir` silently skips its own shared fragments while
    scanning whatever happens to sit under the process working directory.
    """
    workflows = tmp_path / '.github' / 'workflows'
    _write(
        workflows / 'pydantic-ai-x.md',
        '---\ntimeout-minutes: 30\nenv:\n  PYDANTIC_AI_JOB_TIMEOUT_MINUTES: "30"\n---\nprompt\n',
    )
    _write(
        workflows / 'pydantic-ai-x.lock.yml',
        """
jobs:
  agent:
    steps:
      - run: >-
          awf -- /bin/bash -c '/tmp/gh-aw/bin/pydantic-ai-runner-launch
          --allowed-tools "Read,mcp__safeoutputs"
          --output-format stream-json
          --mcp-config mcp-servers.json'
""",
    )
    _write(workflows / 'shared' / 'ctx.md', '---\nname: ctx\n---\nRead /tmp/gh-aw/.review-context/x\n')

    violations = run_checks(workflows)

    assert 'prompt-path-outside-workspace' in {violation.check for violation in violations}


def test_provider_engine_config_rejects_drift_from_shared_values(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    shared = workflows / 'shared'
    engine = _write(
        shared / 'engine-zai.md',
        """---
engine:
  env:
    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
safe-outputs:
  threat-detection:
    engine:
      env:
        ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
        ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
---
""",
    )
    _write(shared / 'provider-health.md', '---\n# provider health\n---\n')
    local = _write(
        workflows / 'pydantic-ai-local.md',
        '---\nengine:\n  env:\n    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic\n'
        '    ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}\n---\nPrompt\n',
    )

    alternate_endpoint = 'https://provider.example/api/anthropic'
    alternate_credential = '${{ secrets.CURRENT_PROVIDER_API_KEY }}'
    _write(
        engine,
        engine.read_text(encoding='utf-8')
        .replace('https://api.z.ai/api/anthropic', alternate_endpoint)
        .replace('${{ secrets.ZAI_API_KEY }}', alternate_credential),
    )
    _write(
        local,
        local.read_text(encoding='utf-8')
        .replace('https://api.z.ai/api/anthropic', alternate_endpoint)
        .replace('${{ secrets.ZAI_API_KEY }}', alternate_credential),
    )

    assert check_provider_engine_config(workflows) == []

    _write(
        engine,
        engine.read_text(encoding='utf-8').replace(
            'safe-outputs:\n  threat-detection:\n    engine:\n      env:\n        ANTHROPIC_BASE_URL: '
            + alternate_endpoint,
            'safe-outputs:\n  threat-detection:\n    engine:\n      env:\n        ANTHROPIC_BASE_URL: https://provider.example/anthropic',
            1,
        ),
    )
    assert 'provider-engine-config' in {violation.check for violation in check_provider_engine_config(workflows)}

    _write(
        engine,
        engine.read_text(encoding='utf-8').replace(
            'safe-outputs:\n  threat-detection:\n    engine:\n      env:\n        ANTHROPIC_BASE_URL: https://provider.example/anthropic',
            'safe-outputs:\n  threat-detection:\n    engine:\n      env:\n        ANTHROPIC_BASE_URL: '
            + alternate_endpoint,
            1,
        ),
    )
    _write(local, local.read_text(encoding='utf-8').replace(alternate_credential, '${{ secrets.OTHER_API_KEY }}'))
    assert 'provider-engine-config' in {violation.check for violation in check_provider_engine_config(workflows)}

    _write(local, local.read_text(encoding='utf-8').replace('${{ secrets.OTHER_API_KEY }}', alternate_credential))
    _write(
        local,
        local.read_text(encoding='utf-8').replace(alternate_endpoint, 'https://provider.example/anthropic'),
    )
    assert 'provider-engine-config' in {violation.check for violation in check_provider_engine_config(workflows)}

    _write(
        engine,
        engine.read_text(encoding='utf-8').replace(
            'ANTHROPIC_API_KEY: ' + alternate_credential,
            'ANTHROPIC_API_KEY: ""',
            1,
        ),
    )
    assert 'provider-engine-config' in {violation.check for violation in check_provider_engine_config(workflows)}

    _write(
        engine,
        engine.read_text(encoding='utf-8').replace(
            'ANTHROPIC_BASE_URL: ' + alternate_endpoint, 'ANTHROPIC_BASE_URL: ""', 1
        ),
    )
    assert 'provider-engine-config' in {violation.check for violation in check_provider_engine_config(workflows)}


def test_provider_engine_config_checks_workflow_local_threat_detection(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    shared = workflows / 'shared'
    _write(
        shared / 'engine-zai.md',
        """---
engine:
  env:
    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
safe-outputs:
  threat-detection:
    engine:
      env:
        ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
        ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
---
""",
    )
    _write(
        workflows / 'pydantic-ai-local.md',
        """---
safe-outputs:
  threat-detection:
    engine:
      env:
        ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
        ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
---
Prompt
""",
    )

    assert check_provider_engine_config(workflows) == []

    local = workflows / 'pydantic-ai-local.md'
    matching_config = local.read_text(encoding='utf-8')
    _write(
        local,
        matching_config.replace('https://api.z.ai/api/anthropic', 'https://provider.example/api/anthropic'),
    )
    assert 'provider-engine-config' in {violation.check for violation in check_provider_engine_config(workflows)}

    _write(local, matching_config.replace('secrets.ZAI_API_KEY', 'secrets.OTHER_API_KEY'))
    assert 'provider-engine-config' in {violation.check for violation in check_provider_engine_config(workflows)}


def test_provider_engine_config_accepts_a_disabled_shared_detection_engine(tmp_path: Path):
    """The shared OpenCode detector replaces gh-aw's detection engine, so it has no provider endpoint."""
    workflows = tmp_path / '.github' / 'workflows'
    engine = _write(
        workflows / 'shared' / 'engine-zai.md',
        """---
engine:
  env:
    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
safe-outputs:
  threat-detection:
    engine: false
---
""",
    )
    assert check_provider_engine_config(workflows) == []

    _write(engine, engine.read_text(encoding='utf-8').replace('engine: false', 'engine:\n      id: claude'))
    assert 'provider-engine-config' in {violation.check for violation in check_provider_engine_config(workflows)}


def test_provider_health_wiring_requires_the_shared_gate_and_compiled_activation_dependency(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    source = _write(
        workflows / 'pydantic-ai-mini.md',
        '---\nname: Z.AI\nimports:\n  - shared/engine-zai.md\nif: true\n---\nPrompt\n',
    )
    lock = _write(
        source.with_suffix('.lock.yml'),
        """
jobs:
  activation:
    needs: [pre_activation]
  provider_health:
    needs: [activation]
""",
    )

    violations = check_provider_health_wiring(workflows)

    assert {violation.check for violation in violations} == {
        'provider-health-import',
        'provider-health-prompt',
        'provider-health-gate',
        'provider-health-reporting',
        'provider-health-activation-needs',
        'provider-health-activation-if',
        'provider-health-job-order',
    }
    assert all(violation.path in (str(source), str(lock)) for violation in violations)


def test_provider_health_recovery_wiring_requires_incident_bound_dispatch(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    source = _write(
        workflows / 'pydantic-ai-mini.md',
        """---
name: Z.AI
imports:
  - shared/engine-zai.md
on:
  workflow_dispatch:
    inputs:
      recovery_issue:
        type: string
        required: false
---
Prompt
""",
    )
    health = _write(
        workflows / 'shared' / 'provider-health.md',
        "job:\n  env:\n    PYDANTIC_AI_RECOVERY_ISSUE: ${{ inputs.recovery_issue || '' }}\n",
    )

    assert check_provider_health_recovery_wiring(workflows) == []

    source.write_text(
        source.read_text(encoding='utf-8').replace('required: false', 'required: true', 1), encoding='utf-8'
    )
    health.write_text('job: {}\n', encoding='utf-8')
    assert {violation.path for violation in check_provider_health_recovery_wiring(workflows)} == {
        str(source),
        str(health),
    }


def test_provider_health_recovery_input_must_be_an_optional_string(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    _write(
        workflows / 'shared' / 'provider-health.md',
        "job:\n  env:\n    PYDANTIC_AI_RECOVERY_ISSUE: ${{ inputs.recovery_issue || '' }}\n",
    )
    ordinary = _write(
        workflows / 'pydantic-ai-mini.md',
        """---
name: Z.AI
imports:
  - shared/engine-zai.md
on:
  workflow_dispatch: {}
---
Prompt
""",
    )
    violations = check_provider_health_recovery_wiring(workflows)
    assert [(violation.path, violation.check) for violation in violations] == [
        (str(ordinary), 'provider-health-recovery-input')
    ]

    ordinary.write_text(
        ordinary.read_text(encoding='utf-8').replace(
            'workflow_dispatch: {}',
            'workflow_dispatch:\n    inputs:\n      recovery_issue:\n        type: number\n        required: false',
            1,
        ),
        encoding='utf-8',
    )
    violations = check_provider_health_recovery_wiring(workflows)
    assert [(violation.path, violation.check) for violation in violations] == [
        (str(ordinary), 'provider-health-recovery-input')
    ]


_GATE_ENV: dict[str, str] = {
    'GITHUB_TOKEN': '${{ github.token }}',
    'ZAI_API_KEY': '${{ secrets.ZAI_API_KEY }}',
    'ZAI_BACKUP_API_KEY': '${{ secrets.ZAI_BACKUP_API_KEY }}',
    'GITHUB_WORKFLOW': '${{ github.workflow }}',
    'PYDANTIC_AI_RUN_ATTEMPT': '${{ github.run_attempt }}',
    'PYDANTIC_AI_TASK_KEY': '${{ github.workflow }}:${{ github.event_name }}',
}
_RECHECK_RUN = (
    "gh api -H 'Accept: application/vnd.github.raw+json' "
    '"repos/${GITHUB_REPOSITORY}/contents/.github/scripts/agent_provider_health.py" '
    '> "$RUNNER_TEMP/provider-health/agent_provider_health.py"\n'
    'python3 "$RUNNER_TEMP/provider-health/agent_provider_health.py" check '
    '--output "$RUNNER_TEMP/provider-health/provider-health.json"\n'
)
# The `CI Review` agent job's step order before agent-start rechecks, trimmed.
_SHIPPED_AGENT_STEPS: list[dict[str, object]] = [
    {'name': 'Restore inline skills from activation artifact', 'run': 'bash restore_inline_skills.sh'},
    {'name': 'Stage Pydantic AI gh-aw shim launcher', 'run': 'install -m 755 launcher /tmp/gh-aw/bin/'},
    {'name': 'Execute Claude Code CLI', 'id': 'agentic_execution', 'run': 'awf -- claude_harness.cjs'},
]


def _agent_start_steps(
    *,
    run: str = _RECHECK_RUN,
    env: dict[str, str] = _GATE_ENV,
    stop_continues: bool = False,
    stop_name: str = AGENT_START_HEALTH_STEPS[2],
    recheck_id: str = 'agent_start_health',
) -> list[dict[str, object]]:
    stop: dict[str, object] = {
        'name': stop_name,
        'if': f"steps.{recheck_id}.outputs.ready != 'true'",
        'run': 'echo "::error::blocked"\nexit 1\n',
    }
    if stop_continues:
        stop['continue-on-error'] = True
    return [
        {'name': AGENT_START_HEALTH_STEPS[0], 'id': recheck_id, 'env': env, 'run': run},
        {
            'name': AGENT_START_HEALTH_STEPS[1],
            'id': 'agent_start_health_artifact',
            'uses': 'actions/upload-artifact',
            'continue-on-error': True,
        },
        stop,
        {
            'name': AGENT_START_HEALTH_STEPS[3],
            'if': f"steps.agent_start_health_artifact.outcome == 'failure' && steps.{recheck_id}.outputs.account "
            '!= needs.provider_health.outputs.account',
            'run': 'echo "::error::hidden account"\nexit 1\n',
        },
    ]


def _write_agent_start_workflow(
    workflows: Path, agent_steps: list[dict[str, object]], gate_env: dict[str, str] = _GATE_ENV
) -> Path:
    source = _write(
        workflows / 'pydantic-ai-mini.md', '---\nname: Z.AI\nimports:\n  - shared/engine-zai.md\n---\nPrompt\n'
    )
    jobs: dict[str, object] = {
        'provider_health': {
            'outputs': {'account': '${{ steps.health.outputs.account }}'},
            'steps': [{'id': 'health', 'env': gate_env, 'run': 'python3 controller check'}],
        },
        'agent': {'needs': ['activation', 'provider_health'], 'steps': agent_steps},
    }
    return _write(source.with_suffix('.lock.yml'), yaml.safe_dump({'jobs': jobs}, sort_keys=False))


def test_provider_health_agent_start_catches_the_queued_ci_review_defect(tmp_path: Path):
    """An agent job that waited for its slot ran on a gate decision from workflow start.

    Run 38967083667 passed its gate at 2026-10-16 21:07Z and started inference at
    03:13Z the next day, after provider incident #10572 had opened.
    """
    workflows = tmp_path / '.github' / 'workflows'
    lock = _write_agent_start_workflow(workflows, _SHIPPED_AGENT_STEPS)

    assert [(violation.path, violation.check) for violation in check_provider_health_agent_start(workflows)] == [
        (str(lock), 'provider-health-agent-start')
    ]

    _write_agent_start_workflow(workflows, [_SHIPPED_AGENT_STEPS[0], *_agent_start_steps(), *_SHIPPED_AGENT_STEPS[1:]])
    assert check_provider_health_agent_start(workflows) == []


@pytest.mark.parametrize(
    'agent_steps',
    [
        pytest.param([*_SHIPPED_AGENT_STEPS, *_agent_start_steps()], id='after-engine-execution'),
        pytest.param(
            [*_agent_start_steps(run='python3 .github/scripts/agent_provider_health.py check'), *_SHIPPED_AGENT_STEPS],
            id='workspace-controller',
        ),
        pytest.param(
            [
                *_agent_start_steps(
                    run=_RECHECK_RUN.replace(
                        'python3 "$RUNNER_TEMP/provider-health/agent_provider_health.py"',
                        'python3 .github/scripts/agent_provider_health.py',
                        1,
                    )
                ),
                *_SHIPPED_AGENT_STEPS,
            ],
            id='fetched-but-runs-workspace-controller',
        ),
        pytest.param(
            [
                *_agent_start_steps(
                    run=_RECHECK_RUN.replace('agent_provider_health.py"', 'agent_provider_health.py?ref=v1"', 1)
                ),
                *_SHIPPED_AGENT_STEPS,
            ],
            id='pinned-controller-ref',
        ),
        pytest.param(
            [
                *_agent_start_steps(env={key: value for key, value in _GATE_ENV.items() if key != 'ZAI_API_KEY'}),
                *_SHIPPED_AGENT_STEPS,
            ],
            id='env-differs-from-gate',
        ),
        pytest.param([*_agent_start_steps(stop_continues=True), *_SHIPPED_AGENT_STEPS], id='stop-continues'),
        pytest.param([*_agent_start_steps()[:2], *_SHIPPED_AGENT_STEPS], id='no-stop'),
        pytest.param(
            [
                *_agent_start_steps(stop_name='Stop before inference when provider health is blocked'),
                *_SHIPPED_AGENT_STEPS,
            ],
            id='renamed-stop-step',
        ),
        pytest.param(
            [*_agent_start_steps(recheck_id='provider_recheck'), *_SHIPPED_AGENT_STEPS],
            id='recheck-id-the-engine-key-does-not-read',
        ),
        pytest.param([*_agent_start_steps()[:3], *_SHIPPED_AGENT_STEPS], id='hidden-account-runs'),
        pytest.param(
            [*_agent_start_steps()[:3], {**_agent_start_steps()[3], 'continue-on-error': True}, *_SHIPPED_AGENT_STEPS],
            id='hidden-account-stop-continues',
        ),
    ],
)
def test_provider_health_agent_start_rejects_a_recheck_that_cannot_stop_inference(
    tmp_path: Path, agent_steps: list[dict[str, object]]
):
    workflows = tmp_path / '.github' / 'workflows'
    _write_agent_start_workflow(workflows, agent_steps)

    assert [violation.check for violation in check_provider_health_agent_start(workflows)] == [
        'provider-health-agent-start'
    ]


@pytest.mark.parametrize('key', ['ZAI_API_KEY', 'ZAI_BACKUP_API_KEY'])
def test_provider_health_agent_start_requires_both_keys_in_the_gate(tmp_path: Path, key: str):
    """A gate and recheck without a key cannot select that account, even when their environments agree."""
    workflows = tmp_path / '.github' / 'workflows'
    env = {name: value for name, value in _GATE_ENV.items() if name != key}
    _write_agent_start_workflow(workflows, [*_agent_start_steps(env=env), *_SHIPPED_AGENT_STEPS], gate_env=env)

    assert [violation.check for violation in check_provider_health_agent_start(workflows)] == [
        'provider-health-agent-start'
    ]


def test_provider_health_agent_start_requires_the_gate_account_output(tmp_path: Path):
    """Without the gate's account, the job cannot tell that a failed replacement hides the recheck's account."""
    workflows = tmp_path / '.github' / 'workflows'
    lock = _write_agent_start_workflow(workflows, [*_agent_start_steps(), *_SHIPPED_AGENT_STEPS])
    workflow = yaml.safe_load(lock.read_text(encoding='utf-8'))
    del workflow['jobs']['provider_health']['outputs']
    _write(lock, yaml.safe_dump(workflow, sort_keys=False))

    assert [violation.check for violation in check_provider_health_agent_start(workflows)] == [
        'provider-health-agent-start'
    ]


_RECORD_ENV: dict[str, str] = {key: value for key, value in _GATE_ENV.items() if key != 'GITHUB_TOKEN'}
_RECORD_RUN = (
    'python3 -I .github/scripts/agent_provider_health.py record \\\n'
    '  --agent-session agent/agent-session.jsonl \\\n'
    '  --health-artifact provider-health/provider-health.json \\\n'
    '  --output provider-health/run-result.json\n'
)
# The authored condition inside gh-aw's clean-detection guard, as the compiled lock carries it.
_AFTER_AGENT = (
    "${{ needs.detection.result == 'success' && needs.detection.outputs.detection_success == 'true' && "
    "((!cancelled()) && needs.agent.result != 'skipped') }}"
)
_CONTROLLER_CHECKOUT: dict[str, object] = {
    'persist-credentials': False,
    'ref': '${{ github.event.repository.default_branch }}',
    'repository': '${{ github.repository }}',
    'sparse-checkout': '.github/scripts/agent_provider_health.py',
}
_AGENT_DOWNLOAD: dict[str, str] = {'name': 'agent', 'path': 'agent'}
_RECORD_UPLOAD_PATH = 'provider-health/provider-health.json\nprovider-health/run-result.json\n'


def _host_result_job(
    *,
    needs: tuple[str, ...] = ('agent', 'detection'),
    condition: str = _AFTER_AGENT,
    job_fields: dict[str, object] | None = None,
    continue_on_error: bool = True,
    checkouts: tuple[dict[str, object], ...] = (_CONTROLLER_CHECKOUT,),
    download_id: str = 'agent_artifact',
    download: dict[str, str] = _AGENT_DOWNLOAD,
    download_env: dict[str, str] | None = None,
    run: str = _RECORD_RUN,
    record_fields: dict[str, object] | None = None,
    env: dict[str, str] = _RECORD_ENV,
    upload_path: str = _RECORD_UPLOAD_PATH,
    record_first: bool = False,
    upload_first: bool = False,
) -> dict[str, object]:
    """The record job as gh-aw compiles it from `shared/provider-health.md`, trimmed, with one part changed."""
    download_step: dict[str, object] = {
        'name': "Download the agent's terminal session record",
        'id': download_id,
        'uses': 'actions/download-artifact@sha',
        'with': download,
    }
    if download_env is not None:
        download_step['env'] = download_env
    downloads: list[dict[str, object]] = [
        download_step,
        {
            'name': 'Download the health decision the run acted on',
            'id': 'health_artifact',
            'uses': 'actions/download-artifact@sha',
            'with': {'name': 'provider-health', 'path': 'provider-health'},
        },
    ]
    record: dict[str, object] = {
        'name': 'Record the terminal agent result',
        'if': "steps.agent_artifact.outcome == 'success' && steps.health_artifact.outcome == 'success'",
        'run': run,
        'env': env,
        **(record_fields or {}),
    }
    upload: dict[str, object] = {
        'name': 'Add the host-recorded agent result to the health decision artifact',
        'uses': 'actions/upload-artifact@sha',
        'with': {'name': 'provider-health', 'overwrite': True, 'path': upload_path},
    }
    if record_first:
        after_checkout = [record, *downloads, upload]
    else:
        after_checkout = [*downloads, *([upload, record] if upload_first else [record, upload])]
    return {
        'needs': list(needs),
        'if': condition,
        'continue-on-error': continue_on_error,
        'steps': [*({'uses': 'actions/checkout@sha', 'with': checkout} for checkout in checkouts), *after_checkout],
        **(job_fields or {}),
    }


def _write_host_result_workflow(workflows: Path, record_job: dict[str, object] | None) -> None:
    source = _write(
        workflows / 'pydantic-ai-mini.md', '---\nname: Z.AI\nimports:\n  - shared/engine-zai.md\n---\nPrompt\n'
    )
    jobs: dict[str, object] = {
        'provider_health': {'steps': [{'id': 'health', 'env': _GATE_ENV, 'run': 'python3 controller check'}]},
        'agent': {'needs': ['activation', 'provider_health'], 'steps': _SHIPPED_AGENT_STEPS},
        'detection': {'needs': ['agent']},
    }
    if record_job is not None:
        jobs[HOST_RESULT_JOB] = record_job
    _write(source.with_suffix('.lock.yml'), yaml.safe_dump({'jobs': jobs}, sort_keys=False))


def test_provider_health_host_result_accepts_the_shipped_record_job(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    _write_host_result_workflow(workflows, _host_result_job())

    assert check_provider_health_host_result(workflows) == []


@pytest.mark.parametrize(
    'record_job',
    [
        pytest.param(None, id='no-record-job'),
        pytest.param(_host_result_job(needs=('detection',)), id='not-after-agent'),
        pytest.param(
            _host_result_job(condition=_AFTER_AGENT.replace("!= 'skipped'", "== 'success'")),
            id='skipped-after-a-failed-agent',
        ),
        pytest.param(
            _host_result_job(
                condition="${{ needs.detection.result == 'success' && needs.detection.outputs.detection_success == "
                "'true' && (!((!cancelled()) && needs.agent.result != 'skipped')) }}"
            ),
            id='negated-condition',
        ),
        pytest.param(_host_result_job(continue_on_error=False), id='record-failure-fails-the-run'),
        pytest.param(
            _host_result_job(checkouts=({**_CONTROLLER_CHECKOUT, 'ref': '${{ github.event.pull_request.head.sha }}'},)),
            id='pr-head-controller',
        ),
        pytest.param(
            _host_result_job(checkouts=({**_CONTROLLER_CHECKOUT, 'repository': 'someone/fork'},)),
            id='other-repository-controller',
        ),
        pytest.param(
            _host_result_job(checkouts=({**_CONTROLLER_CHECKOUT, 'persist-credentials': True},)),
            id='checkout-keeps-credentials',
        ),
        pytest.param(
            _host_result_job(checkouts=({**_CONTROLLER_CHECKOUT, 'sparse-checkout': '.github/scripts'},)),
            id='checkout-beyond-the-controller',
        ),
        pytest.param(
            _host_result_job(
                checkouts=(
                    _CONTROLLER_CHECKOUT,
                    {'ref': '${{ github.event.pull_request.head.sha }}', 'persist-credentials': False},
                )
            ),
            id='second-checkout',
        ),
        pytest.param(
            _host_result_job(download={'name': 'agent-output-fallback', 'path': 'agent'}), id='other-artifact'
        ),
        pytest.param(_host_result_job(download={**_AGENT_DOWNLOAD, 'run-id': '${{ inputs.run_id }}'}), id='other-run'),
        pytest.param(_host_result_job(download_id='session_artifact'), id='download-the-record-step-does-not-check'),
        pytest.param(_host_result_job(record_first=True), id='record-before-download'),
        pytest.param(
            _host_result_job(run=_RECORD_RUN.replace('agent/agent-session.jsonl', 'agent/agent-stdio.log')),
            id='reads-the-stdio-log',
        ),
        pytest.param(_host_result_job(run=_RECORD_RUN.replace('python3 -I ', 'python3 ', 1)), id='not-isolated'),
        pytest.param(
            _host_result_job(run=f'cp agent/agent_provider_health.py .github/scripts/\n{_RECORD_RUN}'),
            id='command-before-record',
        ),
        pytest.param(_host_result_job(record_fields={'if': 'false'}), id='record-step-disabled'),
        pytest.param(_host_result_job(record_fields={'working-directory': 'agent'}), id='record-in-another-directory'),
        pytest.param(
            _host_result_job(env={key: value for key, value in _RECORD_ENV.items() if key != 'ZAI_API_KEY'}),
            id='no-zai-key',
        ),
        pytest.param(
            _host_result_job(env={key: value for key, value in _RECORD_ENV.items() if key != 'ZAI_BACKUP_API_KEY'}),
            id='no-backup-key',
        ),
        pytest.param(
            _host_result_job(
                run=_RECORD_RUN.replace('  --health-artifact provider-health/provider-health.json \\\n', '')
            ),
            id='record-without-the-engine-account',
        ),
        pytest.param(
            _host_result_job(env={**_RECORD_ENV, 'PYDANTIC_AI_TASK_KEY': '${{ github.workflow }}'}),
            id='identity-differs-from-gate',
        ),
        pytest.param(
            _host_result_job(download_env={'ZAI_API_KEY': '${{ secrets.ZAI_API_KEY }}'}), id='key-in-another-step'
        ),
        pytest.param(
            _host_result_job(download_env={'KEY': "${{ secrets['ZAI_API_KEY'] }}"}), id='bracket-key-in-another-step'
        ),
        pytest.param(
            _host_result_job(download_env={'KEYS': '${{ toJSON(secrets) }}'}), id='all-secrets-in-another-step'
        ),
        pytest.param(_host_result_job(download_env={'KEY': '${{ secrets.OTHER }}'}), id='other-secret-in-another-step'),
        pytest.param(
            _host_result_job(job_fields={'env': {'ZAI_API_KEY': '${{ secrets.ZAI_API_KEY }}'}}), id='key-in-job-env'
        ),
        pytest.param(
            _host_result_job(
                job_fields={'container': {'image': 'ubuntu', 'env': {'ZAI_API_KEY': '${{ secrets.ZAI_API_KEY }}'}}}
            ),
            id='key-in-job-container',
        ),
        pytest.param(
            _host_result_job(
                job_fields={
                    'services': {'cache': {'image': 'redis', 'env': {'ZAI_API_KEY': '${{ secrets.ZAI_API_KEY }}'}}}
                }
            ),
            id='key-in-job-services',
        ),
        pytest.param(
            _host_result_job(job_fields={'defaults': {'run': {'working-directory': '${{ SECRETS.ZAI_API_KEY }}'}}}),
            id='key-in-job-defaults',
        ),
        pytest.param(_host_result_job(upload_path='provider-health/provider-health.json\n'), id='record-not-uploaded'),
        pytest.param(_host_result_job(upload_first=True), id='upload-before-record'),
    ],
)
def test_provider_health_host_result_rejects_a_record_the_monitor_cannot_trust(
    tmp_path: Path, record_job: dict[str, object] | None
):
    workflows = tmp_path / '.github' / 'workflows'
    _write_host_result_workflow(workflows, record_job)

    assert [violation.check for violation in check_provider_health_host_result(workflows)] == [
        'provider-health-host-result'
    ]


def test_provider_health_readiness_must_be_required_in_every_or_branch(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    source = _write(
        workflows / 'pydantic-ai-mini.md',
        """---
name: Z.AI
imports:
  - shared/engine-zai.md
  - shared/provider-health.md
if: ${{ needs.provider_health.outputs.ready == 'true' }}
safe-outputs:
  report-failure-as-issue: false
  report-failed-jobs: false
  noop:
    report-as-issue: false
---
Readiness: ${{ needs.provider_health.outputs.ready }}
""",
    )
    lock = _write(
        source.with_suffix('.lock.yml'),
        """jobs:
  activation:
    needs: [provider_health]
    if: needs.provider_health.outputs.ready == 'true'
  provider_health:
    needs: [pre_activation]
""",
    )

    assert check_provider_health_wiring(workflows) == []

    # CI Review's sources disabled the other two reporters but left this one, which duplicated every task incident
    # with an `[aw] Failed jobs` issue.
    configured = source.read_text(encoding='utf-8')
    source.write_text(configured.replace('  report-failed-jobs: false\n', '', 1), encoding='utf-8')
    assert [violation.check for violation in check_provider_health_wiring(workflows)] == ['provider-health-reporting']
    source.write_text(configured, encoding='utf-8')

    source.write_text(
        source.read_text(encoding='utf-8').replace(
            "if: ${{ needs.provider_health.outputs.ready == 'true' }}",
            "if: ${{ needs.provider_health.outputs.ready == 'true' || true }}",
            1,
        ),
        encoding='utf-8',
    )
    assert [violation.check for violation in check_provider_health_wiring(workflows)] == ['provider-health-gate']

    source.write_text(
        source.read_text(encoding='utf-8').replace(
            "needs.provider_health.outputs.ready == 'true' || true",
            "needs.provider_health.outputs.ready == 'true'",
            1,
        ),
        encoding='utf-8',
    )
    lock.write_text(
        lock.read_text(encoding='utf-8').replace(
            "needs.provider_health.outputs.ready == 'true'",
            "needs.provider_health.outputs.ready == 'true' || true",
            1,
        ),
        encoding='utf-8',
    )
    assert [violation.check for violation in check_provider_health_wiring(workflows)] == [
        'provider-health-activation-if'
    ]


def test_provider_health_identity_is_shared_and_stable_for_workflow_run_retries(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    shared = workflows / 'shared'
    task_key = (
        '${{ github.workflow }}:${{ github.event_name }}:${{ github.event.workflow_run.head_branch }}:'
        "${{ github.event.workflow_run.head_sha }}:${{ (github.event_name == 'workflow_dispatch' && github.run_id) || '' }}"
    )
    engine = _write(
        shared / 'engine-zai.md',
        f'engine:\n  env:\n    GITHUB_WORKFLOW: ${{{{ github.workflow }}}}\n'
        f'    PYDANTIC_AI_TRIGGER_EVENT: ${{{{ github.event_name }}}}\n'
        f'    PYDANTIC_AI_RUN_ATTEMPT: ${{{{ github.run_attempt }}}}\n    PYDANTIC_AI_TASK_KEY: {task_key}\n',
    )
    health = _write(
        shared / 'provider-health.md',
        f'job:\n  env:\n    PYDANTIC_AI_RUN_ATTEMPT: ${{{{ github.run_attempt }}}}\n'
        f'    PYDANTIC_AI_TASK_KEY: {task_key}\n',
    )

    assert check_provider_health_identity(workflows) == []
    expression = re.search(r'^\s*PYDANTIC_AI_TASK_KEY: (.*)$', engine.read_text(), re.MULTILINE)
    assert expression is not None
    assert expression.group(1).count('github.run_id') == 1
    assert "github.event_name == 'workflow_dispatch' && github.run_id" in expression.group(1)
    assert 'github.run_attempt' not in expression.group(1)

    health_text = health.read_text(encoding='utf-8')
    _write(health, health_text.replace('    PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}\n', '', 1))
    assert [violation.check for violation in check_provider_health_identity(workflows)] == ['provider-health-identity']
    _write(
        health,
        health_text.replace(
            '    PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}',
            '    PYDANTIC_AI_RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}',
            1,
        ),
    )
    assert [violation.check for violation in check_provider_health_identity(workflows)] == ['provider-health-identity']
    unstable_task_key = f'{task_key}:${{{{ github.run_attempt }}}}'
    _write(engine, engine.read_text().replace(task_key, unstable_task_key, 1))
    _write(health, health_text.replace(task_key, unstable_task_key, 1))
    assert [violation.check for violation in check_provider_health_identity(workflows)] == [
        'provider-health-task-stability'
    ]


def test_provider_health_identity_rejects_gate_shim_drift(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    shared = workflows / 'shared'
    _write(
        shared / 'engine-zai.md',
        'engine:\n  env:\n    GITHUB_WORKFLOW: ${{ github.workflow }}\n'
        '    PYDANTIC_AI_TRIGGER_EVENT: ${{ github.event_name }}\n'
        '    PYDANTIC_AI_TASK_KEY: ${{ github.workflow }}: ${{ github.run_id }}\n',
    )
    _write(shared / 'provider-health.md', 'job:\n  env:\n    PYDANTIC_AI_TASK_KEY: ${{ github.workflow }}\n')

    assert [violation.check for violation in check_provider_health_identity(workflows)] == ['provider-health-identity']


def test_provider_health_identity_requires_run_attempt_for_local_engine_config(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    shared = workflows / 'shared'
    task_key = (
        '${{ github.workflow }}:${{ github.event_name }}:${{ github.event.workflow_run.head_branch }}:'
        "${{ github.event.workflow_run.head_sha }}:${{ (github.event_name == 'workflow_dispatch' && github.run_id) || '' }}"
    )
    _write(
        shared / 'engine-zai.md',
        f'engine:\n  env:\n    GITHUB_WORKFLOW: ${{{{ github.workflow }}}}\n'
        f'    PYDANTIC_AI_TRIGGER_EVENT: ${{{{ github.event_name }}}}\n'
        f'    PYDANTIC_AI_RUN_ATTEMPT: ${{{{ github.run_attempt }}}}\n    PYDANTIC_AI_TASK_KEY: {task_key}\n',
    )
    _write(
        shared / 'provider-health.md',
        f'job:\n  env:\n    PYDANTIC_AI_RUN_ATTEMPT: ${{{{ github.run_attempt }}}}\n'
        f'    PYDANTIC_AI_TASK_KEY: {task_key}\n',
    )
    local = _write(
        workflows / 'pydantic-ai-local.md',
        f'---\nengine:\n  env:\n    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic\n'
        f'    GITHUB_WORKFLOW: ${{{{ github.workflow }}}}\n'
        f'    PYDANTIC_AI_TRIGGER_EVENT: ${{{{ github.event_name }}}}\n'
        f'    PYDANTIC_AI_TASK_KEY: {task_key}\n---\nPrompt\n',
    )

    assert [violation.path for violation in check_provider_health_identity(workflows)] == [str(local)]
    local.write_text(
        local.read_text(encoding='utf-8').replace(
            '    PYDANTIC_AI_TASK_KEY:',
            '    PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}\n    PYDANTIC_AI_TASK_KEY:',
            1,
        ),
        encoding='utf-8',
    )
    assert check_provider_health_identity(workflows) == []


def test_provider_health_monitor_is_explicitly_scoped_and_cannot_recurse(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    _write(
        workflows / 'pydantic-ai-a.md',
        '---\nname: Agent A\nimports:\n  - shared/engine-zai.md\n---\nPrompt\n',
    )
    _write(
        workflows / 'pydantic-ai-b.md',
        '---\nname: Agent B\nimports:\n  - shared/engine-zai.md\n---\nPrompt\n',
    )
    monitor = _write(
        workflows / 'agent-provider-health.yml',
        """
name: Agent Provider Health
on:
  workflow_run:
    workflows: [Agent A, Agent B, Agent Provider Health]
    types: [completed]
  schedule: [{cron: '0 */6 * * *'}]
  workflow_dispatch:
    inputs:
      recover_issue: {type: number, required: true}
jobs:
  triage:
    if: >-
      github.repository == 'pydantic/pydantic-ai' &&
      github.event_name == 'workflow_run' &&
      (github.event.workflow_run.event != 'pull_request' ||
       (github.event.workflow_run.head_repository.full_name == github.repository &&
        github.event.workflow_run.actor.login != 'dependabot[bot]')) &&
      (github.event.workflow_run.event != 'pull_request_review' ||
       github.event.workflow_run.head_repository.full_name == github.repository)
    permissions:
      actions: read
      contents: read
    outputs:
      reconcile: ${{ steps.triage.outputs.reconcile }}
    steps:
      - uses: actions/checkout@sha
        with:
          repository: ${{ github.repository }}
          ref: ${{ github.event.repository.default_branch }}
          persist-credentials: false
          sparse-checkout: .github/scripts/agent_provider_health.py
      - if: github.event_name == 'workflow_run'
        uses: actions/download-artifact@sha
        with:
          name: agent
          run-id: ${{ github.event.workflow_run.id }}
      - if: github.event_name == 'workflow_run' && steps.agent-artifact.outcome != 'success'
        run: |
          mkdir -p agent
          : > agent/agent-stdio.log
      - if: github.event_name == 'workflow_run'
        id: triage
        env:
          GITHUB_TOKEN: ${{ github.token }}
          RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}
        run: >-
          python3 .github/scripts/agent_provider_health.py triage
          --repository "$GITHUB_REPOSITORY"
          --run-id "$RUN_ID"
          --run-attempt "$RUN_ATTEMPT"
          --conclusion "$CONCLUSION"
          --health-artifact provider-health/provider-health.json
          --agent-artifact agent
  monitor:
    needs: triage
    if: >-
      !cancelled() &&
      github.repository == 'pydantic/pydantic-ai' &&
      (github.event_name != 'workflow_run' || needs.triage.outputs.reconcile == 'true')
    concurrency:
      group: provider-health
      cancel-in-progress: false
    env:
      PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}
    permissions:
      actions: read
      contents: read
      issues: write
      pull-requests: read
    steps:
      - uses: actions/checkout@sha
        with:
          repository: ${{ github.repository }}
          ref: ${{ github.event.repository.default_branch }}
          persist-credentials: false
          sparse-checkout: .github/scripts/agent_provider_health.py
      - if: github.event_name == 'workflow_run'
        uses: actions/download-artifact@sha
        with:
          name: agent
          run-id: ${{ github.event.workflow_run.id }}
      - if: github.event_name == 'workflow_run' && steps.agent-artifact.outcome != 'success'
        run: |
          mkdir -p agent
          : > agent/agent-stdio.log
      - if: github.event_name == 'workflow_run'
        env:
          GITHUB_TOKEN: ${{ github.token }}
          RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}
        run: >-
          python3 .github/scripts/agent_provider_health.py monitor
          --repository "$GITHUB_REPOSITORY"
          --run-id "$RUN_ID"
          --run-attempt "$RUN_ATTEMPT"
          --conclusion "$CONCLUSION"
          --health-artifact provider-health/provider-health.json
          --agent-artifact agent
      - if: github.event_name == 'schedule'
        env:
          ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}
          ZAI_BACKUP_API_KEY: ${{ secrets.ZAI_BACKUP_API_KEY }}
        run: python3 .github/scripts/agent_provider_health.py monitor
      - if: github.event_name == 'workflow_dispatch'
        env:
          GITHUB_TOKEN: ${{ github.token }}
          ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}
          ZAI_BACKUP_API_KEY: ${{ secrets.ZAI_BACKUP_API_KEY }}
          ISSUE_NUMBER: ${{ inputs.recover_issue }}
        run: |
          python3 .github/scripts/agent_provider_health.py monitor \
            --repository "$GITHUB_REPOSITORY" \
            --recover-issue "$ISSUE_NUMBER"
  """,
    )

    violations = check_provider_health_monitor(workflows)

    assert [violation.check for violation in violations] == [
        'provider-health-monitor-workflows',
        'provider-health-monitor-recursion',
        'provider-health-monitor-queue',
    ]
    assert all(violation.path == str(monitor) for violation in violations)

    trusted_config = monitor.read_text(encoding='utf-8')
    _write(
        monitor,
        trusted_config.replace(
            'ref: ${{ github.event.repository.default_branch }}',
            'ref: ${{ github.event.workflow_run.head_sha }}',
            1,
        ),
    )
    assert 'provider-health-monitor-trusted-checkout' in {
        violation.check for violation in check_provider_health_monitor(workflows)
    }
    _write(
        monitor,
        trusted_config.replace(
            'python3 .github/scripts/agent_provider_health.py monitor',
            'python3 agent/agent-stdio.log',
            1,
        ),
    )
    assert 'provider-health-monitor-artifact-execution' in {
        violation.check for violation in check_provider_health_monitor(workflows)
    }
    _write(
        monitor,
        trusted_config.replace(
            "- if: github.event_name == 'schedule'\n        env:",
            "- if: github.event_name == 'workflow_run'\n        env:",
            1,
        ),
    )
    assert 'provider-health-monitor-provider-event-scope' in {
        violation.check for violation in check_provider_health_monitor(workflows)
    }
    _write(
        monitor,
        trusted_config.replace(
            '        run: >-\n'
            '          python3 .github/scripts/agent_provider_health.py monitor\n'
            '          --repository "$GITHUB_REPOSITORY"\n'
            '          --run-id "$RUN_ID"\n'
            '          --run-attempt "$RUN_ATTEMPT"\n'
            '          --conclusion "$CONCLUSION"\n'
            '          --health-artifact provider-health/provider-health.json\n'
            '          --agent-artifact agent',
            '        run: bash ./agent/payload',
            1,
        ),
    )
    assert 'provider-health-monitor-artifact-execution' in {
        violation.check for violation in check_provider_health_monitor(workflows)
    }
    _write(
        monitor,
        trusted_config.replace(
            '    env:\n      PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}\n',
            '',
            1,
        ),
    )
    assert 'provider-health-monitor-run-attempt' in {
        violation.check for violation in check_provider_health_monitor(workflows)
    }
    _write(
        monitor,
        trusted_config.replace(
            'RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}',
            'RUN_ATTEMPT: ${{ github.run_attempt }}',
            1,
        ),
    )
    assert 'provider-health-monitor-source-run-attempt' in {
        violation.check for violation in check_provider_health_monitor(workflows)
    }


@pytest.mark.parametrize(
    ('gate', 'weakened_gate'),
    [
        # The ungated monitor that opened #9753 on a fork run.
        (
            "if: >-\n      github.repository == 'pydantic/pydantic-ai' &&\n"
            "      github.event_name == 'workflow_run' &&\n"
            "      (github.event.workflow_run.event != 'pull_request' ||\n"
            '       (github.event.workflow_run.head_repository.full_name == github.repository &&\n'
            "        github.event.workflow_run.actor.login != 'dependabot[bot]'))",
            "if: github.repository == 'pydantic/pydantic-ai'",
        ),
        # The fork-only gate that opened #10181 on a Dependabot run.
        (
            '       (github.event.workflow_run.head_repository.full_name == github.repository &&\n'
            "        github.event.workflow_run.actor.login != 'dependabot[bot]'))",
            '       github.event.workflow_run.head_repository.full_name == github.repository)',
        ),
        # A gate that accepts a fork's `pull_request_review` run.
        (
            " &&\n      (github.event.workflow_run.event != 'pull_request_review' ||\n"
            '       github.event.workflow_run.head_repository.full_name == github.repository)',
            '',
        ),
    ],
)
def test_provider_health_monitor_rejects_credentialless_runs(tmp_path: Path, gate: str, weakened_gate: str):
    """A monitor that reconciles runs without Actions secrets must not pass CI policy checks."""
    workflows = tmp_path / '.github' / 'workflows'
    for source in WORKFLOWS_DIR.glob('pydantic-ai-*.md'):
        _write(workflows / source.name, source.read_text(encoding='utf-8'))
    trusted_config = (WORKFLOWS_DIR / 'agent-provider-health.yml').read_text(encoding='utf-8')
    monitor = _write(workflows / 'agent-provider-health.yml', trusted_config)
    assert check_provider_health_monitor(workflows) == []

    assert gate in trusted_config
    _write(monitor, trusted_config.replace(gate, weakened_gate, 1))
    assert [violation.check for violation in check_provider_health_monitor(workflows)] == [
        'provider-health-monitor-credential-gate'
    ]


def test_provider_health_monitor_serializes_only_triaged_incident_work(tmp_path: Path):
    """A monitor that queues every completion must not pass CI policy checks."""
    workflows = tmp_path / '.github' / 'workflows'
    for source in WORKFLOWS_DIR.glob('pydantic-ai-*.md'):
        _write(workflows / source.name, source.read_text(encoding='utf-8'))
    trusted_config = (WORKFLOWS_DIR / 'agent-provider-health.yml').read_text(encoding='utf-8')
    monitor = _write(workflows / 'agent-provider-health.yml', trusted_config)
    assert check_provider_health_monitor(workflows) == []

    job_group = (
        '    concurrency:\n'
        '      group: agent-provider-health-reconciliation\n'
        '      cancel-in-progress: false\n'
        '      queue: max\n'
    )
    assert job_group in trusted_config
    # The workflow-level queue that cancelled new runs once 100 were pending, e.g. run 39816847357.
    _write(
        monitor,
        trusted_config.replace(job_group, '', 1).replace(
            'permissions: {}\n',
            'permissions: {}\n\nconcurrency:\n  group: agent-provider-health-reconciliation\n'
            '  cancel-in-progress: false\n  queue: max\n',
            1,
        ),
    )
    assert [violation.check for violation in check_provider_health_monitor(workflows)] == [
        'provider-health-monitor-concurrency',
        'provider-health-monitor-queue',
    ]
    _write(
        monitor,
        trusted_config.replace(
            'permissions: {}\n',
            'permissions: {}\n\nconcurrency:\n  group: agent-provider-health-reconciliation\n'
            '  cancel-in-progress: false\n  queue: max\n',
            1,
        ),
    )
    assert [violation.check for violation in check_provider_health_monitor(workflows)] == [
        'provider-health-monitor-concurrency'
    ]
    _write(monitor, trusted_config.replace('  triage:\n', '  triage:\n    concurrency: provider-health-triage\n', 1))
    assert [violation.check for violation in check_provider_health_monitor(workflows)] == [
        'provider-health-monitor-concurrency'
    ]

    triage_handoff = "needs.triage.outputs.reconcile == 'true'"
    assert triage_handoff in trusted_config
    _write(monitor, trusted_config.replace(triage_handoff, 'true', 1))
    assert [violation.check for violation in check_provider_health_monitor(workflows)] == [
        'provider-health-monitor-triage'
    ]
    # Triage that reads a different agent artifact than the monitor misclassifies typed failures as unclassified.
    head, prefixed_pattern, tail = trusted_config.rpartition("pattern: '*-agent'")
    assert head.count(prefixed_pattern) == 1
    _write(monitor, head + "pattern: 'agent-*'" + tail)
    assert [violation.check for violation in check_provider_health_monitor(workflows)] == [
        'provider-health-monitor-triage'
    ]
    # Triage without the source conclusion leaves a failed run unclassified that the monitor reconciles.
    for triage_input in (
        '          --conclusion "$CONCLUSION"\n',
        '          CONCLUSION: ${{ github.event.workflow_run.conclusion }}\n',
    ):
        assert triage_input in trusted_config
        _write(monitor, trusted_config.replace(triage_input, '', 1))
        assert [violation.check for violation in check_provider_health_monitor(workflows)] == [
            'provider-health-monitor-triage'
        ]
    # Triage gated more narrowly than the monitor step drops completions the monitor would reconcile.
    triage_condition = (
        "if: github.event_name == 'workflow_run' && steps.health-artifact.outcome == 'success'\n        id: triage\n"
    )
    assert triage_condition in trusted_config
    _write(
        monitor,
        trusted_config.replace(
            triage_condition,
            "if: github.event_name == 'workflow_run' && steps.health-artifact.outcome == 'success'"
            " && github.event.workflow_run.conclusion == 'failure'\n        id: triage\n",
            1,
        ),
    )
    assert [violation.check for violation in check_provider_health_monitor(workflows)] == [
        'provider-health-monitor-triage'
    ]

    triage_permissions = '      contents: read\n    outputs:\n'
    assert triage_permissions in trusted_config
    _write(
        monitor,
        trusted_config.replace(triage_permissions, '      contents: read\n      issues: write\n    outputs:\n', 1),
    )
    assert [violation.check for violation in check_provider_health_monitor(workflows)] == [
        'provider-health-monitor-permissions'
    ]


def test_assigned_alert_metadata_gate_excludes_operational_incidents(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    _write(
        workflows / 'at-claude.yml', "name: '@claude'\non: issues\nif: contains(github.event.issue.body, '@claude')\n"
    )

    violations = check_assigned_alert_metadata_gate(workflows)

    assert [violation.check for violation in violations] == ['assigned-alert-metadata-gate']


def test_assigned_alert_metadata_gate_checks_each_event_route(tmp_path: Path):
    workflows = tmp_path / '.github' / 'workflows'
    _write(
        workflows / 'at-claude.yml',
        """name: '@claude'
on:
  issue_comment:
    types: [created]
  issues:
    types: [opened, assigned]
jobs:
  get-pr-info:
    if: |
      (github.event_name == 'issue_comment' &&
        !contains(github.event.issue.labels.*.name, 'pydanty:meta') &&
        !contains(github.event.issue.labels.*.name, 'pydanty:meta') &&
        contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'issues' &&
        contains(github.event.issue.body, '@claude'))
""",
    )

    violations = check_assigned_alert_metadata_gate(workflows)

    assert [violation.check for violation in violations] == ['assigned-alert-metadata-gate']
