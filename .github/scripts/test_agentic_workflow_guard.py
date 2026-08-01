"""Tests for the agentic workflow policy guard.

Each check has a regression test built from the *actual* pre-fix configuration
that shipped to `main` (reconstructed from the parent of #6761), so the guard is
verified against the defects it exists to prevent rather than against invented
shapes. The final test asserts the live repository is clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import agentic_workflow_guard
from agentic_workflow_guard import (
    Violation,
    changed_files,
    check_compiler_versions,
    check_dangling_needs,
    check_job_timeout_env,
    check_lock_regenerated,
    check_prompt_paths,
    check_safe_output_job_max,
    check_timeout_declared,
    run_checks,
)

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOWS_DIR = REPO_ROOT / '.github' / 'workflows'


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


def test_prompt_paths_catches_the_review_context_defect(tmp_path: Path):
    """The prompt told the agent to read a path its file tools cannot open."""
    source = _write(
        tmp_path / 'shared.md',
        """---
name: x
---

A pre-agent step wrote everything you need to `/tmp/gh-aw/.review-context/`.
**Read these files instead of calling the GitHub API.**
""",
    )

    violations = check_prompt_paths(source)

    assert [v.check for v in violations] == ['prompt-path-outside-workspace']
    assert '/tmp/gh-aw/.review-context/' in violations[0].message


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
        # A sibling of an allowlisted *file*, the counterpart of `bindings/` vs `bin/`.
        '/tmp/gh-aw/agent/open-issues.tsv.bak',
        '/tmp/gh-aw/agent/issues-private/1.json',
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
        '/tmp/gh-aw/agent/open-issues.tsv',
        '/tmp/gh-aw/agent/issues/1.json',
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
    _write(workflows / 'pydantic-ai-x.lock.yml', 'jobs: {}\n')
    _write(workflows / 'shared' / 'ctx.md', '---\nname: ctx\n---\nRead /tmp/gh-aw/.review-context/x\n')

    violations = run_checks(workflows)

    assert [v.check for v in violations] == ['prompt-path-outside-workspace']
