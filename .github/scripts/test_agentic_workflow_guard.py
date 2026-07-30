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

from agentic_workflow_guard import (
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


def test_dangling_needs_ignores_a_lock_without_jobs(tmp_path: Path):
    assert check_dangling_needs(_write(tmp_path / 'w.lock.yml', 'name: nothing\n')) == []


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


def test_job_timeout_env_is_optional(tmp_path: Path):
    """Workflows that keep the default budget need not declare it."""
    source = _write(tmp_path / 'w.md', '---\ntimeout-minutes: 30\n---\nprompt\n')

    assert check_job_timeout_env(source) == []


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


def test_lock_regenerated_ignores_an_unimported_shared_fragment(workflows_dir: Path):
    _write(workflows_dir / 'shared' / 'unused.md', '---\nname: unused\n---\nbody\n')

    assert check_lock_regenerated([str(workflows_dir / 'shared' / 'unused.md')], workflows_dir) == []


# --- rendering, parsing edge cases, and the CLI -------------------------------


def test_violation_renders_as_one_line():
    from agentic_workflow_guard import Violation

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


def test_changed_files_returns_empty_for_an_unresolvable_ref():
    from agentic_workflow_guard import changed_files

    assert changed_files('definitely-not-a-ref-8f3a2b') == []


def test_main_reports_success_on_a_clean_tree(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    import agentic_workflow_guard
    from agentic_workflow_guard import Violation

    def no_violations(workflows_dir: Path = WORKFLOWS_DIR, changed: list[str] | None = None) -> list[Violation]:
        return []

    monkeypatch.setattr(agentic_workflow_guard, 'run_checks', no_violations)

    assert agentic_workflow_guard.main(['check']) == 0
    assert 'passed' in capsys.readouterr().out


def test_main_exits_nonzero_and_prints_each_violation(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    import agentic_workflow_guard
    from agentic_workflow_guard import Violation

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
    _write(workflows / 'pydantic-ai-x.md', '---\ntimeout-minutes: 30\n---\nprompt\n')
    _write(workflows / 'pydantic-ai-x.lock.yml', 'jobs: {}\n')
    _write(workflows / 'shared' / 'ctx.md', '---\nname: ctx\n---\nRead /tmp/gh-aw/.review-context/x\n')

    violations = run_checks(workflows)

    assert [v.check for v in violations] == ['prompt-path-outside-workspace']
