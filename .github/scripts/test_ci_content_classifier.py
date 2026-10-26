from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter
from typing_extensions import TypedDict

WORKFLOW = Path(__file__).parents[1] / 'workflows' / 'ci.yml'
BENCHMARK_WORKFLOW = Path(__file__).parents[1] / 'workflows' / 'benchmark.yml'


class Workflow(TypedDict):
    jobs: dict[str, object]


class ClassifierStep(TypedDict):
    run: str


class ClassifierJob(TypedDict):
    steps: list[ClassifierStep]


CheckWith = TypedDict('CheckWith', {'allowed-skips': str})


CheckStep = TypedDict('CheckStep', {'with': CheckWith})


class CheckJob(TypedDict):
    needs: list[str]
    steps: list[CheckStep]


class JobSteps(TypedDict):
    steps: list[dict[str, object]]


BenchmarkJob = TypedDict('BenchmarkJob', {'needs': str, 'if': str})
ConditionalJob = TypedDict('ConditionalJob', {'if': str})


WORKFLOW_ADAPTER: TypeAdapter[Workflow] = TypeAdapter(Workflow)
CLASSIFIER_JOB_ADAPTER: TypeAdapter[ClassifierJob] = TypeAdapter(ClassifierJob)
CHECK_JOB_ADAPTER: TypeAdapter[CheckJob] = TypeAdapter(CheckJob)
JOB_STEPS_ADAPTER: TypeAdapter[JobSteps] = TypeAdapter(JobSteps)
BENCHMARK_JOB_ADAPTER: TypeAdapter[BenchmarkJob] = TypeAdapter(BenchmarkJob)
CONDITIONAL_JOB_ADAPTER: TypeAdapter[ConditionalJob] = TypeAdapter(ConditionalJob)


def _workflow(path: Path = WORKFLOW) -> Workflow:
    loaded: object = yaml.safe_load(path.read_text(encoding='utf-8'))
    return WORKFLOW_ADAPTER.validate_python(loaded)


def _classify(
    tmp_path: Path,
    changed_files: list[tuple[str, str]],
    *,
    workflow_path: Path = WORKFLOW,
    event_name: str = 'pull_request',
    total: int | str | None = None,
    count_api_failure: bool = False,
    files_api_failure: bool = False,
) -> dict[str, str]:
    workflow = _workflow(workflow_path)
    classify_job = CLASSIFIER_JOB_ADAPTER.validate_python(workflow['jobs']['classify'])
    run = classify_job['steps'][0]['run']
    run = run.replace('${{ github.event_name }}', event_name)

    gh = tmp_path / 'gh'
    gh.write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *"--jq .changed_files"* ]]; then
  [[ "${GH_COUNT_FAILURE:-false}" != true ]] || exit 1
  printf '%s\\n' "$GH_CHANGED_COUNT"
elif [[ "$*" == *"pulls/$PR_NUMBER/files"* ]]; then
  [[ "$*" == *"--paginate"* ]] || exit 3
  [[ "${GH_FILES_FAILURE:-false}" != true ]] || exit 1
  if [[ -n "$GH_CHANGED_FILES" ]]; then
    printf '%s\\n' "$GH_CHANGED_FILES"
  fi
else
  exit 2
fi
""",
        encoding='utf-8',
    )
    gh.chmod(0o755)
    runner_temp = tmp_path / 'runner-temp'
    runner_temp.mkdir()
    output = tmp_path / 'github-output'
    files = '\n'.join(f'{new}\t{old}' for new, old in changed_files)
    env = {
        **os.environ,
        'PATH': f'{tmp_path}:{os.environ["PATH"]}',
        'GITHUB_OUTPUT': str(output),
        'RUNNER_TEMP': str(runner_temp),
        'GITHUB_REPOSITORY': 'pydantic/pydantic-ai',
        'PR_NUMBER': '123',
        'GH_TOKEN': 'test-token',
        'GH_CHANGED_COUNT': str(len(changed_files) if total is None else total),
        'GH_CHANGED_FILES': files,
        'GH_COUNT_FAILURE': str(count_api_failure).lower(),
        'GH_FILES_FAILURE': str(files_api_failure).lower(),
    }
    process = subprocess.run(['bash', '-e', '-c', run], env=env, text=True, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr
    return dict(line.split('=', maxsplit=1) for line in output.read_text(encoding='utf-8').splitlines())


@pytest.mark.parametrize(
    ('changed_files', 'expected'),
    [
        ([('README.md', '')], {'content_only': 'true', 'docs_changed': 'true'}),
        (
            [('docs/agents.md', ''), ('.agents/data/catalog.tsv', '')],
            {'content_only': 'true', 'docs_changed': 'true'},
        ),
        ([('docs/guides/agents.md', '')], {'content_only': 'true', 'docs_changed': 'true'}),
        ([('docs/AGENTS.md', '')], {'content_only': 'true', 'docs_changed': 'true'}),
        ([('docs/img/logo.svg', '')], {'content_only': 'true', 'docs_changed': 'true'}),
        ([('docs/navigation.yml', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('pydantic_ai_slim/README.md', '')], {'content_only': 'true', 'docs_changed': 'true'}),
        ([('src/pydantic_ai_harness/README.md', '')], {'content_only': 'true', 'docs_changed': 'true'}),
        ([('.agents/skills/review/SKILL.md', '')], {'content_only': 'true', 'docs_changed': 'false'}),
        ([('.macroscope/ignore.md', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('.macroscope/correctness/review-discipline.md', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('.macroscope/AGENTS.md', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('.agents/skills/review/references.md', '')], {'content_only': 'true', 'docs_changed': 'false'}),
        ([('.claude/skills/review/SKILL.md', '')], {'content_only': 'true', 'docs_changed': 'false'}),
        ([('agent_docs/index.md', '')], {'content_only': 'true', 'docs_changed': 'false'}),
        ([('CONTRIBUTING.md', '')], {'content_only': 'true', 'docs_changed': 'false'}),
        ([('.github/pull_request_template.md', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('.github/workflows/AGENTS.md', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('.github/ISSUE_TEMPLATE/bug.md', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('.github/ISSUE_TEMPLATE/bug.yaml', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('src/pydantic_clai2/AGENTS.md', '')], {'content_only': 'true', 'docs_changed': 'false'}),
        (
            [('docs/agents.md', ''), ('.agents/skills/review/SKILL.md', '')],
            {'content_only': 'true', 'docs_changed': 'true'},
        ),
        (
            [('.agents/skills/review/SKILL.md', 'docs/review.md')],
            {'content_only': 'true', 'docs_changed': 'true'},
        ),
        ([('.agents/skills/review/nested/SKILL.md', '')], {'content_only': 'true', 'docs_changed': 'false'}),
        ([('.agents/skills/SKILL.md', '')], {'content_only': 'true', 'docs_changed': 'false'}),
        ([('AGENTS.md', '')], {'content_only': 'true', 'docs_changed': 'false'}),
        (
            [('tests/harness/skills/temporal_workspace/skills/reviewer/SKILL.md', '')],
            {'content_only': 'false', 'docs_changed': 'false'},
        ),
        (
            [('tests/harness/repo_context/fixtures/AGENTS.md', '')],
            {'content_only': 'false', 'docs_changed': 'false'},
        ),
        ([('.github/workflows/pydantic-ai-pr-review.md', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('.github/workflows/ci.yml', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('pyproject.toml', '')], {'content_only': 'false', 'docs_changed': 'false'}),
        ([('src/code.py', 'docs/old.md')], {'content_only': 'false', 'docs_changed': 'true'}),
    ],
)
def test_classifies_content_without_python_test_impact(
    tmp_path: Path, changed_files: list[tuple[str, str]], expected: dict[str, str]
):
    outputs = _classify(tmp_path, changed_files)

    assert {key: outputs[key] for key in expected} == expected


@pytest.mark.parametrize(
    ('changed_files', 'docs_changed'),
    [
        ([('.github/workflows/benchmark.yml', '')], 'false'),
        ([('.github/workflows/example.yaml', '')], 'false'),
        ([('.github/workflows/pydantic-ai-pr-review.md', '')], 'false'),
        ([('.github/workflows/AGENTS.md', '')], 'false'),
        ([('.github/ISSUE_TEMPLATE/bug.yaml', '')], 'false'),
        ([('.github/ISSUE_TEMPLATE/bug.md', '')], 'false'),
        ([('.github/pull_request_template.md', '')], 'false'),
        ([('.github/dependabot.yml', '')], 'false'),
        ([('.github/CODEOWNERS', '')], 'false'),
        ([('docs/navigation.yml', '')], 'false'),
        ([('pyproject.toml', '')], 'false'),
        ([('docs/agent.md', ''), ('.github/workflows/benchmark.yml', '')], 'true'),
    ],
)
def test_workflow_and_config_changes_keep_full_ci(
    tmp_path: Path, changed_files: list[tuple[str, str]], docs_changed: str
):
    outputs = _classify(tmp_path, changed_files)

    assert outputs['content_only'] == 'false'
    assert outputs['docs_changed'] == docs_changed


@pytest.mark.parametrize(
    ('changed_files', 'total', 'count_api_failure', 'files_api_failure'),
    [
        ([('README.md', '')], 0, False, False),
        ([('README.md', '')], 3000, False, False),
        ([], 1, False, False),
        ([('README.md', '')], 2, False, False),
        ([('README.md', '')], 'not-a-number', False, False),
        ([('README.md', '')], 1, True, False),
        ([('README.md', '')], 1, False, True),
    ],
)
def test_incomplete_pr_file_list_defaults_to_full_ci(
    tmp_path: Path,
    changed_files: list[tuple[str, str]],
    total: int | str,
    count_api_failure: bool,
    files_api_failure: bool,
):
    outputs = _classify(
        tmp_path,
        changed_files,
        total=total,
        count_api_failure=count_api_failure,
        files_api_failure=files_api_failure,
    )

    assert outputs == {
        'content_only': 'false',
        'docs_changed': 'false',
        'clai2_only': 'false',
        'clai2_changed': 'true',
        'clai2_web_changed': 'true',
        'harness_only': 'false',
        'pyright_changed': 'true',
    }


def test_non_pr_events_keep_full_ci_defaults(tmp_path: Path):
    outputs = _classify(tmp_path, [], event_name='push')

    assert outputs == {
        'content_only': 'false',
        'docs_changed': 'false',
        'clai2_only': 'false',
        'clai2_changed': 'true',
        'clai2_web_changed': 'true',
        'harness_only': 'false',
        'pyright_changed': 'true',
    }


@pytest.mark.parametrize(
    ('changed_files', 'clai2_changed'),
    [
        ([('src/pydantic_clai2/pydantic_clai2/ui/prompt/image_input.py', '')], 'true'),
        ([('tests/clai2/test_image_input.py', '')], 'true'),
        ([('.github/workflows/ci.yml', '')], 'true'),
        ([('.github/workflows/clai2-web.yml', '')], 'false'),
        ([('.github/scripts/test_ci_content_classifier.py', '')], 'false'),
        ([('.github/scripts/test_agentic_workflow_guard.py', '')], 'false'),
        ([('src/pydantic_ai_harness/pydantic_ai_harness/media/__init__.py', '')], 'false'),
        ([('docs/agent.md', '')], 'false'),
        ([('src/code.py', 'src/pydantic_clai2/pydantic_clai2/code.py')], 'true'),
    ],
)
def test_clai2_clipboard_runs_only_for_clai2_changes(
    tmp_path: Path, changed_files: list[tuple[str, str]], clai2_changed: str
):
    outputs = _classify(tmp_path, changed_files)

    assert outputs['clai2_changed'] == clai2_changed


@pytest.mark.parametrize(
    ('changed_files', 'expected'),
    [
        ([('src/pydantic_clai2/web-ui/src/App.tsx', '')], 'true'),
        ([('tests/clai2/browser/test_web_browser.py', '')], 'true'),
        ([('.github/workflows/ci.yml', '')], 'true'),
        ([('.github/workflows/clai2-web.yml', '')], 'true'),
        ([('pyproject.toml', '')], 'true'),
        ([('uv.lock', '')], 'true'),
        ([('.github/scripts/test_ci_content_classifier.py', '')], 'false'),
        ([('docs/agent.md', '')], 'false'),
        ([('src/renamed.py', '.github/workflows/clai2-web.yml')], 'true'),
    ],
)
def test_clai2_web_runs_for_its_workflow_and_dependencies(
    tmp_path: Path, changed_files: list[tuple[str, str]], expected: str
):
    outputs = _classify(tmp_path, changed_files)

    assert outputs['clai2_web_changed'] == expected


HARNESS_SOURCE = 'src/pydantic_ai_harness/pydantic_ai_harness/media/__init__.py'


@pytest.mark.parametrize(
    ('changed_files', 'expected'),
    [
        ([(HARNESS_SOURCE, '')], 'true'),
        ([('tests/harness/media/test_media.py', '')], 'true'),
        ([(HARNESS_SOURCE, ''), ('tests/clai2/test_image_input.py', '')], 'true'),
        ([(HARNESS_SOURCE, ''), ('docs/harness/media.md', '')], 'true'),
        ([(HARNESS_SOURCE, ''), ('src/pydantic_ai_harness/README.md', '')], 'true'),
        ([(HARNESS_SOURCE, ''), ('docs/agent.md', ''), ('AGENTS.md', '')], 'true'),
        ([('src/pydantic_clai2/pydantic_clai2/code.py', ''), ('README.md', '')], 'true'),
        ([('src/pydantic_ai_harness/pydantic_ai_harness/new.py', HARNESS_SOURCE)], 'true'),
        # The narrower tiers win.
        ([('docs/harness/media.md', '')], 'false'),
        ([('src/pydantic_clai2/pydantic_clai2/code.py', '')], 'false'),
        # Anything outside the two packages and content reaches the rest of the workspace.
        ([(HARNESS_SOURCE, ''), ('pydantic_ai_slim/pydantic_ai/agent/__init__.py', '')], 'false'),
        ([(HARNESS_SOURCE, ''), ('uv.lock', '')], 'false'),
        ([(HARNESS_SOURCE, ''), ('pyproject.toml', '')], 'false'),
        ([('src/pydantic_ai_harness/pyproject.toml', '')], 'false'),
        ([(HARNESS_SOURCE, ''), ('tests/conftest.py', '')], 'false'),
        ([(HARNESS_SOURCE, ''), ('tests/cost_guards.py', '')], 'false'),
        ([(HARNESS_SOURCE, ''), ('tests/test_examples.py', '')], 'false'),
        ([(HARNESS_SOURCE, ''), ('docs/navigation.yml', '')], 'false'),
        ([(HARNESS_SOURCE, ''), ('.github/workflows/ci.yml', '')], 'false'),
        ([(HARNESS_SOURCE, 'pydantic_ai_slim/pydantic_ai/media.py')], 'false'),
    ],
)
def test_harness_only_tier(tmp_path: Path, changed_files: list[tuple[str, str]], expected: str):
    outputs = _classify(tmp_path, changed_files)

    assert outputs['harness_only'] == expected
    if expected == 'true':
        assert outputs['content_only'] == outputs['clai2_only'] == 'false'


def test_harness_only_tier_skips_only_jobs_outside_the_harness():
    jobs = _workflow()['jobs']
    harness_skip = "needs.classify.outputs.harness_only != 'true'"
    for name in ('mypy', 'test-temporal-latest', 'test-examples', 'test-fastmcp-4'):
        assert harness_skip in CONDITIONAL_JOB_ADAPTER.validate_python(jobs[name])['if']
    for name in (
        'quality',
        'docs-assets',
        'test',
        'test-all-extras',
        'test-durable-exec',
        'test-lowest-versions',
        'test-harness-browser-use',
        'coverage',
    ):
        assert 'harness_only' not in CONDITIONAL_JOB_ADAPTER.validate_python(jobs[name])['if']
    docs_only = CONDITIONAL_JOB_ADAPTER.validate_python(jobs['docs-only'])
    assert "needs.classify.outputs.harness_only == 'true'" in docs_only['if']


@pytest.mark.parametrize(
    ('changed_files', 'expected'),
    [
        ([('.agents/skills/pushing-commits-to-the-repo/labels.tsv', '')], 'true'),
        ([('.agents/data/catalog.tsv', ''), ('.agents/skills/review/SKILL.md', '')], 'true'),
        ([('.agents/data/catalog.tsv', ''), ('README.md', '')], 'true'),
        ([('.agents/data/renamed.tsv', '.agents/data/catalog.tsv')], 'true'),
        ([('.agents/data/catalog.tsv', 'tests/data/catalog.tsv')], 'false'),
        ([('tests/data/catalog.tsv', '.agents/data/catalog.tsv')], 'false'),
        ([('.agents/data/catalog.tsv', ''), ('src/code.py', '')], 'false'),
        ([('.agents/skills/pushing-commits-to-the-repo/label-catalog', '')], 'false'),
        ([('.agents/config/settings.yaml', '')], 'false'),
        ([('.agents/scripts/check.py', '')], 'false'),
    ],
)
def test_agent_data_uses_content_only_route(tmp_path: Path, changed_files: list[tuple[str, str]], expected: str):
    outputs = _classify(tmp_path, changed_files)

    assert outputs['content_only'] == expected


@pytest.mark.parametrize(
    ('filename', 'expected'),
    [
        ('pydantic_ai_slim/pydantic_ai/agent/__init__.py', 'true'),
        ('pydantic_graph/pydantic_graph/graph.py', 'true'),
        ('pydantic_evals/pydantic_evals/evaluators/common.py', 'true'),
        ('tests/benchmarks/test_agent.py', 'true'),
        ('tests/benchmarks/conftest.py', 'true'),
        ('tests/conftest.py', 'true'),
        ('tests/__init__.py', 'true'),
        ('tests/cost_guards.py', 'true'),
        ('tests/_inline_snapshot.py', 'true'),
        ('tests/data/input.json', 'true'),
        ('tests/test_tools.py', 'true'),
        ('pyproject.toml', 'true'),
        ('pydantic_ai_slim/pyproject.toml', 'true'),
        ('src/pydantic_ai_harness/pyproject.toml', 'true'),
        ('src/pydantic_clai2/pyproject.toml', 'true'),
        ('clai/pyproject.toml', 'true'),
        ('uv.lock', 'true'),
        ('uv.toml', 'true'),
        ('.python-version', 'true'),
        ('requirements-dev.txt', 'true'),
        ('src/pydantic_ai_harness/requirements.in', 'true'),
        ('src/pydantic_clai2/web-ui/package-lock.json', 'true'),
        ('src/pydantic_clai2/web-ui/package.json', 'true'),
        ('src/pydantic_ai_harness/uv.lock', 'true'),
        ('src/pydantic_ai_harness/setup.cfg', 'true'),
        ('tests/harness/pytest.ini', 'true'),
        ('pytest.ini', 'true'),
        ('.github/workflows/benchmark.yml', 'true'),
        ('.github/scripts/test_ci_content_classifier.py', 'true'),
        ('Makefile', 'true'),
        ('codspeed.yml', 'true'),
        ('unknown/input.json', 'true'),
        ('src/new_package/module.py', 'true'),
        ('', 'true'),
        ('.agents/config/settings.yaml', 'true'),
        ('.agents/scripts/check.py', 'true'),
        ('.macroscope/correctness/review-discipline.md', 'true'),
        ('README.md', 'false'),
        ('THREAT_MODEL.md', 'false'),
        ('docs/harness/filesystem.md', 'false'),
        ('docs/navigation.yml', 'false'),
        ('docs/img/logo.svg', 'false'),
        ('examples/README.md', 'false'),
        ('examples/guide.mdc', 'false'),
        ('pydantic_ai_slim/README.md', 'false'),
        ('pydantic_evals/pydantic_evals/README.md', 'false'),
        ('pydantic_graph/README.md', 'false'),
        ('agent_docs/index.md', 'false'),
        ('.agents/skills/review/SKILL.md', 'false'),
        ('.agents/data/catalog.tsv', 'false'),
        ('.claude/skills/review/SKILL.md', 'false'),
        ('src/pydantic_ai_harness/pydantic_ai_harness/filesystem/_capability.py', 'false'),
        ('tests/harness/repo_context/fixtures/AGENTS.md', 'false'),
        ('tests/harness/conftest.py', 'false'),
        ('src/pydantic_clai2/pydantic_clai2/_app.py', 'false'),
        ('tests/clai2/test_app.py', 'false'),
        ('clai/clai/__init__.py', 'false'),
        ('tests/clai/test_cli.py', 'false'),
    ],
)
def test_benchmark_classifier_uses_benchmark_inputs(tmp_path: Path, filename: str, expected: str):
    outputs = _classify(tmp_path, [(filename, '')], workflow_path=BENCHMARK_WORKFLOW)

    assert outputs == {'run_benchmarks': expected}


@pytest.mark.parametrize(
    ('changed_files', 'expected'),
    [
        ([('docs/new.md', 'pydantic_ai_slim/pydantic_ai/old.py')], 'true'),
        ([('pydantic_ai_slim/pydantic_ai/new.py', 'docs/old.md')], 'true'),
        ([('src/pydantic_ai_harness/new.py', 'pydantic_evals/pydantic_evals/old.py')], 'true'),
        ([('pydantic_graph/pydantic_graph/new.py', 'src/pydantic_clai2/old.py')], 'true'),
        ([('docs/new.md', 'docs/old.md')], 'false'),
        ([('src/pydantic_clai2/new.py', 'src/pydantic_ai_harness/old.py')], 'false'),
        ([('docs/guide.md', ''), ('tests/conftest.py', '')], 'true'),
    ],
)
def test_benchmark_classifier_checks_both_rename_paths(
    tmp_path: Path, changed_files: list[tuple[str, str]], expected: str
):
    outputs = _classify(tmp_path, changed_files, workflow_path=BENCHMARK_WORKFLOW)

    assert outputs == {'run_benchmarks': expected}


def test_benchmark_classifier_skips_filesystem_documentation_pr(tmp_path: Path):
    changed_files = [
        ('THREAT_MODEL.md', ''),
        ('docs/harness/filesystem.md', ''),
        ('src/pydantic_ai_harness/pydantic_ai_harness/filesystem/README.md', ''),
        ('src/pydantic_ai_harness/pydantic_ai_harness/filesystem/_capability.py', ''),
    ]
    outputs = _classify(tmp_path, changed_files, workflow_path=BENCHMARK_WORKFLOW)

    assert outputs == {'run_benchmarks': 'false'}


@pytest.mark.parametrize(
    ('changed_files', 'total', 'count_api_failure', 'files_api_failure'),
    [
        ([], 0, False, False),
        ([], 1, False, False),
        ([('docs/guide.md', '')], 3000, False, False),
        ([('docs/guide.md', '')], 2, False, False),
        ([('docs/guide.md', '')], 'not-a-number', False, False),
        ([('docs/guide.md', '')], 1, True, False),
        ([('docs/guide.md', '')], 1, False, True),
    ],
)
def test_benchmark_classifier_runs_on_count_or_file_api_fallback(
    tmp_path: Path,
    changed_files: list[tuple[str, str]],
    total: int | str,
    count_api_failure: bool,
    files_api_failure: bool,
):
    outputs = _classify(
        tmp_path,
        changed_files,
        workflow_path=BENCHMARK_WORKFLOW,
        total=total,
        count_api_failure=count_api_failure,
        files_api_failure=files_api_failure,
    )

    assert outputs == {'run_benchmarks': 'true'}


@pytest.mark.parametrize('event_name', ['push', 'workflow_dispatch'])
def test_benchmark_non_pr_events_keep_full_run_defaults(tmp_path: Path, event_name: str):
    outputs = _classify(tmp_path, [], workflow_path=BENCHMARK_WORKFLOW, event_name=event_name)

    assert outputs == {'run_benchmarks': 'true'}


def test_benchmark_job_is_gated_on_the_classifier_output():
    benchmark = BENCHMARK_JOB_ADAPTER.validate_python(_workflow(BENCHMARK_WORKFLOW)['jobs']['benchmarks'])

    assert benchmark['needs'] == 'classify'
    assert benchmark['if'] == "always() && !cancelled() && needs.classify.outputs.run_benchmarks != 'false'"


def test_benchmark_triggers_include_main_and_manual_runs():
    # BaseLoader preserves the `on` key instead of interpreting it as a YAML 1.1 boolean.
    workflow: object = yaml.load(BENCHMARK_WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    assert isinstance(workflow, dict)
    assert workflow['on'] == {
        'push': {'branches': ['main']},
        'pull_request': {'branches': ['main']},
        'workflow_dispatch': '',
    }


def test_python_test_jobs_use_content_only_output():
    jobs = _workflow()['jobs']
    for name in (
        'mypy',
        'test',
        'test-all-extras',
        'test-durable-exec',
        'test-lowest-versions',
        'test-temporal-latest',
        'test-examples',
        'test-fastmcp-4',
        'test-harness-browser-use',
        'test-clai2-clipboard',
        'coverage',
    ):
        job = CONDITIONAL_JOB_ADAPTER.validate_python(jobs[name])
        assert "needs.classify.outputs.content_only != 'true'" in job['if']

    quality = CONDITIONAL_JOB_ADAPTER.validate_python(jobs['quality'])
    assert quality['if'] == "needs.classify.outputs.content_only != 'true'"


def test_docs_checks_cover_harness_readmes():
    docs_job = JOB_STEPS_ADAPTER.validate_python(_workflow()['jobs']['docs-only'])
    snippet_steps = [step for step in docs_job['steps'] if step.get('name') == 'Test documentation snippets']
    assert len(snippet_steps) == 1
    command = snippet_steps[0].get('run')
    assert isinstance(command, str)
    for test_path in (
        'tests/test_examples.py',
        'tests/harness/test_docs_installation.py',
        'tests/test_docs_parity.py',
        'tests/test_docs_navigation.py',
        'tests/harness/test_docs_parity.py',
        'tests/harness/test_doc_snippets.py',
        'tests/harness/test_workspace_quickstarts.py',
    ):
        assert test_path in command


def test_aggregate_requires_the_selected_lightweight_job():
    jobs = _workflow()['jobs']
    content_checks = CONDITIONAL_JOB_ADAPTER.validate_python(jobs['content-checks'])
    assert content_checks['if'] == (
        "needs.classify.outputs.content_only == 'true' && needs.classify.outputs.docs_changed != 'true'"
    )
    check = CHECK_JOB_ADAPTER.validate_python(jobs['check'])
    assert 'content-checks' in check['needs']
    allowed_skips = check['steps'][0]['with']['allowed-skips']

    branches = re.findall(r"(?:&&|\|\|)\s*'([^']+)'", allowed_skips)
    docs_skips = set(branches[0].split(','))
    content_skips = set(branches[1].split(','))
    clai2_skips = set(branches[2].split(','))
    harness_docs_skips = set(branches[3].split(','))
    harness_skips = set(branches[4].split(','))
    tag_skips = set(branches[5].split(','))
    default_skips = set(branches[6].split(','))
    assert len(branches) == 7

    assert 'content-checks' in docs_skips
    assert 'docs-only' not in docs_skips
    assert {'docs-only', 'docs-assets', 'test-clai2-web'} <= content_skips
    assert 'content-checks' not in content_skips
    assert {'docs-only', 'content-checks'} <= clai2_skips
    assert {'docs-only', 'content-checks'} <= tag_skips
    assert {'docs-only', 'content-checks'} <= default_skips
    assert 'test-clai2-web' in default_skips
    assert 'test-clai2-clipboard' in check['needs']
    assert 'test-clai2-web' in check['needs']
    assert 'test-clai2-clipboard' not in clai2_skips
    assert 'test-clai2-web' not in clai2_skips
    assert 'test-clai2-clipboard' in tag_skips
    assert harness_skips - harness_docs_skips == {'docs-only'}
    assert harness_docs_skips <= harness_skips
    assert {'test-clai2-web'} <= harness_skips
    for job in (
        'quality',
        'docs-assets',
        'test',
        'test-all-extras',
        'test-durable-exec',
        'test-lowest-versions',
        'test-harness-browser-use',
        'coverage',
    ):
        assert job not in harness_skips
    assert harness_skips <= set(check['needs'])


def test_clai2_clipboard_job_is_gated_on_the_classifier_output():
    job = CONDITIONAL_JOB_ADAPTER.validate_python(_workflow()['jobs']['test-clai2-clipboard'])

    assert "needs.classify.outputs.clai2_changed == 'true'" in job['if']
    assert "github.ref_type != 'tag'" in job['if']


def test_clai2_web_workflow_is_gated_on_the_classifier_output():
    job = CONDITIONAL_JOB_ADAPTER.validate_python(_workflow()['jobs']['test-clai2-web'])

    assert job['if'] == (
        "needs.classify.outputs.content_only != 'true' && needs.classify.outputs.clai2_web_changed == 'true'"
    )


def test_clai2_bleeding_validates_the_published_commit():
    bleeding = _workflow(WORKFLOW.with_name('clai2-bleeding.yml'))
    jobs = bleeding['jobs']
    validate = TypeAdapter(dict[str, object]).validate_python(jobs['validate'])
    publish = TypeAdapter(dict[str, object]).validate_python(jobs['publish'])
    assert validate['uses'] == './.github/workflows/clai2-web.yml'
    assert publish['needs'] == 'validate'
    for workflow, job in (
        (bleeding, 'publish'),
        (_workflow(WORKFLOW.with_name('clai2-web.yml')), 'web-ui'),
    ):
        steps = JOB_STEPS_ADAPTER.validate_python(workflow['jobs'][job])['steps']
        checkout = steps[0]
        assert str(checkout['uses']).startswith('actions/checkout@')
        options = TypeAdapter(dict[str, object]).validate_python(checkout['with'])
        assert options['ref'] == '${{ github.sha }}'
