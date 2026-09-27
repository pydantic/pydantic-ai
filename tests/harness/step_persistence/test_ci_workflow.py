"""Guards on the `mongodb-integration` CI job.

Imports no pymongo on purpose: `conftest.collect_ignore` drops `test_mongo.py`
where the extra is absent, and these assertions must hold on every matrix leg.
"""

from __future__ import annotations

from pathlib import Path


def _workflow_lines() -> list[str]:
    workflow = Path(__file__).parents[2] / '.github' / 'workflows' / 'main.yml'
    return workflow.read_text().splitlines()


def _mongodb_condition() -> str:
    """The `mongodb-integration` job's `if:` expression, collapsed to one line.

    The expression is folded across several lines for readability, so match on
    its content rather than its formatting. The slice ends at the next key at
    job-body indentation, which keeps unrelated job settings out of the string --
    the `not in` assertions below would otherwise be reading a wider haystack
    than they claim to.
    """
    lines = _workflow_lines()
    start = lines.index('  mongodb-integration:')
    first = next(i for i in range(start + 1, len(lines)) if lines[i].strip().startswith('if:'))
    end = next(i for i in range(first + 1, len(lines)) if len(lines[i]) - len(lines[i].lstrip(' ')) <= 4)
    return ' '.join(' '.join(lines[first:end]).split())


def test_mongodb_integration_is_scoped_to_mongo_changes() -> None:
    condition = _mongodb_condition()

    # Pull requests and branch pushes run the live job only when the Mongo
    # backends change; tags run it unconditionally so releases exercise the live
    # path. `always()` is needed because `changes` is skipped on tag events.
    assert 'always()' in condition
    assert "github.ref_type == 'tag' || needs.changes.outputs.mongodb == 'true'" in condition


def test_mongodb_integration_runs_on_fork_pull_requests() -> None:
    condition = _mongodb_condition()

    # Inverted against `localstack-integration` on purpose. That job skips runs
    # that cannot read repository secrets because LOCALSTACK_AUTH_TOKEN is one and
    # its absence is a hard failure -- a secrets guard, not a security boundary.
    # The mongod service container starts with authentication disabled and this
    # job reads no secret, so an external contributor's pull request can and
    # should get the live signal. Copying the LocalStack condition here would
    # silently withhold it.
    assert 'head.repo.full_name' not in condition
    assert 'dependabot' not in condition
    assert "github.event_name != 'pull_request'" not in condition


def test_mongodb_integration_gates_the_aggregate_check() -> None:
    lines = _workflow_lines()

    # In `needs` so a live failure blocks merges, and in `allowed-skips` so the
    # path filter skipping it does not.
    needs = next(line for line in lines if line.strip().startswith('needs: [') and 'coverage' in line)
    assert 'mongodb-integration' in needs
    assert any('allowed-skips:' in line and 'mongodb-integration' in line for line in lines)


def test_mongodb_detection_covers_both_mongo_modules() -> None:
    lines = _workflow_lines()

    detect_index = lines.index('      - id: detect-mongodb')
    lint_index = lines.index('  lint:')
    detect_block = lines[detect_index:lint_index]

    # `MongoStepStore` externalizes large parts through `MongoMediaStore`, so a
    # change to either module can break the live path.
    assert '            pydantic_ai_harness/step_persistence \\' in detect_block
    assert '            pydantic_ai_harness/media \\' in detect_block
    assert '            integration_tests/mongodb \\' in detect_block
    assert '            .github/workflows/main.yml' in detect_block
    assert any('echo \'mongodb=true\' >> "$GITHUB_OUTPUT"' in line for line in detect_block)


def test_mongodb_integration_runs_against_a_real_server() -> None:
    lines = _workflow_lines()

    job_index = lines.index('  mongodb-integration:')
    coverage_index = lines.index('  coverage:')
    job_block = lines[job_index:coverage_index]

    # A service container, plus the env var that turns an unreachable one into a
    # failure. Without it the suite skips and the job reports green having
    # exercised nothing -- which is the failure mode this whole job exists to
    # close, since the unit tests already pass against `mongomock-motor`.
    assert '        image: mongo:8' in job_block
    assert any('--health-cmd' in line for line in job_block)
    assert "      MONGODB_REQUIRE_LIVE: '1'" in job_block
    assert '      - run: make integration-mongodb' in job_block
