from __future__ import annotations

from pathlib import Path


def _workflow_lines() -> list[str]:
    workflow = Path(__file__).parents[2] / '.github' / 'workflows' / 'main.yml'
    return workflow.read_text().splitlines()


def test_localstack_ci_lets_the_capability_manage_the_container() -> None:
    lines = _workflow_lines()
    action_index = lines.index(
        '      - uses: LocalStack/setup-localstack@7c8a0cb3405bc58be4c8f763f812aa000bc46303 # v0.3.2'
    )
    action_block = lines[action_index : action_index + 4]

    assert "          skip-startup: 'true'" in action_block


def test_localstack_ci_does_not_advertise_an_external_endpoint() -> None:
    lines = _workflow_lines()

    assert not any('LOCALSTACK_ENDPOINT_URL:' in line for line in lines)


def test_localstack_ci_authenticates_with_an_auth_token() -> None:
    lines = _workflow_lines()

    # The single image requires a token since LocalStack 2026.03.0; the deprecated
    # LOCALSTACK_ACKNOWLEDGE_ACCOUNT_REQUIREMENT bypass expired and must not return.
    assert "      LOCALSTACK_REQUIRE_AUTH_TOKEN: '1'" in lines
    assert not any('LOCALSTACK_ACKNOWLEDGE_ACCOUNT_REQUIREMENT' in line for line in lines)


def test_localstack_ci_gates_the_aggregate_check() -> None:
    lines = _workflow_lines()

    # The aggregate `check` job must depend on localstack-integration so a live
    # test failure blocks merges rather than passing silently.
    needs = next(line for line in lines if line.strip().startswith('needs: [') and 'coverage' in line)
    assert 'localstack-integration' in needs


def test_localstack_integration_is_scoped_to_localstack_changes() -> None:
    lines = _workflow_lines()

    # On a pull request the live job runs only when LocalStack paths change, so
    # unrelated PRs don't pull the image, install the AWS CLI, or spend the token.
    assert any('needs.changes.outputs.localstack' in line for line in lines)
    assert any("- 'pydantic_ai_harness/localstack/**'" in line for line in lines)

    # A skipped live job must not fail the required aggregate check; a job that
    # actually runs and fails still votes (see allowed-skips semantics).
    assert any('allowed-skips: changes, localstack-integration' in line for line in lines)


def test_changes_job_can_read_pull_request_files() -> None:
    lines = _workflow_lines()

    # paths-filter runs without a checkout, so it lists PR files via the GitHub
    # API, which needs `pull-requests: read`. Without it the job fails on private
    # repos (or under tightened default token scopes) and the live job is skipped.
    changes_index = lines.index('  changes:')
    lint_index = lines.index('  lint:')
    changes_block = lines[changes_index:lint_index]
    assert '      pull-requests: read' in changes_block


def test_localstack_ci_scopes_the_auth_token_to_the_test_step() -> None:
    lines = _workflow_lines()

    # The token is scoped to the integration-test step, not the job-level env, so
    # the checkout and setup steps never receive the secret.
    run_index = lines.index('      - run: make integration-localstack')
    step_block = lines[run_index : run_index + 4]
    assert '          LOCALSTACK_AUTH_TOKEN: ${{ secrets.LOCALSTACK_AUTH_TOKEN }}' in step_block
    assert '      LOCALSTACK_AUTH_TOKEN: ${{ secrets.LOCALSTACK_AUTH_TOKEN }}' not in lines
