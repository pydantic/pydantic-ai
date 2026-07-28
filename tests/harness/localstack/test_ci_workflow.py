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


def _localstack_condition() -> str:
    """The `localstack-integration` job's `if:` expression, collapsed to one line.

    The expression is folded across several lines for readability, so match on its
    content rather than its formatting.
    """
    lines = _workflow_lines()
    start = lines.index('  localstack-integration:')
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith('    steps:'))
    block = ' '.join(line.strip() for line in lines[start:end] if not line.strip().startswith('#'))
    condition = block.split('if:', 1)[1]
    return ' '.join(condition.split())


def test_localstack_integration_is_scoped_to_localstack_changes() -> None:
    lines = _workflow_lines()
    condition = _localstack_condition()

    # Pull requests and branch pushes run the live job only when LocalStack paths
    # change; tags run it unconditionally so releases exercise the live path.
    assert "    if: github.event_name == 'pull_request' || github.ref_type == 'branch'" in lines
    assert 'always()' in condition
    assert "github.ref_type == 'tag' || needs.changes.outputs.localstack == 'true'" in condition

    # A skipped live job must not fail the required aggregate check; a job that
    # actually runs and fails still votes (see allowed-skips semantics).
    assert any('allowed-skips: changes, localstack-integration' in line for line in lines)


def test_localstack_integration_skips_pull_requests_without_secrets() -> None:
    condition = _localstack_condition()

    # LOCALSTACK_AUTH_TOKEN comes from repository secrets, and
    # LOCALSTACK_REQUIRE_AUTH_TOKEN turns an empty token into a failure rather than a
    # skip. `check` tolerates a skip but not a failure, so a pull request that cannot
    # read secrets goes red for a reason unrelated to its changes.
    assert "github.event_name != 'pull_request'" in condition

    # Cross-repository (fork) branches never receive repository secrets.
    assert 'github.event.pull_request.head.repo.full_name == github.repository' in condition

    # Dependabot reads a separate secrets store, and its branches live in this repo,
    # so the head-repo check alone would let those runs through.
    assert "github.actor != 'dependabot[bot]'" in condition


def test_changes_job_uses_owned_git_diff_with_read_only_checkout() -> None:
    lines = _workflow_lines()

    changes_index = lines.index('  changes:')
    lint_index = lines.index('  lint:')
    changes_block = lines[changes_index:lint_index]

    assert not any('dorny/paths-filter' in line for line in changes_block)
    assert '      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2' in changes_block
    assert '          fetch-depth: 0' in changes_block
    assert '          persist-credentials: false' in changes_block
    assert '          if git diff --quiet "$BASE_SHA" "$HEAD_SHA" -- \\' in changes_block
    assert '            pydantic_ai_harness/localstack \\' in changes_block
    assert '            tests/localstack \\' in changes_block
    assert '            integration_tests/localstack \\' in changes_block
    assert '            Makefile \\' in changes_block
    assert '            pyproject.toml \\' in changes_block
    assert '            uv.lock \\' in changes_block
    assert '            .github/workflows/main.yml' in changes_block


def test_localstack_ci_scopes_the_auth_token_to_the_test_step() -> None:
    lines = _workflow_lines()

    # The token is scoped to the integration-test step, not the job-level env, so
    # the checkout and setup steps never receive the secret. The job must not use
    # a protected environment that creates deployment approval notifications.
    run_index = lines.index('      - run: make integration-localstack')
    step_block = lines[run_index : run_index + 5]
    assert (
        '          LOCALSTACK_AUTH_TOKEN: ${{ secrets.LOCALSTACK_AUTH_TOKEN }} # zizmor: ignore[secrets-outside-env]'
    ) in step_block
    assert '      LOCALSTACK_AUTH_TOKEN: ${{ secrets.LOCALSTACK_AUTH_TOKEN }}' not in lines
    assert '    environment: localstack-integration' not in lines
