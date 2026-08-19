"""Regression tests for the cross-repository docs preview dispatcher."""

from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / 'workflows' / 'docs-preview.yml'


def test_privileged_dispatcher_never_executes_pull_request_code() -> None:
    """The secret-bearing public workflow must remain a dispatcher only."""
    workflow = WORKFLOW.read_text()

    assert 'pull_request_target:' in workflow
    assert "github.event.label.name == 'trigger:docs'" in workflow
    assert 'actions/checkout@' not in workflow
    assert '/collaborators/${ACTOR}/permission' in workflow
    assert 'admin|maintain|write)' in workflow
    assert 'permission-contents: write' in workflow
    assert workflow.index('Verify a maintainer triggered the preview') < workflow.index('Generate app token')


def test_public_comments_do_not_disclose_private_workflow_links() -> None:
    """Public comments should not link users to inaccessible private runs."""
    workflow = WORKFLOW.read_text()

    assert 'github.com/pydantic/unified-docs' not in workflow
    assert 'The preview build for commit' in workflow
    assert 'has been queued' in workflow
