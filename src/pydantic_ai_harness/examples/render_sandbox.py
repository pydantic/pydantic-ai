"""Run a real model-driven repair, independently check it, and verify cleanup.

Environment: RENDER_API_KEY, RENDER_WORKSPACE_ID, ANTHROPIC_API_KEY.
The host verifies the repair in a fresh sandbox before reporting success.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from render import RenderAsync
from render.experimental.sandbox.errors import SandboxNotFoundError

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.usage import UsageLimits
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness import Coder, FileSystem, Shell
from pydantic_ai_harness.render_sandbox import (
    RenderSandbox,
    RenderSandboxBackend,
    RenderSandboxConfig,
    RenderSandboxEvent,
)

from .render_sandbox_fixture import HELD_OUT_TESTS, SOURCE, VISIBLE_TESTS

TASK = """Fix invoice_total in totals.py. First run python3 -m unittest -v test_totals
and observe the defect before editing. Read the implementation. Each line has
nonnegative integer unit_price_cents and quantity. Return the exact integer sum
of price times quantity, including empty invoices, without changing input.
Do not edit test_totals.py. Use the standard write_file tool for the fix, then run the
visible tests again. Work only in the sandbox. Do not run background commands,
install packages, deploy, or contact external services. End with the test result.
"""


def arguments() -> argparse.Namespace:
    """Select the model, standard tool composition, and local evidence destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=os.getenv('PYDANTIC_MODEL', 'anthropic:claude-sonnet-4-6'))
    parser.add_argument('--harness', choices=('coder', 'tools'), default='coder')
    parser.add_argument('--evidence', type=Path, default=Path('.cache/render-sandboxes/workspace-repair.json'))
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    """Repair a project, independently verify the candidate, and audit owned resources."""
    token = os.environ.get('RENDER_API_KEY')
    workspace = os.environ.get('RENDER_WORKSPACE_ID')
    if not token or not workspace:
        raise ValueError('Set RENDER_API_KEY and RENDER_WORKSPACE_ID on the host')
    client = RenderAsync(token=token, owner_id=workspace, region='oregon')
    report: dict[str, object] = {
        'status': 'running',
        'started_at': datetime.now(UTC).isoformat(),
        'model': args.model,
        'workspace_id': workspace,
        'harness': args.harness,
        'versions': {
            name: version(name) for name in ('render', 'pydantic-ai-slim', 'pydantic-graph', 'pydantic-ai-harness')
        },
    }
    report['anthropic_version'] = version('anthropic')
    events: list[dict[str, object]] = []
    report['events'] = events
    ids: set[str] = set()
    args.evidence.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.evidence.write_text(json.dumps(report, indent=2))

    def record(event: RenderSandboxEvent) -> None:
        events.append(asdict(event))
        if event.sandbox_id:
            ids.add(event.sandbox_id)
        save()

    async def verify(candidate_workspace: Workspace) -> None:
        candidate = await candidate_workspace.read_bytes('totals.py')
        observed_tests = await candidate_workspace.read_bytes('test_totals.py')
        if observed_tests != VISIBLE_TESTS:
            raise AssertionError('The agent changed the protected visible tests')
        assert candidate_workspace.ref is not None
        report['agent_sandbox_id'] = candidate_workspace.ref.id
        report['candidate_sha256'] = hashlib.sha256(candidate).hexdigest()
        report['protected_tests_unchanged'] = True
        # Generated source is only executed in a fresh remote environment. Original
        # checks and their imports are not trusted from the agent-editable sandbox.
        verifier = RenderSandboxBackend(
            client, config=RenderSandboxConfig(lifetime_seconds=300), on_sandbox_event=record
        )
        try:
            checked_workspace = Workspace(verifier)
            await checked_workspace.working_dir()
            report['verifier_sandbox_id'] = verifier.ref.id if verifier.ref else None
            assert verifier.ref is not None and verifier.ref.id != candidate_workspace.ref.id
            await checked_workspace.write_bytes('totals.py', SOURCE)
            await checked_workspace.write_bytes('test_totals.py', VISIBLE_TESTS)
            await checked_workspace.write_bytes('test_acceptance.py', HELD_OUT_TESTS)
            original = await checked_workspace.run(
                ['python3', '-m', 'unittest', '-v', 'test_totals', 'test_acceptance'], timeout=30
            )
            assert original.exit_code != 0, 'The held-out checks failed to detect the original defect'
            report['original_defect_exit_code'] = original.exit_code
            await checked_workspace.write_bytes('totals.py', candidate)
            checked = await checked_workspace.run(
                ['python3', '-m', 'unittest', '-v', 'test_totals', 'test_acceptance'], timeout=30
            )
            report['independent_verification'] = asdict(checked)
            assert checked.exit_code == 0 and 'Ran 7 tests' in checked.stderr, 'Candidate failed independent checks'
            report['candidate_source'] = candidate.decode('utf-8')
            save()
        finally:
            await verifier.destroy()

    capability = RenderSandbox(
        client=client,
        config=RenderSandboxConfig(lifetime_seconds=600),
        files={'totals.py': SOURCE, 'test_totals.py': VISIBLE_TESTS},
        on_finish=verify,
        on_sandbox_event=record,
    )
    # Standard Harness capabilities use ctx.workspace. The Render capability
    # supplies that workspace and contributes no provider-specific model tools.
    harness = [Coder(sub_agents=False)] if args.harness == 'coder' else [Shell(), FileSystem()]
    agent = Agent(
        args.model,
        capabilities=[capability, *harness],
        # A test depends on the preceding write. Prevent models from issuing both
        # in one response, where standard tools would otherwise run concurrently.
        model_settings={'parallel_tool_calls': False},
    )
    try:
        async with asyncio.timeout(480):
            result = await agent.run(TASK, usage_limits=UsageLimits(request_limit=15))
        tool_results: list[dict[str, str]] = []
        for message in result.all_messages():
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, ToolReturnPart):
                        tool_results.append({'tool': part.tool_name, 'result': str(part.content)})
        names = [item['tool'] for item in tool_results]
        report['agent_tool_names'] = names
        report['agent_tool_results'] = tool_results
        report['agent_output'] = str(result.output)
        report['model_requests'] = result.usage.requests
        report['workspace_ref'] = asdict(result.workspace.ref) if result.workspace.ref else None
        shell_results = [item['result'] for item in tool_results if item['tool'] in {'shell', 'run_command'}]
        assert shell_results and 'write_file' in names
        assert not any(name.startswith('sandbox_') for name in names), 'Unexpected custom tools'
        assert any('FAILED' in output for output in shell_results), 'Agent did not observe failing tests'
        assert any('Ran 2 tests' in output and 'OK' in output for output in shell_results), (
            'Agent did not rerun passing tests'
        )
        cleanup: dict[str, object] = {}
        for sandbox_id in sorted(ids):
            try:
                sandbox = await asyncio.wait_for(client.experimental.sandboxes.from_id(sandbox_id), 20)
                cleanup[sandbox_id] = {
                    'status': sandbox.status,
                    'terminated_at': str(sandbox.terminated_at),
                }
                assert sandbox.status == 'terminated', 'Owned sandbox is not terminated'
            except SandboxNotFoundError:
                cleanup[sandbox_id] = {'status': 'not_found_after_owned_cleanup'}
        report['cleanup_verification'] = cleanup
        report['status'] = 'passed'
        print(
            'PASS: real Pydantic AI agent repaired totals.py using standard '
            + ('Coder' if args.harness == 'coder' else 'Shell + FileSystem')
            + ' tools'
        )
        print('PASS: original defect rejected; seven independent tests passed')
        print('PASS: owned sandbox termination verified')
        print('Evidence:', args.evidence)
    except BaseException as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        raise
    finally:
        report['finished_at'] = datetime.now(UTC).isoformat()
        save()
        await capability.aclose()


if __name__ == '__main__':
    asyncio.run(main(arguments()))
