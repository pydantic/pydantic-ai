"""Run the review gates on the pull request a successful `CI` run tested, or backtest them.

`review-gates.yml` runs `python -m review_gates run` from the default branch's checkout, so this code,
the reach map and the sibling files it reads are the default branch's. The pull request is resolved and
read through the API; none of its code is checked out, installed, imported or run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import httpx2
import logfire

from pydantic_ai.models import infer_model

from .backtest import backtest
from .checks import check_output, conclusion
from .gates import GATES, MODEL, GateKey, evaluate
from .github import CheckOutput, GitHub, PullRequest
from .prep import prepare
from .reach import reach_map


async def run(env: Mapping[str, str], root: Path) -> int:
    """Judge the PR the triggering `CI` run tested, unless it is gone or has moved on."""
    head_sha = env['HEAD_SHA']
    github = GitHub.connect(env['GITHUB_REPOSITORY'], env['GITHUB_TOKEN'])
    try:
        pull = await resolve(github, env['HEAD_REPOSITORY'], env['HEAD_BRANCH'], head_sha)
        if isinstance(pull, str):
            print(f'Not running the review gates: {pull}')
            return 0
        with logfire.span('review gates on PR {number}', number=pull.number, head_sha=head_sha):
            await judge_and_report(github, pull, head_sha, root)
        return 0
    finally:
        await github.aclose()


async def resolve(github: GitHub, head_repository: str, head_branch: str, head_sha: str) -> PullRequest | str:
    """The one open PR whose head CI tested, or why there is none to judge."""
    owner = head_repository.split('/', 1)[0]
    pulls = [
        p
        for p in await github.open_pull_requests(f'{owner}:{head_branch}')
        if p.head.repo is not None and p.head.repo.full_name == head_repository
    ]
    if len(pulls) != 1:
        return f'{len(pulls)} open PRs have this head branch; cannot tell which one CI ran for'
    if pulls[0].head.sha != head_sha:
        return f'PR #{pulls[0].number} moved on to a new head'
    return pulls[0]


async def judge_and_report(github: GitHub, pull: PullRequest, head_sha: str, root: Path) -> None:
    """Post a check run for each gate not yet reported on this head with its current question."""
    pending: dict[GateKey, str] = {}
    for key, gate in GATES.items():
        external_id = f'{key}:{head_sha}:{gate.fingerprint}'
        # Only a completed run counts: one left `in_progress` by a killed job is judged again.
        runs = await github.check_runs(head_sha, gate.name)
        if not any(run.status == 'completed' and run.external_id == external_id for run in runs):
            pending[key] = external_id
    if not pending:
        print('Every gate has already reported on this head.')
        return
    started: dict[GateKey, int] = {
        key: await github.start_check_run(name=GATES[key].name, head_sha=head_sha, external_id=external_id)
        for key, external_id in pending.items()
    }
    try:
        prepared = await prepare(github, pull, head_sha)
        results = await evaluate(prepared, infer_model(MODEL), root=root, reach=reach_map(root), keys=set(started))
    except BaseException as e:
        # A check left `in_progress` reads as still running forever, so close each before the error propagates,
        # each on its own, under an external id a rerun does not count as reported.
        for key, check_run_id in started.items():
            try:
                await github.finish_check_run(
                    check_run_id,
                    conclusion='neutral',
                    output={'title': 'The gate failed before judging', 'summary': type(e).__name__, 'annotations': []},
                    external_id=f'{pending[key]}:failed',
                )
            except httpx2.HTTPError:
                print(f'Could not close the {key} check run {check_run_id}')
        raise
    moved = (await github.pull_request(pull.number)).head.sha != head_sha
    for result in results:
        output: CheckOutput = check_output(result)
        if moved:
            output = {
                'title': 'Skipped: the PR moved on to a new head',
                'summary': output['summary'],
                'annotations': [],
            }
        verdict = 'skipped' if moved else conclusion(result)
        logfire.info(
            '{gate} concluded {verdict}',
            gate=result.gate,
            verdict=verdict,
            outcomes=[j.outcome for j in result.judgements],
            notes=result.notes,
        )
        await github.finish_check_run(started[result.gate], conclusion=verdict, output=output)


def main() -> int:
    """The command line: `run` in CI, `backtest` locally."""
    parser = argparse.ArgumentParser(prog='review_gates')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('run', help='judge the PR the triggering CI run tested and post its check runs')
    replay = commands.add_parser('backtest', help='judge a labelled set of historical PR heads; posts nothing')
    replay.add_argument('--set', type=Path, default=Path(__file__).parent / 'backtest_set.json')
    replay.add_argument('--out', type=Path, required=True, help='where to write the per-item JSONL')
    args = parser.parse_args()
    if token := os.environ.get('LOGFIRE_TOKEN'):
        logfire.configure(token=token, service_name='review-gates', environment='github-actions', console=False)
        logfire.instrument_pydantic_ai()
    root = Path.cwd()
    if args.command == 'run':
        return asyncio.run(run(os.environ, root))
    return asyncio.run(backtest(args.set, args.out, os.environ['GITHUB_TOKEN'], root))


if __name__ == '__main__':
    sys.exit(main())
