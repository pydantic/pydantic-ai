"""Workspace setup and launcher failures through LocalStack's public agent API."""

from pathlib import Path
from shutil import which

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import LocalStack


class TestLocalStackWorkspace:
    @pytest.mark.parametrize('utility', ['env', 'sed'])
    @pytest.mark.parametrize('exit_code', [None, 2, 127])
    async def test_scrub_failure_does_not_launch_cli(self, tmp_path: Path, utility: str, exit_code: int | None) -> None:
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        for program in ('sh', 'env', 'sed'):
            if program != utility:
                executable = which(program)
                assert executable is not None
                (bin_dir / program).symlink_to(executable)
        if exit_code is not None:
            failed_utility = bin_dir / utility
            failed_utility.write_text(f'#!/bin/sh\nexit {exit_code}\n')
            failed_utility.chmod(0o755)

        cli = tmp_path / 'aws'
        cli.write_text('#!/bin/sh\necho ran > cli-ran\n')
        cli.chmod(0o755)
        agent = Agent(
            TestModel(custom_output_text='done', call_tools=['aws_cli']),
            capabilities=[
                LocalWorkspace(
                    tmp_path,
                    env={'PATH': str(bin_dir), 'AWS_PROFILE': 'prod', 'AWS_SESSION_TOKEN': 'prod-token'},
                ),
                LocalStack(aws_cli_path=str(cli)),
            ],
        )

        result = await agent.run('List the buckets.')

        assert not (tmp_path / 'cli-ran').exists()
        outputs = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'aws_cli'
        ]
        assert len(outputs) == 1
        output = outputs[0]
        assert isinstance(output, str)
        assert '[stderr]' in output
        assert 'Could not' in output
        assert '[exit code: 1]' in output

    @pytest.mark.parametrize(
        'doc_path',
        [
            'docs/harness/localstack.md',
            'src/pydantic_ai_harness/pydantic_ai_harness/localstack/README.md',
        ],
    )
    async def test_documented_agent_spec_runs(
        self, request: pytest.FixtureRequest, tmp_path: Path, doc_path: str
    ) -> None:
        document = (request.config.rootpath / doc_path).read_text()
        spec = document.split('```yaml\n', 1)[1].split('```', 1)[0]
        spec_file = tmp_path / 'agent.yaml'
        spec_file.write_text(spec)
        agent = Agent.from_file(spec_file, custom_capability_types=[LocalStack], defer_model_check=True)

        result = await agent.run('Say done.', model=TestModel(custom_output_text='done', call_tools=[]))

        assert result.output == 'done'
