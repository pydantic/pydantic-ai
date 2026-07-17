from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('inventory_langchain.py')


class InventoryLangChainTests(unittest.TestCase):
    """Exercise inventory detection and failure behavior."""

    def run_inventory(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(root), '--format', 'json', *args],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_aliases_are_detected_and_unrelated_methods_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / 'agent.py').write_text(
                """
from langchain.agents import create_agent as make_agent
from langchain.tools import tool as lc_tool

@lc_tool
def lookup(query: str) -> str:
    return query

agent = make_agent(model='provider:model', tools=[lookup])
database.invoke('not a LangChain runnable')
agent.invoke({'messages': []})
""".lstrip()
            )

            completed = self.run_inventory(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            symbols = {finding['symbol'] for finding in report['findings']}
            self.assertIn('make_agent', symbols)
            self.assertIn('lc_tool', symbols)
            self.assertIn('agent.invoke', symbols)
            self.assertNotIn('database.invoke', symbols)

    def test_dependency_config_and_notebook_are_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / 'pyproject.toml').write_text('[project]\ndependencies = ["langchain>=1"]\n')
            (root / 'setup.cfg').write_text('install_requires = langgraph\n')
            (root / 'requirements-dev.in').write_text('langsmith\n')
            (root / 'pdm.lock').write_text('name = "deepagents"\n')
            (root / 'example.ipynb').write_text('{"cells": [{"cell_type": "code", "source": ["import langgraph"]}]}')

            completed = self.run_inventory(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report['config_files_scanned'], 5)
            self.assertEqual(report['category_counts']['dependency-config'], 4)
            self.assertEqual(report['category_counts']['notebook'], 1)
            self.assertEqual(
                report['scanner_python'],
                f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
            )
            self.assertEqual(report['routing'][0]['skill'], 'migrate-deep-agents-to-pydantic-ai')

    def test_deep_agent_usage_routes_to_the_dedicated_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / 'deep_agent.py').write_text(
                "from deepagents import create_deep_agent\nagent = create_deep_agent(model='provider:model')\n"
            )

            completed = self.run_inventory(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report['category_counts']['harness'], 2)
            self.assertEqual(
                report['routing'],
                [
                    {
                        'condition': 'Deep Agents usage detected',
                        'skill': 'migrate-deep-agents-to-pydantic-ai',
                        'reason': (
                            'Use the dedicated skill for create_deep_agent, Harness, sandbox, and deployment semantics.'
                        ),
                    }
                ],
            )

    def test_framework_neutral_runtime_call_is_reported_as_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / 'factory.py').write_text(
                "from langchain.agents import create_agent\nruntime = create_agent(model='provider:model', tools=[])\n"
                'app = runtime\n'
                'workflow = runtime\n'
                'executor = runtime\n'
            )
            (root / 'api.py').write_text(
                'from factory import app, executor, runtime, workflow\n\n'
                'async def events():\n'
                "    await app.ainvoke({'messages': []})\n"
                "    async for item in workflow.astream({'messages': []}):\n"
                '        yield item\n'
                "    executor.invoke({'messages': []})\n"
                "    async for event in runtime.astream_events({'messages': []}):\n"
                '        yield event\n'
                "    database.invoke('not a framework runtime')\n"
            )

            completed = self.run_inventory(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            candidates = [finding for finding in report['findings'] if finding['kind'] == 'candidate-call']
            self.assertEqual(
                [finding['symbol'] for finding in candidates],
                ['app.ainvoke', 'workflow.astream', 'executor.invoke', 'runtime.astream_events'],
            )
            self.assertEqual(
                [finding['category'] for finding in candidates],
                [
                    'candidate-invocation',
                    'candidate-streaming',
                    'candidate-invocation',
                    'candidate-streaming',
                ],
            )

    def test_python_encoding_declaration_is_honored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / 'latin1_agent.py').write_bytes(
                b"# -*- coding: latin-1 -*-\nfrom langchain import agents\nlabel = '\xe9'\n"
            )

            completed = self.run_inventory(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report['parse_errors'], [])
            self.assertEqual(report['findings'][0]['symbol'], 'langchain: agents')

    def test_parse_errors_fail_by_default_and_can_be_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / 'broken.py').write_text('def broken(:\n')

            strict = self.run_inventory(root)
            exploratory = self.run_inventory(root, '--allow-parse-errors')

            self.assertEqual(strict.returncode, 2)
            self.assertEqual(exploratory.returncode, 0)
            parse_errors = json.loads(strict.stdout)['parse_errors']
            self.assertEqual(len(parse_errors), 1)
            self.assertIn('newer Python grammar', parse_errors[0]['hint'])


if __name__ == '__main__':
    unittest.main()
