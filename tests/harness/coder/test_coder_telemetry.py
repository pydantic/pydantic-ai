"""What a `Coder` run records about its capabilities on the agent run span.

Not a VCR test: the claim is about the `pydantic_ai.capabilities` span attribute of an in-memory model run,
which no provider exchange captures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter

from pydantic_ai import Agent
from pydantic_ai._instrumentation import CapabilityRecord
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.coder import Coder
from tests.conftest import try_import

with try_import() as otel_sdk_installed:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = [pytest.mark.skipif(not otel_sdk_installed(), reason='opentelemetry-sdk not installed')]

_RECORDS = TypeAdapter(dict[str, list[CapabilityRecord]])


async def _recorded_capabilities(tmp_path: Path, *, include_content: bool) -> dict[str, list[CapabilityRecord]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentation = Instrumentation(
        settings=InstrumentationSettings(include_content=include_content, tracer_provider=provider)
    )
    agent = Agent(TestModel(call_tools=[], custom_output_text='done'), capabilities=[Coder(), instrumentation])

    await agent.run('go', capabilities=[LocalWorkspace(tmp_path)])

    (span,) = [span for span in exporter.get_finished_spans() if span.name.startswith('invoke_agent')]
    # A run-assigned id ends in a random suffix that differs every run.
    return _RECORDS.validate_json(
        re.sub(r':[0-9a-f]{6}>', ':*>', str((span.attributes or {})['pydantic_ai.capabilities']))
    )


async def test_a_coder_run_records_coder_with_its_members(tmp_path: Path) -> None:
    # `Coder` is the capability the user configured, so the run records it once, with the capabilities it
    # registers as its members rather than beside it. Its unnamed private pieces roll up into it, and
    # `FileSystem` holds the `AnnounceWorkingDirectory` its `for_run` brings along.
    assert await _recorded_capabilities(tmp_path, include_content=True) == snapshot(
        {
            'pydantic_ai.capabilities.Instrumentation': [
                {'id': 'instrumentation', 'defer_loading': False, 'active': True}
            ],
            'pydantic_ai.capabilities.ToolSearch': [{'id': 'tool_search', 'defer_loading': False, 'active': True}],
            'pydantic_ai_harness.coder.Coder': [
                {
                    'defer_loading': False,
                    'active': True,
                    'members': {
                        'pydantic_ai.capabilities.AbstractCapability': [
                            {'id': 'coder_instructions', 'defer_loading': False, 'active': True}
                        ],
                        'pydantic_ai_harness.file_system.FileSystem': [
                            {
                                'id': '<file_system:*>',
                                'defer_loading': False,
                                'active': True,
                                'members': {
                                    'pydantic_ai_harness.announce_working_directory.AnnounceWorkingDirectory': [
                                        {'id': 'announce_working_directory', 'defer_loading': False, 'active': True}
                                    ]
                                },
                            }
                        ],
                        'pydantic_ai_harness.shell.Shell': [
                            {'id': '<shell:*>', 'defer_loading': False, 'active': True}
                        ],
                        'pydantic_ai_harness.repo_context.RepoContext': [
                            {'id': '<repo_context:*>', 'defer_loading': False, 'active': True}
                        ],
                        'pydantic_ai_harness.sub_agents.SubAgents': [
                            {'id': 'sub_agents', 'defer_loading': False, 'active': True}
                        ],
                        'pydantic_ai_harness.compaction.ClearToolResults': [
                            {'id': '<clear_tool_results:*>', 'defer_loading': False, 'active': True}
                        ],
                        'pydantic_ai_harness.tool_output_limits.ToolOutputLimits': [
                            {'id': 'coder_tool_output_limits', 'defer_loading': False, 'active': True}
                        ],
                        'pydantic_ai_harness.repair_tool_arguments.RepairToolArguments': [
                            {'id': '<repair_tool_arguments:*>', 'defer_loading': False, 'active': True}
                        ],
                    },
                }
            ],
            'pydantic_ai.capabilities.LocalWorkspace': [
                {'id': 'local_workspace', 'defer_loading': False, 'active': True}
            ],
        }
    )


async def test_a_coder_run_without_content_records_only_chosen_ids(tmp_path: Path) -> None:
    capabilities = await _recorded_capabilities(tmp_path, include_content=False)

    assert list(capabilities) == snapshot(
        [
            'pydantic_ai.capabilities.Instrumentation',
            'pydantic_ai.capabilities.ToolSearch',
            'pydantic_ai_harness.coder.Coder',
            'pydantic_ai.capabilities.LocalWorkspace',
        ]
    )
    assert capabilities['pydantic_ai_harness.coder.Coder'][0].get('members') == snapshot(
        {
            'pydantic_ai.capabilities.AbstractCapability': [
                {'id': 'coder_instructions', 'defer_loading': False, 'active': True}
            ],
            'pydantic_ai_harness.file_system.FileSystem': [
                {
                    'defer_loading': False,
                    'active': True,
                    'members': {
                        'pydantic_ai_harness.announce_working_directory.AnnounceWorkingDirectory': [
                            {'id': 'announce_working_directory', 'defer_loading': False, 'active': True}
                        ]
                    },
                }
            ],
            'pydantic_ai_harness.shell.Shell': [{'defer_loading': False, 'active': True}],
            'pydantic_ai_harness.repo_context.RepoContext': [{'defer_loading': False, 'active': True}],
            'pydantic_ai_harness.sub_agents.SubAgents': [{'id': 'sub_agents', 'defer_loading': False, 'active': True}],
            'pydantic_ai_harness.compaction.ClearToolResults': [{'defer_loading': False, 'active': True}],
            'pydantic_ai_harness.tool_output_limits.ToolOutputLimits': [
                {'id': 'coder_tool_output_limits', 'defer_loading': False, 'active': True}
            ],
            'pydantic_ai_harness.repair_tool_arguments.RepairToolArguments': [{'defer_loading': False, 'active': True}],
        }
    )
