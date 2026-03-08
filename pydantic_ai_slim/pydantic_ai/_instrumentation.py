from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

DEFAULT_INSTRUMENTATION_VERSION = 2
"""Default instrumentation version for `InstrumentationSettings`.

Versions:
- 1: Original span/attribute names (e.g. 'agent run', 'tool_arguments').
- 2: Same names as v1 but with additional attributes.
- 3: GenAI semantic convention names (e.g. 'invoke_agent', 'gen_ai.tool.call.arguments').
- 4: Like v3 but with GenAI semantic conventions for multimodal content (URI/blob parts, modality fields).
- 5: Like v4 but CallDeferred/ApprovalRequired no longer produce ERROR spans (opt-in).
"""


@dataclass(frozen=True)
class InstrumentationNames:
    """Configuration for instrumentation span names and attributes based on version."""

    # Agent run span configuration
    agent_run_span_name: str
    agent_name_attr: str

    # Tool execution span configuration
    tool_span_name: str
    tool_arguments_attr: str
    tool_result_attr: str

    # Output Tool execution span configuration
    output_tool_span_name: str

    # Deferral span attributes (CallDeferred / ApprovalRequired)
    tool_deferral_name_attr: str
    tool_deferral_metadata_attr: str

    @classmethod
    def for_version(cls, version: int) -> Self:
        """Create instrumentation configuration for a specific version.

        Args:
            version: The instrumentation version (1, 2, 3, or 4+)

        Returns:
            InstrumentationConfig instance with version-appropriate settings
        """
        if version <= 2:
            return cls(
                agent_run_span_name='agent run',
                agent_name_attr='agent_name',
                tool_span_name='running tool',
                tool_arguments_attr='tool_arguments',
                tool_result_attr='tool_response',
                output_tool_span_name='running output function',
                tool_deferral_name_attr='pydantic_ai.tool.deferral.name',
                tool_deferral_metadata_attr='pydantic_ai.tool.deferral.metadata',
            )
        else:
            # Version 3 and 4+ share the same span/attribute names.
            # The only difference between v3 and v4 is behavioral (gated in _tool_manager.py):
            # v4 suppresses ERROR status for CallDeferred/ApprovalRequired.
            return cls(
                agent_run_span_name='invoke_agent',
                agent_name_attr='gen_ai.agent.name',
                tool_span_name='execute_tool',  # Will be formatted with tool name
                tool_arguments_attr='gen_ai.tool.call.arguments',
                tool_result_attr='gen_ai.tool.call.result',
                output_tool_span_name='execute_tool',
                tool_deferral_name_attr='pydantic_ai.tool.deferral.name',
                tool_deferral_metadata_attr='pydantic_ai.tool.deferral.metadata',
            )

    def get_agent_run_span_name(self, agent_name: str) -> str:
        """Get the formatted agent span name.

        Args:
            agent_name: Name of the agent being executed

        Returns:
            Formatted span name
        """
        if self.agent_run_span_name == 'invoke_agent':
            return f'invoke_agent {agent_name}'
        return self.agent_run_span_name

    def get_tool_span_name(self, tool_name: str) -> str:
        """Get the formatted tool span name.

        Args:
            tool_name: Name of the tool being executed

        Returns:
            Formatted span name
        """
        if self.tool_span_name == 'execute_tool':
            return f'execute_tool {tool_name}'
        return self.tool_span_name

    def get_output_tool_span_name(self, tool_name: str) -> str:
        """Get the formatted output tool span name.

        Args:
            tool_name: Name of the tool being executed

        Returns:
            Formatted span name
        """
        if self.output_tool_span_name == 'execute_tool':
            return f'execute_tool {tool_name}'
        return self.output_tool_span_name
