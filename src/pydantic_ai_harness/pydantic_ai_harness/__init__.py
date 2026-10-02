"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

from ._mcp import MCPReadOnlyNoToolsWarning
from ._warn import HarnessDeprecationWarning

if TYPE_CHECKING:
    from .absurd import AbsurdDurability
    from .advisor import Advisor
    from .ask_user import AskUser
    from .background_tools import BackgroundTools
    from .browser_use import BrowserUse
    from .bubblewrap_sandbox import BubblewrapSandbox, BubblewrapWorkspace
    from .capability_creation import CapabilityCreation
    from .code_mode import CodeMode
    from .coder import Coder
    from .compaction import (
        ClampOversizedMessages,
        ClearToolResults,
        DeduplicateFileReads,
        FallbackCompaction,
        ReportContextUsage,
        SlidingWindowCompaction,
        SummarizingCompaction,
        TieredCompaction,
        WarnNearLimits,
    )
    from .conversation_search import ConversationSearch
    from .day_ai import DayAI
    from .dynamic_workflow import DynamicWorkflow
    from .e2b_sandbox import E2BSandbox, E2BSandboxBackend
    from .exa import ExaAgent, ExaSearch
    from .filesystem import READ_ONLY_TOOL_NAMES, FileSystem
    from .grain import Grain
    from .guardrails import (
        GuardrailError,
        GuardrailResult,
        InputBlocked,
        InputGuardrail,
        InputGuardrailFunc,
        OutputBlocked,
        OutputGuardrail,
        OutputGuardrailFunc,
        ToolGuardrail,
    )
    from .localstack import LocalStack
    from .logfire import ManagedPrompt
    from .macroscope import Macroscope
    from .memory import Memory
    from .modal_sandbox import ModalSandbox, ModalSandboxBackend
    from .ordinal import Ordinal
    from .planning import Planning
    from .posthog import PostHog
    from .prompt_injection_defender import PromptInjectionDefender
    from .pydantic_ai_docs import PydanticAIDocs
    from .pylon import Pylon
    from .repo_context import RepoContext
    from .researcher import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher
    from .shell import LLM_API_KEY_ENV_PATTERNS, Shell
    from .skills import Skills
    from .spend import SpendLimits
    from .sprites_sandbox import SpritesSandbox, SpritesSandboxBackend
    from .ssh_workspace import SSHWorkspace, SSHWorkspaceBackend
    from .stackone import StackOne
    from .step_persistence import StepPersistence
    from .subagents import SubAgent, SubAgents
    from .system_reminders import SystemReminders
    from .tool_call_judge import ToolCallJudge
    from .tool_output_limits import ToolOutputLimits
    from .trajectory_judge import TrajectoryJudge
    from .warn_on_cache_busts import WarnOnCacheBusts
    from .youdotcom import YouResearch, YouSearch

__all__ = [
    'AbsurdDurability',
    'Advisor',
    'AskUser',
    'BackgroundTools',
    'BrowserUse',
    'BubblewrapSandbox',
    'BubblewrapWorkspace',
    'CapabilityCreation',
    'ClampOversizedMessages',
    'ClearToolResults',
    'CodeMode',
    'Coder',
    'ConversationSearch',
    'DEFAULT_RESEARCHER_INSTRUCTIONS',
    'DayAI',
    'DeduplicateFileReads',
    'DynamicWorkflow',
    'E2BSandbox',
    'E2BSandboxBackend',
    'ExaAgent',
    'ExaSearch',
    'FallbackCompaction',
    'FileSystem',
    'Grain',
    'GuardrailError',
    'GuardrailResult',
    'HarnessDeprecationWarning',
    'InputBlocked',
    'InputGuardrail',
    'InputGuardrailFunc',
    'LLM_API_KEY_ENV_PATTERNS',
    'LocalStack',
    'MCPReadOnlyNoToolsWarning',
    'Macroscope',
    'ManagedPrompt',
    'Memory',
    'ModalSandbox',
    'ModalSandboxBackend',
    'Ordinal',
    'OutputBlocked',
    'OutputGuardrail',
    'OutputGuardrailFunc',
    'Planning',
    'PostHog',
    'PromptInjectionDefender',
    'PydanticAIDocs',
    'Pylon',
    'READ_ONLY_TOOL_NAMES',
    'ReportContextUsage',
    'RepoContext',
    'Researcher',
    'SSHWorkspace',
    'SSHWorkspaceBackend',
    'Shell',
    'Skills',
    'SlidingWindowCompaction',
    'SpendLimits',
    'SpritesSandbox',
    'SpritesSandboxBackend',
    'StackOne',
    'StepPersistence',
    'SubAgent',
    'SubAgents',
    'SummarizingCompaction',
    'SystemReminders',
    'TieredCompaction',
    'ToolCallJudge',
    'ToolGuardrail',
    'ToolOutputLimits',
    'TrajectoryJudge',
    'WarnNearLimits',
    'WarnOnCacheBusts',
    'YouResearch',
    'YouSearch',
]

_CAPABILITY_EXPORTS = {
    'AbsurdDurability': 'absurd',
    'Advisor': 'advisor',
    'AskUser': 'ask_user',
    'BackgroundTools': 'background_tools',
    'BrowserUse': 'browser_use',
    'BubblewrapSandbox': 'bubblewrap_sandbox',
    'CapabilityCreation': 'capability_creation',
    'ClampOversizedMessages': 'compaction',
    'ClearToolResults': 'compaction',
    'CodeMode': 'code_mode',
    'Coder': 'coder',
    'ConversationSearch': 'conversation_search',
    'DayAI': 'day_ai',
    'DeduplicateFileReads': 'compaction',
    'DynamicWorkflow': 'dynamic_workflow',
    'E2BSandbox': 'e2b_sandbox',
    'ExaAgent': 'exa',
    'ExaSearch': 'exa',
    'FallbackCompaction': 'compaction',
    'FileSystem': 'filesystem',
    'Grain': 'grain',
    'LocalStack': 'localstack',
    'Macroscope': 'macroscope',
    'ManagedPrompt': 'logfire',
    'Memory': 'memory',
    'ModalSandbox': 'modal_sandbox',
    'Ordinal': 'ordinal',
    'Planning': 'planning',
    'PostHog': 'posthog',
    'PromptInjectionDefender': 'prompt_injection_defender',
    'PydanticAIDocs': 'pydantic_ai_docs',
    'Pylon': 'pylon',
    'ReportContextUsage': 'compaction',
    'RepoContext': 'repo_context',
    'Researcher': 'researcher',
    'Shell': 'shell',
    'Skills': 'skills',
    'SlidingWindowCompaction': 'compaction',
    'SpendLimits': 'spend',
    'SpritesSandbox': 'sprites_sandbox',
    'SSHWorkspace': 'ssh_workspace',
    'StackOne': 'stackone',
    'StepPersistence': 'step_persistence',
    'SubAgents': 'subagents',
    'SummarizingCompaction': 'compaction',
    'SystemReminders': 'system_reminders',
    'TieredCompaction': 'compaction',
    'ToolCallJudge': 'tool_call_judge',
    'ToolGuardrail': 'guardrails',
    'ToolOutputLimits': 'tool_output_limits',
    'TrajectoryJudge': 'trajectory_judge',
    'WarnNearLimits': 'compaction',
    'WarnOnCacheBusts': 'warn_on_cache_busts',
    'YouResearch': 'youdotcom',
    'YouSearch': 'youdotcom',
}

_CONSTANT_EXPORTS = {
    'BubblewrapWorkspace': 'bubblewrap_sandbox',
    'DEFAULT_RESEARCHER_INSTRUCTIONS': 'researcher',
    'E2BSandboxBackend': 'e2b_sandbox',
    'LLM_API_KEY_ENV_PATTERNS': 'shell',
    'ModalSandboxBackend': 'modal_sandbox',
    'READ_ONLY_TOOL_NAMES': 'filesystem',
    'SpritesSandboxBackend': 'sprites_sandbox',
    'SSHWorkspaceBackend': 'ssh_workspace',
    'SubAgent': 'subagents',
}

_GUARDRAIL_EXPORTS = {
    'GuardrailError',
    'GuardrailResult',
    'InputBlocked',
    'InputGuardrail',
    'InputGuardrailFunc',
    'OutputBlocked',
    'OutputGuardrail',
    'OutputGuardrailFunc',
    # Pre-rename names; `pydantic_ai_harness.guardrails.__getattr__` emits the
    # deprecation warning when these resolve.
    'GuardResult',
    'InputGuard',
    'InputGuardFunc',
    'OutputGuard',
    'OutputGuardFunc',
}


def __getattr__(name: str) -> object:
    module_name = _CAPABILITY_EXPORTS.get(name) or _CONSTANT_EXPORTS.get(name)
    if module_name is not None:
        from importlib import import_module

        module = import_module(f'.{module_name}', __name__)
        return getattr(module, name)
    if name in _GUARDRAIL_EXPORTS:
        from . import guardrails

        return getattr(guardrails, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
