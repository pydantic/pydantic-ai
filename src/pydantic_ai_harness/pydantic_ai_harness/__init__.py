"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

from ._warn import HarnessDeprecationWarning

if TYPE_CHECKING:
    from .advisor import Advisor
    from .browser_use import BrowserUse
    from .capability_creation import CapabilityCreation
    from .code_mode import CodeMode
    from .coder import DEFAULT_ALLOWED_COMMANDS, Coder
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
    from .dynamic_workflow import DynamicWorkflow
    from .exa import ExaAgent, ExaSearch
    from .filesystem import READ_ONLY_TOOL_NAMES, FileSystem
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
    from .modal_sandbox import ModalSandbox
    from .planning import Planning
    from .prompt_injection_defender import PromptInjectionDefender
    from .pydantic_ai_docs import PydanticAIDocs
    from .repo_context import RepoContext
    from .researcher import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher
    from .shell import LLM_API_KEY_ENV_PATTERNS, Shell
    from .skills import Skills
    from .spend import SpendLimits
    from .stackone import StackOne
    from .step_persistence import StepPersistence
    from .subagents import SubAgent, SubAgents
    from .system_reminders import SystemReminders
    from .tool_output_limits import ToolOutputLimits
    from .warn_on_cache_busts import WarnOnCacheBusts
    from .youdotcom import YouResearch, YouSearch

__all__ = [
    'Advisor',
    'BrowserUse',
    'CapabilityCreation',
    'ClampOversizedMessages',
    'ClearToolResults',
    'CodeMode',
    'Coder',
    'ConversationSearch',
    'DEFAULT_ALLOWED_COMMANDS',
    'DEFAULT_RESEARCHER_INSTRUCTIONS',
    'DeduplicateFileReads',
    'DynamicWorkflow',
    'ExaAgent',
    'ExaSearch',
    'FallbackCompaction',
    'FileSystem',
    'GuardrailError',
    'GuardrailResult',
    'HarnessDeprecationWarning',
    'InputBlocked',
    'InputGuardrail',
    'InputGuardrailFunc',
    'LLM_API_KEY_ENV_PATTERNS',
    'LocalStack',
    'Macroscope',
    'ManagedPrompt',
    'Memory',
    'ModalSandbox',
    'OutputBlocked',
    'OutputGuardrail',
    'OutputGuardrailFunc',
    'Planning',
    'PromptInjectionDefender',
    'PydanticAIDocs',
    'READ_ONLY_TOOL_NAMES',
    'ReportContextUsage',
    'RepoContext',
    'Researcher',
    'Shell',
    'Skills',
    'SlidingWindowCompaction',
    'SpendLimits',
    'StackOne',
    'StepPersistence',
    'SubAgent',
    'SubAgents',
    'SummarizingCompaction',
    'SystemReminders',
    'TieredCompaction',
    'ToolGuardrail',
    'ToolOutputLimits',
    'WarnNearLimits',
    'WarnOnCacheBusts',
    'YouResearch',
    'YouSearch',
]

_CAPABILITY_EXPORTS = {
    'Advisor': 'advisor',
    'BrowserUse': 'browser_use',
    'CapabilityCreation': 'capability_creation',
    'ClampOversizedMessages': 'compaction',
    'ClearToolResults': 'compaction',
    'CodeMode': 'code_mode',
    'Coder': 'coder',
    'ConversationSearch': 'conversation_search',
    'DeduplicateFileReads': 'compaction',
    'DynamicWorkflow': 'dynamic_workflow',
    'ExaAgent': 'exa',
    'ExaSearch': 'exa',
    'FallbackCompaction': 'compaction',
    'FileSystem': 'filesystem',
    'LocalStack': 'localstack',
    'Macroscope': 'macroscope',
    'ManagedPrompt': 'logfire',
    'Memory': 'memory',
    'ModalSandbox': 'modal_sandbox',
    'Planning': 'planning',
    'PromptInjectionDefender': 'prompt_injection_defender',
    'PydanticAIDocs': 'pydantic_ai_docs',
    'ReportContextUsage': 'compaction',
    'RepoContext': 'repo_context',
    'Researcher': 'researcher',
    'Shell': 'shell',
    'Skills': 'skills',
    'SlidingWindowCompaction': 'compaction',
    'SpendLimits': 'spend',
    'StackOne': 'stackone',
    'StepPersistence': 'step_persistence',
    'SubAgents': 'subagents',
    'SummarizingCompaction': 'compaction',
    'SystemReminders': 'system_reminders',
    'TieredCompaction': 'compaction',
    'ToolGuardrail': 'guardrails',
    'ToolOutputLimits': 'tool_output_limits',
    'WarnNearLimits': 'compaction',
    'WarnOnCacheBusts': 'warn_on_cache_busts',
    'YouResearch': 'youdotcom',
    'YouSearch': 'youdotcom',
}

_CONSTANT_EXPORTS = {
    'DEFAULT_ALLOWED_COMMANDS': 'coder',
    'DEFAULT_RESEARCHER_INSTRUCTIONS': 'researcher',
    'LLM_API_KEY_ENV_PATTERNS': 'shell',
    'READ_ONLY_TOOL_NAMES': 'filesystem',
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
