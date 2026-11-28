"""Capabilities for Pydantic AI agents."""

from typing import TYPE_CHECKING

from ._warn import HarnessDeprecationWarning, MCPReadOnlyNoToolsWarning

if TYPE_CHECKING:
    from .absurd import AbsurdDurability
    from .advisor import Advisor
    from .announce_working_directory import AnnounceWorkingDirectory
    from .approval_reviewer import ApprovalReviewer
    from .artifact_scanning import ArtifactScanner
    from .ask_user import AskUser
    from .azure_durable_task import AzureDurableTaskDurability
    from .background_tools import BackgroundTools
    from .boxd_sandbox import BoxdSandbox, BoxdSandboxBackend
    from .boxlite_sandbox import BoxLiteSandbox, BoxLiteSandboxBackend
    from .browser_use import BrowserUse
    from .bubblewrap_sandbox import BubblewrappedWorkspace, BubblewrapWorkspace
    from .capability_creation import CapabilityCreation
    from .cdp_browser import CDPBrowser
    from .code_mode import CodeMode
    from .coder import BenchmarkInstructions, Coder
    from .compaction import (
        CannotSummarizeError,
        ClampOversizedMessages,
        ClearToolResults,
        CompactOnRequest,
        DeduplicateFileReads,
        FallbackCompaction,
        PruneToolResults,
        RelevanceCompaction,
        ReportContextUsage,
        SlidingWindowCompaction,
        SummarizingCompaction,
        TieredCompaction,
        WarnNearLimits,
    )
    from .computer_use import ComputerUse
    from .conversation_search import ConversationSearch
    from .current_time import CurrentTime
    from .day_ai import DayAI
    from .daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend
    from .dbos import DBOSDurability
    from .deny_unanswered import DenyUnanswered
    from .detect_loops import DetectLoops
    from .docker_container import DockerContainer, DockerContainerBackend

    # Deprecated names: the redundant aliases type them for users; at runtime they resolve lazily, with a warning.
    from .docker_sandbox import DockerSandbox as DockerSandbox, DockerSandboxBackend as DockerSandboxBackend  # pyright: ignore[reportDeprecated]
    from .dynamic_workflow import DynamicWorkflow
    from .e2b_sandbox import E2BSandbox, E2BSandboxBackend
    from .exa import ExaAgent, ExaSearch
    from .file_system import READ_ONLY_TOOL_NAMES, FileSystem
    from .goal import Goal, GoalVerifier
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
        StreamGuardrail,
        StreamGuardrailFunc,
        ToolGuardrail,
    )
    from .ipython_code_mode import IPythonCodeMode
    from .keenable import KeenableSearch
    from .kubernetes_sandbox import KubernetesSandbox, KubernetesSandboxBackend
    from .label_cross_model_history import DEFAULT_CROSS_MODEL_NOTICE, LabelCrossModelHistory
    from .localstack import LocalStack
    from .logfire import ManagedPrompt
    from .macroscope import Macroscope
    from .memory import Memory
    from .modal_sandbox import ModalSandbox, ModalSandboxBackend
    from .model_router import ModelChoice, ModelRouter
    from .opensandbox_sandbox import OpenSandbox, OpenSandboxBackend
    from .ordinal import Ordinal
    from .peers import Peers
    from .permissions import Permissions
    from .pgtask import PGTaskDurability
    from .pixeltable import Pixeltable
    from .planning import Planning
    from .posthog import PostHog
    from .prefect import PrefectDurability
    from .prompt_injection_defender import PromptInjectionDefender
    from .pydantic_ai_docs import PydanticAIDocs
    from .pylon import Pylon
    from .refine_output import DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, RefineOutput
    from .render import RenderWorkflows
    from .render_sandbox import RenderSandbox, RenderSandboxBackend
    from .repo_context import RepoContext
    from .report_remaining_budget import ReportRemainingBudget
    from .report_stale_files import ReportStaleFiles
    from .report_tool_errors import ReportToolErrors
    from .require_verification import DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS, RequireVerification
    from .researcher import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher
    from .shell import LLM_API_KEY_ENV_PATTERNS, Shell
    from .skills import Skills
    from .smart_search import SmartSearch
    from .smol_sandbox import SmolSandbox, SmolSandboxBackend
    from .spend_limits import SpendLimits
    from .sprites_sandbox import SpritesSandbox, SpritesSandboxBackend
    from .ssh_workspace import SSHWorkspace, SSHWorkspaceBackend
    from .stackone import StackOne
    from .step_persistence import StepPersistence
    from .sub_agents import SubAgent, SubAgents
    from .system_reminders import SystemReminders
    from .temporal import TemporalDurability
    from .tool_bridge import ToolBridge
    from .tool_budgets import ToolBudgets
    from .tool_call_judge import ToolCallJudge
    from .tool_output_limits import ToolOutputLimits
    from .trajectory_judge import TrajectoryJudge
    from .warn_on_cache_busts import WarnOnCacheBusts
    from .wrap_up_at_limits import DEFAULT_WRAP_UP_PROMPT, WrapUpAtLimits, wrapped_up_by
    from .youdotcom import YouResearch, YouSearch

__all__ = [
    'AbsurdDurability',
    'Advisor',
    'AnnounceWorkingDirectory',
    'ApprovalReviewer',
    'ArtifactScanner',
    'AskUser',
    'AzureDurableTaskDurability',
    'BackgroundTools',
    'BenchmarkInstructions',
    'BoxdSandbox',
    'BoxdSandboxBackend',
    'BoxLiteSandbox',
    'BoxLiteSandboxBackend',
    'BrowserUse',
    'BubblewrappedWorkspace',
    'BubblewrapWorkspace',
    'CannotSummarizeError',
    'CapabilityCreation',
    'CDPBrowser',
    'ClampOversizedMessages',
    'ClearToolResults',
    'CodeMode',
    'Coder',
    'CompactOnRequest',
    'ComputerUse',
    'ConversationSearch',
    'CurrentTime',
    'DEFAULT_CROSS_MODEL_NOTICE',
    'DEFAULT_REFINE_OUTPUT_INSTRUCTIONS',
    'DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS',
    'DEFAULT_RESEARCHER_INSTRUCTIONS',
    'DEFAULT_WRAP_UP_PROMPT',
    'DayAI',
    'DaytonaSandbox',
    'DaytonaSandboxBackend',
    'DBOSDurability',
    'DeduplicateFileReads',
    'DenyUnanswered',
    'DetectLoops',
    'DockerContainer',
    'DockerContainerBackend',
    'DynamicWorkflow',
    'E2BSandbox',
    'E2BSandboxBackend',
    'ExaAgent',
    'ExaSearch',
    'FallbackCompaction',
    'FileSystem',
    'Goal',
    'GoalVerifier',
    'Grain',
    'GuardrailError',
    'GuardrailResult',
    'HarnessDeprecationWarning',
    'IPythonCodeMode',
    'InputBlocked',
    'InputGuardrail',
    'InputGuardrailFunc',
    'KeenableSearch',
    'KubernetesSandbox',
    'KubernetesSandboxBackend',
    'LabelCrossModelHistory',
    'LLM_API_KEY_ENV_PATTERNS',
    'LocalStack',
    'MCPReadOnlyNoToolsWarning',
    'Macroscope',
    'ManagedPrompt',
    'Memory',
    'ModalSandbox',
    'ModalSandboxBackend',
    'ModelChoice',
    'ModelRouter',
    'OpenSandbox',
    'OpenSandboxBackend',
    'Ordinal',
    'OutputBlocked',
    'OutputGuardrail',
    'OutputGuardrailFunc',
    'PGTaskDurability',
    'Peers',
    'Permissions',
    'Pixeltable',
    'Planning',
    'PostHog',
    'PrefectDurability',
    'PromptInjectionDefender',
    'PruneToolResults',
    'PydanticAIDocs',
    'Pylon',
    'READ_ONLY_TOOL_NAMES',
    'RefineOutput',
    'RelevanceCompaction',
    'RenderSandbox',
    'RenderSandboxBackend',
    'RenderWorkflows',
    'ReportContextUsage',
    'ReportRemainingBudget',
    'ReportStaleFiles',
    'ReportToolErrors',
    'RepoContext',
    'RequireVerification',
    'Researcher',
    'SSHWorkspace',
    'SSHWorkspaceBackend',
    'Shell',
    'Skills',
    'SlidingWindowCompaction',
    'SmartSearch',
    'SmolSandbox',
    'SmolSandboxBackend',
    'SpendLimits',
    'SpritesSandbox',
    'SpritesSandboxBackend',
    'StackOne',
    'StepPersistence',
    'StreamGuardrail',
    'StreamGuardrailFunc',
    'SubAgent',
    'SubAgents',
    'SummarizingCompaction',
    'SystemReminders',
    'TemporalDurability',
    'TieredCompaction',
    'ToolBridge',
    'ToolBudgets',
    'ToolCallJudge',
    'ToolGuardrail',
    'ToolOutputLimits',
    'TrajectoryJudge',
    'WarnNearLimits',
    'WarnOnCacheBusts',
    'WrapUpAtLimits',
    'wrapped_up_by',
    'YouResearch',
    'YouSearch',
]

_LAZY_EXPORTS = {
    'AbsurdDurability': 'absurd',
    'Advisor': 'advisor',
    'AnnounceWorkingDirectory': 'announce_working_directory',
    'ApprovalReviewer': 'approval_reviewer',
    'ArtifactScanner': 'artifact_scanning',
    'AskUser': 'ask_user',
    'AzureDurableTaskDurability': 'azure_durable_task',
    'BackgroundTools': 'background_tools',
    'BenchmarkInstructions': 'coder',
    'BoxdSandbox': 'boxd_sandbox',
    'BoxdSandboxBackend': 'boxd_sandbox',
    'BoxLiteSandbox': 'boxlite_sandbox',
    'BoxLiteSandboxBackend': 'boxlite_sandbox',
    'BrowserUse': 'browser_use',
    'BubblewrappedWorkspace': 'bubblewrap_sandbox',
    'BubblewrapWorkspace': 'bubblewrap_sandbox',
    'CannotSummarizeError': 'compaction',
    'CapabilityCreation': 'capability_creation',
    'CDPBrowser': 'cdp_browser',
    'ClampOversizedMessages': 'compaction',
    'ClearToolResults': 'compaction',
    'CodeMode': 'code_mode',
    'Coder': 'coder',
    'CompactOnRequest': 'compaction',
    'ComputerUse': 'computer_use',
    'ConversationSearch': 'conversation_search',
    'CurrentTime': 'current_time',
    'DayAI': 'day_ai',
    'DaytonaSandbox': 'daytona_sandbox',
    'DaytonaSandboxBackend': 'daytona_sandbox',
    'DBOSDurability': 'dbos',
    'DeduplicateFileReads': 'compaction',
    'DEFAULT_CROSS_MODEL_NOTICE': 'label_cross_model_history',
    'DEFAULT_REFINE_OUTPUT_INSTRUCTIONS': 'refine_output',
    'DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS': 'require_verification',
    'DEFAULT_RESEARCHER_INSTRUCTIONS': 'researcher',
    'DEFAULT_WRAP_UP_PROMPT': 'wrap_up_at_limits',
    'DenyUnanswered': 'deny_unanswered',
    'DetectLoops': 'detect_loops',
    'DockerContainer': 'docker_container',
    'DockerContainerBackend': 'docker_container',
    'DynamicWorkflow': 'dynamic_workflow',
    'E2BSandbox': 'e2b_sandbox',
    'E2BSandboxBackend': 'e2b_sandbox',
    'ExaAgent': 'exa',
    'ExaSearch': 'exa',
    'FallbackCompaction': 'compaction',
    'FileSystem': 'file_system',
    'Goal': 'goal',
    'GoalVerifier': 'goal',
    'Grain': 'grain',
    'GuardrailError': 'guardrails',
    'GuardrailResult': 'guardrails',
    'InputBlocked': 'guardrails',
    'InputGuardrail': 'guardrails',
    'InputGuardrailFunc': 'guardrails',
    'IPythonCodeMode': 'ipython_code_mode',
    'KeenableSearch': 'keenable',
    'KubernetesSandbox': 'kubernetes_sandbox',
    'KubernetesSandboxBackend': 'kubernetes_sandbox',
    'LabelCrossModelHistory': 'label_cross_model_history',
    'LLM_API_KEY_ENV_PATTERNS': 'shell',
    'LocalStack': 'localstack',
    'Macroscope': 'macroscope',
    'ManagedPrompt': 'logfire',
    'Memory': 'memory',
    'ModalSandbox': 'modal_sandbox',
    'ModalSandboxBackend': 'modal_sandbox',
    'ModelChoice': 'model_router',
    'ModelRouter': 'model_router',
    'OpenSandbox': 'opensandbox_sandbox',
    'OpenSandboxBackend': 'opensandbox_sandbox',
    'Ordinal': 'ordinal',
    'OutputBlocked': 'guardrails',
    'OutputGuardrail': 'guardrails',
    'OutputGuardrailFunc': 'guardrails',
    'PGTaskDurability': 'pgtask',
    'Peers': 'peers',
    'Permissions': 'permissions',
    'Pixeltable': 'pixeltable',
    'Planning': 'planning',
    'PostHog': 'posthog',
    'PrefectDurability': 'prefect',
    'PromptInjectionDefender': 'prompt_injection_defender',
    'PruneToolResults': 'compaction',
    'PydanticAIDocs': 'pydantic_ai_docs',
    'Pylon': 'pylon',
    'READ_ONLY_TOOL_NAMES': 'file_system',
    'RefineOutput': 'refine_output',
    'RelevanceCompaction': 'compaction',
    'RenderSandbox': 'render_sandbox',
    'RenderSandboxBackend': 'render_sandbox',
    'RenderWorkflows': 'render',
    'RepoContext': 'repo_context',
    'ReportContextUsage': 'compaction',
    'ReportRemainingBudget': 'report_remaining_budget',
    'ReportStaleFiles': 'report_stale_files',
    'ReportToolErrors': 'report_tool_errors',
    'RequireVerification': 'require_verification',
    'Researcher': 'researcher',
    'Shell': 'shell',
    'Skills': 'skills',
    'SlidingWindowCompaction': 'compaction',
    'SmartSearch': 'smart_search',
    'SmolSandbox': 'smol_sandbox',
    'SmolSandboxBackend': 'smol_sandbox',
    'SpendLimits': 'spend_limits',
    'SpritesSandbox': 'sprites_sandbox',
    'SpritesSandboxBackend': 'sprites_sandbox',
    'SSHWorkspace': 'ssh_workspace',
    'SSHWorkspaceBackend': 'ssh_workspace',
    'StackOne': 'stackone',
    'StepPersistence': 'step_persistence',
    'StreamGuardrail': 'guardrails',
    'StreamGuardrailFunc': 'guardrails',
    'SubAgent': 'sub_agents',
    'SubAgents': 'sub_agents',
    'SummarizingCompaction': 'compaction',
    'SystemReminders': 'system_reminders',
    'TemporalDurability': 'temporal',
    'TieredCompaction': 'compaction',
    'ToolBridge': 'tool_bridge',
    'ToolBudgets': 'tool_budgets',
    'ToolCallJudge': 'tool_call_judge',
    'ToolGuardrail': 'guardrails',
    'ToolOutputLimits': 'tool_output_limits',
    'TrajectoryJudge': 'trajectory_judge',
    'WarnNearLimits': 'compaction',
    'WarnOnCacheBusts': 'warn_on_cache_busts',
    'WrapUpAtLimits': 'wrap_up_at_limits',
    'wrapped_up_by': 'wrap_up_at_limits',
    'YouResearch': 'youdotcom',
    'YouSearch': 'youdotcom',
    # Pre-rename names; the target module's `__getattr__` emits the
    # deprecation warning when these resolve.
    'BubblewrapSandbox': 'bubblewrap_sandbox',
    'GuardResult': 'guardrails',
    'InputGuard': 'guardrails',
    'InputGuardFunc': 'guardrails',
    'OutputGuard': 'guardrails',
    'OutputGuardFunc': 'guardrails',
    # Pre-rename names; importing the `pydantic_ai_harness.docker_sandbox` shim emits
    # the deprecation warning.
    'DockerSandbox': 'docker_sandbox',
    'DockerSandboxBackend': 'docker_sandbox',
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        from importlib import import_module

        module = import_module(f'.{module_name}', __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
