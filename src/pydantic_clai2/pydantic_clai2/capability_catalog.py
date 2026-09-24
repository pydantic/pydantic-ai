"""Opt-in harness capabilities, declared without importing their optional dependencies."""

from .config import PluginSettings

# Coder, AskUser and RepoContext already have shell-integrated built-in entries.
_FACTORIES = (
    ('advisor', 'advisor:Advisor'),
    ('aws_lambda', 'aws_lambda:AWSLambdaDurability'),
    ('background_tools', 'background_tools:BackgroundTools'),
    ('browser_use', 'browser_use:BrowserUse'),
    ('capability_creation', 'capability_creation:CapabilityCreation'),
    ('code_mode', 'code_mode:CodeMode'),
    ('clamp_oversized_messages', 'compaction:ClampOversizedMessages'),
    ('clear_tool_results', 'compaction:ClearToolResults'),
    ('deduplicate_file_reads', 'compaction:DeduplicateFileReads'),
    ('fallback_compaction', 'compaction:FallbackCompaction'),
    ('report_context_usage', 'compaction:ReportContextUsage'),
    ('sliding_window_compaction', 'compaction:SlidingWindowCompaction'),
    ('summarizing_compaction', 'compaction:SummarizingCompaction'),
    ('tiered_compaction', 'compaction:TieredCompaction'),
    ('warn_near_limits', 'compaction:WarnNearLimits'),
    ('conversation_search', 'conversation_search:ConversationSearch'),
    ('dynamic_workflow', 'dynamic_workflow:DynamicWorkflow'),
    ('exa_agent', 'exa:ExaAgent'),
    ('exa_search', 'exa:ExaSearch'),
    ('filesystem', 'filesystem:FileSystem'),
    ('github', 'github:GitHub'),
    ('google_workspace', 'google_workspace:GoogleWorkspace'),
    ('input_guardrail', 'guardrails:InputGuardrail'),
    ('output_guardrail', 'guardrails:OutputGuardrail'),
    ('tool_guardrail', 'guardrails:ToolGuardrail'),
    ('linear', 'linear:Linear'),
    ('localstack', 'localstack:LocalStack'),
    ('managed_prompt', 'logfire:ManagedPrompt'),
    ('logfire_mcp', 'logfire_mcp:LogfireMCP'),
    ('macroscope', 'macroscope:Macroscope'),
    ('memory', 'memory:Memory'),
    ('modal_sandbox', 'modal_sandbox:ModalSandbox'),
    ('notion', 'notion:Notion'),
    ('planning', 'planning:Planning'),
    ('playwright', 'playwright:PlaywrightBrowser'),
    ('prompt_injection_defender', 'prompt_injection_defender:PromptInjectionDefender'),
    ('pydantic_ai_docs', 'pydantic_ai_docs:PydanticAIDocs'),
    ('repair_tool_arguments', 'repair_tool_arguments:RepairToolArguments'),
    ('researcher', 'researcher:Researcher'),
    ('shell', 'shell:Shell'),
    ('skills', 'skills:Skills'),
    ('slack', 'slack:Slack'),
    ('spend_limits', 'spend:SpendLimits'),
    ('stackone', 'stackone:StackOne'),
    ('step_persistence', 'step_persistence:StepPersistence'),
    ('subagents', 'subagents:SubAgents'),
    ('system_reminders', 'system_reminders:SystemReminders'),
    ('tool_output_limits', 'tool_output_limits:ToolOutputLimits'),
    ('trajectory_judge', 'trajectory_judge:TrajectoryJudge'),
    ('warn_on_cache_busts', 'warn_on_cache_busts:WarnOnCacheBusts'),
    ('you_search', 'youdotcom:YouSearch'),
    ('you_research', 'youdotcom:YouResearch'),
)

HARNESS_PLUGINS: tuple[PluginSettings, ...] = tuple(
    PluginSettings(id=name, factory=f'pydantic_ai_harness.{factory}', enabled=False) for name, factory in _FACTORIES
)
"""Disabled built-ins. The loader imports a capability only when the user enables it."""
