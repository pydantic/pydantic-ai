"""Typed passthrough for the browser-use `Agent` constructor options.

`BrowserAgentSettings` mirrors the supported options of `browser_use.Agent`
(v0.13.x) with browser-use's own defaults. Four groups are deliberately absent:
options the capability already owns as first-class fields
(task, llm, session, vision, output schema, sensitive data, system-message
extension, signal handling), options that are objects the caller has to build in
code (browser-use's callbacks, `injected_agent_state`, `skill_service` -- these
belong in a `BrowserAgentFactory`), and options the toolset controls or that are
not user-facing (`browser`, `browser_profile`, the deprecated `controller` alias
of `tools`, per-run identity like `task_id` and `source`, and private ones), and
`available_file_paths`, which needs an application-specific upload approval or
destination policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

try:
    from browser_use import Tools
    from browser_use.agent.views import MessageCompactionSettings
    from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'browser-use is required for BrowserUse. Install it with: pip install "pydantic-ai-harness[browser-use]"'
    ) from _import_error

if TYPE_CHECKING:
    from pydantic_ai_harness.browser_use._model import ChatModelInput


@dataclass
class BrowserAgentSettings:
    """Supported `browser_use.Agent` options, with browser-use's defaults.

    Pass an instance as `BrowserUse.agent_settings`. The defaults are a
    snapshot of browser-use's own (the pinned minimum version), so an empty
    instance behaves exactly like not passing one; a test asserts that mirror
    field by field, so an upgrade that moves a default is caught rather than
    silently changing behaviour. The `*_llm` fields accept the same inputs as
    the capability's `llm` field: a browser-use chat model, a Pydantic AI model,
    or a model name string.
    """

    tools: Tools[None] | None = None
    """Custom action registry (browser-use `Tools`): register your own actions, exclude built-ins."""

    override_system_message: str | None = None
    """Replace the browser agent's system prompt entirely (`BrowserUse.extend_system_message` appends instead)."""

    max_failures: int = 5
    """Consecutive step failures before the agent gives up."""

    max_actions_per_step: int = 5
    """How many actions the model may emit per step."""

    use_thinking: bool = True
    """Include a thinking field in the agent's output schema."""

    flash_mode: bool = False
    """Minimal output schema (skips evaluation/memory/goal fields) for speed."""

    max_history_items: int | None = None
    """Cap on agent-history items kept in the model's context; `None` keeps all."""

    page_extraction_llm: ChatModelInput | None = None
    """Separate model for page-content extraction; `None` uses the main model."""

    fallback_llm: ChatModelInput | None = None
    """Model to fall back to when the main model errors."""

    use_judge: bool = True
    """Run a judge model call over the finished task (one extra LLM call per task)."""

    judge_llm: ChatModelInput | None = None
    """Separate model for the judge; `None` uses the main model."""

    ground_truth: str | None = None
    """Reference answer for the judge to evaluate the result against."""

    calculate_cost: bool = False
    """Track token costs via browser-use's pricing data."""

    vision_detail_level: Literal['auto', 'low', 'high'] = 'auto'
    """Screenshot detail level sent to the model."""

    llm_screenshot_size: tuple[int, int] | None = None
    """Resize screenshots to (width, height) before sending them to the model."""

    llm_timeout: int | None = None
    """Seconds to wait for a single model call; `None` uses browser-use's per-model default."""

    step_timeout: int = 180
    """Seconds to wait for a single agent step."""

    directly_open_url: bool = True
    """Open a URL found in the task as the first action, before the first model call."""

    include_recent_events: bool = False
    """Include recent browser events in the model's context."""

    final_response_after_failure: bool = True
    """Ask the model for a final summary even when the task failed or ran out of steps."""

    enable_planning: bool = True
    """Run browser-use's planning loop alongside the action loop."""

    planning_replan_on_stall: int = 3
    """Steps without progress before the planner replans."""

    planning_exploration_limit: int = 5
    """Cap on exploratory planning steps."""

    loop_detection_enabled: bool = True
    """Detect and break repeated-action loops."""

    loop_detection_window: int = 20
    """How many recent steps the loop detector inspects."""

    message_compaction: MessageCompactionSettings | bool | None = True
    """Compact older messages in the sub-agent's context; pass settings for fine control."""

    max_clickable_elements_length: int = 40000
    """Character cap for the serialized clickable-elements listing."""

    include_tool_call_examples: bool = False
    """Include tool-call examples in the system prompt."""

    initial_actions: list[dict[str, dict[str, object]]] | None = None
    """Actions to run before the first model call, e.g. `[{'navigate': {'url': ...}}]`."""

    file_system_path: str | None = None
    """Directory backing the sub-agent's own file system; `None` uses a temporary one per run."""

    display_files_in_done_text: bool = True
    """Include the contents of files the agent wrote in its final message."""

    save_conversation_path: str | Path | None = None
    """Write the full sub-agent conversation to this path for debugging."""

    save_conversation_path_encoding: str | None = 'utf-8'
    """Encoding for the saved conversation file."""

    include_attributes: list[str] | None = None
    """DOM attributes serialized for the model with each element; `None` uses browser-use's set."""

    extraction_schema: dict[str, object] | None = None
    """JSON schema for browser-use's page-extraction action (distinct from the task's `output_schema`)."""

    sample_images: list[ContentPartTextParam | ContentPartImageParam] | None = None
    """Reference images (with captions) prepended to the sub-agent's context, e.g. what to look for."""

    skills: list[str | Literal['*']] | None = None
    """browser-use skills to enable by name, or `'*'` for all; needs a browser-use account."""

    skill_ids: list[str | Literal['*']] | None = None
    """browser-use skills to enable by id; the id-addressed counterpart of `skills`."""

    pricing_url: str | None = None
    """Override the pricing data source used when `calculate_cost` is on."""

    generate_gif: bool | str = False
    """Record the run as a GIF (`True` for a default path, or a target path)."""

    demo_mode: bool | None = None
    """Slow the browser down and highlight interactions, for demos; `None` uses browser-use's default."""
