"""Agent banner display — logo, agent info, and next-steps guide.

Prints a branded banner to stderr the first time an agent runs. Subsequent calls
for the same agent instance are no-ops. Each unique agent instance gets its own
banner.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from . import __version__

if TYPE_CHECKING:
    from .agent import Agent as _Agent

# Track which agent instances have already been displayed (process-wide).
_displayed_agent_ids: set[int] = set()

LOGO_PREAMBLE = r"""
pydantic-ai v{version} • Python {python_version}
"""

LOGO = r"""
      /\        
     /  \       
    /    \      
   /      \     
  /________\    
 /    ||    \   
/     ||     \  
\_____||_____/  
"""

LOGO_TEXT = r"""
[agent]
name:         {name}
model:        {model}
output_type:  {output_type}
tools:        {tools}
logfire:      {logfire_status}


"""

# line by line join of Logo and text for easier maintenance
LOGO_PLUS_TEXT = '\n'.join(
    f'{logo_line}{text_line}'
    for logo_line, text_line in zip(
        LOGO.splitlines(),
        LOGO_TEXT.splitlines(),
    )
)

NEXT_STEPS = """
─── Next steps ────────────────────────────────────
• Setup Logfire (Free):   https://ai.pydantic.dev/logfire/
• Configure Logfire:      logfire.configure(...)
• Instrument this agent:  logfire.instrument_pydantic_ai()
• Instrument its calls:   .instrument_httpx · .instrument_openai
• Export anywhere:        OTLP → Logfire · etc

• Docs:                   https://ai.pydantic.dev/logfire/
• Turn off this banner:   os.environ['PYDANTIC_AI_HIDE_BANNER'] = '1'

"""

BANNER = f'{LOGO_PREAMBLE}\n{LOGO_PLUS_TEXT}\n{NEXT_STEPS}'


def _get_model_name(agent: _Agent[Any, Any]) -> str:
    """Get a display-friendly model name from the agent."""
    model = agent.model
    if model is None:
        return '(not set)'
    if isinstance(model, str):
        return model
    # Model instance
    try:
        return model.model_id
    except Exception:
        return getattr(model, 'model_name', str(model))


def _get_output_type_name(agent: _Agent[Any, Any]) -> str:
    """Get a display-friendly output type name."""
    try:
        ot = agent.output_type
        return getattr(ot, '__name__', str(ot))
    except Exception:
        return 'str'


def _get_tool_names(agent: _Agent[Any, Any]) -> str:
    """Get comma-separated tool names from the agent's function toolset."""
    try:
        tools = agent._function_toolset.tools  # type: ignore[union-attr]
        if tools:
            return ', '.join(tools.keys())
        return '(none)'
    except Exception:
        return '(none)'


def _get_logfire_status(agent: _Agent[Any, Any]) -> str:
    """Check whether Logfire instrumentation is configured."""
    try:
        settings = agent._resolve_instrumentation_settings()
        return 'configured' if settings is not None else 'not configured'
    except Exception:
        return 'unknown'


def display_agent_banner(agent: _Agent[Any, Any]) -> None:
    """Display the agent banner to stderr, once per agent instance per process.

    Args:
        agent: The agent to display the banner for.
    """
    import os

    if os.environ.get('PYDANTIC_AI_HIDE_BANNER'):
        return

    agent_id = id(agent)
    if agent_id in _displayed_agent_ids:
        return
    _displayed_agent_ids.add(agent_id)

    name = agent.name or '(unnamed)'
    model = _get_model_name(agent)
    output_type = _get_output_type_name(agent)
    tools = _get_tool_names(agent)
    logfire_status = _get_logfire_status(agent)
    python_version = platform.python_version()
    timestamp = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    banner = BANNER.format(
        version=__version__,
        python_version=python_version,
        timestamp=timestamp,
        name=name,
        model=model,
        output_type=output_type,
        tools=tools,
        logfire_status=logfire_status,
    )

    print(f'\n{banner}\n', file=sys.stderr)
