"""
Agent hook management system for Pydantic-AI.

This module provides the hook routing system that connects the public API
`agent.on.tool(my_tool).before(request_approval_func)` to the actual node instances
during agent execution.
"""


class HookRegistry:
    """Connects agent-level hooks to node-level traits during execution.
    
    This registry follows the registry pattern to provide a centralized way for nodes
    to query and access hooks that were set via agent.on.tool(my_tool).before.
    """
    
    def __init__(self, agent_hooks: 'AgentHooks'):
        self.agent_hooks = agent_hooks
    
    def get_tool_hook(self, tool_name: str, hook_type: str):
        """Get a specific hook for a tool.
        
        Args:
            tool_name: Name of the tool to get hooks for
            hook_type: Type of hook ('before', 'after', 'error')
            
        Returns:
            The hook function if found, None otherwise
        """
        tool_hooks = self.agent_hooks.get_tool_hooks(tool_name)
        if tool_hooks:
            return getattr(tool_hooks, hook_type, None)
        return None


class ToolHookManager:
    """Manages tool-specific hooks for individual tools."""
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.before = None
        self.after = None
        self.error = None
    
    def set_before(self, hook):
        """Set the before hook for this tool."""
        self.before = hook
    
    def set_after(self, hook):
        """Set the after hook for this tool."""
        self.after = hook
    
    def set_error(self, hook):
        """Set the error hook for this tool."""
        self.error = hook


class AgentHooks:
    """Tool-specific hooks for agents."""
    
    def __init__(self):
        # Tool-specific hooks storage
        self._tool_hooks = {}
    
    def tool(self, tool_or_name):
        """Get or create tool-specific hooks.
        
        Enables the beautiful API: agent.on.tool(my_tool).before = func
        """
        if hasattr(tool_or_name, 'name'):
            tool_name = tool_or_name.name
        else:
            tool_name = str(tool_or_name)
        
        if tool_name not in self._tool_hooks:
            self._tool_hooks[tool_name] = ToolHookManager(tool_name)
        
        return self._tool_hooks[tool_name]
    
    def get_tool_hooks(self, tool_name: str):
        """Get hooks for a specific tool. Used by HookRegistry."""
        return self._tool_hooks.get(tool_name)
