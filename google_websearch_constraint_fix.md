# Fix for pydantic/pydantic-ai #9706

## Problem
`WebSearchTool` has domain filtering and constraint fields that are silently ignored by providers that can't apply them (like Google/Gemini's native `google_search`).

This violates the intent of `_requires_native()` which was designed to prevent silent constraint violations.

## Root Cause
In `google.py:832-833`, when a `WebSearchTool` is encountered, it's converted to a bare `GoogleSearchDict()` without checking if the tool has unsupported constraint fields.

```python
if isinstance(tool, WebSearchTool):
    tools.append(ToolDict(google_search=GoogleSearchDict()))  # Ignores all constraints!
```

## Solution
Add validation before converting to check if the tool has constraints that Google cannot honor.

### File: `pydantic_ai_slim/pydantic_ai/models/google.py`

Add check around line 832:

```python
from ..exceptions import UserError

# In _build_tools() method, before line 832:
if isinstance(tool, WebSearchTool):
    # Check for unsupported constraints
    unsupported = []
    if tool.allowed_domains is not None:
        unsupported.append('allowed_domains')
    if tool.blocked_domains is not None:
        unsupported.append('blocked_domains')
    if tool.max_uses is not None:
        unsupported.append('max_uses')
    if tool.external_web_access is not None:
        unsupported.append('external_web_access')
    
    if unsupported:
        raise UserError(
            f'Google native web search (google_search) does not support these WebSearchTool '
            f'constraint fields: {", ".join(unsupported)}. '
            f'These fields exist for compliance and trust reasons - silently dropping them '
            f'would violate the constraint. Use WebSearch(local=True) with a DuckDuckGo or '
            f'custom search engine that supports your required constraints, or remove these fields.'
        )
    
    # Original code (only reached if no constraints)
    tools.append(ToolDict(google_search=GoogleSearchDict()))
```

## Why This Approach

1. **Fail fast** - Prevents silent security/compliance violations
2. **Consistent** - Matches pattern used elsewhere in `google.py` (see lines 844-846)
3. **Clear guidance** - Error message tells user exactly what to do
4. **Matches `_requires_native()` intent** - That method returns True specifically to prevent silent dropping

## Alternative Considered: Warning

Could use `warnings.warn()` and continue, but:
- Warnings are often ignored or missed in logs
- Domain filters are often security/compliance requirements
- Not consistent with how other unsupported features are handled in this file

## Testing

Should add test in `tests/models/google/test_native_tools.py`:

```python
def test_websearch_unsupported_constraints():
    \"\"\"Test that WebSearchTool with unsupported constraints raises UserError.\"\"\"
    from pydantic_ai import Agent
    from pydantic_ai.native_tools import WebSearchTool
    from pydantic_ai.exceptions import UserError
    
    tool = WebSearchTool(allowed_domains=['example.com'])
    agent = Agent('gemini-2.0-flash', tools=[tool])
    
    with pytest.raises(UserError, match='does not support.*allowed_domains'):
        agent.run_sync('test')
```

## Related Issues
- #8218 - Mapping `blocked_domains` to Gemini's `exclude_domains` (SDK feature)
- #9386 - ExaSearch/You.com native domain filters
