# Brief: Less Invasive Temporal Dynamic Toolset Changes

## Problem

The current changes to `temporal/_dynamic_toolset.py` are too invasive:
1. Line 69: Uses private methods and repeats logic from `DynamicToolset`
2. Line 106: Changed from `_call_tool_in_activity` to manual steps

## Goal

Find a less invasive approach that:
- Works with the new `for_run`/`for_run_step` lifecycle
- Reuses `_call_tool_in_activity` or similar existing patterns
- Doesn't duplicate DynamicToolset logic
- Passes toolset instances through where needed instead of reimplementing

## Also Check

- `temporal/_toolset.py` line 80: "Is this right? We don't have to pass anything through?" — verify the for_run/for_run_step pass-through is correct

## Reference

- PR #4640 comments on `durable_exec/temporal/_dynamic_toolset.py`
- PR #4688 (toolset-state) — original implementation
