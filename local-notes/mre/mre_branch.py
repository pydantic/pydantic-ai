# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic-ai @ file:///home/pydanty/pydantic-ai",
# ]
# ///
"""
MRE for pydantic-ai issue #5413 fix verification.

Demonstrates that `input_fidelity` is omitted from the OpenAI API payload
when not explicitly set, preventing `gpt-image-2` errors.

Run: uv run local-notes/mre/mre_branch.py
"""

from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.models.openai import _map_openai_image_generation_tool

tool = ImageGenerationTool(model='gpt-image-2')
mapped = _map_openai_image_generation_tool(tool)

print('Mapped tool payload:')
print(mapped)

# On the fixed branch, input_fidelity is absent when not set
if 'input_fidelity' not in mapped:
    print('\nFIX VERIFIED: input_fidelity is absent from payload when not set')
else:
    print('\nFIX NOT VERIFIED: input_fidelity is still present in payload')
    print(f"Value: {mapped['input_fidelity']!r}")

# When explicitly set, it should still be present
tool_with_fidelity = ImageGenerationTool(model='gpt-image-2', input_fidelity='high')
mapped_with_fidelity = _map_openai_image_generation_tool(tool_with_fidelity)
print('\nWith input_fidelity="high":')
print(mapped_with_fidelity)
if mapped_with_fidelity.get('input_fidelity') == 'high':
    print('Explicit input_fidelity correctly preserved')
else:
    print('ERROR: Explicit input_fidelity lost')
