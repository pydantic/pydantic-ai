# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic-ai==1.95.0",
# ]
# ///
"""
MRE for pydantic-ai issue #5413.

Demonstrates that `input_fidelity` is passed as `None` to the OpenAI API
even when not explicitly set, causing `gpt-image-2` to error.

Run: uv run local-notes/mre/mre_release.py
"""

from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.models.openai import _map_openai_image_generation_tool

tool = ImageGenerationTool(model='gpt-image-2')
mapped = _map_openai_image_generation_tool(tool)

print('Mapped tool payload:')
print(mapped)

# On the release, input_fidelity is present as None
if 'input_fidelity' in mapped:
    print('\nBUG REPRODUCED: input_fidelity is present in payload')
    print(f"Value: {mapped['input_fidelity']!r}")
else:
    print('\nBUG NOT REPRODUCED: input_fidelity is absent from payload')
