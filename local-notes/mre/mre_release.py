# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic-ai==1.75.0",
# ]
# ///

"""MRE: OpenAIJsonSchemaTransformer mutates top-level oneOf schemas when strict=None.

Run with: uv run local-notes/mre/mre_release.py
"""

from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

schema = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {"kind": {"const": "final_answer"}, "text": {"type": "string"}},
            "required": ["kind", "text"],
        },
        {
            "type": "object",
            "properties": {"kind": {"const": "tool_call"}, "tool": {"type": "string"}, "args": {"type": "object"}},
            "required": ["kind", "tool", "args"],
        },
    ],
}

result = OpenAIJsonSchemaTransformer(schema, strict=None).walk()
print("Result keys:", list(result.keys()))
print("Has 'properties':", "properties" in result)
print("Has 'additionalProperties':", "additionalProperties" in result)

if "properties" in result or "additionalProperties" in result:
    print("BUG REPRODUCED: outer object was mutated with properties/additionalProperties")
else:
    print("OK: outer object was left intact")
