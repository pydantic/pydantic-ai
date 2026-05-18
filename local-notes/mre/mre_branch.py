# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic-ai @ file:///home/pydanty/pydantic-ai",
# ]
# ///

"""MRE: OpenAIJsonSchemaTransformer no longer mutates top-level oneOf schemas when strict=None.

Run with: uv run local-notes/mre/mre_branch.py
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

transformer = OpenAIJsonSchemaTransformer(schema, strict=None)
result = transformer.walk()
print("Result keys:", list(result.keys()))
print("Has 'properties':", "properties" in result)
print("Has 'additionalProperties':", "additionalProperties" in result)
print("is_strict_compatible:", transformer.is_strict_compatible)

if "properties" not in result and "additionalProperties" not in result and not transformer.is_strict_compatible:
    print("FIX VERIFIED: outer object was left intact and marked as not strict-compatible")
else:
    print("FIX NOT VERIFIED")
