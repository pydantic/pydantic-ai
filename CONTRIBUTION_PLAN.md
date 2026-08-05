# Issue 7169 plan

Issue: https://github.com/pydantic/pydantic-ai/issues/7169

Goal: validate and document the combination of `OpenAIResponsesModel`, `DeepSeekProvider`, and the public `deepseek-v4-flash` model identifier for DeepSeek V4 Flash 0731.

Proposed scope after maintainer alignment:

1. Confirm whether the existing OpenAI Responses implementation already handles DeepSeek's streamed and non-streamed response shapes.
2. Add provider/model compatibility tests for ordinary output, reasoning content, tool calls, structured output, and response continuation.
3. Add a short documentation example only if the generic implementation is already compatible.
4. Avoid provider-specific production code unless a reproducible DeepSeek incompatibility requires it.

Current status: plan only. Pydantic AI's contribution guide requires maintainer alignment and assignment before implementing a non-trivial provider feature.
