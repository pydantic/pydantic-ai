# pydantic_ai_slim/pydantic_ai/models/openai/ Guidelines

These rules add to [`models/AGENTS.md`](../AGENTS.md) and apply to the OpenAI package only.

- Keep each model class and each streamed response in its own submodule: `OpenAIChatModel` in `_chat_model.py`, `OpenAIStreamedResponse` in `_chat_streaming.py`, `OpenAIResponsesModel` in `_responses_model.py`, `OpenAIResponsesStreamedResponse` in `_responses_streaming.py`. Put a module-level helper in the `_*.py` submodule whose docstring names its concern. Do not split a class's methods into mixins: a mixin spreads one object's state across files.
- Never import from `__init__.py` at runtime in a submodule. `__init__.py` imports the submodules, so that import is circular.
- Keep the runtime imports between submodules acyclic. When a submodule needs a name only in annotations and a runtime import would close a cycle, import the name under `TYPE_CHECKING`, as `_chat_model.py` does for `OpenAIResponsesModel`.
- Import each submodule that defines a public API name inside the `try` block of `__init__.py` that imports the `openai` SDK. A missing SDK then raises the install hint, not a bare `ImportError`.
- Import `_batch.py` and `_responses_websocket.py` at runtime only inside the function that uses them, never at a module's top level. `_responses_websocket.py` needs the `openai-realtime` extra, and an error importing `_batch.py` must not read as a missing `openai` SDK.
- In a test, patch a name in the module that calls it: `pydantic_ai.models.openai._chat_model.download_item` for a call in `_chat_model.py`. Each `from ... import name` creates its own binding, so a patch on `pydantic_ai.models.openai` reaches no call.
