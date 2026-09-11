# Pydantic AI vs Google ADK

Google ADK is the Gemini-native kit: `LlmAgent`, a `Runner`, Vertex, Search, A2A, a web UI. Pydantic
AI isn't tied to a cloud. The production gap is stop: there is no `cancel`, `stop`, or `abort` on
`Runner` or `LlmAgent`.

## Side by side

| | Google ADK | Pydantic AI |
|---|---|---|
| Models | Gemini first (`LiteLlm`, `AnthropicLlm` exist) | Any provider |
| Stop | Cancel the asyncio task | A stop signal; you get the messages back |
| Trusted state | App / user / invocation state | A typed object your tools read; the model never sees it |
| Compose | `LoopAgent`, `ParallelAgent` | `async` / `gather` |
| Deploy | Vertex | Anywhere |
| Test offline | Subclass `BaseLlm` | A fake model you script; no API key |

## FAQ

**Gemini?** Yes, including Vertex. This isn't about the model.

**Drop-in?** No. Sessions become history. `ParallelAgent` becomes `gather`.
