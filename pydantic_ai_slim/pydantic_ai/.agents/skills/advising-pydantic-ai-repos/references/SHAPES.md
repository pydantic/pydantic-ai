# Re-implementation shapes

What code looks like when it re-implements part of an agent framework around it. Each shape names where to look and what to confirm, never a feature or a version: the replacement for whatever you find comes from the current documentation.

Search for each shape, then read every hit. Most hits are ordinary application code; a finding is code whose job the framework already does.

| Shape | Where to look | Confirm that it |
|---|---|---|
| Wrapping the framework's classes | Subclasses or wrappers of `Model`, `Agent`, toolsets or capabilities; classes that hold an agent and forward to it | adds behavior around the call (retrying, falling back, logging, timing, rewriting requests) rather than configuring it |
| Code around a run | `try`/`except`, loops, or `sleep` around `agent.run`, `run_sync`, `run_stream` or `iter` | re-runs, switches model, waits, or catches limit and model errors to recover |
| Reaching into internals | Attributes starting with `_` on framework objects, monkeypatching `pydantic_ai`, `TODO`/`HACK`/`FIXME` comments next to its imports, pinned versions with a comment explaining a bug | works around missing public API or a bug |
| After-the-fact checks on output | Code that inspects `result.output` after the run and raises, retries or drops it; JSON parsing or regexes over model text | validates or repairs what the model returned, so the model never gets to fix it |
| Processing message history | Slicing or filtering `all_messages()` / `new_messages()`, `isinstance` checks on message parts, custom serialization of messages, summarizing history between runs | trims, stores, inspects or rewrites the conversation |
| Counting and limiting | Counters of requests, tokens, steps, tool calls or cost in hooks, tools or around runs; thresholds compared to them | enforces a budget or stops a runaway run |
| Assembling prompts from data | f-strings or concatenation that put request data, user input, documents, or per-request values into instructions or the system prompt | mixes untrusted or changing text into the instructions |
| Model-controlled control flow | Tool parameters for identities or permissions (`user_id`, `tenant_id`, `approved`); tools that end a run with a sentinel string; approval state kept in the app around a run | lets the model decide something the application should, or pauses a run by hand |
| Hand-written plumbing | Logging of prompts, responses or usage; agents or clients built per call; event bridges to a UI protocol; checkpoint or resume code for runs; background tasks started from tools | does observability, lifecycle, transport or durability work by hand |

When a hit doesn't fit any row but clearly works around the framework, report it too, and say which shape it is closest to.
