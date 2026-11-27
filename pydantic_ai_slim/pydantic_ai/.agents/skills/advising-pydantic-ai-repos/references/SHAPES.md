# Re-implementation shapes

What code looks like when it re-implements part of an agent framework around it. Each shape names where to look and what to confirm, never a feature or a version: the replacement for whatever you find comes from the current documentation.

Search for each shape, then read every hit. Most hits are ordinary application code; a finding is code whose job the framework already does.

| Shape | Where to look | Confirm that it |
|---|---|---|
| Wrapping the framework's classes | Subclasses or wrappers of `Model`, `Agent`, toolsets or capabilities; classes that hold an agent and forward to it | adds behavior around the call (retrying, falling back, logging, timing, rewriting requests) rather than configuring it |
| Code around a run | `try`/`except`, loops, `sleep` or `asyncio.timeout`/`wait_for` around `agent.run`, `run_sync`, `run_stream` or `iter`; `Agent.override(model=...)` outside tests | re-runs, switches or routes models, waits, times out or cancels, or catches limit and model errors to recover |
| Custom models and adapters | Subclasses of `Model` or `StreamedResponse`, and code that converts Pydantic AI messages to another API | matches every message part kind the current version sends (an exhaustive `match` or `assert_never` breaks when one is added), forwards only the settings that API accepts, and records usage even when a stream is cancelled |
| Errors returned as tool results | Tools that catch exceptions and return strings like `"Error: ..."`, or a dict with an error field | tells the model a tool failed in a form the framework can't see, retry, or count |
| Reaching into internals | Attributes starting with `_` on framework objects, monkeypatching `pydantic_ai`, `TODO`/`HACK`/`FIXME` comments next to its imports, pinned versions with a comment explaining a bug | works around missing public API or a bug; for the upgrade, check each private member it uses at the target version (see [Upgrade notes](UPGRADE.md#private-members)) |
| After-the-fact checks on output | Code that inspects `result.output` after the run and raises, retries or drops it; JSON parsing or regexes over model text | validates or repairs what the model returned, so the model never gets to fix it |
| Processing message history | Slicing or filtering `all_messages()` / `new_messages()`, `isinstance` checks on message parts, custom serialization of messages, summarizing history between runs | trims, stores, inspects or rewrites the conversation |
| Counting and limiting | Counters of requests, tokens, steps, tool calls or cost in hooks, tools or around runs; thresholds compared to them; semaphores or locks around model calls or runs | enforces a budget, stops a runaway run, or limits concurrency |
| Assembling prompts from data | f-strings or concatenation that put request data, user input, documents, or per-request values into instructions or the system prompt | mixes untrusted or changing text into the instructions. Text from other users, documents, tool results or the web in the instructions is a security finding even when it's deliberate: never put it under "Considered and kept" |
| Model-controlled control flow | Tool parameters for identities or permissions (`user_id`, `tenant_id`, `approved`); tools that end a run with a sentinel string; approval state kept in the app around a run | lets the model decide something the application should, or pauses a run by hand |
| Settings that don't reach the framework | Application settings or environment variables named like a framework option (retries, temperature, model, limits) | is never passed to the agent, model or run, so the framework's default applies instead |
| Hand-written plumbing | Logging of prompts, responses or usage; agents or clients built per call; event bridges to a UI protocol; checkpoint or resume code for runs; background tasks started from tools; loops that submit and poll provider batch jobs | does observability, lifecycle, transport or durability work by hand |

When a hit doesn't fit any row but clearly works around the framework, report it too, and say which shape it is closest to.
