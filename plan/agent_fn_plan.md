# Implementation Plan: Lightweight Functional `@Agent` API

## Core Design Principles
This document outlines the procedural implementation plan for building a functional wrapper for Pydantic AI Agents (`agent_fn_proposed.py`). The goal is to provide a declarative, syntactic sugar layer, turning typed Python function signatures into `agent.run*()` execution flows. 

The implementation centers around adding a wrapper inside `Agent.__call__`. This allows instances of `Agent` to double as decorators (`@agent_instance`).

## 1. Core API & Execution Flow (`Agent.__call__`)
The main hook is implemented by modifying `Agent.__call__` so that when `__call__` receives a callable, it delegates to an internal `_wrap_agent_function(target_agent, func)`.

Inside the wrapper, execution goes through:
### a. Signature Inference
- **Async/Sync & Streaming:** Use the `inspect` and `typing` modules (`get_origin`, `get_args`) to determine if the function is a coroutine (`async def`), a generator (`yield`), an async generator, or synchronous.
- **Output Types:** Extract the function's `return_annotation` to populate `output_type`. If wrapped in standard streaming annotations (`Iterable`, `AsyncIterable`, etc.), extract the inner base type. Also be able to handle PromptedOutput/NativeOutput for models that do not support streaming tool calls.
- **Model Output Classes:** Handle cases where the type hint is `PromptedOutput` or `NativeOutput`. The decorator strips these to provide clean types to external systems while handling the data extraction during `res.stream_output()` or standard returns.

### b. Input Prompts & XML formatting
- Argument names and values are matched via `inspect.Signature.bind()`.
- Standard python base types, dicts, and Pydantic models are passed into Pydantic AI's `format_as_xml` to construct the `user_prompt`. Solution should consider how to handle arguments that may not serialize for some reason - should the function raise a runtime exception when the decorator builds or catch the error and either note the argument as not renderable or just ignore the argument?

### c. Injecting Instructions
- The wrapped function's `__doc__` string natively becomes the prompt instructions via `.run(instructions=func.__doc__)`.

## 2. Advanced Handling 

### Multimodal Inputs (Processing Non-Text Parts)
When inspecting bound arguments prior to XML generation, check for Pydantic AI multimodal types (`ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`, `BinaryContent`, `UploadedFile`, `TextContent`) either as direct arguments or inside lists. 
- **Crucial Action:** These must bypass `format_as_xml` and instead be appended organically as `Part`s alongside the text buffer in the final `user_prompt` list.

### Graph Ecosystem (StepContext)
- In order to stack `@g.step` and `@Agent`, look for `StepContext` in type hints.
- Extract `ctx.inputs` to pass into the `user_prompt` implicitly.
- Extract `ctx.deps` from the StepContext and forward it to `target_agent.run*(deps=...)`.

## 3. Nice-to-Have / Non-Core Features: **Discuss with developer before trying to incorporate**
**Handling `RunContext`, `deps_type`, and Usage Tracking**
Capturing `RunContext` or matching the agent's `deps_type` directly in the signature is a *nice-to-have* expansion to facilitate the "Agents as tools" pattern.
- Pulling `RunContext.usage` to map token tracking
- Look out for `RunContext` or matching parameter annotations against `target_agent.deps_type`. 
- Extract `.deps` and forward it to `target_agent.run*(deps=...)`.

## 4. Edge Cases & Future Considerations
A procedural rewrite must account for the following open edge cases gracefully:
- **Debouncing Streaming Outputs**: When returning structured lists via `res.stream_output()`, yielding diffs requires cleanly capturing items as they finish validating, avoiding yielding partial or unvalidated structures. Resolve optimal debouncing; the default debouncing in `stream_output()` is 0.1, which may be too small, but the user has no clean way to specify this via the function signature unless we grab `debounce_by` as a kwarg which might border on being too magical. 
- **Type Checking for Prompted and Native Output**: Models that require Prompted/Native Output will require the user to wrap the type hint in PromptedOutput/NativeOutput on a separate line, but it is unclear if this simply sidesteps python type checking concerns.
- **Handling DeferredToolRequests/DeferredToolResults**: The decorator logic does not consider if the agent calls a tool that requires approval and returns a `DeferredToolRequest`. This will cause the decorator to fail to return the correct type. Similarly, the argument handling for the decorator does not detect `DeferredToolResult` and include it in the agent run.
- **Passing message history** Currently does not contain logic for detecting arguments typehinted as an input sequence of `ModelMessages` for usage as the message history. 
- **Multiple Arguments matching `deps_type`**: How to resolve which argument to pass to the `.run(deps=...)` param if multiple are typed similarly?


## 5. Dependencies
- **Standard Library:** `inspect`, `functools`, `typing`, `collections.abc`
- **Pydantic Ecosystem:** `pydantic.BaseModel`
- **Pydantic AI:** `Agent`, `format_as_xml`, `PromptedOutput`, `NativeOutput`, `RunContext`, and Media/URL content types.
- **Pydantic Graph (Beta):** `GraphBuilder`, `StepContext`, `join`

---

## 6. Development Experience Test Cases (Demos)
Use the following demos as behavioral test cases for the implementation:

### Demo 1: Multimodal Input & `@g.step`
Demonstrates stacking `@g.step` with the functional agent for an OCR pipeline, handling `BinaryContent` and `StepContext`.
```python
if __name__ == "__main__":
    import asyncio
    import os
    from dataclasses import dataclass
    from pathlib import Path
    
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_graph.beta import GraphBuilder, StepContext
    from pydantic_graph.beta.join import reduce_list_append

    # Map to local Ollama API
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"

    gemma = OpenAIChatModel('gemma4:e2b')
    @dataclass
    class StepState:
        pass

    g1 = GraphBuilder(state_type=StepState, input_type=Path, output_type=str)

    @g1.step
    async def load_image(ctx: StepContext[StepState, None, Path]) -> BinaryContent:
        """Load an image file from disk."""
        print(f"-> Loading {ctx.inputs}...")
        return BinaryContent(data=ctx.inputs.read_bytes(), media_type='image/png')

    @g1.step
    @Agent(gemma)
    async def ocr_extract(ctx: StepContext[StepState, None, BinaryContent]) -> str:
        """You are an OCR extraction agent. Extract all visible text from the provided image. Return ONLY the raw text."""
        pass

    g1.add(
        g1.edge_from(g1.start_node).to(load_image),
        g1.edge_from(load_image).to(ocr_extract),
        g1.edge_from(ocr_extract).to(g1.end_node),
    )

async def main():
        print("=== Demo 1: OCR Pipeline (gemma4:e2b) ===\n")

        graph1 = g1.build()
        result1 = await graph1.run(state=StepState(), inputs=Path('test.png'))
        print(f"OCR Result: {result1}\n")

    asyncio.run(main())

```

### Demo 2: Streaming & `@g.stream`
Showcases returning an `AsyncIterable` to organically invoke `run_stream`, combined with a graph step.
```python
if __name__ == "__main__":
    import asyncio
    import os
    from dataclasses import dataclass
    from pathlib import Path
    
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_graph.beta import GraphBuilder, StepContext
    from pydantic_graph.beta.join import reduce_list_append

    # Map to local Ollama API
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"

    gemma = OpenAIChatModel('gemma4:e2b')

class Idea(BaseModel):
        title: str
        description: str

    @dataclass
    class StreamState:
        chunks_processed: int = 0

    g2 = GraphBuilder(state_type=StreamState, output_type=list[str])

    @g2.stream
    @Agent(gemma)
    async def stream_ideas(ctx: StepContext[StreamState, None, None]) -> AsyncIterable[PromptedOutput(Idea)]:
        """Generate 10 creative Python project ideas. Each idea should have a short title and a one-sentence description."""
        pass

    @g2.step
    async def format_idea(ctx: StepContext[StreamState, None, Idea]) -> str:
        """Format a single idea with a bullet prefix."""
        ctx.state.chunks_processed += 1
        print(f"Processing idea {ctx.inputs.title}")
        return f"• {ctx.inputs.title}: {ctx.inputs.description}"

    collect = g2.join(reduce_list_append, initial_factory=list[str])

    g2.add(
        g2.edge_from(g2.start_node).to(stream_ideas),
        g2.edge_from(stream_ideas).map().to(format_idea),
        g2.edge_from(format_idea).to(collect),
        g2.edge_from(collect).to(g2.end_node),
    )
    async def main():
        print("=== Demo 2: Streaming @g.stream + @Agent (gemma4:e2b) ===\n")

        state2 = StreamState()
        graph2 = g2.build()
        result2 = await graph2.run(state=state2)
        print(f"Ideas generated: {state2.chunks_processed}")
        for idea in result2:
            print(idea)

    asyncio.run(main())
```