import inspect
import collections.abc
from functools import wraps
from typing import AsyncIterable, Iterable, Generator, AsyncGenerator, get_origin, get_args, Any

from pydantic import BaseModel
from pydantic_ai import (
    Agent, format_as_xml, PromptedOutput, NativeOutput,
    ImageUrl, AudioUrl, VideoUrl, DocumentUrl, 
    BinaryContent, UploadedFile, TextContent
)

def _is_stream(annotation):
    origin = get_origin(annotation)
    return origin in (
        Iterable, AsyncIterable,
        collections.abc.Iterable, collections.abc.AsyncIterable,
        Generator, AsyncGenerator,
        collections.abc.Generator, collections.abc.AsyncGenerator
    )

def _get_base_type(annotation):
    if _is_stream(annotation):
        args = get_args(annotation)
        return args[0] if args else str
    return annotation

def _wrap_agent_function(target_agent: Agent, func):
    sig = inspect.signature(func)
    raw_rt = sig.return_annotation
    is_async = inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
    is_stream = _is_stream(raw_rt)
    
    # 1. Infer Output Types
    raw_base_type = _get_base_type(raw_rt) if raw_rt is not inspect._empty else str
    
    # Check if the user explicitly opted into PromptedOutput or NativeOutput
    # via their type annotation (e.g. AsyncIterable[PromptedOutput(Idea)])
    _is_output_wrapper = isinstance(raw_base_type, (PromptedOutput, NativeOutput))
    
    if _is_output_wrapper:
        # User annotated e.g. AsyncIterable[PromptedOutput(Idea)] or -> NativeOutput(Idea)
        # For streaming: wrap the inner type in list[] for diff-based fan-out
        inner_type = raw_base_type.outputs
        wrapper_cls = type(raw_base_type)
        if is_stream:
            output_type = wrapper_cls(list[inner_type])
        else:
            output_type = raw_base_type
    elif is_stream and raw_base_type is not str:
        # Streaming structured output without PromptedOutput: wrap in list[]
        # for diff-based fan-out, using tool-call-based extraction
        output_type = list[raw_base_type]
    else:
        output_type = raw_base_type
    
    # 2. Infer Instructions
    doc = inspect.getdoc(func)
    instructions = doc if doc else None

    def _prep_run_context(*args, **kw):
        bound = sig.bind(*args, **kw)
        bound.apply_defaults()

        input_dict = {}
        multimodal_parts = []
        deps = None
        
        for name, value in bound.arguments.items():
            param = sig.parameters[name]
            param_anno_str = str(param.annotation)
            
            # --- Pr.md Proposed Additions ---
            
            # 1. StepContext Handling (Graph API)
            if 'StepContext' in param_anno_str:
                # Extract the RunContext argument deps and forward it to run*()
                deps = getattr(value, 'deps', None)
                # Extract the inputs field and pass it as part of the userprompt
                value = getattr(value, 'inputs', value)
                # Fall through to standard payload parsing with the extracted inputs

            # 2. RunContext Handling (Agent-As-A-Tool API)
            if 'RunContext' in param_anno_str:
                # Extract the RunContext argument deps and forward it to run*()
                deps = getattr(value, 'deps', None)
                continue
                
            # TODO: Resolve handling multiple args of deps_type?
            # 3. Dependencies Injection Mapping
            # deps_type = getattr(target_agent, 'deps_type', None)
            # if deps_type and deps_type is not type(None) and param.annotation == deps_type:
            #     deps = value
            #     continue

            # --- Base Logic ---

            # Multimodal Payload Checks
            if isinstance(value, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent, UploadedFile, TextContent)):
                multimodal_parts.append(value)
                continue
            
            # Lists of Multimodal Payloads
            if isinstance(value, list) and value and all(isinstance(v, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent, UploadedFile, TextContent)) for v in value):
                multimodal_parts.extend(value)
                continue

            input_dict[name] = value
            
        xml_prompt = format_as_xml(input_dict) if input_dict else None
        
        if multimodal_parts:
            user_prompt = [xml_prompt] + multimodal_parts if xml_prompt else multimodal_parts
        else:
            user_prompt = xml_prompt
            
        return user_prompt, deps

    # Execute using the target_agent with injected variables
    run_kwargs = {}
    if instructions:
        run_kwargs['instructions'] = instructions

    if is_async and is_stream:
        @wraps(func)
        async def async_gen_wrapper(*args, **kw):
            prompt, extracted_deps = _prep_run_context(*args, **kw)
            async with target_agent.run_stream(prompt, output_type=output_type, deps=extracted_deps, **run_kwargs) as res:
                if raw_base_type is str:
                    async for chunk in res.stream_text(delta=True): yield chunk
                else:
                    # Structured streaming: agent produces a JSON array via PromptedOutput.
                    # Diff successive partial parses to yield only newly completed items.
                    # Debounce to let partial objects fill in before yielding.
                    prev_len = 0 # note, the user cannot set debouncing when using the decorator.
                    # TODO: resolve debouncing.
                    async for partial_list in res.stream_output(): # could try to capture from decorated function as a kwarg but would start to get too magical.
                        if isinstance(partial_list, list) and len(partial_list) > prev_len:
                            for item in partial_list[prev_len:]:
                                yield item
                            prev_len = len(partial_list)
        wrapper = async_gen_wrapper

    elif is_async and not is_stream:
        @wraps(func)
        async def async_wrapper(*args, **kw):
            prompt, extracted_deps = _prep_run_context(*args, **kw)
            res = await target_agent.run(prompt, output_type=output_type, deps=extracted_deps, **run_kwargs)
            return res.output if hasattr(res, 'output') else res.data
        wrapper = async_wrapper
        
    elif not is_async and is_stream:
        @wraps(func)
        def sync_gen_wrapper(*args, **kw):
            prompt, extracted_deps = _prep_run_context(*args, **kw)
            with target_agent.run_sync_stream(prompt, output_type=output_type, deps=extracted_deps, **run_kwargs) as res:
                if output_type is str:
                    for chunk in res.stream_text(delta=True): yield chunk
                else:
                    for chunk in res.stream_output(): yield chunk
        wrapper = sync_gen_wrapper

    else:
        @wraps(func)
        def sync_wrapper(*args, **kw):
            prompt, extracted_deps = _prep_run_context(*args, **kw)
            res = target_agent.run_sync(prompt, output_type=output_type, deps=extracted_deps, **run_kwargs)
            return res.output if hasattr(res, 'output') else res.data
        wrapper = sync_wrapper

    wrapper.__agent__ = target_agent
    
    # Sanitize annotations so external decorators (e.g. @agent.tool) see
    # clean types, not PromptedOutput/NativeOutput wrappers
    if hasattr(wrapper, '__annotations__') and 'return' in wrapper.__annotations__:
        ret_anno = wrapper.__annotations__['return']
        if isinstance(ret_anno, (PromptedOutput, NativeOutput)):
            wrapper.__annotations__ = {**wrapper.__annotations__, 'return': ret_anno.outputs}
    
    return wrapper


def patched_agent_call(self, arg: Any = None, **kwargs):
    if callable(arg):
        return _wrap_agent_function(self, arg)
    return lambda f: _wrap_agent_function(self, f)

# Patch the Agent class globally
Agent.__call__ = patched_agent_call


# --- Graph Demonstrations ---

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

    # ──────────────────────────────────────────────
    # Demo 0: Agent-as-a-Tool with Dependency Injection
    # ──────────────────────────────────────────────
    # 
    # The @Agent decorator handles RunContext natively:
    # - Strips RunContext from the XML prompt
    # - Extracts deps and forwards them to the inner agent
    # - The decorated function can be directly registered as a tool
    #   on another agent — no manual wrapper needed.

    from pydantic_ai import RunContext

    class Theme(BaseModel):
        topic: str
        tone: str  # e.g. "dark", "wholesome", "dad-joke"

    # The selection agent — standard Pydantic AI Agent with deps_type
    literary_agent = Agent(
        gemma,
        deps_type=Theme,
        instructions=(
            'Delegate to the correct tool to return the requested writing product.'
        ),
    )

    writer_agent = Agent(
        gemma,
        deps_type=Theme,
        instructions="Get the theme before writing your response."
    )

    @writer_agent.tool
    async def get_theme(ctx: RunContext):
        print(f"Theme: {ctx.deps}")
        return format_as_xml(ctx.deps)

    # Stack @tool directly on @Agent — the decorated function receives
    # RunContext from the parent agent and structured args from the LLM.
    # The decorator strips RunContext, formats (count,) as XML, and runs the inner agent.
    # TODO: is this the recommended way to handle this?
    JokeList = PromptedOutput(list[str]) # PyreFly doesn't like function calls as the type hint.

    @literary_agent.tool
    @writer_agent
    async def jokes(ctx: RunContext[Theme], count: int = 5) -> JokeList:
        """Generate jokes. The count parameter controls how many jokes to generate."""
        pass

    @literary_agent.tool
    @writer_agent
    async def poem(ctx: RunContext[Theme]) -> str:
        """Generate a poem based on the theme."""
        pass

    @literary_agent.tool
    @writer_agent
    async def essay(ctx: RunContext[Theme], num_paragraphs: int = 3) -> str:
        """Generate a poem based on the theme."""
        pass

    # ──────────────────────────────────────────────
    # Demo 1: OCR Pipeline — @g.step + @Agent with multimodal input
    # ──────────────────────────────────────────────

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

    # ──────────────────────────────────────────────
    # Demo 2: Streaming @g.stream + @Agent decorator
    # ──────────────────────────────────────────────

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
        print("=== Demo 0: Agent-as-a-Tool + Deps (gemma4:e2b) ===\n")

        theme = Theme(topic="programming", tone="dad-joke")
        result0 = await literary_agent.run(
            'Tell me a joke.',
            deps=theme,
        )
        print(f"Best joke: {result0.output}\n")

        theme = Theme(topic="The moon", tone="wonder")
        result0 = await literary_agent.run(
            'write a poem.',
            deps=theme,
        )
        print(f"Poem: {result0.output}\n")

        print("=== Demo 1: OCR Pipeline (gemma4:e2b) ===\n")

        graph1 = g1.build()
        result1 = await graph1.run(state=StepState(), inputs=Path('test.png'))
        print(f"OCR Result: {result1}\n")

        print("=== Demo 2: Streaming @g.stream + @Agent (gemma4:e2b) ===\n")

        state2 = StreamState()
        graph2 = g2.build()
        result2 = await graph2.run(state=state2)
        print(f"Ideas generated: {state2.chunks_processed}")
        for idea in result2:
            print(idea)

    asyncio.run(main())

