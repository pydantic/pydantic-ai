```python
# ---------------------------------------------------------------------------
# Functional decorator helpers  (@agent_instance as a decorator)
# ---------------------------------------------------------------------------

def _is_fn_stream(annotation: Any) -> bool:
    """Return True if *annotation* represents a streaming return type."""
    origin = get_origin(annotation)
    return origin in (
        Iterable, AsyncIterable,
        Generator, AsyncGenerator,
        # also accept the bare collections.abc variants
        __import__('collections').abc.Iterable,
        __import__('collections').abc.AsyncIterable,
        __import__('collections').abc.Generator,
        __import__('collections').abc.AsyncGenerator,
    )


def _get_fn_base_type(annotation: Any) -> Any:
    """Strip streaming wrapper and return the inner type (or the annotation itself)."""
    if _is_fn_stream(annotation):
        args = get_args(annotation)
        return args[0] if args else str
    return annotation


def _wrap_agent_function(target_agent: AbstractAgent[Any, Any], func: Callable[..., Any]) -> Callable[..., Any]:
    """Return a callable that, when invoked, runs *func*'s arguments through
    *target_agent* using the appropriate ``run`` / ``run_stream`` variant.

    The wrapped function:
    * Uses *func*'s docstring as per-run ``instructions``.
    * Binds all arguments via :func:`inspect.Signature.bind` and formats them
      as XML for the ``user_prompt`` (skipping ``RunContext`` / ``StepContext``
      args, which are used for dependency injection instead).
    * Passes multimodal values (``ImageUrl``, ``BinaryContent``, …) directly
      as prompt parts rather than serialising them to XML.
    * Returns / yields the agent output matching *func*'s return annotation.
    """
    sig = inspect.signature(func)
    raw_rt = sig.return_annotation
    is_async = inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
    is_stream = _is_fn_stream(raw_rt)

    # --- 1. Infer output type ---
    raw_base_type = _get_fn_base_type(raw_rt) if raw_rt is not inspect.Parameter.empty else str

    _is_output_wrapper = isinstance(raw_base_type, (PromptedOutput, NativeOutput))

    if _is_output_wrapper:
        inner_type = raw_base_type.outputs
        wrapper_cls = type(raw_base_type)
        output_type: Any = wrapper_cls(list[inner_type]) if is_stream else raw_base_type  # type: ignore[valid-type]
    elif is_stream and raw_base_type is not str:
        output_type = list[raw_base_type]  # type: ignore[valid-type]
    else:
        output_type = raw_base_type

    # --- 2. Infer instructions from docstring ---
    doc = inspect.getdoc(func)
    run_kwargs: dict[str, Any] = {}
    if doc:
        run_kwargs['instructions'] = doc

    # --- 3. Multimodal type sentinel (imported lazily to avoid circular deps) ---
    from ..messages import (
        AudioUrl, BinaryContent, DocumentUrl, ImageUrl, TextContent, UploadedFile, VideoUrl,
    )
    _MULTIMODAL_TYPES = (ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent, UploadedFile, TextContent)

    def _prep_run_context(*args: Any, **kw: Any) -> tuple[Any, Any]:
        """Bind *args*/*kw* to *func*'s signature and return ``(user_prompt, deps)``."""
        bound = sig.bind(*args, **kw)
        bound.apply_defaults()

        input_dict: dict[str, Any] = {}
        multimodal_parts: list[Any] = []
        deps: Any = None

        for name, value in bound.arguments.items():
            param = sig.parameters[name]
            param_anno_str = str(param.annotation)

            # StepContext: extract .inputs for the prompt, .deps for injection
            if 'StepContext' in param_anno_str:
                deps = getattr(value, 'deps', None)
                value = getattr(value, 'inputs', value)
                # fall through to normal payload handling with extracted inputs

            # RunContext: extract .deps, skip this arg entirely
            if 'RunContext' in param_anno_str:
                deps = getattr(value, 'deps', None)
                continue

            # Multimodal single value
            if isinstance(value, _MULTIMODAL_TYPES):
                multimodal_parts.append(value)
                continue

            # List of multimodal values
            if (
                isinstance(value, list)
                and value
                and all(isinstance(v, _MULTIMODAL_TYPES) for v in value)
            ):
                multimodal_parts.extend(value)
                continue

            input_dict[name] = value

        xml_prompt: str | None = format_as_xml(input_dict) if input_dict else None

        if multimodal_parts:
            user_prompt: Any = ([xml_prompt] + multimodal_parts) if xml_prompt else multimodal_parts
        else:
            user_prompt = xml_prompt

        return user_prompt, deps

    # --- 4. Build the appropriate wrapper ---
    if is_async and is_stream:
        @wraps(func)
        async def _async_gen_wrapper(*args: Any, **kw: Any) -> AsyncGenerator[Any, None]:  # type: ignore[misc]
            prompt, extracted_deps = _prep_run_context(*args, **kw)
            async with target_agent.run_stream(
                prompt, output_type=output_type, deps=extracted_deps, **run_kwargs
            ) as res:
                if raw_base_type is str:
                    async for chunk in res.stream_text(delta=True):
                        yield chunk
                else:
                    prev_len = 0
                    async for partial_list in res.stream_output():
                        if isinstance(partial_list, list) and len(partial_list) > prev_len:
                            for item in partial_list[prev_len:]:
                                yield item
                            prev_len = len(partial_list)
        wrapper: Callable[..., Any] = _async_gen_wrapper

    elif is_async:
        @wraps(func)
        async def _async_wrapper(*args: Any, **kw: Any) -> Any:
            prompt, extracted_deps = _prep_run_context(*args, **kw)
            res = await target_agent.run(
                prompt, output_type=output_type, deps=extracted_deps, **run_kwargs
            )
            return res.output if hasattr(res, 'output') else res.data  # type: ignore[union-attr]
        wrapper = _async_wrapper

    elif is_stream:
        @wraps(func)
        def _sync_gen_wrapper(*args: Any, **kw: Any) -> Generator[Any, None, None]:
            prompt, extracted_deps = _prep_run_context(*args, **kw)
            with target_agent.run_stream_sync(  # type: ignore[attr-defined]
                prompt, output_type=output_type, deps=extracted_deps, **run_kwargs
            ) as res:
                if output_type is str:
                    for chunk in res.stream_text(delta=True):
                        yield chunk
                else:
                    for chunk in res.stream_output():
                        yield chunk
        wrapper = _sync_gen_wrapper

    else:
        @wraps(func)
        def _sync_wrapper(*args: Any, **kw: Any) -> Any:
            prompt, extracted_deps = _prep_run_context(*args, **kw)
            res = target_agent.run_sync(
                prompt, output_type=output_type, deps=extracted_deps, **run_kwargs
            )
            return res.output if hasattr(res, 'output') else res.data  # type: ignore[union-attr]
        wrapper = _sync_wrapper

    wrapper.__agent__ = target_agent  # type: ignore[attr-defined]

    # Sanitise annotations: strip PromptedOutput/NativeOutput wrappers so that
    # external decorators (e.g. @agent.tool) see clean, plain types.
    if hasattr(wrapper, '__annotations__') and 'return' in wrapper.__annotations__:
        ret_anno = wrapper.__annotations__['return']
        if isinstance(ret_anno, (PromptedOutput, NativeOutput)):
            wrapper.__annotations__ = {**wrapper.__annotations__, 'return': ret_anno.outputs}

    return wrapper

```