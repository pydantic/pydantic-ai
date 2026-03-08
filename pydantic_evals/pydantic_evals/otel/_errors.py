class SpanTreeRecordingError(Exception):
    """An exception that is used to provide the reason why a SpanTree was not recorded by `context_subtree`.

    This may be due to missing dependencies, a tracer provider not having been set, or a custom TracerProvider
    that does not support `add_span_processor`.
    """

    pass
