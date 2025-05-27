class SpanTreeRecordingError(Exception):
    """An exception that is used to provide the reason why a SpanTree was not recorded by `context_subtree`.

    This will either be due to missing dependencies or because a tracer provider had not been set.
    """

    pass
