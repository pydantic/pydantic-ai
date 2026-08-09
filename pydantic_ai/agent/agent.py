def process_history(self, message_history: List[Message], context: ProcessorContext) -> List[Message]:
    # Updated to support realtime model context
    processed = message_history
    for processor in self.history_processors:
        processed = processor.process(processed, context)
    # Apply ProcessHistory capability if available
    if hasattr(self, 'process_history_capability'):
        processed = self.process_history_capability.apply(processed, context)
    return processed
