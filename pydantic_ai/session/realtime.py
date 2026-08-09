def seed_message_history(self, message_history: List[Message]) -> None:
    # Process message history through history processors and ProcessHistory capability before seeding
    processed_history = self.agent.process_history(message_history, context=self.get_processor_context())
    self.provider_conversation.messages = processed_history
