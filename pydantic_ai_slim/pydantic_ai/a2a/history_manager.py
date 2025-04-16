from abc import ABC, abstractmethod

from pydantic_ai.a2a.schema import Message


class HistoryManager(ABC):
    @abstractmethod
    async def retrieve_history(self, task_id: str, history_length: int | None) -> list[Message]: ...
