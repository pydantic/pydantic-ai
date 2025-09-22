from typing import Any, Optional

from pydantic_ai.durable_exec.restate._serde import PydanticTypeAdapter
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel

from restate import Context, RunOptions

class RestateModelWrapper(WrapperModel):
    
    def __init__(self,
                 wrapped: Model,
                 context: Context,
                 max_attempts: Optional[int] = None):
        super().__init__(wrapped)
        self.options = RunOptions[ModelResponse](serde=PydanticTypeAdapter(ModelResponse), max_attempts=max_attempts)
        self.context = context
        
    async def request(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return await self.context.run_typed("Model call", self.wrapped.request, self.options, *args, **kwargs)
 