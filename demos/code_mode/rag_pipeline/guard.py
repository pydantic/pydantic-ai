"""Guard toolset for Pinecone API limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.toolsets.abstract import AgentDepsT, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset


@dataclass(kw_only=True)
class PineconeGuardToolset(WrapperToolset[AgentDepsT]):
    """Guards against Pinecone API limits by pre-truncating documents.

    The rerank API has token limits per query+document pair:
    - pinecone-rerank-v0: 512 tokens
    - bge-reranker-v2-m3: 1024 tokens
    - cohere-rerank-3.5: 40,000 tokens (requires authorization)

    This toolset truncates documents to stay within limits.
    Using ~3 chars per token as a conservative estimate.
    """

    max_chars_per_doc: int = 3000  # ~730 tokens (at 4.1 chars/token), safe for 1024 limit

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        if name == 'rerank-documents':
            tool_args = self._truncate_documents(tool_args)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    def _truncate_documents(self, args: dict[str, Any]) -> dict[str, Any]:
        docs = args.get('documents', [])
        if not docs:
            return args

        truncated: list[str | dict[str, Any]] = []
        for doc in docs:
            if isinstance(doc, str):
                truncated.append(doc[: self.max_chars_per_doc])
            elif isinstance(doc, dict):
                doc = dict(doc)
                for key in ('text', 'content'):
                    if key in doc and isinstance(doc[key], str):
                        doc[key] = doc[key][: self.max_chars_per_doc]
                truncated.append(doc)
            else:
                truncated.append(doc)

        return {**args, 'documents': truncated}
