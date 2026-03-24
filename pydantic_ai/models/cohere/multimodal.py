from typing import List, Union
from pydantic import BaseModel

class ImagePart(BaseModel):
    url: str

class MultimodalMessage(BaseModel):
    """
    Multimodal message support for Cohere Command A Vision.
    Addresses issue #3703.
    """
    role: str
    content: Union[str, List[Union[str, ImagePart]]]

class CohereMultimodalHandler:
    @staticmethod
    def format_parts(parts: List[Union[str, ImagePart]]):
        formatted = []
        for part in parts:
            if isinstance(part, ImagePart):
                formatted.append({"type": "image_url", "image_url": {"url": part.url}})
            else:
                formatted.append({"type": "text", "text": part})
        return formatted
