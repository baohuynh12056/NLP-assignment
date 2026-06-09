from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class Chunk(BaseModel):
    """
    Represents a retrieved document chunk.
    Acts as the standard data transfer object across the pipeline.
    """

    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class ParsedQuery(BaseModel):
    """
    Represents the query after being optimized and parsed by the Mini-LLM.
    Contains both the refined text and extracted metadata filters.
    """

    optimized_query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
