from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

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

class SourceChunk(BaseModel):
    id: str
    function_name: str
    library_name: str
    score: Optional[float] = None
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    snippet: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class RAGResponse(BaseModel):
    query: str
    optimized_query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    answer: str
    sources: List[SourceChunk] = Field(default_factory=list)
