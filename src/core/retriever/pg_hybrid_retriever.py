from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from core.retriever.base import BaseRetriever
from core.schemas import Chunk
# Fixed Import: Using the exact class name present in your database module
from database.hybrid_search import PostgreSQLHybridSearch
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

class PGHybridRetrieverModel(BaseRetriever):
    """
    Implementation of BaseRetriever using PostgreSQL.
    Delegates the raw SQL execution to PostgreSQLHybridSearch.
    """

    def __init__(self, config: Dict[str, Any]):
        embed_model_path = config.get("embed_model_path", "BAAI/bge-small-en-v1.5")
        logger.info(f"Loading local Embedding model from: {embed_model_path}")
        self.embed_model = SentenceTransformer(embed_model_path)

        weights = config.get("weights", {})
        self.semantic_weight = float(weights.get("semantic", 0.7))

        logger.info("Initializing PostgreSQLHybridSearch connection...")
        
        # Load database credentials from global config to pass into the search class
        db_config = GLOBAL_CONFIG.get("database", {})
        self.db_search = PostgreSQLHybridSearch(db_config=db_config)

    def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        logger.info(f"Executing hybrid search for query: '{query}'")

        try:
            # 1. Generate vector embedding for the search query
            query_vector = self.embed_model.encode(query, normalize_embeddings=True).tolist()

            # 2. Execute the database SQL query using the correct method name `search`
            raw_results = self.db_search.search(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters
            )

            # 3. Map raw database rows into standard Chunk objects
            chunks = []
            for row in raw_results:
                metadata = {
                    "library_name": row.get("library_name", ""),
                    "func_name": row.get("func_name") or row.get("full_name", ""),
                    "parameters": row.get("parameters", {})
                }

                chunks.append(
                    Chunk(
                        id=str(row.get("id")),
                        content=row.get("docstring", ""),
                        metadata=metadata,
                        score=float(row.get("hybrid_score", 0.0)),
                    )
                )
                
            logger.info(f"Successfully retrieved {len(chunks)} chunks from Database.")
            return chunks
            
        except Exception as e:
            logger.error(f"Retrieval pipeline failed during execution: {str(e)}")
            return []