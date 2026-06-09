from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from core.retriever.base import BaseRetriever
from core.schemas import Chunk
from database.hybrid_search import PostgresHybridSearch
from utils.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)

class PGHybridRetrieverModel(BaseRetriever):
    """
    Implementation of BaseRetriever using PostgreSQL.
    Delegates the raw SQL execution to PostgresHybridSearch.
    """

    def __init__(self, config: Dict[str, Any]):
        # 1. Load embedding model
        embed_model_path = config.get("embed_model_path", "BAAI/bge-small-en-v1.5")
        logger.info(f"Loading local Embedding model from: {embed_model_path}")
        self.embed_model = SentenceTransformer(embed_model_path)

        # 2. Load fusion weights
        weights = config.get("weights", {})
        self.semantic_weight = float(weights.get("semantic", 0.7))

        # 3. Khởi tạo đối tượng giao tiếp Database chuyên dụng
        logger.info("Initializing PostgresHybridSearch connection...")
        self.db_search = PostgresHybridSearch()

    def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        logger.info(f"Executing hybrid search for query: '{query}'")

        try:
            # 1. Nhúng (Embed) câu hỏi thành vector
            # Đã sửa lỗi: dùng self.embed_model thay vì self.model
            query_vector = self.embed_model.encode(query, normalize_embeddings=True).tolist()

            # 2. Gọi tầng Database thực thi truy vấn SQL
            raw_results = self.db_search.execute_search(
                query_vector=query_vector,
                query_text=query,
                top_k=top_k,
                semantic_weight=self.semantic_weight,
                filters=filters
            )

            # 3. Map (Ánh xạ) kết quả từ Database thành cấu trúc Chunk chuẩn
            chunks = []
            for row in raw_results:
                # Trích xuất metadata
                metadata = {
                    "library_name": row.get("library_name", ""),
                    "func_name": row.get("func_name", ""),
                    "parameters": row.get("parameters", {})
                }

                # Tạo đối tượng Chunk
                chunks.append(
                    Chunk(
                        id=str(row.get("id")),
                        content=row.get("docstring", ""),
                        metadata=metadata,
                        score=float(row.get("hybrid_score", 0.0)),
                    )
                )
                
            logger.info(f"Retrieved {len(chunks)} chunks successfully from Database.")
            return chunks
            
        except Exception as e:
            logger.error(f"Retrieval pipeline failed: {str(e)}")
            return []
            
