import torch
from typing import List
from sentence_transformers import CrossEncoder

from core.base_reranker import BaseReranker
from core.schemas import Chunk

class BGEReranker(BaseReranker):
    """
    Implementation of the BaseReranker using BAAI's BGE-Reranker models.
    It uses a Cross-Encoder architecture to compute the semantic relevance 
    between a query and a list of chunks with full attention.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 512):
        """
        Initializes the BGE Reranker model.
        
        Args:
            model_name (str): HuggingFace model ID or local path. Default is the 'base' version 
                              for optimal performance on end-devices.
            max_length (int): Maximum token length for the cross-encoder inputs.
        """
        print(f"[BGEReranker] Loading cross-encoder model: {model_name}...")
        
        # Automatically detect GPU if available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the CrossEncoder model
        # Using float16 if on GPU to save VRAM and increase inference speed
        model_kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
        
        self.model = CrossEncoder(
            model_name, 
            max_length=max_length, 
            device=self.device,
            model_kwargs=model_kwargs
        )
        print(f"[BGEReranker] Model loaded successfully on {self.device.upper()}.")

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
        """
        Re-evaluates the chunks against the query using the cross-encoder.
        
        Args:
            query (str): The user's search query.
            chunks (List[Chunk]): The initial candidates retrieved by the fast retriever.
            top_k (int): Number of top chunks to return after reranking.
            
        Returns:
            List[Chunk]: The reranked chunks sorted by cross-encoder score.
        """
        if not chunks:
            return []

        # Step 1: Format inputs for the CrossEncoder
        # The model requires a list of pairs: [[query, doc1], [query, doc2], ...]
        sentence_pairs = [[query, chunk.content] for chunk in chunks]

        # Step 2: Predict relevance scores
        # show_progress_bar=False to keep terminal logs clean during API calls
        scores = self.model.predict(sentence_pairs, show_progress_bar=False)

        # Step 3: Assign the new scores to the chunks
        for chunk, score in zip(chunks, scores):
            # Convert numpy float32 to standard Python float
            chunk.score = float(score)

        # Step 4: Sort chunks in descending order based on the new scores
        chunks.sort(key=lambda x: x.score, reverse=True)

        # Step 5: Return only the top_k most relevant chunks
        return chunks[:top_k]