import json
from typing import List
from llama_cpp import Llama

from core.base_llm import BaseLLM
from core.schemas import ParsedQuery, Chunk

class MicroParserLLM(BaseLLM):
    """
    An ultra-lightweight LLM dedicated ONLY to parsing queries.
    It uses a strictly constrained JSON format to guarantee 100% valid outputs.
    """

    def __init__(self, model_path: str, n_ctx: int = 512):
        """
        Initializes the micro LLM. Context size is kept very small (512) for max speed.
        """
        print(f"[MicroParserLLM] Loading tiny model from: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1, # Fast GPU offloading
            verbose=False
        )

    def parse_query(self, raw_query: str) -> ParsedQuery:
        """
        Extracts library_name and rewrites the query using JSON Schema enforcement.
        """
        print("[MicroParserLLM] Parsing raw query extremely fast...")
        
        system_prompt = (
            "You are a routing agent. Extract 'library_name' from the query if it exists "
            "(e.g., pandas, numpy, sklearn). Rewrite the query to be clear for semantic search. "
            "Respond strictly in JSON."
        )
        
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{raw_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Force the tiny model to output JSON adhering to our exact schema
        json_schema = {
            "type": "object",
            "properties": {
                "optimized_query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "library_name": {"type": "string"}
                    }
                }
            },
            "required": ["optimized_query", "filters"]
        }

        response = self.llm(
            prompt,
            max_tokens=100,
            temperature=0.0, # 0.0 for absolute deterministic extraction
            response_format={
                "type": "json_object",
                "schema": json_schema
            },
            stop=["<|im_end|>"]
        )
        
        output_text = response["choices"][0]["text"].strip()
        parsed_data = json.loads(output_text)
        
        return ParsedQuery(
            optimized_query=parsed_data.get("optimized_query", raw_query),
            filters=parsed_data.get("filters", {})
        )

    def generate_answer(self, query: str, context_chunks: List[Chunk]) -> str:
        """
        This micro model is NOT used for generating answers. 
        """
        raise NotImplementedError("MicroParserLLM is only designed for query parsing.")