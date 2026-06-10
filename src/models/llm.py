import json
import re
from typing import List
from llama_cpp import Llama

from core.base_llm import BaseLLM
from core.schemas import ParsedQuery, Chunk
from pipeline.prompt_builder import PromptBuilder

class QwenLocalLLM(BaseLLM):
    """
    Implementation of BaseLLM using a local Qwen instruct model in GGUF format.
    Runs locally via llama.cpp, highly optimized for end-devices with GPU offloading.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        """
        Initializes the local Qwen model.
        
        Args:
            model_path (str): Path to the downloaded .gguf file.
            n_ctx (int): Context window size. 4096 is usually enough for RAG snippets.
            n_gpu_layers (int): Number of layers to offload to GPU. 
                                -1 means offload ALL layers (maximum speed).
        """
        print(f"[QwenLocalLLM] Loading GGUF model from: {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False  # Set to True if you want to see C++ engine logs
        )
        print("[QwenLocalLLM] Model loaded successfully.")

    def parse_query(self, raw_query: str) -> ParsedQuery:
        """
        Uses the LLM to extract metadata filters (like library names) 
        and rewrite the query for better semantic vector search.
        """
        print("[QwenLocalLLM] Parsing raw query...")
        
        # Qwen uses ChatML prompt format: <|im_start|>role\ncontent<|im_end|>
        system_prompt = (
            "You are an AI assistant for Python developers. "
            "Extract 'library_name' from the user's query if mentioned (e.g., pandas, numpy, sklearn). "
            "Rewrite the query into a clear, semantic sentence for a vector database search. "
            "Always respond in strictly valid JSON format with keys: 'optimized_query' and 'filters'."
        )
        
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{raw_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # We constrain the output to a small number of tokens since it's just JSON
        response = self.llm(
            prompt,
            max_tokens=150,
            temperature=0.1, # Low temperature for deterministic JSON output
            stop=["<|im_end|>"]
        )
        
        output_text = response["choices"][0]["text"].strip()
        
        # Safely parse the JSON output
        try:
            # Strip markdown code blocks if the LLM adds them (e.g., ```json ... ```)
            if output_text.startswith("```json"):
                output_text = output_text[7:-3].strip()
                
            parsed_data = json.loads(output_text)
            return ParsedQuery(
                optimized_query=parsed_data.get("optimized_query", raw_query),
                filters=parsed_data.get("filters", {})
            )
        except json.JSONDecodeError:
            print(f"[QwenLocalLLM] Warning: Failed to parse JSON. Raw output: {output_text}")
            # Fallback to the original query if parsing fails
            return ParsedQuery(optimized_query=raw_query, filters={})

    def generate_answer(self, query: str, context_chunks: List[Chunk]) -> str:
        """
        Generates the final comprehensive answer using the retrieved context chunks.
        """
        print(f"[QwenLocalLLM] Generating answer using {len(context_chunks)} chunks...")
        
        system_prompt = (
            "You are a concise Python documentation assistant for students and developers. "
            "Use ONLY the retrieved documentation in the user message. "
            "Answer in the detected user language specified in the user message. "
            "Prefer a practical answer over a long explanation: name the API, show one short code example, "
            "then explain the key parameters or caveats that matter for the question. "
            "Do not invent arguments, defaults, or behavior not present in the retrieved documentation. "
            "If the context is weak or unrelated, say that the exact answer was not found in the documentation."
        )
        
        user_message = PromptBuilder.build_user_message(query, context_chunks) + "\n/no_think"
        
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # 3. Generate stream or full response
        response = self.llm(
            prompt,
            max_tokens=800,
            temperature=0.3,
            stop=["<|im_end|>"]
        )
        
        return self._strip_thinking(response["choices"][0]["text"].strip())

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
