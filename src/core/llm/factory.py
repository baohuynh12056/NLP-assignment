# src/core/llm/factory.py
from typing import Dict, Any
from core.llm.base import BaseLLM
from core.llm.llama_cpp_model import LlamaCPPModel

class LLMFactory:
    """Creates LLM instances dynamically based on configuration."""

    @staticmethod
    def create_llm(config: Dict[str, Any]) -> BaseLLM:
        llm_type = config.get("type", "llama_cpp").lower()

        if llm_type == "llama_cpp":
            return LlamaCPPModel(config)
        # Add support for other providers (e.g., openai) here in the future
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")