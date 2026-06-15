from core.llm.base import BaseLLM
from core.llm.llama_cpp_model import LlamaCPPModel
from typing import Dict, Any


class LLMFactory:
    """Creates LLM instances based on configuration (Factory Pattern)."""

    @staticmethod
    def create_llm(config: Dict[str, Any]) -> BaseLLM:
        llm_type = config.get("type", "llama_cpp")

        if llm_type == "llama_cpp":
            return LlamaCPPModel(config)
        elif llm_type == "openai":
            raise NotImplementedError("OpenAI LLM backend is not implemented.")
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
