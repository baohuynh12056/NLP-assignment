# src/core/llm/llama_cpp_model.py
from llama_cpp import Llama
from typing import List, Dict, Any

from core.llm.base import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)

class LlamaCPPModel(BaseLLM):
    """Implementation of BaseLLM specifically for the local llama.cpp engine."""

    def __init__(self, config: Dict[str, Any]):
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for LlamaCPPModel configuration.")

        logger.info(f"Loading local GGUF model from: {model_path}")
        chat_format = config.get("chat_format", "auto")
        logger.info(f"Using chat format: {chat_format}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=config.get("context_window", 4096),
            n_gpu_layers=config.get("gpu_layers", -1),
            chat_format=chat_format,
            verbose=False,
        )

        self.default_params = {
            "temperature": config.get("temperature", 0.3),
            "max_tokens": config.get("max_tokens", 512),
        }
        logger.info("Local LLM model loaded successfully.")

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        params = {**self.default_params, **kwargs}

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            stop=["<|im_end|>"]
        )
        return response["choices"][0]["message"]["content"].strip()