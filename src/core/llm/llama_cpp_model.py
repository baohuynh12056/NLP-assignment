# core/llm/llama_cpp_model.py

from core.llm.base import BaseLLM
from llama_cpp import Llama
from typing import Any, Dict, Iterator, List
from utils.logger import get_logger

logger = get_logger(__name__)


class LlamaCPPModel(BaseLLM):
    """Implementation of BaseLLM specifically for the llama.cpp engine."""

    def __init__(self, config: Dict[str, Any]):
        logger.info(f"Loading model from: {config.get('model_path')}")

        # Fetch chat_format directly from the injected config, NOT from global prompts
        chat_format = config.get("chat_format", "auto")
        logger.info(f"Using chat format: {chat_format}")

        # Initialize the underlying Llama engine
        self.llm = Llama(
            model_path=config["model_path"],
            n_ctx=config.get("context_window", 4096),
            n_gpu_layers=config.get("gpu_layers", -1),
            chat_format=chat_format,
            verbose=False,
        )

        # Store default generation parameters
        self.default_params = {
            "temperature": config.get("temperature", 0.3),
            "max_tokens": config.get("max_tokens", 512),
        }
        logger.info("Model loaded successfully.")

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        params = {**self.default_params, **kwargs}

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        return response["choices"][0]["message"]["content"].strip()

    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Iterator[str]:
        params = {**self.default_params, **kwargs}
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            stream=True,
        )
        for event in response:
            delta = event["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
