from core.llm.factory import LLMFactory
from core.llm.base import BaseLLM
from utils.config_loader import GLOBAL_CONFIG


class LLMManager:
    """Singleton that manages and provides LLM instances across the system."""

    _instance = None

    def __new__(cls):
        # Ensure only one instance exists
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._initialize_models()
        return cls._instance

    def _initialize_models(self):
        models_cfg = GLOBAL_CONFIG.get("models", {})

        # Initialize the lightweight parser model
        self.parser_llm: BaseLLM = LLMFactory.create_llm(models_cfg.get("parser", {}))

        # Initialize the primary generator model
        self.generator_llm: BaseLLM = LLMFactory.create_llm(
            models_cfg.get("generator", {})
        )
