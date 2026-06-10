# src/core/llm/manager.py
from core.llm.factory import LLMFactory
from core.llm.base import BaseLLM
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

class LLMManager:
    """Singleton that manages and provides LLM instances across the system to save VRAM."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._initialize_models()
        return cls._instance

    def _initialize_models(self):
        models_cfg = GLOBAL_CONFIG.get("models", {})

        logger.info("Initializing Parser LLM instance...")
        self.parser_llm: BaseLLM = LLMFactory.create_llm(models_cfg.get("parser", {}))

        logger.info("Initializing Generator LLM instance...")
        self.generator_llm: BaseLLM = LLMFactory.create_llm(models_cfg.get("generator", {}))