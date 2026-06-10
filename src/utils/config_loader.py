import yaml
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Automatically loads and merges all YAML files in the config directory."""

    def __init__(self, config_dir="configs"):
        self.config_dir = Path(__file__).resolve().parents[2] / config_dir
        self.config = {}
        self._load_all_configs()

    def get(self, key, default=None):
        """Retrieve a config value by key."""
        return self.config.get(key, default)

    def _load_all_configs(self):
        if not self.config_dir.exists():
            logger.error(f"Config directory not found: {self.config_dir}")
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        for yaml_file in self.config_dir.glob("*.yaml"):
            module_name = yaml_file.stem
            with open(yaml_file, "r", encoding="utf-8") as f:
                self.config[module_name] = yaml.safe_load(f) or {}
            logger.info(f"Loaded config: {yaml_file.name}")


# Global singleton instance for system-wide access
GLOBAL_CONFIG = ConfigLoader().config
