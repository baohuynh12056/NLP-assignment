import yaml
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigLoader:
    """Load all yaml configs from configs directory."""

    def __init__(self, config_dir=None):
        self.config_dir = (
            Path(config_dir)
            if config_dir
            else PROJECT_ROOT / "configs"
        )

        self.config = {}
        self._load_all_configs()

    def _load_all_configs(self):
        if not self.config_dir.exists():
            logger.error(
                f"Config directory not found: {self.config_dir}"
            )
            raise FileNotFoundError(
                f"Config directory not found: {self.config_dir}"
            )

        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    self.config[yaml_file.stem] = (
                        yaml.safe_load(f) or {}
                    )

                logger.info(
                    f"Loaded config: {yaml_file.name}"
                )

            except Exception as e:
                logger.exception(
                    f"Failed to load {yaml_file.name}: {e}"
                )
                raise

    def get(self, key, default=None):
        return self.config.get(key, default)


config_loader = ConfigLoader()
GLOBAL_CONFIG = config_loader.config