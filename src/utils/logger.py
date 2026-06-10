# utils/logger.py
import logging
import sys


def get_logger(module_name: str) -> logging.Logger:
    """
    Creates and configures a standard logger for the application.
    Perfectly optimized for Docker container logs.
    """
    logger = logging.getLogger(module_name)

    # Prevent adding multiple handlers if the logger already exists
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Standardize the log format: [Time] - [Level] - [Module] - Message
        formatter = logging.Formatter(
            fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Use stdout so Docker logs command can capture it immediately
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Prevent propagation to the root logger to avoid duplicate logs
        logger.propagate = False

    return logger
