"""Logging configuration using loguru."""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any


def setup_logging(config: Optional[Dict[str, Any]] = None):
    """
    Set up logging configuration.

    Args:
        config: Logging configuration dictionary
    """
    # Remove default handler
    logger.remove()

    # Default configuration
    default_config = {
        "level": "INFO",
        "file": "./logs/tome_raider.log",
        "max_size_mb": 100,
        "backup_count": 5,
        "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    }

    if config:
        default_config.update(config)

    log_format = default_config["format"]
    log_level = default_config["level"]

    # Console handler with colors
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler with rotation
    log_file = Path(default_config["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    max_size = f"{default_config['max_size_mb']} MB"
    rotation = max_size

    logger.add(
        log_file,
        format=log_format,
        level=log_level,
        rotation=rotation,
        retention=default_config["backup_count"],
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Thread-safe logging
    )

    logger.info(f"Logging initialized at level: {log_level}")


def get_logger(name: str):
    """
    Get a logger instance with a specific name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)
