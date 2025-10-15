"""Configuration management with YAML profiles and environment overrides."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy
from loguru import logger


class ConfigManager:
    """Manages application configuration with profiles and inheritance."""

    def __init__(self, config_path: Optional[str] = None, profile: str = "default"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to user configuration file (optional)
            profile: Configuration profile to load (default, dev, prod)
        """
        self.profile = profile
        self.config_dir = Path(__file__).parent.parent.parent / "configs"
        self.config = self._load_base_config()

        # Load profile configuration
        profile_config = self._load_profile_config(profile)
        self.config = self._merge_configs(self.config, profile_config)

        # Load user configuration if provided
        if config_path:
            user_config = self._load_user_config(config_path)
            self.config = self._merge_configs(self.config, user_config)

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Validate configuration
        self._validate_config()

        logger.info(f"Configuration loaded with profile: {profile}")

    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration with sensible defaults."""
        return {
            "name": "Dataset Builder",
            "version": "0.1.0",

            "storage": {
                "base_path": "./datasets",
                "index_path": "./.index",
                "cache_path": "./.cache",
            },

            "llm": {
                "server_command": "llama-server",
                "base_url": "http://localhost:8080",
                "default_context_size": 4096,
                "health_check_interval": 5,
                "request_timeout": 300,
            },

            "validation": {
                "strict_mode": True,
                "required_fields": ["instruction", "response"],
                "min_instruction_length": 10,
                "max_instruction_length": 5000,
                "min_response_length": 20,
                "max_response_length": 10000,
            },

            "quality": {
                "scoring_model": None,
                "auto_recommend_thresholds": True,
                "duplicate_threshold": 0.9,
                "near_duplicate_threshold": 0.85,
            },

            "generation": {
                "default_temperature": 0.7,
                "default_top_p": 0.9,
                "default_max_tokens": 1024,
                "batch_size": 8,
                "error_retry_attempts": 3,
                "error_retry_delay": 5,
            },

            "review": {
                "batch_size": 50,
                "default_strategy": "prioritized",
                "priority_weights": {
                    "flagged": 1000,
                    "low_quality": 500,
                    "near_duplicate": 300,
                    "pending": 100,
                },
            },

            "export": {
                "default_format": "jsonl",
                "validate_before_export": True,
                "never_block": True,
            },

            "logging": {
                "level": "INFO",
                "file": "./logs/tome_raider.log",
                "max_size_mb": 100,
                "backup_count": 5,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            },

            "web_scraping": {
                "rate_limit": 1.0,
                "max_retries": 3,
                "timeout": 30,
                "respect_robots_txt": True,
                "user_agent": "GRPO-tome-raider/0.1.0",
            },
        }

    def _load_profile_config(self, profile: str) -> Dict[str, Any]:
        """
        Load profile configuration with inheritance support.

        Args:
            profile: Profile name (default, dev, prod)

        Returns:
            Profile configuration dictionary
        """
        profile_path = self.config_dir / f"{profile}.yaml"

        if not profile_path.exists():
            if profile != "default":
                logger.warning(f"Profile '{profile}' not found, using defaults")
            return {}

        with open(profile_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        # Handle inheritance via 'extends' key
        if "extends" in config:
            parent_profile = config.pop("extends").replace(".yaml", "")
            parent_config = self._load_profile_config(parent_profile)
            config = self._merge_configs(parent_config, config)

        return config

    def _load_user_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load user-provided configuration file.

        Args:
            config_path: Path to user configuration file

        Returns:
            User configuration dictionary
        """
        path = Path(config_path)

        if not path.exists():
            logger.warning(f"User config file not found: {config_path}")
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded user configuration from: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load user config: {e}")
            return {}

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Environment variable format: DB_SECTION_KEY -> config[section][key]
        # Example: DB_STORAGE_BASE_PATH -> config[storage][base_path]

        env_prefix = "DB_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue

            # Remove prefix and split into parts
            config_path = env_key[len(env_prefix):].lower().split("_")

            # Navigate to the configuration location
            current = self.config
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value (try to parse as int/float/bool)
            final_key = config_path[-1]
            current[final_key] = self._parse_env_value(env_value)

            logger.debug(f"Applied environment override: {env_key} = {env_value}")

    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.

        Args:
            value: Environment variable value

        Returns:
            Parsed value (str, int, float, bool)
        """
        # Try boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _validate_config(self):
        """Validate configuration for required fields and reasonable values."""
        # Validate required fields
        required_sections = ["storage", "llm", "validation", "generation", "export", "logging"]
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing configuration section: {section}")

        # Validate storage paths
        storage = self.config.get("storage", {})
        for path_key in ["base_path", "index_path", "cache_path"]:
            if path_key in storage:
                path = Path(storage[path_key])
                path.mkdir(parents=True, exist_ok=True)

        # Validate numeric ranges
        validation = self.config.get("validation", {})
        if validation.get("min_instruction_length", 0) >= validation.get("max_instruction_length", 1):
            logger.warning("Invalid instruction length configuration")

        if validation.get("min_response_length", 0) >= validation.get("max_response_length", 1):
            logger.warning("Invalid response length configuration")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Configuration key path (e.g., "storage.base_path")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        current = self.config

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]

        return current

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Configuration key path (e.g., "storage.base_path")
            value: Value to set
        """
        keys = key_path.split(".")
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def save(self, output_path: str):
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {output_path}")

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to configuration."""
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        """Dict-like membership test."""
        return key in self.config

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return deepcopy(self.config)
