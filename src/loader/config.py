"""Configuration management for Loader."""

import json
from pathlib import Path
from typing import Any


def get_config_dir() -> Path:
    """Get the configuration directory, creating it if needed."""
    config_dir = Path.home() / ".config" / "loader"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the path to the config file."""
    return get_config_dir() / "config.json"


def load_config() -> dict[str, Any]:
    """Load configuration from file.

    Returns:
        Configuration dict, empty if file doesn't exist.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return {}

    try:
        return json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dict to save.
    """
    config_path = get_config_path()
    try:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    except OSError:
        pass  # Silently fail if we can't write


def get_last_model() -> str | None:
    """Get the last used model from config.

    Returns:
        Model name, or None if not set.
    """
    config = load_config()
    return config.get("last_model")


def set_last_model(model: str) -> None:
    """Save the last used model to config.

    Args:
        model: Model name to save.
    """
    config = load_config()
    config["last_model"] = model
    save_config(config)


def get_default_model() -> str:
    """Get the default model to use.

    Returns last used model if available, otherwise 'llama3.1:8b'.
    """
    return get_last_model() or "llama3.1:8b"
