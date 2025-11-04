"""
Utility Functions for DocuFlow Service Colony

This module provides common utility functions used across the service colony.
"""

import os
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration data

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise


def get_env_var(var_name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get an environment variable with optional default and validation.

    Args:
        var_name: Name of the environment variable
        default: Default value if variable is not set
        required: If True, raises ValueError when variable is not set and no default provided

    Returns:
        The environment variable value or default

    Raises:
        ValueError: If variable is required but not set and no default provided
    """
    value = os.getenv(var_name, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{var_name}' is not set")

    if value:
        logger.debug(f"Environment variable '{var_name}' = '{value}'")

    return value


def ensure_directory(directory_path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory

    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {directory_path}")
    return path


def validate_file_exists(file_path: str) -> bool:
    """
    Check if a file exists and is accessible.

    Args:
        file_path: Path to the file

    Returns:
        True if file exists and is accessible, False otherwise
    """
    exists = os.path.exists(file_path) and os.path.isfile(file_path)

    if not exists:
        logger.warning(f"File not found or not accessible: {file_path}")

    return exists


def safe_json_loads(json_string: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Safely parse a JSON string with a default fallback.

    Args:
        json_string: JSON string to parse
        default: Default value to return if parsing fails (default: empty dict)

    Returns:
        Parsed JSON dictionary or default value
    """
    if default is None:
        default = {}

    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}. Returning default value.")
        return default


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def retry_operation(operation, max_attempts: int = 3, delay: float = 1.0):
    """
    Retry an operation with exponential backoff.

    Args:
        operation: Callable to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds

    Returns:
        Result of the operation

    Raises:
        Exception: Last exception if all retries fail
    """
    import time

    last_exception = None

    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{max_attempts}). "
                    f"Retrying in {wait_time:.1f}s... Error: {e}"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Operation failed after {max_attempts} attempts")

    raise last_exception
