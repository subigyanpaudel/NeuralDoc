"""
NeuralDoc — Helper Utilities
Logging setup, formatting, and general helper functions.
"""

import logging
import sys
from datetime import datetime
from typing import Optional

from config import LOG_LEVEL


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to config.
    """
    log_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logging.info("Logging initialized at level: %s", log_level)


def format_timestamp(iso_string: str) -> str:
    """
    Format an ISO timestamp into a human-readable string.

    Args:
        iso_string: ISO 8601 formatted timestamp.

    Returns:
        Human-readable timestamp string.
    """
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%b %d, %Y at %I:%M %p")
    except (ValueError, TypeError):
        return iso_string


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Args:
        text: Input text.
        max_length: Maximum length before truncation.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted size string (e.g., '1.5 MB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count for a text string.

    Uses the approximation of ~4 characters per token.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    return len(text) // 4
