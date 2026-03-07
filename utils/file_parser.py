"""
NeuralDoc — File Parser Utility
Handles file validation, temporary file management, and type detection.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB, DOCUMENT_STORAGE_PATH

logger = logging.getLogger(__name__)


def validate_file(
    filename: str,
    file_size: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Validate an uploaded file.

    Args:
        filename: Name of the uploaded file.
        file_size: Size of the file in bytes (optional).

    Returns:
        Tuple of (is_valid, error_message).
    """
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return False, (
            f"❌ Unsupported file type: **{ext}**\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if file_size is not None:
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            return False, (
                f"❌ File too large: **{file_size / (1024*1024):.1f} MB**\n"
                f"Maximum allowed: {MAX_FILE_SIZE_MB} MB"
            )

        if file_size == 0:
            return False, "❌ File is empty."

    return True, ""


async def save_uploaded_file(
    file_content: bytes,
    filename: str,
    session_id: str,
) -> str:
    """
    Save an uploaded file to the document storage directory.

    Creates a session-specific subdirectory to organize uploads.

    Args:
        file_content: Raw file bytes.
        filename: Original filename.
        session_id: Current session ID.

    Returns:
        Path to the saved file.
    """
    # Create session directory
    session_dir = DOCUMENT_STORAGE_PATH / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = session_dir / filename
    file_path.write_bytes(file_content)

    logger.info("Saved uploaded file: %s (%d bytes)", file_path, len(file_content))
    return str(file_path)


def get_file_type_emoji(extension: str) -> str:
    """Return an emoji for a file type."""
    emoji_map = {
        ".pdf": "📄",
        ".docx": "📝",
        ".txt": "📃",
        ".pptx": "📊",
        ".xlsx": "📈",
        ".xls": "📈",
        ".csv": "📋",
        ".md": "📑",
    }
    return emoji_map.get(extension.lower(), "📁")


def cleanup_session_files(session_id: str) -> None:
    """Remove all files for a given session."""
    session_dir = DOCUMENT_STORAGE_PATH / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
        logger.info("Cleaned up files for session: %s", session_id)
