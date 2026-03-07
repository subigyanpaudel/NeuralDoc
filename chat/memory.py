"""
NeuralDoc — Chat Memory Module
Persistent chat history storage using SQLite.
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from config import CHAT_DB_PATH

logger = logging.getLogger(__name__)


class ChatMemory:
    """
    Manages persistent chat history using SQLite.

    Stores chat sessions and individual messages for continuity
    across conversations.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize chat memory with SQLite backend.

        Args:
            db_path: Path to the SQLite database file. Defaults to config value.
        """
        self.db_path = db_path or str(CHAT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("ChatMemory initialized at %s", self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT 'New Chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    documents_used TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id);
            """)

    def create_session(self, session_id: str, title: str = "New Chat") -> None:
        """
        Create a new chat session.

        Args:
            session_id: Unique session identifier.
            title: Display title for the session.
        """
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO chat_sessions (id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (session_id, title, now, now),
            )
        logger.info("Created chat session: %s", session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to a chat session.

        Args:
            session_id: Session to add the message to.
            role: Message role ('user' or 'assistant').
            content: Message content.
        """
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO messages (session_id, role, content, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    def get_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get chat history for a session.

        Args:
            session_id: Session ID to retrieve history for.
            limit: Maximum number of messages to return.

        Returns:
            List of message dictionaries with role, content, and timestamp.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT role, content, timestamp FROM messages
                   WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()

        # Return in chronological order
        return [dict(row) for row in reversed(rows)]

    def update_session_documents(
        self,
        session_id: str,
        document_names: list[str],
    ) -> None:
        """
        Update the list of documents used in a session.

        Args:
            session_id: Session to update.
            document_names: List of document filenames.
        """
        with self._get_connection() as conn:
            # Get existing documents
            row = conn.execute(
                "SELECT documents_used FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

            existing = json.loads(row["documents_used"]) if row else []
            updated = list(set(existing + document_names))

            conn.execute(
                "UPDATE chat_sessions SET documents_used = ? WHERE id = ?",
                (json.dumps(updated), session_id),
            )

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """
        List all chat sessions, most recent first.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session dictionaries.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT id, title, created_at, updated_at, documents_used
                   FROM chat_sessions
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        sessions = []
        for row in rows:
            session = dict(row)
            session["documents_used"] = json.loads(session["documents_used"])
            sessions.append(session)

        return sessions

    def get_session_title(self, session_id: str) -> str:
        """Get the title of a chat session."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT title FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return row["title"] if row else "New Chat"

    def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a chat session."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )
