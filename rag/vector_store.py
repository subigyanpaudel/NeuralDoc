"""
NeuralDoc — Vector Store Module
Manages ChromaDB and FAISS vector stores for persistent document storage and retrieval.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import VECTOR_DB_PATH

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages dual vector stores: ChromaDB (persistent, metadata-rich)
    and FAISS (high-performance similarity search).
    """

    def __init__(self, embedding_function: Embeddings) -> None:
        """
        Initialize the vector store manager.

        Args:
            embedding_function: LangChain-compatible embedding function.
        """
        self.embedding_function = embedding_function
        self._chroma_store = None
        self._faiss_store = None
        self._chroma_path = str(VECTOR_DB_PATH / "chroma")
        self._faiss_path = str(VECTOR_DB_PATH / "faiss")

        # Ensure directories exist
        Path(self._chroma_path).mkdir(parents=True, exist_ok=True)
        Path(self._faiss_path).mkdir(parents=True, exist_ok=True)

    # ── ChromaDB ─────────────────────────────────────────────────────────

    def _get_chroma(self, collection_name: str = "neuraldoc"):
        """Get or create ChromaDB store."""
        if self._chroma_store is None:
            from langchain_community.vectorstores import Chroma

            self._chroma_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self._chroma_path,
            )
            # Ensure the collection exists and is flushed
            logger.info("ChromaDB initialized at %s", self._chroma_path)
        return self._chroma_store

    # ── FAISS ────────────────────────────────────────────────────────────

    def _get_faiss(self):
        """Get or create FAISS store, loading from disk if available."""
        if self._faiss_store is None:
            from langchain_community.vectorstores import FAISS

            index_file = Path(self._faiss_path) / "index.faiss"
            if index_file.exists():
                try:
                    self._faiss_store = FAISS.load_local(
                        self._faiss_path,
                        self.embedding_function,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info("FAISS index loaded from %s", self._faiss_path)
                except Exception as e:
                    logger.warning("Failed to load FAISS index: %s. Creating new.", e)
                    self._faiss_store = None
        return self._faiss_store

    # ── Public API ───────────────────────────────────────────────────────

    async def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to both ChromaDB and FAISS stores.

        Args:
            documents: List of LangChain Document objects to store.
        """
        if not documents:
            logger.warning("No documents to add.")
            return

        # Add to ChromaDB
        chroma = self._get_chroma()
        chroma.add_documents(documents)
        logger.info("Added %d documents to ChromaDB.", len(documents))

        # Add to / create FAISS
        from langchain_community.vectorstores import FAISS

        faiss = self._get_faiss()
        if faiss is None:
            self._faiss_store = FAISS.from_documents(documents, self.embedding_function)
            logger.info("Created new FAISS index with %d documents.", len(documents))
        else:
            faiss.add_documents(documents)
            logger.info("Added %d documents to existing FAISS index.", len(documents))

        # Persist FAISS to disk
        self._faiss_store.save_local(self._faiss_path)
        logger.info("FAISS index saved to %s", self._faiss_path)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
        use_faiss: bool = True,
    ) -> list[Document]:
        """
        Perform similarity search across the vector store.

        Args:
            query: User's search query.
            k: Number of top results to return.
            filter_dict: Optional metadata filter (ChromaDB only).
            use_faiss: If True, prioritize FAISS for speed; fall back to ChromaDB.

        Returns:
            List of relevant Document objects.
        """
        # Try FAISS first for speed
        if use_faiss:
            faiss = self._get_faiss()
            if faiss is not None:
                try:
                    results = faiss.similarity_search(query, k=k)
                    logger.info("FAISS returned %d results.", len(results))
                    return results
                except Exception as e:
                    logger.warning("FAISS search failed: %s. Falling back to ChromaDB.", e)

        # Fall back to ChromaDB (supports metadata filtering)
        chroma = self._get_chroma()
        try:
            if filter_dict:
                results = chroma.similarity_search(query, k=k, filter=filter_dict)
            else:
                results = chroma.similarity_search(query, k=k)
            logger.info("ChromaDB returned %d results.", len(results))
            return results
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

    def has_documents(self) -> bool:
        """Check if any documents have been indexed."""
        try:
            chroma = self._get_chroma()
            count = chroma._collection.count()
            logger.info("Vector store document count: %d", count)
            return count > 0
        except Exception as e:
            logger.error("Error checking document count: %s", e)
            return False

    def get_document_count(self) -> int:
        """Return the number of indexed document chunks."""
        try:
            chroma = self._get_chroma()
            return chroma._collection.count()
        except Exception:
            return 0

    def delete_session_documents(self, session_id: str) -> None:
        """Delete all documents associated with a session."""
        try:
            chroma = self._get_chroma()
            chroma._collection.delete(where={"session_id": session_id})
            logger.info("Deleted documents for session %s from ChromaDB.", session_id)
        except Exception as e:
            logger.warning("Failed to delete session documents: %s", e)

    def clear_all_knowledge(self) -> None:
        """Completely wipe all vector database collections and FAISS indices."""
        try:
            # 1. Clear ChromaDB logically first
            if self._chroma_store is not None:
                try:
                    self._chroma_store.delete_collection()
                except Exception as e:
                    logger.warning("Could not delete Chroma collection: %s", e)
                self._chroma_store = None
            
            # 2. Clear FAISS store
            self._faiss_store = None
            
            # 3. Attempt physical deletion of indices
            # On Windows, files might be locked. We try our best but don't crash if locked.
            def safe_rmtree(path: str):
                p = Path(path)
                if p.exists():
                    try:
                        shutil.rmtree(path)
                        p.mkdir(parents=True, exist_ok=True)
                        logger.info("Successfully deleted directory: %s", path)
                    except PermissionError:
                        logger.warning("Directory %s is locked. Content cleared but folder remains.", path)
                    except Exception as e:
                        logger.error("Error deleting %s: %s", path, e)

            safe_rmtree(self._chroma_path)
            safe_rmtree(self._faiss_path)
            
            logger.info("Knowledge base clearing attempt completed.")
        except Exception as e:
            logger.error("Error in clear_all_knowledge: %s", e)
            # We don't re-raise here to allow the UI to continue and clear other states
