"""
NeuralDoc — Retriever Module
High-level retrieval interface with re-ranking support.
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from config import RETRIEVAL_TOP_K
from rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Retrieves relevant document chunks from the vector store
    and optionally re-ranks results for better quality.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store_manager: Initialized VectorStoreManager instance.
            top_k: Number of top results to retrieve.
        """
        self.vector_store = vector_store_manager
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: The user's question.
            session_id: Optional session ID to filter documents.
            top_k: Override the default number of results.

        Returns:
            List of relevant Document objects, ordered by relevance.
        """
        k = top_k or self.top_k
        filter_dict = {"session_id": session_id} if session_id else None

        # Retrieve from vector store
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict,
        )

        if not results:
            logger.info("No relevant documents found for query: '%s'", query[:80])
            return []

        # Re-rank: deduplicate and sort by source diversity
        results = self._rerank(results, query)

        logger.info(
            "Retrieved %d chunks for query: '%s'",
            len(results),
            query[:80],
        )
        return results

    def _rerank(self, documents: list[Document], query: str) -> list[Document]:
        """
        Re-rank retrieved documents for better source diversity.

        Ensures results aren't dominated by a single document
        by interleaving chunks from different sources.

        Args:
            documents: Raw retrieval results.
            query: Original query (for future scoring use).

        Returns:
            Re-ranked list of documents.
        """
        if len(documents) <= 1:
            return documents

        # Group by source document
        source_groups: dict[str, list[Document]] = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            source_groups.setdefault(source, []).append(doc)

        # Interleave: round-robin from each source
        reranked: list[Document] = []
        seen_contents: set[str] = set()
        max_rounds = max(len(group) for group in source_groups.values())

        for round_idx in range(max_rounds):
            for source, group in source_groups.items():
                if round_idx < len(group):
                    doc = group[round_idx]
                    content_hash = doc.page_content[:200]
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        reranked.append(doc)

        return reranked

    def format_context(self, documents: list[Document]) -> str:
        """
        Format retrieved documents into a context string for the prompt.

        Args:
            documents: List of retrieved documents.

        Returns:
            Formatted context string with source attribution.
        """
        if not documents:
            return "No relevant documents found."

        context_parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"[Source {i}: {source}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_source_citations(self, documents: list[Document]) -> list[dict]:
        """
        Extract source citation information from retrieved documents.

        Args:
            documents: List of retrieved documents.

        Returns:
            List of citation dictionaries with source info.
        """
        citations: list[dict] = []
        seen_sources: set[str] = set()

        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                citations.append({
                    "source": source,
                    "file_type": doc.metadata.get("file_type", ""),
                    "chunk_preview": doc.page_content[:150] + "...",
                })

        return citations
