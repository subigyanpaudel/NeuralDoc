"""
NeuralDoc — Document Chunking Module
Splits documents into smaller chunks for embedding and retrieval.
"""

import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits documents into chunks using RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )
        logger.info(
            "DocumentChunker initialized (chunk_size=%d, overlap=%d)",
            chunk_size,
            chunk_overlap,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of documents into smaller chunks.

        Each chunk inherits the metadata of its parent document,
        with an additional `chunk_index` field.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked Document objects.
        """
        if not documents:
            logger.warning("No documents provided for chunking.")
            return []

        chunks = self.splitter.split_documents(documents)

        # Add chunk index metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx

        logger.info(
            "Chunked %d documents into %d chunks.", len(documents), len(chunks)
        )
        return chunks
