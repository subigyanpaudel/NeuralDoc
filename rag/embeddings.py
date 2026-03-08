"""
NeuralDoc — Embeddings Module
Provides embedding functions with Gemini primary and sentence-transformers fallback.
"""

import logging
from langchain_core.embeddings import Embeddings

from config import GOOGLE_API_KEY, GEMINI_EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL

import logging
import streamlit as st

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="Running...")
def get_embedding_function() -> Embeddings:
    """
    Return the best available embedding function.

    Tries Google Generative AI embeddings first (requires GOOGLE_API_KEY).
    Falls back to HuggingFace sentence-transformers if unavailable.

    Returns:
        A LangChain-compatible Embeddings instance.
    """
    if GOOGLE_API_KEY:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(
                model=GEMINI_EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY,
            )
            # Test the key actually works before committing to Gemini
            embeddings.embed_query("test")
            logger.info("Using Google Generative AI embeddings (%s)", GEMINI_EMBEDDING_MODEL)
            return embeddings
        except Exception as e:
            logger.warning("Gemini embeddings failed (%s), falling back.", e)

    # Fallback to sentence-transformers
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=FALLBACK_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        logger.info("Using HuggingFace embeddings (%s)", FALLBACK_EMBEDDING_MODEL)
        return embeddings
    except Exception as e:
        logger.error("Failed to initialize any embedding model: %s", e)
        raise RuntimeError(
            "Could not initialize an embedding model. "
            "Ensure either GOOGLE_API_KEY is set or sentence-transformers is installed."
        ) from e
