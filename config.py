import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import google.generativeai as genai

# Base Paths 
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENT_STORAGE_PATH = Path(os.getenv("DOCUMENT_STORAGE_PATH", str(DATA_DIR / "documents")))
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "vectordb")))
CHAT_DB_PATH = DATA_DIR / "chat_history.db"

# Ensure directories exist
DOCUMENT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Google Gemini 
api_key = os.getenv("GOOGLE_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)
GOOGLE_API_KEY = api_key
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

# Chunking 
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embeddings 
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval 
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# File Upload 
MAX_FILE_SIZE_MB = 100
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".pptx",
    ".xlsx", ".xls", ".csv", ".md",
}

# Prompt Template 
RAG_PROMPT_TEMPLATE = """You are an intelligent assistant that answers questions based ONLY on the provided context from uploaded documents.

Context:
{context}

Question:
{question}

Instructions:
- Answer clearly and concisely based on the context above.
- If the answer spans multiple documents, mention which documents the information comes from.
- If the answer is not in the context, say: "I cannot find the answer in the uploaded documents."
- Use bullet points or structured formatting when appropriate.
"""

# Logging 
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
