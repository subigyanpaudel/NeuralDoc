import logging
import chainlit as cl

from pathlib import Path

from utils.helpers import setup_logging
from rag.document_loader import DocumentLoader
from rag.chunking import DocumentChunker
from rag.embeddings import get_embedding_function
from rag.vector_store import VectorStoreManager
from rag.retriever import DocumentRetriever
from chat.chat_engine import ChatEngine
from chat.memory import ChatMemory
from utils.file_parser import validate_file, save_uploaded_file, get_file_type_emoji

# Initialize Logging 
setup_logging()
logger = logging.getLogger(__name__)

# Global Components 
# These are initialized once and shared across sessions
document_loader = DocumentLoader()
chunker = DocumentChunker()
chat_memory = ChatMemory()

# Lazy-initialized (requires embedding model download on first use)
_embedding_fn = None
_vector_store_manager = None


def get_vector_store() -> VectorStoreManager:
    """Get or initialize the global vector store manager."""
    global _embedding_fn, _vector_store_manager
    if _vector_store_manager is None:
        _embedding_fn = get_embedding_function()
        _vector_store_manager = VectorStoreManager(_embedding_fn)
    return _vector_store_manager


# Chainlit Events 


@cl.on_chat_start
async def on_chat_start():
    """Handle new chat session initialization."""
    session_id = cl.user_session.get("id")

    # Initialize session state
    cl.user_session.set("uploaded_files", [])

    # Create chat session in memory
    chat_memory.create_session(session_id)

    logger.info("New chat session started: %s", session_id)

    # Welcome message
    welcome_text = (
        "# Welcome to NeuralDoc!\n\n"
        "I'm your AI document assistant. Upload documents and ask me anything about them.\n\n"
        "### Supported Formats\n"
        "PDF, DOCX, TXT, PPTX, XLSX, CSV, Markdown\n\n"
        "### Getting Started\n"
        "1. **Upload** one or more documents using the attach button below\n"
        "2. **Wait** for indexing to complete\n"
        "3. **Ask** any question about your documents!\n\n"
        "---\n"
        "*Powered by Google Gemini & RAG*"
    )
    await cl.Message(content=welcome_text).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages and file uploads."""
    session_id = cl.user_session.get("id")

    # Handle file uploads 
    if message.elements:
        await _handle_file_uploads(message.elements, session_id)

        # If the message also has text, process it as a query
        if message.content.strip():
            await _handle_query(message.content.strip(), session_id)
        return

    # Handle text queries 
    if message.content.strip():
        await _handle_query(message.content.strip(), session_id)
    else:
        await cl.Message(
            content="Please upload a document or ask a question! (No message content)"
        ).send()


async def _handle_file_uploads(elements: list, session_id: str) -> None:
    """
    Process uploaded files: validate, save, load, chunk, and embed.

    Args:
        elements: List of Chainlit file elements.
        session_id: Current session ID.
    """
    vector_store = get_vector_store()
    uploaded_files = cl.user_session.get("uploaded_files") or []
    processed_files: list[str] = []
    failed_files: list[str] = []

    # Processing status message
    status_msg = cl.Message(content="Processing uploaded files...")
    await status_msg.send()

    for element in elements:
        if not hasattr(element, "name") or not hasattr(element, "path"):
            continue

        filename = element.name
        file_path = element.path

        # Validate
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else None
        is_valid, error = validate_file(filename, file_size)
        if not is_valid:
            failed_files.append(f"{filename}: {error}")
            continue

        try:
            # Read file content and save to storage
            file_content = Path(file_path).read_bytes()
            saved_path = await save_uploaded_file(file_content, filename, session_id)

            # Load document
            documents = await document_loader.load(
                file_path=saved_path,
                session_id=session_id,
                original_filename=filename,
            )

            # Chunk document
            chunks = chunker.chunk_documents(documents)

            # Embed and store
            await vector_store.add_documents(chunks)

            ext = Path(filename).suffix.lower()
            label = get_file_type_emoji(ext)
            processed_files.append(f"{label} **{filename}** — {len(chunks)} chunks")
            uploaded_files.append(filename)

            logger.info(
                "Processed '%s': %d pages → %d chunks",
                filename,
                len(documents),
                len(chunks),
            )

        except Exception as e:
            logger.error("Failed to process '%s': %s", filename, e)
            failed_files.append(f"{filename}: {str(e)}")

    # Update session state
    cl.user_session.set("uploaded_files", uploaded_files)
    chat_memory.update_session_documents(session_id, [f for f in uploaded_files])

    # Build result message
    result_parts: list[str] = []

    if processed_files:
        result_parts.append("### Documents Indexed Successfully\n")
        result_parts.extend(processed_files)
        total_chunks = vector_store.get_document_count()
        result_parts.append(f"\n\n**Total chunks in knowledge base:** {total_chunks}")
        result_parts.append("\n\n(No message content) You can now ask questions about your documents!")

    if failed_files:
        result_parts.append("\n\n### Failed to Process\n")
        result_parts.extend(failed_files)

    # Update the status message
    status_msg.content = "\n".join(result_parts) if result_parts else "No files were processed."
    await status_msg.update()


async def _handle_query(query: str, session_id: str) -> None:
    """
    Process a user query using the RAG pipeline.

    Args:
        query: User's question text.
        session_id: Current session ID.
    """
    vector_store = get_vector_store()

    # Check if documents are available
    if not vector_store.has_documents():
        await cl.Message(
            content=(
                "**No documents indexed yet.**\n\n"
                "Please upload one or more documents first using the attach button, "
                "then I can answer your questions!"
            )
        ).send()
        return

    # Initialize chat engine
    retriever = DocumentRetriever(vector_store)
    engine = ChatEngine(retriever=retriever, memory=chat_memory)

    # Stream the response
    response_msg = cl.Message(content="")
    await response_msg.send()

    sources = []
    try:
        async for chunk in engine.generate_stream(query=query, session_id=session_id):
            if chunk.get("token"):
                await response_msg.stream_token(chunk["token"])

            if chunk.get("done") and chunk.get("sources"):
                sources = chunk["sources"]

    except Exception as e:
        logger.error("Error generating response: %s", e)
        response_msg.content = (
            "An error occurred while generating the response.\n\n"
            f"Error: {str(e)}\n\n"
            "Please check your API key and try again."
        )
        await response_msg.update()
        return

    # Add source citations
    if sources:
        citation_text = "\n\n---\n**Sources:**\n"
        for src in sources:
            label = get_file_type_emoji(src.get("file_type", ""))
            citation_text += f"- {label} {src['source']}\n"
        response_msg.content += citation_text
        await response_msg.update()


# Chat Resume (for persistent sessions) 


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Handle chat session resume."""
    session_id = cl.user_session.get("id")
    logger.info("Chat session resumed: %s", session_id)

    # Restore session state
    cl.user_session.set("uploaded_files", [])

    await cl.Message(
        content=(
            "Welcome back!\n\n"
            "Your previous documents may still be available in the knowledge base. "
            "Feel free to upload new documents or ask questions!"
        )
    ).send()
