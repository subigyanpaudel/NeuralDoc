import logging
import asyncio
import streamlit as st
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

# Setup page
st.set_page_config(page_title="NeuralDoc", page_icon="📄")

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Cache global components
@st.cache_resource
def get_global_components():
    return {
        "document_loader": DocumentLoader(),
        "chunker": DocumentChunker(),
        "chat_memory": ChatMemory(),
    }

@st.cache_resource
def get_vector_store():
    embedding_fn = get_embedding_function()
    return VectorStoreManager(embedding_fn)

components = get_global_components()
document_loader = components["document_loader"]
chunker = components["chunker"]
chat_memory = components["chat_memory"]

vector_store = get_vector_store()

# Session State Initialization
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
    chat_memory.create_session(st.session_state.session_id)

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to NeuralDoc!\n\nI'm your AI document assistant. Upload documents in the sidebar and ask me anything about them."}
    ]

# Layout: Sidebar
with st.sidebar:
    st.header("📄 Documents")
    uploaded_files = st.file_uploader("Upload documents here", accept_multiple_files=True, type=["pdf", "docx", "txt", "pptx", "xlsx", "csv", "md"])

    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing files..."):
            session_id = st.session_state.session_id
            
            async def process_files():
                p_files = []
                f_files = []
                for file in uploaded_files:
                    filename = file.name
                    file_content = file.read()
                    file_size = len(file_content)
                    
                    is_valid, error = validate_file(filename, file_size)
                    if not is_valid:
                        f_files.append(f"{filename}: {error}")
                        continue
                    
                    try:
                        saved_path = await save_uploaded_file(file_content, filename, session_id)
                        documents = await document_loader.load(
                            file_path=saved_path, session_id=session_id, original_filename=filename
                        )
                        chunks = chunker.chunk_documents(documents)
                        await vector_store.add_documents(chunks)
                        
                        ext = Path(filename).suffix.lower()
                        label = get_file_type_emoji(ext)
                        p_files.append(f"{label} **{filename}** — {len(chunks)} chunks")
                        
                        if filename not in st.session_state.uploaded_files:
                            st.session_state.uploaded_files.append(filename)
                            
                        logger.info("Processed '%s': %d pages → %d chunks", filename, len(documents), len(chunks))
                    except Exception as e:
                        logger.error("Failed to process '%s': %s", filename, e)
                        f_files.append(f"{filename}: {str(e)}")
                return p_files, f_files
            
            p_files, f_files = asyncio.run(process_files())
                
            if p_files:
                st.success(f"Successfully processed {len(p_files)} files!")
                for p in p_files:
                    st.write(p)
                chat_memory.update_session_documents(session_id, st.session_state.uploaded_files)
                
            if f_files:
                st.error("Failed to process:")
                for f in f_files:
                    st.write(f)

# Layout: Main Chat
st.title("NeuralDoc")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handling User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        if not vector_store.has_documents():
            response = "**No documents indexed yet.**\n\nPlease upload one or more documents in the sidebar first."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            response_placeholder = st.empty()
            full_response = ""
            
            retriever = DocumentRetriever(vector_store)
            engine = ChatEngine(retriever=retriever, memory=chat_memory)
            
            async def generate_response():
                res = ""
                sources = []
                try:
                    async for chunk in engine.generate_stream(query=prompt, session_id=st.session_state.session_id):
                        if chunk.get("token"):
                            res += chunk["token"]
                            response_placeholder.markdown(res + "▌")
                        
                        if chunk.get("done") and chunk.get("sources"):
                            sources = chunk["sources"]
                except Exception as e:
                    logger.error("Error generating response: %s", e)
                    res = f"An error occurred: {str(e)}"
                    response_placeholder.markdown(res)
                
                return res, sources
                
            full_response, sources = asyncio.run(generate_response())
            
            if sources:
                full_response += "\n\n---\n**Sources:**\n"
                for src in sources:
                    label = get_file_type_emoji(src.get("file_type", ""))
                    full_response += f"- {label} {src['source']}\n"
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
