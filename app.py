import logging
import asyncio
import uuid
import streamlit as st
from pathlib import Path

# --- Streamlit Cloud SQLite3 Fix ---
# ChromaDB requires a newer version of sqlite3 than what's often available on Streamlit Cloud.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -----------------------------------

from utils.helpers import setup_logging
from rag.document_loader import DocumentLoader
from rag.chunking import DocumentChunker
from rag.embeddings import get_embedding_function
from rag.vector_store import VectorStoreManager
from rag.retriever import DocumentRetriever
from chat.chat_engine import ChatEngine
from chat.memory import ChatMemory
from utils.file_parser import validate_file, save_uploaded_file, get_file_type_emoji, purge_all_documents

# Setup page
st.set_page_config(page_title="NeuralDoc", page_icon="📄", layout="wide")

# Custom CSS with Dynamic Theme Support
st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    
    /* Use Streamlit variables for sidebar to support dark mode */
    section[data-testid="stSidebar"] { 
        background-color: var(--secondary-background-color); 
        border-right: 1px solid var(--border-color); 
    }
    
    .stChatMessage { 
        border-radius: 12px; 
        padding: 10px; 
        margin-bottom: 10px; 
    }
    
    /* User message background that works in both modes */
    [data-testid="stChatMessageUser"] { 
        background-color: var(--secondary-background-color); 
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Cache global components for Cloud stability
@st.cache_resource(show_spinner="Running...")
def get_global_components():
    return {
        "document_loader": DocumentLoader(),
        "chunker": DocumentChunker(),
        "chat_memory": ChatMemory(),
    }

@st.cache_resource(show_spinner="Running...")
def get_vector_store():
    # This also uses cached get_embedding_function() from rag.embeddings
    embedding_fn = get_embedding_function()
    return VectorStoreManager(embedding_fn)

components = get_global_components()
document_loader = components["document_loader"]
chunker = components["chunker"]
chat_memory = components["chat_memory"]
vector_store = get_vector_store()

# Session State PERSISTENCE
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}

if "current_session_id" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.current_session_id = initial_id
    st.session_state.all_chats[initial_id] = {
        "messages": [{"role": "assistant", "content": "Welcome to NeuralDoc! I'm your AI document assistant. Click below to attach files and start chatting."}],
        "files": []
    }
    chat_memory.create_session(initial_id)

# Current Session Data - Define these once at the top level
session_data = st.session_state.all_chats[st.session_state.current_session_id]
messages = session_data["messages"]
if "files" not in session_data:
    session_data["files"] = []
uploaded_files_history = session_data["files"]

# Helper for Session Reset
def start_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.current_session_id = new_id
    st.session_state.all_chats[new_id] = {
        "messages": [{"role": "assistant", "content": "Started a new conversation. How can I help you today?"}],
        "files": []
    }
    chat_memory.create_session(new_id)

# Sidebar: Chat History & Diagnostic Status
with st.sidebar:
    st.title("NeuralDoc")
    if st.button("New Chat", use_container_width=True, icon=":material/add:"):
        start_new_chat()
        st.rerun()

    st.divider()
    
    # Cloud Diagnostic: Database Status & Document Management
    doc_count = vector_store.get_document_count()
    st.info(f"Indexing Status\n\nKnowledge Base: {doc_count} chunks")
    
    if uploaded_files_history:
        st.caption("Active Documents")
        for doc_name in list(uploaded_files_history):
            col1, col2 = st.columns([0.8, 0.2])
            col1.text(f"File: {doc_name}")
            if col2.button("", key=f"del_{doc_name}", icon=":material/delete:", help="Remove this document"):
                if asyncio.run(vector_store.delete_document(doc_name, st.session_state.current_session_id)):
                    uploaded_files_history.remove(doc_name)
                    st.toast(f"Removed '{doc_name}'")
                    st.rerun()

        st.divider()
        st.subheader("System Reset")
        if st.button("Purge Knowledge Base", use_container_width=True, icon=":material/dangerous:", help="WARNING: This will permanently delete all files and clear all indexed data."):
            # Global purge
            vector_store.clear_all_knowledge()
            purge_all_documents()
            uploaded_files_history.clear()
            st.toast("Knowledge base has been completely purged.")
            st.rerun()

    st.divider()
    st.subheader("Recent Chats")
    for sid in reversed(list(st.session_state.all_chats.keys())):
        chat_msgs = st.session_state.all_chats[sid]["messages"]
        # Find first non-assistant message or first message
        label_msg = chat_msgs[0]["content"]
        if len(chat_msgs) > 1:
            label_msg = chat_msgs[1]["content"] if chat_msgs[1]["role"] == "user" else chat_msgs[1]["content"]
            
        title = (label_msg[:25] + '...') if len(label_msg) > 25 else label_msg
        if st.button(title, key=f"btn_{sid}", use_container_width=True, icon=":material/chat:"):
            st.session_state.current_session_id = sid
            st.rerun()

# Current Session Data
session_data = st.session_state.all_chats[st.session_state.current_session_id]
messages = session_data["messages"]
uploaded_files_history = session_data["files"]

# Layout: Main Chat
st.title("NeuralDoc Chat")

# Display chat messages
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Attachment Area
with st.container():
    with st.expander("📎 Attach Documents to this Session", expanded=not vector_store.has_documents()):
        uploaded_files = st.file_uploader("Upload PDF, DOCX, TXT, etc.", accept_multiple_files=True, type=["pdf", "docx", "txt", "pptx", "xlsx", "csv", "md"])
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in uploaded_files_history]
            if new_files:
                if st.button("Index Selected Files"):
                    with st.spinner("Processing documents into knowledge base..."):
                        session_id = st.session_state.current_session_id
                        
                        async def process_files():
                            p_files = []
                            f_files = []
                            for file in new_files:
                                filename = file.name
                                file_content = file.read()
                                file_size = len(file_content)
                                
                                is_valid, error = validate_file(filename, file_size)
                                if not is_valid:
                                    f_files.append(f"{filename}: {error}")
                                    continue
                                
                                try:
                                    saved_path = await save_uploaded_file(file_content, filename, session_id)
                                    docs = await document_loader.load(
                                        file_path=saved_path, session_id=session_id, original_filename=filename
                                    )
                                    chunks = chunker.chunk_documents(docs)
                                    await vector_store.add_documents(chunks)
                                    
                                    ext = Path(filename).suffix.lower()
                                    label = get_file_type_emoji(ext)
                                    p_files.append(f"{label} {filename}")
                                    
                                    if filename not in uploaded_files_history:
                                        uploaded_files_history.append(filename)
                                        
                                    logger.info("Processed '%s': %d chunks", filename, len(chunks))
                                except Exception as e:
                                    logger.error("Failed to process '%s': %s", filename, e)
                                    f_files.append(f"{filename}: {str(e)}")
                            return p_files, f_files

                        p_files, f_files = asyncio.run(process_files())
                        if p_files:
                            st.success(f"Indexed successfully: {', '.join(p_files)}")
                            chat_memory.update_session_documents(session_id, uploaded_files_history)
                            st.rerun() # Refresh to update counter
                        if f_files:
                            st.error(f"Failed to process: {', '.join(f_files)}")

# Handling User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not vector_store.has_documents():
            response = "**No documents indexed yet.**\n\nPlease attach your documents using the clip icon above first."
            st.markdown(response)
            messages.append({"role": "assistant", "content": response})
        else:
            response_placeholder = st.empty()
            
            retriever = DocumentRetriever(vector_store)
            engine = ChatEngine(retriever=retriever, memory=chat_memory)
            
            async def generate_response():
                res = ""
                sources = []
                try:
                    async for chunk in engine.generate_stream(query=prompt, session_id=st.session_state.current_session_id):
                        if chunk.get("token"):
                            res += chunk["token"]
                            response_placeholder.markdown(res + "▌")
                        
                        if chunk.get("done") and chunk.get("sources"):
                            sources = chunk["sources"]
                except Exception as e:
                    logger.error("Error generating response: %s", e)
                    res = f"An error occurred while generating response: {str(e)}"
                    response_placeholder.markdown(res)
                
                return res, sources
                
            full_response, sources = asyncio.run(generate_response())
            
            if sources:
                full_response += "\n\n---\n**Sources:**\n"
                for src in sources:
                    label = get_file_type_emoji(src.get("file_type", ""))
                    full_response += f"- {label} {src['source']}\n"
            
            response_placeholder.markdown(full_response)
            messages.append({"role": "assistant", "content": full_response})
