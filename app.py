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

    /* Hide the radio button circle/dot */
    [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    
    /* Style the navigation labels */
    [data-testid="stRadio"] div[role="radiogroup"] label {
        font-size: 1.2rem !important;
        font-weight: 500;
        padding: 5px 0;
        cursor: pointer;
    }

    /* Style Recent Chat buttons to have theme-consistent borders */
    section[data-testid="stSidebar"] .stButton > button {
        border: 1px solid var(--border-color) !important;
        background-color: transparent !important;
        color: var(--text-color) !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: block !important;
        transition: all 0.2s ease;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--secondary-background-color) !important;
        border-color: var(--primary-color) !important;
    }

    /* Active chat button: Stroke/Outline only (no red tint) */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        border: 2px solid var(--primary-color) !important;
        background-color: transparent !important; /* Removed red tint */
        box-shadow: none !important;
    }
    
    /* Add a subtle indicator for the selected page if needed, 
       but user requested no red/white highlight. 
       Standard hover effect is usually enough. */
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

# Sidebar: Navigation & History
with st.sidebar:
    st.title("NeuralDoc")
    
    # Navigation Dashboard
    nav_options = {
        "Chat": "Chat",
        "Settings": "Settings"
    }
    nav_icons = {
        "Chat": ":material/chat:",
        "Settings": ":material/settings:"
    }
    
    page = st.radio(
        "Navigation",
        options=list(nav_options.keys()),
        format_func=lambda x: f"{nav_icons[x]} {nav_options[x]}",
        label_visibility="collapsed",
        index=0,
        key="nav_radio"
    )
    st.divider()

    if page == "Chat":
        if st.button("New Chat", use_container_width=True, icon=":material/add:"):
            start_new_chat()
            st.rerun()

        st.divider()
        
        # Current Session Info
        doc_count = vector_store.get_document_count()
        st.info(f"Session Status\n\nKnowledge Base: {doc_count} chunks")
        
        if uploaded_files_history:
            st.caption("Active Documents (This Session)")
            for doc_name in list(uploaded_files_history):
                col1, col2 = st.columns([0.8, 0.2])
                col1.text(f"File: {doc_name}")
                if col2.button("", key=f"del_{doc_name}", icon=":material/delete:", help="Remove this document"):
                    # Use a safer check for the method existence
                    if hasattr(vector_store, "delete_document"):
                        if asyncio.run(vector_store.delete_document(doc_name, st.session_state.current_session_id)):
                            uploaded_files_history.remove(doc_name)
                            st.toast(f"Removed '{doc_name}'")
                            st.rerun()
                    else:
                        st.error("Deletion service unavailable. Please check logs.")

        st.divider()
        st.subheader("Recent Chats")
        for sid in reversed(list(st.session_state.all_chats.keys())):
            chat_container = st.container()
            with chat_container:
                col1, col2 = st.columns([0.85, 0.15])
                
                chat_msgs = st.session_state.all_chats[sid]["messages"]
                label_msg = chat_msgs[0]["content"] if chat_msgs else "Empty Chat"
                if len(chat_msgs) > 1:
                    label_msg = chat_msgs[1]["content"] if chat_msgs[1]["role"] == "user" else chat_msgs[1].get("content", "...")
                    
                title_words = label_msg.split()[:3]
                title = " ".join(title_words)
                if len(label_msg.split()) > 3:
                    title += "..."
                
                # Highlight active chat
                is_active = sid == st.session_state.current_session_id
                btn_type = "primary" if is_active else "secondary"
                
                if col1.button(title, key=f"btn_{sid}", use_container_width=True, icon=":material/chat:", type=btn_type):
                    st.session_state.current_session_id = sid
                    st.rerun()
                
                if col2.button("", key=f"del_chat_{sid}", icon=":material/close:", help="Delete this chat"):
                    chat_memory.delete_session(sid)
                    del st.session_state.all_chats[sid]
                    if sid == st.session_state.current_session_id:
                        if st.session_state.all_chats:
                            st.session_state.current_session_id = list(st.session_state.all_chats.keys())[0]
                        else:
                            start_new_chat()
                    st.rerun()
    
    else: # Settings Page Dashboard
        st.subheader("System Reset")
        st.caption("Permanently wipe all documents and clear the entire knowledge base.")
        if st.button("Purge Knowledge Base", use_container_width=True, icon=":material/dangerous:", help="CAUTION: This affects ALL sessions."):
            vector_store.clear_all_knowledge()
            purge_all_documents()
            for chat in st.session_state.all_chats.values():
                chat["files"] = []
            st.toast("Global Knowledge Base Purged.")
            st.rerun()

# Layout: Main Content Area
if page == "Chat":
    st.title("NeuralDoc Chat")
    
    # Display chat messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
else:
    st.title("NeuralDoc Settings")
    st.divider()
    
    st.subheader("Global Knowledge Base")
    doc_stats = vector_store.get_document_stats()
    
    if not doc_stats:
        st.info("The knowledge base is currently empty. Start by uploading documents in a chat session.")
    else:
        st.write(f"Total Unique Documents: **{len(doc_stats)}**")
        
        # Display as a professional table
        import pandas as pd
        df = pd.DataFrame(doc_stats)
        df.columns = ["Filename", "Chunks Indexed"]
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.write("---")
        st.caption("Individual document deletion is available within the 'Chat' view for the current session.")
    
    st.divider()
    st.info("Settings are global and affect the entire application instance.")

# Attachment Area
with st.container():
    with st.expander("Attach Documents to this Session", expanded=not vector_store.has_documents(), icon=":material/attach_file:"):
        uploaded_files = st.file_uploader("Upload PDF, DOCX, TXT, etc.", accept_multiple_files=True, type=["pdf", "docx", "txt", "pptx", "xlsx", "csv", "md"], label_visibility="collapsed")
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in uploaded_files_history]
            if new_files:
                with st.status("Auto-indexing documents...", expanded=True) as status:
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
                                status.update(label=f"Processing {filename}...", state="running")
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
                        status.update(label="Indexing complete!", state="complete", expanded=False)
                        st.toast(f"Successfully indexed: {len(p_files)} files")
                        chat_memory.update_session_documents(session_id, uploaded_files_history)
                        st.rerun()
                    if f_files:
                        status.update(label="Some files failed to index", state="error")
                        for error in f_files:
                            st.error(error)

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
