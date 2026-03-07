"""
NeuralDoc — Chat Engine Module
Orchestrates the RAG pipeline with Google Gemini for conversational document Q&A.
"""

import logging
from typing import AsyncGenerator, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config import GOOGLE_API_KEY, GEMINI_MODEL, RAG_PROMPT_TEMPLATE
from rag.retriever import DocumentRetriever
from chat.memory import ChatMemory

logger = logging.getLogger(__name__)


class ChatEngine:
    """
    Main chat engine that orchestrates RAG retrieval and LLM generation.

    Combines document retrieval, prompt construction, chat memory,
    and streaming Gemini responses.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        memory: ChatMemory,
    ) -> None:
        """
        Initialize the chat engine.

        Args:
            retriever: DocumentRetriever instance for fetching context.
            memory: ChatMemory instance for conversation history.
        """
        self.retriever = retriever
        self.memory = memory
        self._llm = None
        self._prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        logger.info("ChatEngine initialized with model: %s", GEMINI_MODEL)

    def _get_llm(self):
        """Lazy-initialize the Gemini LLM."""
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                streaming=True,
                max_retries=3,
                convert_system_message_to_human=True,
            )
        return self._llm

    async def generate_response(
        self,
        query: str,
        session_id: str,
    ) -> dict:
        """
        Generate a non-streaming response for a user query.

        Args:
            query: The user's question.
            session_id: Current chat session ID.

        Returns:
            Dictionary with 'answer', 'sources', and 'context_docs'.
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve(query=query, session_id=None)

        if not relevant_docs:
            return {
                "answer": "I cannot find the answer in the uploaded documents. "
                          "Please upload documents first or try a different question.",
                "sources": [],
                "context_docs": [],
            }

        # Build context
        context = self.retriever.format_context(relevant_docs)
        sources = self.retriever.get_source_citations(relevant_docs)

        # Build the prompt with chat history
        history = self.memory.get_history(session_id, limit=10)
        history_text = self._format_history(history)

        full_context = context
        if history_text:
            full_context = f"Previous conversation:\n{history_text}\n\n---\n\n{context}"

        # Generate response
        prompt = self._prompt.format(context=full_context, question=query)
        llm = self._get_llm()
        response = await llm.ainvoke(prompt)

        answer = response.content

        # Store in memory
        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(session_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": sources,
            "context_docs": relevant_docs,
        }

    async def generate_stream(
        self,
        query: str,
        session_id: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Generate a streaming response for a user query.

        Yields chunks of the response as they are generated.

        Args:
            query: The user's question.
            session_id: Current chat session ID.

        Yields:
            Dictionaries with 'token' (str), and on the final chunk,
            'sources' and 'done' keys.
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve(query=query, session_id=None)

        if not relevant_docs:
            yield {
                "token": "I cannot find the answer in the uploaded documents. "
                         "Please upload documents first or try a different question.",
                "sources": [],
                "done": True,
            }
            return

        # Build context
        context = self.retriever.format_context(relevant_docs)
        sources = self.retriever.get_source_citations(relevant_docs)

        # Include chat history
        history = self.memory.get_history(session_id, limit=10)
        history_text = self._format_history(history)

        full_context = context
        if history_text:
            full_context = f"Previous conversation:\n{history_text}\n\n---\n\n{context}"

        # Build prompt
        prompt = self._prompt.format(context=full_context, question=query)

        # Store user message
        self.memory.add_message(session_id, "user", query)

        # Stream response
        llm = self._get_llm()
        full_response = ""

        async for chunk in llm.astream(prompt):
            token = chunk.content
            if token:
                full_response += token
                yield {"token": token, "done": False}

        # Store assistant response
        self.memory.add_message(session_id, "assistant", full_response)

        # Final chunk with sources
        yield {
            "token": "",
            "sources": sources,
            "done": True,
        }

    def _format_history(self, history: list[dict]) -> str:
        """
        Format chat history messages into a text block.

        Args:
            history: List of message dicts with 'role' and 'content'.

        Returns:
            Formatted history string.
        """
        if not history:
            return ""

        lines: list[str] = []
        for msg in history[-6:]:  # Last 6 messages for context window
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)
