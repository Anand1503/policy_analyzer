"""
RAG (Retrieval-Augmented Generation) pipeline using ChromaDB + T5.

Architecture:
  - Embeddings: sentence-transformers (all-MiniLM-L6-v2) via HuggingFaceEmbeddings
  - Vector Store: ChromaDB with clause-level embeddings
  - Generator: T5-base (replaces Ollama for self-contained local inference)
  - Retriever: ChromaDB similarity search → top-k clause chunks
  - Pipeline: Retrieve relevant clauses → T5 generates grounded answer
"""

import logging
import torch
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# ─── Singletons ─────────────────────────────────────────────
_embeddings = None
_rag_service = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-loaded singleton embedding model (sentence-transformers)."""
    global _embeddings
    if _embeddings is None:
        logger.info(f"[model_loader] Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        logger.info("[model_loader] Embedding model loaded")
    return _embeddings


def _get_vector_store(collection: str = "policy_documents") -> Chroma:
    """Get or create a ChromaDB collection."""
    return Chroma(
        collection_name=collection,
        embedding_function=_get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def get_rag_service() -> "RAGService":
    """Singleton RAG service."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


class RAGService:
    """
    RAG pipeline: document ingestion, clause-level embeddings, and Q&A retrieval.
    Uses T5 for answer generation (self-contained, no external LLM required).
    """

    def __init__(self):
        # T5 is loaded via model_loader singleton — no per-request loading
        self._t5_loaded = False

    def _ensure_t5(self):
        """Lazy-load T5 once."""
        if not self._t5_loaded:
            from app.ml.model_loader import get_summarizer
            get_summarizer()  # Ensures T5 is loaded
            self._t5_loaded = True

    # ─── Document-Level Ingestion (legacy) ───────────────────

    async def ingest_document(self, document_id: str, text: str, metadata: Dict[str, Any]) -> int:
        """Ingest full document text into ChromaDB for RAG Q&A."""
        doc = Document(page_content=text, metadata={"source": document_id, **metadata})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents([doc])
        logger.info(f"RAG: Split document '{document_id}' into {len(chunks)} chunks")

        store = _get_vector_store()
        store.add_documents(chunks)
        return len(chunks)

    # ─── Clause-Level Embedding Storage (Module 1) ───────────

    def store_clause_embeddings(
        self,
        document_id: str,
        clauses: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> List[str]:
        """
        Generate and store embeddings for individual clauses.
        Uses chunked clause embeddings (mandatory per architecture rules).

        Args:
            document_id: UUID string of the parent document
            clauses: List of clause dicts with 'index' and 'text' keys
            batch_size: Number of clauses to embed per batch

        Returns:
            List of embedding_id strings (ChromaDB document IDs)
        """
        try:
            store = _get_vector_store(collection="clause_embeddings")
            embedding_ids = []

            for batch_start in range(0, len(clauses), batch_size):
                batch = clauses[batch_start:batch_start + batch_size]

                # Build deterministic IDs
                ids = [f"{document_id}_clause_{c['index']}" for c in batch]

                # Cache check: skip clauses that already have embeddings
                existing = self._check_existing_embeddings(store, ids)
                new_clauses = []
                new_ids = []

                for clause, cid in zip(batch, ids):
                    if cid in existing:
                        logger.debug(f"Embedding cache hit: {cid}")
                    else:
                        new_clauses.append(clause)
                        new_ids.append(cid)

                # Store new embeddings
                if new_clauses:
                    documents = [
                        Document(
                            page_content=c["text"],
                            metadata={
                                "document_id": document_id,
                                "clause_index": c["index"],
                                "category": c.get("category", ""),
                            },
                        )
                        for c in new_clauses
                    ]
                    store.add_documents(documents, ids=new_ids)
                    logger.info(
                        f"RAG: Stored {len(new_clauses)} new clause embeddings "
                        f"(batch {batch_start // batch_size + 1})"
                    )

                embedding_ids.extend(ids)

            logger.info(f"RAG: Total {len(embedding_ids)} clause embeddings for document {document_id}")
            return embedding_ids

        except Exception as e:
            raise EmbeddingError(
                f"Failed to store clause embeddings: {str(e)}",
                document_id=document_id,
                step="store_clause_embeddings",
            )

    def _check_existing_embeddings(self, store: Chroma, ids: List[str]) -> set:
        """Check which IDs already exist in ChromaDB. Returns set of existing IDs."""
        try:
            collection = store._collection
            result = collection.get(ids=ids, include=[])
            return set(result["ids"]) if result and result.get("ids") else set()
        except Exception:
            return set()

    # ─── Q&A with T5 Generator ────────────────────────────────

    def answer_question(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        RAG Q&A: Retrieve relevant clauses, then generate answer with T5.
        Must always retrieve before answering (no direct full-document queries).

        Args:
            question: User's question
            top_k: Number of clauses to retrieve

        Returns:
            Dict with answer, source documents, and retrieval metadata
        """
        self._ensure_t5()

        # Step 1: Retrieve relevant clause chunks
        store = _get_vector_store()
        retriever = store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)

        if not docs:
            return {
                "answer": "I could not find relevant information in the document.",
                "source_documents": [],
                "retrieval_count": 0,
            }

        # Step 2: Build context from retrieved chunks
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 3: Generate answer with T5
        from app.ml.model_loader import get_summarizer, get_summarizer_device
        model, tokenizer = get_summarizer()
        device = get_summarizer_device()

        # T5 question-answering prompt
        input_text = (
            f"answer question based on context: "
            f"question: {question} "
            f"context: {context}"
        )

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=200,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "answer": answer,
            "source_documents": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
            "retrieval_count": len(docs),
        }

    # ─── Legacy Q&A Chain Interface ──────────────────────────

    def get_qa_chain(self):
        """
        Legacy interface — returns a callable that answers questions.
        Wraps the T5-based answer_question method.
        """
        class T5QAChain:
            def __init__(self, rag_service):
                self._rag = rag_service

            def __call__(self, query_dict):
                question = query_dict.get("query", query_dict.get("question", ""))
                result = self._rag.answer_question(question)
                return {
                    "result": result["answer"],
                    "source_documents": [
                        Document(page_content=d["content"], metadata=d.get("metadata", {}))
                        for d in result["source_documents"]
                    ],
                }

        return T5QAChain(self)
