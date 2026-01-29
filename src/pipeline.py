"""NeuroLitRAG Pipeline - Main RAG Orchestration"""

import os
from typing import Dict, Any

from .embeddings import CohereEmbedder
from .vector_store import VectorStore
from .reranker import CohereReranker
from .generator import AnswerGenerator
from .data_ingestion import TextChunker, DEMO_PAPERS


class NeuroLitRAG:
    """
    Main RAG pipeline for neuroscience literature.
    
    Usage:
        rag = NeuroLitRAG()
        rag.load_demo_data()
        result = rag.query("What is the role of the hippocampus?")
    """
    
    def __init__(self, top_k_retrieve: int = 20, top_n_rerank: int = 5):
        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("COHERE_API_KEY not found!")
        
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank = top_n_rerank
        
        self.embedder = CohereEmbedder()
        self.vector_store = VectorStore()
        self.reranker = CohereReranker()
        self.generator = AnswerGenerator()
        self.chunker = TextChunker()
    
    def load_demo_data(self) -> Dict[str, int]:
        """Load demo papers."""
        all_chunks = []
        for paper in DEMO_PAPERS:
            chunks = self.chunker.chunk_paper(paper)
            all_chunks.extend(chunks)
        
        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed_documents(texts, show_progress=False)
        
        self.vector_store.add(
            ids=[c.chunk_id for c in all_chunks],
            embeddings=embeddings,
            texts=texts,
            metadatas=[c.metadata for c in all_chunks]
        )
        
        return {"papers": len(DEMO_PAPERS), "chunks": len(all_chunks)}
    
    def query(self, question: str, use_reranking: bool = True) -> Dict[str, Any]:
        """Query the RAG system."""
        
        if self.vector_store.count == 0:
            return {"error": "No documents loaded."}
        
        # 1. Embed query
        query_embedding = self.embedder.embed_query(question)
        
        # 2. Retrieve candidates
        retrieved = self.vector_store.query(
            embedding=query_embedding,
            top_k=self.top_k_retrieve
        )
        
        # 3. Rerank
        if use_reranking and retrieved:
            reranked = self.reranker.rerank_with_metadata(
                query=question,
                documents=retrieved,
                text_key="text",
                top_n=self.top_n_rerank
            )
            context_docs = [
                {"text": r.text, "metadata": r.metadata}
                for r in reranked
            ]
            rerank_scores = [r.relevance_score for r in reranked]
        else:
            context_docs = [
                {"text": r["text"], "metadata": r["metadata"]}
                for r in retrieved[:self.top_n_rerank]
            ]
            rerank_scores = None
        
        # 4. Generate answer
        result = self.generator.generate(
            query=question,
            context_docs=context_docs
        )
        
        return {
            "question": question,
            "answer": result.answer,
            "citations": [
                {
                    "number": c.number,
                    "title": c.title,
                    "authors": c.authors,
                    "year": c.year,
                    "journal": c.journal
                }
                for c in result.citations
            ],
            "sources_used": result.sources_used,
            "reranking_used": use_reranking,
            "rerank_scores": rerank_scores[:3] if rerank_scores else None
        }
