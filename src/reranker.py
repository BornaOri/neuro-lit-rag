"""Cohere Reranker Module - KEY DIFFERENTIATOR!"""

import os
import cohere
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RerankedResult:
    """A reranked search result."""
    index: int
    text: str
    relevance_score: float
    metadata: Optional[Dict[str, Any]] = None


class CohereReranker:
    """
    Reranks documents using Cohere Rerank API.
    
    Why Reranking Matters:
    - Vector search is fast but approximate
    - Reranking uses cross-encoder for precise relevance
    - Dramatically improves results for technical queries
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-v3.5"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found")
        
        self.client = cohere.Client(self.api_key)
        self.model = model
    
    def rerank(self, query: str, documents: List[str], 
               top_n: Optional[int] = None) -> List[RerankedResult]:
        """Rerank documents by relevance to query."""
        if not documents:
            return []
        
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_n or len(documents),
            return_documents=True
        )
        
        results = []
        for r in response.results:
            results.append(RerankedResult(
                index=r.index,
                text=r.document.text,
                relevance_score=r.relevance_score
            ))
        
        return results
    
    def rerank_with_metadata(self, query: str, documents: List[Dict[str, Any]], 
                             text_key: str = "text", 
                             top_n: Optional[int] = None) -> List[RerankedResult]:
        """Rerank documents while preserving metadata."""
        texts = [doc[text_key] for doc in documents]
        results = self.rerank(query, texts, top_n)
        
        # FIXED: Properly extract nested metadata
        for result in results:
            original = documents[result.index]
            result.metadata = original.get("metadata", {})
        
        return results
