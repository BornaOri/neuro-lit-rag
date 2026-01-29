"""Cohere Answer Generator Module"""

import os
import re
import cohere
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Citation:
    """A citation reference."""
    number: int
    pmid: str
    title: str
    authors: str
    year: str
    journal: str


@dataclass
class GeneratedAnswer:
    """A generated answer with citations."""
    answer: str
    citations: List[Citation]
    sources_used: int


class AnswerGenerator:
    """Generates answers using Cohere Command."""
    
    SYSTEM_PROMPT = """You are a neuroscience research assistant. Answer questions based on the provided research excerpts.

Guidelines:
- Base answers ONLY on provided sources
- Cite using [1], [2], etc.
- If sources don't have enough info, say so
- Use precise scientific terminology
- Be concise but thorough"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "command-r-plus-08-2024"):  # Updated model name!
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found")
        
        self.client = cohere.Client(self.api_key)
        self.model = model
    
    def generate(self, query: str, context_docs: List[Dict[str, Any]], 
                 temperature: float = 0.3, max_tokens: int = 1024) -> GeneratedAnswer:
        """Generate answer from retrieved documents."""
        
        formatted_context, citations = self._format_context(context_docs)
        
        prompt = f"""Based on these research excerpts, answer the question:

Question: {query}

Sources:
{formatted_context}

Provide a comprehensive answer citing sources using [1], [2], etc."""
        
        response = self.client.chat(
            message=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            preamble=self.SYSTEM_PROMPT
        )
        
        used_citations = self._extract_used_citations(response.text, citations)
        
        return GeneratedAnswer(
            answer=response.text,
            citations=used_citations,
            sources_used=len(used_citations)
        )
    
    def _format_context(self, docs: List[Dict[str, Any]]) -> tuple:
        """Format documents for the prompt."""
        parts = []
        citations = []
        
        for i, doc in enumerate(docs):
            num = i + 1
            meta = doc.get("metadata", {})
            
            authors = meta.get("authors", ["Unknown"])
            if isinstance(authors, str):
                author_str = authors.split(",")[0] if authors else "Unknown"
            elif isinstance(authors, list) and authors:
                author_str = authors[0]
            else:
                author_str = "Unknown"
            
            year = meta.get("year", "n.d.")
            title = meta.get("title", "Untitled")
            journal = meta.get("journal", "Unknown Journal")
            
            citations.append(Citation(
                number=num,
                pmid=meta.get("pmid", f"doc_{i}"),
                title=title,
                authors=author_str,
                year=year,
                journal=journal
            ))
            
            text = doc.get("text", "")
            parts.append(f"[{num}] {author_str} ({year}) - {journal}\nTitle: {title}\n{text}")
        
        return "\n\n---\n\n".join(parts), citations
    
    def _extract_used_citations(self, answer: str, 
                                 all_citations: List[Citation]) -> List[Citation]:
        """Find which citations were used."""
        matches = re.findall(r'\[(\d+)\]', answer)
        used_nums = set(int(m) for m in matches)
        return [c for c in all_citations if c.number in used_nums]
