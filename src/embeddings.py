"""Cohere Embeddings Module"""

import os
import cohere
from typing import List, Optional
from tqdm import tqdm


class CohereEmbedder:
    """Generates embeddings using Cohere Embed API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "embed-english-v3.0"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found")
        
        self.client = cohere.Client(self.api_key)
        self.model = model
        self.embedding_dim = 1024
    
    def embed_documents(self, texts: List[str], batch_size: int = 96, 
                        show_progress: bool = True) -> List[List[float]]:
        """Embed documents for storage."""
        all_embeddings = []
        
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Embedding")
        
        for i in batches:
            batch = texts[i:i + batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document",
                truncate="END"
            )
            all_embeddings.extend(response.embeddings)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a search query."""
        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="search_query",
            truncate="END"
        )
        return response.embeddings[0]
