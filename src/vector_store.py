"""Vector Store Module using ChromaDB"""

from typing import List, Dict, Any, Optional
from pathlib import Path


class VectorStore:
    """Vector store using ChromaDB."""
    
    def __init__(self, collection_name: str = "neuro_lit_rag", 
                 persist_directory: str = "./data/chroma_db"):
        
        try:
            import chromadb
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    @property
    def count(self) -> int:
        return self.collection.count()
    
    def add(self, ids: List[str], embeddings: List[List[float]], 
            texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add documents to store."""
        
        if metadatas:
            clean_meta = []
            for m in metadatas:
                clean = {}
                for k, v in m.items():
                    if v is None:
                        continue
                    elif isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    elif isinstance(v, list):
                        clean[k] = ", ".join(str(x) for x in v[:5])
                    else:
                        clean[k] = str(v)
                clean_meta.append(clean)
            metadatas = clean_meta
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def query(self, embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Query for similar documents."""
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
            })
        
        return formatted
