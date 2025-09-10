from typing import List, Dict, Any
from src.vector_store import VectorStore
from src.embedding_manager import EmbeddingManager

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_emb = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.query(query_emb, top_k=top_k)
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_id, doc, meta, dist) in enumerate(zip(results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0])):
                similarity = 1 - dist
                if similarity >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': doc,
                        'metadata': meta,
                        'similarity_score': similarity,
                        'distance': dist,
                        'rank': i + 1
                    })
        return retrieved_docs
