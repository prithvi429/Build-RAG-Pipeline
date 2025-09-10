import os
import uuid
import chromadb
from typing import List, Any
from langchain.schema import Document

class VectorStore:
    def __init__(self, collection_name: str = "documents", persist_directory: str = "../data/vector_store"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document embeddings for RAG"}
        )

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings count mismatch")
        ids, metadatas, docs_text, embs = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            docs_text.append(doc.page_content)
            embs.append(emb)
        self.collection.add(ids=ids, embeddings=embs, metadatas=metadatas, documents=docs_text)

    def query(self, query_embedding: List[float], top_k: int = 5):
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
