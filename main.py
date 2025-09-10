import os
from src.ingestion import load_documents
from src.text_splitter import split_documents
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.llm import GroqLLM
from src.rag_pipeline import rag_simple
from src.history import QueryHistory

def main():
    # Load documents from data directory
    data_dir = "./data"
    print("Loading documents...")
    documents = load_documents(data_dir)
    
    # Split documents into chunks
    print("Splitting documents...")
    chunks = split_documents(documents)
    
    # Generate embeddings
    embedding_manager = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    
    # Initialize vector store and add documents
    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)
    
    # Initialize retriever and LLM
    retriever = RAGRetriever(vector_store, embedding_manager)
    llm = GroqLLM()
    
    # Initialize query history
    history = QueryHistory()
    
    # Simple CLI loop for queries
    print("Ready for queries. Type 'exit' to quit.")
    while True:
        query = input("Enter your question: ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = rag_simple(query, retriever, llm)
        print(f"Answer:\n{answer}\n")
        history.add(query, answer, [])
    
    print("Session ended.")

if __name__ == "__main__":
    main()
