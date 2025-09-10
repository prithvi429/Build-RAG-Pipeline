# main.py
from src.embedding_manager import DummyEmbedder
from src.vector_store import InMemoryVectorStore
from src.ingestion import ingest_files
from src.retriever import Retriever
from src.llm import GroqLLM
from src.rag_pipeline import RAGPipeline

def build_and_run():
    # Step 1: Initialize components
    embedder = DummyEmbedder(dim=64)  # swap later with real embedding model
    vector_store = InMemoryVectorStore()

    # Step 2: Ingest text files
    files = [
        "data/text_files/machine_learning.txt",
        "data/text_files/python_intro.txt",
        # TODO: add PDF ingestion later
    ]
    ingested = ingest_files(files, embedder, vector_store)
    print(f"[INFO] Ingested {ingested} chunks")

    # Step 3: Create retriever + LLM
    retriever = Retriever(vector_store, embedder, top_k=3)
    llm = GroqLLM(model="llama3-8b-8192")  # ✅ uses Groq API
    rag = RAGPipeline(retriever, llm)

    # Step 4: Interactive Q&A loop
    while True:
        q = input("\nEnter your question (or type 'exit' to quit): ")
        if q.strip().lower() == "exit":
            break
        ans = rag.answer(q)
        print("\nAnswer:\n", ans)

if __name__ == "__main__":
    build_and_run()
