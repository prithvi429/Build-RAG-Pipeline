from typing import Dict, Any
from src.retriever import RAGRetriever
from src.llm import GroqLLM

def rag_simple(query: str, retriever: RAGRetriever, llm: GroqLLM, top_k: int = 3) -> str:
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."
    return llm.generate_response(query, context)

def rag_advanced(query: str, retriever: RAGRetriever, llm: GroqLLM, top_k: int = 5, min_score: float = 0.2, return_context: bool = False) -> Dict[str, Any]:
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', 'unknown'),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]
    confidence = max(doc['similarity_score'] for doc in results)
    answer = llm.generate_response(query, context)
    output = {'answer': answer, 'sources': sources, 'confidence': confidence}
    if return_context:
        output['context'] = context
    return output
