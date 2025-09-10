from typing import List, Dict, Any

class QueryHistory:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add(self, question: str, answer: str, sources: List[Dict[str, Any]], summary: str = None):
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
