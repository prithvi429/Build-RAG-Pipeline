import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

class GroqLLM:
    def __init__(self, model_name: str = "gemma2-9b-it", api_key: str = None):
        # Use passed api_key or environment variable
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set. "
                "Please set it in your environment or in a .env file."
            )
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=1024
        )

    def generate_response(self, query: str, context: str) -> str:
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""
        )
        prompt = prompt_template.format(context=context, question=query)
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content
