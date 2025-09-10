# 🚀 Build-RAG-Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG) pipeline** for building knowledge-grounded AI systems. This project enables you to ingest, chunk, embed, index, retrieve, and generate answers from your own data using state-of-the-art LLMs and vector databases.

---

## 🌟 Key Features

- **Flexible Data Ingestion:** Load text, PDF, CSV, JSON, and database sources.
- **Smart Chunking:** Split documents for optimal retrieval and context.
- **Powerful Embeddings:** Use Sentence Transformers for high-quality vectorization.
- **Vector Store Integration:** Fast similarity search with ChromaDB.
- **Custom Retriever:** Retrieve the most relevant chunks for any query.
- **LLM Integration:** Seamless connection to Groq LLM or your own LLM.
- **Advanced RAG Pipeline:** Streaming, citations, summarization, and query history.

---

## 💡 Use Case Example

**"Ask your documents anything!"**

Suppose you have a folder of company policies, research papers, or technical docs. With Build-RAG-Pipeline, you can:

1. Ingest all your files (PDF, TXT, CSV, etc.)
2. Automatically chunk and embed them
3. Store embeddings in a vector database
4. Ask natural language questions and get answers grounded in your data, with citations and context

**Example:**
> _"How can sensitive or private company data remain secure in an Agentic AI environment?"_

The pipeline retrieves the most relevant chunks, feeds them to the LLM, and returns a concise, cited answer.

---

## 🛠️ How to Use

1. **Clone the repository:**
	```bash
	git clone https://github.com/prithvi429/Build-RAG-Pipeline.git
	cd Build-RAG-Pipeline
	```

2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

3. **Set up your environment:**
	- Create a `.env` file and add your API keys (e.g., `GROQ_API_KEY`)

4. **Organize your data:**
	- Place your files in the `data/text_files/` directory

5. **Run the pipeline:**
	- Use the provided notebook (`notbook/document.ipynb`) or Python scripts in `src/` to ingest, embed, and query your data

---



## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss your ideas.

---

## 📜 License

This project is licensed under the MIT License.
