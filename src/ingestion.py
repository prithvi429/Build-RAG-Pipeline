from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader, JSONLoader, SQLDatabaseLoader
)
from sqlalchemy import create_engine

LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".json": JSONLoader,
}

def load_documents(data_dir: str) -> List[Document]:
    all_documents = []
    data_path = Path(data_dir)
    for file in data_path.rglob("*"):
        if not file.is_file():
            continue
        ext = file.suffix.lower()
        if ext in LOADER_MAPPING:
            loader_cls = LOADER_MAPPING[ext]
            try:
                loader = loader_cls(str(file))
                docs = loader.load()
                for d in docs:
                    d.metadata["source_file"] = file.name
                    d.metadata["file_type"] = ext
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
    return all_documents

def load_from_database(conn_str: str, query: str) -> List[Document]:
    try:
        loader = SQLDatabaseLoader(create_engine(conn_str), query=query)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = "database"
            d.metadata["file_type"] = "db"
        return docs
    except Exception as e:
        print(f"DB error: {e}")
        return []
