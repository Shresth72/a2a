from typing import List

from langchain_community.vectorstores import (
    FAISS,  # Facebook AI Similarity Search
    VectorStore,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


def embed_and_store(file_texts: List[Document]) -> VectorStore:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        documents=file_texts,
        embedding=embeddings,
    )
    return vector_store
