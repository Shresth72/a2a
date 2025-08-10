from langchain_community.vectorstores import VectorStore


def query_vector_store(vector_store: VectorStore, query: str):
    retriever = vector_store.as_retriever(top_k=4)
    docs = retriever.invoke(query)
    return retriever, docs
