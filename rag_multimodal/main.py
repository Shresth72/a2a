from embed_store import emded_images
from query import invoke_llm
from dotenv import load_dotenv

load_dotenv()


def main():
    vector_store = emded_images("../rag_embed_images/images/*.jpeg")
    retriever = vector_store.as_retriever()

    invoke_llm(retriever)
