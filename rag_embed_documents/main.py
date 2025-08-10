from dotenv import load_dotenv

from chunking import create_chunks_from_files
from embed_store import embed_and_store
from query import query_vector_store
from llm import invoke_llm


load_dotenv()


def main():
    data_dir = "./Big Star Collectibles"
    file_texts = create_chunks_from_files(data_dir)

    vector_store = embed_and_store(file_texts)
    print("Successfully populated vector_store")

    query = "What year was Big Star Collectibles Started?"
    query = "I want to join Big Star Collectibles as a E-Commerce Web Developer"
    query = "Tell me about Big Star Collectibles Trading Cards"
    retriever, _ = query_vector_store(vector_store, query)

    response = invoke_llm(
        query,
        retriever,
    )
    print(f"\nFinal Response: {response}")


if __name__ == "__main__":
    main()
