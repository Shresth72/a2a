import glob
import base64

from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore


def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def emded_images(path: str) -> VectorStore:
    lc_docs = []
    paths = glob.glob(path, recursive=True)

    for path in paths:
        doc = Document(
            page_content=encode_image(path),
            metadata={"source": path},
        )
        lc_docs.append(doc)

    vector_store = FAISS.from_documents(lc_docs, embedding=OpenCLIPEmbeddings())
    return vector_store


def retrieve_dog_similar_to_cat(retriever: VectorStoreRetriever):
    dog_paths = glob.glob("./images/dog*.jpeg", recursive=True)
    dog_to_cat = {}

    for dog_pic in dog_paths:
        docs = retriever.invoke(encode_image(dog_pic))
        cats_retrieved = 0
        for i, doc in enumerate(docs):
            if "cat" in doc.metadata["source"]:
                cats_retrieved += 4 - i
        dog_to_cat[dog_pic] = cats_retrieved


def retrieve_similar_images(retriever: VectorStoreRetriever, query_image: str):
    docs = retriever.invoke(encode_image(query_image), k=4)
    for doc in docs:
        print(doc.metadata)
