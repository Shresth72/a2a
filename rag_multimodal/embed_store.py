import glob
import base64

from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


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
