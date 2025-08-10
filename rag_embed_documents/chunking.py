import os
from typing import List

from langchain.text_splitter import (
    CharacterTextSplitter,
)  # Split strings based on preset parameters
from langchain.schema import (
    Document,
)  # Add metadata to text and prepare it for vector storage


def create_chunks_from_files(data_dir) -> List[Document]:
    file_texts = []

    files = os.listdir(data_dir)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=128, chunk_overlap=32, separator="\n"
    )

    for file in files:
        with open(f"{data_dir}/{file}") as f:
            file_text = f.read()
        texts = text_splitter.split_text(file_text)
        for i, chunked_text in enumerate(texts):
            file_texts.append(
                Document(
                    page_content=chunked_text,
                    metadata={
                        "doc_title": file.split(".")[0],
                        "chunk_num": i,
                    },
                )
            )

    return file_texts
