import base64
from io import BytesIO
from PIL import Image
from typing import List
import os

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import VectorStoreRetriever


def invoke_llm(retriever: VectorStoreRetriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # All the elements in a chain must be a runnable
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(prompt_func)
        | llm
        | StrOutputParser()
    )

    return chain.invoke("rotweiler")


def prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)

    text_messsage = {
        "type": "text",
        "text": (
            "As an animal lover, your task is to analyze and interpret images of cute animals, "
            "Please use your extensive knowledge and analytical skills to provide a "
            "summary that includes:\n"
            "- A detailed description of the visual elements in the image.\n"
            f"User-provided keywords: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_messsage)

    return [HumanMessage(content=messages)]


def split_image_text_types(docs: List[Document]):
    images = []
    text = []

    for doc in docs:
        doc = doc.page_content
        if is_base64(doc):
            images.append(resize_base64_image(doc))
        else:
            text.append(doc)
    return {"images": images, "texts": text}


def resize_base64_image(base64_string: bytes, size=(128, 128)) -> bytes:
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))

    resized_img = img.resize(size, Image.LANCZOS)

    buffered = BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b16encode(buffered.getvalue()).decode("utf-8")


def is_base64(s: str) -> bool:
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False
