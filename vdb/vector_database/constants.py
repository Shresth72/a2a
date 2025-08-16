import os
import uuid
from dotenv import load_dotenv

load_dotenv()

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash"

PROJECT_ID = os.getenv("PROJECT_ID")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")
