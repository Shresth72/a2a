import weaviate
from weaviate.client import WeaviateClient

import constants
from typing import List
from utils import refresh_token
from models import BaseCollection


class Database:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "client") or self.client is None:
            token = refresh_token()
            self.client: WeaviateClient = weaviate.connect_to_weaviate_cloud(
                cluster_url=constants.WEAVIATE_URL,
                auth_credentials=weaviate.auth.AuthApiKey(
                    constants.WEAVIATE_API_KEY,
                ),
                headers={
                    "X-HuggingFace-Api-Key": constants.HF_API_KEY,
                    "X-Goog-Vertex-Api-Key": token,
                },
            )
            assert self.client.is_ready(), "Weaviate client is not ready"

    def init_collections(self, collections: List[BaseCollection]):
        existing = set(self.client.collections.list_all())

        for cls in collections:
            if cls.__name__ not in existing:
                print(
                    f"Collection '{cls.__name__}' missing. Creating and populating..."
                )
                cls.create(self.client)
                cls.populate(self.client)
            else:
                print(
                    f"Collection '{cls.__name__}' already exists. Skipping population."
                )

    def query_generate(self, collection_cls: BaseCollection, query, limit, prompt):
        collection = self.client.collections.get(collection_cls.__name__)
        response = collection.generate.near_text(
            query=query,
            limit=limit,
            single_prompt=prompt,
        )

        return [
            {
                "generated": obj.generated,
            }
            for obj in response.objects
        ]

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
