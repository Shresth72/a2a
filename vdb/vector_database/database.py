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
        existing_collections = set(self.client.collections.list_all())

        for cls in collections:
            collection_name = cls.__name__

            if collection_name in existing_collections:
                existing_props = self._get_collection_schema(collection_name)
                new_props = {name: dtype.value for name, dtype in cls.properties}

                if existing_props != new_props and existing_props is not None:
                    print(
                        f"Schema change detected for '{collection_name}'. Recreating..."
                    )
                    self.client.collections.delete(collection_name)
                    cls.create(self.client)
                    cls.populate(self.client)
                else:
                    print(
                        f"Collection '{collection_name}' already exists. Skipping population..."
                    )
            else:
                print(f"Creating and populating: '{collection_name}'...")
                cls.create(self.client)
                cls.populate(self.client)

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

    def _get_collection_schema(self, collection_name):
        if self.client:
            try:
                config = self.client.collections.get(collection_name).config
                properties = config.get().properties
                if not properties:
                    return None
                return {prop.name: prop.data_type.value for prop in properties}
            except Exception:
                return None
        return None

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
