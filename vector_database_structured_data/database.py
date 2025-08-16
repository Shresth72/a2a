import weaviate
from weaviate.client import WeaviateClient

import constants
from typing import List
from utils import refresh_token, _normalize_props
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

    def fetch_objects(self, collection_cls: BaseCollection, filters, limit, ref):
        collection = self.client.collections.get(collection_cls.name)  # updated
        return collection.query.fetch_objects(
            filters=filters,
            limit=limit,
            return_references=ref,
        )

    def fetch_objects_by_id(self, collection_cls: BaseCollection, cid, ref):
        collection = self.client.collections.get(collection_cls.name)  # updated
        return collection.query.fetch_object_by_id(
            uuid=cid,
            return_references=ref
        )

    def query_generate(self, collection_cls: BaseCollection, query, filters, limit, ref):
        collection = self.client.collections.get(collection_cls.name)  # updated
        return collection.generate.near_text(
            query=query,
            filters=filters,
            limit=limit,
            return_references=ref,
        )

    def query_hybrid(self, collection_cls: BaseCollection, query, filters, limit, alpha, ref):
        collection = self.client.collections.get(collection_cls.name)  # updated
        return collection.query.hybrid(
            query=query,
            filters=filters,
            limit=limit,
            alpha=alpha,
            return_references=ref,
        )

    def init_collections(self, collections: List[BaseCollection]):
        existing_collections = set(self.client.collections.list_all())
        recreated = {}

        for cls in collections:
            collection_name = cls.name

            if collection_name in existing_collections:
                if self._should_recreate_collection(cls):
                    print(f"Schema change detected for '{collection_name}'. Recreating...")
                    self.client.collections.delete(collection_name)
                    cls.create(self.client)
                    recreated[collection_name] = True
                else:
                    print(f"Collection '{collection_name}' already exists")
                    recreated[collection_name] = False
            else:
                print(f"Creating collection: '{collection_name}'...")
                cls.create(self.client)
                recreated[collection_name] = True

        for cls in collections:
            if recreated[cls.name] or self._should_add_references(cls):
                cls.add_references(self.client)

        for cls in collections:
            if recreated[cls.name] or self._should_add_references(cls) or self._should_populate_collection(cls):
                cls.populate(self.client)

    def _should_recreate_collection(self, cls: BaseCollection) -> bool:
        existing_props = _normalize_props(
            self._get_collection_schema(cls.name)
        )
        new_props = _normalize_props(
            {name: dtype.value for name, dtype, *_ in cls.properties}
        )
        return existing_props is not None and existing_props != new_props

    def _should_add_references(self, cls: BaseCollection) -> bool:
        if not cls.references:
            return False

        try:
            config = self.client.collections.get(cls.name).config.get()
            existing_refs = _normalize_props(
                {ref.name: ref.target_collections for ref in (config.references or [])}
            )
            new_refs = _normalize_props(
                {ref.name: [ref.target_collection] for ref in cls.references}
            )
            return existing_refs != new_refs
        except Exception:
            return True

    def _should_populate_collection(self, cls: BaseCollection) -> bool:
        if not cls.data_file:
            return False

        try:
            collection = self.client.collections.get(cls.name)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count == 0
        except Exception:
            return True

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
