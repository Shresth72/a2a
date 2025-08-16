import os
import uuid
import pandas as pd
from abc import ABC, abstractmethod

import weaviate.classes as wvc
from weaviate.client import WeaviateClient
from weaviate.classes.config import Configure

import constants
from utils import make_properties


class BaseCollection(ABC):
    name = None
    data_file = None
    properties = []
    references = None

    vector_config = Configure.Vectors.text2vec_huggingface(
        model=constants.HF_MODEL,
    )
    generative_config = Configure.Generative.google(
        project_id=constants.PROJECT_ID,
        model_id="gemini-2.0-flash",
    )

    @classmethod
    def create(cls, client: WeaviateClient):
        print(f"Creating collection: '{cls.name}'...")
        client.collections.create(
            name=cls.name,
            vector_config=cls.vector_config,
            generative_config=cls.generative_config,
            properties=make_properties(cls.properties),
            references=cls.references,
        )

    @classmethod
    @abstractmethod
    def build_properties(cls, row):
        raise NotImplementedError

    @classmethod
    def populate(cls, client: WeaviateClient):
        if not cls.data_file or not os.path.exists(cls.data_file):
            print(f"No data file for '{cls.name}', Skipping population...")
            return

        print(f"Populating collection '{cls.name}'...")
        collection = client.collections.get(cls.name)
        df = pd.read_csv(cls.data_file)

        objs = []
        for _, row in df.iterrows():
            row_objs = cls.build_properties(row)
            if isinstance(row_objs, dict):
                row_objs = [row_objs]

            for obj in row_objs:
                objs.append(
                    wvc.data.DataObject(
                        uuid=obj["uuid"],
                        properties=obj["properties"],
                        references=obj.get("references") or None,
                    )
                )

        if objs:
            collection.data.insert_many(objs)


class ReviewsCollection(BaseCollection):
    name = "Reviews"
    data_file = "movies.csv"
    properties = [
        ("body", wvc.config.DataType.TEXT, None),
    ]

    @classmethod
    def build_properties(cls, row):
        objs = []
        for c in [1, 2, 3]:
            col = f"Critic Review {c}"
            text = row.get(col)

            if pd.notna(text) and str(text).strip():
                review_uuid = uuid.uuid5(constants.NAMESPACE, text.strip())
                objs.append(
                    {
                        "uuid": review_uuid,
                        "properties": {"body": text.strip()},
                    }
                )
        return objs


class MoviesCollection(BaseCollection):
    name = "Movies"
    data_file = "movies.csv"
    properties = [
        ("title", wvc.config.DataType.TEXT, lambda row: row["Movie Title"]),
        ("description", wvc.config.DataType.TEXT, lambda row: row["Description"]),
        ("rating", wvc.config.DataType.NUMBER, lambda row: float(row["Star Rating"])),
        ("movie_id", wvc.config.DataType.INT, lambda row: int(row["ID"])),
        ("year", wvc.config.DataType.INT, lambda row: int(row["Year"])),
        ("director", wvc.config.DataType.TEXT, lambda row: row["Director"]),
    ]
    references = [
        wvc.config.ReferenceProperty(
            name="hasReview",
            target_collection=ReviewsCollection.name,
        )
    ]

    @classmethod
    def build_properties(cls, row):
        props = {name: fn(row) for name, _, fn in cls.properties}

        review_uuids = []
        for c in [1, 2, 3]:
            col = f"Critic Review {c}"
            text = row.get(col)

            if pd.notna(text) and str(text).strip():
                review_uuids.append(uuid.uuid5(constants.NAMESPACE, text.strip()))

        return {
            "uuid": uuid.uuid4(),
            "properties": props,
            "references": {"hasReview": review_uuids} if review_uuids else None,
        }
